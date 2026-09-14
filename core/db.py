import uuid
import time
import threading
import logging
from typing import List, Dict

from shared_functions import get_connection

logger = logging.getLogger(__name__)


# ==============================================================================
# Background Status Flusher (Solves High-Frequency Update Timeouts & Deadlocks)
# ==============================================================================
class _StatusFlusher:
    """
    Batches and throttles high-frequency status updates to prevent SQL Server
    lock contention, deadlocks, and timeouts when multiple threads update the same row.
    """

    def __init__(self, interval=2.0):
        self.interval = interval
        self._lock = threading.Lock()
        self._pending = {}  # {job_id: {status, detail, error, output}}
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def push(self, job_id, status, detail, error, output):
        with self._lock:
            self._pending[job_id] = {
                "status": status,
                "detail": detail,
                "error": error,
                "output": output
            }

    def _loop(self):
        while not self._stop_event.is_set():
            self._stop_event.wait(self.interval)
            self._flush_all()
        self._flush_all()  # Final flush on exit

    def _flush_all(self):
        with self._lock:
            if not self._pending:
                return
            to_flush = self._pending.copy()
            self._pending.clear()

        for job_id, data in to_flush.items():
            self._write_to_db(job_id, data["status"], data["detail"], data["error"], data["output"])

    def _write_to_db(self, job_id, status, detail, error, output, max_retries=3):
        sql = """
            UPDATE trn_Jobs 
            SET Status = ?, Status_detail = ?, ErrorMessage = ?, OutputFilePath = ?, UpdatedAt = SYSDATETIME()
            WHERE JobID = ?
        """
        for attempt in range(max_retries):
            try:
                with get_connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute(sql, (status, detail, error, output, job_id))
                    conn.commit()
                return  # Success
            except Exception as e:
                logger.warning(f"DB update retry {attempt + 1}/{max_retries} for {job_id}: {e}")
                if attempt < max_retries - 1:
                    time.sleep(0.5 * (attempt + 1))  # Exponential backoff
                else:
                    logger.error(f"Failed to update job {job_id} status after {max_retries} attempts.")

    def flush_immediate(self, job_id, status, detail, error, output):
        """Forces an immediate write and removes from pending queue. Used for terminal states."""
        with self._lock:
            if job_id in self._pending:
                del self._pending[job_id]
        self._write_to_db(job_id, status, detail, error, output)

    def shutdown(self):
        self._stop_event.set()
        self._thread.join(timeout=5)


# Initialize the global flusher (runs silently in the background)
_flusher = _StatusFlusher(interval=2.0)


# ==============================================================================
# Core DB Functions
# ==============================================================================

def create_job(user_id: int, original_filename: str, input_filepath: str) -> str:
    job_id = str(uuid.uuid4())
    sql = """
        INSERT INTO trn_Jobs (JobID, UserID, Status, OriginalFileName, InputFilePath)
        VALUES (?, ?, 'PENDING', ?, ?)
    """
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(sql, (job_id, user_id, original_filename, input_filepath))
        conn.commit()
    return job_id


def update_job_status(job_id: str, status: str, error_message: str = None, output_filepath: str = None,
                      status_detail: str = None):
    """
    Throttled update function.
    High-frequency progress updates are queued and flushed every 2 seconds.
    Terminal states (COMPLETED, FAILED) are written immediately.
    """
    is_terminal = status in ('COMPLETED', 'FAILED') or error_message is not None

    if is_terminal:
        _flusher.flush_immediate(job_id, status, status_detail, error_message, output_filepath)
    else:
        _flusher.push(job_id, status, status_detail, error_message, output_filepath)


def get_next_pending_job() -> dict | None:
    """Atomically fetches and locks the oldest PENDING job."""
    sql = """
        WITH cte AS (
            SELECT TOP (1) *
            FROM trn_Jobs
            WHERE Status = 'PENDING'
            ORDER BY CreatedAt ASC
        )
        UPDATE cte
        SET Status = 'OCR_PROCESSING',
            UpdatedAt = SYSDATETIME()
        OUTPUT inserted.JobID,
               inserted.UserID,
               inserted.OriginalFileName,
               inserted.InputFilePath;
    """
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(sql)
        row = cursor.fetchone()
        if row:
            return {
                "JobID": str(row[0]),
                "UserID": row[1],
                "OriginalFileName": row[2],
                "InputFilePath": row[3],
            }
    return None


def get_job(job_id: str, user_id: int) -> dict | None:
    # WITH (NOLOCK) prevents SELECT from being blocked by the background UPDATE thread
    sql = """
        SELECT Status, OriginalFileName, ErrorMessage, CreatedAt, UpdatedAt, Status_detail 
        FROM trn_Jobs WITH (NOLOCK) 
        WHERE JobID = ? AND UserID = ?
    """
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(sql, (job_id, user_id))
        row = cursor.fetchone()
        if row:
            return {
                "Status": row[0],
                "Status_detail": row[5],
                "OriginalFileName": row[1],
                "ErrorMessage": row[2],
                "CreatedAt": row[3],
                "UpdatedAt": row[4]
            }
    return None


def get_job_for_download(job_id: str, user_id: int) -> dict | None:
    # WITH (NOLOCK) for read performance
    sql = """
        SELECT Status, OriginalFileName, OutputFilePath 
        FROM trn_Jobs WITH (NOLOCK) 
        WHERE JobID = ? AND UserID = ?
    """
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(sql, (job_id, user_id))
        row = cursor.fetchone()
        if row:
            return {"Status": row[0], "OriginalFileName": row[1], "OutputFilePath": row[2]}
    return None


def get_jobs_for_user(user_id: int, skip: int = 0, limit: int = 50) -> List[Dict]:
    """
    Retrieve a paginated list of jobs for a given user.

    The query uses NOLOCK for non‑blocking reads (consistent with the
    rest of the codebase) and orders by CreatedAt descending
    so the newest jobs appear first.
    """
    sql = """
        SELECT JobID,
               Status,
               OriginalFileName,
               InputFilePath,
               OutputFilePath,
               ErrorMessage,
               CreatedAt,
               UpdatedAt,
               Status_detail
        FROM trn_Jobs WITH (NOLOCK)
        WHERE UserID = ?
        ORDER BY CreatedAt DESC
        OFFSET ? ROWS FETCH NEXT ? ROWS ONLY;
    """

    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(sql, (user_id, skip, limit))
        rows = cursor.fetchall()

    result: List[Dict] = []
    for row in rows:
        result.append(
            {
                "job_id": str(row[0]),
                "status": row[1],
                "original_file_name": row[2],
                "error_message": row[5],
                "created_at": row[6],
                "updated_at": row[7],
                "status_detail": row[8],
            }
        )
    return result


def get_stuck_failed_jobs(timeout_hours=1):
    """
    Fetches jobs that are in FAILED/ERROR state and haven't been updated
    for more than `timeout_hours`.
    """
    # NOTE: The date function below is for SQL Server (T-SQL).
    # If using PostgreSQL, use: NOW() - INTERVAL '1 hour'
    # If using MySQL, use: DATE_SUB(NOW(), INTERVAL 1 HOUR)
    query = """
  SELECT JobID, InputFilePath, Status, UpdatedAt 
FROM trn_Jobs 
WHERE (Status IN ('FAILED', 'ERROR')
AND UpdatedAt < DATEADD(hour, ?, GETDATE())) or (Status not IN ('COMPLETED','FAILED', 'ERROR'))
    """

    # --- Example implementation using pyodbc (adjust to your actual DB connection logic) ---
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(query, (timeout_hours,))
    columns = [column[0] for column in cursor.description]
    results = [dict(zip(columns, row)) for row in cursor.fetchall()]
    cursor.close()
    conn.close()
    return results

    raise NotImplementedError("Implement this using your existing DB connection logic.")


def reset_job_to_pending(job_id):
    """
    Resets a job's status back to PENDING and clears the error message.
    """
    query = """
        UPDATE trn_Jobs 
        SET Status = 'PENDING', 
            ErrorMessage = NULL, 
            UpdatedAt = GETDATE() 
        WHERE JobID = ?
    """

    # --- Example implementation using pyodbc ---
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(query, (job_id,))
    conn.commit()
    cursor.close()
    conn.close()

    raise NotImplementedError("Implement this using your existing DB connection logic.")