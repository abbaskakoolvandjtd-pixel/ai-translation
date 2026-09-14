import os
import sys
import time
import logging
from pathlib import Path


PROJECT_ROOT=os.path.dirname(os.path.dirname(Path(__file__)))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from core.utils import TranslationConfig
from core.db import get_next_pending_job, get_stuck_failed_jobs, reset_job_to_pending, update_job_status
from core.pipeline_executor import run_pipeline

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("translation_worker")


def check_and_reset_stuck_jobs(timeout_hours=1):
    """
    Checks for jobs in FAILED/ERROR state older than timeout_hours
    and resets them to PENDING for a retry.
    """
    try:
        stuck_jobs = get_stuck_failed_jobs(timeout_hours=timeout_hours)
        if stuck_jobs:
            logger.info(
                f"Found {len(stuck_jobs)} stuck/failed jobs older than {timeout_hours} hour(s). Resetting to PENDING...")
            for job in stuck_jobs:
                job_id = job['JobID']
                reset_job_to_pending(job_id)
                logger.info(f"✅ Reset job {job_id} to PENDING for retry.")
    except Exception as e:
        logger.error(f"Error checking/resetting stuck jobs: {e}")


def start_worker():
    logger.info("Worker started. Polling database for PENDING jobs and resetting stuck ERROR jobs...")

    while True:
        # 1. Check and reset stuck jobs first (Runs every loop iteration)
        check_and_reset_stuck_jobs(timeout_hours=1)

        # 2. Poll for next pending job
        job = get_next_pending_job()
        if job:
            job_id = job['JobID']
            logger.info(f"Picked up job {job_id}")
            try:
                run_pipeline(job)
            except Exception as e:
                logger.error(f"Job {job_id} crashed worker: {e}")
                # Ensure it's marked as FAILED in case it crashed before pipeline_executor could catch it
                update_job_status(job_id, "FAILED", error_message=str(e))

        # Poll every 5 seconds
        time.sleep(5)


if __name__ == "__main__":
    print(TranslationConfig.__dict__)
    start_worker()