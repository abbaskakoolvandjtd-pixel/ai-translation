from typing import List, Dict, Optional

from fastapi import APIRouter, Depends, UploadFile, File, HTTPException, Query, status
from fastapi.responses import FileResponse
from pathlib import Path
import shutil
import uuid
import logging
from core.utils import TranslationConfig
from login_functions import verify_jwt_and_db
from shared_functions import get_user_permissions_for_service
from core.db import create_job, get_job, get_job_for_download, get_jobs_for_user

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/translation/v1", tags=["Translation"])


def _has_module_access(df_service, module_id: int = 92) -> bool:
    """
    Returns True if the user has an 'access' action for the given module_id.
    Handles the situation where the module_id is missing gracefully.
    """
    if df_service.empty:
        return False
    matched = df_service[df_service["module_id"] == module_id]
    if matched.empty:
        return False
    # Guard against missing 'action' column
    if "action" not in matched.columns:
        return False
    return matched.iloc[0]["action"] == "access"


@router.get("/jobs")
def list_user_jobs(
    skip: int = Query(0, ge=0, description="Number of records to skip"),
    limit: int = Query(
        50,
        ge=1,
        le=200,
        description="Maximum number of records to return (capped at 200)",
    ),
    context: dict = Depends(verify_jwt_and_db),
) -> Dict:
    """
    Return a paginated list of all translation jobs belonging to the
    authenticated user.
    """
    payload = context["payload"]
    uid = int(payload.get("uid"))   # user id from JWT
    sid = int(payload.get("sid"))   # service id – used for permission check

    # ---- Permission check -------------------------------------------------
    df_service = get_user_permissions_for_service(user_id=uid, service_id=sid)
    if not _has_module_access(df_service, 92):
        raise HTTPException(status_code=403, detail="User does not have access.")

    # ---- Fetch jobs --------------------------------------------------------
    jobs: List[Dict] = get_jobs_for_user(user_id=uid, skip=skip, limit=limit)

    # ---- Response ----------------------------------------------------------
    return {
        "skip": skip,
        "limit": limit,
        "total_returned": len(jobs),
        "jobs": jobs,
    }

# Get max file size from config (default 100MB)
MAX_FILE_SIZE_MB = getattr(TranslationConfig, "MAX_UPLOAD_SIZE_MB", 100)
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024


@router.post("/upload")
async def upload_pdf(
    file: UploadFile = File(..., description=f"PDF file to upload (max {MAX_FILE_SIZE_MB}MB)"),
    context=Depends(verify_jwt_and_db)
) -> Dict:
    """
    Upload a PDF file for translation.
    
    Args:
        file: PDF file to upload (max {MAX_FILE_SIZE_MB}MB)
        context: Authentication context from JWT
        
    Returns:
        Dictionary with job_id and status
        
    Raises:
        HTTPException: If file is not PDF, too large, or user lacks permission
    """
    payload = context["payload"]
    uid = int(payload.get("uid"))
    sid = int(payload.get("sid"))
    df_service = get_user_permissions_for_service(user_id=uid, service_id=sid)
    logger.info(f"User {uid} uploading file: {file.filename}")
    
    if not _has_module_access(df_service, 92):
        logger.warning(f"User {uid} denied access to translation module")
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="کاربر دسترسی ندارد.")
    
    if not file.filename:
        logger.warning(f"No filename provided")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="نام فایل نامعتبر است.")
    
    if not file.filename.lower().endswith(".pdf"):
        logger.warning(f"Invalid file type uploaded: {file.filename}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="فقط فایل‌های PDF مجاز هستند.")

    # Check file size by reading content if size not available in header
    file_size = 0
    file_content = await file.read()
    file_size = len(file_content)
    
    if file_size > MAX_FILE_SIZE_BYTES:
        logger.warning(f"File too large: {file_size} bytes (max: {MAX_FILE_SIZE_BYTES} bytes)")
        raise HTTPException(
            status_code=status.HTTP_413_PAYLOAD_TOO_LARGE,
            detail=f"فایل بسیار بزرگ است. حداکثر حجم مجاز {MAX_FILE_SIZE_MB} مگابایت است."
        )
    
    if file_size == 0:
        logger.warning(f"Empty file uploaded: {file.filename}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="فایل خالی است.")

    base_storage = getattr(TranslationConfig, "LOCAL_UPLOAD_PATH", "./uploads")
    user_dir = Path(base_storage) / "translations" / str(uid) / str(uuid.uuid4())
    
    try:
        user_dir.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        logger.error(f"Failed to create directory {user_dir}: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to create storage directory")

    file_path = user_dir / file.filename
    
    try:
        with open(file_path, "wb") as f:
            f.write(file_content)
    except IOError as e:
        logger.error(f"Failed to save file {file_path}: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to save file")
    finally:
        file.file.close()

    job_id = create_job(uid, file.filename, str(file_path))
    logger.info(f"Job {job_id} created for user {uid}, file size: {file_size} bytes")
    return {"job_id": job_id, "status": "PENDING", "message": f"File uploaded ({file_size / (1024*1024):.2f}MB). Worker will pick it up shortly."}


@router.get("/status/{job_id}")
def get_status(job_id: str, context=Depends(verify_jwt_and_db)) -> Dict:
    """
    Get the status of a translation job.
    
    Args:
        job_id: Unique identifier of the job
        context: Authentication context from JWT
        
    Returns:
        Job status information
        
    Raises:
        HTTPException: If job not found or user lacks permission
    """
    payload = context["payload"]
    user_id = int(payload.get("uid"))
    sid = int(payload.get("sid"))
    df_service = get_user_permissions_for_service(user_id=user_id, service_id=sid)
    logger.info(f"User {user_id} checking status for job {job_id}")
    
    if not _has_module_access(df_service, 92):
        logger.warning(f"User {user_id} denied access to translation module")
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="کاربر دسترسی ندارد.")
        
    job = get_job(job_id, user_id)
    if not job:
        logger.warning(f"Job {job_id} not found for user {user_id}")
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found or access denied.")
    return job


@router.get("/download/{job_id}")
def download_file(job_id: str, context=Depends(verify_jwt_and_db)) -> FileResponse:
    """
    Download the translated file for a completed job.
    
    Args:
        job_id: Unique identifier of the job
        context: Authentication context from JWT
        
    Returns:
        FileResponse with the translated document
        
    Raises:
        HTTPException: If job not found, not completed, or user lacks permission
    """
    payload = context["payload"]
    user_id = int(payload.get("uid"))
    sid = int(payload.get("sid"))
    df_service = get_user_permissions_for_service(user_id=user_id, service_id=sid)
    logger.info(f"User {user_id} requesting download for job {job_id}")
    
    if not _has_module_access(df_service, 92):
        logger.warning(f"User {user_id} denied access to translation module")
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="کاربر دسترسی ندارد.")
        
    job = get_job_for_download(job_id, user_id)
    if not job:
        logger.warning(f"Job {job_id} not found for user {user_id}")
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found or access denied.")
        
    if job["Status"] != "COMPLETED":
        logger.warning(f"Job {job_id} not ready for download, status: {job['Status']}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"File not ready. Current status: {job['Status']}")
        
    if not job["OutputFilePath"]:
        logger.error(f"Job {job_id} has no output file path")
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Output file path not recorded.")
        
    output_path = Path(job["OutputFilePath"])
    if not output_path.exists():
        logger.error(f"Output file not found on server: {output_path}")
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Output file not found on server.")

    original_name = job["OriginalFileName"]
    if original_name and original_name.lower().endswith(".pdf"):
        download_filename = original_name[:-4] + "_translated.docx"
    else:
        download_filename = f"{original_name}_translated.docx" if original_name else "translated.docx"
        
    logger.info(f"Serving file {output_path} as {download_filename}")
    return FileResponse(
        path=str(output_path),
        filename=download_filename,
        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    )