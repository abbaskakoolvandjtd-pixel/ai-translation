from typing import List, Dict, Optional
from fastapi import APIRouter, Depends, UploadFile, File, HTTPException, Query, status, Form
from fastapi.responses import FileResponse
from pathlib import Path
import shutil
import uuid
import logging
import hashlib
from core.utils import TranslationConfig
from login_functions import verify_jwt_and_db
from shared_functions import get_user_permissions_for_service
from core.db import create_job, get_job, get_job_for_download, get_jobs_for_user

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/translation/v1", tags=["Translation"])


# Chunked upload configuration
CHUNK_SIZE_MB = getattr(TranslationConfig, "CHUNK_SIZE_MB", 5)  # Default 5MB per chunk
CHUNK_SIZE_BYTES = CHUNK_SIZE_MB * 1024 * 1024
MAX_FILE_SIZE_MB = getattr(TranslationConfig, "MAX_UPLOAD_SIZE_MB", 1000)  # Default 1GB total
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024


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

# Get max file size from config (default 1GB for chunked uploads)
# Note: Individual chunks are limited to CHUNK_SIZE_MB (default 5MB)


@router.post("/upload")
async def upload_pdf(
    file: UploadFile = File(..., description=f"PDF file to upload (max {MAX_FILE_SIZE_MB}MB)"),
    context=Depends(verify_jwt_and_db)
) -> Dict:
    """
    Upload a PDF file for translation (for files up to CHUNK_SIZE_MB).
    For larger files, use the chunked upload endpoints.
    
    Args:
        file: PDF file to upload (max {CHUNK_SIZE_MB}MB for single upload)
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
    
    # For single upload, limit to CHUNK_SIZE_MB to avoid server limits
    if file_size > CHUNK_SIZE_BYTES:
        logger.warning(f"File too large for single upload: {file_size} bytes. Use chunked upload.")
        raise HTTPException(
            status_code=status.HTTP_413_PAYLOAD_TOO_LARGE,
            detail=f"فایل برای آپلود مستقیم بسیار بزرگ است. حداکثر حجم مجاز {CHUNK_SIZE_MB} مگابایت است. از آپلود تکه‌ای استفاده کنید."
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


@router.post("/upload/initiate")
async def initiate_chunked_upload(
    filename: str = Form(..., description="Original filename"),
    total_size: int = Form(..., ge=1, description="Total file size in bytes"),
    context=Depends(verify_jwt_and_db)
) -> Dict:
    """
    Initiate a chunked upload session for large files.
    
    Args:
        filename: Original filename of the PDF
        total_size: Total size of the file in bytes
        context: Authentication context from JWT
        
    Returns:
        Dictionary with upload_session_id and chunk information
        
    Raises:
        HTTPException: If validation fails or user lacks permission
    """
    payload = context["payload"]
    uid = int(payload.get("uid"))
    sid = int(payload.get("sid"))
    df_service = get_user_permissions_for_service(user_id=uid, service_id=sid)
    logger.info(f"User {uid} initiating chunked upload for: {filename}")
    
    if not _has_module_access(df_service, 92):
        logger.warning(f"User {uid} denied access to translation module")
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="کاربر دسترسی ندارد.")
    
    if not filename or not filename.lower().endswith(".pdf"):
        logger.warning(f"Invalid filename for chunked upload: {filename}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="فقط فایل‌های PDF مجاز هستند.")
    
    if total_size > MAX_FILE_SIZE_BYTES:
        logger.warning(f"File too large: {total_size} bytes (max: {MAX_FILE_SIZE_BYTES} bytes)")
        raise HTTPException(
            status_code=status.HTTP_413_PAYLOAD_TOO_LARGE,
            detail=f"فایل بسیار بزرگ است. حداکثر حجم مجاز {MAX_FILE_SIZE_MB} مگابایت است."
        )
    
    if total_size == 0:
        logger.warning(f"Empty file specified for chunked upload: {filename}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="فایل خالی است.")
    
    # Generate unique session ID
    session_id = str(uuid.uuid4())
    upload_session_hash = hashlib.md5(f"{uid}:{session_id}:{filename}".encode()).hexdigest()[:16]
    
    base_storage = getattr(TranslationConfig, "LOCAL_UPLOAD_PATH", "./uploads")
    session_dir = Path(base_storage) / "translations" / str(uid) / "chunks" / upload_session_hash
    
    try:
        session_dir.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        logger.error(f"Failed to create session directory {session_dir}: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to create storage directory")
    
    # Calculate number of chunks
    num_chunks = (total_size + CHUNK_SIZE_BYTES - 1) // CHUNK_SIZE_BYTES
    
    # Store session metadata
    session_metadata = {
        "session_id": session_id,
        "upload_session_hash": upload_session_hash,
        "filename": filename,
        "total_size": total_size,
        "chunk_size": CHUNK_SIZE_BYTES,
        "num_chunks": num_chunks,
        "uploaded_chunks": [],
        "uid": uid,
        "created_at": str(uuid.uuid4())  # Using UUID as timestamp placeholder
    }
    
    # Save session metadata
    metadata_path = session_dir / "session_metadata.json"
    try:
        import json
        with open(metadata_path, "w") as f:
            json.dump(session_metadata, f)
    except IOError as e:
        logger.error(f"Failed to save session metadata: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to initialize upload session")
    
    logger.info(f"Chunked upload session {session_id} created for user {uid}, {num_chunks} chunks expected")
    
    return {
        "upload_session_id": upload_session_hash,
        "session_id": session_id,
        "filename": filename,
        "total_size": total_size,
        "chunk_size": CHUNK_SIZE_BYTES,
        "num_chunks": num_chunks,
        "message": f"Session initialized. Upload {num_chunks} chunks sequentially."
    }


@router.post("/upload/chunk")
async def upload_chunk(
    upload_session_id: str = Form(..., description="Upload session ID"),
    chunk_index: int = Form(..., ge=0, description="Chunk index (0-based)"),
    chunk_data: UploadFile = File(..., description="Chunk data"),
    context=Depends(verify_jwt_and_db)
) -> Dict:
    """
    Upload a single chunk of a large file.
    
    Args:
        upload_session_id: Session ID from initiate_chunked_upload
        chunk_index: Index of this chunk (0-based)
        chunk_data: The chunk file data
        context: Authentication context from JWT
        
    Returns:
        Dictionary with chunk upload status
        
    Raises:
        HTTPException: If validation fails or chunk is invalid
    """
    payload = context["payload"]
    uid = int(payload.get("uid"))
    sid = int(payload.get("sid"))
    df_service = get_user_permissions_for_service(user_id=uid, service_id=sid)
    logger.info(f"User {uid} uploading chunk {chunk_index} for session {upload_session_id}")
    
    if not _has_module_access(df_service, 92):
        logger.warning(f"User {uid} denied access to translation module")
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="کاربر دسترسی ندارد.")
    
    if not upload_session_id:
        logger.warning(f"No upload session ID provided")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="شناسه جلسه آپلود نامعتبر است.")
    
    # Load session metadata
    base_storage = getattr(TranslationConfig, "LOCAL_UPLOAD_PATH", "./uploads")
    session_dir = Path(base_storage) / "translations" / str(uid) / "chunks" / upload_session_id
    metadata_path = session_dir / "session_metadata.json"
    
    if not metadata_path.exists():
        logger.warning(f"Session {upload_session_id} not found for user {uid}")
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="جلسه آپلود یافت نشد.")
    
    import json
    try:
        with open(metadata_path, "r") as f:
            session_metadata = json.load(f)
    except (IOError, json.JSONDecodeError) as e:
        logger.error(f"Failed to load session metadata: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to load session information")
    
    # Verify session belongs to user
    if session_metadata.get("uid") != uid:
        logger.warning(f"User {uid} attempted to access session belonging to another user")
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="دسترسی غیرمجاز به جلسه آپلود.")
    
    # Validate chunk index
    if chunk_index >= session_metadata["num_chunks"]:
        logger.warning(f"Invalid chunk index {chunk_index} for session {upload_session_id}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="اندیس تکه نامعتبر است.")
    
    # Read and validate chunk
    chunk_content = await chunk_data.read()
    chunk_size = len(chunk_content)
    
    # Last chunk can be smaller, others must match chunk size
    expected_size = CHUNK_SIZE_BYTES if chunk_index < session_metadata["num_chunks"] - 1 else None
    if expected_size and chunk_size != expected_size:
        logger.warning(f"Chunk {chunk_index} size mismatch: expected {expected_size}, got {chunk_size}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"اندازه تکه نامعتبر است. انتظار می‌رفت {expected_size} بایت باشد."
        )
    
    if chunk_size == 0:
        logger.warning(f"Empty chunk {chunk_index} uploaded")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="تکه خالی است.")
    
    # Save chunk
    chunk_filename = f"chunk_{chunk_index:05d}"
    chunk_path = session_dir / chunk_filename
    
    try:
        with open(chunk_path, "wb") as f:
            f.write(chunk_content)
    except IOError as e:
        logger.error(f"Failed to save chunk {chunk_index}: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to save chunk")
    finally:
        chunk_data.file.close()
    
    # Update session metadata
    if chunk_index not in session_metadata["uploaded_chunks"]:
        session_metadata["uploaded_chunks"].append(chunk_index)
        session_metadata["uploaded_chunks"].sort()
        try:
            with open(metadata_path, "w") as f:
                json.dump(session_metadata, f)
        except IOError as e:
            logger.error(f"Failed to update session metadata: {e}")
            # Non-fatal, continue
    
    logger.info(f"Chunk {chunk_index}/{session_metadata['num_chunks']} uploaded successfully")
    
    all_uploaded = len(session_metadata["uploaded_chunks"]) == session_metadata["num_chunks"]
    
    return {
        "upload_session_id": upload_session_id,
        "chunk_index": chunk_index,
        "chunk_size": chunk_size,
        "status": "uploaded",
        "all_chunks_uploaded": all_uploaded,
        "message": f"Chunk {chunk_index + 1}/{session_metadata['num_chunks']} uploaded successfully"
    }


@router.post("/upload/complete")
async def complete_chunked_upload(
    upload_session_id: str = Form(..., description="Upload session ID"),
    context=Depends(verify_jwt_and_db)
) -> Dict:
    """
    Complete a chunked upload by assembling all chunks into the final file.
    
    Args:
        upload_session_id: Session ID from initiate_chunked_upload
        context: Authentication context from JWT
        
    Returns:
        Dictionary with job_id and status
        
    Raises:
        HTTPException: If validation fails or not all chunks are uploaded
    """
    payload = context["payload"]
    uid = int(payload.get("uid"))
    sid = int(payload.get("sid"))
    df_service = get_user_permissions_for_service(user_id=uid, service_id=sid)
    logger.info(f"User {uid} completing chunked upload for session {upload_session_id}")
    
    if not _has_module_access(df_service, 92):
        logger.warning(f"User {uid} denied access to translation module")
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="کاربر دسترسی ندارد.")
    
    if not upload_session_id:
        logger.warning(f"No upload session ID provided")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="شناسه جلسه آپلود نامعتبر است.")
    
    # Load session metadata
    base_storage = getattr(TranslationConfig, "LOCAL_UPLOAD_PATH", "./uploads")
    session_dir = Path(base_storage) / "translations" / str(uid) / "chunks" / upload_session_id
    metadata_path = session_dir / "session_metadata.json"
    
    if not metadata_path.exists():
        logger.warning(f"Session {upload_session_id} not found for user {uid}")
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="جلسه آپلود یافت نشد.")
    
    import json
    try:
        with open(metadata_path, "r") as f:
            session_metadata = json.load(f)
    except (IOError, json.JSONDecodeError) as e:
        logger.error(f"Failed to load session metadata: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to load session information")
    
    # Verify session belongs to user
    if session_metadata.get("uid") != uid:
        logger.warning(f"User {uid} attempted to access session belonging to another user")
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="دسترسی غیرمجاز به جلسه آپلود.")
    
    # Verify all chunks are uploaded
    if len(session_metadata["uploaded_chunks"]) != session_metadata["num_chunks"]:
        logger.warning(f"Incomplete upload: {len(session_metadata['uploaded_chunks'])}/{session_metadata['num_chunks']} chunks")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"همه تکه‌ها آپلود نشده‌اند. {len(session_metadata['uploaded_chunks'])}/{session_metadata['num_chunks']} تکه موجود است."
        )
    
    # Assemble chunks into final file
    filename = session_metadata["filename"]
    user_dir = Path(base_storage) / "translations" / str(uid) / str(uuid.uuid4())
    
    try:
        user_dir.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        logger.error(f"Failed to create directory {user_dir}: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to create storage directory")
    
    file_path = user_dir / filename
    
    try:
        with open(file_path, "wb") as final_file:
            for chunk_index in range(session_metadata["num_chunks"]):
                chunk_filename = f"chunk_{chunk_index:05d}"
                chunk_path = session_dir / chunk_filename
                
                if not chunk_path.exists():
                    logger.error(f"Missing chunk {chunk_index} during assembly")
                    raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Incomplete chunk data")
                
                with open(chunk_path, "rb") as chunk_file:
                    shutil.copyfileobj(chunk_file, final_file)
    except (IOError, HTTPException) as e:
        logger.error(f"Failed to assemble chunks: {e}")
        if isinstance(e, IOError):
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to assemble file")
        raise
    
    # Clean up chunk files
    try:
        for chunk_index in range(session_metadata["num_chunks"]):
            chunk_filename = f"chunk_{chunk_index:05d}"
            chunk_path = session_dir / chunk_filename
            if chunk_path.exists():
                chunk_path.unlink()
        # Remove session directory
        if session_dir.exists():
            shutil.rmtree(session_dir)
    except OSError as e:
        logger.warning(f"Failed to clean up chunk files: {e}")
        # Non-fatal, continue
    
    # Create job
    job_id = create_job(uid, filename, str(file_path))
    logger.info(f"Job {job_id} created from chunked upload for user {uid}, file size: {session_metadata['total_size']} bytes")
    
    return {
        "job_id": job_id,
        "status": "PENDING",
        "message": f"File assembled and uploaded ({session_metadata['total_size'] / (1024*1024):.2f}MB). Worker will pick it up shortly."
    }


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