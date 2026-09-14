from typing import List, Dict

from fastapi import APIRouter, Depends, UploadFile, File, HTTPException, Query
from fastapi.responses import FileResponse
from pathlib import Path
import shutil
import uuid
from core.utils import TranslationConfig
from login_functions import verify_jwt_and_db
from shared_functions import get_user_permissions_for_service
from core.db import create_job, get_job, get_job_for_download, get_jobs_for_user

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

@router.post("/upload")
async def upload_pdf(file: UploadFile = File(...), context=Depends(verify_jwt_and_db)):
    payload = context["payload"]
    uid = int(payload.get("uid"))
    sid = int(payload.get("sid"))
    df_service = get_user_permissions_for_service(user_id=uid, service_id=sid)
    print(df_service)
    if not _has_module_access(df_service, 92):
        raise HTTPException(400, "کاربر دسترسی ندارد.")
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="فقط فایل‌های PDF مجاز هستند.")

    base_storage = getattr(TranslationConfig, "LOCAL_UPLOAD_PATH", "./uploads")
    user_dir = Path(base_storage) / "translations" / str(uid) / str(uuid.uuid4())
    user_dir.mkdir(parents=True, exist_ok=True)

    file_path = user_dir / file.filename
    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    job_id = create_job(uid, file.filename, str(file_path))
    return {"job_id": job_id, "status": "PENDING", "message": "File uploaded. Worker will pick it up shortly."}


@router.get("/status/{job_id}")
def get_status(job_id: str, context=Depends(verify_jwt_and_db)):
    payload = context["payload"]
    user_id = int(payload.get("uid"))
    uid = int(payload.get("uid"))
    sid = int(payload.get("sid"))
    df_service = get_user_permissions_for_service(user_id=uid, service_id=sid)
    print(df_service)
    if not _has_module_access(df_service, 92):
        raise HTTPException(400, "کاربر دسترسی ندارد.")
    job = get_job(job_id, user_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found or access denied.")
    return job


@router.get("/download/{job_id}")
def download_file(job_id: str, context=Depends(verify_jwt_and_db)):
    payload = context["payload"]
    user_id = int(payload.get("uid"))
    uid = int(payload.get("uid"))
    sid = int(payload.get("sid"))
    df_service = get_user_permissions_for_service(user_id=uid, service_id=sid)
    print(df_service)
    if not _has_module_access(df_service, 92):
        raise HTTPException(400, "کاربر دسترسی ندارد.")
    job = get_job_for_download(job_id, user_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found or access denied.")
    if job["Status"] != "COMPLETED":
        raise HTTPException(status_code=400, detail=f"File not ready. Current status: {job['Status']}")
    if not job["OutputFilePath"] or not Path(job["OutputFilePath"]).exists():
        raise HTTPException(status_code=404, detail="Output file not found on server.")

    return FileResponse(
        path=job["OutputFilePath"],
        filename=job["OriginalFileName"].replace(".pdf", "_translated.docx"),
        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    )