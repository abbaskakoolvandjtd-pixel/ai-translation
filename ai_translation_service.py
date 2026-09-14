from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from core.utils import TranslationConfig
from api.translation_router import router as translation_router

# Maximum file size: 1GB for chunked uploads (individual chunks limited to CHUNK_SIZE_MB)
MAX_FILE_SIZE_MB = getattr(TranslationConfig, "MAX_UPLOAD_SIZE_MB", 1000)
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024

app = FastAPI(
    title="Translation Service",
    docs_url="/docs",
    redoc_url="/redoc",
)

origins = [
    "http://localhost",
    "http://localhost:80",
    "http://localhost/spmbi",
    "http://localhost/translation",
    "*",  # Allow all origins for flexibility (restrict in production)
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Note: No file size middleware needed - chunked upload handles large files
# Individual chunk size is enforced at the endpoint level

app.include_router(translation_router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5020)
    print(TranslationConfig.__dict__)