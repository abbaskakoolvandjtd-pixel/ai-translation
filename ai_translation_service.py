from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from core.utils import TranslationConfig
from api.translation_router import router as translation_router

# Maximum file size: 100MB (adjust as needed)
MAX_FILE_SIZE_MB = getattr(TranslationConfig, "MAX_UPLOAD_SIZE_MB", 100)
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

# Add custom middleware to enforce max file size at the app level
@app.middleware("http")
async def check_file_size_middleware(request, call_next):
    """Middleware to check Content-Length header for large uploads."""
    content_length = request.headers.get("content-length")
    if content_length:
        try:
            size = int(content_length)
            if size > MAX_FILE_SIZE_BYTES:
                from fastapi.responses import JSONResponse
                return JSONResponse(
                    status_code=413,
                    content={"detail": f"File too large. Maximum allowed size is {MAX_FILE_SIZE_MB}MB"}
                )
        except ValueError:
            pass
    return await call_next(request)

app.include_router(translation_router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5020)
    print(TranslationConfig.__dict__)