from fastapi import FastAPI
from core.utils import TranslationConfig
from api.translation_router import router as translation_router
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Translation Service")

origins = [
    "http://localhost",
    "http://localhost:80",
    "http://localhost/spmbi",
    "http://localhost/translation",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(translation_router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5020)
    print(TranslationConfig.__dict__)