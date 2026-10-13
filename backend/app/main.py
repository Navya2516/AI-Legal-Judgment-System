from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.app.api.judgment import router as judgment_router


app = FastAPI(
    title="AI Legal Judgment Explanation System",
    description="Evidence-grounded multilingual legal judgment analysis system",
    version="1.0.0"
)


# Allow React frontend to communicate with FastAPI
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.include_router(judgment_router)


@app.get("/")
def root():
    return {
        "message": "AI Legal Judgment System API is running"
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy"
    }