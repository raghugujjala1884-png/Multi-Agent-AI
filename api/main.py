from fastapi import FastAPI
from api.routes import router


app = FastAPI(
    title="Multi-Agent AI API",
    description="API for the LangGraph Multi-Agent AI application",
    version="1.0.0"
)


@app.get("/")
async def root():
    return {
        "message": "Multi-Agent AI API is running"
    }


@app.get("/health")
async def health_check():
    return {
        "status": "healthy"
    }


app.include_router(
    router,
    prefix="/api/v1",
    tags=["Chat"]
)