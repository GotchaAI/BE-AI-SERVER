import uvicorn
from fastapi import FastAPI
from src.api.image_routes import router as image_router

app = FastAPI(
    title="Gotcha! AI Server",
    description="AI Server",
    docs_url="/docs",
    openapi_url="/openapi.json",
    redoc_url="/redoc"
)

app.include_router(image_router, prefix='/api/v1')
