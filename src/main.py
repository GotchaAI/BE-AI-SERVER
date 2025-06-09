import uvicorn
from fastapi import FastAPI
from src.api.image_routes import router as image_router
from src.api.myomyo_routes import router as chat_router
from src.api.lulu_routes import router as lulu_router
app = FastAPI(
    title="Gotcha! AI Server",
    description="AI Server",
    docs_url="/docs",
    openapi_url="/openapi.json",
    redoc_url="/redoc"
)

app.include_router(image_router, prefix='/api/v1')

app.include_router(chat_router, prefix='/api/v1')


app.include_router(lulu_router, prefix='/api/v1')