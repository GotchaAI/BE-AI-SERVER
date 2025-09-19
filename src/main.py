from fastapi import FastAPI
from src.api.captioning import router as caption_router
from src.api.myomyo import router as myomyo_router
from src.api.lulu import router as lulu_router
from src.api.classifying import router as classification_router
from src.api.masking import router as masking_router
import httpx

async def lifespan(app):
    app.state.http = httpx.AsyncClient(
        timeout=httpx.Timeout(10.0),
        limits=httpx.Limits(max_keepalive_connections=100, max_connections=200),
    )
    yield
    await app.state.http.aclose()
app = FastAPI(
    title="Gotcha! AI Server",
    description="AI Server",
    docs_url="/docs",
    openapi_url="/openapi.json",
    redoc_url="/redoc"
)

app.include_router(caption_router, prefix='/api/v1')
app.include_router(classification_router, prefix='/api/v1')
app.include_router(masking_router, prefix='/api/v1')
app.include_router(myomyo_router, prefix='/api/v1')
app.include_router(lulu_router, prefix='/api/v1')