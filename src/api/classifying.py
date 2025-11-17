from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field
from src.core.classify import classify
router = APIRouter(
    tags=["Image Classification"],
)

class ImageReq(BaseModel):
    image_url: str = Field(description="S3 이미지 URL")

class AiPrediction(BaseModel):
    predicted: str = Field(description="예측된 카테고리 (한국어)")
    confidence: float = Field(description="신뢰도 점수")

class ClassifyRes(BaseModel):
    filename: str = Field(description="이미지 파일 이름")
    result: list[AiPrediction] = Field(description="분류 결과 리스트")




@router.post(
    "/classify",
    summary="이미지 분류 API",
    description="S3 이미지 URL을 받아 해당 이미지의 분류 결과를 반환합니다.",
    response_model=ClassifyRes,
)
async def classify_image(request: Request, body: ImageReq):
    try:
        response = await request.app.state.http.get(body.image_url)
        response.raise_for_status()

        if not response.headers.get("content-type", "").startswith("image/"):
            # fallback: 실제 바이트로 이미지 여부 검사
            try:
                from io import BytesIO
                from PIL import Image
                Image.open(BytesIO(response.content)).verify()
            except Exception:
                raise HTTPException(status_code=415, detail="Unsupported content-type")
        predictions = classify(response.content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {e}")

    filename = body.image_url.split("/")[-1]
    result = [AiPrediction(**pred) for pred in predictions]
    return ClassifyRes(filename=filename, result=result)