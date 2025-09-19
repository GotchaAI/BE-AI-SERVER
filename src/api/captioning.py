from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from src.core.caption import generate_caption

router = APIRouter(
    tags=["Image Captioning"]
)

class ImageReq(BaseModel):
    image_url: str = Field(description="S3 이미지 URL")

class CaptionRes(BaseModel):
    caption: str = Field(description="이미지를 묘사하는 문장")

@router.post(
    "/caption",
    summary="이미지 문장 추출 API",
    description="S3 이미지 URL을 받아 해당 이미지를 묘사하는 적절한 문장을 반환합니다.",
    response_model=CaptionRes,
)
async def caption_image(request: Request, body: ImageReq):
    try:
        response = await request.app.state.http_client.get(body.image_url)
        response.raise_for_status()
        if not response.headers.get("content-type", "").startswith("image/"):
            raise HTTPException(415, "지원하지 않는 콘텐츠 유형입니다. 이미지 파일만 허용됩니다.")
        caption = generate_caption(response.content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"이미지 처리 중 오류가 발생했습니다: {e}")

    return CaptionRes(caption=caption)