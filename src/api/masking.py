from http.client import HTTPResponse
from io import BytesIO

from fastapi import APIRouter, UploadFile, File, HTTPException
from src.core.mask import mask_text, upload_to_s3, S3_BUCKET_NAME
from PIL import Image
import uuid
router = APIRouter(
    tags=["Image Masking"]
)

@router.post(
    "/upload",
    summary="이미지 텍스트 마스킹 및 S3 업로드 API",
    description="업로드된 이미지 파일에서 텍스트를 마스킹하고, 마스킹된 이미지를 S3에 업로드한 후 해당 이미지의 URL을 반환합니다.",
    responses={
        200: {"message": "업로드된 이미지의 S3 URL"},

    }
)
async def mask_image(file: UploadFile = File(...)):
    if not S3_BUCKET_NAME:
        raise HTTPException(status_code=500, detail="S3_BUCKET_NAME 환경 변수가 설정되지 않았습니다.")

    try:
        contents = await file.read()
        img = Image.open(BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"이미지 파일을 읽는 도중 오류가 발생했습니다: {e}")

    masked_img = mask_text(img)

    masked_img_buffer = BytesIO()
    masked_img.save(masked_img_buffer, format="PNG")
    masked_img_buffer.seek(0)

    file_extension = file.filename.split(".")[-1] if "." in file.filename else "png"
    s3_filename = f"masked_images/{uuid.uuid4()}.{file_extension}"

    try:
        s3_url = upload_to_s3(masked_img_buffer, s3_filename)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"S3 이미지 업로드 중에 오류가 발생했습니다 : {e}")

    return {"message" : s3_url}
