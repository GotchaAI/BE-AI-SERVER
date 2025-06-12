from typing import Dict, Any, List

from fastapi import APIRouter, File, UploadFile, Body, HTTPException
from src.image import classifier, preprocessor, img_caption
from pydantic import BaseModel, Field
import requests
from io import BytesIO
router = APIRouter(prefix="/image", tags=['Image'])


class AiPrediction(BaseModel):
    predicted: str
    confidence: float

class ClassifyRes(BaseModel):
    filename: str = Field(description="Image filename")
    result: List[AiPrediction] = Field(description="Classifying result")

class ImageReq(BaseModel):
    imageURL: str = Field(description = "Image URL")

@router.post(
    "/classify",
    summary="이미지 분류 API",
    description="S3 이미지 URL을 받아 QuickDraw 345개 클래스를 기반으로 분류합니다.",
    response_model=ClassifyRes,
)
async def classify(request: ImageReq = Body(...)):
    try:
        response = requests.get(request.imageURL)
        response.raise_for_status()  # HTTPError 발생시 예외 처리
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Image processing error: {str(e)}")

    try:
        bytes_img = response.content
        img = preprocessor.preproc(bytes_img)
        result = classifier.classify(img)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Classification error: {str(e)}")


    filename = request.imageURL.split("/")[-1]
    return ClassifyRes(filename=filename, result=result)



@router.post(
    '/caption',
    summary="이미지 문장 추출 API",
    description="S3 이미지 URL을 받아 해당 이미지를 묘사하는 적절한 문장을 반환합니다.",
responses={
    200:{
        "description":"성공",
        "content" :{
            "application/json" : {
                "example": "a black and white drawing of cat"
            }
        }
    }
})
async def captioning(request: ImageReq = Body(...)):
    try:
        response = requests.get(request.imageURL)
        response.raise_for_status()  # HTTPError 발생시 예외 처리
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Image processing error: {str(e)}")

    try:
        bytes_img = response.content
        img = preprocessor.preproc(bytes_img)
        caption = img_caption.get_caption(img)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Captioning error: {str(e)}")

    return caption
