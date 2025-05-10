from fastapi import APIRouter, File, UploadFile
from fastapi.responses import JSONResponse
from src.image import classifier, preprocessor, img_caption
from pydantic import BaseModel, Field

router = APIRouter(prefix="/image")


class ClassifyRes(BaseModel):
    filename: str = Field(description="Image filename")
    result: str = Field(description="Image result")

@router.post(
    "/classify",
    summary="이미지 분류 API",
    description="이미지 파일을 받아 QuickDraw 345개 클래스를 기반으로 분류합니다.",
    response_model=ClassifyRes,
    responses={
    200: {
        "description" : "성공",
        "content" :{
            "application/json" : {
                "example": {
                    "filename" : "cat.png",
                    "result" :"cat"
                }
            }
        }
    }
}
)
async def classify(file: UploadFile = File(..., description="이미지 파일")):
    contents = await file.read()

    # 이미지 전처리
    img = preprocessor.preproc(contents) # returns PIL

    # 분류기 예측
    result = classifier.classify(img)

    return ClassifyRes(filename=file.filename, result=result)



class CaptioningRes(BaseModel):
    filename: str = Field(description="Image filename")
    result: str = Field(description="Image result")

@router.post(
    '/caption',
    summary="이미지 문장 추출 API",
    description="이미지 파일을 받아 해당 이미지를 묘사하는 적절한 문장을 반환합니다.",
response_model=CaptioningRes,
responses={
    200:{
        "description":"성공",
        "content" :{
            "application/json" : {
                "example": {
                    "filename" : "cat.png",
                    "result":"a black and white drawing of cat"
                }
            }
        }
    }
})
async def captioning(file: UploadFile = File(...)):
    contents = await file.read()

    # 이미지 전처리
    img = preprocessor.preproc(contents) # returns PIL

    # 문장으로 변환
    caption = img_caption.get_caption(img)

    return CaptioningRes(filename=file.filename, result=caption)
