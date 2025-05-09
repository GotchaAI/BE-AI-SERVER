from fastapi import APIRouter, File, UploadFile
from src.chat import gpt_handler
from src.image import classifier, preprocessor, img_caption

router = APIRouter(prefix="/image")

@router.post("/classify")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()

    # 이미지 전처리
    img = preprocessor.preproc(contents) # returns PIL

    # 분류기 예측
    result = classifier.classify(img)

    return {
        'result' : result
    }

@router.post('/caption')
async def predict(file: UploadFile = File(...)):
    contents = await file.read()

    # 이미지 전처리
    img = preprocessor.preproc(contents) # returns PIL

    # 문장으로 변환
    caption = img_caption.get_caption(img)

    return {
        'caption': caption
    }
