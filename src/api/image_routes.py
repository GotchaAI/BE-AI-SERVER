from fastapi import APIRouter, File, UploadFile
from src.services import img_preproc, gpt_handler
from src.models import classifier

router = APIRouter(prefix="/image")

@router.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()

    # 이미지 전처리
    img_arr = img_preproc.preproc(contents)

    # 분류기 예측
    result = classifier.classify(img_arr)

    # GPT API 호출
    message = gpt_handler.get_message(
        predicted_class=result['predicted_class'], 
        confidence=result['confidence']
    )

    return {
        "filename": file.filename,
        "predicted_class": result['predicted_class'],
        "confidence": result['confidence'],
        "message" : message
    }