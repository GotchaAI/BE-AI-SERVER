from fastapi import APIRouter, File, UploadFile
from src.chat import gpt_handler
from src.preproc import img_preproc
from src.models import classifier

router = APIRouter(prefix="/image")

@router.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()

    # 이미지 전처리
    img = img_preproc.preproc(contents) # returns PIL

    # 분류기 예측
    result = classifier.classify(img)

    # GPT API 호출
    for i in range(3):
        result[i]['message'] = gpt_handler.get_message(result[i])


    return {
        'filename': file.filename,
        'result' : result
    }

@router.post('/context')
async def predict(file: UploadFile = File(...)):
    contents = await file.read()

    # 이미지 전처리
    img = img_preproc.preproc(contents) # returns PIL