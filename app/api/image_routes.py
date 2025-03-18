from fastapi import APIRouter, File, UploadFile
from services import img_preproc

router = APIRouter(prefix="/image")

@router.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()

    # 이미지 전처리
    img_arr = img_preproc.preproc(contents)

    