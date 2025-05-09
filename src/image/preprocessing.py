import io
from PIL import Image
from src.image.text_masking import mask_text

def preproc(image_bytes: bytes):
    """
    이미지 전처리 함수
    1. PIL.Image로 변환
    2. 텍스트 검출 후 masking
    """
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    masked_img = mask_text(image)
    return masked_img
