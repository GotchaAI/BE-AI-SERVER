from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
from io import BytesIO
import torch
from src.config import settings

print('BLIP 모델 로딩중....')
processor = BlipProcessor.from_pretrained(settings.CAPTIONING_MODEL)
model = BlipForConditionalGeneration.from_pretrained(settings.CAPTIONING_MODEL)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print('BLIP 모델 로딩완료!')



def generate_caption(image: bytes) -> str:
    """
    이미지 캡션 생성 함수

    Args:
        image (bytes): 이미지 바이트 데이터

    Returns:
        str: 생성된 캡션
    """
    image = Image.open(BytesIO(image)).convert("RGB")
    inputs = processor(image, return_tensors="pt").to(device)

    with torch.no_grad():
        out = model.generate(**inputs)

    caption = processor.decode(out[0], skip_special_tokens=True)
    return caption
