import io
from PIL import Image
from src.services.text_masking import recog_text, mask_text
import matplotlib.pyplot as plt

def preproc(image_bytes: bytes):
    """
    이미지 전처리 함수
    1. PIL.Image로 변환
    2. 텍스트 검출 후 masking
    """
    masking_box = recog_text(image_bytes)
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    masked_img = mask_text(image, masking_box) # PIL
    return masked_img


def show_image(image):
    """
    Args:
        image (PIL.Image): 출력할 이미지
    """
    plt.figure(figsize=(10, 6))
    plt.imshow(image)
    plt.axis("off")
    plt.show()