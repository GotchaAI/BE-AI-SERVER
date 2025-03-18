import config
import numpy as np
from app.services.text_masking import recog_text, mask_text
import matplotlib.pyplot as plt

def preproc(image_bytes: bytes, image_size=config.IMAGE_SIZE):
    """
    이미지 전처리 함수
    1. text masking 수행
    2. 흑백 변환
    3. 크기 조정
    4. 정규화
    5. 차원 확장
    """
    masking_box = recog_text(image_bytes=image_bytes)
    masked_img = mask_text(image_bytes=image_bytes, boxes=masking_box) # PIL
    show_image(masked_img)


    img = masked_img.resize(image_size)

    show_image(img)
    img_arr = np.array(img) / 255.0
    img_arr = np.expand_dims(img_arr, axis=-1)
    img_arr = np.expand_dims(img_arr, axis=0)

    return img_arr



def show_image(image):
    """
    Args:
        image (PIL.Image): 출력할 이미지
    """
    plt.figure(figsize=(10, 6))
    plt.imshow(image)
    plt.axis("off")
    plt.show()