import io
from PIL import Image, ImageDraw
import config
import easyocr
import numpy as np

reader = easyocr.Reader(['en', 'ko'])

def recog_text(image: Image):
    """
    Args: image
    Returns: 이미지가 인식된 바운딩 박스
    """
    image_np = np.array(image)

    results = reader.readtext(image_np)

    filtered_boxes = [box for box, text, conf in results if conf >= config.TEXT_THRESHOLD]

    for i, box in enumerate(filtered_boxes):
        print(f"[{i + 1}] 박스 좌표 (신뢰도 ≥ {config.TEXT_THRESHOLD}): {box}")

    return filtered_boxes



def mask_text(image: Image):
    """
    Args: PIL Image(RGB), 바운딩 박스
    Returns: 마스킹 된 이미지 데이터
    """
    boxes = recog_text(image)
    masked = image.copy()
    draw = ImageDraw.Draw(masked)
    for box in boxes:
        box = [(int(point[0]), int(point[1])) for point in box]
        draw.polygon(box, fill=(255, 255, 255))
    return masked




