import io
from PIL import Image, ImageDraw
import config
from craft_text_detector import (
    read_image,
    load_craftnet_model,
    load_refinenet_model,
    get_prediction
)

print("CRAFT_NET 가중치 로딩중 ...")
craft_net = load_craftnet_model(cuda=config.CUDA)
print("CRAFT_NET 가중치 로딩 완료.")

print("REFINE_NET 가중치 로딩중...")
refine_net = load_refinenet_model(cuda=config.CUDA)
print("REFINE_NET 가중치 로딩 완료.")


def recog_text(image_bytes: bytes):
    """
    Args: bytes 이미지 데이터
    Returns: 이미지가 인식된 바운딩 박스
    """
    img_data = read_image(image_bytes)
    print("텍스트 검출/마스킹 중...")
    prediction_res = get_prediction(
        image = img_data,
        craft_net=craft_net,
        refine_net=refine_net,
        text_threshold=config.TEXT_THRESHOLD,
        link_threshold=config.LINK_THRESHOLD,
        low_text=config.LOW_TEXT,
        cuda=config.CUDA,
        long_size=config.LONG_SIZE
    )
    print('텍스트 검출/마스킹 완료.')
    # print(prediction_res['times'])
    return prediction_res['boxes']



def mask_text(image_bytes: bytes, boxes):
    """
    Args: bytes 이미지 데이터, 바운딩 박스
    Returns: 마스킹 된 이미지 데이터터
    """
    img = Image.open(io.BytesIO(image_bytes)).convert('L')
    draw = ImageDraw.Draw(img)
    for box in boxes:
        box = [(int(point[0]), int(point[1])) for point in box]
        draw.polygon(box, fill=255)
    return img




