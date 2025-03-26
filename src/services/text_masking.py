import io
from PIL import Image, ImageDraw
import config
from craft_text_detector import (
    load_craftnet_model,
    load_refinenet_model,
    get_prediction
)
import logging

logging.info("CRAFT_NET 가중치 로딩중 ...")
craft_net = load_craftnet_model(cuda=config.CUDA)
logging.info("CRAFT_NET 가중치 로딩 완료.")

logging.info("REFINE_NET 가중치 로딩중...")
refine_net = load_refinenet_model(cuda=config.CUDA)
logging.info("REFINE_NET 가중치 로딩 완료.")


def recog_text(image: bytes):
    """
    Args: image
    Returns: 이미지가 인식된 바운딩 박스
    """
    logging.info("텍스트 검출 중...")
    prediction_res = get_prediction(
        image = image,
        craft_net=craft_net,
        refine_net=refine_net,
        text_threshold=config.TEXT_THRESHOLD,
        link_threshold=config.LINK_THRESHOLD,
        low_text=config.LOW_TEXT,
        cuda=config.CUDA,
        long_size=config.LONG_SIZE
    )
    logging.info('텍스트 검출 완료.')
    logging.info(f'텍스트 검출 걸린 시간: {sum(prediction_res["times"].values()):.2f}초.')
    return prediction_res['boxes']



def mask_text(image: Image, boxes):
    """
    Args: PIL Image(RGB), 바운딩 박스
    Returns: 마스킹 된 이미지 데이터
    """
    masked = image.copy()
    draw = ImageDraw.Draw(masked)
    for box in boxes:
        box = [(int(point[0]), int(point[1])) for point in box]
        draw.polygon(box, fill=(255, 255, 255))
    return masked




