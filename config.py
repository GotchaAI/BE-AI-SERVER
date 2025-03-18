# CRAFT
TEXT_THRESHOLD=0.7
LINK_THRESHOLD=0.4
LOW_TEXT=0.4
CUDA=False
LONG_SIZE=1280


# QuickDraw 데이터 설정
MAX_DRAWINGS = 10000  # 각 클래스별 샘플 개수
IMAGE_SIZE = (28, 28)  # 이미지 크기 (너비, 높이)
CLASSES = ['cat', 'dog']
MODEL_PATH='./train/model/cat_dog_model.h5'