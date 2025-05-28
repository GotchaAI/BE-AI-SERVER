import config
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from torchvision import transforms as T
from torchvision.models import efficientnet_b0
import glob
import os

# EfficientNet에 맞는 이미지 전처리 (ImageNet 표준)
encode_image = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.RandomHorizontalFlip(),
        T.RandomRotation(10),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

# 최신 모델 파일 찾기
pattern = os.path.join(config.MODEL_PATH, "*.pth")
file_list = glob.glob(pattern)
latest_file = max(file_list, key=os.path.getctime)

logging.info("EfficientNet 모델 로딩 중...")
device = "cuda" if torch.cuda.is_available() else "cpu"

# EfficientNet 모델 생성 및 로드
def load_efficientnet_model(model_path, num_classes):
    """EfficientNet 모델 로드"""
    try:
        # 저장된 모델 정보 로드
        checkpoint = torch.load(model_path, map_location=device)
        
        # EfficientNet-B0 모델 생성
        model = efficientnet_b0(weights=None)  # 가중치 없이 모델 구조만 로드
        
        # 분류기 레이어 수정
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_ftrs, num_classes)
        
        # 저장된 가중치 로드
        if 'model_state_dict' in checkpoint:
            # 새로운 형식 (딕셔너리 형태)
            model.load_state_dict(checkpoint['model_state_dict'])
            logging.info("딕셔너리 형태의 체크포인트에서 모델 로드")
        else:
            # 이전 형식 (직접 state_dict)
            model.load_state_dict(checkpoint)
            logging.info("직접 state_dict에서 모델 로드")
        
        return model
        
    except Exception as e:
        logging.error(f"EfficientNet 모델 로드 실패: {e}")
        # 대안: 기본 EfficientNet 모델 생성 (사전 훈련된 가중치 사용)
        logging.info("기본 EfficientNet 모델로 대체...")
        model = efficientnet_b0(weights='IMAGENET1K_V1')
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_ftrs, num_classes)
        return model

# 모델 로드
model = load_efficientnet_model(latest_file, len(config.CATEGORIES))
model.to(device)
model.eval()  # 평가 모드
logging.info("EfficientNet 모델 로드 완료!")

def classify(image):
    """
    이미지를 분류하고 상위 3개 예측 결과를 반환
    
    Args:
        image: PIL Image 객체
        
    Returns:
        list: 상위 3개 예측 결과 (클래스명, 신뢰도 포함)
    """
    try:
        # 이미지 전처리
        image_tensor = encode_image(image).unsqueeze(0).to(device)
        
        o1 = time.time()
        logging.info("EfficientNet 모델 예측중 ....")
        
        with torch.no_grad():
            outputs = model(image_tensor)  # 모델 추론
            probabilities = F.softmax(outputs, dim=1)  # 확률 변환
            top3_prob, top3_indices = torch.topk(probabilities, 3)  # 상위 3개 예측 가져오기
        
        o2 = time.time()
        logging.info(f"EfficientNet 모델 예측 걸린 시간 : {o2-o1:.2f}초.")

        # 결과 반환 (기존 형식과 동일)
        results = []
        for i in range(3):
            class_idx = top3_indices[0][i].item()
            confidence = top3_prob[0][i].item() * 100
            
            results.append({
                'predicted': config.CATEGORIES[class_idx],
                'confidence': confidence
            })
        
        return results
        
    except Exception as e:
        logging.error(f"분류 중 오류 발생: {e}")
        # 오류 발생 시 기본값 반환
        return [
            {'predicted': 'unknown', 'confidence': 0.0},
            {'predicted': 'unknown', 'confidence': 0.0},
            {'predicted': 'unknown', 'confidence': 0.0}
        ]