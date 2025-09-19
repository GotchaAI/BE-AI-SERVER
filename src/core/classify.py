from PIL import Image
from io import BytesIO
import torch
import torch.nn as nn
from torchvision import transforms as T
from torchvision.models import efficientnet_b0
from typing import List
from src.config import settings

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model(model_path: str, num_classes: int) -> nn.Module:
    model = efficientnet_b0(weights=None)
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, num_classes)
    checkpoint = torch.load(model_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    return model


classifier = load_model(settings.CLASSIFYING_MODEL_PATH, settings.NUM_CLASSES)

encode_image = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.RandomHorizontalFlip(),
        T.RandomRotation(10),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


def classify(image_bytes: bytes) -> List[dict]:
    """
    이미지 분류 함수
    1. 이미지 전처리
    2. 모델 추론
    3. 상위 3개 카테고리 및 신뢰도 반환
    4. 한글 카테고리로 매핑하여 반환
    5. 반환 형식: List[{"predicted": 카테고리, "confidence": 신뢰도}]
    6. 신뢰도는 퍼센트(%)로 반환
    """
    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    img_tensor = encode_image(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = classifier(img_tensor)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        top3_prob, top3_catid = torch.topk(probabilities, 3)
    results = []
    for i in range(top3_prob.size(0)):
        results.append({
            "predicted": settings.KOR_CATEGORIES[top3_catid[i]], # 한글 카테고리로 변경
            "confidence": top3_prob[i].item() * 100
        })
    return results