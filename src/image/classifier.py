import config
import time
import torch
import torch.nn.functional as F
import logging
from torchvision import transforms as T
from src.image.model import CNNModel
import glob
import os
encode_image = T.Compose([
    T.Resize((32, 32)),
    T.ToTensor(),
    T.Normalize(0.5, 0.5)
])

pattern = os.path.join(config.MODEL_PATH, "*.pth")
file_list = glob.glob(pattern)
latest_file=max(file_list, key=os.path.getctime)



logging.info("모델 로딩 중...")
device = "cuda" if torch.cuda.is_available() else "cpu"
model = CNNModel(output_classes = len(config.CATEGORIES))
model.load_state_dict(torch.load(latest_file, map_location=device))
model.to(device)
model.eval()  # 평가 모드
logging.info("모델 로드 완료!")

def classify(image):
    image = encode_image(image).unsqueeze(0).to(device)
    o1 = time.time()
    logging.info("모델 예측중 ....")
    with torch.no_grad():
        outputs = model(image)  # 모델 추론
        probabilities = F.softmax(outputs, dim=1)  # 확률 변환
        top3_prob, top3_indices = torch.topk(probabilities, 3)  # 상위 3개 예측 가져오기
    o2 = time.time()
    logging.info(f"모델 예측 걸린 시간 : {o2-o1:.2f}초.")

    return [
        {
            'predicted_class': config.CATEGORIES[top3_indices[0][0].item()],
            'confidence': top3_prob[0][0].item() * 100
        }, {
            "predicted_class": config.CATEGORIES[top3_indices[0][1].item()],
            "confidence": top3_prob[0][1].item() * 100,
        }, {
            "predicted_class": config.CATEGORIES[top3_indices[0][2].item()],
            "confidence": top3_prob[0][2].item() * 100
        }
    ]