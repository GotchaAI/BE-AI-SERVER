# handle trained model(CNN)
import config
import os
from tensorflow import keras

print("모델 로딩 중...")
print("현재 경로:", os.getcwd())  # 현재 실행 중인 Python 스크립트의 경로 확인
print("모델 경로:", os.path.abspath(config.MODEL_PATH))  # 모델 경로 확인인
model = keras.models.load_model(config.MODEL_PATH)
print("모델 로드 완료!")