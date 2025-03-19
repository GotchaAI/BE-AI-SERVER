import numpy as np
from quickdraw import QuickDrawDataGroup
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
import config

# ----------------------
# QuickDraw 데이터 로드 (고양이, 개 각각 10000개)
# ----------------------
print("QuickDraw 데이터 로딩 중 ...")
cat_data = QuickDrawDataGroup("cat", max_drawings=config.MAX_DRAWINGS)
dog_data = QuickDrawDataGroup("dog", max_drawings=config.MAX_DRAWINGS)
cat_drawings = cat_data.drawings
dog_drawings = dog_data.drawings

# ----------------------
# NumPy 변환
# ----------------------
X_cat = [np.array(d.get_image(stroke_width=2).resize((config.IMAGE_SIZE[0], config.IMAGE_SIZE[1])).convert("L")) for d in cat_drawings]
X_dog = [np.array(d.get_image(stroke_width=2).resize((config.IMAGE_SIZE[0], config.IMAGE_SIZE[1])).convert("L")) for d in dog_drawings]

X_cat = np.array(X_cat)
X_dog = np.array(X_dog)

# 레이블 생성 (고양이=0, 개=1)
y_cat = np.zeros(len(X_cat), dtype=np.int32)
y_dog = np.ones(len(X_dog), dtype=np.int32)

# 데이터 합치기
X = np.concatenate([X_cat, X_dog], axis=0)
y = np.concatenate([y_cat, y_dog], axis=0)

# 스케일링 및 차원 확장 (CNN 입력용)
X = X / 255.0
X = np.expand_dims(X, axis=-1)  # (4000, 28, 28, 1)

# ----------------------
# 학습용/검증용 데이터 분할
# ----------------------
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# ----------------------
# CNN 모델 정의
# ----------------------
model = keras.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(config.IMAGE_SIZE[0],config.IMAGE_SIZE[1],1)),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(2, activation='softmax')  # 2개의 클래스 (cat/dog)
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()

# ----------------------
# 모델 학습
# ----------------------
print("모델 학습 시작...")
history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=config.EPOCHS, batch_size=config.BATCH_SIZE)

# ----------------------
# 모델 저장
# ----------------------
model.save(config.MODEL_PATH)
print("모델이 'cat_dog_model.h5' 파일로 저장되었습니다!")