import numpy as np
import config
from src.models import model

def classify(image_bytes: bytes):
    pred = model.predict(image_bytes)
    pred_label = np.argmax(pred[0])
    predicted_class = config.CLASSES[pred_label]
    confidence = float(pred[0][pred_label])

    return {
        "predicted_class" : predicted_class,
        "confidence" : confidence
    }