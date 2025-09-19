from pydantic import BaseModel
import os

class Settings(BaseModel):
    CAPTIONING_MODEL: str = "Salesforce/blip-image-captioning-base"
    CLASSIFYING_MODEL_PATH: str = "ml_models/classifying_model.pth"
    AWS_ACCESS_KEY_ID: str = os.getenv("AWS_ACCESS_KEY_ID")
    AWS_SECRET_ACCESS_KEY: str = os.getenv("AWS_SECRET_ACCESS_KEY")
    AWS_REGION: str = os.getenv("AWS_REGION")
    S3_BUCKET_NAME: str = os.getenv("S3_BUCKET_NAME")

settings = Settings()
