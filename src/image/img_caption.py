from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch


print('BLIP 모델 로딩중....')
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
print('BLIP 모델 로딩완료!')



def get_caption(image: Image) -> str:
    inputs = processor(images=image, return_tensors="pt")

    with torch.no_grad():
        output_ids = model.generate(**inputs)
    caption = processor.decode(output_ids[0], skip_special_tokens=True)

    return caption