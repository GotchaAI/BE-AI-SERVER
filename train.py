from torch.utils.data import DataLoader
import config
import torch.utils.data
from torchvision import transforms as T
from train.trainer import train_model, test_model
from datetime import datetime
from src.models.model import CNNModel
from train.datamodule import QuickDrawAllDataSet

encode_image = T.Compose([ # 이미지 데이터 전처리 : 32x32 크기, 정규화
    T.Resize(32),
    T.ToTensor(),
    T.Normalize(0.5, 0.5)
])

# Load CNN Model
model = CNNModel(output_classes=len(config.CATEGORIES))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


# Load Dataset
dataset = QuickDrawAllDataSet(max_drawings=1000, transform = encode_image)

# train, val split
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

# init DataLoader
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)


# Train Model
train_model(model, train_loader, val_loader)


# Test Model
test_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_model(model, test_loader)

# Save Model
file_dir = config.MODEL_PATH + f"quickdraw_cnn_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pth"
torch.save(model.state_dict(), file_dir)
print(f"모델 저장 완료: {file_dir}")