import torch.nn as nn

class CNNModel(nn.Module):
    def __init__(self, output_classes: int, dropout=0.2):
        super(CNNModel, self).__init__()

        # CNN Layer 정의
        self.conv_layer = nn.Sequential(
            nn.Conv2d(3, 32, 2),  # (3, 32, 32) → (32, 31, 31)
            nn.ReLU(),
            nn.Conv2d(32, 64, 2),  # (32, 31, 31) → (64, 30, 30)
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # (64, 30, 30) → (64, 15, 15)

            nn.Conv2d(64, 128, 3),  # (64, 15, 15) → (128, 13, 13)
            nn.ReLU(),
            nn.Conv2d(128, 256, 3),  # (128, 13, 13) → (256, 11, 11)
            nn.ReLU(),
            nn.MaxPool2d(3, 2),  # (256, 11, 11) → (256, 5, 5)
        )

        # Fully Connected Layer 정의
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(256 * 5 * 5, output_classes),  # Flatten 후 최종 분류
            nn.LogSoftmax(dim=1)  # LogSoftmax (NLLLoss 사용)
        )

    def forward(self, x):
        x = self.conv_layer(x)  # CNN Layer
        x = x.view(x.size(0), -1)  # Flatten
        x = self.classifier(x)  # FC Layer
        return x