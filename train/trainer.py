import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter



def train_model(model, train_loader, val_loader, num_epochs=10, lr=0.001):
    """
    trained_model : 학습 시킬 신경망 모델 객체
    train_loader : 훈련 데이터 객체(DataLoader)
    val_loader : 검증 데이터 객체(DataLoader)
    num_epoch : 학습할 EPOCH 수
    lr : 학습률
    """
    # GPU 학습 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)


    optimizer = optim.Adam(model.parameters(), lr=lr) # 신경망 옵티마이저 설정 : adam
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.85) # lr_scheduler: 일정스텝마다 학습률을 감소시킴
    criterion = nn.NLLLoss() # LogSoftmax 출력을 위한 손실 함수
    writer = SummaryWriter(log_dir='runs/experiment')

    for epoch in range(num_epochs):
        print(f"EPOCH {epoch + 1}/{num_epochs} Training..")

        # ----------------------------
        # Training Loop
        # ----------------------------
        model.train()
        train_loss, correct, total = 0, 0, 0 # 전체 훈련 손실, accuracy 계산을 위한 correct/total

        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device) # device로 데이터 이동

            optimizer.zero_grad() # set Gradient to Zero
            outputs = model(images) # 모델 예측 결과
            loss = criterion(outputs, labels) # 손실 함수 계산
            loss.backward() # 역전파
            optimizer.step()

            train_loss += loss.item() # 훈련 총 손실 업데이트
            correct += (outputs.argmax(1) == labels).sum().item() # 예측값 맞는지 카운트
            total += labels.size(0) # 샘플 갯수 누적

            writer.add_scalar('Loss/train', loss.item(), epoch * len(train_loader) + i)

        avg_loss = train_loss / len(train_loader)
        train_acc = 100 * correct / total # accuracy 업데이트

        writer.add_scalar('Loss/epoch', avg_loss, epoch)
        writer.add_scalar('Accuracy/epoch', train_acc, epoch)

        # ----------------------------
        # Validation Loop
        # ----------------------------
        model.eval()
        val_loss, correct, total = 0, 0, 0 # validation 손실, accuracy 계산을 위한 correct/total
        with torch.no_grad(): # 역전파 사용하지 않으므로 Gradient 저장 X
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images) # 예측 수행
                loss = criterion(outputs, labels) # 손실 계산

                val_loss += loss.item() # 손실값 누적
                correct += (outputs.argmax(1) == labels).sum().item()
                total += labels.size(0)

        val_acc = 100 * correct / total

        scheduler.step()

        print(f"Epoch {epoch+1}/{num_epochs} | "
              f"Train Loss: {train_loss/len(train_loader):.4f}, Train Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss/len(val_loader):.4f}, Val Acc: {val_acc:.2f}%")


def test_model(model, test_loader):
    """
    trained_model : test할 모델 객체
    test_loader : 테스트 데이터 객체(DataLoader)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    criterion = nn.NLLLoss()
    test_loss, correct, total = 0, 0, 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            test_loss += loss.item()
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)

    test_acc = 100 * correct / total
    print(f"Test Loss: {test_loss/len(test_loader):.4f}, Test Accuracy: {test_acc:.2f}%")




