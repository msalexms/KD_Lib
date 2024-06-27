import os

from data.loaders import get_data_loaders
from data.transforms import *
from model.resnet import ResNet50, ResNet18
import torch.nn as nn
import torch.optim as optim
import torch
from train.train import train_models
import wandb

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

batch_size = 128
epochs = 200
learning_rate = 0.001
dropout = 0.3
data_dir = 'tiny-imagenet-200/'
experiment = 'resnet18-3'
criterion = nn.CrossEntropyLoss()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


wandb.init(sync_tensorboard=False,
               project='kd',
               id=experiment,
               config={
                   "kd_method": "VanillaKD",
                   "tensorBoard": False,
                   "learning_rate": learning_rate,
                   "epochs": epochs,
                   "batch_size": batch_size,
                   "dropout": dropout,
               }
               )

transform = get_transform_tensor_normalize()

trainloader, val_loader, testloader = get_data_loaders(data_dir, batch_size, transform, interpolation=False)


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    running_corrects = 0
    total_samples = 0

    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        _, preds = torch.max(outputs, 1)
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)
        total_samples += inputs.size(0)

    epoch_loss = running_loss / total_samples
    epoch_acc = running_corrects.double() / total_samples

    return epoch_loss, epoch_acc


# Función de validación
def validate_one_epoch(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    total_samples = 0

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            total_samples += inputs.size(0)

    epoch_loss = running_loss / total_samples
    epoch_acc = running_corrects.double() / total_samples

    return epoch_loss, epoch_acc


# Ejemplo de uso
def train_resnet18(train_dataloader, val_dataloader, num_epochs, device):
    model = ResNet18(3, len(trainloader.dataset.classes), dropout_prob=dropout, interpolation=False)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(num_epochs):
        train_loss, train_acc = train_one_epoch(model, train_dataloader, criterion, optimizer, device)
        val_loss, val_acc = validate_one_epoch(model, val_dataloader, criterion, device)
        wandb.log({"train_loss": train_loss, "train_acc": train_acc, "val_loss": val_loss, "val_acc": val_acc})

        print(f'Epoch {epoch}/{num_epochs - 1}')
        print(f'Train Loss: {train_loss:.4f} Train Acc: {train_acc:.4f}')
        print(f'Val Loss: {val_loss:.4f} Val Acc: {val_acc:.4f}')

    return model

print('Finished Training')

train_resnet18(trainloader,val_loader,200, device)

