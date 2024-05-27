import torch
import torch.nn as nn
from torchvision import models


class ResNet101(nn.Module):
    def __init__(self):
        super(ResNet101, self).__init__()
        self.model = models.resnet101(pretrained=True)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, 10)
        print("Resnet101")

    def forward(self, x):
        return self.model(x)


class ResNet50(nn.Module):
    def __init__(self, channels, classes, pretrained, dropout_prob=0.5, weights="../weights/resnet50-11ad3fa6.pth",
                 interpolation=False):
        super(ResNet50, self).__init__()
        if pretrained:
            self.model = models.resnet50(pretrained=True)
        else:
            print("Custom Weights")
            self.model = models.resnet50(pretrained=False)
            state_dict = torch.load(weights)
            self.model.load_state_dict(state_dict=state_dict)
        # Adjust the first convolutional layer to accept 'channels' input channels
        if interpolation:
            self.model.conv1 = nn.Conv2d(channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        else:
            self.model.conv1 = nn.Conv2d(channels, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        num_ftrs = self.model.fc.in_features
        # Add dropout before the final fully connected layer
        self.model.fc = nn.Sequential(
            nn.Dropout(p=dropout_prob),
            nn.Linear(num_ftrs, classes)
        )
        self.model.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        print("Resnet50")

    def forward(self, x):
        return self.model(x)


class ResNet18(nn.Module):
    def __init__(self, channels, classes,dropout_prob=0.5, interpolation=False):
        super(ResNet18, self).__init__()
        self.model = models.resnet18(pretrained=False)
        # Adjust the first convolutional layer to accept 1 input channel
        if interpolation:
            self.model.conv1 = nn.Conv2d(channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        else:
            self.model.conv1 = nn.Conv2d(channels, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Dropout(p=dropout_prob),
            nn.Linear(num_ftrs, classes)
        )
        self.model.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        print("Resnet18")

    def forward(self, x):
        return self.model(x)
