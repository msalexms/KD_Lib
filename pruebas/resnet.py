import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from KD_Lib.KD import VanillaKD,LabelSmoothReg
import wandb
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# wandb.init(sync_tensorboard=False,
#     project='kd',
#     entity='gram-uah',
#     id='LabelSmoothReg-02',
#     config={
#         "tensorBoard": True,
#         "learning_rate": 0.001,
#         "architecture": "resnet50-18",
#         "dataset": "MNIST",
#         "epochs": (10, 5),
#     }
# )
#
# wandb.define_metric("teacher/epoch")
# wandb.define_metric("teacher/*", step_metric="teacher/epoch")
# wandb.define_metric("student/epoch")
# wandb.define_metric("student/*", step_metric="student/epoch")


transform = transforms.Compose([
    transforms.Resize(256),                          # Redimensionar la imagen a 256x256
    transforms.CenterCrop(224),                      # Recortar la imagen al centro a un tamaño de 224x224
    transforms.ToTensor(),                           # Convertir la imagen a un tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], # Normalizar la imagen con la media y la desviación estándar
                         std=[0.229, 0.224, 0.225])
])

train_loader = torch.utils.data.DataLoader(
    datasets.MNIST(
        "mnist_data",
        train=True,
        download=True,
        transform=transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        ),
    ),
    batch_size=512,
    shuffle=True,
)

test_loader = torch.utils.data.DataLoader(
    datasets.MNIST(
        "mnist_data",
        train=False,
        transform=transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        ),
    ),
    batch_size=512,
    shuffle=True,
)


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
    def __init__(self,chanels, classes):
        super(ResNet50, self).__init__()
        self.model = models.resnet50(pretrained=True)
        # Adjust the first convolutional layer to accept 1 input channel
        self.model.conv1 = nn.Conv2d(chanels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, classes)
        print("Resnet50")

    def forward(self, x):
        return self.model(x)


class ResNet18(nn.Module):
    def __init__(self,chanels, classes):
        super(ResNet18, self).__init__()
        self.model = models.resnet18(pretrained=False)
        # Adjust the first convolutional layer to accept 1 input channel
        self.model.conv1 = nn.Conv2d(chanels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, classes)
        print("Resnet18")

    def forward(self, x):
        return self.model(x)


teacher_model = ResNet50(1, 10)


student_model = ResNet18(1, 10)
# wandb.watch(teacher_model, log_freq=5, idx=0)
# wandb.watch(student_model, log_freq=5, idx=1)

teacher_optimizer = optim.SGD(teacher_model.parameters(), lr=0.001)
student_optimizer = optim.SGD(student_model.parameters(), lr=0.001)

distiller = LabelSmoothReg(teacher_model, student_model, train_loader, test_loader, teacher_optimizer,
                           student_optimizer, correct_prob=0.9, device='cuda')
distiller.train_teacher(epochs=10)                                       # Train the teacher model
distiller.train_student(epochs=5)                                      # Train the student model
distiller.evaluate(teacher=True)                                        # Evaluate the teacher model
distiller.evaluate()

# distiller = VanillaKD(teacher_model, student_model, train_loader, test_loader,
#                       teacher_optimizer, student_optimizer, device='cuda', log=False)
#
# distiller.train_teacher(epochs=10, plot_losses=True, save_model=False)
#
# distiller.train_student(epochs=5, plot_losses=True, save_model=False)
#
# distiller.evaluate(teacher=False)
#
# distiller.get_parameters()
#
# wandb.finish()
