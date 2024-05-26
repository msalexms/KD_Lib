import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from KD_Lib.KD import VanillaKD, LabelSmoothReg, ProbShift
import wandb
import os
from CustomDataset import CustomDataset

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

learning_rate = 0.001
batch_size = 768
epoch_teacher = 150
epoch_student = 100
decay = 0.001
dropout = 0.5
momentum = 0.9
optimizador = "SGD"
experiment = "ProbShift-IN1"

if True:
    wandb.init(sync_tensorboard=False,
               project='kd',
               entity='gram-uah',
               id=experiment,
               config={
                   "kd_method": "VanillaKD",
                   "tensorBoard": False,
                   "learning_rate": learning_rate,
                   "architecture": "resnet50-18",
                   "dataset": "ImagenetTiny",
                   "epochs": (epoch_teacher, epoch_student),
                   "batch_size": batch_size,
                   "optim": optimizador,
                   "regularizacion l2": decay,
                   "dropout": dropout,
                   "momentum": momentum,
               }
               )

    wandb.define_metric("teacher/epoch")
    wandb.define_metric("teacher/*", step_metric="teacher/epoch")
    wandb.define_metric("student/epoch")
    wandb.define_metric("student/*", step_metric="student/epoch")

# Definir las transformaciones que deseas aplicar a las imágenes
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Ruta al directorio raíz del conjunto de datos ImageNet Tiny
data_dir = 'tiny-imagenet-200/'

# Crear el conjunto de datos de ImageFolder para train, val y test
train_dataset = datasets.ImageFolder(root=os.path.join(data_dir, 'train'),
                                     transform=transform,
                                     is_valid_file=lambda file: file.lower().endswith('.jpeg'))
print(train_dataset)
print(f"CLASES: {len(train_dataset.classes)}")
print(f"CLASES: {train_dataset.classes}")
print(f"CLASES: {train_dataset.class_to_idx}")


test_dataset = datasets.ImageFolder(root=os.path.join(data_dir, 'test'),
                                    transform=transform,
                                    is_valid_file=lambda file: file.lower().endswith('.jpeg'))
print(test_dataset)

val_dataset = CustomDataset(root_dir=os.path.join(data_dir, 'val', 'images'),
                            annotations_file=os.path.join(data_dir, 'val', 'val_annotations.txt'),
                            transform=transform,
                            label_map=train_dataset.class_to_idx,
                            interpolation=True)
print(len(val_dataset.labels))

# Crear los DataLoaders para train, val y test


train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


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
    def __init__(self, channels, classes, pretrained, dropout_prob=0.5, weights="../weights/resnet50-11ad3fa6.pth"):
        super(ResNet50, self).__init__()
        if pretrained:
            self.model = models.resnet50(pretrained=True)
        else:
            print("Custom Weights")
            self.model = models.resnet50(pretrained=False)
            state_dict = torch.load(weights)
            self.model.load_state_dict(state_dict=state_dict)
        # Adjust the first convolutional layer to accept 'channels' input channels
        self.model.conv1 = nn.Conv2d(channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        num_ftrs = self.model.fc.in_features
        # Add dropout before the final fully connected layer
        self.model.fc = nn.Sequential(
            nn.Dropout(p=dropout_prob),
            nn.Linear(num_ftrs, classes)
        )
        self.model.avgpool = nn.AdaptiveAvgPool1d(1)
        print("Resnet50")

    def forward(self, x):
        return self.model(x)


class ResNet18(nn.Module):
    def __init__(self, chanels, classes):
        super(ResNet18, self).__init__()
        self.model = models.resnet18(pretrained=False)
        # Adjust the first convolutional layer to accept 1 input channel
        self.model.conv1 = nn.Conv2d(chanels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, classes)
        print("Resnet18")

    def forward(self, x):
        return self.model(x)


teacher_model = ResNet50(3, len(train_dataset.classes), pretrained=True, dropout_prob=dropout)

student_model = ResNet18(3, len(train_dataset.classes))
# wandb.watch(teacher_model, log_freq=5, idx=0)
# wandb.watch(student_model, log_freq=5, idx=1)

teacher_optimizer = optim.SGD(teacher_model.parameters(), lr=learning_rate, weight_decay=decay)
student_optimizer = optim.SGD(student_model.parameters(), lr=learning_rate, weight_decay=decay)

# distiller = LabelSmoothReg(teacher_model, student_model, train_loader, test_loader, teacher_optimizer,
#                            student_optimizer, correct_prob=0.9, device='cuda')
# distiller.train_teacher(epochs=10)                                       # Train the teacher model
# distiller.train_student(epochs=5)                                      # Train the student model
# distiller.evaluate(teacher=True)                                        # Evaluate the teacher model
# distiller.evaluate()

distiller = ProbShift(teacher_model, student_model, train_loader, val_loader,
                      teacher_optimizer, student_optimizer, device='cuda', log=False)

distiller.train_teacher(epochs=epoch_teacher, plot_losses=False, save_model=True, save_model_pth=f"./models/teacher_{experiment}.pt")

distiller.train_student(epochs=epoch_student, plot_losses=False, save_model=True, save_model_pth=f"./models/student_{experiment}.pt")

distiller.evaluate(teacher=False)

distiller.evaluate(teacher=True)

distiller.get_parameters()

wandb.finish()
