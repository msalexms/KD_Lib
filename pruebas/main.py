import os
import torch
from data.loaders import get_data_loaders
from data.transforms import get_transform
from model.resnet import ResNet50, ResNet18
from train.train import train_models
from utils.wandb_utils import initialize_wandb, finish_wandb

# https://github.com/tjmoon0104/Tiny-ImageNet-Classifier/blob/master/ResNet18_224.ipynb
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

data_dir = 'tiny-imagenet-200/'
learning_rate = 0.001
batch_size = 256
epoch_teacher = 150
epoch_student = 100
decay = 0.001
dropout = 0.8
momentum = 0.9
lr_decay = 10
optimizador = "SGD"
number = 14
method_name = "VanillaKD"
experiment = f"{method_name}-{number}"

print("Parámetros:")
print(f"data_dir: {data_dir}")
print(f"learning_rate: {learning_rate}")
print(f"batch_size: {batch_size}")
print(f"epoch_teacher: {epoch_teacher}")
print(f"epoch_student: {epoch_student}")
print(f"decay: {decay}")
print(f"dropout: {dropout}")
print(f"momentum: {momentum}")
print(f"lr_decay: {lr_decay}")
print(f"optimizador: {optimizador}")
print(f"number: {number}")
print(f"method_name: {method_name}")
print(f"experiment: {experiment}")

initialize_wandb(experiment, learning_rate, "resnet50-18", "ImagenetTiny", (epoch_teacher, epoch_student),
                 batch_size, optimizador, decay, dropout, momentum, lr_decay)

transform = get_transform()
train_loader, val_loader, test_loader = get_data_loaders(data_dir, batch_size, interpolation=False)

teacher_model = ResNet50(3, len(train_loader.dataset.classes), pretrained=True, dropout_prob=dropout)
student_model = ResNet18(3, len(train_loader.dataset.classes))

distiller = train_models(method_name=method_name ,teacher_model=teacher_model, student_model=student_model, train_loader=train_loader,
                         val_loader=val_loader,epochs_teacher=epoch_teacher, epochs_student=epoch_student,
                         learning_rate=learning_rate,
                         decay=decay, momentum=momentum, lr_decay=lr_decay, experiment=experiment,
                         optimizer_name=optimizador)

finish_wandb()
