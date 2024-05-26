from collections import OrderedDict
import torch
import torch.nn as nn
from PIL import Image
from torchvision import datasets, transforms, models
import os

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "3"
#
# class ResNet50(nn.Module):
#     def __init__(self, channels, classes, pretrained, weights="../weights/resnet50-11ad3fa6.pth"):
#         super(ResNet50, self).__init__()
#         if pretrained:
#             self.model = models.resnet50(pretrained=True)
#             self.model.conv1 = nn.Conv2d(channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
#             num_ftrs = self.model.fc.in_features
#             self.model.fc = nn.Linear(num_ftrs, classes)
#         else:
#             print("Custom Weights")
#             self.model = models.resnet50(pretrained=False)
#             state_dict = torch.load(weights)
#             # Eliminar el prefijo "model." de las claves del estado
#             new_state_dict = {}
#             for key, value in state_dict.items():
#                 if key.startswith("model."):
#                     new_key = key[len("model."):]  # Eliminar el prefijo "model."
#                     new_state_dict[new_key] = value
#                 else:
#                     new_state_dict[key] = value
#             new_state_dict_ordered = OrderedDict(new_state_dict)
#             self.model.conv1 = nn.Conv2d(channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
#             num_ftrs = self.model.fc.in_features
#             self.model.fc = nn.Linear(num_ftrs, classes)
#             self.model.load_state_dict(state_dict=new_state_dict_ordered)
#         # Adjust the first convolutional layer to accept 1 input channel
#         print("Resnet50")
#
#     def forward(self, x):
#         return self.model(x)
#
#
# model = ResNet50(channels=3, classes=200, pretrained=False, weights='./models/teacher_VanillaKD-IN8.pt')
#
# model.eval()
#
# transform = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# ])
#
# # Cargar la imagen y aplicar la transformación
# image = transform(Image.open('./tiny-imagenet-200/train/n01443537/images/n01443537_1.JPEG')).unsqueeze(0)
#
# # Mover la imagen a la GPU si es necesario
# if torch.cuda.is_available():
#     print("CUDA")
#     model = model.cuda()
#     image = image.cuda()
#
# # Obtener los resultados
# resultados = model(image)
#
# print(resultados) #n03599486
#
# probabilidades = torch.softmax(resultados, dim=1)
#
# print("Probabilidades normalizadas:", probabilidades)
#
# valor_maximo, indice_maximo = probabilidades.max(1)
# print("Índice de la clase con la mayor probabilidad:", indice_maximo.item())
# print("Valor de la probabilidad:", valor_maximo.item())


random_tensor = torch.rand(512, 1)
random_tensor1 = torch.rand(512)
print(random_tensor1)
print(random_tensor1.view_as(random_tensor))
