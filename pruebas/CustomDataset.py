import os
from PIL import Image
from torch.utils.data import Dataset
import torch
import cv2

class CustomDataset(Dataset):
    def __init__(self, root_dir, annotations_file, label_map, transform=None, interpolation=False):
        self.root_dir = root_dir
        self.annotations_file = annotations_file
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self.label_map = label_map  # Diccionario de mapeo de etiquetas
        self.interpolation = interpolation

        # Leer las anotaciones del archivo
        with open(annotations_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                parts = line.split()
                image_name = parts[0]
                label = parts[1]
                self.image_paths.append(os.path.join(root_dir, image_name))
                # Mapear la etiqueta al valor correspondiente en el diccionario
                label = label_map[label]
                self.labels.append(label)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        if not self.interpolation:
            img = Image.open(img_path).convert('RGB')
        else:
            img = cv2.imread(img_path)
            img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_CUBIC)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)


        if self.transform:
            img = self.transform(img)

        label = torch.tensor(label)

        return img, label
