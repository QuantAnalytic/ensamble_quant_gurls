from torch.utils.data import Dataset
from typing import Tuple
import torch


class TaskDataset(Dataset):
    def __init__(self, transform=None):
        self.ids = []
        self.imgs = []
        self.labels = []
        self.transform = transform

    def add_image(self, image, label):
        self.ids.append(len(self.ids))  # Dodaj unikalny ID oparty na aktualnej długości listy
        self.imgs.append(image)  # Dodaj obraz
        self.labels.append(label)  # Dodaj etykietę

    def __getitem__(self, index) -> Tuple[int, torch.Tensor, int]:
        id_ = self.ids[index]
        img = self.imgs[index]
        if self.transform is not None:
            img = self.transform(img)
        label = self.labels[index]
        return id_, img, label

    def __len__(self):
        return len(self.ids)
