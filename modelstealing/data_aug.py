from __future__ import print_function

import matplotlib.pyplot as plt
from torchvision import transforms
from taskdataset import TaskDataset

import torch
from torchvision import transforms

dataset_path = '/home/hack10/task1/ModelStealingPub.pt'

dataset = torch.load(dataset_path)

transform = transforms.Compose([
    transforms.RandomRotation(45),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor(),
])


task_dataset = TaskDataset()

for image, label in zip(dataset.imgs, dataset.labels):
    task_dataset.add_image(image, label)

print(f'Liczba elementów w zbiorze danych: {len(task_dataset)}')
# Stosowanie transformacji i dodawanie zaugmentowanych obrazów do datasetu
for image, label in zip(dataset.imgs, dataset.labels):
    transformed_image = transform(image)
    task_dataset.add_image(transformed_image, label)

# W tym momencie `task_dataset` zawiera oryginalne oraz zaugmentowane obrazy
print(f'Liczba elementów w zbiorze danych: {len(task_dataset)}')

"""
# PyTorch tensor (C x H x W) należy przekonwertować na format numpy (H x W x C) dla Matplotlib
# Należy również "odnormalizować" obraz
transformed_image = transformed_image.numpy().transpose((1, 2, 0))  # Przekształć tensor do formatu (H x W x C)

# Wyświetl obraz
plt.imshow(transformed_image)
plt.title(f'fajne')
plt.show()
plt.savefig(f'image.png', bbox_inches='tight', pad_inches=0)"""