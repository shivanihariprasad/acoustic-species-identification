# -*- coding: utf-8 -*-
"""resnet_pytorch.ipynb
Run on Google Colab.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models, datasets
import numpy as np

path_folders = os.listdir("./drive/MyDrive/IMAGES2/")
IM_SIZE = (224, 224)
print(path_folders)
path_folders.remove('.DS_Store')
print(len(path_folders))

data_transforms = transforms.Compose([
    transforms.RandomResizedCrop(IM_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# Path to folder containing spectogram images
data_dir = './drive/MyDrive/IMAGES2/'

image_datasets = datasets.ImageFolder(data_dir, data_transforms)
train_size = int(0.8 * len(image_datasets))
val_size = len(image_datasets) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(image_datasets, [train_size, val_size])

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=25, shuffle=True, num_workers=7)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=25, shuffle=False, num_workers=7)

num_classes = len((torch.tensor(image_datasets.targets)))
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# load the pretrained model
base_model = models.resnet50(pretrained=True)
num_ftrs = base_model.fc.in_features
base_model.fc = nn.Identity()

classifier = nn.Sequential(
    nn.Linear(num_ftrs, num_classes),
    nn.Softmax(dim=1)
)

model = nn.Sequential(
    base_model,
    classifier
)

model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())

# Train the model
for epoch in range(20):
    print('Epoch {}/{}'.format(epoch + 1, 20))
    print('-' * 10)

    running_loss = 0.0
    running_corrects = 0

    model.train()

    for inputs, labels in train_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        with torch.set_grad_enabled(True):
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

    epoch_loss = running_loss / train_size
    epoch_acc = running_corrects.double() / train_size

    print('Train Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))

    # Validation phase
    running_loss = 0.0
    running_corrects = 0

    model.eval()  # set the model to evaluation mode

    for inputs, labels in val_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        with torch.set_grad_enabled(False):
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

    epoch_loss = running_loss / val_size
    epoch_acc = running_corrects.double() / val_size

    print('Val Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))