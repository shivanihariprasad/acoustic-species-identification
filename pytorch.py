
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.transforms import transforms
import os
from PIL import Image
import torch
import ssl
import torchvision
import torchvision.models as models
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
import splitfolders
def calculate_accuracy(y_pred, y):
    top_pred = y_pred.argmax(1, keepdim=True)
    correct = top_pred.eq(y.view_as(top_pred)).sum()
    acc = correct.float() / y.shape[0]
    return acc

ssl._create_default_https_context = ssl._create_unverified_context
if __name__ == '__main__':
    # Define the device to be used for training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Set up the transform to resize and normalize the images
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5]),
    ])

    # Create the dataset
    # dataset = torchvision.datasets.ImageFolder('C:\\Users\\shiva\\Desktop\\Spring_2023\\237D\\PyHa\\IMAGES', transform=transform)
    # #print(dataset.targets)
   
    # train_data, test_data, train_labels, test_labels = train_test_split(dataset, labels, test_size=0.2, random_state=42)
    # train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    input_folder = "C:\\Users\\shiva\\Desktop\\Spring_2023\\237D\\PyHa\\IMAGES"
    output_folder = "C:\\Users\\shiva\\Desktop\\Spring_2023\\237D\\PyHa\\IMAGES_Split"
    splitfolders.ratio(input_folder, output_folder, seed=42, ratio=(0.8, 0.2), group_prefix=None)
    # Create datasets for the training and testing sets
    train_dataset = torchvision.datasets.ImageFolder(output_folder + '/train', transform=transform)
    val_dataset = torchvision.datasets.ImageFolder(output_folder + '/val', transform=transform)
    train_size = len(train_dataset)
    val_size = len(val_dataset)
    # Create the data loaders for training and validation
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True,num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=True,num_workers=4)
    # Define the VGG model

    model = torchvision.models.vgg16(pretrained=True)
    num_features = model.classifier[0].in_features
    model.classifier = nn.Sequential(
        nn.Linear(num_features, 4096),
        nn.ReLU(inplace=True),
        nn.Dropout(0.5),
        nn.Linear(4096, 4096),
        nn.ReLU(inplace=True),
        nn.Dropout(0.5),
        nn.Linear(4096, 205)
    )
    #   model.to(device)
    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    START_LR = 1e-7
    optimizer = optim.Adam(model.parameters(), lr=START_LR)
    model = model.to(device)
    criterion = criterion.to(device)

        # Train the model
    for epoch in range(10):
        print('Epoch {}/{}'.format(epoch + 1, 10))
        print('-' * 10)

        running_loss = 0
        running_corrects = 0

        model.train()

        for inputs, labels in tqdm(train_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            #acc = calculate_accuracy(y_pred, labels)

            loss.backward()

            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / train_size
        epoch_acc = running_corrects.double() / train_size

        print('Train Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))

        # Validation phase
        running_loss = 0
        running_corrects = 0
        model.eval()  # set the model to evaluation mode

        for inputs, labels in tqdm(val_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            #acc = calculate_accuracy(preds, labels)

            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / val_size
        epoch_acc = running_corrects.double() / val_size

        print('Val Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))

    # Save the model
    torch.save(model, 'vgg16_model.pth')

