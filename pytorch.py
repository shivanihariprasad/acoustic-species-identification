
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
ssl._create_default_https_context = ssl._create_unverified_context
if __name__ == '__main__':
    # Define the device to be used for training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Set up the transform to resize and normalize the images
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    # Create the dataset
    dataset = torchvision.datasets.ImageFolder('C:\\Users\\shiva\\Desktop\\Spring_2023\\237D\\PyHa\\IMAGES_HighPassFilter', transform=transform)
    #print(dataset.targets)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    # Create the data loaders for training and validation
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True,num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=True,num_workers=4)
    # Define the VGG model

    model = torchvision.models.vgg16(pretrained=True)
    num_features = model.classifier[0].in_features
    model.classifier = nn.Sequential(
        nn.Linear(num_features, 2048),
        nn.ReLU(inplace=True),
        nn.Dropout(),
        nn.Linear(2048, 2048),
        nn.ReLU(inplace=True),
        nn.Dropout(),
        nn.Linear(2048, 6)
    )
    model.to(device)
    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.classifier.parameters())


    # Train the model
    # num_epochs = 10
    # for epoch in range(num_epochs):
    #     running_loss = 0.0
    #     for i, (inputs, labels) in tqdm(train_loader):
    #         inputs, labels = inputs.to(device), labels.to(device)

    #         optimizer.zero_grad()

    #         outputs = model(inputs)
    #         loss = criterion(outputs, labels)
    #         loss.backward()
    #         optimizer.step()

    #         running_loss += loss.item()

    #         if i % 50 == 49:
    #             print('[%d, %5d] loss: %.3f' %
    #                 (epoch + 1, i + 1, running_loss / 50))
    #             running_loss = 0.0

    #     # Compute train and validation accuracies
    #     correct_train = 0
    #     total_train = 0
    #     with torch.no_grad():
    #         for inputs, labels in tqdm(train_loader):
    #             inputs, labels = inputs.to(device), labels.to(device)
    #             outputs = model(inputs)
    #             _, predicted = torch.max(outputs.data, 1)
    #             total_train += labels.size(0)
    #             correct_train += (predicted == labels).sum().item()
    #     train_accuracy = 100 * correct_train / total_train

    #     correct_valid = 0
    #     total_valid = 0
    #     with torch.no_grad():
    #         for inputs, labels in tqdm(val_loader):
    #             inputs, labels = inputs.to(device), labels.to(device)
    #             outputs = model(inputs)
    #             _, predicted = torch.max(outputs.data, 1)
    #             total_valid += labels.size(0)
    #             correct_valid += (predicted == labels).sum().item()
    #     valid_accuracy = 100 * correct_valid / total_valid

    #     # Print progress
    #     print('Epoch [%d/%d], Loss: %.4f, Train Accuracy: %.2f%%, Valid Accuracy: %.2f%%'
    #         % (epoch+1, num_epochs, loss.item(), train_accuracy, valid_accuracy))
        # Train the model
    for epoch in range(20):
        print('Epoch {}/{}'.format(epoch + 1, 20))
        print('-' * 10)

        running_loss = 0.0
        running_corrects = 0

        model.train()

        for inputs, labels in tqdm(train_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            with torch.set_grad_enabled(True):
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                #print(preds)
                #print(labels)
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

        for inputs, labels in tqdm(val_loader):
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

