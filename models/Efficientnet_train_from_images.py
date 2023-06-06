import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import torchvision.models as models
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim 
from sklearn.metrics import confusion_matrix
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt

# load the pretrained model
model = models.efficientnet_b0(pretrained=True)
device = 'cpu'
batch_size = 100
num_classes = 205
num_epochs = 20

def train():
    # the training transforms
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5]
        )
    ])

    # the validation transforms
    valid_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5]
        )
    ])
    
    # Use path of images to be used for training and validation
    train_dataset = torchvision.datasets.ImageFolder(root='SPLITIMAGES/train', transform=train_transform)
    valid_dataset = torchvision.datasets.ImageFolder(root='SPLITIMAGES/val', transform=valid_transform)

    # class names
    classes = list(train_dataset.class_to_idx.keys())
    classes.sort()

    # Train and validation loader
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    # Unfreeze model weights
    for param in model.parameters():
        param.requires_grad = False

    num_inputs = model.classifier[1].in_features

    model.classifier = nn.Sequential(
        nn.Linear(num_inputs, 2048),
        nn.SiLU(), # Sigmoid Weighted Linear Unit
        nn.Dropout(0.2),
        # Note that the last layer is 2048 * Number of Classes
        # Reshape the final layer(s) to have the same number of outputs as the number of classes in the new dataset
        nn.Linear(2048, num_classes)
    )

    # Define optimiser and loss function
    optimizer = optim.Adam(model.classifier.parameters())
    loss_func = nn.CrossEntropyLoss()

    # Train the model
    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1}/{num_epochs}')
        predictions = []
        true_values = []
        train_size = 0
        val_size = 0
        running_loss = 0.0
        running_corrects = 0

        model.train()

        for inputs, labels in tqdm(train_loader):
            train_size += inputs.size(0)
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()

            with torch.set_grad_enabled(True):
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = loss_func(outputs, labels)

                loss.backward()
                optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / train_size
        epoch_acc = running_corrects.double() / train_size
        print("Train size: ", train_size)
        print('Train Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))

        # Validation phase
        running_loss = 0.0
        running_corrects = 0
       
        model.eval()  # set the model to evaluation mode

        for inputs, labels in tqdm(valid_loader):
            val_size += inputs.size(0)
            inputs = inputs.to(device)
            labels = labels.to(device)
            true_values.extend(labels.numpy())
            with torch.set_grad_enabled(False):
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                predictions.extend(preds.numpy())
                loss = loss_func(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / val_size
        epoch_acc = running_corrects.double() / val_size
        print("Val size: ", val_size)
        print('Val Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))

        # Generate confusion matrix
        cf_matrix = confusion_matrix(true_values, predictions)
        df_cm = pd.DataFrame(cf_matrix / np.sum(cf_matrix, axis=1)[:, None], index = [i for i in classes],
                        columns = [i for i in classes])
        df_cm.to_csv('epoch' + str(epoch + 1) + '.csv')
    
    # Save model
    torch.save(model, 'efficient_net.pt')

if __name__ == '__main__':
    train()