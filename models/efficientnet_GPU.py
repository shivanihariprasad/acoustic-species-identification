import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
#from efficientnet_pytorch import EfficientNet
import torchvision.models as models
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim 
from sklearn.metrics import confusion_matrix
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report

#model = EfficientNet.from_name('efficientnet-b1', num_classes=29)
#model = models.efficientnet_b0(pretrained=True)
model = models.efficientnet_b7(weights=models.efficientnet.EfficientNet_B7_Weights.DEFAULT)
device = 'cuda'
batch_size = 50
test_batch_size = 25
num_classes = 264
num_epochs = 15
START_LR = 0.0001
accuracy_list = []

def train():
    # the training transforms
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        #transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
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
    
    train_dataset = torchvision.datasets.ImageFolder(root='SPLITIMAGES/train', transform=train_transform)
    valid_dataset = torchvision.datasets.ImageFolder(root='SPLITIMAGES/val', transform=valid_transform)
    #print("Train dataset\n", train_dataset.class_to_idx)
    #print("Valid dataset\n", valid_dataset.class_to_idx)
    classes = list(train_dataset.class_to_idx.keys())
    classes.sort()
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset, batch_size=test_batch_size, shuffle=True, num_workers=4)

    #data_dir = "./IMAGES/"
    #image_datasets = torchvision.datasets.ImageFolder(data_dir, train_transform)
    #train_size = int(0.8 * len(image_datasets))
    #val_size = len(image_datasets) - train_size
    #train_size = 4758
    #train_size = 43417
    #val_size = 1204
    #val_size = 10951
    #train_dataset, val_dataset = torch.utils.data.random_split(image_datasets, [train_size, val_size])

    #train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=25, shuffle=True, num_workers=7)
    #valid_loader = torch.utils.data.DataLoader(val_dataset, batch_size=25, shuffle=False, num_workers=1)

    # Unfreeze model weights
    for param in model.parameters():
        param.requires_grad = True

    num_inputs = model.classifier[1].in_features
    #print("Num inputs:", num_inputs)
    model.classifier = nn.Sequential(
        nn.Linear(num_inputs, 2048),
        nn.SiLU(), # Sigmoid Weighted Linear Unit
        nn.Dropout(0.2),
        # Note that the last layer is 2048 * Number of Classes
        # Reshape the final layer(s) to have the same number of outputs as the number of classes in the new dataset
        nn.Linear(2048, num_classes)
    )



    #num_ftrs = model._fc.in_features
    #num_ftrs = model.fc.in_features
    #model._fc = nn.Linear(num_ftrs, 29)

    #optimizer = optim.Adam(model.parameters())
    optimizer = optim.Adam(model.classifier.parameters(), lr=START_LR)
    loss_func = nn.CrossEntropyLoss()
    model.cuda()

    # Train the model
    for epoch in range(num_epochs):
        torch.cuda.empty_cache()
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
            true_values.extend(labels.cpu().numpy())
            optimizer.zero_grad()


            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            predictions.extend(preds.cpu().numpy())
            loss = loss_func(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

            del outputs
            del preds
            del labels

        epoch_loss = running_loss / train_size
        epoch_acc = running_corrects.double() / train_size
        report_dict = classification_report(true_values, predictions, target_names=classes,output_dict=True)
        report_pd = pd.DataFrame(report_dict)
        report_pd.to_csv('training-classification-epoch' + str(epoch + 1) + '.csv')
        cnf_matrix = confusion_matrix(true_values, predictions)
        df_cm = pd.DataFrame(cnf_matrix / np.sum(cnf_matrix, axis=1)[:, None], index = [i for i in classes],
                        columns = [i for i in classes])
        df_cm.to_csv('train-confusion-matrix-epoch' + str(epoch + 1) + '.csv')        #acc = matrix.diagonal()/matrix.sum(axis=1)
        FP = cnf_matrix.sum(axis=0) - np.diag(cnf_matrix) 
        FN = cnf_matrix.sum(axis=1) - np.diag(cnf_matrix)
        TP = np.diag(cnf_matrix)
        TN = cnf_matrix.sum() - (FP + FN + TP)
        FP = FP.astype(float)
        FN = FN.astype(float)
        TP = TP.astype(float)
        TN = TN.astype(float)
        ACC = (TP+TN)/(TP+FP+FN+TN)
        TPR = TP/(TP+FN)
        PPV = TP/(TP+FP)
        pd.DataFrame(ACC, columns=['Accuracy']).to_csv('accuracy-train-epoch' + str(epoch + 1) + '.csv')
        # print("accuracy for all classes in train phase", ACC)
        # print("recall for all classes in train phase", TPR)
        # print("precision for all classes in train phase", PPV)
        pd.DataFrame(ACC, columns=['Accuracy']).to_csv('accuracy-train-epoch' + str(epoch + 1) + '.csv')
        print('Train Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))
        print("Train size: ", train_size)
        print('Train Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))
        a = 'Epoch: {} Train Loss: {:.4f} Acc: {:.4f}'.format(epoch + 1, epoch_loss, epoch_acc)
        accuracy_list.append(a)

        torch.cuda.empty_cache()

        # Validation phase
        running_loss = 0.0
        running_corrects = 0
        predictions = []
        true_values = []
        model.eval()  # set the model to evaluation mode

        for inputs, labels in tqdm(valid_loader):
            val_size += inputs.size(0)
            inputs = inputs.to(device)
            labels = labels.to(device)
            true_values.extend(labels.cpu().numpy())
            
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            predictions.extend(preds.cpu().numpy())
            loss = loss_func(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

            del outputs
            del preds
            del labels

        epoch_loss = running_loss / val_size
        epoch_acc = running_corrects.double() / val_size
        print("Val size: ", val_size)
        print('Val Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))
        a = 'Epoch: {} Val Loss: {:.4f} Acc: {:.4f}'.format(epoch + 1, epoch_loss, epoch_acc)
        accuracy_list.append(a)
        FP = cnf_matrix.sum(axis=0) - np.diag(cnf_matrix) 
        FN = cnf_matrix.sum(axis=1) - np.diag(cnf_matrix)
        TP = np.diag(cnf_matrix)
        TN = cnf_matrix.sum() - (FP + FN + TP)
        FP = FP.astype(float)
        FN = FN.astype(float)
        TP = TP.astype(float)
        TN = TN.astype(float)
        ACC = (TP+TN)/(TP+FP+FN+TN)
        TPR = TP/(TP+FN)
        PPV = TP/(TP+FP)
        pd.DataFrame(ACC, columns=['Accuracy']).to_csv('accuracy-val-epoch' + str(epoch + 1) + '.csv')
        # Print confusion matrix
        report_dict = classification_report(true_values, predictions, target_names=classes,output_dict=True)
        report_pd = pd.DataFrame(report_dict)
        report_pd.to_csv('val-classification-epoch' + str(epoch + 1) + '.csv')
        cf_matrix = confusion_matrix(true_values, predictions)
        df_cm = pd.DataFrame(cf_matrix / np.sum(cf_matrix, axis=1)[:, None], index = [i for i in classes],
                        columns = [i for i in classes])
        df_cm.to_csv('val-confusion-matrix-epoch' + str(epoch + 1) + '.csv')
        #plt.figure(figsize = (12,7))
        #sn.heatmap(df_cm, annot=True)
        #plt.savefig('output.png')
        #plt.show()
    
    torch.save(model, 'efficient_netb7_264.pt')
    # open file in write mode
    with open('accuracies_b7_264_0.0001.txt', 'w') as fp:
        for item in accuracy_list:
            # write each item on a new line
            fp.write("%s\n" % item)

if __name__ == '__main__':
    train()