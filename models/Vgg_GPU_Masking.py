import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
from torchvision import models as models
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms
import torchvision
import torchvision.models as models
import torch.optim as optim
import torch.nn as nn
from sklearn.metrics import confusion_matrix
import scipy.signal as scipy_signal
import os
import ssl
import librosa
from sklearn.metrics import classification_report
from collections import Counter
import torchaudio
import random

# set computation device
device = 'cuda'
num_epochs = 10
num_classes = 264
ssl._create_default_https_context = ssl._create_unverified_context

classes = ['abethr1', 'abhori1', 'abythr1', 'afbfly1', 'afdfly1', 'afecuc1', 'affeag1', 'afgfly1', 'afghor1', 'afmdov1', 'afpfly1', 'afpkin1', 'afpwag1', 'afrgos1', 'afrgrp1', 'afrjac1', 'afrthr1', 'amesun2', 'augbuz1', 'bagwea1', 'barswa', 'bawhor2', 'bawman1', 'bcbeat1', 'beasun2', 'bkctch1', 'bkfruw1', 'blacra1', 'blacuc1', 'blakit1', 'blaplo1', 'blbpuf2', 'blcapa2', 'blfbus1', 'blhgon1', 'blhher1', 'blksaw1', 'blnmou1', 'blnwea1', 'bltapa1', 'bltbar1', 'bltori1', 'blwlap1', 'brcale1', 'brcsta1', 'brctch1', 'brcwea1', 'brican1', 'brobab1', 'broman1', 'brosun1', 'brrwhe3', 'brtcha1', 'brubru1', 'brwwar1', 'bswdov1', 'btweye2', 'bubwar2', 'butapa1', 'cabgre1', 'carcha1', 'carwoo1', 'categr', 'ccbeat1', 'chespa1', 'chewea1', 'chibat1', 'chtapa3', 'chucis1', 'cibwar1', 'cohmar1', 'colsun2', 'combul2', 'combuz1', 'comsan', 'crefra2', 'crheag1', 'crohor1', 'darbar1', 'darter3', 'didcuc1', 'dotbar1', 'dutdov1', 'easmog1', 'eaywag1', 'edcsun3', 'egygoo', 'equaka1', 'eswdov1', 'eubeat1', 'fatrav1', 'fatwid1', 'fislov1', 'fotdro5', 'gabgos2', 'gargan', 'gbesta1', 'gnbcam2', 'gnhsun1', 'gobbun1', 'gobsta5', 'gobwea1', 'golher1', 'grbcam1', 'grccra1', 'grecor', 'greegr', 'grewoo2', 'grwpyt1', 'gryapa1', 'grywrw1', 'gybfis1', 'gycwar3', 'gyhbus1', 'gyhkin1', 'gyhneg1', 'gyhspa1', 'gytbar1', 'hadibi1', 'hamerk1', 'hartur1', 'helgui', 'hipbab1', 'hoopoe', 'huncis1', 'hunsun2', 'joygre1', 'kerspa2', 'klacuc1', 'kvbsun1', 'laudov1', 'lawgol', 'lesmaw1', 'lessts1', 'libeat1', 'litegr', 'litswi1', 'litwea1', 'loceag1', 'lotcor1', 'lotlap1', 'luebus1', 'mabeat1', 'macshr1', 'malkin1', 'marsto1', 'marsun2', 'mcptit1', 'meypar1', 'moccha1', 'mouwag1', 'ndcsun2', 'nobfly1', 'norbro1', 'norcro1', 'norfis1', 'norpuf1', 'nubwoo1', 'pabspa1', 'palfly2', 'palpri1', 'piecro1', 'piekin1', 'pitwhy', 'purgre2', 'pygbat1', 'quailf1', 'ratcis1', 'raybar1', 'rbsrob1', 'rebfir2', 'rebhor1', 'reboxp1', 'reccor', 'reccuc1', 'reedov1', 'refbar2', 'refcro1', 'reftin1', 'refwar2', 'rehblu1', 'rehwea1', 'reisee2', 'rerswa1', 'rewsta1', 'rindov', 'rocmar2', 'rostur1', 'ruegls1', 'rufcha2', 'sacibi2', 'sccsun2', 'scrcha1', 'scthon1', 'shesta1', 'sichor1', 'sincis1', 'slbgre1', 'slcbou1', 'sltnig1', 'sobfly1', 'somgre1', 'somtit4', 'soucit1', 'soufis1', 'spemou2', 'spepig1', 'spewea1', 'spfbar1', 'spfwea1', 'spmthr1', 'spwlap1', 'squher1', 'strher', 'strsee1', 'stusta1', 'subbus1', 'supsta1', 'tacsun1', 'tafpri1', 'tamdov1', 'thrnig1', 'trobou1', 'varsun2', 'vibsta2', 'vilwea1', 'vimwea1', 'walsta1', 'wbgbir1', 'wbrcha2', 'wbswea1', 'wfbeat1', 'whbcan1', 'whbcou1', 'whbcro2', 'whbtit5', 'whbwea1', 'whbwhe3', 'whcpri2', 'whctur2', 'wheslf1', 'whhsaw1', 'whihel1', 'whrshr1', 'witswa1', 'wlwwar', 'wookin1', 'woosan', 'wtbeat1', 'yebapa1', 'yebbar1', 'yebduc1', 'yebere1', 'yebgre1', 'yebsto1', 'yeccan1', 'yefcan', 'yelbis1', 'yenspu1', 'yertin1', 'yesbar1', 'yespet1', 'yetgre1', 'yewgre1']

# read the data.csv file and get the audio paths, audio offset and duration and labels
df = pd.read_csv('dataset_264_aug.csv',index_col=False)

# train test split needs atleast 3 images to be present for a class so that the images gets split into train and val set
# Add duplicate entries to satisfy this requirement
df.loc[len(df.index)] = [230514, './kaggletest/afpkin1/', 'XC704863.ogg', 7.751995464852610, 0, 0.0, 5, 44100, 'bird', 1.0, 'NO_AUGMENT', 11]
df.loc[len(df.index)] = [230515,'./kaggletest/golher1/', 'XC248014.ogg', 8.28, 0, 0.0, 5, 44100, 'bird', 1.0, 'NO_AUGMENT', 102]
df.loc[len(df.index)] = [230516,'./kaggletest/whctur2/', 'XC444635.ogg', 9.508548752834470, 0, 0.0, 5, 44100, 'bird', 1.0, 'NO_AUGMENT', 239]

X = df.iloc[:, 1:-1]
Y = pd.DataFrame(df.iloc[:, -1])

image_shape = (224, 224)
(xtrain, xtest, ytrain, ytest) = (train_test_split(X, Y, test_size=0.2, random_state=42, shuffle=True, stratify=Y))
xtrain = xtrain.reset_index(drop=True)
ytrain = ytrain.reset_index(drop=True)
xtest = xtest.reset_index(drop=True)
ytest = ytest.reset_index(drop=True)

masks = np.arange(20,51)
augment_choice = [0,1,2]
f_masks = np.arange(10,31)

# Custom bird dataset class
class BirdAudioDataset(Dataset):
    def __init__(self, info, labels, tfms=None):
        self.folder = info['FOLDER']
        self.file = info['IN FILE']
        self.offset = info['OFFSET']
        self.duration = info['DURATION']
        self.label = labels['Label']
        self.augmentation = info['AUGMENT_TYPE']


    def __len__(self):
        return (len(self.file))

    def __getitem__(self, i):
        file_path = self.folder[i] + self.file[i]
        normalized_sample_rate=32000
        try:
            SIGNAL, SAMPLE_RATE = librosa.load(file_path, offset=self.offset[i], duration=self.duration[i], sr=None, mono=True)
            SIGNAL = SIGNAL * 32768
        except BaseException:
            print("Failed to load" + file_path)
                
        # Resample the audio if it isn't the normalized sample rate
        try:
            if SAMPLE_RATE != normalized_sample_rate:
                rate_ratio = normalized_sample_rate / SAMPLE_RATE
                SIGNAL = scipy_signal.resample(
                SIGNAL, int(len(SIGNAL) * rate_ratio))
                SAMPLE_RATE = normalized_sample_rate
        except Exception as e:
            print("Failed to Downsample" + file_path + str(e))
        
        # convert stereo to mono if needed
        # Might want to compare to just taking the first set of data.
        if len(SIGNAL.shape) == 2:
            SIGNAL = SIGNAL.sum(axis=1) / 2

        D = np.abs(librosa.stft(SIGNAL))**2
        # Assuming `D` and `normalized_sample_rate` are defined

        S = librosa.feature.melspectrogram(S=D, sr=normalized_sample_rate, fmax=normalized_sample_rate/2)
        if str(self.augmentation[i]) == "AUGMENT":
            choice = random.choice(augment_choice)
            if(choice == 0):
                S = torchaudio.transforms.TimeMasking(time_mask_param=random.choice(masks))(torch.from_numpy(S).unsqueeze(0))
                S = S.squeeze(0).numpy()
            elif (choice == 1):
                S = torchaudio.transforms.FrequencyMasking(freq_mask_param=random.choice(f_masks))(torch.from_numpy(S).unsqueeze(0))
                S = S.squeeze(0).numpy()
            else:
                S = librosa.effects.time_stretch(S, rate=1.25)
                

        S_dB = librosa.power_to_db(S, ref=np.max)


        # Scale the spectrogram to the range [0, 255]
        S_scaled = (S_dB - np.min(S_dB)) / (np.max(S_dB) - np.min(S_dB)) * 255
        S_scaled = S_scaled.astype('uint8')

        # Create a PIL image object from the scaled spectrogram
        pil_image = Image.fromarray(S_scaled)
        # Resize the PIL image to 224 x 224 and convert it to an RGB image
        pil_image = pil_image.resize((224, 224)).convert('RGB')
        transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5]),
    ])
        img_tensor = transform(pil_image)
        label = self.label[i]
        return img_tensor, torch.tensor(label, dtype=torch.long)

# Create train and test dataset
train_data = BirdAudioDataset(xtrain, ytrain, tfms=1)
test_data = BirdAudioDataset(xtest, ytest, tfms=0)

# Define the VGG model
def train():
    # load the pretrained model
    model = torchvision.models.vgg16(pretrained=True)
    train_loader = DataLoader(train_data, batch_size=10, shuffle=True, num_workers=4)
    val_loader = DataLoader(test_data, batch_size=5, shuffle=True, num_workers=4)
    num_features = model.classifier[0].in_features
    model.classifier = nn.Sequential(
        nn.Linear(num_features, 2048),
        nn.ReLU(inplace=True),
        nn.Dropout(0.5),
        nn.Linear(2048, 2048),
        nn.ReLU(inplace=True),
        nn.Dropout(0.5),
        nn.Linear(2048, num_classes)
    )

    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    START_LR = 0.0001
    optimizer = optim.Adam(model.parameters(), lr=START_LR)
    model = model.to(device)
    criterion = criterion.to(device)

    # Train the model
    for epoch in range(10):
        print('Epoch {}/{}'.format(epoch + 1, 10))
        print('-' * 10)
        running_loss = 0
        running_corrects = 0
        train_size = 0
        val_size = 0
        model.train()
        predictions = []
        true_labels = []
        for inputs, labels in tqdm(train_loader):
            train_size += inputs.size(0)
            inputs = inputs.to(device)
            labels = labels.to(device)
            true_labels.extend(labels.cpu().numpy())
            optimizer.zero_grad()
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            predictions.extend(preds.cpu().numpy())
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
        epoch_loss = running_loss / train_size
        epoch_acc = running_corrects.double() / train_size

        # Generate classification report
        report_dict = classification_report(true_labels, predictions, target_names=classes,output_dict=True)
        report_pd = pd.DataFrame(report_dict)
        report_pd.to_csv('training-classification-epoch' + str(epoch + 1) + '.csv')

        # Generate confusion matrix
        cnf_matrix = confusion_matrix(true_labels, predictions)
        df_cm = pd.DataFrame(cnf_matrix / np.sum(cnf_matrix, axis=1)[:, None], index = [i for i in classes],
                        columns = [i for i in classes])
        df_cm.to_csv('confusion-matrix-train-epoch' + str(epoch + 1) + '.csv')
        print('Train Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))

        # Delete data to clear GPU memory
        del outputs
        del preds
        del labels
        del inputs
        torch.cuda.empty_cache()

        # Validation phase
        running_loss = 0
        running_corrects = 0
        train_size = 0
        val_size = 0
        model.eval()  # set the model to evaluation mode
        predictions = []
        true_labels = []
        for inputs, labels in tqdm(val_loader):
            val_size += inputs.size(0)
            inputs = inputs.to(device)
            labels = labels.to(device)
            true_labels.extend(labels.cpu().numpy())
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            predictions.extend(preds.cpu().numpy())
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
        epoch_loss = running_loss / val_size
        epoch_acc = running_corrects.double() / val_size

        # Generate classification report
        report_dict = classification_report(true_labels, predictions, target_names=classes,output_dict=True)
        report_pd = pd.DataFrame(report_dict)
        report_pd.to_csv('val-classification-epoch' + str(epoch + 1) + '.csv')

        # Generate confusion matrix
        cnf_matrix = confusion_matrix(true_labels, predictions)
        df_cm = pd.DataFrame(cnf_matrix / np.sum(cnf_matrix, axis=1)[:, None], index = [i for i in classes],
                        columns = [i for i in classes])
        df_cm.to_csv('confusion-matrix-val-epoch' + str(epoch + 1) + '.csv')
        print('Val Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))

        # Delete data to clear GPU memory
        del outputs
        del preds
        del labels
        del inputs

    # Save the model
    torch.save(model, 'vgg16_model.pth')


if __name__ == '__main__':
    train()