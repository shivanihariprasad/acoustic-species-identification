import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
from torchvision import models as models
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim 
import sklearn.metrics
from sklearn.metrics import confusion_matrix
import librosa
import scipy.signal as scipy_signal
import os
from opensoundscape import Audio, Spectrogram
from sklearn.metrics import classification_report
from collections import Counter
#from opensoundscape import MelSpectrogram

# set computation device
device = ('cuda' if torch.cuda.is_available() else 'cpu')
num_epochs = 10

classes = ['abethr1', 'abhori1', 'abythr1', 'afbfly1', 'afdfly1', 'afecuc1', 'affeag1', 'afgfly1', 'afghor1', 'afmdov1', 'afpfly1', 'afpkin1', 'afpwag1', 'afrgos1', 'afrgrp1', 'afrjac1', 'afrthr1', 'amesun2', 'augbuz1', 'bagwea1', 'barswa', 'bawhor2', 'bawman1', 'bcbeat1', 'beasun2', 'bkctch1', 'bkfruw1', 'blacra1', 'blacuc1', 'blakit1', 'blaplo1', 'blbpuf2', 'blcapa2', 'blfbus1', 'blhgon1', 'blhher1', 'blksaw1', 'blnmou1', 'blnwea1', 'bltapa1', 'bltbar1', 'bltori1', 'blwlap1', 'brcale1', 'brcsta1', 'brctch1', 'brcwea1', 'brican1', 'brobab1', 'broman1', 'brosun1', 'brrwhe3', 'brtcha1', 'brubru1', 'brwwar1', 'bswdov1', 'btweye2', 'bubwar2', 'butapa1', 'cabgre1', 'carcha1', 'carwoo1', 'categr', 'ccbeat1', 'chespa1', 'chewea1', 'chibat1', 'chtapa3', 'chucis1', 'cibwar1', 'cohmar1', 'colsun2', 'combul2', 'combuz1', 'comsan', 'crefra2', 'crheag1', 'crohor1', 'darbar1', 'darter3', 'didcuc1', 'dotbar1', 'dutdov1', 'easmog1', 'eaywag1', 'edcsun3', 'egygoo', 'equaka1', 'eswdov1', 'eubeat1', 'fatrav1', 'fatwid1', 'fislov1', 'fotdro5', 'gabgos2', 'gargan', 'gbesta1', 'gnbcam2', 'gnhsun1', 'gobbun1', 'gobsta5', 'gobwea1', 'golher1', 'grbcam1', 'grccra1', 'grecor', 'greegr', 'grewoo2', 'grwpyt1', 'gryapa1', 'grywrw1', 'gybfis1', 'gycwar3', 'gyhbus1', 'gyhkin1', 'gyhneg1', 'gyhspa1', 'gytbar1', 'hadibi1', 'hamerk1', 'hartur1', 'helgui', 'hipbab1', 'hoopoe', 'huncis1', 'hunsun2', 'joygre1', 'kerspa2', 'klacuc1', 'kvbsun1', 'laudov1', 'lawgol', 'lesmaw1', 'lessts1', 'libeat1', 'litegr', 'litswi1', 'litwea1', 'loceag1', 'lotcor1', 'lotlap1', 'luebus1', 'mabeat1', 'macshr1', 'malkin1', 'marsto1', 'marsun2', 'mcptit1', 'meypar1', 'moccha1', 'mouwag1', 'ndcsun2', 'nobfly1', 'norbro1', 'norcro1', 'norfis1', 'norpuf1', 'nubwoo1', 'pabspa1', 'palfly2', 'palpri1', 'piecro1', 'piekin1', 'pitwhy', 'purgre2', 'pygbat1', 'quailf1', 'ratcis1', 'raybar1', 'rbsrob1', 'rebfir2', 'rebhor1', 'reboxp1', 'reccor', 'reccuc1', 'reedov1', 'refbar2', 'refcro1', 'reftin1', 'refwar2', 'rehblu1', 'rehwea1', 'reisee2', 'rerswa1', 'rewsta1', 'rindov', 'rocmar2', 'rostur1', 'ruegls1', 'rufcha2', 'sacibi2', 'sccsun2', 'scrcha1', 'scthon1', 'shesta1', 'sichor1', 'sincis1', 'slbgre1', 'slcbou1', 'sltnig1', 'sobfly1', 'somgre1', 'somtit4', 'soucit1', 'soufis1', 'spemou2', 'spepig1', 'spewea1', 'spfbar1', 'spfwea1', 'spmthr1', 'spwlap1', 'squher1', 'strher', 'strsee1', 'stusta1', 'subbus1', 'supsta1', 'tacsun1', 'tafpri1', 'tamdov1', 'thrnig1', 'trobou1', 'varsun2', 'vibsta2', 'vilwea1', 'vimwea1', 'walsta1', 'wbgbir1', 'wbrcha2', 'wbswea1', 'wfbeat1', 'whbcan1', 'whbcou1', 'whbcro2', 'whbtit5', 'whbwea1', 'whbwhe3', 'whcpri2', 'whctur2', 'wheslf1', 'whhsaw1', 'whihel1', 'whrshr1', 'witswa1', 'wlwwar', 'wookin1', 'woosan', 'wtbeat1', 'yebapa1', 'yebbar1', 'yebduc1', 'yebere1', 'yebgre1', 'yebsto1', 'yeccan1', 'yefcan', 'yelbis1', 'yenspu1', 'yertin1', 'yesbar1', 'yespet1', 'yetgre1', 'yewgre1']
#print("Length of classes: ", len(classes))
# read the data.csv file and get the audio paths, audio offset and duration and labels
df = pd.read_csv('data_labels_264n.csv',index_col=False)
#print(df.columns)
#print("Before adding data dataset length is ", len(df))
#print(df.columns)
df.loc[len(df.index)] = [115259, './kaggletest/afpkin1/', 'XC704863.ogg', 7.751995464852610, 0, 0.0, 5, 44100, 'bird', 1.0, 11]
df.loc[len(df.index)] = [115260, './kaggletest/afpkin1/', 'XC704863.ogg', 7.751995464852610, 0, 0.0, 5, 44100, 'bird', 1.0, 11]
#df.loc[len(df.index)] = [115261, './kaggletest/afpkin1/', 'XC704863.ogg', 7.751995464852610, 0, 0.0, 5, 44100, 'bird', 1.0, 11]
df.loc[len(df.index)] = [115261,'./kaggletest/golher1/', 'XC248014.ogg', 8.28, 0, 0.0, 5, 44100, 'bird', 1.0, 102]
df.loc[len(df.index)] = [115262,'./kaggletest/golher1/', 'XC248014.ogg', 8.28, 0, 0.0, 5, 44100, 'bird', 1.0, 102]
#df.loc[len(df.index)] = [115264,'./kaggletest/golher1/', 'XC248014.ogg', 8.28, 0, 0.0, 5, 44100, 'bird', 1.0, 102]
df.loc[len(df.index)] = [115263,'./kaggletest/whctur2/', 'XC444635.ogg', 9.508548752834470, 0, 0.0, 5, 44100, 'bird', 1.0, 239]
df.loc[len(df.index)] = [115264,'./kaggletest/whctur2/', 'XC444635.ogg', 9.508548752834470, 0, 0.0, 5, 44100, 'bird', 1.0, 239]
#df.loc[len(df.index)] = [115267,'./kaggletest/whctur2/', 'XC444635.ogg', 9.508548752834470, 0, 0.0, 5, 44100, 'bird', 1.0, 239]
df.loc[len(df.index)] = [115265, './kaggletest/whhsaw1/', 'XC289267.ogg', 10.3967346938776, 0, 5, 5, 44100, 'bird', 1.0, 241]
df.loc[len(df.index)] = [115266, './kaggletest/lotlap1/', 'XC213636.ogg', 7.15755102040816, 0, 0.0, 5, 44100, 'bird', 1.0, 140]
#print("Before adding data dataset length is ", len(df))

X = df.iloc[:, 1:-1]
Y = pd.DataFrame(df.iloc[:, -1])

image_shape = (224, 224)
#print(X.head())
#print("X shape: ", X.shape)
#print("X Columns: ", X.columns)
#print(Y.head())
#print("Y Shape: ", Y.shape)
#print("Y Columns: ", Y.columns)

(xtrain, xtest, ytrain, ytest) = (train_test_split(X, Y, test_size=0.2, random_state=49, shuffle=True, stratify=Y))
xtrain = xtrain.reset_index(drop=True)
ytrain = ytrain.reset_index(drop=True)
xtest = xtest.reset_index(drop=True)
ytest = ytest.reset_index(drop=True)
#print("Length of train: ", len(ytrain))
#print("Length of test: ", len(ytest))

l = []
for index, row in ytest.iterrows():
    l.append(row['Label'])
#print("Test labels count:")
#print(Counter(l))
s = set(l)
actual = [i for i in range(264)]
target_ids = [x for x in actual if x not in s]
#print("Length of the test set is: ", len(s))
print("Missing: ", len(target_ids))
#print("data: ", target_ids)

metadata_file =  './birdclef-2023/train_metadata.csv'
df = pd.read_csv(metadata_file)

df['relative_path'] = 'train_audio/' + df['filename'].astype(str)

df = df[['relative_path', 'primary_label']]

birds = list(pd.get_dummies(df['primary_label']).columns)

birds = np.transpose(birds)
#print(birds)
print("Birds shape: ", birds.shape)

def padded_cmap(solution, submission, padding_factor=5):
    solution = solution.drop(['row_id'], axis=1, errors='ignore')
    submission = submission.drop(['row_id'], axis=1, errors='ignore')
    new_rows = []
    for i in range(padding_factor):
        new_rows.append([1 for i in range(len(solution.columns))])
    new_rows = pd.DataFrame(new_rows)
    new_rows.columns = solution.columns
    padded_solution = pd.concat([solution, new_rows]).reset_index(drop=True).copy()
    padded_submission = pd.concat([submission, new_rows]).reset_index(drop=True).copy()
    score = sklearn.metrics.average_precision_score(
        padded_solution.values,
        padded_submission.values,
        average='macro',
    )
    return score

def mono_to_color(X, eps=1e-6, mean=None, std=None):
        mean = mean or X.mean()
        std = std or X.std()
        X = (X - mean) / (std + eps)
        
        _min, _max = X.min(), X.max()

        if (_max - _min) > eps:
            V = np.clip(X, _min, _max)
            V = 255 * (V - _min) / (_max - _min)
            V = V.astype(np.uint8)
        else:
            V = np.zeros_like(X, dtype=np.uint8)

        return V

class BirdAudioDataset(Dataset):
    def __init__(self, info, labels, tfms=None):
        self.folder = info['FOLDER']
        self.file = info['IN FILE']
        self.offset = info['OFFSET']
        self.duration = info['DURATION']
        self.label = labels['Label']

    def __len__(self):
        return (len(self.file))

    def __getitem__(self, i):
        file_path = self.folder[i] + self.file[i]
        normalized_sample_rate=44100
        #try:
        #    SIGNAL, SAMPLE_RATE = librosa.load(file_path, offset=self.offset[i], duration=self.duration[i], sr=None, mono=True)
        #    SIGNAL = SIGNAL * 32768
        #except BaseException:
        #    print("Failed to load" + file_path)
                
        # Resample the audio if it isn't the normalized sample rate
        #try:
        #    if SAMPLE_RATE != normalized_sample_rate:
        #        rate_ratio = normalized_sample_rate / SAMPLE_RATE
        #        SIGNAL = scipy_signal.resample(SIGNAL, int(len(SIGNAL) * rate_ratio))
        #        SAMPLE_RATE = normalized_sample_rate
        #except Exception as e:
        #    print("Failed to Downsample" + file_path + str(e))
                
        # convert stereo to mono if needed
        # Might want to compare to just taking the first set of data.
        #if len(SIGNAL.shape) == 2:
        #    SIGNAL = SIGNAL.sum(axis=1) / 2

        #D = np.abs(librosa.stft(SIGNAL))**2
        #melspec = librosa.feature.melspectrogram(S=D, sr=normalized_sample_rate, fmax=normalized_sample_rate//2)
        #melspec = librosa.power_to_db(melspec).astype(np.float32)
        #image = mono_to_color(melspec)
        #print(image.shape)
        #newsize = (224, 224)
        #image.resize(newsize)
        #Should use load audio of opensoundscape for this: 
        D = Audio.from_file(file_path,offset=self.offset[i],duration=self.duration[i], sample_rate=normalized_sample_rate)

        #MelSpectrogram.from_audio(D, n_fft=1024, n_mels=128, fmin=None, fmax=normalized_sample_rate//2)
        spec_obj = Spectrogram.from_audio(D)
        img = spec_obj.to_image(shape=image_shape, channels=3, return_type='torch')
        #img = np.array(img)
        label = self.label[i]
        return img, torch.tensor(label, dtype=torch.long)


train_data = BirdAudioDataset(xtrain, ytrain, tfms=1)
test_data = BirdAudioDataset(xtest, ytest, tfms=0)
 
model = models.efficientnet_b0(weights=models.efficientnet.EfficientNet_B0_Weights.DEFAULT)
num_classes = 264

def train():
    # dataloaders
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True, num_workers=4)
    valid_loader = DataLoader(test_data, batch_size=32, shuffle=True, num_workers=4)

    # Unfreeze model weights
    for param in model.parameters():
        param.requires_grad = False
    
    num_inputs = model.classifier[1].in_features
    #print("Num inputs:", num_inputs)
    model.classifier = nn.Sequential(
        nn.Linear(num_inputs, 2048),
        nn.SiLU(), # Sigmoid Weighted Linear Unit
        nn.Dropout(0.2),
        # Note that the last layer is 2048 * Number of Classes
        # Reshape the final layer(s) to have the same number of outputs as the number of classes in the new dataset
        nn.Linear(2048, 1056),
        nn.Linear(1056, num_classes)
    )

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
        all_outputs = np.empty((0,264), int)
        model.train()
        for inputs, labels in tqdm(train_loader):
            train_size += inputs.size(0)
            #print("Input size: ---->", inputs.size())
            inputs = inputs.to(device)
            labels = labels.to(device)
            true_values.extend(labels.numpy())
            optimizer.zero_grad()

            outputs = model(inputs)
            #all_outputs.extend(outputs.detach().numpy())
            #print("Shape of outputs: ", outputs.detach().numpy().shape)
            all_outputs = np.concatenate((all_outputs, outputs.detach().numpy()), axis=0)
            #print(len(all_outputs))
            #print("Shape after concatenattion", all_outputs.shape)
            _, preds = torch.max(outputs, 1)
            predictions.extend(preds.numpy())
            loss = loss_func(outputs, labels)
            #print(len(predictions))
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

            #output_val = outputs.sigmoid().detach().numpy()
        target_one_hot = torch.eye(264)[true_values]
        target_val = target_one_hot.numpy()

        val_df = pd.DataFrame(target_val, columns = birds)
        pred_df = pd.DataFrame(all_outputs, columns = birds)
        val_df.to_csv('val_df.csv')
        pred_df.to_csv('pred_df.csv')
        
        avg_score = padded_cmap(val_df, pred_df, padding_factor = 5)

        print(f"cmAP score pad 5: {avg_score}")

        epoch_loss = running_loss / train_size
        epoch_acc = running_corrects.double() / train_size
        print("Train size: ", train_size)
        print('Train Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))
        report_dict = classification_report(true_values, predictions, target_names=classes,output_dict=True)
        report_pd = pd.DataFrame(report_dict)
        report_pd.to_csv('./opensoundscape/train-classification-epoch' + str(epoch + 1) + '.csv')

        # Print confusion matrix
        cf_matrix = confusion_matrix(true_values, predictions)

        df_cm = pd.DataFrame(cf_matrix / np.sum(cf_matrix, axis=1)[:, None], index = [i for i in classes],
                        columns = [i for i in classes])
        df_cm.to_csv('./opensoundscape/train-epoch' + str(epoch + 1) + '.csv')

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
            true_values.extend(labels.numpy())
            
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
        print("True values and predictions length: ", len(set(true_values)), len(set(predictions)))
        report_dict = classification_report(true_values, predictions, target_names=classes,output_dict=True)
        report_pd = pd.DataFrame(report_dict)
        report_pd.to_csv('./opensoundscape/val-classification-epoch' + str(epoch + 1) + '.csv')

        # Print confusion matrix
        cf_matrix = confusion_matrix(true_values, predictions)
        print("Validation mconfusion matrix shape:", cf_matrix.shape)
        df_cm = pd.DataFrame(cf_matrix / np.sum(cf_matrix, axis=1)[:, None], index = [i for i in classes],
                        columns = [i for i in classes])
        df_cm.to_csv('./opensoundscape/val-epoch' + str(epoch + 1) + '.csv')

    torch.save(model, 'efficientnet_b0_264_spectogram.pt')

if __name__ == '__main__':
    train()
