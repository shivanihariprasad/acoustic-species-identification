'''
    Code to split dataset into train and validation such that if audio is present in train, it is not included in validation set
'''
import pandas as pd
import math 
import random

df = pd.read_csv('data_labels_264n.csv')
classes = ['abethr1', 'abhori1', 'abythr1', 'afbfly1', 'afdfly1', 'afecuc1', 'affeag1', 'afgfly1', 'afghor1', 'afmdov1', 'afpfly1', 'afpkin1', 'afpwag1', 'afrgos1', 'afrgrp1', 'afrjac1', 'afrthr1', 'amesun2', 'augbuz1', 'bagwea1', 'barswa', 'bawhor2', 'bawman1', 'bcbeat1', 'beasun2', 'bkctch1', 'bkfruw1', 'blacra1', 'blacuc1', 'blakit1', 'blaplo1', 'blbpuf2', 'blcapa2', 'blfbus1', 'blhgon1', 'blhher1', 'blksaw1', 'blnmou1', 'blnwea1', 'bltapa1', 'bltbar1', 'bltori1', 'blwlap1', 'brcale1', 'brcsta1', 'brctch1', 'brcwea1', 'brican1', 'brobab1', 'broman1', 'brosun1', 'brrwhe3', 'brtcha1', 'brubru1', 'brwwar1', 'bswdov1', 'btweye2', 'bubwar2', 'butapa1', 'cabgre1', 'carcha1', 'carwoo1', 'categr', 'ccbeat1', 'chespa1', 'chewea1', 'chibat1', 'chtapa3', 'chucis1', 'cibwar1', 'cohmar1', 'colsun2', 'combul2', 'combuz1', 'comsan', 'crefra2', 'crheag1', 'crohor1', 'darbar1', 'darter3', 'didcuc1', 'dotbar1', 'dutdov1', 'easmog1', 'eaywag1', 'edcsun3', 'egygoo', 'equaka1', 'eswdov1', 'eubeat1', 'fatrav1', 'fatwid1', 'fislov1', 'fotdro5', 'gabgos2', 'gargan', 'gbesta1', 'gnbcam2', 'gnhsun1', 'gobbun1', 'gobsta5', 'gobwea1', 'golher1', 'grbcam1', 'grccra1', 'grecor', 'greegr', 'grewoo2', 'grwpyt1', 'gryapa1', 'grywrw1', 'gybfis1', 'gycwar3', 'gyhbus1', 'gyhkin1', 'gyhneg1', 'gyhspa1', 'gytbar1', 'hadibi1', 'hamerk1', 'hartur1', 'helgui', 'hipbab1', 'hoopoe', 'huncis1', 'hunsun2', 'joygre1', 'kerspa2', 'klacuc1', 'kvbsun1', 'laudov1', 'lawgol', 'lesmaw1', 'lessts1', 'libeat1', 'litegr', 'litswi1', 'litwea1', 'loceag1', 'lotcor1', 'lotlap1', 'luebus1', 'mabeat1', 'macshr1', 'malkin1', 'marsto1', 'marsun2', 'mcptit1', 'meypar1', 'moccha1', 'mouwag1', 'ndcsun2', 'nobfly1', 'norbro1', 'norcro1', 'norfis1', 'norpuf1', 'nubwoo1', 'pabspa1', 'palfly2', 'palpri1', 'piecro1', 'piekin1', 'pitwhy', 'purgre2', 'pygbat1', 'quailf1', 'ratcis1', 'raybar1', 'rbsrob1', 'rebfir2', 'rebhor1', 'reboxp1', 'reccor', 'reccuc1', 'reedov1', 'refbar2', 'refcro1', 'reftin1', 'refwar2', 'rehblu1', 'rehwea1', 'reisee2', 'rerswa1', 'rewsta1', 'rindov', 'rocmar2', 'rostur1', 'ruegls1', 'rufcha2', 'sacibi2', 'sccsun2', 'scrcha1', 'scthon1', 'shesta1', 'sichor1', 'sincis1', 'slbgre1', 'slcbou1', 'sltnig1', 'sobfly1', 'somgre1', 'somtit4', 'soucit1', 'soufis1', 'spemou2', 'spepig1', 'spewea1', 'spfbar1', 'spfwea1', 'spmthr1', 'spwlap1', 'squher1', 'strher', 'strsee1', 'stusta1', 'subbus1', 'supsta1', 'tacsun1', 'tafpri1', 'tamdov1', 'thrnig1', 'trobou1', 'varsun2', 'vibsta2', 'vilwea1', 'vimwea1', 'walsta1', 'wbgbir1', 'wbrcha2', 'wbswea1', 'wfbeat1', 'whbcan1', 'whbcou1', 'whbcro2', 'whbtit5', 'whbwea1', 'whbwhe3', 'whcpri2', 'whctur2', 'wheslf1', 'whhsaw1', 'whihel1', 'whrshr1', 'witswa1', 'wlwwar', 'wookin1', 'woosan', 'wtbeat1', 'yebapa1', 'yebbar1', 'yebduc1', 'yebere1', 'yebgre1', 'yebsto1', 'yeccan1', 'yefcan', 'yelbis1', 'yenspu1', 'yertin1', 'yesbar1', 'yespet1', 'yetgre1', 'yewgre1']

audio_class_count = {}
folder_name = './kaggletest/'

for bird in classes:
    count = df['FOLDER'].value_counts()[folder_name + bird + '/']
    audio_class_count[bird] = count

#print(len(audio_class_count))
print(len(df))
print(len(classes))
train_df = pd.DataFrame()
val_df = pd.DataFrame()
one_files = []
two_files = []
three_files = []
no_val = []

for bird in classes:
    chunk_count =  audio_class_count[bird]
    train_count = math.floor(0.8*chunk_count)
    val_count = chunk_count - train_count
    b_df = df.loc[df['FOLDER'] == folder_name + bird + '/']
    audio_files = b_df['IN FILE'].unique()
    if (chunk_count == 1):
        #print("Bird with one chunk count: ", bird)
        audio_df = df.loc[df['IN FILE'] == audio_files[0]]
        train_df = pd.concat([train_df, audio_df], ignore_index=True)
        val_df = pd.concat([val_df, audio_df], ignore_index=True)
        continue
    if (chunk_count == 2):
        t_df = b_df.iloc[:]
        v_df = b_df.iloc[1:]
        train_df = pd.concat([train_df, t_df], ignore_index=True)
        val_df = pd.concat([val_df, v_df], ignore_index=True)
        continue

    if (len(audio_files) == 1):
        t_df = b_df.iloc[:train_count - 1]
        v_df = b_df.iloc[train_count - 1:]
        
        train_df = pd.concat([train_df, t_df], ignore_index=True)
        val_df = pd.concat([val_df, v_df], ignore_index=True)
        continue

    random.shuffle(audio_files)
    train_audio_list = []
    current_train_count = 0
    for a in audio_files:
        c = df['IN FILE'].value_counts()[a]
        current_train_count += c
        train_audio_list.append(a)
        if (current_train_count >= train_count):
            if current_train_count - train_count > 0.5*chunk_count:
                print(chunk_count, train_count, current_train_count)
                print("WARNING for bird ", bird)
            break

    val_audio_list = [a for a in audio_files if a not in train_audio_list]
    temp_df = pd.DataFrame()

    for a in train_audio_list:
        t_df = df.loc[df['IN FILE'] == a]
        temp_df = pd.concat([temp_df, t_df], ignore_index=True)

    if (len(val_audio_list) == 0):
        train_split_df = temp_df.iloc[:train_count - 1]
        val_split_df = temp_df.iloc[train_count - 1:]
        val_df = pd.concat([val_df, val_split_df], ignore_index=True)
        train_df = pd.concat([train_df, train_split_df], ignore_index=True)
        continue

    train_df = pd.concat([train_df, temp_df], ignore_index=True)

    for a in val_audio_list:
        v_df = df.loc[df['IN FILE'] == a]
        val_df = pd.concat([val_df, v_df], ignore_index=True)
    #print(train_audio_list)
    #print(val_audio_list)

print("Done: ", len(val_df), len(train_df))
print(" percentage of data in train: ", len(train_df)/len(df))
print(" percentage of data in val: ", len(val_df)/len(df))
train_df.drop(df.columns[df.columns.str.contains('unnamed',case = False)],axis = 1, inplace = True)
val_df.drop(df.columns[df.columns.str.contains('unnamed',case = False)],axis = 1, inplace = True)
print("Count of unique labels in df: ", len(df['Label'].unique()))
print("Count of unique labels in train df: ", len(train_df['Label'].unique()))
print("Count of unique labels in val df: ", len(val_df['Label'].unique()))
train_df.to_csv("newtrain.csv",index=True)
val_df.to_csv("newval.csv",index=True)