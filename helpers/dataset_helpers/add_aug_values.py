'''
    Code to add augmentation type in the dataset CSV file
'''

import pandas as pd
import os
import numpy as np

df = pd.read_csv('newtrain_aug.csv')
classes = ['abethr1', 'abhori1', 'abythr1', 'afbfly1', 'afdfly1', 'afecuc1', 'affeag1', 'afgfly1', 'afghor1', 'afmdov1', 'afpfly1', 'afpkin1', 'afpwag1', 'afrgos1', 'afrgrp1', 'afrjac1', 'afrthr1', 'amesun2', 'augbuz1', 'bagwea1', 'barswa', 'bawhor2', 'bawman1', 'bcbeat1', 'beasun2', 'bkctch1', 'bkfruw1', 'blacra1', 'blacuc1', 'blakit1', 'blaplo1', 'blbpuf2', 'blcapa2', 'blfbus1', 'blhgon1', 'blhher1', 'blksaw1', 'blnmou1', 'blnwea1', 'bltapa1', 'bltbar1', 'bltori1', 'blwlap1', 'brcale1', 'brcsta1', 'brctch1', 'brcwea1', 'brican1', 'brobab1', 'broman1', 'brosun1', 'brrwhe3', 'brtcha1', 'brubru1', 'brwwar1', 'bswdov1', 'btweye2', 'bubwar2', 'butapa1', 'cabgre1', 'carcha1', 'carwoo1', 'categr', 'ccbeat1', 'chespa1', 'chewea1', 'chibat1', 'chtapa3', 'chucis1', 'cibwar1', 'cohmar1', 'colsun2', 'combul2', 'combuz1', 'comsan', 'crefra2', 'crheag1', 'crohor1', 'darbar1', 'darter3', 'didcuc1', 'dotbar1', 'dutdov1', 'easmog1', 'eaywag1', 'edcsun3', 'egygoo', 'equaka1', 'eswdov1', 'eubeat1', 'fatrav1', 'fatwid1', 'fislov1', 'fotdro5', 'gabgos2', 'gargan', 'gbesta1', 'gnbcam2', 'gnhsun1', 'gobbun1', 'gobsta5', 'gobwea1', 'golher1', 'grbcam1', 'grccra1', 'grecor', 'greegr', 'grewoo2', 'grwpyt1', 'gryapa1', 'grywrw1', 'gybfis1', 'gycwar3', 'gyhbus1', 'gyhkin1', 'gyhneg1', 'gyhspa1', 'gytbar1', 'hadibi1', 'hamerk1', 'hartur1', 'helgui', 'hipbab1', 'hoopoe', 'huncis1', 'hunsun2', 'joygre1', 'kerspa2', 'klacuc1', 'kvbsun1', 'laudov1', 'lawgol', 'lesmaw1', 'lessts1', 'libeat1', 'litegr', 'litswi1', 'litwea1', 'loceag1', 'lotcor1', 'lotlap1', 'luebus1', 'mabeat1', 'macshr1', 'malkin1', 'marsto1', 'marsun2', 'mcptit1', 'meypar1', 'moccha1', 'mouwag1', 'ndcsun2', 'nobfly1', 'norbro1', 'norcro1', 'norfis1', 'norpuf1', 'nubwoo1', 'pabspa1', 'palfly2', 'palpri1', 'piecro1', 'piekin1', 'pitwhy', 'purgre2', 'pygbat1', 'quailf1', 'ratcis1', 'raybar1', 'rbsrob1', 'rebfir2', 'rebhor1', 'reboxp1', 'reccor', 'reccuc1', 'reedov1', 'refbar2', 'refcro1', 'reftin1', 'refwar2', 'rehblu1', 'rehwea1', 'reisee2', 'rerswa1', 'rewsta1', 'rindov', 'rocmar2', 'rostur1', 'ruegls1', 'rufcha2', 'sacibi2', 'sccsun2', 'scrcha1', 'scthon1', 'shesta1', 'sichor1', 'sincis1', 'slbgre1', 'slcbou1', 'sltnig1', 'sobfly1', 'somgre1', 'somtit4', 'soucit1', 'soufis1', 'spemou2', 'spepig1', 'spewea1', 'spfbar1', 'spfwea1', 'spmthr1', 'spwlap1', 'squher1', 'strher', 'strsee1', 'stusta1', 'subbus1', 'supsta1', 'tacsun1', 'tafpri1', 'tamdov1', 'thrnig1', 'trobou1', 'varsun2', 'vibsta2', 'vilwea1', 'vimwea1', 'walsta1', 'wbgbir1', 'wbrcha2', 'wbswea1', 'wfbeat1', 'whbcan1', 'whbcou1', 'whbcro2', 'whbtit5', 'whbwea1', 'whbwhe3', 'whcpri2', 'whctur2', 'wheslf1', 'whhsaw1', 'whihel1', 'whrshr1', 'witswa1', 'wlwwar', 'wookin1', 'woosan', 'wtbeat1', 'yebapa1', 'yebbar1', 'yebduc1', 'yebere1', 'yebgre1', 'yebsto1', 'yeccan1', 'yefcan', 'yelbis1', 'yenspu1', 'yertin1', 'yesbar1', 'yespet1', 'yetgre1', 'yewgre1']

audio_class_count = {}
folder_name = './kaggletest/'

for bird in classes:
    count = df['FOLDER'].value_counts()[folder_name + bird + '/']
    audio_class_count[bird] = count

print(len(audio_class_count))

values = []
for k in audio_class_count.keys():
    values.append(audio_class_count[k])

m = max(values)
print(m)

to_augment = ['abethr1', 'abythr1', 'afbfly1', 'afgfly1', 'afpfly1', 'afpkin1', 'afrgrp1', 'afrjac1', 'amesun2', 'augbuz1', 'bawhor2',
              'bawman1', 'beasun2', 'blcapa2', 'blfbus1', 'blhgon1', 'blhher1', 'blnwea1', 'bltapa1', 
              'brctch1', 'brcwea1', 'brobab1', 'broman1', 'brrwhe3', 'bubwar2', 'cabgre1', 'chespa1', 'chtapa3',
              'darter3', 'dotbar1', 'easmog1', 'edcsun3', 'equaka1', 'fatrav1', 'gargan', 'gbesta1', 'gobwea1', 'golher1',
              'grccra1', 'gryapa1', 'gybfis1', 'gyhkin1', 'gyhneg1', 'gytbar1', 'hamerk1', 'huncis1', 
              'klacuc1', 'lesmaw1', 'lessts1', 'libeat1', 'litwea1', 'loceag1', 'luebus1', 'mabeat1', 'malkin1',
              'mcptit1', 'ndcsun2', 'nobfly1', 'norbro1', 'norcro1', 'norfis1', 'norpuf1', 'pabspa1', 'palfly2',
              'purgre2', 'raybar1', 'reboxp1', 'reedov1', 'refbar2', 'refwar2', 'rehblu1', 'rehwea1', 'reisee2',
              'rewsta1', 'rocmar2', 'rostur1', 'ruegls1', 'rufcha2', 'sacibi2', 'sccsun2', 'scrcha1', 'shesta1', 'slbgre1', 'spemou2',
              'spewea1', 'spfbar1', 'spfwea1', 'squher1', 'stusta1', 'supsta1', 'tacsun1', 'vibsta2', 'vimwea1', 
              'whbcan1', 'whbcou1', 'whbcro2', 'whbtit5', 'whcpri2', 'whctur2', 'wheslf1', 'whrshr1', 'wtbeat1', 'yebduc1' ,'yebere1', 
              'yeccan1', 'yelbis1', 'yenspu1', 'yespet1' ,'yetgre1']

count = 0
to_update = 0
for bird in to_augment:
    to_update = to_update + audio_class_count[bird]
    df_new = df[df['FOLDER'] == folder_name + bird + '/']
    for index, row in df_new.iterrows():
        #row['AUGMENT_TYPE'] = 'SPEC_AUGMENT'
        #df = df.append(row,ignore_index=True) 
        row['AUGMENT_TYPE'] = 'AUGMENT_TIME_STRETCH'
        df = df.append(row,ignore_index=True) 
        row['AUGMENT_TYPE'] = 'AUGMENT_TIME_MASK'
        #df = df.append(row,ignore_index=True) 
        ##row['AUGMENT_TYPE'] = 'SPEC_AUGMENT_FREQ_MASK'
        df = df.append(row,ignore_index=True) 
        count = count + 2

print("Number of rows added: ", count)
print("Count from dictionary: ", 2*to_update)
df.drop(df.columns[df.columns.str.contains('unnamed',case = False)],axis = 1, inplace = True)
df.to_csv("newtrain_aug_updated.csv",index=True)




