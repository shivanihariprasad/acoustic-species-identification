import librosa
import numpy as np
from scipy.io.wavfile import write
import os

path = "./kaggletest/"

# Some of the files in these folders failed with PyHa
# On debugging we found out that if audio chunks are very small, PyHa fails.
# For audio chunks < 5 secs, we add noise to make it a chunk of 5 sec
folders = ['carwoo1/', 'gargan/','rerswa1/', 'eaywag1/', 'thrnig1/', 'cibwar1/', 'lessts1/', 'purgre2/', 'spfbar1/',
         'gobbun1/', 'varsun2/', 'piekin1/', 'piecro1/', 'hoopoe/', 'blfbus1/', 'gyhneg1/', 'grecor/', 'beasun2/' ,
         'greegr/', 'cabgre1/', 'eubeat1/', 'yertin1/', 'categr/', 'combuz1/', 'witswa1/', 'libeat1/', 'rebfir2/',
        'afrthr1/', 'spfwea1/', 'tafpri1/', "woosan/", 'afdfly1/', 'rocmar2/', 'wlwwar/', 'blakit1/', 'slcbou1/', 'afbfly1/',
       'reedov1/', 'laudov1/', 'gobbun1/', 'squher1/', 'greegr/', 'litegr/', 'refbar2/', 'afpfly1/', 'bcbeat1/',
         'sltnig1/', 'amesun2/', 'brubru1/', 'litswi1/', 'blksaw1/', 'ratcis1/', 'carcha1/', 'brican1/', 'spemou2/',
         'brwwar1/', 'combul2/', 'spepig1/', 'hadibi1/', 'strher/', 'wfbeat1/']

sr = 44100
STD_n = 0.005
total_len = 5*sr
for f in folders:
    print("-------------")
    audio_files = os.listdir(path + f)
    print(f"Number of files in {f} are {len(audio_files)}")
    modified_files = []
    for a in audio_files:
        if ".DS_Store" == a:
            print("Skipping", a)
            continue
        signal, sr = librosa.load(path+f+a,sr=sr, mono=True)
        if (len(signal) < 220500):
            modified_files.append(a)
            print(a)
            noise=np.random.normal(0, STD_n, 5*sr)
            signal_noise = np.concatenate((signal,noise[:total_len-signal.shape[0]]), axis=None)  # create 5 sec audio file by adding noise to short audio clips
            write(path + f + a[:-3] + 'wav', sr, signal_noise) # save the audio in wav format

    # open file in write mode
    with open('modifiedFiles' + '/' + f[:-1]+'.txt', 'w') as fp:
        for item in modified_files:
            # write each item on a new line
            fp.write("%s\n" % item)

    for item in modified_files:
        print("Removing file", path + f + item)
        os.remove(path + f + item)
