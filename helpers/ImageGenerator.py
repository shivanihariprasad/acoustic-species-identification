from PyHa.statistics import *
from PyHa.IsoAutio import *
from PyHa.visualizations import *
from PyHa.annotation_post_processing import *
import pandas as pd
import os
import librosa.display
import numpy as np
import scipy.signal as scipy_signal
import matplotlib.pyplot as plt

path = "./kaggletest/"

# Example parameters for TweetyNET
isolation_parameters = {
    "model" : "tweetynet",
    "tweety_output": True,
   "technique" : "steinberg",
    "threshold_type" : "median",
    "threshold_const" : 2.0,
    "threshold_min" : 0.0,
    "window_size" : 2.0,
    "chunk_size" : 5.0
}



test_path = "./kaggletest/"
path_folders = os.listdir(test_path)

#failed = ['carwoo1', 'rerswa1', 'eaywag1', 'thrnig1', 'cibwar1', 'lessts1', 'purgre2', 'spfbar1',
#          'gobbun1', 'varsun2', 'piekin1', 'piecro1', 'hoopoe', 'blfbus1', 'gyhneg1', 'grecor', 'beasun2' ,
#          'greegr', 'cabgre1', 'eubeat1', 'yertin1', 'categr', 'combuz1', 'witswa1', 'libeat1', 'rebfir2',
#          'afrthr1', 'spfwea1', 'tafpri1', "woosan", 'afdfly1', 'rocmar2', 'wlwwar', 'blakit1', 'slcbou1', 'afbfly1',
#          'reedov1', 'laudov1', 'gobbun1', 'squher1', 'greegr', 'litegr', 'refbar2', 'afpfly1', 'bcbeat1',
#          'sltnig1', 'amesun2', 'brubru1', 'litswi1', 'blksaw1', 'ratcis1', 'carcha1', 'brican1', 'spemou2',
#          'brwwar1', 'combul2', 'spepig1', 'hadibi1', 'strher', 'wfbeat1']

for folder in path_folders:
    if ".DS_Store" == folder:
        print("Skipping .DS_Store")
        continue
    #if folder in completed:
    #    print("Skipping ", folder)
    #    continue
    try:
        automated_df = generate_automated_labels(test_path+folder+'/',isolation_parameters);
        df = annotation_chunker_no_duplicates(automated_df, 5)
    except Exception as e:
        print("Failed ", folder)
        continue
    print("Annotation done for ", folder)
    for index, row in df.iterrows():
        file_name = row["IN FILE"]
        folder_name = row["FOLDER"]
        
        path = folder_name+file_name
        offset = float(row["OFFSET"])
        duratiom = float(row["DURATION"])
        normalized_sample_rate=44100
        try:
            SIGNAL, SAMPLE_RATE = librosa.load(path, offset=offset, duration=duratiom, sr=None, mono=True)
            SIGNAL = SIGNAL * 32768
        except BaseException:
            print("Failed to load" + path)
                
        # Resample the audio if it isn't the normalized sample rate
        try:
            if SAMPLE_RATE != normalized_sample_rate:
                rate_ratio = normalized_sample_rate / SAMPLE_RATE
                SIGNAL = scipy_signal.resample(
                SIGNAL, int(len(SIGNAL) * rate_ratio))
                SAMPLE_RATE = normalized_sample_rate
        except Exception as e:
            print("Failed to Downsample" + path + str(e))
                
        # convert stereo to mono if needed
        # Might want to compare to just taking the first set of data.
        if len(SIGNAL.shape) == 2:
            SIGNAL = SIGNAL.sum(axis=1) / 2

        D = np.abs(librosa.stft(SIGNAL))**2
        S = librosa.feature.melspectrogram(S=D, sr=normalized_sample_rate)
        # save s directly -> as an image preferably
        fig, ax = plt.subplots()
        S_dB = librosa.power_to_db(S, ref=np.max)
        img = librosa.display.specshow(S_dB, x_axis='time', y_axis='mel', sr=normalized_sample_rate,fmax=normalized_sample_rate//2, ax=ax)
        fig.colorbar(img, ax=ax, format='%+2.0f dB')
        ax.set(title='Mel-frequency spectrogram')
        folder_name = folder_name[:-1]
        bird_name = folder_name[13:]
        bird_path = "./IMAGES/"+bird_name+"/"
        isExist = os.path.exists(bird_path)
        if not isExist:
            # Create a new directory because it does not exist
            os.makedirs(bird_path)
        plt.savefig(bird_path+str(index)+".png")
        plt.close()
#print(automated_df)

