# Import necessary modules and libraries
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

# Define isolation parameters
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

# Define the path for test files
test_path = "./TEST/"

# Get the list of folders/directories in the test path
path_folders = os.listdir(test_path)

# Define the path for storing completed images
completed_path = "./IMAGES_HighPassFilter/"

# Get the list of folders/directories in the completed path
completed_path_folders = os.listdir(completed_path)

# Iterate over each folder in the test path
for folder in path_folders:
    # Check if the folder is already completed, skip if so
    if folder in completed_path_folders:
        print("Skipping ", folder)
        continue
    
    try:
        # Generate automated labels for the files in the folder using isolation parameters, used PyHa code
        automated_df = generate_automated_labels(test_path+folder+'/', isolation_parameters)
        
        # Chunk the automated labels without duplicates
        df = annotation_chunker_no_duplicates(automated_df, 5)
    except Exception as e:
        print("Issue in folder: ", folder)
        continue
    
    # Process each row in the chunked dataframe
    for index, row in df.iterrows():
        file_name = row["IN FILE"]
        folder_name = row["FOLDER"]
        path = folder_name + file_name
        offset = float(row["OFFSET"])
        duration = float(row["DURATION"])
        normalized_sample_rate = 44100
        
        try:
            # Load the audio file using librosa
            SIGNAL, SAMPLE_RATE = librosa.load(path, offset=offset, duration=duration, sr=None, mono=True)
            SIGNAL = SIGNAL * 32768
        except BaseException:
            print("Failed to load" + path)
        
        # Resample the audio if it isn't the normalized sample rate
        try:
            if SAMPLE_RATE != normalized_sample_rate:
                rate_ratio = normalized_sample_rate / SAMPLE_RATE
                SIGNAL = scipy_signal.resample(SIGNAL, int(len(SIGNAL) * rate_ratio))
                SAMPLE_RATE = normalized_sample_rate
        except Exception as e:
            print("Failed to Downsample" + path + str(e))
        
        # Convert stereo audio to mono if needed
        if len(SIGNAL.shape) == 2:
            SIGNAL = SIGNAL.sum(axis=1) / 2
        
        # Compute the mel spectrogram of the audio
        D = np.abs(librosa.stft(SIGNAL)) ** 2
        S = librosa.feature.melspectrogram(S=D, sr=normalized_sample_rate, fmax=normalized_sample_rate/2, fmin=1400)
        
        # Plot and save the mel spectrogram as an image
        fig, ax = plt.subplots()
        S_dB = librosa.power_to_db(S, ref=np.max)
        img = librosa.display.specshow(S_dB, x_axis='time', y_axis='mel', sr=normalized_sample_rate