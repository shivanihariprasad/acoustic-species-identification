import librosa
import numpy as np
from PIL import Image
import pandas as pd
import os
import librosa.display
import numpy as np
import scipy.signal as scipy_signal
import matplotlib.pyplot as plt
import torchaudio
import torch
from torchvision.transforms import transforms
import noisereduce as nr

normalized_sample_rate=44100

# Path to audio
path="./kaggletest/abethr1/XC128013.ogg"

try:
    SIGNAL, SAMPLE_RATE = librosa.load(path, offset=0, duration=4, sr=None, mono=True)
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

SIGNAL = nr.reduce_noise(y=SIGNAL, sr=normalized_sample_rate, use_tqdm=False)

D = np.abs(librosa.stft(SIGNAL))**2
# Assuming `D` and `normalized_sample_rate` are defined
S = librosa.feature.melspectrogram(S=D, sr=normalized_sample_rate, fmax=normalized_sample_rate/2)

#Time stretching
time_stretched_signal = librosa.effects.time_stretch(S, rate=1.25)

# Apply time and frequency masking
time_masked = torchaudio.transforms.TimeMasking(time_mask_param=50)(torch.from_numpy(time_stretched_signal).unsqueeze(0))
frequency_masked = torchaudio.transforms.FrequencyMasking(freq_mask_param=30)(time_masked)

S_dB = librosa.power_to_db(frequency_masked.squeeze(0).numpy(), ref=np.max)

# Scale the spectrogram to the range [0, 255]
S_scaled = (S_dB - np.min(S_dB)) / (np.max(S_dB) - np.min(S_dB)) * 255
S_scaled = S_scaled.astype('uint8')

# Create a PIL image object from the scaled spectrogram
print(S_scaled.shape)
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

# Show the image
pil_image.save('melspec'+str(4)+'.png')