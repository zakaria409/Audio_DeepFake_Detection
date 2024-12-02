import matplotlib.pyplot as plt
import tensorflow as tf
import keras
from keras.models import load_model
import librosa
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

import numpy as np 
import pandas as pd
import os
import shutil
import matplotlib.pyplot as plt
print(keras.__version__)

model = load_model('speakRec2.h5')
def extract_features(file_path):
    features = []

    audio, sr = librosa.load(file_path, sr=None, duration=1)
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13, hop_length=64)
                
        # Normalize MFCC features
    mfccs = StandardScaler().fit_transform(mfccs)
    
    desired_length = 32
    if mfccs.shape[1] < desired_length:
        # If the array has fewer than 32 frames, zero-pad it
        mfccs_padded = np.pad(mfccs, ((0, 0), (0, desired_length - mfccs.shape[1])), mode='constant')
        features.append(mfccs_padded.T)
    elif mfccs.shape[1] > desired_length:
        # If the array has more than 32 frames, truncate it
        mfccs_truncated = mfccs[:, :desired_length]
        features.append(mfccs_truncated.T)
    else:
        # If the array has exactly 32 frames, use it as is
        features.append(mfccs.T)    
        
    return np.array(features)

def predict_genre(model, audio_file_path, genre_mapping):

    # Load audio file
    signal, sample_rate = librosa.load(audio_file_path, sr=22050)

    # Extract MFCC features
    mfcc = librosa.feature.mfcc(y=signal, sr=sample_rate, n_mfcc=13, n_fft=2048, hop_length=512)
    mfcc = mfcc.T

    # MFCC'leri uygun boyuta getir
    mfcc = np.resize(mfcc, (130, 13, 1))

    # Reshape MFCC'leri uygun boyuta
    mfcc = mfcc[np.newaxis, ...]

    # Predict using the model
    prediction = model.predict(mfcc)
    predicted_index = np.argmax(prediction, axis=1)

    # Map predicted index to genre label
    genre_label = genre_mapping[predicted_index[0]]
    print("Raw prediction:", prediction)

    return genre_label

speaker_folders = ['Nelson_Mandela', 'Magaret_Tarcher', 'Benjamin_Netanyau', 'Jens_Stoltenberg', 'Julia_Gillard', 'Zakaria_Sameh']
file_path = "recording1.wav"

features = extract_features(file_path)
y_pred = model.predict(features)
rnd = np.random.randint(0, 1, 1)

y_pred = np.argmax(y_pred, axis=-1)[rnd]

print(
            "Speaker:\33{} {}\33[0m\tPredicted:\33{} {}\33[0m".format(
            "[92m",y_pred,
                "[92m", y_pred
                )
            )
            
print("Speaker Predicted:",speaker_folders[int(y_pred)])

print("Welcome")