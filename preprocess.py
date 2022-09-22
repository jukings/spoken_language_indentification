import torch
import os
import librosa
import numpy as np

DATA_DIR = './data'

def build_mfccs(DATA_DIR = './data', train=True) :

    if train :
        for file in os.listdir(f'{DATA_DIR}/train/train_audio/') :
            signal, sr = librosa.load(f'{DATA_DIR}/train/train_audio/{file}')
            mfccs = librosa.feature.mfcc(y=signal, n_mfcc=10,sr=sr)
            new_file = file[:-4]
            np.save(f'{DATA_DIR}/train/train_mfccs/{new_file}mfccs',mfccs)
    if not train :
        for file in os.listdir(f'{DATA_DIR}/test/test_audio/') :
            signal, sr = librosa.load(f'{DATA_DIR}/test/test_audio/{file}')
            mfccs = librosa.feature.mfcc(y=signal, n_mfcc=10,sr=sr)
            new_file = file[:-4]
            np.save(f'{DATA_DIR}/test/test_mfccs/{new_file}mfccs',mfccs)

if __name__=='__main__' :
    build_mfccs(train=True)
    build_mfccs(train=False)