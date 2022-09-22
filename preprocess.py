import torch
import os
import librosa
import numpy as np
import argparse

DATA_DIR = './data'

def build_mfccs(data_dir = DATA_DIR, train=True) :

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

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default = './data', help='directory where to find the data')
    args = parser.parse_args()
    data_dir = args.data_dir
    

    build_mfccs(data_dir,train=True)
    build_mfccs(data_dir, train=False)