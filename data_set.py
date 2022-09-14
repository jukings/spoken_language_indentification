from torch.utils.data import Dataset
import torch
import os
import librosa
import numpy as np

DATA_DIR = './data'
DIC_VALUE_TO_LABEL = {0 : 'en', 1 : 'de', 2 : 'es'}
DIC_LABEL_TO_VALUE = {'en' : 0, 'de' : 1, 'es' : 2}

class SpokenLanguageIdentification(Dataset):
    
    def __init__(self, data_dir, train=True, transform=None):
        
        if train is True :
          self.data_dir = f'{data_dir}/train/train'
          self.path_list = os.listdir(self.data_dir)
        else :
          self.data_dir = f'{data_dir}/test/test'
          self.path_list = os.listdir(self.data_dir)

        self.transform = transform

    def __len__(self):
        return len(self.path_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        audio = librosa.load(f'{self.data_dir}/{self.path_list[idx]}')
        label = self.path_list[idx][:2]
        tmp = np.zeros(3)
        tmp[int(DIC_LABEL_TO_VALUE[label])]=1
        label = tmp

        sample = {'audio': audio, 'label': label}

        if self.transform is not None:
            sample = self.transform(sample)

        return sample