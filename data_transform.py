import torch
import numpy as np
import librosa
from torchvision import transforms as transforms

class Resize_Audio(object) :

  def __init__(self, std_length=220500) :
    self.std_length = std_length

  def __call__(self, sample) :
    signal, sr = sample['audio']
    signal_length = signal.shape[0]

    if(signal_length > self.std_length) :
      signal = signal[:self.std_length]

    elif (signal_length < self.std_length):
      padding_begin_length = np.ceil((self.std_length-signal_length)/2)
      padding_end_length = np.floor((self.std_length-signal_length)/2)
      padding_begin = np.zeros(int(padding_begin_length))
      padding_end = np.zeros(int(padding_end_length))
      signal = np.concatenate((padding_begin,signal,padding_end),0)

    return   {'audio' : (signal, sr), 'label' : sample['label']}

class Build_MFCCS(object) :

  def __call__(self, sample) :
    signal, sr = sample['audio']
    mfccs = librosa.feature.mfcc(y=signal, n_mfcc=13,sr=sr)
    #delta_mfccs = librosa.feature.delta(mfccs)
    #ddelta_mfccs = librosa.feature.delta(mfccs, order=2)
    #return {'mfccs' : mfccs, 'delta_mfccs' : delta_mfccs, 'ddelta_mfccs' : ddelta_mfccs}
    return {'audio' : mfccs, 'label' : sample['label']}

class To_Tensor(object):

    def __call__(self, sample):
        audio, label = sample['audio'], sample['label']
        audio = torch.unsqueeze(torch.from_numpy(audio),0)
        label = torch.from_numpy(label)
        return {'audio': audio,'label': label}
