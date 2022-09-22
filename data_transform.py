import torch
import numpy as np
import librosa
from torchvision import transforms as transforms
from sklearn.preprocessing import StandardScaler

#extraida de https://github.com/tomasz-oponowicz/spoken_language_identification
def generate_fb_and_mfcc(signal, sample_rate):

    # Pre-Emphasis
    pre_emphasis = 0.97
    emphasized_signal = np.append(
        signal[0],
        signal[1:] - pre_emphasis * signal[:-1])

    # Framing
    frame_size = 0.025
    frame_stride = 0.01

    # Convert from seconds to samples
    frame_length, frame_step = (
        frame_size * sample_rate,
        frame_stride * sample_rate)
    signal_length = len(emphasized_signal)
    frame_length = int(round(frame_length))
    frame_step = int(round(frame_step))

    # Make sure that we have at least 1 frame
    num_frames = int(
        np.ceil(float(np.abs(signal_length - frame_length)) / frame_step))

    pad_signal_length = num_frames * frame_step + frame_length
    z = np.zeros((pad_signal_length - signal_length))

    # Pad Signal to make sure that all frames have equal
    # number of samples without truncating any samples
    # from the original signal
    pad_signal = np.append(emphasized_signal, z)

    indices = (
        np.tile(np.arange(0, frame_length), (num_frames, 1)) +
        np.tile(
            np.arange(0, num_frames * frame_step, frame_step),
            (frame_length, 1)
        ).T
    )
    frames = pad_signal[indices.astype(np.int32, copy=False)]

    # Window
    frames *= np.hamming(frame_length)

    # Fourier-Transform and Power Spectrum
    NFFT = 512

    # Magnitude of the FFT
    mag_frames = np.absolute(np.fft.rfft(frames, NFFT))

    # Power Spectrum
    pow_frames = ((1.0 / NFFT) * ((mag_frames) ** 2))

    # Filter Banks
    nfilt = 40

    low_freq_mel = 0

    # Convert Hz to Mel
    high_freq_mel = (2595 * np.log10(1 + (sample_rate / 2) / 700))

    # Equally spaced in Mel scale
    mel_points = np.linspace(low_freq_mel, high_freq_mel, nfilt + 2)

    # Convert Mel to Hz
    hz_points = (700 * (10**(mel_points / 2595) - 1))
    bin = np.floor((NFFT + 1) * hz_points / sample_rate)

    fbank = np.zeros((nfilt, int(np.floor(NFFT / 2 + 1))))
    for m in range(1, nfilt + 1):
        f_m_minus = int(bin[m - 1])   # left
        f_m = int(bin[m])             # center
        f_m_plus = int(bin[m + 1])    # right

        for k in range(f_m_minus, f_m):
            fbank[m - 1, k] = (k - bin[m - 1]) / (bin[m] - bin[m - 1])
        for k in range(f_m, f_m_plus):
            fbank[m - 1, k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m])
    filter_banks = np.dot(pow_frames, fbank.T)

    # Numerical Stability
    filter_banks = np.where(
        filter_banks == 0,
        np.finfo(float).eps,
        filter_banks)

    # dB
    filter_banks = 20 * np.log10(filter_banks)

    # MFCCs
    # num_ceps = 12
    # cep_lifter = 22

    # ### Keep 2-13
    # mfcc = dct(
    #     filter_banks,
    #     type=2,
    #     axis=1,
    #     norm='ortho'
    # )[:, 1 : (num_ceps + 1)]

    # (nframes, ncoeff) = mfcc.shape
    # n = np.arange(ncoeff)
    # lift = 1 + (cep_lifter / 2) * np.sin(np.pi * n / cep_lifter)
    # mfcc *= lift
    #filter_banks -= (np.mean(filter_banks, axis=0) + 1e-8)
    return filter_banks

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

class Build_MFCCS_librosa(object) :

  def __call__(self, sample) :
    signal, sr = sample['audio']
    mfccs = librosa.feature.mfcc(y=signal, n_mfcc=10,sr=sr)
    #delta_mfccs = librosa.feature.delta(mfccs)
    #ddelta_mfccs = librosa.feature.delta(mfccs, order=2)
    #return {'mfccs' : mfccs, 'delta_mfccs' : delta_mfccs, 'ddelta_mfccs' : ddelta_mfccs}
    return {'audio' : mfccs, 'label' : sample['label']}

class Build_MFCCS_kaggle(object) :
    
  def __call__(self, sample) :
    signal, sr = sample['audio']
    mfccs = generate_fb_and_mfcc(signal,sr)
    mfccs = np.array(mfccs)
    #delta_mfccs = librosa.feature.delta(mfccs)
    #ddelta_mfccs = librosa.feature.delta(mfccs, order=2)
    #return {'mfccs' : mfccs, 'delta_mfccs' : delta_mfccs, 'ddelta_mfccs' : ddelta_mfccs}
    return {'audio' : mfccs, 'label' : sample['label']}

class Normalize_Audio(object) :
      
    def __call__(self,sample) :
      signal = sample['audio']
      sc = StandardScaler()
      signal = sc.fit_transform(signal)
      return {'audio' : signal, 'label' : sample['label']}

class To_Tensor(object):

    def __call__(self, sample):
        signal, label = sample['audio'], sample['label']
        signal = torch.unsqueeze(torch.from_numpy(signal),0)
        if label is not None :
          label = torch.from_numpy(label)
        return {'audio': signal,'label': label}
