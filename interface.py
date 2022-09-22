import gradio as gr
import librosa
import numpy as np
import sys
from network import MFCCS_net 
from data_transform import Resize_Audio, Build_MFCCS_librosa, To_Tensor, Normalize_Signal
import torchvision.transforms as transforms
import torch
from torch.utils.data import DataLoader
import librosa
import argparse

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DIC_VALUE_TO_LABEL = {0 : 'en', 1 : 'de', 2 : 'es'}

def predict(audio_file_path) :
    print(audio_file_path)
    audio = librosa.load(audio_file_path)
    sample = {'audio' : audio, 'label' : np.zeros([0,0,0])}
    sample = transform(sample)
    input = sample['audio'].to(device, dtype=torch.float32)
    input = torch.unsqueeze(input,0)
    prediction = net.forward(input)
    prediction = int(torch.argmax(prediction))
    prediction = DIC_VALUE_TO_LABEL[prediction]
    return prediction

if __name__=='__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--weight_file', type=str, default = 'mfccs_net.pth', help='file containing the weight to use')

    args = parser.parse_args()
    weight_file = args.weight_file

    net = MFCCS_net()
    net = net.to(device)
    net.load_state_dict(torch.load(weight_file,map_location=device))
    net.eval()

    transform = transforms.Compose([Resize_Audio(),Build_MFCCS_librosa(),To_Tensor()])

    app = gr.Interface(
    fn=predict, 
    inputs=gr.Audio(source='microphone', type='filepath'), 
    outputs='text')

    app.launch(show_error=True, debug=True, share=True)
