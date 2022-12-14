from torch.nn import Module
from torch.nn import Conv2d
from torch.nn import Linear
from torch.nn import MaxPool2d
from torch.nn import ReLU
from torch.nn import LogSoftmax
from torch.nn import BatchNorm2d
from torch.nn import Dropout
from torch import flatten
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch

class MFCCS_net(nn.Module):

    def __init__(self, input_dim=(13, 431), output_dim=3):

        super().__init__()
       
        self.output_dim = output_dim
        self.input_dim = input_dim

        self.conv1 = nn.Conv2d(in_channels=1,out_channels=32,kernel_size=7)
        self.batch_normalization1 = BatchNorm2d(32)
        self.conv2 = nn.Conv2d(in_channels=32,out_channels=64,kernel_size=5, padding='same')
        self.batch_normalization2 = BatchNorm2d(64)
        self.conv3 = nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3, padding='same')
        self.batch_normalization3 = BatchNorm2d(128)
        self.conv4 = nn.Conv2d(in_channels=128,out_channels=256,kernel_size=3, padding='same')
        self.batch_normalization4 = BatchNorm2d(256)
        self.conv5 = nn.Conv2d(in_channels=256,out_channels=512,kernel_size=3, padding='same')
        self.batch_normalization5 = BatchNorm2d(512)
        self.fc1 = Linear(512*1*14,256)
        self.fc2 = Linear(256,self.output_dim)
        
        self.dropout = Dropout(p=0.5)

    def forward(self, x, mode=1):

        x = F.relu(F.max_pool2d(self.conv1(x),kernel_size=2, stride=2, padding=1))
        x = self.batch_normalization1(x)
        x = F.relu(F.max_pool2d(self.conv2(x),kernel_size=3, stride=2, padding=1))
        x = self.batch_normalization2(x)
        x = F.relu(F.max_pool2d(self.conv3(x),kernel_size=3, stride=2, padding=1))
        x = self.batch_normalization3(x)
        x = F.relu(F.max_pool2d(self.conv4(x),kernel_size=3, stride=2, padding=1))
        x = self.batch_normalization4(x)
        x = F.relu(F.max_pool2d(self.conv5(x),kernel_size=3, stride=2, padding=1))
        x = self.batch_normalization5(x)
        x = flatten(x,mode)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.softmax(self.fc2(x), dim=1)
        
        return x