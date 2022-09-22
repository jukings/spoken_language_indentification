from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
import numpy as np
import torch
import argparse
from network import MFCCS_net
import torchvision.transforms as transforms
from torchvision.utils import make_grid
import torch.nn as nn
from tqdm import tqdm
from data_transform import Resize_Audio, Build_MFCCS_librosa, Build_MFCCS_kaggle, To_Tensor, Normalize_Signal
from data_set import Data_FromAudioFile, Data_FromMFCCSFile
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def loader_loss(net, dataloader) :
    test_corrects = 0
    total = 0
    running_loss = []
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for data in dataloader:
            inputs = data['audio'].to(device)
            labels = data['label'].to(device)
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            running_loss.append(loss.item())
    return running_loss

def train(net, optimizer, trainloader, testloader, writer, epochs, scheduler=None):
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        running_loss = []
        t = tqdm(trainloader)

        for data in t:
            inputs = data['audio'].to(device)
            labels = data['label'].to(device)
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            running_loss.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            test_loss = np.mean(loader_loss(net,testloader))
            train_loss = np.mean(running_loss)

            t.set_description(f'Epochs {epoch+1}/{epochs} -- training loss : {train_loss} -- test loss : {test_loss}')
        
        if scheduler is not None :
            scheduler.step()

        writer.add_scalars('loss', {'train' : np.mean(running_loss), 'test' : test_loss}, epoch)
        
        test_acc = loader_accuracy(net, testloader)
        train_acc = loader_accuracy(net, trainloader)
        writer.add_scalars('accuracy loss', {'train' : train_acc, 'test' : test_acc}, epoch)

def loader_accuracy(model, dataloader):
    test_corrects = 0
    total = 0
    with torch.no_grad():
        for data in dataloader:
            inputs = data['audio'].to(device)
            labels = data['label'].to(device)
            prediction = model(inputs).argmax(1)
            test_corrects += prediction.eq(labels.argmax(1)).sum().item()
            total += labels.size(0)
    return test_corrects / total

if __name__=='__main__':

    print(f'Using device {device}')

    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, default = 'MFCCS_net', help='experiment name')
    parser.add_argument('--batch_size', type=int, default = 64, help='batch size')
    parser.add_argument('--epochs', type=int, default = 10, help='epochs')
    parser.add_argument('--lr', type=float, default = 10**-3, help='learning rate')
    parser.add_argument('--optimizer', type=str, default = 'SGD', help='optimizer to use for learning')
    parser.add_argument('--data_dir', type=str, default = './data', help='directory where to find the data')

    args = parser.parse_args()
    exp_name = args.exp_name
    epochs = args.epochs
    batch_size = args.batch_size
    lr = args.lr
    optimizer = args.optimizer
    data_dir = args.data_dir

    writer = SummaryWriter(f'runs/{args.exp_name}')

    
    transform = transforms.Compose([To_Tensor()])
    
    trainset = Data_FromMFCCSFile(data_dir, train=True, transform=transform)
    testset = Data_FromMFCCSFile(data_dir, train=False, transform=transform)

    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=True, num_workers=0)

    net = MFCCS_net()
    net = net.to(device)

    if optimizer == 'SGD' :
        optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9)
    elif optimizer == 'Adam' :
        optimizer = optim.Adam(net.parameters(),lr=lr)
    else :
        raise TypeError("Not valid optimizer")

    scheduler = StepLR(optimizer, step_size= 1, gamma=0.9)

    train(net, optimizer, trainloader, testloader, writer, epochs)
    test_acc = loader_accuracy(net,testloader)
    print(f'Test accuracy : {test_acc}')
    torch.save(net.state_dict(), "mfccs_net.pth")
    
    audio_example = os.listdir(f'{data_dir}/train/train')[0]
    
    mfccs_example = os.listdir(f'{data_dir}/train/train_mfccs')[0]
    mfccs_example = np.load(f'{data_dir}/train/train_mfccs/{mfccs_example}')
    mfccs_example = torch.from_numpy(mfccs_example).unsqueeze(0).unsqueeze(0).to(device)

    writer.add_graph(net, mfccs_example)


    
    