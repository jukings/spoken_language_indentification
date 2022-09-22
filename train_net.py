from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
import numpy as np
import torch
import argparse
from network import MFCCS_net
import torchvision.transforms as transforms
import torch.nn as nn
from tqdm import tqdm
from data_transform import Resize_Audio, Build_MFCCS, To_Tensor, Normalize_Audio
from data_set import SpokenLanguageIdentification
from torch.utils.data import DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train(net, optimizer, loader, writer, epochs, train_sample):
    criterion = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        running_loss = []
        t = tqdm(loader)
        tmp_count = 0
        
        for data in t:
            if tmp_count > train_sample :
                break
            tmp_count+=1
            inputs = data['audio'].to(device)
            labels = data['label'].to(device)
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            running_loss.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            t.set_description(f'training loss: {np.mean(running_loss)}')
        writer.add_scalar('training loss', np.mean(running_loss), epoch)

def test(model, dataloader):
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
    parser.add_argument('--train_sample', type=float, default = 10**6, help='number of maximum sample for training')

    args = parser.parse_args()
    exp_name = args.exp_name
    epochs = args.epochs
    batch_size = args.batch_size
    lr = args.lr
    train_sample = args.train_sample

    writer = SummaryWriter(f'runs/{args.exp_name}')

    transform = transforms.Compose([Resize_Audio(),Build_MFCCS(),To_Tensor()])
    
    trainset = SpokenLanguageIdentification('./data', train=True, transform=transform)
    testset = SpokenLanguageIdentification('./data', train=False, transform=transform)

    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=True, num_workers=0)

    net = MFCCS_net()
    net = net.to(device)
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9)

    train(net, optimizer, trainloader, writer, epochs, train_sample)
    test_acc = test(net,testloader)
    print(f'Test accuracy : {test_acc}')
    torch.save(net.state_dict(), "mfccs_net.pth")
    
    image_example = torch.zeros(1,1,13,431)
    writer.add_graph(net, image_example)
    