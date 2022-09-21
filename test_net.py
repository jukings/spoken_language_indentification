from data_set import SpokenLanguageIdentification
import argparse
from network import MFCCS_net
from data_transform import Resize_Audio, Build_MFCCS, To_Tensor
import torchvision.transforms as transforms
import torch
from torch.utils.data import DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def test(model, dataloader):
    test_corrects = 0
    total = 0
    with torch.no_grad():
        for data in dataloader:
            inputs = data['audio'].to(device)
            labels = data['label'].to(device)
            prediction = model(inputs)
            prediction = model(inputs).argmax(1)
            test_corrects += prediction.eq(labels.argmax(1)).sum().item()
            total += labels.size(0)
    return test_corrects / total

if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--weight_file', type=str, default = 'mfccs_net.pth', help='file containing the weight to use')

    args = parser.parse_args()
    weight_file = args.weight_file

    net = MFCCS_net()
    net.to(device)
    net.load_state_dict(torch.load(weight_file,map_location=device))

    transform = transforms.Compose([Resize_Audio(),Build_MFCCS(),To_Tensor()])
    testset = SpokenLanguageIdentification('./data', train=False, transform=transform)
    testloader = DataLoader(testset, batch_size=1, shuffle=True, num_workers=0)

    test_acc = test(net,testloader)
    print(f'Test accuracy : {test_acc}')



