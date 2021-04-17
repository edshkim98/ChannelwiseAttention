import os
import sys
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torch.utils.data.dataset import random_split
from torchvision import datasets, models, transforms
import torchvision
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import torch.optim as optim


device = torch.device('cuda')

criterion = nn.CrossEntropyLoss()

transform = transforms.Compose(
[transforms.Resize(224),
 transforms.ToTensor(),
 transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])])

batch_size = 16

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, drop_last = True, num_workers=2)

class SE(nn.Module):
    """
    Re-implementation of Squeeze-and-Excitation (SE) block described in:
        *Hu et al., Squeeze-and-Excitation Networks, arXiv:1709.01507*
    """

    def __init__(self, num_channels, reduction_ratio=2):
        """
        :param num_channels: No of input channels
        :param reduction_ratio: By how much should the num_channels should be reduced
        """
        super(SE, self).__init__()
        num_channels_reduced = num_channels // reduction_ratio
        self.reduction_ratio = reduction_ratio
        self.fc1 = nn.Linear(num_channels, num_channels_reduced, bias=True)
        self.fc2 = nn.Linear(num_channels_reduced, num_channels, bias=True)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_tensor):
        """
        :param input_tensor: X, shape = (batch_size, num_channels, H, W)
        :return: output tensor
        """
        batch_size, num_channels, H, W = input_tensor.size()
        # Average along each channel
        squeeze_tensor = input_tensor.view(batch_size, num_channels, -1).mean(dim=2)

        # channel excitation
        fc_out_1 = self.relu(self.fc1(squeeze_tensor))
        fc_out_2 = self.sigmoid(self.fc2(fc_out_1))

        a, b = squeeze_tensor.size()
        output_tensor = torch.mul(input_tensor, fc_out_2.view(a, b, 1, 1))
        return output_tensor
    
class Model(nn.Module):
    def __init__(self, version,trained = True):#, args):
        super(Model, self).__init__()
        self.model = models.resnet50(pretrained=True)

        if version == 'SE':
            for i in range(len(self.model.layer3)):
                self.model.layer3[i].se = SE(1024)

            for i in range(len(self.model.layer2)):
                self.model.layer2[i].se = SE(512)

            for i in range(len(self.model.layer1)):
                self.model.layer1[i].se = SE(256)

            for i in range(len(self.model.layer4)):
                self.model.layer4[i].se = SE(2048)
        
        else:
            raise NameError('Type correct attention modules')
                
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        for param in self.model.parameters():
            param.requires_grad = trained        
    def forward(self,inputs):
        out = self.model(inputs)
        return out

if __name__ == "__main__":  
    model = Model(version='SE')
    for param in model.parameters():
        param.requires_grad = True
        
    model.fc = nn.Linear(2048,10,bias=True)
    model.fc.requires_grad = True
    model.to(device);
    
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    np.random.seed(42)
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(10):  # loop over the dataset multiple times
        total = 0
        correct= 0
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            model.train()
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        with torch.no_grad(): 
            for i, data in enumerate(testloader):
                model.eval()
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                running_loss = loss.item()
                _,pred = torch.max(outputs.data,1)

                total += batch_size 
                correct += (pred == labels).sum().item()

        acc = 100. *correct / total
        print("Epoch {} ACC {} LOSS {}".format(epoch,acc,running_loss/len(testloader)))
    print('Finished Training')