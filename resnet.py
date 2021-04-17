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

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

if __name__ == "__main__":  
    model = models.resnet50(pretrained=True)
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