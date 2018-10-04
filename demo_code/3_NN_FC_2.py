 
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from torchvision import datasets, transforms

# Training settings
batch_size = 64

# MNIST Dataset
dataset_path = "../data/mnist"
train_dataset = datasets.MNIST(root=dataset_path,
                               train=True,
                               transform=transforms.ToTensor(),
                               download=True)

test_dataset = datasets.MNIST(root=dataset_path,
                              train=False,
                              transform=transforms.ToTensor())

# Data Loader (Input Pipeline)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)


# define Network
seq_net = nn.Sequential(
    nn.Linear(28*28, 300),
    nn.ReLU(),
    nn.Linear(300, 100),
    nn.ReLU(),
    nn.Linear(100, 10)
)

# define optimizer & criterion
param = seq_net.parameters()
optim = torch.optim.Adam(param, 0.01)
criterion = nn.CrossEntropyLoss()

# train the network
for e in range(100):
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = Variable(data), Variable(target)
        data = data.view(-1, 784)

        out = seq_net(data)
        loss = criterion(out, target)
        optim.zero_grad()
        loss.backward()
        optim.step()

        if batch_idx % 100 == 0:
            pred = out.data.max(1, keepdim=True)[1]
            c = float(pred.eq(target.data.view_as(pred)).cpu().sum())/out.size(0)

            print("epoch: %5d, loss: %f, acc: %f" %
                  (e+1, loss.data[0], c))

