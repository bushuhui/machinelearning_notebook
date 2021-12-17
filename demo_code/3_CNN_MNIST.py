
import torch
import torch.nn as nn
import torch.nn.functional as F
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


# Define the network
class Net_CNN(nn.Module):
    def __init__(self):
        super(Net_CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1   = nn.Linear(16*4*4, 120)
        self.fc2   = nn.Linear(120, 84)
        self.fc3   = nn.Linear(84, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        x = x.view(x.size()[0], -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# define optimizer & criterion
model = Net_CNN()
optim = torch.optim.Adam(model.parameters(), 0.01)
criterion = nn.CrossEntropyLoss()


# train the network
for e in range(100):
    # train
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = Variable(data), Variable(target)

        # inference, loss calculation
        out = model(data)
        loss = criterion(out, target)
        
        # backward, optimize
        optim.zero_grad()
        loss.backward()
        optim.step()

        # print loss, acc
        if batch_idx % 100 == 0:
            pred = out.data.max(1, keepdim=True)[1]
            c = float(pred.eq(target.data.view_as(pred)).cpu().sum() ) /out.size(0)

            print("epoch: %5d, loss: %f, acc: %f" %
                  ( e +1, loss.data.item(), c))

    # test
    model.eval()

    test_loss = 0.0
    correct = 0.0
    for data, target in test_loader:
        data, target = Variable(data), Variable(target)
        output = model(data)

        # sum up batch loss
        test_loss += criterion(output, target).item()

        # get the index of the max
        pred = output.data.max(1, keepdim=True)[1]
        correct += float(pred.eq(target.data.view_as(pred)).cpu().sum())

    test_loss /= len(test_loader.dataset)
    print("\nTest set: Average loss: %.4f, Accuracy: %6d/%6d (%4.2f %%)\n" %
          (test_loss,
           correct, len(test_loader.dataset),
           100.0*correct / len(test_loader.dataset)) )
