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
class NN_FC1(nn.Module):
    def __init__(self):
        super(NN_FC1, self).__init__()
        self.l1 = nn.Linear(784, 520)
        self.l2 = nn.Linear(520, 320)
        self.l3 = nn.Linear(320, 240)
        self.l4 = nn.Linear(240, 120)
        self.l5 = nn.Linear(120, 10)

    def forward(self, x):
        x = x.view(-1, 784)  # Flatten the data (n, 1, 28, 28)-> (n, 784)
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        x = F.relu(self.l4(x))
        return self.l5(x)

# Define the network
class NN_FC2(nn.Module):
    def __init__(self):
        super(NN_FC2, self).__init__()

        in_dim      = 28*28
        n_hidden_1  = 300
        n_hidden_2  = 100
        out_dim     = 10

        self.layer1 = nn.Linear(in_dim, n_hidden_1)
        self.layer2 = nn.Linear(n_hidden_1, n_hidden_2)
        self.layer3 = nn.Linear(n_hidden_2, out_dim)

    def forward(self, x):
        x = x.view(-1, 784)
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = self.layer3(x)
        return x


# create the NN object
model = NN_FC2()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)


def train(epoch):
    model.train()

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print("Train epoch: %6d [%6d/%6d (%.0f %%)] \t Loss: %.6f" % (
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()) )


def test():
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

for epoch in range(1, 10):
    train(epoch)
    test()
