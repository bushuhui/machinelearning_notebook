import torch
from torch import nn, optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.nn.functional as F

from torchvision import transforms
from torchvision import datasets

# set parameters
batch_size      = 32
learning_rate   = 1e-2
num_epoches     = 50

# download & load MNIST dataset
dataset_path = "../data/mnist"

train_dataset = datasets.MNIST(
    root=dataset_path, train=True, transform=transforms.ToTensor(), download=True)

test_dataset = datasets.MNIST(
    root=dataset_path, train=False, transform=transforms.ToTensor())

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False)


# Define the network
class NeuralNetwork(nn.Module):
    def __init__(self, in_dim, n_hidden_1, n_hidden_2, out_dim):
        super(NeuralNetwork, self).__init__()
        self.layer1 = nn.Linear(in_dim, n_hidden_1)
        self.layer2 = nn.Linear(n_hidden_1, n_hidden_2)
        self.layer3 = nn.Linear(n_hidden_2, out_dim)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = self.layer3(x)
        return x


# create network & define loss function
model = NeuralNetwork(28 * 28, 300, 100, 10)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

# train
for epoch in range(num_epoches):
    print("epoch %6d" % int(epoch+1))
    print('-' * 40)

    running_loss = 0.0
    running_acc = 0.0

    for i, data in enumerate(train_loader, 1):
        img, label = data
        
        img   = Variable(img.view(img.size(0), -1))
        label = Variable(label)
        
        # 向前传播
        optimizer.zero_grad()

        out = model(img)
        loss = criterion(out, label)
        running_loss += loss.data[0] * label.size(0)
        
        pred = out.data.max(1, keepdim=True)[1]
        running_acc += float(pred.eq(label.data.view_as(pred)).cpu().sum())
        
        # 向后传播
        loss.backward()
        optimizer.step()

        if i % 300 == 0:
            print('[{}/{}] Loss: {:.6f}, Acc: {:.2f}%'.format(
                epoch + 1, num_epoches, 
                1.0*running_loss / (batch_size * i),
                100.0*running_acc  / (batch_size * i)))
    
    # do test
    model.eval()
    eval_loss = 0.
    eval_acc = 0.

    for data in test_loader:
        img, label = data
        img = img.view(img.size(0), -1)

        img = Variable(img)
        label = Variable(label)
        
        out = model(img)
        loss = criterion(out, label)
        
        eval_loss += loss.data[0] * label.size(0)
        pred = out.data.max(1, keepdim=True)[1]
        eval_acc += float(pred.eq(label.data.view_as(pred)).cpu().sum())
        

    print('\nTest Loss: {:.6f}, Acc: {:.2f}%'.format(
        1.0*eval_loss / (len(test_dataset)), 
        100.0*eval_acc  / (len(test_dataset))))
    print()


# save model
torch.save(model.state_dict(), './model_Neural_Network.pth')
