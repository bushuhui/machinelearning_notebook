import time

import torch as t
from torch import nn, optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import datasets

"""
    Use pytorch nn.Module to implement logistic regression
    
"""


# define hyper parameters
batch_size = 32
learning_rate = 1e-3
num_epoches = 100

# download/load MNIST dataset
dataset_path = "../data/mnist"

train_dataset = datasets.MNIST(
    root=dataset_path, train=True, transform=transforms.ToTensor(), download=True)

test_dataset = datasets.MNIST(
    root=dataset_path, train=False, transform=transforms.ToTensor())

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False)


# define Logistic Regression model
class Logstic_Regression(nn.Module):
    def __init__(self, in_dim, n_class):
        super(Logstic_Regression, self).__init__()
        self.logstic = nn.Linear(in_dim, n_class)

    def forward(self, x):
        out = self.logstic(x)
        return out


model = Logstic_Regression(28 * 28, 10)     # model's input/output node size
use_gpu = t.cuda.is_available()             # GPU use or not
if use_gpu: model = model.cuda()

# define loss & optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

# training
for epoch in range(num_epoches):
    print('-' * 40)
    print('epoch {}'.format(epoch + 1))

    since = time.time()
    running_loss = 0.0
    running_acc = 0.0
    for i, data in enumerate(train_loader, 1):
        img, label = data
        img = img.view(img.size(0), -1)  # convert input image to dimensions of (n, 28x28)
        if use_gpu:
            img = Variable(img).cuda()
            label = Variable(label).cuda()
        else:
            img = Variable(img)
            label = Variable(label)

        # forward calculation
        out = model(img)
        loss = criterion(out, label)
        running_loss += loss.data[0] * label.size(0)
        _, pred = t.max(out, 1)
        num_correct = (pred == label).sum()
        running_acc += float(num_correct.data[0])

        # bp
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 300 == 0:
            print('[{}/{}] Loss: {:.6f}, Acc: {:.6f}'.format(
                epoch + 1, num_epoches, running_loss / (batch_size * i),
                running_acc / (batch_size * i)))

    print('Finish {} epoch, Loss: {:.6f}, Acc: {:.6f}'.format(
        epoch + 1, running_loss / (len(train_dataset)), running_acc / (len(
            train_dataset))))

    model.eval()
    eval_loss = 0.
    eval_acc = 0.

    for data in test_loader:
        img, label = data
        img = img.view(img.size(0), -1)
        if use_gpu:
            img = Variable(img, volatile=True).cuda()
            label = Variable(label, volatile=True).cuda()
        else:
            img = Variable(img, volatile=True)
            label = Variable(label, volatile=True)
        out = model(img)
        loss = criterion(out, label)
        eval_loss += loss.data[0] * label.size(0)
        _, pred = t.max(out, 1)
        num_correct = (pred == label).sum()
        eval_acc += float(num_correct.data[0])

    print('Test Loss: {:.6f}, Acc: {:.6f}'.format(
        eval_loss / (len(test_dataset)),
        eval_acc / (len(test_dataset))))
    print('Time:{:.1f} s'.format(time.time() - since))
    print()

# save model's parameters
#t.save(model.state_dict(), './model_LogsticRegression.pth')
