
import torch
from torch import nn, optim
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt


torch.manual_seed(2018)

# model's real-parameters
w_target = 3
b_target = 10

# generate data
n_data = 100
x_train = np.random.rand(n_data, 1)*20 - 10
y_train = w_target*x_train + b_target + (np.random.randn(n_data, 1)*10-5.0)

# draw the data
plt.plot(x_train, y_train, 'bo')
plt.show()


# convert to tensor
x_train = torch.from_numpy(x_train).float()
y_train = torch.from_numpy(y_train).float()



# Linear Regression Model
class LinearRegression(nn.Module):
    def __init__(self):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(1, 1)  # input and output is 1 dimension

    def forward(self, x):
        out = self.linear(x)
        return out

# create the model
model = LinearRegression()

# 定义loss和优化函数
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=1e-4)

# 开始训练
num_epochs = 1000
for epoch in range(num_epochs):
    inputs = Variable(x_train)
    target = Variable(y_train)

    # forward
    out = model(inputs)
    loss = criterion(out, target)

    # backward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch+1) % 20 == 0:
        print('Epoch[{}/{}], loss: {:.6f}'
              .format(epoch+1, num_epochs, loss.data[0]))


# do evaluation & plot
model.eval()
predict = model(Variable(x_train))
predict = predict.data.numpy()
plt.plot(x_train.numpy(), y_train.numpy(), 'bo', label='Real')
plt.plot(x_train.numpy(), predict, 'ro', label='Estimated')

plt.legend()
plt.show()

