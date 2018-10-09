 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch.autograd import Variable


"""
Using torch to do time series analysis by LSTM model
"""

# load data
data_csv = pd.read_csv("./lstm_data.csv", usecols=[1])

#plt.plot(data_csv)
#plt.show()

# data pre-processing
data_csv = data_csv.dropna()
dataset = data_csv.values
dataset = dataset.astype("float32")
val_max = np.max(dataset)
val_min = np.min(dataset)
val_scale = val_max - val_min
dataset = (dataset - val_min) / val_scale


# generate dataset
def create_dataset(dataset, look_back=6):
    dataX, dataY = [], []
    dataset = dataset.tolist()
    for i in range(len(dataset) - look_back):
        a = dataset[i:(i+look_back)]
        dataX.append(a)
        dataY.append(dataset[i+look_back])

    return np.array(dataX), np.array(dataY)

look_back = 1
data_X, data_Y = create_dataset(dataset, look_back)


# split train/test dataset
train_size = int(len(data_X) * 0.7)
test_size = len(data_X) - train_size

train_X = data_X[:train_size]
train_Y = data_Y[:train_size]
test_X  = data_X[train_size:]
test_Y  = data_Y[train_size:]


# convert data for torch
train_X = train_X.reshape(-1, 1, look_back)
train_Y = train_Y.reshape(-1, 1, 1)
test_X  = test_X.reshape(-1, 1, look_back)

train_x = torch.from_numpy(train_X).float()
train_y = torch.from_numpy(train_Y).float()
test_x  = torch.from_numpy(test_X).float()

# define LSTM model
class LSTM_Reg(nn.Module):
    def __init__(self, input_size, hidden_size, output_size=1, num_layer=2):
        super(LSTM_Reg, self).__init__()

        self.rnn = nn.LSTM(input_size, hidden_size, num_layer)
        self.reg = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x, _ = self.rnn(x)
        s, b, h = x.shape
        x = x.view(s*b, h)
        x = self.reg(x)
        x = x.view(s, b, -1)
        return x

net = LSTM_Reg(look_back, 4, num_layer=1)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(net.parameters(), lr=1e-2)


for e in range(1000):
    var_x = Variable(train_x)
    var_y = Variable(train_y)

    # forward
    out = net(var_x)
    loss = criterion(out, var_y)

    # backward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # print progress
    if e % 100 == 0:
        print("epoch: %5d, loss: %.5f" % (e, loss.data[0]))

# do test
net = net.eval()

data_X = data_X.reshape(-1, 1, look_back)
data_X = torch.from_numpy(data_X).float()
var_data = Variable(data_X)
pred_test = net(var_data)

pred_test = pred_test.view(-1).data.numpy()

# plot
plt.plot(pred_test, 'r', label="Prediction")
plt.plot(dataset, 'b', label="Real")
plt.legend(loc="best")
plt.show()