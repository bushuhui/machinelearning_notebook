# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext_format_version: '1.2'
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
#   language_info:
#     codemirror_mode:
#       name: ipython
#       version: 3
#     file_extension: .py
#     mimetype: text/x-python
#     name: python
#     nbconvert_exporter: python
#     pygments_lexer: ipython3
#     version: 3.5.2
# ---

# # 多层神经网络和反向传播
#

# ## 神经元
#
# 神经元和感知器本质上是一样的，只不过我们说感知器的时候，它的激活函数是阶跃函数；而当我们说神经元时，激活函数往往选择为sigmoid函数或tanh函数。如下图所示：
# ![neuron](images/neuron.gif)
#
# 计算一个神经元的输出的方法和计算一个感知器的输出是一样的。假设神经元的输入是向量$\vec{x}$，权重向量是$\vec{w}$(偏置项是$w_0$)，激活函数是sigmoid函数，则其输出y：
# $$
# y = sigmod(\vec{w}^T \cdot \vec{x})
# $$
#
# sigmoid函数的定义如下：
# $$
# sigmod(x) = \frac{1}{1+e^{-x}}
# $$
# 将其带入前面的式子，得到
# $$
# y = \frac{1}{1+e^{-\vec{w}^T \cdot \vec{x}}}
# $$
#
# sigmoid函数是一个非线性函数，值域是(0,1)。函数图像如下图所示
# ![sigmod_function](images/sigmod.jpg)
#
# sigmoid函数的导数是：
# $$
# y = sigmod(x) \ \ \ \ \ \ (1) \\
# y' = y(1-y)
# $$
# 可以看到，sigmoid函数的导数非常有趣，它可以用sigmoid函数自身来表示。这样，一旦计算出sigmoid函数的值，计算它的导数的值就非常方便。
#
#

# ## 神经网络是啥?
#
# ![nn1](images/nn1.jpeg)
#
# 神经网络其实就是按照一定规则连接起来的多个神经元。上图展示了一个全连接(full connected, FC)神经网络，通过观察上面的图，我们可以发现它的规则包括：
#
# * 神经元按照层来布局。最左边的层叫做输入层，负责接收输入数据；最右边的层叫输出层，我们可以从这层获取神经网络输出数据。输入层和输出层之间的层叫做隐藏层，因为它们对于外部来说是不可见的。
# * 同一层的神经元之间没有连接。
# * 第N层的每个神经元和第N-1层的所有神经元相连(这就是full connected的含义)，第N-1层神经元的输出就是第N层神经元的输入。
# * 每个连接都有一个权值。
#
# 上面这些规则定义了全连接神经网络的结构。事实上还存在很多其它结构的神经网络，比如卷积神经网络(CNN)、循环神经网络(RNN)，他们都具有不同的连接规则。
#

# ## 计算神经网络的输出
#
# 神经网络实际上就是一个输入向量$\vec{x}$到输出向量$\vec{y}$的函数，即：
#
# $$
# \vec{y} = f_{network}(\vec{x})
# $$
# 根据输入计算神经网络的输出，需要首先将输入向量$\vec{x}$的每个元素的值$x_i$赋给神经网络的输入层的对应神经元，然后根据式1依次向前计算每一层的每个神经元的值，直到最后一层输出层的所有神经元的值计算完毕。最后，将输出层每个神经元的值串在一起就得到了输出向量$\vec{y}$。
#
# 接下来举一个例子来说明这个过程，我们先给神经网络的每个单元写上编号。
#
# ![nn2](images/nn2.png)
#
# 如上图，输入层有三个节点，我们将其依次编号为1、2、3；隐藏层的4个节点，编号依次为4、5、6、7；最后输出层的两个节点编号为8、9。因为我们这个神经网络是全连接网络，所以可以看到每个节点都和上一层的所有节点有连接。比如，我们可以看到隐藏层的节点4，它和输入层的三个节点1、2、3之间都有连接，其连接上的权重分别为$w_{41}$,$w_{42}$,$w_{43}$。那么，我们怎样计算节点4的输出值$a_4$呢？
#
#
# 为了计算节点4的输出值，我们必须先得到其所有上游节点（也就是节点1、2、3）的输出值。节点1、2、3是输入层的节点，所以，他们的输出值就是输入向量$\vec{x}$本身。按照上图画出的对应关系，可以看到节点1、2、3的输出值分别是$x_1$,$x_2$,$x_3$。我们要求输入向量的维度和输入层神经元个数相同，而输入向量的某个元素对应到哪个输入节点是可以自由决定的，你偏非要把$x_1$赋值给节点2也是完全没有问题的，但这样除了把自己弄晕之外，并没有什么价值。
#
# 一旦我们有了节点1、2、3的输出值，我们就可以根据式1计算节点4的输出值$a_4$：
# ![eqn_3_4](images/eqn_3_4.png)
#
# 上式的$w_{4b}$是节点4的偏置项，图中没有画出来。而$w_{41}$,$w_{42}$,$w_{43}$分别为节点1、2、3到节点4连接的权重，在给权重$w_{ji}$编号时，我们把目标节点的编号$j$放在前面，把源节点的编号$i$放在后面。
#
# 同样，我们可以继续计算出节点5、6、7的输出值$a_5$,$a_6$,$a_7$。这样，隐藏层的4个节点的输出值就计算完成了，我们就可以接着计算输出层的节点8的输出值$y_1$：
# ![eqn_5_6](images/eqn_5_6.png)
#
# 同理，我们还可以计算出$y_2$的值。这样输出层所有节点的输出值计算完毕，我们就得到了在输入向量$\vec{x} = (x_1, x_2, x_3)^T$时，神经网络的输出向量$\vec{y} = (y_1, y_2)^T$。这里我们也看到，输出向量的维度和输出层神经元个数相同。
#
#

# ## 神经网络的矩阵表示
#
# 神经网络的计算如果用矩阵来表示会很方便（当然逼格也更高），我们先来看看隐藏层的矩阵表示。
#
# 首先我们把隐藏层4个节点的计算依次排列出来：
# ![eqn_hidden_units](images/eqn_hidden_units.png)
#
# 接着，定义网络的输入向量$\vec{x}$和隐藏层每个节点的权重向量$\vec{w}$。令
#
# ![eqn_7_12](images/eqn_7_12.png)
#
# 代入到前面的一组式子，得到：
#
# ![eqn_13_16](images/eqn_13_16.png)
#
# 现在，我们把上述计算$a_4$, $a_5$,$a_6$,$a_7$的四个式子写到一个矩阵里面，每个式子作为矩阵的一行，就可以利用矩阵来表示它们的计算了。令
# ![eqn_matrix1](images/eqn_matrix1.png)
#
# 带入前面的一组式子，得到
# ![formular_2](images/formular_2.png)
#
# 在式2中，$f$是激活函数，在本例中是$sigmod$函数；$W$是某一层的权重矩阵；$\vec{x}$是某层的输入向量；$\vec{a}$是某层的输出向量。式2说明神经网络的每一层的作用实际上就是先将输入向量左乘一个数组进行线性变换，得到一个新的向量，然后再对这个向量逐元素应用一个激活函数。
#
# 每一层的算法都是一样的。比如，对于包含一个输入层，一个输出层和三个隐藏层的神经网络，我们假设其权重矩阵分别为$W_1$,$W_2$,$W_3$,$W_4$，每个隐藏层的输出分别是$\vec{a}_1$,$\vec{a}_2$,$\vec{a}_3$，神经网络的输入为$\vec{x}$，神经网络的输出为$\vec{y}$，如下图所示：
# ![nn_parameters_demo](images/nn_parameters_demo.png)
#
# 则每一层的输出向量的计算可以表示为：
# ![eqn_17_20](images/eqn_17_20.png)
#
#
# 这就是神经网络输出值的矩阵计算方法。
#

# ## 神经网络的训练 - 反向传播算法
#
# 现在，我们需要知道一个神经网络的每个连接上的权值是如何得到的。我们可以说神经网络是一个模型，那么这些权值就是模型的参数，也就是模型要学习的东西。然而，一个神经网络的连接方式、网络的层数、每层的节点数这些参数，则不是学习出来的，而是人为事先设置的。对于这些人为设置的参数，我们称之为超参数(Hyper-Parameters)。
#
# 反向传播算法其实就是链式求导法则的应用。然而，这个如此简单且显而易见的方法，却是在Roseblatt提出感知器算法将近30年之后才被发明和普及的。对此，Bengio这样回应道：
#
# > 很多看似显而易见的想法只有在事后才变得显而易见。
#
# 按照机器学习的通用套路，我们先确定神经网络的目标函数，然后用随机梯度下降优化算法去求目标函数最小值时的参数值。
#
# 我们取网络所有输出层节点的误差平方和作为目标函数：
# ![bp_loss](images/bp_loss.png)
#
# 其中，$E_d$表示是样本$d$的误差。
#
# 然后，使用随机梯度下降算法对目标函数进行优化：
# ![bp_weight_update](images/bp_weight_update.png)
#
# 随机梯度下降算法也就是需要求出误差$E_d$对于每个权重$w_{ji}$的偏导数（也就是梯度），怎么求呢？
# ![nn3](images/nn3.png)
#
# 观察上图，我们发现权重$w_{ji}$仅能通过影响节点$j$的输入值影响网络的其它部分，设$net_j$是节点$j$的加权输入，即
# ![eqn_21_22](images/eqn_21_22.png)
#
# $E_d$是$net_j$的函数，而$net_j$是$w_{ji}$的函数。根据链式求导法则，可以得到：
#
# ![eqn_23_25](images/eqn_23_25.png)
#
#
# 上式中，$x_{ji}$是节点传递给节点$j$的输入值，也就是节点$i$的输出值。
#
# 对于的$\frac{\partial E_d}{\partial net_j}$推导，需要区分输出层和隐藏层两种情况。
#
#

# ### 输出层权值训练
#
# ![nn3](images/nn3.png)
#
# 对于输出层来说，$net_j$仅能通过节点$j$的输出值$y_j$来影响网络其它部分，也就是说$E_d$是$y_j$的函数，而$y_j$是$net_j$的函数，其中$y_j = sigmod(net_j)$。所以我们可以再次使用链式求导法则：
# ![eqn_26](images/eqn_26.png)
#
# 考虑上式第一项:
# ![eqn_27_29](images/eqn_27_29.png)
#
#
# 考虑上式第二项：
# ![eqn_30_31](images/eqn_30_31.png)
#
# 将第一项和第二项带入，得到：
# ![eqn_ed_net_j.png](images/eqn_ed_net_j.png)
#
# 如果令$\delta_j = - \frac{\partial E_d}{\partial net_j}$，也就是一个节点的误差项$\delta$是网络误差对这个节点输入的偏导数的相反数。带入上式，得到：
# ![eqn_delta_j.png](images/eqn_delta_j.png)
#
# 将上述推导带入随机梯度下降公式，得到：
# ![eqn_32_34.png](images/eqn_32_34.png)
#

# ### 隐藏层权值训练
#
# 现在我们要推导出隐藏层的$\frac{\partial E_d}{\partial net_j}$。
#
# ![nn3](images/nn3.png)
#
# 首先，我们需要定义节点$j$的所有直接下游节点的集合$Downstream(j)$。例如，对于节点4来说，它的直接下游节点是节点8、节点9。可以看到$net_j$只能通过影响$Downstream(j)$再影响$E_d$。设$net_k$是节点$j$的下游节点的输入，则$E_d$是$net_k$的函数，而$net_k$是$net_j$的函数。因为$net_k$有多个，我们应用全导数公式，可以做出如下推导：
# ![eqn_35_40](images/eqn_35_40.png)
#
# 因为$\delta_j = - \frac{\partial E_d}{\partial net_j}$，带入上式得到：
# ![eqn_delta_hidden.png](images/eqn_delta_hidden.png)
#
#
# 至此，我们已经推导出了反向传播算法。需要注意的是，我们刚刚推导出的训练规则是根据激活函数是sigmoid函数、平方和误差、全连接网络、随机梯度下降优化算法。如果激活函数不同、误差计算方式不同、网络连接结构不同、优化算法不同，则具体的训练规则也会不一样。但是无论怎样，训练规则的推导方式都是一样的，应用链式求导法则进行推导即可。
#

# ###  具体解释
#
# 我们假设每个训练样本为$(\vec{x}, \vec{t})$，其中向量$\vec{x}$是训练样本的特征，而$\vec{t}$是样本的目标值。
#
# ![nn3](images/nn3.png)
#
# 首先，我们根据上一节介绍的算法，用样本的特征$\vec{x}$，计算出神经网络中每个隐藏层节点的输出$a_i$，以及输出层每个节点的输出$y_i$。
#
# 然后，我们按照下面的方法计算出每个节点的误差项$\delta_i$：
#
# * **对于输出层节点$i$**
# ![formular_3.png](images/formular_3.png)
# 其中，$\delta_i$是节点$i$的误差项，$y_i$是节点$i$的输出值，$t_i$是样本对应于节点$i$的目标值。举个例子，根据上图，对于输出层节点8来说，它的输出值是$y_1$，而样本的目标值是$t_1$，带入上面的公式得到节点8的误差项应该是：
# ![forumlar_delta8.png](images/forumlar_delta8.png)
#
# * **对于隐藏层节点**
# ![formular_4.png](images/formular_4.png)
#
# 其中，$a_i$是节点$i$的输出值，$w_{ki}$是节点$i$到它的下一层节点$k$的连接的权重，$\delta_k$是节点$i$的下一层节点$k$的误差项。例如，对于隐藏层节点4来说，计算方法如下：
# ![forumlar_delta4.png](images/forumlar_delta4.png)
#
#
# 最后，更新每个连接上的权值：
# ![formular_5.png](images/formular_5.png)
#
# 其中，$w_{ji}$是节点$i$到节点$j$的权重，$\eta$是一个成为学习速率的常数，$\delta_j$是节点$j$的误差项，$x_{ji}$是节点$i$传递给节点$j$的输入。例如，权重$w_{84}$的更新方法如下：
# ![eqn_w84_update.png](images/eqn_w84_update.png)
#
# 类似的，权重$w_{41}$的更新方法如下：
# ![eqn_w41_update.png](images/eqn_w41_update.png)
#
#
# 偏置项的输入值永远为1。例如，节点4的偏置项$w_{4b}$应该按照下面的方法计算：
# ![eqn_w4b_update.png](images/eqn_w4b_update.png)
#
# 我们已经介绍了神经网络每个节点误差项的计算和权重更新方法。显然，计算一个节点的误差项，需要先计算每个与其相连的下一层节点的误差项。这就要求误差项的计算顺序必须是从输出层开始，然后反向依次计算每个隐藏层的误差项，直到与输入层相连的那个隐藏层。这就是反向传播算法的名字的含义。当所有节点的误差项计算完毕后，我们就可以根据式5来更新所有的权重。
#
#

# ## Program

# +
% matplotlib inline

import numpy as np
from sklearn import datasets, linear_model
import matplotlib.pyplot as plt

# generate sample data
np.random.seed(0)
X, y = datasets.make_moons(200, noise=0.20)

# generate nn output target
t = np.zeros((X.shape[0], 2))
t[np.where(y==0), 0] = 1
t[np.where(y==1), 1] = 1

# plot data
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral)
plt.show()

# +
# generate the NN model
class NN_Model:
    epsilon = 0.01               # learning rate
    n_epoch = 1000               # iterative number
    
nn = NN_Model()
nn.n_input_dim = X.shape[1]      # input size
nn.n_output_dim = 2              # output node size
nn.n_hide_dim = 4                # hidden node size

nn.X = X
nn.y = y

# initial weight array
nn.W1 = np.random.randn(nn.n_input_dim, nn.n_hide_dim) / np.sqrt(nn.n_input_dim)
nn.b1 = np.zeros((1, nn.n_hide_dim))
nn.W2 = np.random.randn(nn.n_hide_dim, nn.n_output_dim) / np.sqrt(nn.n_hide_dim)
nn.b2 = np.zeros((1, nn.n_output_dim))

# defin sigmod & its derivate function
def sigmod(X):
    return 1.0/(1+np.exp(-X))

def sigmod_derivative(X):
    f = sigmod(X)
    return f*(1-f)

# network forward calculation
def forward(n, X):
    n.z1 = sigmod(X.dot(n.W1) + n.b1)
    n.z2 = sigmod(n.z1.dot(n.W2) + n.b2)
    return n


# use random weight to perdict
forward(nn, X)
y_pred = np.argmax(nn.z2, axis=1)

# plot data
plt.scatter(X[:, 0], X[:, 1], c=y_pred, cmap=plt.cm.Spectral)
plt.show()

# +
from sklearn.metrics import accuracy_score

y_true = np.array(nn.y).astype(float)

# back-propagation
def backpropagation(n, X, y):
    for i in range(n.n_epoch):
        # forward to calculate each node's output
        forward(n, X)
        
        # print loss, accuracy
        L = np.sum((n.z2 - y)**2)
        
        y_pred = np.argmax(nn.z2, axis=1)
        acc = accuracy_score(y_true, y_pred)
        
        print("epoch [%4d] L = %f, acc = %f" % (i, L, acc))
        
        # calc weights update
        d2 = n.z2*(1-n.z2)*(y - n.z2)
        d1 = n.z1*(1-n.z1)*(np.dot(d2, n.W2.T))
        
        # update weights
        n.W2 += n.epsilon * np.dot(n.z1.T, d2)
        n.b2 += n.epsilon * np.sum(d2, axis=0)
        n.W1 += n.epsilon * np.dot(X.T, d1)
        n.b1 += n.epsilon * np.sum(d1, axis=0)

nn.n_epoch = 2000
backpropagation(nn, X, t)


# +
# plot data
y_pred = np.argmax(nn.z2, axis=1)

plt.scatter(X[:, 0], X[:, 1], c=nn.y, cmap=plt.cm.Spectral)
plt.title("ground truth")
plt.show()

plt.scatter(X[:, 0], X[:, 1], c=y_pred, cmap=plt.cm.Spectral)
plt.title("predicted")
plt.show()

# -

# ## 如何使用类的方法封装多层神经网络?

# +
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt


# define sigmod
def sigmod(X):
    return 1.0/(1+np.exp(-X))


# generate the NN model
class NN_Model:
    def __init__(self, nodes=None):
        self.epsilon = 0.01                 # learning rate
        self.n_epoch = 1000                 # iterative number
        
        if not nodes:
            self.nodes = [2, 4, 2]          # default nodes size (from input -> output)
        else:
            self.nodes = nodes
    
    def init_weight(self):
        W = []
        B = []
        
        n_layer = len(self.nodes)
        for i in range(n_layer-1):
            w = np.random.randn(self.nodes[i], self.nodes[i+1]) / np.sqrt(self.nodes[i])
            b = np.random.randn(1, self.nodes[i+1])
            
            W.append(w)
            B.append(b)
            
        self.W = W
        self.B = B
    
    def forward(self, X):
        Z = []
        x0 = X
        for i in range(len(self.nodes)-1):
            z = sigmod(np.dot(x0, self.W[i]) + self.B[i])
            x0 = z
            
            Z.append(z)
        
        self.Z = Z
        return Z[-1]
        
    # back-propagation
    def backpropagation(self, X, y, n_epoch=None, epsilon=None):
        if not n_epoch: n_epoch = self.n_epoch
        if not epsilon: epsilon = self.epsilon
        
        self.X = X
        self.Y = y
        
        for i in range(n_epoch):
            # forward to calculate each node's output
            self.forward(X)

            self.evaluate()
            
            # calc weights update
            W = self.W
            B = self.B
            Z = self.Z
            
            D = []
            d0 = y
            n_layer = len(self.nodes)
            for j in range(n_layer-1, 0, -1):
                jj = j - 1
                z = self.Z[jj]
                
                if j == n_layer - 1:
                    d = z*(1-z)*(d0 - z)
                else:
                    d = z*(1-z)*np.dot(d0, W[j].T)
                    
                d0 = d
                D.insert(0, d)
            
            # update weights
            for j in range(n_layer-1, 0, -1):
                jj = j - 1
                
                if jj != 0:
                    W[jj] += epsilon * np.dot(Z[jj-1].T, D[jj])
                else:
                    W[jj] += epsilon * np.dot(X.T, D[jj])
                    
                B[jj] += epsilon * np.sum(D[jj], axis=0)
        
    def evaluate(self):
        z = self.Z[-1]
        
        # print loss, accuracy
        L = np.sum((z - self.Y)**2)
            
        y_pred = np.argmax(z, axis=1)
        y_true = np.argmax(self.Y, axis=1)
        acc = accuracy_score(y_true, y_pred)
        
        print("L = %f, acc = %f" % (L, acc))
        

# +
# generate sample data
np.random.seed(0)
X, y = datasets.make_moons(200, noise=0.20)

# generate nn output target
t = np.zeros((X.shape[0], 2))
t[np.where(y==0), 0] = 1
t[np.where(y==1), 1] = 1

# plot data
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral)
plt.show()

# +
# use the NN model and training
nn = NN_Model([2, 6, 2])
nn.init_weight()
nn.backpropagation(X, t, 2000)



# +
# predict results & plot results
y_res  = nn.forward(X)
y_pred = np.argmax(y_res, axis=1)

# plot data
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral)
plt.title("ground truth")
plt.show()

plt.scatter(X[:, 0], X[:, 1], c=y_pred, cmap=plt.cm.Spectral)
plt.title("predicted")
plt.show()
# -

# ## 深入分析

# +
# print some results

print(y_res[1:10, :])
# -

# **问题**
# 1. 我们希望得到的每个类别的概率
# 2. 如何做多分类问题？
# 3. 如何能让神经网络更快的训练好？
# 4. 如何抽象，让神经网络的类支持更多的类型的层

# ## Softmax & 交叉熵代价函数
#
# softmax经常被添加在分类任务的神经网络中的输出层，神经网络的反向传播中关键的步骤就是求导，从这个过程也可以更深刻地理解反向传播的过程，还可以对梯度传播的问题有更多的思考。
#
# ### softmax 函数
#
# softmax(柔性最大值)函数，一般在神经网络中， softmax可以作为分类任务的输出层。其实可以认为softmax输出的是几个类别选择的概率，比如我有一个分类任务，要分为三个类，softmax函数可以根据它们相对的大小，输出三个类别选取的概率，并且概率和为1。
#
# softmax函数的公式是这种形式：
# ![softmax](images/softmax.png)
#
# * $S_i$是经过softmax的类别概率输出
# * $z_k$是神经元的输出
#
# 更形象的如下图表示：
# ![softmax_demo](images/softmax_demo.png)
# softmax直白来说就是将原来输出是3,1,-3通过softmax函数一作用，就映射成为(0,1)的值，而这些值的累和为1（满足概率的性质），那么我们就可以将它理解成概率，在最后选取输出结点的时候，我们就可以选取概率最大（也就是值对应最大的）结点，作为我们的预测目标！
#
#
#
# 首先是神经元的输出，一个神经元如下图：
# ![softmax_neuron](images/softmax_neuron.png)
#
# 神经元的输出设为：
# ![softmax_neuron_output_eqn.png](images/softmax_neuron_output_eqn.png)
# 其中$W_{ij}$是第$i$个神经元的第$j$个权重，$b$是偏置。$z_i$表示该网络的第$i$个输出。
#
# 给这个输出加上一个softmax函数，那就变成了这样：
# ![softmax_neuron_output2_eqn.png](images/softmax_neuron_output2_eqn.png)
# $a_i$代表softmax的第$i$个输出值，右侧套用了softmax函数。
#
#
# ### 损失函数 loss function
#
# 在神经网络反向传播中，要求一个损失函数，这个损失函数其实表示的是真实值与网络的估计值的误差，知道误差了，才能知道怎样去修改网络中的权重。
#
# 损失函数可以有很多形式，这里用的是交叉熵函数，主要是由于这个求导结果比较简单，易于计算，并且交叉熵解决某些损失函数学习缓慢的问题。交叉熵的函数是这样的：
#
# ![cross_entropy_loss](images/cross_entropy_loss.png)
#
# 其中$y_i$表示真实的分类结果。
#
#

# ## References
# * 反向传播算法
#   * [零基础入门深度学习(3) - 神经网络和反向传播算法](https://www.zybuluo.com/hanbingtao/note/476663)
#   * [Neural Network Using Python and Numpy](https://www.python-course.eu/neural_networks_with_python_numpy.php)
#   * http://www.cedar.buffalo.edu/%7Esrihari/CSE574/Chap5/Chap5.3-BackProp.pdf
#   * https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/
# * Softmax & 交叉熵
#   * [交叉熵代价函数（作用及公式推导）](https://blog.csdn.net/u014313009/article/details/51043064)
#   * [手打例子一步一步带你看懂softmax函数以及相关求导过程](https://www.jianshu.com/p/ffa51250ba2e)
#   * [简单易懂的softmax交叉熵损失函数求导](https://www.jianshu.com/p/c02a1fbffad6)
