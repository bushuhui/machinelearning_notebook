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

# ## 感知机
#
# 感知机（perceptron）是二分类的线性分类模型，输入为实例的特征向量，输出为实例的类别（取+1和-1）。感知机对应于输入空间中将实例划分为两类的分离超平面。感知机旨在求出该超平面，为求得超平面导入了基于误分类的损失函数，利用梯度下降法 对损失函数进行最优化（最优化）。感知机的学习算法具有简单而易于实现的优点，分为原始形式和对偶形式。感知机预测是用学习得到的感知机模型对新的实例进行预测的，因此属于判别模型。感知机由Rosenblatt于1957年提出的，是神经网络和支持向量机的基础。
#
# 模仿的是生物神经系统内的神经元，它能够接受来自多个源的信号输入，然后将信号转化为便于传播的信号在进行输出(在生物体内表现为电信号)。
#
# ![neuron](images/neuron.png)
#
# * dendrites - 树突
# * nucleus - 细胞核
# * axon - 轴突
#
# 心理学家Rosenblatt构想了感知机，它作为简化的数学模型解释大脑神经元如何工作：它取一组二进制输入值（附近的神经元），将每个输入值乘以一个连续值权重（每个附近神经元的突触强度），并设立一个阈值，如果这些加权输入值的和超过这个阈值，就输出1，否则输出0（同理于神经元是否放电）。对于感知机，绝大多数输入值不是一些数据，就是别的感知机的输出值。
#
# 麦卡洛克-皮兹模型缺乏一个对AI而言至关重要的学习机制。这就是感知机更出色的地方所在——罗森布拉特受到唐纳德·赫布(Donald Hebb) 基础性工作的启发，想出一个让这种人工神经元学习的办法。赫布提出了一个出人意料并影响深远的想法，称知识和学习发生在大脑主要是通过神经元间突触的形成与变化，简要表述为赫布法则：
#
# >当细胞A的轴突足以接近以激发细胞B，并反复持续地对细胞B放电，一些生长过程或代谢变化将发生在某一个或这两个细胞内，以致A作为对B放电的细胞中的一个，效率增加。
#
#
# 感知机并没有完全遵循这个想法，**但通过调输入值的权重，可以有一个非常简单直观的学习方案：给定一个有输入输出实例的训练集，感知机应该「学习」一个函数：对每个例子，若感知机的输出值比实例低太多，则增加它的权重，否则若设比实例高太多，则减少它的权重。**
#

# ## 1. 感知机模型
#
# 假设输入空间(特征向量)为X⊆Rn，输出空间为Y={-1, +1}。输入x∈X表示实例的特征向量，对应于输入空间的点；输出y∈Y表示示例的类别。由输入空间到输出空间的函数为
#
# $$
# f(x) = sign(w x + b)
# $$
#
# 称为感知机。其中，参数w叫做权值向量，b称为偏置。w·x表示w和x的内积。sign为符号函数，即
# ![sign_function](images/sign.png)
#
# ### 几何解释    
# 感知机模型是线性分类模型，感知机模型的假设空间是定义在特征空间中的所有线性分类模型，即函数集合{f|f(x)=w·x+b}。线性方程 w·x+b=0对应于特征空间Rn中的一个超平面S，其中w是超平面的法向量，b是超平面的截踞。这个超平面把特征空间划分为两部分。位于两侧的点分别为正负两类。超平面S称为分离超平面，如下图：
# ![perceptron_geometry_def](images/perceptron_geometry_def.png)
#
# ### 生物学类比
# ![perceptron_2](images/perceptron_2.PNG)
#
#
#

# ## 2. 感知机学习策略
#
# 假设训练数据集是线性可分的，感知机学习的目标是求得一个能够将训练数据的正负实例点完全分开的分离超平面，即最终求得参数w、b。这需要一个学习策略，即定义（经验）损失函数并将损失函数最小化。
#
# 损失函数的一个自然的选择是误分类的点的总数。但是这样得到的损失函数不是参数w、b的连续可导函数，不宜优化。损失函数的另一个选择是误分类点到分里面的距离之和。
#
# 首先，对于任意一点xo到超平面的距离为
# $$
# \frac{1}{||w||} | w \cdot xo + b |
# $$
#
# 其次，对于误分类点（xi,yi）来说 -yi(w·xi+b)>0
#
# 这样，假设超平面S的总的误分类点集合为M，那么所有误分类点到S的距离之和为
# $$
# -\frac{1}{||w||} \sum_{x_i \in M} y_i (w \cdot x_i + b)
# $$
# 不考虑1/||w||，就得到了感知机学习的损失函数。
#
# ### 经验风险函数
#
# 给定数据集T={(x1,y1),(x2,y2)...(xN,yN)}（其中xi∈X=Rn，yi∈Y={-1, +1}，i=1,2...N），感知机sign(w·x+b)学习的损失函数定义为
# $$
# L(w, b) = - \sum_{x_i \in M} y_i (w \cdot x_i + b)
# $$
# 其中M为误分类点的集合，这个损失函数就是感知机学习的[经验风险函数](https://blog.csdn.net/zhzhx1204/article/details/70163099)。
#
# 显然，损失函数L(w,b)是非负的。如果没有误分类点，那么L(w,b)为0，误分类点数越少，L(w,b)值越小。一个特定的损失函数：在误分类时是参数w,b的线性函数，在正确分类时，是0.因此，给定训练数据集T,损失函数L(w,b)是w,b的连续可导函数。
#

# ## 3. 感知机学习算法
#
#
# 最优化问题：给定数据集T={(x1,y1),(x2,y2)...(xN,yN)}（其中xi∈X=Rn，yi∈Y={-1, +1}，i=1,2...N），求参数w,b,使其成为损失函数的解（M为误分类的集合）：
#
# $$
# min_{w,b} L(w, b) =  - \sum_{x_i \in M} y_i (w \cdot x_i + b)
# $$
#
# 感知机学习是误分类驱动的，具体采用[随机梯度下降法](https://blog.csdn.net/zbc1090549839/article/details/38149561)。首先，任意选定$w_0$、$b_0$，然后用梯度下降法不断极小化目标函数，极小化的过程不知一次性的把M中的所有误分类点梯度下降，而是一次随机选取一个误分类点使其梯度下降。
#
# 假设误分类集合M是固定的，那么损失函数L(w,b)的梯度为
# $$
# \triangledown_w L(w, b) = - \sum_{x_i \in M} y_i x_i \\
# \triangledown_b L(w, b) = - \sum_{x_i \in M} y_i \\
# $$
#
# 随机选取一个误分类点$(x_i,y_i)$,对$w,b$进行更新：
# $$
# w = w + \eta y_i x_i \\
# b = b + \eta y_i
# $$
#
# 式中$\eta$（0 ≤ $ \eta $ ≤ 1）是步长，在统计学是中成为学习速率。步长越大，梯度下降的速度越快，更能接近极小点。如果步长过大，有可能导致跨过极小点，导致函数发散；如果步长过小，有可能会耗很长时间才能达到极小点。
#
# 直观解释：当一个实例点被误分类时，调整w,b，使分离超平面向该误分类点的一侧移动，以减少该误分类点与超平面的距离，直至超越该点被正确分类。
#
#
#
# 算法
# ```
# 输入：T={(x1,y1),(x2,y2)...(xN,yN)}（其中xi∈X=Rn，yi∈Y={-1, +1}，i=1,2...N，学习速率为η）
# 输出：w, b;感知机模型f(x)=sign(w·x+b)
# (1) 初始化w0,b0
# (2) 在训练数据集中选取（xi, yi）
# (3) 如果yi(w xi+b)≤0
#            w = w + ηyixi
#            b = b + ηyi
# (4) 转至（2）
# ```
#
#

# ## 4. Program
#

# +
import random
import numpy as np

# 符号函数
def sign(v):
    if v > 0:  return 1
    else:      return -1
    
def perceptron_train(train_data, eta=0.5, n_iter=100):
    weight = [0, 0]  # 权重
    bias = 0  # 偏置量
    learning_rate = eta  # 学习速率

    train_num = n_iter  # 迭代次数

    for i in range(train_num):
        #FIXME: the random chose sample is to slow
        train = random.choice(train_data)
        x1, x2, y = train
        predict = sign(weight[0] * x1 + weight[1] * x2 + bias)  # 输出
        #print("train data: x: (%2d, %2d) y: %2d  ==> predict: %2d" % (x1, x2, y, predict))
        
        if y * predict <= 0:  # 判断误分类点
            weight[0] = weight[0] + learning_rate * y * x1  # 更新权重
            weight[1] = weight[1] + learning_rate * y * x2
            bias      = bias      + learning_rate * y       # 更新偏置量
            print("update weight and bias: ", weight[0], weight[1], bias)

    #print("stop training: ", weight[0], weight[1], bias)

    return weight, bias

def perceptron_pred(data, w, b):
    y_pred = []
    for d in data:
        x1, x2, y = d
        yi = sign(w[0]*x1 + w[1]*x2 + b)
        y_pred.append(yi)
        
    return y_pred

# set training data
train_data = np.array([[1, 3,  1], [2, 5,  1], [3, 8,  1], [2, 6,  1], 
                       [3, 1, -1], [4, 1, -1], [6, 2, -1], [7, 3, -1]])

# do training
w, b = perceptron_train(train_data)
print("w = ", w)
print("b = ", b)

# predict 
y_pred = perceptron_pred(train_data, w, b)

print(train_data[:, 2])
print(y_pred)


# -

# ## Reference
# * [感知机（Python实现）](http://www.cnblogs.com/kaituorensheng/p/3561091.html)
# * [Programming a Perceptron in Python](https://blog.dbrgn.ch/2013/3/26/perceptrons-in-python/)
# * [损失函数、风险函数、经验风险最小化、结构风险最小化](https://blog.csdn.net/zhzhx1204/article/details/70163099)
