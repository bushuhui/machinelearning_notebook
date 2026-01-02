

# GANs

## GANs 原理基本介绍

<p align="center">
<img src="./images/title.jpeg" width ="850" height="400"/>
</p>


GANs (Generative Adversarial Networks) 自从14年被提出，现在已经成为人工智能领域最火热的研究热点之一。我们所熟知的神经网络(CNN)只擅长分类任务，如在猫和狗、飞机和汽车之间进行分类。但现在，新的一种神经网络可以用来生成图片的猫或狗（有些生成出来看起来很奇怪），但是这告诉我们他们已经学会记住特征，这为很多具有创新性的工作提供了更多可能。

生成对抗网络是一种无监督模型，尽管大部分时候，有监督学习比无监督的能获得更好的训练效果，但真实世界中，有监督学习需要的数据标注（label）是相对少的。所以研究者们从未放弃去探索更好的无监督学习策略，希望能从海量的无标注数据中学到对于这个真实世界的表示（representation）甚至知识，从而去更好地理解我们的真实世界。

评价无监督学习好坏的方式有很多，其中生成任务就是最直接的一个。只有当我们能生成/创造我们的真实世界，才能说明我们是完完全全理解了它。然而，生成任务所依赖的生成式模型（generative models）往往会遇到两大困难。

**首先是我们需要大量的先验知识去对真实世界进行建模，其中包括选择什么样的先验、什么样的分布等等。而建模的好坏直接影响着我们的生成模型的表现。**

**另一个困难是，真实世界的数据往往很复杂，我们要用来拟合模型的计算量往往非常庞大，甚至难以承受。**

而在过去一两年中，有一个让人兴奋的新模型，则很好地避开了这两大困难。这个模型叫做 Generative Adversarial Networks（GAN）。在原始的 GAN paper中，作者是用博弈论来阐释了 GAN 框架背后的思想。每一个 GAN 框架，都包含着一对模型 —— 一个生成模型（G）和一个判别模型（D）。因为 D 的存在，才使得 GAN 中的 G 不再需要对于真实数据的先验知识和复杂建模，也能学习去逼近真实数据，最终让其生成的数据达到以假乱真的地步 —— D 也无法分辨—— 从而 G 和 D 达到了某种均衡。

由此我们可以看出生成对抗网络是一种生成模型（Generative Model），其背后基本思想是从训练库里获取很多训练样本，从而学习这些训练案例生成的概率分布。实现的方法，是让两个网络相互竞争，‘玩一个游戏’。其中一个叫做生成器网络（ Generator Network），它不断捕捉训练库里真实图片的概率分布，将输入的随机噪声（Random Noise）转变成新的样本（也就是假数据）。另一个叫做判别器网络（Discriminator Network），它可以同时观察真实和假造的数据，判断这个数据到底是不是真的。” 

更加直白的解释是：“GANs 的基本原理是它有两个模型：一个生成器，一个判别器。判别器的任务是判断给定图像是否看起来‘自然’，换句话说，是否像是人为（机器）生成的。而生成器的任务是，顾名思义，生成看起来‘自然’的图像，要求与原始数据分布尽可能一致。GANs 的运作方式可被看作是两名玩家之间的零和游戏。原论文的类比是，生成器就像一支造假币的团伙，试图用假币蒙混过关。而判别器就像是警察，目标是检查出假币。生成器想要骗过判别器，判别器想要不上当。当两组模型不断训练，生成器不断生成新的结果进行尝试，它们的能力互相提高，直到生成器生成的人造样本看起来与原始样本没有区别。虽然 GAN 背后的思想十分直观与朴素，但我们需要更进一步了解该理论背后的证明与推导。

https://zhuanlan.zhihu.com/p/29837245 对GANs的基本介绍了解
***http://speech.ee.ntu.edu.tw/~tlkagk/courses_MLDS17.html 李宏毅老师的视频讲解，基于公式的推导，讲解的非常好，入门GANs，这套视频就够了。***

---

## 参考文献

* [一文看懂生成对抗网络——GANS](http://xiaoqiang.me/?p=4592)  				
* [一文帮你发现各种出色的GAN变体](https://mp.weixin.qq.com/s?__biz=MzA3MzI4MjgzMw==&mid=2650724769&idx=2&sn=6fa540106cf6a5fd55fc39d057092888&chksm=871b1ddfb06c94c9e11d3a8281f60c0fce06a4e021fcd8eaab858c7f08ab9c939c4ad130e4b2&scene=21#wechat_redirect)					
*  [GANS——Zoo](https://github.com/zuokuijun/the-gan-zoo)	
*  [GAN原理与应用入门介绍](https://mp.weixin.qq.com/s?__biz=MzA3MzI4MjgzMw==&mid=2650730028&idx=1&sn=21d57cf54f257aeab15ebd4058671a2b&chksm=871b2a52b06ca3449f255549a914e8ab8d85bb4d43e0487a95fd9ffd97e708d9eac7a1f9943b&scene=21#wechat_redirect)				          
*  [GAN完整理论推导与实现](https://zhuanlan.zhihu.com/p/29837245)   		
* [用GAN自动生成二次元萌妹子](https://mp.weixin.qq.com/s?__biz=MzA3MzI4MjgzMw==&mid=2650729957&idx=4&sn=bdeb666588a1e926e2802b2daf3c2836&chksm=871b299bb06ca08db37816a9b8bce1d38ddf957f25f37bd39fb6e9ce2da5c1120373a2a1e888&scene=21#wechat_redirect)	
* [GAN之父NIPS 2016演讲现场直击](https://mp.weixin.qq.com/s?__biz=MzA3MzI4MjgzMw==&mid=2650721284&idx=1&sn=427e7f45c8253ab22a3960978409f5d1&chksm=871b087ab06c816c424ad03810be3e1b3aa9d6e99a5f325047796f110d178a07736f667d1a10&scene=21#wechat_redirect)				
* https://github.com/LynnHo/GAN-Papers



```python
"""
@function: GAN show demo
@data :2021-08-26
@author:zuo_kui_jun
"""
import argparse
import numpy as np
from scipy.stats import norm
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib import animation
import seaborn as sns  # Seaborn是比Matplotlib更高级的免费库
import cProfile

Count = 0
seed = 42
np.random.seed(seed)
tf.random.set_seed(seed)
tf.compat.v1.disable_eager_execution()

class DataDistribution(object):  # 真实数据分布
    def __init__(self):  # 只执行一次
        self.mu = 4  # 线　的参数  高斯分布相关

        self.sigma = 0.5

    def sample(self, N):  # 每次迭代都运行
        samples = np.random.normal(self.mu, self.sigma, N)  # 正态分布
        samples.sort()  # 画图   原来真实分布

        return samples


class GeneratorDistribution(object):  # 随机初始化分布,当成G输入
    def __init__(self, range):
        self.range = range

    def sample(self, N):  # 迭代次数的两倍

        return np.linspace(-self.range, self.range, N) + \
               np.random.random(N) * 0.01


def static_vars(**kwargs):
    def decorate(func):
        for k in kwargs:
            setattr(func, k, kwargs[k])
        return func

    return decorate


def linear(input, output_dim, scope=None, stddev=1.0):  # 100次迭代 运行了14次
    norm = tf.random_normal_initializer(stddev=stddev)  # 随机 w
    const = tf.constant_initializer(0.0)  # b

    with tf.compat.v1.variable_scope(scope or 'linear'):
        w = tf.compat.v1.get_variable('w', [input.get_shape()[1], output_dim], initializer=norm)  # 随机w
        b = tf.compat.v1.get_variable('b', [output_dim], initializer=const)
        return tf.matmul(input, w) + b


def generator(input, h_dim):
    h0 = tf.nn.softplus(linear(input, h_dim, 'g0'))
    h1 = linear(h0, 1, 'g1')

    return h1


def discriminator(input, h_dim):  # #初始化 w b
    h0 = tf.tanh(linear(input, h_dim * 2, 'd0'))
    h1 = tf.tanh(linear(h0, h_dim * 2, 'd1'))
    h2 = tf.tanh(linear(h1, h_dim * 2, scope='d2'))

    h3 = tf.sigmoid(linear(h2, 1, scope='d3'))

    return h3


# @static_vars(counter = 0)
def optimizer(loss, var_list, initial_learning_rate):
    decay = 0.95

    num_decay_steps = 150  # 每迭代150次 做一次学习率的衰减
    batch = tf.Variable(0)

    learning_rate = tf.compat.v1.train.exponential_decay(
        initial_learning_rate,
        batch,
        num_decay_steps,
        decay,
        staircase=True
    )
    optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate).minimize(
        loss,
        global_step=batch,
        var_list=var_list
    )
    # optimizer.counter += 1

    # print("counter:", optimizer.counter)
    return optimizer


class GAN(object):
    def __init__(self, data, gen, num_steps, batch_size, log_every):
        self.data = data  # DataDistribution
        self.gen = gen
        self.num_steps = num_steps
        self.batch_size = batch_size
        self.log_every = log_every
        self.mlp_hidden_size = 4  # 用神经网络模型当G D 网络时,隐层神经元个数
        self.count = 0
        self.learning_rate = 0.03  # 学习率

        self._create_model()

    def _create_model(self):  # 1/100次

        with tf.compat.v1.variable_scope('D_pre'):  # 先训练D的初始化
            self.pre_input = tf.compat.v1.placeholder(tf.float32, shape=(self.batch_size, 1))
            self.pre_labels = tf.compat.v1.placeholder(tf.float32, shape=(self.batch_size, 1))
            D_pre = discriminator(self.pre_input, self.mlp_hidden_size)  # 初始化 w b
            self.pre_loss = tf.reduce_mean(tf.square(D_pre - self.pre_labels))
            self.pre_opt = optimizer(self.pre_loss, None, self.learning_rate)

        # This defines the generator network - it takes samples from a noise
        # distribution as input, and passes them through an MLP.
        with tf.compat.v1.variable_scope('Gen'):
            self.z = tf.compat.v1.placeholder(tf.float32, shape=(self.batch_size, 1))
            self.G = generator(self.z, self.mlp_hidden_size)

        # The discriminator tries to tell the difference between samples from the
        # true data distribution (self.x) and the generated samples (self.z).
        #
        # Here we create two copies of the discriminator network (that share parameters),
        # as you cannot use the same network with different inputs in TensorFlow.
        with tf.compat.v1.variable_scope('Disc') as scope:
            self.x = tf.compat.v1.placeholder(tf.float32, shape=(self.batch_size, 1))
            self.D1 = discriminator(self.x, self.mlp_hidden_size)  # 构造D1网络,真实数据当做输入
            scope.reuse_variables()  # 重用
            self.D2 = discriminator(self.G, self.mlp_hidden_size)  # 构造D2网络,G数据当做输入

        # Define the loss for discriminator and generator networks (see the original
        # paper for details), and create optimizers for both
        self.loss_d = tf.reduce_mean(-tf.compat.v1.log(self.D1) - tf.compat.v1.log(1 - self.D2))  # 希望self.D1 向1...self.D2 向0   计算结果是均值
        self.loss_g = tf.reduce_mean(-tf.compat.v1.log(self.D2))  # 1

        self.d_pre_params = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope='D_pre')
        self.d_params = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope='Disc')
        self.g_params = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope='Gen')

        self.opt_d = optimizer(self.loss_d, self.d_params, self.learning_rate)
        self.opt_g = optimizer(self.loss_g, self.g_params, self.learning_rate)

    def train(self):  # 1

        with tf.compat.v1.Session() as session:
            tf.compat.v1.global_variables_initializer().run()

            # pretraining discriminator
            num_pretrain_steps = 1000
            for step in range(num_pretrain_steps):
                d = (np.random.random(self.batch_size) - 0.5) * 10.0
                labels = norm.pdf(d, loc=self.data.mu, scale=self.data.sigma)  # 高斯
                pretrain_loss, _ = session.run([self.pre_loss, self.pre_opt], {
                    self.pre_input: np.reshape(d, (self.batch_size, 1)),
                    self.pre_labels: np.reshape(labels, (self.batch_size, 1))
                })
            self.weightsD = session.run(self.d_pre_params)
            # copy weights from pre-training over to new D network
            for i, v in enumerate(self.d_params):
                session.run(v.assign(self.weightsD[i]))

            for step in range(self.num_steps):
                # update discriminator
                x = self.data.sample(self.batch_size)  # 原始的分布图  x是随机的
                z = self.gen.sample(self.batch_size)
                loss_d, _ = session.run([self.loss_d, self.opt_d], {  # 自动调网络?????
                    self.x: np.reshape(x, (self.batch_size, 1)),
                    self.z: np.reshape(z, (self.batch_size, 1))
                })

                # update generator
                z = self.gen.sample(self.batch_size)
                loss_g, _ = session.run([self.loss_g, self.opt_g], {
                    self.z: np.reshape(z, (self.batch_size, 1))
                })

                if step % self.log_every == 0:
                    print('{}: {}\t{}'.format(step, loss_d, loss_g))
                if step % 100 == 0 or step == 0 or step == self.num_steps - 1:
                    self._plot_distributions(session)

    def _samples(self, session, num_points=10000, num_bins=100):  # 2/100次

        xs = np.linspace(-self.gen.range, self.gen.range, num_points)
        bins = np.linspace(-self.gen.range, self.gen.range, num_bins)

        # data distribution
        d = self.data.sample(num_points)
        pd, _ = np.histogram(d, bins=bins, density=True)

        # generated samples
        zs = np.linspace(-self.gen.range, self.gen.range, num_points)
        g = np.zeros((num_points, 1))
        for i in range(num_points // self.batch_size):
            g[self.batch_size * i:self.batch_size * (i + 1)] = session.run(self.G, {
                self.z: np.reshape(
                    zs[self.batch_size * i:self.batch_size * (i + 1)],
                    (self.batch_size, 1)
                )
            })
        pg, _ = np.histogram(g, bins=bins, density=True)
        return pd, pg

    def _plot_distributions(self, session):  # 2/100

        pd, pg = self._samples(session)
        p_x = np.linspace(-self.gen.range, self.gen.range, len(pd))
        f, ax = plt.subplots(1)
        ax.set_ylim(0, 1)
        plt.plot(p_x, pd, label='real data')
        plt.plot(p_x, pg, label='generated data')
        plt.title('1D Generative Adversarial Network')
        plt.xlabel('Data values')
        plt.ylabel('Probability density')
        plt.legend()
        plt.show()


def main(args):
    model = GAN(
        DataDistribution(),  # 真实数据分布   只执行一下init
        GeneratorDistribution(range=8),  # 随机初始化分布 当成G 输入
        args.num_steps,  # 迭代次数
        args.batch_size,  # 一次迭代 个
        args.log_every,  # 隔多少次打印loss
    )
    model.train()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-steps', type=int, default=1200,  # 1200
                        help='the number of training steps to take')
    parser.add_argument('--batch-size', type=int, default=12,
                        help='the batch size')
    parser.add_argument('--log-every', type=int, default=10,
                        help='print loss after this many steps')
    return parser.parse_args()


if __name__ == '__main__':
    main(parse_args())


```

