# Transformer

Transformers是神经机器翻译中使用的一种神经网络，它主要涉及将输入序列转换为输出序列的任务。Transformer中抛弃了传统的CNN和RNN，整个网络结构完全是由**Attention机制**组成。采用Attention机制的原因是考虑到RNN（或者LSTM，GRU等）的计算限制为是串行的，即RNN相关算法只能从左向右依次计算或者从右向左依次计算，这种机制带来了两个问题：

1. 时间片`t`的计算依赖`t−1`时刻的计算结果，这样限制了模型的**并行能力**。
2. 串行计算的过程中信息会丢失，尽管LSTM等门机制的结构一定程度上**缓解了长期依赖**的问题，但是对于特别长期的依赖现象，LSTM依旧无能为力。

Transformer的提出解决了上面两个问题，首先它使用了Attention机制，将序列中的任意两个位置之间的距离是缩小为一个常量；然后它使用的不是类似RNN的顺序结构，**具有更好的并行性，符合现有的GPU框架**。

### Transformer模型组成

Transformer是一种基于自注意力机制（self-attention mechanism）的模型架构，Transformer模型通常由以下几个关键部分组成：

* **嵌入层（Embedding Layer）**： 将输入序列中的词或字符转换为连续的向量表示。通常会有两种嵌入层，分别是词嵌入层（Word Embedding）和位置嵌入层（Positional Embedding）。
* **编码器（Encoder）**： 由多个编码器层堆叠而成。每个编码器层都包含了多头自注意力机制（Multi-Head Self-Attention）和全连接前馈网络（Feed-Forward Neural Network）两部分。编码器用于将输入序列转换为上下文感知的编码表示。
* **解码器（Decoder）（可选）**： 在某些任务中可能需要使用解码器来生成目标序列。解码器也由多个解码器层堆叠而成，每个解码器层包含了多头自注意力机制、编码器-解码器注意力机制（Encoder-Decoder Attention）和全连接前馈网络。
* **自注意力机制（Self-Attention Mechanism）**： 用于计算输入序列中每个位置的上下文相关性。自注意力机制使模型能够在输入序列中建立全局的依赖关系，从而更好地捕获序列中的长距离依赖关系。
* **全连接前馈网络（Feed-Forward Neural Network）**： 对于每个位置的表示，都会经过一个全连接前馈网络进行非线性变换和映射。
* **残差连接（Residual Connections）和层归一化（Layer Normalization）**： 在每个编码器和解码器层之间都会添加残差连接和层归一化，以稳定模型训练过程并加速模型收敛。

通过堆叠多个编码器层和解码器层，Transformer模型能够处理不同长度的输入序列，并且具有较好的并行性和可扩展性。这种模型结构已被证明在语言建模、机器翻译、文本生成等各种自然语言处理任务中取得了很好的效果。

### 核心优势

Transformer通过自注意力机制实现了：
- 完全并行计算
- 长距离依赖建模
- 可扩展架构设计

## 🎯 学习路径

### 基础阶段
1. [注意力机制基础](01-attention-mechanism.ipynb)
   - 注意力机制的动机和原理
   - Query、Key、Value概念
   - 简单注意力实现和可视化

2. [自注意力机制](02-self-attention.ipynb)
   - 自注意力vs传统注意力
   - 数学公式推导
   - 掩码注意力（因果掩码、填充掩码）

3. [多头注意力](03-multi-head-attention.ipynb)
   - 多头机制的必要性
   - 并行计算多种注意力模式
   - 注意力头的功能分析

### 核心阶段
4. [位置编码](04-positional-encoding.ipynb)
   - 位置编码的必要性
   - 正弦位置编码详解
   - 可学习位置编码对比
   - RoPE、ALiBi等现代变体

5. [Transformer基本块](05-transformer-block.ipynb)
   - 前馈神经网络
   - 残差连接和层归一化
   - Pre-LN vs Post-LN架构对比

### 进阶阶段
6. [完整Transformer实现](06-complete-transformer.ipynb)
   - 编码器-解码器架构
   - 机器翻译任务实战
   - 训练技巧和优化

7. [Transformer变体](07-transformer-variants.ipynb)
   - BERT、GPT系列模型
   - Vision Transformer
   - 效率优化变体

## 🛠️ 工具和资源

- [utils.py](utils.py) - 完整的工具函数库
  - 核心组件实现（注意力、位置编码等）
  - 可视化函数
  - 训练辅助工具
  - 模型管理函数



## 🚀 快速开始

```python
# 导入工具函数
from utils import *

# 创建多头注意力
attention = MultiHeadAttention(d_model=512, num_heads=8)

# 创建位置编码
pos_encoding = SinusoidalPositionalEncoding(d_model=512)

# 创建Transformer块
transformer_block = TransformerBlock(d_model=512, num_heads=8, d_ff=2048)

# 可视化注意力权重
visualize_attention_weights(attention_weights, tokens=['Hello', 'World'])
```



## 参考

* 10分钟学会Transformer | Pytorch代码讲解  https://zhuanlan.zhihu.com/p/403433120
* Pytorch中 nn.Transformer的使用详解与Transformer的黑盒讲解 https://blog.csdn.net/zhaohongfei_358/article/details/126019181
* 搞懂Transformer结构，看这篇PyTorch实现就够了（上）https://zhuanlan.zhihu.com/p/48731949
* [QKV详细解释](https://mp.weixin.qq.com/s/umEwoVOdTJxb6rAlNmuPQw)

### 视频、教程
中文综合教程：
* [白话transformer](https://www.bilibili.com/video/BV1bu4m137W1)
* [台大李宏毅自注意力机制和Transformer详解！](https://www.bilibili.com/video/BV1v3411r78R)
* [一步一步深入浅出解释Transformer原理](https://www.bilibili.com/video/BV14yQgY5EwJ)

注意力机制：
* [直观解释注意力机制，Transformer的核心](https://www.bilibili.com/video/BV1TZ421j7Ke)
* [Transformer模型的多头注意力机制](https://www.bilibili.com/video/BV12DzfYnErS)

英文教程：
* [Transformers, explained: Understand the model behind GPT, BERT, and T5](https://www.youtube.com/watch?v=SZorAJ4I-sA)
* [Transformer Neural Networks, ChatGPT's foundation, Clearly Explained!!!](https://www.youtube.com/watch?v=zxQyTK8quyY)
* [What are Transformers (Machine Learning Model)?](https://www.youtube.com/watch?v=ZXiruGOCn9s)
* [Illustrated Guide to Transformers Neural Network: A step by step explanation](https://www.youtube.com/watch?v=4Bdc55j80l8)
