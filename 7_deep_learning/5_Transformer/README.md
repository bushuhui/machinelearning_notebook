# Transformer

Transformers是神经机器翻译中使用的一种神经网络，它主要涉及将输入序列转换为输出序列的任务。这些任务包括**语音识别**和**文本转换语音**。

Transformer中抛弃了传统的CNN和RNN，整个网络结构完全是由**Attention机制**组成。采用Attention机制的原因是考虑到RNN（或者LSTM，GRU等）的计算限制为是串行的，即RNN相关算法只能从左向右依次计算或者从右向左依次计算，这种机制带来了两个问题：

1. 时间片`t`的计算依赖`t−1`时刻的计算结果，这样限制了模型的**并行能力**。
2. 串行计算的过程中信息会丢失，尽管LSTM等门机制的结构一定程度上**缓解了长期依赖**的问题，但是对于特别长期的依赖现象，LSTM依旧无能为力。

Transformer的提出解决了上面两个问题，首先它使用了Attention机制，将序列中的任意两个位置之间的距离是缩小为一个常量；然后它使用的不是类似RNN的顺序结构，**具有更好的并行性，符合现有的GPU框架**。



## References

* 10分钟学会Transformer | Pytorch代码讲解  https://zhuanlan.zhihu.com/p/403433120
* Pytorch中 nn.Transformer的使用详解与Transformer的黑盒讲解 https://blog.csdn.net/zhaohongfei_358/article/details/126019181
* 搞懂Transformer结构，看这篇PyTorch实现就够了（上）https://zhuanlan.zhihu.com/p/48731949


### Videos
* [台大李宏毅自注意力机制和Transformer详解！](https://www.bilibili.com/video/BV1v3411r78R)
* [直观解释注意力机制，Transformer的核心](https://www.bilibili.com/video/BV1TZ421j7Ke)
* [Transformers, explained: Understand the model behind GPT, BERT, and T5](https://www.youtube.com/watch?v=SZorAJ4I-sA)
* [Transformer Neural Networks, ChatGPT's foundation, Clearly Explained!!!](https://www.youtube.com/watch?v=zxQyTK8quyY)
* [What are Transformers (Machine Learning Model)?](https://www.youtube.com/watch?v=ZXiruGOCn9s)
