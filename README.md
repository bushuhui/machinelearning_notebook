# 机器学习与人工智能

机器学习越来越多应用到飞行器、机器人等领域，其目的是利用计算机实现类似人类的智能，从而实现装备的智能化与无人化。本课程旨在引导学生掌握机器学习的基本知识、典型方法与技术，通过具体的应用案例激发学生对该学科的兴趣，鼓励学生能够从人工智能的角度来分析、解决飞行器、机器人所面临的问题和挑战。本课程主要内容包括Python编程基础，机器学习模型，无监督学习、监督学习、深度学习基础知识与实现，并学习如何利用机器学习解决实际问题，从而全面提升自我的[《综合能力》](Targets.md)。

由于**本课程需要大量的编程练习才能取得比较好的学习效果**，因此需要认真去完成[《机器学习与人工智能-作业和报告》](https://gitee.com/pi-lab/machinelearning_homework)，写作业的过程可以查阅网上的资料，但是不能直接照抄，需要自己独立思考并独立写出代码。本教程的Python等运行环境的安装说明请参考[《Python环境安装》](references_tips/InstallPython.md)。

为了让大家更好的自学本课程，课程讲座的视频在[《B站 - 机器学习与人工智能》](https://www.bilibili.com/video/BV1oZ4y1N7ei/)，欢迎大家观看学习。



![Machine Learning Cover](images/machine_learning_1.jpg)


## 1. 内容
1. [课程简介](CourseIntroduction.pdf)
2. [Python](0_python/README.md)
   - [Install Python](references_tips/InstallPython.md)
   - [ipython & notebook](0_python/0-ipython_notebook.ipynb)
   - [Python Basics](0_python/1_Basics.ipynb)
   - [Print Statement](0_python/2_Print_Statement.ipynb)
   - [Data Structure 1](0_python/3_Data_Structure_1.ipynb)
   - [Data Structure 2](0_python/4_Data_Structure_2.ipynb)
   - [Control Flow](0_python/5_Control_Flow.ipynb)
   - [Function](0_python/6_Function.ipynb)
   - [Class](0_python/7_Class.ipynb)
3. [numpy & matplotlib](1_numpy_matplotlib_scipy_sympy/README.md)
   - [numpy](1_numpy_matplotlib_scipy_sympy/1-numpy_tutorial.ipynb)
   - [matplotlib](1_numpy_matplotlib_scipy_sympy/2-matplotlib_tutorial.ipynb)
4. [kNN](2_knn/knn_classification.ipynb)
5. [kMeans](3_kmeans/1-k-means.ipynb)
   - [kMeans - Image Compression](3_kmeans/2-kmeans-color-vq.ipynb)
   - [Cluster Algorithms](3_kmeans/3-ClusteringAlgorithms.ipynb)
6. [Logistic Regression](4_logistic_regression/)
   - [Least squares](4_logistic_regression/1-Least_squares.ipynb)
   - [Logistic regression](4_logistic_regression/2-Logistic_regression.ipynb)
   - [PCA and Logistic regression](4_logistic_regression/3-PCA_and_Logistic_Regression.ipynb)
7. [Neural Network](5_nn/)
   - [Perceptron](5_nn/1-Perceptron.ipynb)
   - [Multi-layer Perceptron & BP](5_nn/2-mlp_bp.ipynb)
   - [Softmax & cross-entroy](5_nn/3-softmax_ce.ipynb)
8. [PyTorch](6_pytorch/README.md)
   - [Tensor](6_pytorch/1-tensor.ipynb)
   - [autograd](6_pytorch/2-autograd.ipynb)
   - [linear-regression](6_pytorch/3-linear-regression.ipynb)
   - [logistic-regression](6_pytorch/4-logistic-regression.ipynb)
   - [nn-sequential-module](6_pytorch/5-nn-sequential-module.ipynb)
   - [deep-nn](6_pytorch/6-deep-nn.ipynb)
   - [param_initialize](6_pytorch/7-param_initialize.ipynb)
   - [optim/sgd](6_pytorch/optimizer/6_1-sgd.ipynb)
   - [optim/adam](6_pytorch/optimizer/6_6-adam.ipynb)
9. [Deep Learning](7_deep_learning/README.md)
   - CNN
      - [CNN Introduction](7_deep_learning/1_CNN/CNN_Introduction.pptx)
      - [CNN simple demo](demo_code/3_CNN_MNIST.py)
      - [Basic of Conv](7_deep_learning/1_CNN/1-basic_conv.ipynb)
      - [VGG Network](7_deep_learning/1_CNN/2-vgg.ipynb)
      - [GoogleNet](7_deep_learning/1_CNN/3-googlenet.ipynb)
      - [ResNet](7_deep_learning/1_CNN/4-resnet.ipynb)
      - [DenseNet](7_deep_learning/1_CNN/5-densenet.ipynb)
      - [Batch Normalization](7_deep_learning/1_CNN/6-batch-normalization.ipynb)
      - [Learning Rate Decay](7_deep_learning/2_CNN/7-lr-decay.ipynb)
      - [Regularization](7_deep_learning/1_CNN/8-regularization.ipynb)
      - [Data Augumentation](7_deep_learning/1_CNN/9-data-augumentation.ipynb)
   - RNN
      - [rnn/pytorch-rnn](7_deep_learning/2_RNN/pytorch-rnn.ipynb)
      - [rnn/rnn-for-image](7_deep_learning/2_RNN/rnn-for-image.ipynb)
      - [rnn/lstm-time-series](7_deep_learning/2_RNN/time-series/lstm-time-series.ipynb)
   - GAN
      - [gan/autoencoder](7_deep_learning/3_GAN/autoencoder.ipynb)
      - [gan/vae](7_deep_learning/3_GAN/vae.ipynb)
      - [gan/gan](7_deep_learning/3_GAN/gan.ipynb)



## 2. 学习的建议
1. 为了更好的学习本课程，需要大家把[Python编程](0_python)能力培养好，通过一定数量的练习题、小项目培养Python编程思维，为后续的机器学习理论与实践打好坚实的基础。
2. 每个课程前半部分是理论基础，后半部分是代码实现。如果想学的更扎实，可以自己把各个方法的代码亲自实现一下。做的过程如果遇到问题尽可能自己想解决办法，因为最重要的目标不是代码本身，而是学会分析问题、解决问题的能力。
3. **不能直接抄已有的程序，或者抄别人的程序**，如果自己不会要自己去想，去找解决方法，或者去问。如果直接抄别人的代码，这样的练习一点意义都没有。**如果感觉太难，可以做的慢一些，但是坚持自己思考、自己编写练习代码**。。
4. **请先遍历一遍所有的文件夹，了解有什么内容，资料**。各个目录里有很多说明文档，如果不会先找找有没有文档，如果找不到合适的文档就去网上找找。通过这个过程锻炼自己搜索文献、资料的能力。
5. 本课程的练习题最好使用[《Linux》](https://gitee.com/pi-lab/learn_programming/blob/master/6_tools/linux)以及Linux下的工具来做。逼迫自己使用[《Linux》](https://gitee.com/pi-lab/learn_programming/blob/master/6_tools/linux)，只有多练、多用才能快速进步。如果实在太难，先在虚拟机（建议VirtualBox）里装一个Linux（例如Ubuntu，或者LinuxMint等），先熟悉一下。但是最终需要学会使用Linux。



## 3. [参考资料](References.md)
* [教程、代码](References.md)
* 资料速查
  * [相关学习参考资料汇总](References.md)
  * [一些速查手册](references_tips/cheatsheet)
* 机器学习方面技巧等
  * [Confusion Matrix](references_tips/confusion_matrix.ipynb)
  * [Datasets](references_tips/datasets.ipynb)
  * [构建深度神经网络的一些实战建议](references_tips/构建深度神经网络的一些实战建议.md)
  * [Intro to Deep Learning](references_tips/Intro_to_Deep_Learning.pdf)
* Python技巧等
  * [安装Python环境](references_tips/InstallPython.md)
  * [Python tips](references_tips/python)
* [Git教程](https://gitee.com/pi-lab/learn_programming/blob/master/6_tools/git/README.md)
* [Markdown教程](https://gitee.com/pi-lab/learn_programming/blob/master/6_tools/markdown/README.md)




## 4. 更进一步学习

在上述内容学习完成之后，可以进行更进一步机器学习、计算机视觉方面的学习与研究，具体的资料可以参考：
1. 编程是机器学习研究、实现过程非常重要的能力，编程能力弱则无法快速试错，导致学习、研究进度缓慢；如果编程能力强，则可以快速试错，快速编写实验代码等。强烈建议大家在学习本课程之后或之中，好好把数据结构、算法等基本功锻炼一下。具体的教程可以参考[《一步一步学编程》](https://gitee.com/pi-lab/learn_programming)
2. 飞行器智能感知与控制实验室-培训教程与作业：这个教程是实验室积累的机器学习与计算机视觉方面的教程集合，每个课程介绍基本的原理、编程实现、应用方法等资料，可以作为快速入门的学习材料。
    - [《飞行器智能感知与控制实验室-暑期培训教程》](https://gitee.com/pi-lab/SummerCamp)
    - [《飞行器智能感知与控制实验室-暑期培训作业》](https://gitee.com/pi-lab/SummerCampHomework)
3. 视觉SLAM是一类算法、技巧、编程高度集成的系统，通过学习、练习SLAM能够极大的提高自己的编程、解决问题能力。具体的教程可以参考[《一步一步学SLAM》](https://gitee.com/pi-lab/learn_slam)
3. [《编程代码参考、技巧集合》](https://gitee.com/pi-lab/code_cook)：可以在这个代码、技巧集合中找到某项功能的示例，从而加快自己代码的编写
5. [《学习方法论与技巧》](https://gitee.com/pi-lab/pilab_research_fields)
