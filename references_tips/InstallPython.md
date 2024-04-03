# 安装Python环境

由于Python的库比较多，并且依赖关系比较复杂，所以请仔细阅读下面的说明，并按下面的说明来操作，减少问题出现的可能。 **但是所列的安装方法说明里有较多的细节，也许和你的系统并不适配，所以可能会遇到问题。如果遇到问题请通过搜索引擎去查找解决的办法**，并通过这个方式锻炼自己解决问题的能力。

可以参考后面所列的`1.Winodws`或者`2.Linux`章节所列的将Python环境安装到计算机里。



## 1. Windows下安装

由于Anaconda集成了大部分的python包，因此能够很方便的开始使用。由于网络下载速度较慢，因此推荐使用镜像来提高下载的速度。镜像的使用方法可以参考：[Anaconda镜像的说明文档](https://mirrors.bfsu.edu.cn/help/anaconda/)

1. 在下列镜像网站找到适合自己的安装文件，然后下载
* https://mirrors.bfsu.edu.cn/anaconda/archive/
* https://mirrors.tuna.tsinghua.edu.cn/anaconda/archive/
* https://mirrors.aliyun.com/anaconda/archive/
* https://mirrors.hit.edu.cn/anaconda/archive/

例如： https://mirrors.bfsu.edu.cn/anaconda/archive/Anaconda3-2020.11-Windows-x86_64.exe

2. 按照说明，把Anaconda安装好。



## 2. Linux下安装
在网站下载最新的conda安装文件，例如

```
wget https://mirrors.bfsu.edu.cn/anaconda/archive/Anaconda3-2020.11-Linux-x86_64.sh
```

然后运行
```
bash ./Anaconda3-2020.11-Linux-x86_64.sh
```

按照提示完成安装（记得需要`自动加入环境变量`的设置），**然后关闭终端，再打开终端**



## 3. 设置软件源

### 3.1 设置conda软件源 

参考这里的[conda安装和软件源设置说明](https://mirrors.bfsu.edu.cn/help/anaconda/)


各系统都可以通过修改用户目录下的 `.condarc` 文件。

Windows 用户无法直接创建名为 `.condarc` 的文件，可先执行 `conda config --set show_channel_urls yes` 生成该文件之后再修改。然后在命令行输入 `notepad .condarc`将下面的内容拷贝到文本编辑器里面。

Linux下，打开文件编辑器 `gedit ~/.condarc`，然后把下面的内容拷贝到这个文件中：
```
channels:
  - defaults
show_channel_urls: true
default_channels:
  - https://mirrors.bfsu.edu.cn/anaconda/pkgs/main
  - https://mirrors.bfsu.edu.cn/anaconda/pkgs/r
  - https://mirrors.bfsu.edu.cn/anaconda/pkgs/msys2
custom_channels:
  conda-forge: https://mirrors.bfsu.edu.cn/anaconda/cloud
  msys2: https://mirrors.bfsu.edu.cn/anaconda/cloud
  bioconda: https://mirrors.bfsu.edu.cn/anaconda/cloud
  menpo: https://mirrors.bfsu.edu.cn/anaconda/cloud
  pytorch: https://mirrors.bfsu.edu.cn/anaconda/cloud
  pytorch-lts: https://mirrors.bfsu.edu.cn/anaconda/cloud
  simpleitk: https://mirrors.bfsu.edu.cn/anaconda/cloud
```


### 3.2 设置PIP源

```
pip config set global.index-url https://pypi.mirrors.ustc.edu.cn/simple/
```



## 4. 安装常用软件

新建conda环境
```
conda create -n machinelearning python=3.9
```

打开`conda`的命令行程序，输入下面的命令
```
conda install jupyter scipy numpy sympy matplotlib pandas scikit-learn
```



## 5. 安装PyTorch

GPU 版本
```
# 访问 https://pytorch.org/，查最新的安装命令
# 例如 pytorch-cuda=11.6

# 安装cudatoolkit
conda install cudatoolkit 

# 安装最新版本
conda install pytorch torchvision torchaudio pytorch-cuda -c pytorch -c nvidia

# 安装特定版本
#conda install pytorch torchvision torchaudio pytorch-cuda=11.6 -c pytorch -c nvidia
```

CPU 版本
```
conda install pytorch -c pytorch 
pip3 install torchvision
```



检测GPU是否在PyTorch中可用：

```
>>> import torch
>>> torch.cuda.is_available()
```




## 6. Conda使用技巧

### 6.1 Conda创建自己的环境
```
conda create -n <your_env> python=x.x

# example
conda create -n machinelearning python=3.8
```

上面的`python=x.x`中的`x.x`对应自己系统中的Python版本

### 6.2 Conda怎么激活自己的环境
```
conda activate <your_env>

# example 
conda activate machinelearning
```

### 6.3 Conda常用命令
```
# 帮助命令
conda -h
conda help

# 配置频道(已有)
conda config --add channels https://mirrors.bfsu.edu.cn/anaconda/pkgs/main/

# 退出当前环境
conda deactivate

# 克隆环境
conda create -n BBB --clone AAA

# 查看基本信息
conda info
conda info -h

# 查看当前存在环境
conda env list
conda info --envs

# 删除环境
conda remove -n <your_env> --all
```

## 7. 安装nvidia驱动

### 7.1 查看已有的nvidia驱动
```
dpkg -l | grep -i nvidia
```

### 7.2 卸载驱动
```
sudo apt-get purge nvidia-driver-xxx
```

### 7.3 搜索并安装的驱动

```
apt-cache search nvidia | grep 460
sudo apt-get install nvidia-driverp -460
```

根据自己的需要可以安装更高的版本。

#### 7.4 Conda使用cuda
```
conda install cudatoolkit=8.0 -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/linux-64/
```
**根据自己的需要安装更高的版本**

## 8. pip使用技巧
指定给定的源来安装，可以在pip后面加上 `--extra-index-url https://pypi.mirrors.ustc.edu.cn/simple/`，例如：

```
sudo pip3 install conan==1.61.0 --extra-index-url https://pypi.mirrors.ustc.edu.cn/simple/
```


## 9. [Python技巧](python/)

- [pip的安装、使用等](python/pip.md)
- [virtualenv的安装、使用](python/virtualenv.md)
- [virtualenv便捷管理工具：virtualenv_wrapper](python/virtualenv_wrapper.md)
