# 安装Python环境

由于Python的库比较多，并且依赖关系比较复杂，所以请仔细阅读下面的说明，并按下面的说明来操作，减少问题出现的可能。 **但是所列的安装方法说明里有较多的细节，也许和你的系统并不适配，所以可能会遇到问题。如果遇到问题请通过搜索引擎去查找解决的办法**，并通过这个方式锻炼自己解决问题的能力。

可以参考后面所列的`1.Winodws`或者`2.Linux`章节所列的将Python环境安装到计算机里。



## 1. Windows下安装

由于Anaconda集成了大部分的python包，因此能够很方便的开始使用。由于网络下载速度较慢，因此推荐使用镜像来提高下载的速度。镜像的使用方法可以参考：[Anaconda镜像的说明文档](https://mirrors.bfsu.edu.cn/help/anaconda/)

1. 在下列镜像网站找到适合自己的安装文件，然后下载
* https://mirrors.bfsu.edu.cn/anaconda/archive/
* https://mirrors.tuna.tsinghua.edu.cn/anaconda/archive/
* https://mirrors.ustc.edu.cn/anaconda/archive/

例如： https://mirrors.ustc.edu.cn/anaconda/archive/Anaconda3-2024.06-1-Windows-x86_64.exe

2. 按照说明，把Anaconda安装好。



## 2. Linux下安装
在网站下载最新的conda安装文件，例如

```bash
wget https://mirrors.ustc.edu.cn/anaconda/archive/Anaconda3-2024.06-1-Linux-x86_64.sh
```

然后运行
```bash
bash ./Anaconda3-2024.06-1-Linux-x86_64.sh
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

```bash
pip config set global.index-url https://pypi.mirrors.ustc.edu.cn/simple/
```

指定给定的源来安装，可以在pip后面加上 `--extra-index-url https://pypi.mirrors.ustc.edu.cn/simple/`，例如：

```bash
sudo pip3 install conan==1.61.0 --extra-index-url https://pypi.mirrors.ustc.edu.cn/simple/
```

## 4. 安装常用软件

新建conda环境
```bash
conda create -n machinelearning python=3.9
conda activate machinelearning
```
其中 `machinelearning` 是新建的conda环境的名字

打开`conda`的命令行程序，输入下面的命令
```bash
conda install jupyter scipy numpy sympy matplotlib pandas scikit-learn
```


## 5. 安装PyTorch

GPU 版本
```bash
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
```bash
conda install pytorch -c pytorch 
pip3 install torchvision
```



检测GPU是否在PyTorch中可用：

```bash
>>> import torch
>>> torch.cuda.is_available()
```




## 6. Conda使用技巧

### 6.1 Conda创建自己的环境
```bash
conda create -n <your_env> python=x.x

# example
conda create -n machinelearning python=3.8
```

上面的`python=x.x`中的`x.x`对应自己系统中的Python版本

### 6.2 Conda怎么激活自己的环境
```bash
conda activate <your_env>

# example 
conda activate machinelearning
```

### 6.3 Conda常用命令
```bash
# 帮助命令
conda -h
conda help

# 配置频道(已有)
conda config --add channels https://mirrors.bfsu.edu.cn/anaconda/pkgs/main/

# 退出当前环境
conda deactivate

# 克隆环境
conda create -n BBB --clone AAA

# 删除一个环境
conda env remove --name envname

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
```bash
dpkg -l | grep -i nvidia
```

### 7.2 卸载驱动
```bash
sudo apt-get purge nvidia-driver-xxx
```

### 7.3 搜索并安装的驱动

```bash
apt-cache search nvidia | grep 570
sudo apt-get install nvidia-driver-570 nvidia-utils-570
```

根据自己的需要可以安装更高的版本。

#### 7.4 Conda使用cuda
```bash
conda install cudatoolkit=8.0 -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/linux-64/
```
**根据自己的需要安装更高的版本**



## 8. uv

在 Python 开发中，包管理和环境隔离是每个开发者都会遇到的问题。无论是 pip 的缓慢、virtualenv 的繁琐，还是 conda 的臃肿，uv 都让开发者们期待一个更高效的解决方案。

uv 是由 Astral 公司开发的一款 Rust 编写的 Python 包管理器和环境管理器，它的主要目标是提供比现有工具快 10-100 倍的性能，同时保持简单直观的用户体验。

uv 可以替代 pip、virtualenv、pip-tools 等工具，提供依赖管理、虚拟环境创建、Python 版本管理等一站式服务。

参考资料：
* [Python 包管理工具 uv 使用教程](https://zhuanlan.zhihu.com/p/1888904532131575259)
* [uv 入门教程 -- Python 包与环境管理工具](https://www.runoob.com/python3/uv-tutorial.html)
* [一招配置uv国内镜像](https://zhuanlan.zhihu.com/p/1930714592423703026)


### 8.1. 安装uv
使用pip安装
```bash
pip install uv
```


### 8.2. 设置源
Linux下设置临时的源：
```bash
# 阿里源示例
export UV_DEFAULT_INDEX=https://mirrors.aliyun.com/pypi/simple/
uv pip install -U numpy
```

修改 `~/.config/uv/uv.toml`，镜像一直起作用
```bash
mkdir -p ~/.config/uv
cat >> ~/.config/uv/uv.toml <<'EOF'
[[index]]
url = "https://mirrors.aliyun.com/pypi/simple/"
default = true
EOF
```

设置Python安装文件的镜像
```bash
export UV_PYTHON_INSTALL_MIRROR="https://gh-proxy.com/github.com/indygreg/python-build-standalone/releases/download"
uv python install 3.13.2
```

### 8.3. 管理Python版本
```bash
# 列出所有的Python版本
uv python list

# 安装特点版本的Python
uv python install 3.13.2
```

### 8.4. 管理虚拟环境

#### 创建并激活虚拟环境：

```bash
# 创建名为 .venv 的虚拟环境（默认）
uv venv

# 激活环境（macOS/Linux）
source .venv/bin/activate

# 激活环境（Windows）
.venv\Scripts\activate

# 强制安装基础包（如pip, setuptools, wheel）
uv venv --seed

# 在创建venv的时候指定Python版本
uv venv --python 3.13.2
```

在项目中指定 Python 版本：
```bash
# 为当前项目固定 Python 3.11，但是venv还是创建设定的Python版本
uv python pin 3.11
```

这会创建 `.python-version` 文件，标识项目所需的 Python 版本。


### 8.5. 包管理
安装包：
```bash
# 安装最新版本
uv pip install requests

# 安装特定版本
uv pip install requests==2.31.0

# 从 requirements.txt 安装
uv pip install -r requirements.txt
```

安装包到开发环境：
```bash
uv pip install --dev pytest
```


升级包：
```bash
uv pip upgrade requests
```

卸载包：
```bash
uv pip uninstall requests
```

导出依赖：
```bash
# 导出当前环境的依赖
uv pip freeze > requirements.txt

# 导出生产环境依赖（排除开发依赖）
uv pip freeze --production > requirements.txt
```


### 8.6. 项目管理
uv 支持 pyproject.toml 格式的项目管理，这是现代 Python 项目的标准配置文件。

初始化一个新项目：
```bash
uv init my_project
cd my_project
```

这会创建基本的项目结构和 pyproject.toml 文件。

安装项目的依赖：
```bash
uv sync
```

这个命令会根据 `pyproject.toml` 和 `requirements.txt` 安装所有依赖，类似于 `pip install -e .` 但更高效。

