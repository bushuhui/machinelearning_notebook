# Pyton技巧


## Python的包管理工具: `pip`
由于python是模块化的开发，因此能够能够利用其他人写的现成的包来快速的完成特定的任务。为了加快包的安装，python有很多包管理的工具，其中`pip`是目前使用最多的包管理工具。

* [pip的安装、使用等](pip.md)

但是由于直接使用pip去访问国外的网站慢，所以需要设置好pip的镜像，从而加快包的安装


## Python的虚拟环境： `virtualenv`
由于Python可以通过`pip`工具方便的安装包，因此极大的加快了程序编写的速度。但由于公开的包很多，不可避免的带来了包依赖导致的无法安装某些程序的问题。针对这个问题可以使用`docker`来构建一个隔离的环境来安装所需要的包，但有的时候还是希望在本机安装，因此需要使用`virtualenv`工具来安装虚拟的python环境。

* [virtualenv的安装、使用](virtualenv.md)
* [virtualenv便捷管理工具：virtualenv_wrapper](virtualenv_wrapper.md)