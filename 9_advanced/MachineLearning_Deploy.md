# 机器学习的部署

开发完机器学习模型，并得到最优模型参数之后，其实并未结束。很多情况下，需要将开发的模型部署到嵌入式设备上，从而在终端上能够快速、实时运行。



## TensorRT

TensorRT能够加快所设计的机器学习模型在CUDA设备上的推理速度。

* [Torch-TensorRT](https://pytorch.org/TensorRT/)
* [Accelerating Inference Up to 6x Faster in PyTorch with Torch-TensorRT](https://developer.nvidia.com/blog/accelerating-inference-up-to-6x-faster-in-pytorch-with-torch-tensorrt/)

## C++

将PyTorch所开发的模型转化成TorchScript，然后通过LibTorch加载，这样能够在C++项目中使用。

* [LOADING A TORCHSCRIPT MODEL IN C++](https://pytorch.org/tutorials/advanced/cpp_export.html)
* [https://pytorch.org/tutorials/advanced/super_resolution_with_onnxruntime.html](https://pytorch.org/tutorials/advanced/super_resolution_with_onnxruntime.html)

## Intra-Process Message Passsing

如果不想把PyTorch开发的转化到C++下运行，可以将所设计的程序独立成两个以上的进程，机器学习的模型使用PyTorch运行，其他语言编写的程序也独立运行。两个程序使用进程间通信来实现调用，可以使用ROS，或者ZeroMQ

* [ROS](http://wiki.ros.org/ROS/Tutorials/CreatingMsgAndSrv)
* [ZeroMQ](https://zeromq.org/)