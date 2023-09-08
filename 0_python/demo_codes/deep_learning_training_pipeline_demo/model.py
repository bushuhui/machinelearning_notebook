from torch import nn

class ConvBlock(nn.Module):
	""" Convolution block"""

	def __init__(self,indim,outdim,ksize = 3,stride = 1, activation = nn.ReLU):
		""" Initialization of the custom conv2d """
		# Run initialization for super class
		super(ConvBlock,self).__init__()

		# check ksize(kernel size) and stride requirements
		assert (ksize % 2) == 1
		assert stride == 1
		assert indim == outdim 

		# store activation function depending on configuration
		self.activ = activation

		#compute padding, make sure that will not change the image width and height
		padding = ksize // 2

		# create ResNet Block
		# The architecture of the ResNet Block is 1x1 conv (padding = 0)-> 3x3 conv -> 1x1 conv (padding = 0)
		
		self.layers = nn.Sequential()
		self.layers.add_module("conv_1", self._conv(indim,indim,1, 1, 0))
		self.layers.add_module("conv_2", self._conv(indim,indim, ksize, 1, padding))
		self.layers.add_module("conv_3", self._conv(indim,outdim, 1, 1, 0))

	def _conv(self,indim, outdim, ksize, stride, padding):
		""" Functions to make conv layers easier to create
		
		Return a nn.Sequential object which has bn-conv-activation
		"""

		return nn.Sequential(
			nn.BatchNorm2d(indim),
			nn.Conv2d(indim, outdim, ksize, stride, padding),
			self.activ()
			)

	def forward(self,x):
		"""Forward pass our block. 
		
		Implement the Resnet here. One path go through our "layers" while another 
		path remain intact. Then they should add together.
		"""

		assert (len(x.shape) == 4)

		x_out = self.layers(x) + x

		return x_out


class MyNetWork(nn.Module):
	"""Network class"""
	def __init__(self,config,input_shp):
		"""Initialization of the model
		
		Arguments:
			config {[object]} -- configuration object that holds the command line argument.
			input_shp {[list or tuple]} -- shape of each input data sample.
		"""

		# Run initialization for super class
		super(MyNetWork,self).__init__()

		#store configuration
		self.config = config

		# extract the indim of the data sample
		indim = input_shp[0]

		# Retrive Conv, Act, Pool functions from configuration
		if self.config.conv2d == "torch":
			self.Conv2d = nn.Conv2d
		elif self.config.conv2d == "custom":
			self.Conv2d = ConvBlock

		self.Activation = getattr(nn,self.config.activation)
		self.Pool2d = getattr(nn, self.config.pool2d)
		self.Linear = nn.Linear

		# Implement ResNet Block.
		self.convs = nn.Sequential()
		# save the feature map width and height
		cur_h,cur_w = input_shp[-2:]

		for _i in range(self.config.num_conv_outer):
			# Simply create a pure liner convolution layer.
			outdim = self.config.nchannel_base * 2** _i
			self.convs.add_module(
				"conv_{}_base".format(_i),nn.Conv2d(indim, outdim, 1, 1, 0))
			indim = outdim

			for _j in range(self.config.num_conv_inner):
				if self.config.conv2d == "torch":
					self.convs.add_module(
						"conv_{}_{}".format(_i,_j),
						self.Conv2d(indim, outdim,self.config.ksize,1, 1))
					self.convs.add_module(
						"act_{}_{}".format(_i,_j),
						self.Activation())
					# print the current feathreu map size
					cur_h = cur_h - (self.config.ksize - 1)
					cur_w = cur_w - (self.config.ksize - 1)
				elif self.config.conv2d == "custom":
					self.convs.add_module(
						"conv_{}_{}".format(_i,_j),
						self.Conv2d(indim, outdim, self.config.ksize, 1,self.Activation))
			self.convs.add_module(
				"conv_{}_pool".format(_i),self.Pool2d(2,2))

			cur_h = cur_h // 2
			cur_w = cur_w // 2

		# final output
		self.output = nn.Linear(indim, self.config.num_class)

		print(self)

	def forward(self,x):
		""" Forward pass for the model
		
		Arguments:
			x {[torch.tensor]} -- input data in the BCHW format.

		Returns:
           x torch.tensor
		"""

		# roughly make input to be within -1 to 1 range
		x = (x - 128.) / 128.

		# apply conv layers
		x = self.convs(x)

		# global average pooling
		x = x.mean(-1).mean(-1)

		# output layer
		x = self.output(x)

		return x 

