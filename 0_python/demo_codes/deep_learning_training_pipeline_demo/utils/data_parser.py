import torch
from torch.utils.data import Dataset
import numpy as np
import os

from utils.cifar10 import import_data,load_data

class CIFAR10Dataset(Dataset):
	"""This data wrapper is for CIFAR 10.
	Even though PyTorch already has written CIFAR10 in it,
	But I will implement it by myself to extract the data.
	"""

	def __init__(self,config,mode):
		'''Initialization.
		
		Arguments:
			config {[object]} -- Configuration object that holds the command line arguments.
			mode {[string} -- either 'train','test' or 'valid' to split for the dataset.
		Notes:
		----
		By default, we assume the dataset has downloaded. or it can extract the dataset from external.py to download and unzip the dataset.
		'''
		self.config = config

		if not os.path.exists(self.config.data_dir):
			data_dir = os.path.dirname(self.config.data_dir)
			import_data(data_dir)

		print("Loading CIFAR10 Dataset from {} for {} ing...".format(
			self.config.data_dir,mode),end ="")

		# load data
		data, label = load_data(self.config.data_dir, mode)
		self.data = data
		self.label = label
		self.sample_shp = self.data.shape[1:]

		print("done")

	def __len__(self):
		'''
		Return numbers of samples
		'''
		return len(self.data)

	def __getitem__(self,index):
		'''Function to grab one data sample 
		
		Arguments:
			index {[int]} -- sample index that is going to extract

		Returns:
			data_cur: torch.Tensor
				A torch tensor that holds the data. For GPU compabilitiy in the future,
				we will convert everything in 'float32'

			label_cur:int
				The label of the current data sample. from 0 to 9
		'''
		# grab one data from the dataset based on the index
		data_cur = self.data[index]
		# turn np.array into torch.tensor
		data_cur = torch.from_numpy(data_cur.astype(np.float32))
		# extract the label
		label_cur = self.label[index]

		return data_cur,label_cur


		