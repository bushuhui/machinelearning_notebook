import os 
import numpy as np
import pickle

from utils.external import download

def unpickle(file_name):
	with open(file_name, "rb") as f:
		dict = pickle.load(f,encoding = "bytes")
	return dict

def import_data(data_dir):
	'''If CIFAR10 has not downloaded before, it needs to download and unzip.
	
	Arguments:
		data_dir {[string]} -- absolute path from config.data_dir to save the download CIFAR10.
	'''
	url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"

	download(url,data_dir)

def load_data(data_dir,data_type):
	'''function to load data from CIFAR10
	Arguments:
		data_dir {string} -- data directory (absolute path) that contains the CIFAR10 files
		data_type {[string]} -- "train" or "test" or "valid"

	Returns:
		data {ndarray (uint8)}: data from CIFAR10 related to train/test. 
								The data in NCHW format. 
								N = num_of_imgs(batch size), C=channels, H=height, W=width.
		label {ndarray (int)}: labels for each data from 0 to 9, integer.
	'''
	if data_type == "train":
		data = []
		label = []

		for _i in range(4):
			file_name = os.path.join(data_dir,"data_batch_{}".format(_i + 1))
			curr_dict = unpickle(file_name)
			# save the data and label
			data += [
				np.asarray(curr_dict[b"data"])
			]

			label += [
				np.asarray(curr_dict[b"labels"])
			]

		data = np.concatenate(data)
		label = np.concatenate(label)

	elif data_type =="valid":
		# only choose the data_batch_5 as the validation dataset, it is for simplity
		file_name = os.path.join(data_dir,"data_batch_5")
		curr_dict = unpickle(file_name)

		data = []
		label = []
		data = np.asarray(curr_dict[b"data"])
		label = np.asarray(curr_dict[b"labels"])

	elif data_type == "test":
		file_name = os.path.join(data_dir, "test_batch")
		curr_dict = unpickle(file_name)

		data = []
		label = []
		data = np.asarray(curr_dict[b"data"])
		label = np.asarray(curr_dict[b"labels"])

	else:
		raise ValueError("Wrong data type {}".format(data_type))

	# Turn data in (NxCxHxW) format, which is the pytorch dataloader format.
	# N = num_of_imgs(batch size), C=channels, H=height, W=width
	
	data = np.reshape(data, (-1,3,32,32))

	return data, label

if __name__ == '__main__':
	# for debug
	data_dir = "./data"
	if not os.path.exists(data_dir):
		os.makedirs(data_dir)
	import_data(data_dir)

	