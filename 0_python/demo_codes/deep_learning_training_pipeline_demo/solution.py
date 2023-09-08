import os

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import get_config, print_usage
from model import MyNetWork
from tensorboardX import SummaryWriter
from utils.data_parser import CIFAR10Dataset

def data_criterion(config):
	""" Returns the loss object based on the commandline argument for the data term"""
	if config.loss_type =="cross_entropy":
		data_loss = nn.CrossEntropyLoss()
	elif config.loss_type == "svm":
		data_loss = nn.MultiMarginLoss()


	return data_loss

def model_criterion(config):
	""" loss function based on the commandline argument for the regularizer term"""

	def model_loss(model):
		loss = 0
		for name, param in model.named_parameters():
			if "weight" in name:
				loss += torch.sum(param**2)

		return loss * config.l2_reg

	return model_loss

def train(config):
	#initialize datasets for both training and validation
	train_data = CIFAR10Dataset(
		config, mode = "train"
		)

	valid_data = CIFAR10Dataset(
		config, mode = "valid",
		)

	# create dataloader for training and validation
	tr_data_lodaer = DataLoader(
		dataset = train_data,
		batch_size = config.batch_size,
		num_workers = 2,
		shuffle = True)

	val_data_loader = DataLoader(
		dataset = valid_data,
		batch_size = config.batch_size,
		num_workers = 2,
		shuffle = False)

	# create model instance.
	model = MyNetWork(
		config = config, input_shp = train_data.sample_shp)
	# move model to gpu if cuda is available
	if torch.cuda.is_available():
		model = model.cuda()

	# make sure that the model is set for training 
	model.train()

	# create loss objects
	data_loss = data_criterion(config)
	model_loss = model_criterion(config)

	# create optimizer 
	optimizer = optim.Adam(model.parameters(), lr = config.learning_rate)

	# create summary writer
	tr_writer = SummaryWriter(
		log_dir = os.path.join(config.log_dir, "train"))
	va_writer = SummaryWriter(
		log_dir = os.path.join(config.log_dir, "valid"))

	# create log directory and save logdirectory if it does not exist
	if not os.path.exists(config.log_dir):
		os.makedirs(config.log_dir)
	if not os.path.exists(config.save_dir):
		os.makedirs(config.save_dir)

	# initialize training 
	iter_idx = -1 # make counter start at 0
	best_va_acc = 0 # to check if best validation accuracy

	# prepare checkpoint file and model file to save 
	checkpoint_file = os.path.join(config.save_dir,"checkpoint.pth")
	bestmodel_file = os.path.join(config.save_dir,"best_model.pth")

	# check for existing training results. If it exists, and the configuration is 
	# set to resume "config.resume == True", resume from the previous training.
	if os.path.exists(checkpoint_file):
		if config.resume:
			print("Checkpoint found! Resuming")
			# read checkpoint file
			load_res = torch.load(
				checkpoint_file,
				map_location = "cpu")
			# resume iterations
			iter_idx = load_res["iter_idx"]
			# best va result
			best_va_acc = load_res['best_va_acc']

			# resume the model
			model.load_state_dict(load_res["model"])
			# resume optimizer
			optimizer.load_state_dict(load_res["optimizer"])

		else:
			os.remove(checkpoint_file)

	# Training loop
	for epoch in range(config.num_epoch):
		# learning_rate schedule
		if epoch < 50:
			optimizer = optim.Adam(model.parameters(),lr = config.learning_rate)
		elif 50 < epoch < 80:
			lr = config.learning_rate * 0.1
			optimizer = optim.Adam(model.parameters(), lr = lr)
		else:
			lr = config.learning_rate * 0.01
			optimizer = optim.Adam(model.parameters(), lr = lr)
		# For each iteration
		prefix = "Training Epoch {:3d}".format(epoch)

		for data in tqdm(tr_data_lodaer, desc = prefix):
			# counter 
			iter_idx += 1
			# split the data 
			x, y = data
			# send data to gpu if it were available
			if torch.cuda.is_available():
				x = x.cuda()
				y = y.cuda()

			# apply the model to obtain scores(forward pass)
			logits = model.forward(x)
			# compute the loss
			loss = data_loss(logits,y) + model_loss(model)
			# compute gradients
			loss.backward()
			# update params
			optimizer.step()
			# zero the param. gradients in the optimizer
			optimizer.zero_grad()

			# monitor results every report interval
			if iter_idx % config.rep_intv == 0:
				# compute acc no grads required. we'll wrap this part so that 
				# we prevent torch from computing gradients.
				with torch.no_grad():
					pred = torch.argmax(logits,dim = 1)
					acc = torch.mean(torch.eq(pred,y).float()) * 100.0

					# write loss and acc to tensorboard, using loss and acc
					tr_writer.add_scalar("loss", loss, global_step = iter_idx)
					tr_writer.add_scalar("accuracy", acc, global_step = iter_idx)

					# save 
					torch.save({
						"iter_idx": iter_idx,
						"best_va_acc": best_va_acc,
						"model": model.state_dict(),
						"optimizer": optimizer.state_dict()
						}, checkpoint_file)

			# validation resluts every validation interval:
			if iter_idx % config.val_intv == 0:
				# list to contain all losses and accuracies for all 
				# the training batch
				va_loss =[]
				va_acc = []

				# set model for evaluation
				model = model.eval()
				for data in val_data_loader:
					# split the data
					x,y = data
					# send data to GPU if it were available
					if torch.cuda.is_available():
						x = x.cuda()
						y = y.cuda()
					# apply forward pass to compute the loss and accuracy 
					with torch.no_grad():
						# compute logits
						logits = model.forward(x)
						# compute loss and store as numpy
						loss = data_loss(logits,y) + model_loss(model)
						va_loss += [loss.cpu().numpy()]
						# compute acc and store as numpy 
						pred = torch.argmax(logits,dim = 1)
						acc = torch.mean(torch.eq(pred,y).float()) * 100.0
						va_acc += [acc.cpu().numpy()]

				# set model back for training 
				model = model.train()
				# Take average 
				va_loss = np.mean(va_loss)
				va_acc = np.mean(va_acc)

				# writer to tensorboard
				va_writer.add_scalar("loss", va_loss, global_step = iter_idx)
				va_writer.add_scalar("accuracy", va_acc, global_step = iter_idx)


				# check if best accuarcy
				if va_acc > best_va_acc:
					best_va_acc = va_acc
					torch.save({
						"iter_idx": iter_idx,
						"best_va_acc": best_va_acc,
						"model": model.state_dict(),
						"optimizer": optimizer.state_dict()
						}, bestmodel_file)
def test(config):
	""" test routine"""

	test_data = CIFAR10Dataset(
		config, mode = "test")
	# create data loader for test dataset
	te_data_loader = DataLoader(
		dataset = test_data,
		batch_size = config.batch_size,
		num_workers = 2,
		shuffle = False
		)
	# create model
	model = MyNetWork(
		config, input_shp = test_data.sample_shp)

	# move to GPU if it were available
	if torch.cuda.is_available():
		model = model.cuda()

	# create loss objects
	data_loss = data_criterion(config)
	model_loss = model_criterion(config)

	# load our best model and set model for testing 
	load_res = torch.load(
		os.path.join(config.save_dir,"best_model.pth"),
		map_location = "cpu")
	model.load_state_dict(load_res["model"])
	model.eval()

	# implement test loop
	prefix = "Testing"
	te_loss = []
	te_acc = []

	for data in tqdm(te_data_loader,desc = prefix):
		# split the data 
		x, y = data 
		# send to GPU if it were available
		if torch.cuda.is_available():
			x = x.cuda()
			y = y.cuda()
		with torch.no_grad():
			# compute the scores
			logits = model.forward(x)
			# compute loss and save it to numpy
			loss = data_loss(logits,y) + model_loss(model)
			te_loss += [loss.cpu().numpy()]
			# compute the test_acc and save it to numpy
			pred = torch.argmax(logits, dim = 1)
			acc = torch.mean(torch.eq(pred,y).float())* 100.0
			te_acc += [acc.cpu().numpy()]

	# report test loss and acc
	print("Test loss = {}".format(np.mean(te_loss)))
	print("Test accuracy = {}".format(np.mean(te_acc)))

def main(config):
    """The main function."""

    if config.mode == "train":
        train(config)
    elif config.mode == "test":
        test(config)
    else:
        raise ValueError("Unknown run mode \"{}\"".format(config.mode))


if __name__ == "__main__":

    # ----------------------------------------
    # Parse configuration
    config, unparsed = get_config()
    # If we have unparsed arguments, print usage and exit
    if len(unparsed) > 0:
        print_usage()
        exit(1)

    main(config)