from preprocess import *
from loss import *
from model_vae import *
import os
import torch
import torch.utils.data

import numpy as np
import torch
import torch.utils.data
from torch import optim
import matplotlib.pyplot as plt


def train(epoch, model, train_loader, device, dimx, optimizer, loss_function,beta):
	"""
	Train the Vae model

	Args:
		epoch (int): Number of epochs
		model : VAE model
		train_loader (Dataloader): Dataloader made from the MusicDataset class.
		device (str): cpu or cuda
		dimx (int): size of the input STFT (i.e. 256*128 most of the time)
		optimizer (optim): optimizer
		loss_function (Loss): loss function
		beta (int): beta for the loss function 

	Returns:
		array: contains the train loss
	"""
	model.train()
	train_losses = torch.zeros(3)

	for batch_idx, data in enumerate(train_loader):
		data = data['mix']

		data = data.to(device).view(-1,dimx)
		optimizer.zero_grad()
		recon_y, mu_z, logvar_z, _ = model(data)
		loss, ELL, KLD = loss_function(data,recon_y, mu_z, logvar_z, beta=beta)
		loss.backward()
		optimizer.step()

		train_losses[0] += loss.item()
		train_losses[1] += ELL.item()
		train_losses[2] += KLD.item()

		if batch_idx % 10 == 0:
			print('Train Epoch: {} [{}/{} ({:.0f}%)] \t -ELL: {:5.6f} \t KLD: {:5.6f} \t Loss: {:5.6f}'.format(
				epoch, batch_idx * len(data), len(train_loader.dataset),
				100. * batch_idx / len(train_loader),
				ELL.item() / len(data),KLD.item() / len(data),loss.item() / len(data)))

	train_losses /= len(train_loader.dataset)
	print('====> Epoch: {} Average loss: {:.4f}'.format(
		  epoch, train_losses[0]))

	return train_losses


def test(model, test_loader, device, dimx, loss_function,beta):
	"""Test the VAE model

	Args:
		model : VAE model
		test_loader (DataLoader): DataLoader from the MusicDataset class
		device (str): cpu or cuda
		dimx (int): input size of the STFT (i.e. 256*128 most of the time)
		loss_function (Loss): loss functino
		beta (int): beta for the loss function

	Returns:
		array: returns test loss
	"""
	model.eval()
	test_losses = torch.zeros(3)

	with torch.no_grad():
		for i, data in enumerate(test_loader):
			data = data['mix']
			data = data.to(device).view(-1,dimx)

			recon_y, mu_z, logvar_z, recons = model(data)
			loss, ELL, KLD = loss_function(data,recon_y, mu_z, logvar_z, beta=beta)

			test_losses[0] += loss.item()
			test_losses[1] += ELL.item()
			test_losses[2] += KLD.item()

	return test_losses

def plot_losses(losses):
	"""Use matplotlib to plot the train and test loss

	Args:
		losses (array): Contains the loss
	"""
	plt.figure()
	plt.plot(np.array(range(1,len(losses["train"][:,0]))),losses["train"][:,0].view(-1),label="Train")
	plt.plot(np.array(range(1,len(losses["train"][:,0]))),losses["test"][:,0].view(-1),label="Test")
	plt.xlabel('Epoch'), plt.ylabel('Loss'), plt.legend(), plt.xlim(1,10)
	plt.close()





def launch_model(batch_size = 4,n_freq=256, timeframe=128, dimz=20,n_sources=4, lr=.0002, n_epochs=5,save=False, plot=False, load=False):
	"""
	Load the files, creates the dataloader, initialize the VAE model, train and test it.

	Args:
		batch_size (int, optional): Size of the batch. Defaults to 4.
		n_freq (int, optional): Number of frequency to keep in the STFT. Defaults to 256.
		timeframe (int, optional): Size of the window of STFT to keep. Defaults to 128.
		dimz (int, optional): latent space dimension . Defaults to 20.
		n_sources (int, optional): Number of source to separate. Defaults to 4.
		lr (float, optional): learning rate. Defaults to .0002.
		n_epochs (int, optional): Number of epochs to train. Defaults to 5.
		save (bool, optional): To save the model at the end of the training. Defaults to False.
		plot (bool, optional): To plot the train and test loss. Defaults to False.
		load (bool, optional): To load a pretrained model. Defaults to False.
	"""
	print("Initialization of the Variational autoencoder")
	labels=['drums', 'bass', 'other', 'vocals']
	print("Starting to build the dataset.")
	try:
		filelist_train = []
		for filename in os.listdir("./musdb/train"):
			if filename.endswith("mp4"):
				filelist_train.append("./musdb/train/"+filename)

		filelist_test = []
		for filename in os.listdir("./musdb/test"):
			if filename.endswith("mp4"):
				filelist_test.append("./musdb/test/"+filename)
	except:
		print("Musdb18 dataset is not present.")
		return
	train_set = MusicDataset(filelist_train)
	test_set = MusicDataset(filelist_test)

	print("Dataset built.")
	print("Train set size : ",len(train_set))
	print("Test set size: ",len(test_set))
	
	

	train_loader = torch.utils.data.DataLoader(train_set,batch_size=batch_size, shuffle=True)
	test_loader = torch.utils.data.DataLoader(test_set,batch_size=batch_size, shuffle=True)


	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	dimx = int(n_freq * timeframe)
	model = VAE(dimx=dimx, dimz=dimz,n_sources=n_sources, device=device,variational=True).to(device)
	loss_function = Loss(sources=n_sources, likelihood='laplace',variational=True, prior="gauss",scale=0.5)

	optimizer = optim.Adam(model.parameters(), lr = lr)
	scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=.9998, last_epoch=-1)

	losses = {"train": torch.zeros(n_epochs,3), "test": torch.zeros(n_epochs,3)}

	for epoch in range(1, n_epochs+1):
		beta = min(1.0,(epoch)/min(5000,50)) * 0.5

		losses["train"][epoch-1] = train(epoch, model, train_loader, device, dimx, optimizer, loss_function,beta)
		losses["test"][epoch-1] = test(model, test_loader, device, dimx, loss_function,beta)

		if optimizer.param_groups[0]['lr'] >= 1.0e-5:
			scheduler.step()

		

	if plot:
		plot_losses(losses)
	if save:
		torch.save({
			'epoch': n_epochs,
			'model_state_dict': model.state_dict(),
			'optimizer_state_dict': optimizer.state_dict(),
			'loss': losses,
			}, 'vae.pt')