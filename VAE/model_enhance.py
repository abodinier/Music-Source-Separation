import torch.nn as nn
import torch.nn.functional as F
import torch
import os
from torch import optim
from preprocess import *

class SRNet(nn.Module):
    def __init__(self, n_freq=512, timeframe=128):
        """Super Resolution model for Audio enhancement

        Args:
            n_freq (int, optional): Number of frequency in the output spectogram. Defaults to 512.
            timeframe (int, optional): Size of the window. Defaults to 128.
        """
        super().__init__()
        self.up = nn.UpsamplingBilinear2d(size=(n_freq,timeframe))
        self.conv1 = nn.Conv2d(2, 6, 5,padding='same')
        self.conv2 = nn.Conv2d(6, 16, 3,padding='same')
        self.conv3 = nn.Conv2d(16, 2, 3,padding='same')

    def forward(self, x):
        x = self.up(x)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        return x


def train(model, train_loader,optimizer, criterion, n_epochs):
    """Train a model on the train_loader, with the optimizer and the criterion, for n_epochs

    Args:
        model : SRNet model
        train_loader (Dataloader): Dataloader of [X,y] where X correspond to the stft with half the frequency of y
        optimizer (optim): optimizer
        criterion (Loss): Loss used to train
        n_epochs (int): Number of epochs to train

    Returns:
        loss: list of the loss, used to plot the train loss
    """
    print("Start training.")
    for epoch in range(n_epochs): 
        loss_tot = 0.0
        for batch_idx, data in enumerate(train_loader, 0):
            inputs, labels = data
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print('Train Epoch: {} [{}/{} ({:.0f}%)] \t Loss: {:5.6f}'.format(
				epoch, batch_idx * len(data), len(train_loader.dataset),
				100. * batch_idx / len(train_loader),
				loss.item() / len(data)))

    print('Training finished')
    return loss_tot

def launch_model(batch_size=4, save=False, n_epochs=50):
    """
    Launch this model from the main file

    Args:
        batch_size (int, optional):  Defaults to 4.
        save (bool, optional):  Defaults to False.
        n_epochs (int, optional):  Defaults to 50.
    """
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

    data_list_train = []
    data_list_test = []

    for filepath in filelist_train:
        S, rate = stempeg.read_stems(filepath,dtype=np.float32)
        X = stft(torch.tensor(stereo_to_mono(S[0])),n_freq=256)
        Y = stft(torch.tensor(stereo_to_mono(S[0])),n_freq=512)
        X = torch.cat([X[0][None],X[1][None]],axis=0)
        Y = torch.cat([Y[0][None],Y[1][None]],axis=0)
        data_list_train.append([X,Y])
    for filepath in filelist_test:
        S, rate = stempeg.read_stems(filepath,dtype=np.float32)
        X = stft(torch.tensor(stereo_to_mono(S[0])),n_freq=256)
        Y = stft(torch.tensor(stereo_to_mono(S[0])),n_freq=512)
        X = torch.cat([X[0][None],X[1][None]],axis=0)
        Y = torch.cat([Y[0][None],Y[1][None]],axis=0)
        data_list_test.append([X,Y])


    train_loader = torch.utils.data.DataLoader(data_list_train,batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(data_list_test,batch_size=batch_size, shuffle=True)

    srnet = SRNet()

    criterion = nn.MSELoss()
    optimizer = optim.SGD(srnet.parameters(), lr=0.0001, momentum=0.9)
    losses = train(srnet, train_loader,optimizer, criterion)
    if save:
        torch.save({
			'epoch': n_epochs,
			'model_state_dict': srnet.state_dict(),
			'optimizer_state_dict': optimizer.state_dict(),
			'loss': losses,
			}, 'vae.pt')