import torch.nn as nn
import torch.nn.functional as F
import torch
import os
from torch import optim
from preprocess import *

class SRNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.up = nn.UpsamplingBilinear2d(size=(512,128))
        self.conv1 = nn.Conv2d(2, 6, 5,padding='same')
        # self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 3,padding='same')
        # self.fc1 = nn.Linear(16 * 5 * 5, 120)
        # self.fc2 = nn.Linear(120, 84)
        # self.fc3 = nn.Linear(84, 10)
        self.conv3 = nn.Conv2d(16, 2, 3,padding='same')

    def forward(self, x):
        x = self.up(x)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        return x


def train(model, train_loader,optimizer, criterion):
    print("Start training.")
    for epoch in range(50):  # loop over the dataset multiple times

        loss_tot = 0.0
        for batch_idx, data in enumerate(train_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            # if i % 2000 == 1999:    # print every 2000 mini-batches
        #   print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
        print('Train Epoch: {} [{}/{} ({:.0f}%)] \t -ELL: {:5.6f} \t KLD: {:5.6f} \t Loss: {:5.6f}'.format(
				epoch, batch_idx * len(data), len(train_loader.dataset),
				100. * batch_idx / len(train_loader),
				loss.item() / len(data)))
        loss_tot = 0.0

    print('Training finished')

def launch_model(batch_size=4, save=False):
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
    train(srnet, train_loader,optimizer, criterion)
    if save:
        