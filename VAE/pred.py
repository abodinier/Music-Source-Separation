
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
import numpy as np
# from .utils import *
import os
import shutil

import numpy as np
import librosa
import stempeg

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import soundfile as sf
from tqdm import tqdm

from IPython.display import Audio
import IPython
    
class LinearBlock(nn.Module):
    def __init__(self, in_channels,out_channels,activation=True):
        super(LinearBlock, self).__init__()
        if activation is True:
            self.block = nn.Sequential(
                nn.Linear(in_channels, out_channels),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(inplace=True),
                )
        else:
            self.block = nn.Sequential(
                nn.Linear(in_channels, out_channels),
                )

    def forward(self, x):
        return self.block(x)
import torch
import numpy as np
from scipy.optimize import linear_sum_assignment

def mix_data(data):
    n = data.size(0)//2
    sources = torch.cat([data[:n],data[n:2*n]],1) / 2.0
    data = sources.sum(1).unsqueeze(1)
    return data, sources
    
# KL(q(x)||p(x)) where p(x) is Gaussian, q(x) is Gaussian
def KLD_gauss(mu,logvar):
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return KLD

# KL(q(x)||p(x)) where p(x) is Laplace, q(x) is Gaussian
def KLD_laplace(mu,logvar,scale=1.0):
    v = logvar.exp()
    y = mu/torch.sqrt(2*v)
    y2 = y.pow(2)
    t1 = -2*torch.exp(-y2)*torch.sqrt(2*v/np.pi)
    t2 = -2*mu*torch.erf(y)
    t3 = scale*torch.log(np.pi*v/(2.0*scale*scale))
    temp = scale+t1+t2+t3
    KLD = -1.0/(2*scale)*torch.sum(1+t1+t2+t3)
    return KLD

def vae_masks(mu_s, x):
    mu_sm = mu_s * (x / mu_s.sum(1).unsqueeze(1))
    mu_sm[torch.isnan(mu_sm)] = 0
    return mu_sm

# Optimal permutation based on MSE, with the Hungarian Algorithm (Assignment Problem)
def optimal_permute(y,x):
    n = x.size(0)
    nx = x.size(1)
    ny = y.size(1)
    z = torch.zeros_like(x)
    for i in range(n):
        cost = torch.zeros(ny,nx)
        for j in range(ny):
            cost[j] = (y[i,j].unsqueeze(0) - x[i,:]).pow(2).sum(-1).sum(-1)

        row_ind, col_ind = linear_sum_assignment(cost.detach().numpy().T)
        z[i] = y[i,col_ind]
    return z


class VAE(nn.Module):
    def __init__(self, dimx, dimz, n_sources=1,device='cpu',variational=True):
        super(VAE, self).__init__()

        self.dimx = dimx
        self.dimz = dimz
        self.n_sources = n_sources
        self.device = device
        self.variational = variational

        chans = (2560, 248, 1536, 1024, 512)

        self.out_z = nn.Linear(chans[-1],2*self.n_sources*self.dimz)
        self.out_w = nn.Linear(chans[-1],self.n_sources)

        self.Encoder = nn.Sequential(
            LinearBlock(self.dimx, chans[0]),
            LinearBlock(chans[0],chans[1]),
            LinearBlock(chans[1],chans[2]),
            LinearBlock(chans[2],chans[3]),
            LinearBlock(chans[3],chans[4]),
            )

        self.Decoder = nn.Sequential(
            LinearBlock(self.dimz, chans[4]),
            LinearBlock(chans[4],chans[3]),
            LinearBlock(chans[3],chans[2]),
            LinearBlock(chans[2],chans[1]),
            LinearBlock(chans[1],chans[0]),
            LinearBlock(chans[0],self.dimx,activation=False),
            )


    def encode(self, x):
        d = self.Encoder(x)
        dz = self.out_z(d)
        mu = dz[:,::2]
        logvar = dz[:,1::2]
        return mu,logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        d = self.Decoder(z.view(-1,self.dimz))
        recon_separate = torch.sigmoid(d).view(-1,self.n_sources,self.dimx)
        recon_x = recon_separate.sum(1) 
        return recon_x, recon_separate

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, self.dimx))
        if self.variational is True:
            z = self.reparameterize(mu, logvar)
            recon_x, recons = self.decode(z)
        else:
            recon_x, recons = self.decode(mu)
        return recon_x, mu, logvar, recons

class LaplaceLoss(nn.Module):
    def __init__(self, variance=None,reduction='sum'):
        super(LaplaceLoss, self).__init__()
        if variance is None:
            variance = torch.tensor(1.0)
        self.register_buffer("log2",torch.tensor(2.0).log())
        self.register_buffer("scale",torch.sqrt(0.5*variance))
        self.logscale = self.scale.log()

    def forward(self, estimate, target):
        return torch.sum((target-estimate).abs() / self.scale)

class Loss(nn.Module):
    def __init__(self, sources=2, alpha=None, likelihood='bernoulli',variational=True,prior='gauss',scale=1.0):
        super(Loss, self).__init__()
        self.variational = variational
        self.prior = prior
        self.scale = scale

        if likelihood == 'gauss':
            self.criterion = nn.MSELoss(reduction='sum')
        elif likelihood == 'laplace':
            self.criterion = LaplaceLoss()
        else:
            self.criterion = nn.BCELoss(reduction='sum')

        if alpha is None:
            self.alpha_pior = nn.Parameter(torch.ones(1,sources),requires_grad=False)
        else:
            self.alpha_prior = nn.Parameter(alpha,requires_grad=False)

    def forward(self, x, recon_x, mu, logvar, beta=1):
        ELL = self.criterion(recon_x, x.view(-1,recon_x.size(-1)))

        KLD = 0.0
        if self.variational is True:

            if self.prior == 'laplace':
                KLD = KLD_laplace(mu,logvar,scale=self.scale)
            else:
                KLD = KLD_gauss(mu,logvar)
        
        loss = ELL + beta*KLD
        return loss, ELL,  KLD

import numpy as np
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
import torchvision
from torchvision import datasets, transforms
from torchvision.utils import save_image, make_grid
import matplotlib.pyplot as plt



def eval_audio(idx,filelist):

  audio_data , rate = stempeg.read_stems(filelist[idx])
  audio_mix = stereo_to_mono(audio_data[0])
  audio_drums = stereo_to_mono(audio_data[1])
  audio_bass = stereo_to_mono(audio_data[2])
  audio_other = stereo_to_mono(audio_data[3])
  audio_vocals = stereo_to_mono(audio_data[4])


  track = dict()
  track_stft = dict()

  track['mix'] = torch.tensor(audio_mix)
  track['drums'] = torch.tensor(audio_drums)
  track['bass'] = torch.tensor(audio_bass)
  track['other'] =torch.tensor(audio_other)
  track['vocals'] = torch.tensor(audio_vocals)

  track_stft['mix'] = stft(torch.tensor(audio_mix).float())
  track_stft['drums'] = stft(torch.tensor(audio_drums).float())
  track_stft['bass'] = stft(torch.tensor(audio_bass).float())
  track_stft['other'] = stft(torch.tensor(audio_other).float())
  track_stft['vocals'] = stft(torch.tensor(audio_vocals).float())

  print("Mix")
  IPython.display.display(IPython.display.Audio(track['mix'], rate=rate))
  print("drums")
  IPython.display.display(IPython.display.Audio(track['drums'], rate=rate))
  print("bass")
  IPython.display.display(IPython.display.Audio(track['bass'], rate=rate))
  print("other")
  IPython.display.display(IPython.display.Audio(track['other'], rate=rate))
  print("vocals")
  IPython.display.display(IPython.display.Audio(track['vocals'], rate=rate))

  return track, track_stft

def stereo_to_mono(audio):

  return (audio[:,0]+audio[:,1])/2

def stft(audio_data,n_freq=256, frame=128):

    stft_complex = torch.stft(audio_data, 2048, hop_length=512, win_length=512, window=torch.hann_window(512), center=True, pad_mode='reflect', normalized=True, onesided=None, return_complex=True)[:n_freq,...]
    mag, phase = stft_complex.abs(), stft_complex.angle()       
       
    return mag, phase

def separate_source(model, track):
  model.eval()
  s1 =  torch.tensor(track['drums'][0][:256,:128])
  s2 =  torch.tensor(track['vocals'][0][:256,:128])
  s3 =  torch.tensor(track['bass'][0][:256,:128])
  s4 =  torch.tensor(track['other'][0][:256,:128])

  s_tru = torch.stack([s1,s2,s3,s4]).to(device)
  s_tru = s_tru[None,...]
  model.eval()
  x_vae, mu_z, logvar_z, s_vae = model(track['mix'][0][:256,:128].float().to(device))
  x_vae = x_vae.view(-1,1,256,128)
  s_vae = s_vae.view(-1,4,256,128)

  s_vae = optimal_permute(s_vae,s_tru)

  # Create masks
  s_vaem = vae_masks(s_vae,track['mix'][0][:256,:128].to(device))
  x_vaem = s_vaem.sum(1).unsqueeze(1)

  #Reconstruct
  j = torch.complex(torch.tensor(0.),torch.tensor(1.))
  rate = 18000

  mix = x_vaem[0][0].to('cpu') * ( torch.cos(track['mix'][1][:256,:128]) + j* torch.sin(track['mix'][1][:256,:128]))
  mix = torch.istft(mix.detach(),511,normalized=True)
  
  s1 = s_vaem[0][0].to('cpu') * ( torch.cos(track['mix'][1][:256,:128]) + j* torch.sin(track['mix'][1][:256,:128]))
  s1 = torch.istft(s1.detach(),511,normalized=True)
  s2 = s_vaem[0][1].to('cpu') * ( torch.cos(track['mix'][1][:256,:128]) + j* torch.sin(track['mix'][1][:256,:128]))
  s2 = torch.istft(s2.detach(),511,normalized=True)
  s3 = s_vaem[0][2].to('cpu') * ( torch.cos(track['mix'][1][:256,:128]) + j* torch.sin(track['mix'][1][:256,:128]))
  s3 = torch.istft(s3.detach(),511,normalized=True)
  s4 = s_vaem[0][3].to('cpu') * ( torch.cos(track['mix'][1][:256,:128]) + j* torch.sin(track['mix'][1][:256,:128]))
  s4 = torch.istft(s4.detach(),511,normalized=True)
  sf.write('s1.wav', s1, rate)
  sf.write('s2.wav', s2, rate)
  sf.write('s3.wav', s3, rate)
  sf.write('s4.wav', s4, rate)
  sf.write('mix.wav', mix,rate)
  


  print("Mix")
  IPython.display.display(IPython.display.Audio(mix, rate=rate))
  print("drums")
  IPython.display.display(IPython.display.Audio(s1, rate=rate))
  print("bass")
  IPython.display.display(IPython.display.Audio(s2, rate=rate))
  print("other")
  IPython.display.display(IPython.display.Audio(s3, rate=rate))
  print("vocals")
  IPython.display.display(IPython.display.Audio(s4, rate=rate))

  return s_vaem, x_vaem, s_vae, x_vae


if __name__ == "__main__":
    print("Starting here")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dimx = int(256*128)

    model = VAE(dimx=dimx, dimz=20,n_sources=4, device="cpu",variational=True).to(device)
    optimizer = optim.Adam(model.parameters(), lr = .0002)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=.9998, last_epoch=-1)

    losses = {"train": torch.zeros(500,3), "test": torch.zeros(500,3)}
    print("Chargement du modele")
    checkpoint = torch.load('vae.pt')
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']

    loss = checkpoint['loss']

    filelist_train = []
    for filename in os.listdir("./musdb/train"):
        if filename.endswith("mp4"):
            filelist_train.append("./musdb/train/"+filename)

    filelist_test = []
    for filename in os.listdir("./musdb/test"):
        if filename.endswith("mp4"):
            filelist_test.append("./musdb/test/"+filename)

    print("evaluation")
    track, track_stft = eval_audio(2, filelist_train)

    s_vaem, x_vaem, s_vae, x_vae = separate_source(model,track_stft)


