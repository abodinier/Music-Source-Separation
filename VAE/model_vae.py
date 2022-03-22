
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
import numpy as np
# from .utils import *
    
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