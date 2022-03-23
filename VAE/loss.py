import torch
import numpy as np
from scipy.optimize import linear_sum_assignment
from torch import nn

def KLD_gauss(mu,logvar):
    """
    Compute Kullback-Leibler divergence KL(q(x)||p(x)) where p(x) is Gaussian, q(x) is Gaussian

    Args:
        mu (tensor): mean
        logvar (tensor): logarithm standard variation

    Returns:
        KLD: Return the KLD 
    """
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return KLD

def KLD_laplace(mu,logvar,scale=1.0):
    """ 
    Compute Kullback-Leibler divergence KL(q(x)||p(x)) where p(x) is Laplace, q(x) is Gaussian
    """
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
    """Compute mask from the mixed signal and the reconstructed signal

    Args:
        mu_s (tensor): original mix signal
        x (tensor): reconstructed signal (output of vae)

    Returns:
        tensor: returns the mask for the vaem
    """
    mu_sm = mu_s * (x / mu_s.sum(1).unsqueeze(1))
    mu_sm[torch.isnan(mu_sm)] = 0
    return mu_sm

def optimal_permute(y,x):
    """Optimal permutation based on Mean Squared Error

    Args:
        y (tensor): 
        x (tensor): 

    Returns:
        tensor
    """
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



def mix_data(data):
    n = data.size(0)//2
    sources = torch.cat([data[:n],data[n:2*n]],1) / 2.0
    data = sources.sum(1).unsqueeze(1)
    return data, sources
    