from .base import Game
import torch
from torch import nn
import math
from ..sampler import RandomSampler, ShuffleSampler
from .. import utils


class NonMonotone2dGame(Game):  
    def __init__(self, num_samples, bias=True, scale=7):
        super(NonMonotone2dGame, self).__init__()
       
        self.num_samples = num_samples
        self.a = torch.zeros(num_samples).uniform_()
        self.b = torch.zeros(num_samples).uniform_()
        if not bias:
            self.a = self.a.zero_()
            self.b = self.b.zero_()
       
        self.matrix = torch.zeros(num_samples).uniform_()

        self.scale = scale
       
        self.reset()
       
    def reset(self):
        self.players = nn.ParameterList([nn.Parameter(torch.zeros(1).normal_()+6),
                                         nn.Parameter(torch.zeros(1).normal_()+6)])

    def F(self, x):
        mask_neg = (x <= -math.pi/2).float() 
        mask_pos = (x > math.pi/2).float()
        output = mask_neg*(-3*(x+math.pi/2)) + (1 - (mask_neg + mask_pos))*(-3*x.cos()) + mask_pos*(-x.cos() + 2*x - math.pi)
        return output.mean()

    def loss(self, index):
        loss = self.players[0]*self.matrix[index]*self.players[1] + self.a[index]*self.players[0] + self.b[index]*self.players[1]
        loss = self.F(self.players[0]) + self.scale*loss - self.F(self.players[1])
        return loss

    def forward(self, index):     
        loss_1 = self.loss(index)
        loss_2 = -loss_1
        return [loss_1, loss_2]
  
    def sample(self, n_samples=1):
        return torch.randint(self.num_samples, (n_samples,))
   
    def sample_batch(self, n_samples=None):
        if n_samples is not None:
            return torch.randperm(self.num_samples)[:n_samples]
        return torch.arange(self.num_samples)
   
    def compute_metrics(self):
        metrics = {}
        hamiltonian = self.compute_hamiltonian(torch.arange(self.num_samples)).item()
        metrics["grad_norm"] = hamiltonian
        return metrics


class NonMonotoneGame(Game):
    def __init__(self, dim, scale=4, bias=False):
        super(NonMonotoneGame, self).__init__()

        self.dim = dim
        self.scale = scale
        
        self.a = 1./math.sqrt(self.dim)*torch.zeros(self.dim, self.dim).normal_()
        self.b = 1./math.sqrt(self.dim)*torch.zeros(self.dim, self.dim).normal_()
        if not bias:
            self.a = self.a.zero_()
            self.b = self.b.zero_()

        self.matrix = torch.zeros(self.dim, self.dim, self.dim)
        for i in range(self.dim):
            self.matrix[i, i, i] = 1
        
        self.x_star = torch.zeros(dim)
        self.y_star = torch.zeros(dim)

        self.sampler = RandomSampler(self.dim)

        self.reset()

    def F(self, x):
        mask_neg = (x <= -math.pi/2).float() 
        mask_pos = (x > math.pi/2).float()
        output = mask_neg*(-3*(x+math.pi/2)) + (1 - (mask_neg + mask_pos))*(-3*x.cos()) + mask_pos*(-x.cos() + 2*x - math.pi)
        return output.mean()

    def shuffle(self):
        self.sampler = ShuffleSampler(self.dim)

    def reset(self):
        self.players = nn.ParameterList([nn.Parameter(torch.zeros(self.dim).normal_()*8),
                                         nn.Parameter(torch.zeros(self.dim).normal_()*8)])

    def loss(self, index):
        loss = (self.players[0].view(1,-1)*(self.matrix[index]*self.players[1].view(1, 1, -1)).sum(-1) 
                + self.a[index]*self.players[0].view(1, -1) + self.b[index]*self.players[1].view(1,-1)).sum(-1) 
        loss = self.F(self.players[0]) + self.scale*loss - self.F(self.players[1])
        return loss
    
    def forward(self, index):
        loss_1 = self.loss(index)
        loss_2 = -loss_1
        return (loss_1, loss_2)

    def sample(self, n_samples=1):
        return self.sampler.sample(n_samples)

    def sample_batch(self):
        return self.sampler.sample_batch()

    def compute_metrics(self):
        metrics = {}
        hamiltonian = self.compute_hamiltonian(torch.arange(self.dim)).item()
        metrics["grad_norm"] = hamiltonian
        metrics["distance_to_optimum"] = ((self.players[0]**2).sum() + (self.players[1]**2).sum()).item()
        return metrics


class SufficientlyBilinearGaussianMatrixGame(Game):
    def __init__(self, num_samples, dim, scale=4, bias=True):
        super(SufficientlyBilinearGaussianMatrixGame, self).__init__()

        self.dim = dim
        self.num_samples = num_samples
        self.a = (1./math.sqrt(self.dim) *
                  torch.zeros(num_samples, self.dim).normal_())
        self.b = (1./math.sqrt(self.dim) *
                  torch.zeros(num_samples, self.dim).normal_())
        if not bias:
            self.a = self.a.zero_()
            self.b = self.b.zero_()

        self.matrix = utils.make_spd_matrix(num_samples, dim)
        eigenvalues = torch.symeig(self.matrix.mean(0))[0]
        self.e_min = eigenvalues.min()
        self.e_max = eigenvalues.max()

        self.scale = scale

        self.coeff = math.floor(2*self.e_max/(scale*self.e_min**2))

        self.x_star = None
        self.y_star = None

        self.sampler = RandomSampler(num_samples)

        self.reset()

    def reset(self):
        self.players = nn.ParameterList([nn.Parameter(torch.zeros(self.dim).normal_()),
                                         nn.Parameter(torch.zeros(self.dim).normal_())])

    def shuffle(self):
        self.sampler = ShuffleSampler(self.num_samples)

    def F(self, x):
        mask_neg = (x <= -math.pi/2).float()
        mask_pos = (x > math.pi/2).float()
        output = mask_neg*(-3*(x+math.pi/2)) + (1 - (mask_neg + mask_pos))*(-3*x.cos()) + mask_pos*(-x.cos() + 2*x - math.pi)
        return self.coeff*output.mean()

    def loss(self, index):
        loss = (self.players[0].view(1,-1)*(self.matrix[index]*self.players[1].view(1, 1, -1)).sum(-1) 
                + self.a[index]*self.players[0].view(1, -1) + self.b[index]*self.players[1].view(1,-1)).sum(-1) 
        loss = self.F(self.players[0]) + self.scale*loss - self.F(self.players[1])
        return loss

    def forward(self, index):
        loss_1 = self.loss(index)
        loss_2 = -loss_1
        return (loss_1, loss_2)

    def distance_to_optimum(self):
        d = (((self.players[0])**2).sum() +
             ((self.players[1])**2).sum())
        return d

    def sample(self, n_samples=1):
        return self.sampler.sample(n_samples)

    def sample_batch(self):
        return self.sampler.sample_batch()

    def compute_metrics(self):
        metrics = {}
        hamiltonian = self.compute_hamiltonian(torch.arange(self.num_samples)).item()
        metrics["grad_norm"] = hamiltonian
        metrics["distance_to_optimum"] = self.distance_to_optimum().item()
        return metrics
