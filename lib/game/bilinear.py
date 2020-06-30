from .base import Game
import math
import torch
from torch import nn
from ..sampler import RandomSampler, ShuffleSampler
from numpy import linalg
from .. import utils


class BilinearGame(Game):
    def __init__(self, dim, bias=True):
        super(BilinearGame, self).__init__()

        self.dim = dim
        self.a = (1./math.sqrt(self.dim) *
                  torch.zeros(self.dim, self.dim).normal_())
        self.b = (1./math.sqrt(self.dim) *
                  torch.zeros(self.dim, self.dim).normal_())
        if not bias:
            self.a = self.a.zero_()
            self.b = self.b.zero_()

        self.matrix = torch.zeros(self.dim, self.dim, self.dim)
        for i in range(self.dim):
            self.matrix[i, i, i] = 1

        self.x_star = None
        self.y_star = None

        self.sampler = RandomSampler(self.dim)

        self.L = 1
        self.mu = 1/dim**2

        self.reset()

    def reset(self):
        self.players = nn.ParameterList([nn.Parameter(1./math.sqrt(self.dim) *
                                         torch.zeros(self.dim).normal_()),
                                         nn.Parameter(1./math.sqrt(self.dim) *
                                         torch.zeros(self.dim).normal_())])

    def shuffle(self):
        self.sampler = ShuffleSampler(self.dim)

    def forward(self, index):
        loss_1 = (self.players[0].view(1, -1)*(self.matrix[index] *
                  self.players[1].view(1, 1, -1)).sum(-1) +
                  self.a[index] * self.players[0].view(1, -1) +
                  self.b[index] * self.players[1].view(1, -1)).sum(-1)
        loss_2 = -loss_1
        return [loss_1, loss_2]

    def solve(self):
        x = -self.b.sum(0)
        y = -self.a.sum(0)
        return x, y

    def distance_to_optimum(self):
        if self.x_star is None or self.y_star is None:
            self.x_star, self.y_star = self.solve()
        d = (((self.players[0] - self.x_star)**2).sum() +
             ((self.players[1] - self.y_star)**2).sum())
        return d

    def sample(self, n_samples=1):
        return self.sampler.sample(n_samples)

    def sample_batch(self):
        return self.sampler.sample_batch()

    def compute_metrics(self):
        metrics = {}
        hamiltonian = self.compute_hamiltonian(torch.arange(self.dim)).item()
        metrics["grad_norm"] = hamiltonian
        metrics["distance_to_optimum"] = self.distance_to_optimum().item()
        return metrics


class BilinearGaussianMatrixGame(Game):
    def __init__(self, num_samples, dim, bias=True, symmetric=True):
        super(BilinearGaussianMatrixGame, self).__init__()

        self.dim = dim
        self.num_samples = num_samples
        self.a = (1./math.sqrt(self.dim) *
                  torch.zeros(num_samples, self.dim).normal_())
        self.b = (1./math.sqrt(self.dim) *
                  torch.zeros(num_samples, self.dim).normal_())
        if not bias:
            self.a = self.a.zero_()
            self.b = self.b.zero_()

        if symmetric is True:
            self.matrix = (1./math.sqrt(self.dim)) * torch.zeros(num_samples, self.dim, self.dim).normal_()
            self.matrix = 0.5 * (self.matrix + self.matrix.transpose(1, 2))
            self.L = torch.symeig(self.matrix)[0].max()
            eigenvalues = torch.symeig(self.matrix.mean(0))[0]
        elif symmetric == "spd":
            self.matrix = utils.make_spd_matrix(num_samples, self.dim)
            self.L = torch.symeig(self.matrix)[0].max()
            eigenvalues = torch.symeig(self.matrix.mean(0))[0]
        elif symmetric is False:
            self.matrix = (1./math.sqrt(self.dim)) * torch.zeros(num_samples, self.dim, self.dim).normal_()
            self.L = None
            eigenvalues = torch.eig(self.matrix.mean(0))[0]
        else:
            raise ValueError()
        mask = eigenvalues > 0
        self.mu = eigenvalues[mask].min()

        self.x_star = None
        self.y_star = None

        self.sampler = RandomSampler(num_samples)

        self.reset()

    def reset(self):
        self.players = nn.ParameterList([nn.Parameter(torch.zeros(self.dim).normal_()),
                                         nn.Parameter(torch.zeros(self.dim).normal_())])

    def shuffle(self):
        self.sampler = ShuffleSampler(self.num_samples)

    def forward(self, index):
        loss_1 = (self.players[0].view(1, -1)*(self.matrix[index] *
                  self.players[1].view(1, 1, -1)).sum(-1) +
                  self.a[index] * self.players[0].view(1, -1) +
                  self.b[index] * self.players[1].view(1, -1)).sum(-1)
        loss_2 = -loss_1
        return [loss_1, loss_2]

    def solve(self):
        x = linalg.solve(self.matrix.mean(0).T, -self.b.mean(0))
        y = linalg.solve(self.matrix.mean(0), -self.a.mean(0))
        x = torch.tensor(x)
        y = torch.tensor(y)
        return x, y

    def distance_to_optimum(self):
        if self.x_star is None or self.y_star is None:
            self.x_star, self.y_star = self.solve()
        d = (((self.players[0] - self.x_star)**2).sum() +
             ((self.players[1] - self.y_star)**2).sum())
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


class Bilinear2dGame(Game):
    def __init__(self, num_samples, bias=True):
        super(Bilinear2dGame, self).__init__()

        self.num_samples = num_samples
        self.a = torch.zeros(num_samples).uniform_()
        self.b = torch.zeros(num_samples).uniform_()
        if not bias:
            self.a = self.a.zero_()
            self.b = self.b.zero_()

        self.matrix = torch.zeros(num_samples).uniform_()

        self.x_star = None
        self.y_star = None

        self.reset()

        self.L = self.matrix.max()
        self.mu = self.matrix.mean()

    def reset(self):
        self.players = nn.ParameterList([nn.Parameter(torch.zeros(1).normal_()),
                                         nn.Parameter(torch.zeros(1).normal_())])

    def forward(self, index):
        loss_1 = (self.players[0]*self.matrix[index]*self.players[1] +
                  self.a[index]*self.players[0] +
                  self.b[index]*self.players[1])
        loss_2 = -loss_1
        return [loss_1, loss_2]

    def solve(self):
        x = -self.b.sum()/self.matrix.sum()
        y = -self.a.sum()/self.matrix.sum()
        return x, y

    def distance_to_optimum(self):
        if self.x_star is None or self.y_star is None:
            self.x_star, self.y_star = self.solve()
        d = (((self.players[0] - self.x_star)**2).sum() + 
             ((self.players[1] - self.y_star)**2).sum())
        return d

    def sample(self, n_samples=1):
        return torch.randint(self.num_samples, (n_samples,))

    def sample_batch(self, n_samples=None):
        if n_samples is not None:
            return torch.randperm(self.num_samples)[:n_samples]
        return torch.arange(self.num_samples)

    def compute_metrics(self):
        metrics = {}
        hamiltonian = self.compute_hamiltonian(
                      torch.arange(self.num_samples)).item()
        metrics["grad_norm"] = hamiltonian
        metrics["distance_to_optimum"] = self.distance_to_optimum().item()
        return metrics


class BilinearGame_v2(Game):
    def __init__(self, dim):
        super(BilinearGame_v2, self).__init__()

        self.dim = dim
        self.a = 1./math.sqrt(self.dim)*torch.zeros(self.dim, self.dim).normal_()
        self.b = 1./math.sqrt(self.dim)*torch.zeros(self.dim, self.dim).normal_()
       
        self.matrix = torch.zeros(self.dim, self.dim, self.dim)
        for i in range(self.dim):
            self.matrix[i, i, i] = 1
        
        self.x_star = None
        self.y_star = None
       
        self.reset()    
       
    def compute_grad(self, index):
        grad_0 = ((self.matrix[index]*self.players[1].view(1, 1, -1)).sum(-1) 
                  + self.a[index])
        grad_1 = -((self.matrix[index]*self.players[0].view(1, -1, 1)).sum(1) 
                   + self.b[index])
        return (grad_0, grad_1)

    def hamiltonian_update(self, index1, index2=None):        
        grad_1 = self.compute_grad(index1)
        if index2 is None:
            update_0 = -self.matrix[index1].mean(0).mv(grad_1[1].mean(0))
            update_1 = self.matrix[index1].mean(0).mv(grad_1[0].mean(0))
        else:
            if len(index2) > 1:
                raise NotImplementedError("if index2 is not None, then index1 and index2 should have length 1")
            grad_2 = self.compute_grad(index2)
            update_0 = -0.5*((self.matrix[index1]*(grad_2[1].unsqueeze(1))).sum(-1) + (self.matrix[index2]*(grad_1[1].unsqueeze(1))).sum(-1))
            update_1 = 0.5*((self.matrix[index1].transpose(-1, -2)*(grad_2[0].unsqueeze(-1))).sum(1) +
                            (self.matrix[index2].transpose(-1, -2)*(grad_1[0].unsqueeze(-1))).sum(1))          
        return (update_0, update_1)

    def reset(self):
        self.players = nn.ParameterList([nn.Parameter(1./math.sqrt(self.dim)*torch.zeros(self.dim).normal_()),
                                         nn.Parameter(1./math.sqrt(self.dim)*torch.zeros(self.dim).normal_())])
             
    def forward(self, index):     
        loss_1 = (self.players[0].view(1,-1)*(self.matrix[index]*self.players[1].view(1, 1, -1)).sum(-1) + self.a[index]*self.players[0].view(1, -1) +
                  self.b[index]*self.players[1].view(1, -1)).sum(-1)
        loss_2 = -loss_1
        return [loss_1, loss_2]
   
    def solve(self):
        x = linalg.solve(self.matrix.mean(0).T, -self.b.mean(0))
        y = linalg.solve(self.matrix.mean(0), -self.a.mean(0))
    
        # players = copy.deepcopy(self.players)
        # self.players[0].data = torch.Tensor(x)
        # self.players[1].data = torch.Tensor(y)
        # self.hamiltonian_star = self.compute_hamiltonian(torch.arange(self.dim)).item()
        # self.players = players

        return torch.tensor(x), torch.tensor(y)
  
    def distance_to_optimum(self):
        if self.x_star is None or self.y_star is None:
            self.x_star, self.y_star = self.solve()
        d = ((self.players[0] - self.x_star)**2).sum() + ((self.players[1] - self.y_star)**2).sum()
        return d
   
    def sample(self, n_samples=1):
        return torch.randint(self.dim, (n_samples,))
   
    def sample_batch(self, n_samples=None):
        if n_samples is not None:
            return torch.randperm(self.dim)[:n_samples]
        return torch.arange(self.dim)
   
    def compute_metrics(self):
        metrics = {}
        hamiltonian = self.compute_hamiltonian(torch.arange(self.dim)).item()
        metrics["grad_norm"] = hamiltonian
        metrics["distance_to_optimum"] = self.distance_to_optimum().item()
        return metrics


class BilinearGaussianMatrixGame_v2(Game):
    def __init__(self, num_samples, dim):
        super(BilinearGaussianMatrixGame_v2, self).__init__()
        # How do we make the problem stochastic ?

        X = torch.randn(num_samples, dim)
        u, s, v = torch.svd(X)
        s = 1 / torch.arange(1, min(num_samples, dim) + 1)
        s = s / (s**2).sum().sqrt() * math.sqrt(2 * num_samples)
        self.matrix = u.mm(torch.diag(s)).mm(v.t())

        self.a = torch.zeros(dim)

        x_pred = torch.randn(dim)
        noise = torch.randn(num_samples)
        self.b = (self.matrix.mv(x_pred) + 0.1 * noise * linalg.norm(self.matrix.mv(x_pred), "fro") / torch.norm(noise, "fro")).sign()

        self.dim = dim
        self.num_samples = num_samples
          
        self.x_star = None
        self.y_star = None
        
        self.reset()    
        
    def reset(self):
        self.players = nn.ParameterList([nn.Parameter(1./math.sqrt(self.dim)*torch.zeros(self.dim).normal_()),
                                         nn.Parameter(1./math.sqrt(self.num_samples)*torch.zeros(self.num_samples).normal_())])
             
    def forward(self, index):     
        loss_1 = (self.players[0].view(1, -1)*(self.matrix[index]*self.players[1].view(1, 1, -1)).sum(-1) + self.a[index]*self.players[0].view(1, -1) + 
                  self.b[index]*self.players[1].view(1, -1)).sum(-1)
        loss_2 = -loss_1
        return [loss_1, loss_2]
    
    def solve(self):
        x = linalg.solve(self.matrix.mean(0).T, -self.b.mean(0))
        y = linalg.solve(self.matrix.mean(0), -self.a.mean(0))
        return torch.tensor(x), torch.tensor(y)
    
    def distance_to_optimum(self):
        if self.x_star is None or self.y_star is None:
            self.x_star, self.y_star = self.solve()
        d = ((self.players[0] - self.x_star)**2).sum().sqrt() + ((self.players[1]-self.y_star)**2).sum().sqrt()
        return d
    
    def sample(self):
        return torch.randint(self.dim, (1,))
    
    def sample_batch(self):
        return torch.arange(self.dim)
    
    def compute_metrics(self):
        metrics = {}
        metrics["grad_norm"] = self.compute_hamiltonian(torch.arange(self.dim)).item()
        metrics["distance_to_optimum"] = self.distance_to_optimum().item()
        return metrics