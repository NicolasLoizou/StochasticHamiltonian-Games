from torch import autograd
from torch import nn


def compute_grad(loss, players):
    grad = []
    for l, p in zip(loss, players):
        grad += list(autograd.grad(l.mean(), p, retain_graph=True,
                                   create_graph=True))
    return grad


def compute_hamiltonian(grad1, grad2=None):
    if grad2 is None:
        grad2 = grad1
    hamiltonian = 0
    for g1, g2 in zip(grad1, grad2):
        hamiltonian += (g1*g2).sum()
    hamiltonian /= 2

    return hamiltonian


class Game(nn.Module):
    def __init__(self):
        super(Game, self).__init__()
        self.players = []

    def forward(self, x):
        raise NotImplementedError()

    def compute_grad(self, x):
        loss = self.forward(x)
        grad = compute_grad(loss, self.players)
        return grad

    def compute_hamiltonian(self, x1, x2=None):
        loss1 = self.forward(x1)
        grad1 = compute_grad(loss1, self.players)

        if x2 is not None:
            loss2 = self.forward(x2)
            grad2 = compute_grad(loss2, self.players)
        else:
            grad2 = None

        hamiltonian = compute_hamiltonian(grad1, grad2)
        return hamiltonian

    def sample(self):
        raise NotImplementedError()

    def compute_metrics(self):
        raise NotImplementedError()
