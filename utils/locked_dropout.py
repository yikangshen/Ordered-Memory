import torch
import torch.nn as nn
from torch.autograd import Variable

class LockedDropout(nn.Module):
    def __init__(self, dropout=0.5, dim=0):
        super().__init__()

        assert dim in [0, 1]
        self.dim = dim
        self.dropout = dropout

    def forward(self, x):
        assert len(x.size()) == 3
        if not self.training or not self.dropout:
            return x
        if self.dim == 0:
            m = x.data.new(1, x.size(1), x.size(2)).bernoulli_(1 - self.dropout)
        elif self.dim == 1:
            m = x.data.new(x.size(0), 1, x.size(2)).bernoulli_(1 - self.dropout)
        mask = Variable(m, requires_grad=False) / (1 - self.dropout)
        mask = mask.expand_as(x)
        return mask * x
