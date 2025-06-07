import math
import torch
import torch.nn as nn

class LinearDynamicalSystem(nn.Module):
    def __init__(self, A, timestep):
        super(LinearDynamicalSystem, self).__init__()

        self.A = A + torch.eye(A.shape[0])
        self.timestep = timestep

        self.weight = nn.Parameter(torch.zeros(A.shape[0], A.shape[1]))
        torch.nn.init.xavier_uniform_(self.weight)
        self.bias = nn.Parameter(torch.zeros(A.shape[1]))
        stdv = 1. / math.sqrt(self.bias.size(0))
        self.bias.data.uniform_(-stdv, stdv)

        # index mask representing where A is not zero
        idx = torch.where(self.A == 0)
        # mask weights by adjacency matrix A, except for the diagonal
        self.weight.data[idx] = 0

    def forward(self, x):
        dxdt = (
            x @ (self.A * self.weight).T 
            + self.bias
        )
        x = x + dxdt * self.timestep
        # remove the bottom row
        x = x[:-1]
        return x

class LogisticDynamicalSystem(nn.Module):
    def __init__(self, A, timestep):
        super(LogisticDynamicalSystem, self).__init__()

        self.A = A + torch.eye(A.shape[0])
        self.timestep = timestep

        self.weight = nn.Parameter(torch.zeros(A.shape[0], A.shape[1]))
        # torch.nn.init.xavier_uniform(self.weight)
        self.caps = nn.Parameter(80 * torch.ones(A.shape[1]))
        # stdv = 1. / math.sqrt(self.caps.size(0))
        # self.caps.data.uniform_(-stdv, stdv)
        self.bias = nn.Parameter(torch.zeros(A.shape[1]))
        # stdv = 1. / math.sqrt(self.bias.size(0))
        # self.bias.data.uniform_(-stdv, stdv)

        # index mask representing where A is not zero
        idx = torch.where(self.A == 0)
        # mask weights by adjacency matrix A, except for the diagonal
        self.weight.data[idx] = 0

    def forward(self, x):
        dxdt = (
            (x * (self.caps-x)) @ (self.A * self.weight).T 
            + self.bias
        )
        x = x + dxdt
        # remove the bottom row
        x = x[:-1]
        return x
    
    # def forward(self, x):
    #     # x is T x N
    #     # self.caps is N x N
    #     # self.weight is N x N
    #     # self.bias is N
    #     y = self.caps - x
    #     N x N - N x T
    #     dxdt = self.weights @ x.T * (self.caps - x.T) + self.bias
    #     x = x + dxdt
    #     # remove the bottom row
    #     x = x[:-1]
    #     return x