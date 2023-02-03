import imp
import torch.nn as nn
import torch.nn.init as init
from layers.HornerConv import HornerConv

import torch as th
import torch.nn.functional as F

import math

class HornerNN(nn.Module):
    def __init__(self,
                 edge_index,
                 norm_A, 
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 act_fn,
                 last_act_fn,
                 dropout,
                 dropout2,
                 theta,
                 lamda,
                 theta_decay,
                 init_by_ones=False
                 ):
        super(HornerNN, self).__init__()
        self.edge_index = edge_index
        self.norm_A = norm_A
        
        self.convs = nn.ModuleList()
        for _ in range(n_layers+1):
            self.convs.append(HornerConv(n_hidden, n_hidden, n_layers, theta=theta, lamda=lamda, theta_decay=theta_decay, weight=True, bias=True))
        self.n_layers = n_layers
        self.fcs = nn.ModuleList()
        self.fcs.append(nn.Linear(in_feats, n_hidden))
        self.fcs.append(nn.Linear(n_hidden, n_classes))
        self.params1 = list(self.convs.parameters())
        self.params2 = list(self.fcs.parameters())
        self.act_fn = act_fn
        self.last_act_fn = last_act_fn
        
        self.dropout = nn.Dropout(p=dropout)
        self.dropout2 = nn.Dropout(p=dropout2)
        self.init_alphas(init_by_ones)

    
    def init_alphas(self, init_by_ones):
        if init_by_ones:
            t = th.ones(self.n_layers+1)
        else:
            t = th.zeros(self.n_layers+1)
            t[0] = 1
        # t = th.arange(self.n_layers+1)
        self.alpha_params = nn.Parameter(t.float()) 

    def __relu(self, x):
        x[x<-0.]=0
        return                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           x

    def forward(self, features):
        x = features

        x = self.dropout(x)
        x = self.fcs[0](x)
        x = self.act_fn(x)

        x = self.dropout(x)
        h0 =  x
        last_h = th.zeros_like(h0)

        for i, con in enumerate(self.convs):
            alpha = self.alpha_params[-(i+1)]
            x = con(self.edge_index, self.norm_A, h0, last_h, alpha, i)
            if i < self.n_layers - 1:
                x = self.__relu(x)
                x = self.dropout2(x)
            last_h = x

        x = self.__relu(x)
        x = self.dropout(x)
        x = self.fcs[-1](x)

        x = F.log_softmax(x, dim=1)
        return x