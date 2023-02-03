import imp
import torch.nn as nn
import torch.nn.init as init
from layers.ChebClenshawConv import ChebConv

import torch as th
import torch.nn.functional as F

import math

class ChebNNFix(nn.Module):
    def __init__(self,
                 kind,
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
                 alpha,
                 theta_decay,
                 ):
        super(ChebNNFix, self).__init__()
        self.edge_index = edge_index
        self.norm_A = norm_A
        
        self.convs = nn.ModuleList()
        for _ in range(n_layers+1):
            self.convs.append(ChebConv(kind, n_hidden, n_hidden, n_layers, theta=theta, lamda=lamda, theta_decay=theta_decay, weight=True, bias=True))
        self.n_layers = n_layers
        self.fcs = nn.ModuleList()
        self.fcs.append(nn.Linear(in_feats, n_hidden))
        self.fcs.append(nn.Linear(n_hidden, n_classes))
        self.params1 = list(self.convs.parameters())
        self.params2 = list(self.fcs.parameters())
        # self.weight = nn.Parameter(th.Tensor(n_hidden, n_hidden))
        self.act_fn = act_fn
        self.last_act_fn = last_act_fn
        
        self.dropout = nn.Dropout(p=dropout)
        self.dropout2 = nn.Dropout(p=dropout2)
        self.set_alphas(alpha)

    def set_alphas(self, alpha):
        t = th.zeros(self.n_layers+1)
        TEMP = alpha
        for i in range(self.n_layers+1):
            j = self.n_layers - i
            t[j] = TEMP
            TEMP = TEMP*(1-alpha)
        t[0] = (1-alpha)**(self.n_layers)
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
        second_last_h = th.zeros_like(h0)

        for i, con in enumerate(self.convs):
            alpha = self.alpha_params[-(i+1)]
            x = con(self.edge_index, self.norm_A, h0, last_h, second_last_h, alpha, i)
            if i < self.n_layers - 1:
                x = self.__relu(x)
                x = self.dropout2(x) # 0928
            second_last_h = last_h
            last_h = x

        x = self.__relu(x)
        x = self.dropout(x)
        x = self.fcs[-1](x)

        x = F.log_softmax(x, dim=1)
        return x