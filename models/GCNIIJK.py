# https://github.com/CUAI/Non-Homophily-Benchmarks/blob/main/models.py

import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import JumpingKnowledge
from layers.GCNIIConv import GraphConvII


class GCNIIJK(nn.Module):
    def __init__(self, 
                 edge_index,
                 norm_A,
                 in_channels, 
                 hidden_channels, 
                 out_channels, 
                 num_layers=2,
                 dropout=0.5, 
                 dropout2=0.5,
                 alpha=0.5,
                 lamda=1.0,
                 jk_type='max'):
        super(GCNIIJK, self).__init__()

        self.edge_index = edge_index
        self.norm_A = norm_A
        
        self.convs = nn.ModuleList()
        self.fcs = nn.ModuleList()
        # self.bns = nn.ModuleList()

        # convs
        for _ in range(num_layers):
            self.convs.append(GraphConvII(hidden_channels, hidden_channels))
        self.dropout = nn.Dropout(p=dropout)
        self.dropout2 = nn.Dropout(p=dropout2)
        self.alpha = alpha
        self.lamda = lamda

        # bns
        # for _ in range(num_layers-1):    
        #     self.bns.append(nn.BatchNorm1d(hidden_channels))

        # fcs
        self.fcs = nn.ModuleList()
        self.fcs.append(nn.Linear(in_channels, hidden_channels))
        
        # JK
        self.jump = JumpingKnowledge(jk_type, channels=hidden_channels, num_layers=1)

        self.act_fn = F.relu
        self.params1 = list(self.convs.parameters())
        self.params2 = list(self.fcs.parameters())
        # self.params3 = list(self.bns.parameters())  

        if jk_type == 'cat':
            self.fcs.append(nn.Linear(hidden_channels * (num_layers+1), out_channels))
        elif jk_type == 'max': # max or lstm  
            self.fcs.append(nn.Linear(hidden_channels, out_channels))
        else: # We don't try lstm
            assert False

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        # for bn in self.bns:
        #     bn.reset_parameters()
        for fc in self.fcs:
            fc.reset_parameters()

    def forward(self, features):
        x = features
        x = self.dropout(x)
        x = self.fcs[0](x)
        x = self.act_fn(x)
        xs = [x]
        h0 = x

        for i, con in enumerate(self.convs):
            x = self.dropout2(x)
            x = con(x, self.edge_index, self.norm_A, h0, self.lamda, self.alpha, i+1)
            x = self.act_fn(x)
            xs.append(x)

        x = self.jump(xs)
        x = self.dropout(x)
        x = self.fcs[-1](x)
        x = F.log_softmax(x, dim=1)
        return x