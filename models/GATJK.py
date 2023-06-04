'''
From https://github.com/CUAI/Non-Homophily-Benchmarks/blob/main/models.py

Note: To align with the control groups, we don't use batchnorm.
'''

import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, JumpingKnowledge

class GATJK(nn.Module):
    def __init__(      
            self,   
            edge_index,
            norm_A,
            in_channels, 
            hidden_channels, 
            out_channels, 
            n_layers,
            heads,
            dropout=0.6,
            jk_type='max',
            ):
        super(GATJK, self).__init__()
        self.edge_index = edge_index
        self.norm_A = norm_A
        self.n_layers = n_layers
        self.dropout=dropout
        self.convs = nn.ModuleList()
        self.convs.append(GATConv(in_channels, hidden_channels//heads, heads=heads, dropout=dropout))
        for _ in range(n_layers - 1):
            self.convs.append(GATConv(hidden_channels, hidden_channels//heads, heads=heads, dropout=dropout) ) 
        self.dropout = dropout
        self.activation = F.elu # note: uses elu

        self.jump = JumpingKnowledge(jk_type, channels=hidden_channels, num_layers=1)
        if jk_type == 'cat':
            self.final_project = nn.Linear(hidden_channels*n_layers, out_channels)
        else: # max or lstm
            self.final_project = nn.Linear(hidden_channels, out_channels)


    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        self.jump.reset_parameters()
        self.final_project.reset_parameters()


    def forward(self, x):
        xs = []
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, self.edge_index)
            x = self.activation(x)
            xs.append(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, self.edge_index)
        xs.append(x)
        x = self.jump(xs)
        x = self.final_project(x)
        x = F.log_softmax(x, dim=1)
        return x