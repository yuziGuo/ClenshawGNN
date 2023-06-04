# https://github.com/CUAI/Non-Homophily-Benchmarks/blob/main/models.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, JumpingKnowledge
# from torch_geometric.nn.conv.gcn_conv import gcn_norm
# import scipy.sparse
# import numpy as np

class GCNJK(nn.Module):
    def __init__(self, 
                 edge_index,
                 norm_A,
                 in_channels, 
                 hidden_channels, 
                 out_channels, 
                 num_layers=2,
                 dropout=0.5, 
                 jk_type='max'
                 ):
        super(GCNJK, self).__init__()
        self.norm_A = norm_A
        self.edge_index = edge_index

        self.convs = nn.ModuleList()
        self.convs.append(
            GCNConv(in_channels, hidden_channels, cached=False, normalize=False))

        self.bns = nn.ModuleList()
        self.bns.append(nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(
                GCNConv(hidden_channels, hidden_channels, cached=False, normalize=False))
            self.bns.append(nn.BatchNorm1d(hidden_channels))

        self.convs.append(
            GCNConv(hidden_channels, hidden_channels, cached=False, normalize=False))

        self.dropout = dropout
        self.activation = F.relu
        self.jump = JumpingKnowledge(jk_type, channels=hidden_channels, num_layers=1)
        if jk_type == 'cat':
            self.final_project = nn.Linear(hidden_channels * num_layers, out_channels)
        else: # max or lstm
            self.final_project = nn.Linear(hidden_channels, out_channels)


    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()
        self.jump.reset_parameters()
        self.final_project.reset_parameters()

    def predict(self, x):
        with torch.no_grad():
            self.eval()
            xs = []
            for i, conv in enumerate(self.convs[:-1]):
                x = conv(x, self.edge_index, edge_weight=self.norm_A)
                x = self.bns[i](x)
                x = self.activation(x)
                xs.append(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.convs[-1](x, self.edge_index, edge_weight=self.norm_A)
            xs.append(x)
            x = self.jump(xs)
            x = self.final_project(x)
            return x

    def forward(self, x):
        xs = []
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, self.edge_index, edge_weight=self.norm_A)
            x = self.bns[i](x)
            x = self.activation(x)
            xs.append(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, self.edge_index, edge_weight=self.norm_A)
        xs.append(x)
        x = self.jump(xs)
        x = self.final_project(x)
        x = F.log_softmax(x, dim=1)
        return x