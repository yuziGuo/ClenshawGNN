'''From https://github.com/pyg-team/pytorch_geometric/blob/master/examples/gat.py
'''

import torch
import torch.nn.functional as F
from layers.GATConv import GATConv


class GAT(torch.nn.Module):
    def __init__(self, 
            edge_index,
            norm_A,
            in_channels, 
            hidden_channels, 
            out_channels, 
            n_layers,
            heads,
            out_heads=1,
            dropout=0.6,
            with_negative_residual=False,
            with_initial_residual=False
            ):
        super().__init__()
        self.edge_index = edge_index
        self.norm_A = norm_A
        self.n_layers = n_layers
        self.with_negative_residual=with_negative_residual
        self.with_initial_residual=with_initial_residual
        self.dropout=dropout
        self.convs = torch.nn.ModuleList()
        if self.with_initial_residual:
            if self.n_layers>2:
                self.init_alphas()
        # assert
        self.convs.append(GATConv(in_channels, hidden_channels//heads, heads, dropout=dropout))
        for _ in range(1,self.n_layers-1):
            self.convs.append(GATConv(hidden_channels, hidden_channels//heads, heads, dropout=dropout))
        self.convs.append(GATConv(hidden_channels, out_channels, heads=out_heads, concat=False, dropout=dropout))


    
    def init_alphas(self):
        t = torch.zeros(self.n_layers - 2)
        t[0] = 1
        self.alphas = torch.nn.Parameter(t.float())

    def forward(self, x):
        # first layer
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.convs[0](x, self.edge_index))
        if self.with_initial_residual:
            x_0 = x

        # median layers 
        for i in range(1, self.n_layers-1):
            second_last_x = x
            x = F.dropout(x, p=0.6, training=self.training)
            x = F.elu(self.convs[i](x, self.edge_index))
            if self.with_negative_residual:
                x = x + (x-second_last_x)
            if self.with_initial_residual:
                x = x + self.alphas[i-1] * x_0

        # last layer
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, self.edge_index)
        x = F.log_softmax(x, dim=1)
        return x

        