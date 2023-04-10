'''From https://github.com/pyg-team/pytorch_geometric/blob/master/examples/gat.py
'''

import torch
import torch.nn.functional as F
from layers.GATConv import GATConv   


# When use clenshaw residuals (negative residual + initial residual)
# class GATV2(torch.nn.Module):
#     def __init__(self, 
#             edge_index,
#             norm_A,
#             in_channels, 
#             hidden_channels, 
#             out_channels, 
#             n_layers,
#             heads,
#             out_heads=1,
#             dropout=0.6,
#             dropout2=0.6,
#             with_negative_residual=False,
#             with_initial_residual=False
#             ):
#         super().__init__()
#         self.edge_index = edge_index
#         self.norm_A = norm_A
#         self.K = n_layers
#         self.with_negative_residual=with_negative_residual
#         self.with_initial_residual=with_initial_residual
#         self.dropout=dropout
#         self.dropout2 = dropout2

#         self.fcs = torch.nn.ModuleList()
#         self.fcs.append(torch.nn.Linear(in_channels, hidden_channels))
#         self.fcs.append(torch.nn.Linear(hidden_channels, out_channels))
#         self.convs = torch.nn.ModuleList()
#         if self.with_initial_residual:
#             self.init_alphas()
        
#         # assert
#         for _ in range(0,self.K+1):
#             self.convs.append(GATConv(hidden_channels, hidden_channels//heads, heads, dropout=dropout))

#     def init_alphas(self):
#         t = torch.zeros(self.K+1)
#         t[0] = 1
#         self.alphas = torch.nn.Parameter(t.float())

#     def _forward(self, x):
#         # MLP
#         x = F.dropout(x, p=self.dropout, training=self.training)
#         x = self.fcs[0](x)
#         h0 = x
#         if self.with_initial_residual:
#             last_h = self.alphas[-1]*x
#         else:
#             last_h = x
#         second_last_h = torch.zeros_like(h0)

#         # convlutions
#         for i, con in enumerate(self.convs,1):
#             x = F.dropout(last_h, p=self.dropout2, training=self.training)
#             x = F.elu(con(x, self.edge_index))
#             if self.with_negative_residual:
#                 x = 2*x - second_last_h
#             if self.with_initial_residual:
#                 alpha = self.alphas[-(i + 1)]
#                 x = x + alpha * h0
#             second_last_h = last_h
#             last_h = x

#         # MLP
#         x = F.dropout(x, p=self.dropout, training=self.training)
#         x = self.fcs[-1](x)
#         x = F.log_softmax(x, dim=1)
#         return x
    
#     def forward(self, x):
#         # MLP
#         x = F.dropout(x, p=self.dropout, training=self.training)
#         x = self.fcs[0](x)
#         h0 = x

#         last_h = torch.zeros_like(h0)
#         second_last_h = torch.zeros_like(h0)

#         # convlutions
#         for i, con in enumerate(self.convs,0):
#             x = F.dropout(last_h, p=self.dropout2, training=self.training)
#             x = F.elu(con(x, self.edge_index))
#             if self.with_negative_residual:
#                 x = 2*x - second_last_h
#             if self.with_initial_residual:
#                 alpha = self.alphas[-(i + 1)]
#                 x = x + alpha * h0
#             second_last_h = last_h
#             last_h = x

#         # MLP
#         x = F.dropout(x, p=self.dropout, training=self.training)
#         x = self.fcs[-1](x)
#         x = F.log_softmax(x, dim=1)
#         return x



# When only use negative residuals
class GATV2(torch.nn.Module):
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
            dropout2=0.6,
            with_negative_residual=False,
            with_initial_residual=False
            ):
        super().__init__()
        self.edge_index = edge_index
        self.norm_A = norm_A
        self.K = n_layers
        self.with_negative_residual=with_negative_residual
        self.with_initial_residual=with_initial_residual
        self.dropout=dropout
        self.dropout2 = dropout2

        self.fcs = torch.nn.ModuleList()
        self.fcs.append(torch.nn.Linear(in_channels, hidden_channels))
        self.fcs.append(torch.nn.Linear(hidden_channels, out_channels))
        self.convs = torch.nn.ModuleList()
        if self.with_initial_residual:
            self.init_alphas()
        
        # assert
        for _ in range(0,self.K):
            self.convs.append(GATConv(hidden_channels, hidden_channels//heads, heads, dropout=dropout))

    def init_alphas(self):
        t = torch.zeros(self.K+1)
        t[0] = 1
        self.alphas = torch.nn.Parameter(t.float())

    def _forward(self, x):
        # MLP
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.fcs[0](x)
        h0 = x
        if self.with_initial_residual:
            last_h = self.alphas[-1]*x
        else:
            last_h = x
        second_last_h = torch.zeros_like(h0)

        # convlutions
        for i, con in enumerate(self.convs,1):
            x = F.dropout(last_h, p=self.dropout2, training=self.training)
            x = F.elu(con(x, self.edge_index))
            if self.with_negative_residual:
                x = 2*x - second_last_h
            if self.with_initial_residual:
                alpha = self.alphas[-(i + 1)]
                x = x + alpha * h0
            second_last_h = last_h
            last_h = x

        # MLP
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.fcs[-1](x)
        x = F.log_softmax(x, dim=1)
        return x
    
    def forward(self, x):
        # MLP
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.fcs[0](x)
        h0 = x

        last_h = x
        second_last_h = torch.zeros_like(h0)

        # convlutions
        for i, con in enumerate(self.convs,0):
            x = F.dropout(last_h, p=self.dropout2, training=self.training)
            x = F.elu(con(x, self.edge_index))
            if self.with_negative_residual:
                x = 2*x - second_last_h
            if self.with_initial_residual:
                alpha = self.alphas[-(i + 1)]
                x = x + alpha * h0
            second_last_h = last_h
            last_h = x

        # MLP
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.fcs[-1](x)
        x = F.log_softmax(x, dim=1)
        return x
