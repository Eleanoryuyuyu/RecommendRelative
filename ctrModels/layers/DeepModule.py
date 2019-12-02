import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class DNN(nn.Module):
    def __init__(self, deep_col_idx, embed_input, cont_cols, hidden_units, activation=F.relu, l2_reg=0,
                 embed_dropout=0., dnn_dropout=0, use_bn=False, init_std=0.0001, seed=2019, device='cpu'):
        super(DNN, self).__init__()
        self.activation = activation
        self.dropout = nn.Dropout(dnn_dropout)
        self.seed = seed
        self.l2_reg = l2_reg
        self.use_bn = use_bn
        self.embed_input = embed_input
        self.continus_cols = cont_cols
        self.deep_col_idx = deep_col_idx
        if len(hidden_units) == 0:
            raise ValueError("hidden_units is empty!")

        # embedding layer
        if self.embed_input is not None:
            self.embed_layers = nn.ModuleDict({'embed_layer_' + col: nn.Embedding(val, dim)
                                               for col, val, dim in self.embed_input})
            self.embed_dropout = nn.Dropout(embed_dropout)
            embed_input_dim = np.sum([embed[2] for embed in self.embed_input])
        else:
            embed_input_dim = 0

        # continous dim
        if self.continus_cols is not None:
            cont_input_dim = len(self.continus_cols)
        else:
            cont_input_dim = 0

        # Dense Layer
        deep_input_dim = embed_input_dim + cont_input_dim
        hidden_units = [deep_input_dim] + hidden_units
        self.linears = nn.ModuleList(
            [nn.Linear(hidden_units[i], hidden_units[i + 1]) for i in range(len(hidden_units) - 1)])
        if self.use_bn:
            self.bn = nn.ModuleList([nn.BatchNorm1d(hidden_units[i + 1]) for i in range(len(hidden_units) - 1)])
        for name, tensor in self.linears.named_parameters():
            if 'weight' in name:
                nn.init.normal_(tensor, mean=0, std=init_std)
        self.lst_linear = nn.Linear(hidden_units[-1], 1, bias=False)
        self.to(device)

    def forward(self, X):
        if self.embed_input is not None:
            x = [self.embed_layers['embed_layer_' + col](X[:, self.deep_col_idx[col]].long())
                 for col, _, _ in self.embed_input]
            x = torch.cat(x, 1)
            x = self.embed_dropout(x)
        if self.continus_cols is not None:
            cont_idx = [self.deep_col_idx[col] for col in self.continus_cols]
            x_cont = X[:, cont_idx].float()
            x = torch.cat([x, x_cont], 1) if self.embed_input is not None else x_cont
        deep_input = x
        for i in range(len(self.linears)):
            fc = self.linears[i](deep_input)
            if self.use_bn:
                fc = self.bn[i](fc)
            fc = self.activation(fc)
            fc = self.dropout(fc)
            deep_input = fc
        out = self.lst_linear(deep_input)
        return out
