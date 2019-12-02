import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from ctrModels.layers.EmbeddingLayer import EmbeddingLayer


class DNN(nn.Module):
    def __init__(self, deep_input_dim, hidden_units, activation=F.relu,
                 l2_reg=0, dnn_dropout=0, use_bn=False, init_std=0.0001, seed=2019, device='cpu'):
        super(DNN, self).__init__()
        self.activation = activation
        self.dropout = nn.Dropout(dnn_dropout)
        self.seed = seed
        self.l2_reg = l2_reg
        self.use_bn = use_bn
        if len(hidden_units) == 0:
            raise ValueError("hidden_units is empty!")
        print(deep_input_dim)
        # Dense Layer
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
        deep_input = X
        for i in range(len(self.linears)):
            fc = self.linears[i](deep_input)
            if self.use_bn:
                fc = self.bn[i](fc)
            fc = self.activation(fc)
            fc = self.dropout(fc)
            deep_input = fc
        out = self.lst_linear(deep_input)
        return out
