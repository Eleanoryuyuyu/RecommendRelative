import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class EmbeddingLayer(nn.Module):
    def __init__(self, deep_col_idx, embed_input=None, cont_cols=None, embed_dropout=0):
        super(EmbeddingLayer, self).__init__()
        self.embed_input = embed_input
        self.continus_cols = cont_cols
        self.deep_col_idx = deep_col_idx
        if self.embed_input is not None:
            self.embed_layers = nn.ModuleDict({'embed_layer_' + col: nn.Embedding(val, dim)
                                               for col, val, dim in self.embed_input})
            self.embed_dropout = nn.Dropout(embed_dropout)

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
        return x