import torch
import torch.nn as nn



class LinearModule(nn.Module):
    def __init__(self, wide_dim, output_dim=1):
        super(LinearModule, self).__init__()
        self.wider_linear = nn.Linear(wide_dim, output_dim)

    def forward(self, X):
        out = self.wider_linear(X.float())
        return out


