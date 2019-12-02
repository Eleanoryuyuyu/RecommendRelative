import torch
import torch.nn as nn



class LinearModule(nn.Module):
    def __init__(self, wide_dim, output_dim=1):
        super(LinearModule, self).__init__()
        self.wider_linear = nn.Linear(wide_dim, output_dim)

    def forward(self, X):
        out = self.wider_linear(X.float())
        return out

class FM(nn.Module):
    def __init__(self, n=10, k=5):
        super(FM, self).__init__()
        self.n = n
        self.k = k
        self.linear = nn.Linear(n, 1)
        self.V = nn.Parameter(torch.randn(self.n, self.k))

    def forward(self, x):
        linear_part = self.linear(x)
        intersection_1 = torch.mm(x, self.V)
        intersection_1 = torch.pow(intersection_1, 2)
        intersection_2 = torch.mm(torch.pow(x, 2), torch.pow(self.V, 2))
        output = linear_part + 0.5 * torch.sum(intersection_1 - intersection_2)
        return output

