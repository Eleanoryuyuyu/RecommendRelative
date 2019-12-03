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
    def __init__(self):
        super(FM, self).__init__()

    def forward(self, x):
        """
        :param x: float tensor of size (batch_size, field_size, embed_size)
        :return:
        """
        intersection_1 = torch.sum(x, dim=1)
        intersection_1 = torch.pow(intersection_1, 2)
        intersection_2 = torch.sum(torch.pow(x, 2), dim=1)
        output = 0.5 * torch.sum(intersection_1 - intersection_2)
        return output

