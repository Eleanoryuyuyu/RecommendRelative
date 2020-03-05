import torch
from torch import nn
import torch.nn.functional as F
import torch.nn.init as weight_init
from torch.autograd import Variable

def activation(input, kind):
    if kind == 'relu':
        return F.relu(input)
    elif kind == 'relu6':
        return F.relu6(input)
    elif kind == 'lrelu':
        return F.leaky_relu(input)
    elif kind == 'elu':
        return F.elu(input)
    elif kind == 'selu':
        return F.selu(input)
    elif kind == 'tanh':
        return F.tanh(input)
    elif kind == 'swish':
        return input * F.sigmoid(input)
    elif kind == 'none':
        return input
    else:
        raise NotImplementedError
def MMSELoss(inputs, targets, size_average=False):
    mask = targets!=0
    num_ratings = torch.sum(mask.float())
    criterion = nn.MSELoss(reduction='sum' if not size_average else 'mean')
    return criterion(inputs*mask.float(), targets), Variable(torch.Tensor([1.0])) if size_average else num_ratings
class AutoEncoder(nn.Module):
    def __init__(self, hiddens, nl_type='selu', is_constrained=True, dropout=0.0,
                 last_layer_activations=True):
        super(AutoEncoder,self).__init__()
        self.dropout = dropout
        if dropout > 0:
            self.drop = nn.Dropout(dropout)
        self.last_layer_activations = last_layer_activations
        self.nl_type = nl_type
        self._last = len(hiddens)-2
        self.encode_w = nn.ParameterList([nn.Parameter(torch.Tensor(hiddens[i+1], hiddens[i])) for i in range(len(hiddens)-1)])
        for w in self.encode_w:
            weight_init.xavier_uniform_(w)
        self.encode_b = nn.ParameterList([nn.Parameter(torch.zeros(hiddens[i+1])) for i in range(len(hiddens)-1)])
        reversed_hiddens = list(reversed(hiddens))
        self.is_constrained = is_constrained
        if not self.is_constrained:
            self.decode_w = nn.ParameterList([nn.Parameter(torch.Tensor(reversed_hiddens[i+1], reversed_hiddens[i])) for i in range(len(reversed_hiddens)-1)])
            for w in self.decode_w:
                weight_init.xavier_uniform(w)
        self.decode_b = nn.ParameterList([nn.Parameter(torch.zeros(reversed_hiddens[i+1])) for i in range(len(reversed_hiddens)-1)])
        print("******************************")
        print("******************************")
        print(hiddens)
        print("Dropout drop probability: {}".format(self.dropout))
        print("Encoder pass:")
        for ind, w in enumerate(self.encode_w):
            print(w.data.size())
            print(self.encode_b[ind].size())
        print("Decoder pass:")
        if self.is_constrained:
            print('Decoder is constrained')
            for ind, w in enumerate(list(reversed(self.encode_w))):
                print(w.transpose(0, 1).size())
                print(self.decode_b[ind].size())
        else:
            for ind, w in enumerate(self.decode_w):
                print(w.data.size())
                print(self.decode_b[ind].size())
        print("******************************")
        print("******************************")
    def encode(self, x):
        for ind, w in enumerate(self.encode_w):
            x = F.linear(input=x, weight=w, bias=self.encode_b[ind])
            x = activation(x, self.nl_type)
        if self.dropout > 0:
            x = self.drop(x)
        return x
    def decode(self, z):
        if self.is_constrained:
            for ind, w in enumerate(list(reversed(self.encode_w))):
                z = F.linear(input=z, weight=w.transpose(0,1), bias=self.decode_b[ind])
                z = activation(z, kind=self.nl_type if ind != self._last or self.last_layer_activations else 'none')
        else:
            for ind, w in enumerate(self.decode_w):
                z = F.linear(input=z, weight=w, bias=self.decode_b[ind])
                z = activation(z, kind=self.nl_type if ind != self._last or self.last_layer_activations else 'none')
        return z
    def forward(self, x):
        return self.decode(self.encode(x))



