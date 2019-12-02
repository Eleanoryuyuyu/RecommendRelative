import torch
def get_optimizer(param, optim_type):
    if optim_type == 'sgd':
        return torch.optim.SGD(param)
    if optim_type == 'adam':
        return torch.optim.Adam(param)
    if optim_type == 'adagrad':
        return torch.optim.Adagrad(param)
    if optim_type == 'rmsprop':
        return torch.optim.RMSprop(param)
    else:
        raise  NotImplementedError