import torch.nn.functional as F

def get_loss(loss_type):
    if loss_type == "binary_crossentropy":
        return F.binary_cross_entropy
    if loss_type == 'mse':
        return F.mse_loss
    if loss_type == 'mae':
        return F.l1_loss
    else:
        raise NotImplementedError