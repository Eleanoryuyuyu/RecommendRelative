"""
A pytorch implementation of DeepFM for rates prediction problem.
"""

import torch
from torch import nn, optim
import torch.nn.functional as F

from ctrModels.utils.loss import get_loss
from ctrModels.utils.metrics import get_metric
from ctrModels.utils.optimizers import get_optimizer


class DeepFM(nn.Module):
    def __init__(self, wide_model=None, deep_model=None):
        super(DeepFM, self).__init__()
        self.wide = wide_model
        self.deepdense = deep_model

    def forward(self, X):
        out = self.wide(X)
        out.add_(self.deepdense(X))
        return out

    def compile(self, optimizer='adam', loss_func='binary_crossentropy', metric='acc', verbose=1, seed=2019):
        self.optimizer = get_optimizer(self.parameters(), optim_type=optimizer)
        self.loss_func = get_loss(loss_type=loss_func)
        self.metric = lambda y_pred, y_true: get_metric(metric_type=metric,y_pred=y_pred, y_true=y_true)
        self.verbose = verbose
        self.seed = seed


    def fit(self, loader_train, loader_val, optimizer, epochs=100, verbose=False, print_every=100):
        """
        Training a model and valid accuracy.
        Inputs:
        - loader_train: I
        - loader_val: .
        - optimizer: Abstraction of optimizer used in training process, e.g., "torch.optim.Adam()""torch.optim.SGD()".
        - epochs: Integer, number of epochs.
        - verbose: Bool, if print.
        - print_every: Integer, print after every number of iterations. 
        """
        """
            load input data
        """
        model = self.train().to(device=self.device)
        criterion = F.binary_cross_entropy_with_logits

        for _ in range(epochs):
            for t, (xi, xv, y) in enumerate(loader_train):
                xi = xi.to(device=self.device, dtype=self.dtype)
                xv = xv.to(device=self.device, dtype=torch.float)
                y = y.to(device=self.device, dtype=torch.float)

                total = model(xi, xv)
                loss = criterion(total, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if verbose and t % print_every == 0:
                    print('Iteration %d, loss = %.4f' % (t, loss.item()))
                    self.check_accuracy(loader_val, model)
                    print()

    def check_accuracy(self, loader, model):
        if loader.dataset.train:
            print('Checking accuracy on validation set')
        else:
            print('Checking accuracy on test set')
        num_correct = 0
        num_samples = 0
        model.eval()  # set model to evaluation mode
        with torch.no_grad():
            for xi, xv, y in loader:
                xi = xi.to(device=self.device, dtype=self.dtype)  # move to device, e.g. GPU
                xv = xv.to(device=self.device, dtype=torch.float)
                y = y.to(device=self.device, dtype=torch.bool)
                total = model(xi, xv)
                preds = (F.sigmoid(total) > 0.5)
                num_correct += (preds == y).sum()
                num_samples += preds.size(0)
            acc = float(num_correct) / num_samples
            print('Got %d / %d correct (%.2f%%)' % (num_correct, num_samples, 100 * acc))