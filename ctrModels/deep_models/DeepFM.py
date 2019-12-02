"""
A pytorch implementation of DeepFM for rates prediction problem.
"""

import torch
from torch import nn, optim
import torch.nn.functional as F
import numpy as np
from ctrModels.layers import EmbeddingLayer
from ctrModels.utils.loss import get_loss
from ctrModels.utils.metrics import get_metric
from ctrModels.utils.optimizers import get_optimizer


class DeepFM(nn.Module):
    def __init__(self, wide_dim, output_dim, wide_model, deep_col_idx, deep_model, hidden_units=[64, 32],
                 dnn_dropout=0.5, embed_layer=EmbeddingLayer, embed_input=None, cont_cols=None,
                 embed_dropout=0):
        super(DeepFM, self).__init__()
        self.wide = wide_model
        self.wide = wide_model(wide_dim=wide_dim, output_dim=output_dim)
        if embed_input is not None:
            embed_input_dim = np.sum([embed[2] for embed in embed_input])
        else:
            embed_input_dim = 0
        if cont_cols is not None:
            cont_input_dim = len(cont_cols)
        else:
            cont_input_dim = 0
        deep_input_dim = embed_input_dim + cont_input_dim
        self.deepdense = deep_model(deep_input_dim=deep_input_dim, hidden_units=hidden_units, dnn_dropout=dnn_dropout)
        self.embed_layer = embed_layer(deep_col_idx=deep_col_idx, embed_input=embed_input, cont_cols=cont_cols,
                                       embed_dropout=embed_dropout)

    def forward(self, X):
        embed_input = self.embed_layer(X)
        out = self.wide(embed_input)
        out.add_(self.deepdense(embed_input))
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