"""
A pytorch implementation of DeepFM for rates prediction problem.
"""

import torch
from sklearn.model_selection import train_test_split
from torch import nn, optim
import torch.nn.functional as F
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
import os
from tqdm import trange
from ctrModels.layers.EmbeddingLayer import EmbeddingLayer
from ctrModels.layers.DeepModule import DNN
from ctrModels.layers.WideModule import LinearModule, FM
from ctrModels.utils.loss import get_loss
from ctrModels.utils.metrics import get_metric
from ctrModels.utils.optimizers import get_optimizer

n_cpus = os.cpu_count()


class DeepFM(nn.Module):
    def __init__(self, embed_cols, embed_dim, deep_col_idx, hidden_units=[64, 32],
                 dnn_dropout=0.5, cont_cols=None, embed_dropout=0):
        super(DeepFM, self).__init__()
        self.linear = LinearModule(wide_dim=len(embed_cols), output_dim=1)
        self.embed_layer = EmbeddingLayer(embed_cols=embed_cols, embed_dim=embed_dim, deep_col_idx=deep_col_idx,
                                          embed_dropout=embed_dropout)
        self.fm = FM()
        if embed_cols is not None:
            embed_input_dim = len(embed_cols) * embed_dim
        else:
            embed_input_dim = 0
        if cont_cols is not None:
            cont_input_dim = len(cont_cols)
        else:
            cont_input_dim = 0
        self.deep_input_dim = embed_input_dim + cont_input_dim
        self.deepdense = DNN(deep_input_dim=self.deep_input_dim, hidden_units=hidden_units, dnn_dropout=dnn_dropout,
                             use_bn=True)

    def forward(self, X):
        """
        :param X:  long tensor of size(batch_size, num_filed)
        :return:
        """
        embed_input = self.embed_layer(X)
        linear_part = self.linear(X)
        fm_part = self.fm(embed_input)
        deep_part = self.deepdense(embed_input.view(-1, self.deep_input_dim))
        out = linear_part + fm_part + deep_part
        return torch.sigmoid(out)

    def compile(self, method, optimizer='adam', loss_func='binary_crossentropy', metric='acc', verbose=1, seed=2019):
        self.method = method
        self.optimizer = get_optimizer(self.parameters(), optim_type=optimizer)
        self.loss_func = get_loss(loss_type=loss_func)
        self.metric = lambda y_pred, y_true: get_metric(metric_type=metric, y_pred=y_pred, y_true=y_true)
        self.verbose = verbose
        self.seed = seed

    def fit(self, train_data, eval_data=None, batch_size=32, epochs=100, validation_freq=1):
        self.batch_size = batch_size
        train_set, eval_set = self._train_val_split(train_data, eval_data)
        train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True, num_workers=n_cpus)
        train_steps = len(train_loader)
        if self.verbose:
            print("training...")
        for epoch in range(epochs):
            self.training_loss = 0.0
            with trange(train_steps, disable=self.verbose != 1) as t:
                for batch_idx, (data, target) in zip(t, train_loader):
                    t.set_description("epoch % i" % (epoch + 1))
                    acc, avg_loss = self._train_step(data, target, batch_idx)
                    if acc is not None:
                        t.set_postfix(metrics=acc, loss=avg_loss)
                    else:
                        t.set_postfix(loss=np.sqrt(avg_loss))
            if epoch % validation_freq == (validation_freq - 1):
                if eval_set is not None:
                    eval_loader = DataLoader(dataset=eval_set, batch_size=batch_size, num_workers=n_cpus,
                                             shuffle=False)
                    eval_steps = len(eval_loader)
                    self.eval_loss = 0.0
                    if self.verbose:
                        print("evaluating...")
                    with trange(eval_steps, disable=self.verbose != 1) as v:
                        for i, (data, target) in zip(v, eval_loader):
                            v.set_description("valid")
                            acc, avg_loss = self._valid_step(i, data, target)
                            if acc is not None:
                                v.set_postfix(metric=acc, loss=avg_loss)
                            else:
                                v.set_postfix(loss=np.sqrt(avg_loss))
        self.train()

    def _train_val_split(self, train, eval):
        X_train, y_train = train['data'], train['target']
        if eval is None:
            X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=self.seed)
        else:
            X_val, y_val = eval['data'], eval['target']
        X_train,X_val,y_train,y_val = torch.from_numpy(X_train),torch.from_numpy(X_val),torch.from_numpy(y_train),torch.from_numpy(y_val)
        train_set = TensorDataset(X_train, y_train)
        val_set = TensorDataset(X_val, y_val)
        return train_set, val_set

    def _train_step(self, data, target, batch_idx):
        self.train()
        X = data
        y = target.float()
        self.optimizer.zero_grad()
        y_pred = self.forward(X)
        loss = self.loss_func(y_pred, y.view(-1, 1))
        loss.backward()
        self.optimizer.step()
        self.training_loss += loss.item()
        avg_loss = self.training_loss / (batch_idx + 1)
        if self.metric is not None:
            acc = self.metric(y_pred=y_pred, y_true=y)
        else:
            acc = None
        return acc, avg_loss

    def _valid_step(self, i, data, target):
        self.eval()
        with torch.no_grad():
            X = data,
            y = target.float()
            y_pred = self.forward(X)
            loss = self.loss_func(y_pred, y.view(-1, 1))
            self.eval_loss += loss.item()
            avg_loss = self.eval_loss / (i + 1)
            if self.metric is not None:
                acc = self.metric(y_pred=y_pred, y_true=y)
            else:
                acc = None
        return acc, avg_loss
