from typing import List, Tuple
import os
import torch.nn as nn
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm import trange
import numpy as np
from ctrModels.deep_models.wdl_dataset import WideDeepDataset
from ctrModels.layers.EmbeddingLayer import EmbeddingLayer
from ctrModels.utils.loss import get_loss
from ctrModels.utils.metrics import get_metric
from ctrModels.utils.optimizers import get_optimizer

n_cpus = os.cpu_count()


class WideDeep(nn.Module):
    def __init__(self, wide_dim, output_dim, wide_model, deep_col_idx, deep_model, embed_dim=10,
                 hidden_units=[64, 32], dnn_dropout=0.5,
                 embed_layer=EmbeddingLayer, embed_input=None, cont_cols=None, embed_dropout=0):
        super(WideDeep, self).__init__()
        self.wide = wide_model(wide_dim=wide_dim, output_dim=output_dim)
        if embed_input is not None:
            embed_input_dim = len(embed_input) * embed_dim
        else:
            embed_input_dim = 0
        if cont_cols is not None:
            cont_input_dim = len(cont_cols)
        else:
            cont_input_dim = 0
        deep_input_dim = embed_input_dim + cont_input_dim
        self.deepdense = deep_model(deep_input_dim=deep_input_dim, hidden_units=hidden_units, dnn_dropout=dnn_dropout,
                                    use_bn=True)
        self.embed_layer = embed_layer(embed_cols=embed_input, embed_dim=embed_dim, deep_col_idx=deep_col_idx,
                                       cont_cols=cont_cols, embed_dropout=embed_dropout)

    def forward(self, X):
        out = self.wide(X['wide'])
        out.add_(self.deepdense(self.embed_layer(X['deepdense'])))
        return out

    def compile(self, method, optimizers='adam', loss_func='binary_crossentropy', metric='acc', verbose=1, seed=2019):
        self.verbose = verbose
        self.seed = seed
        self.early_stop = False
        self.method = method
        self.optimizer = get_optimizer(self.parameters(), optim_type=optimizers)
        self.loss_func = get_loss(loss_func)

        self.metric = lambda y_pred, y_true: get_metric(metric, y_pred=y_pred, y_true=y_true)

    def fit(self, X_wide=None, X_deep=None, X_train=None, X_val=None, target=None,
            val_split=None, n_epochs=1, batch_size=32, validation_freq=1):
        if X_train is None and (X_wide is None or X_deep is None or target is None):
            raise ValueError(
                "Training data is missing. Either a dictionary (X_train) with "
                "the training dataset or at least 3 arrays (X_wide, X_deep, "
                "target) must be passed to the fit method")
        self.batch_size = batch_size
        train_set, eval_set = self._train_val_split(X_wide, X_deep, X_train, X_val, val_split, target)
        train_loader = DataLoader(dataset=train_set, batch_size=batch_size, num_workers=n_cpus)
        train_steps = len(train_loader)
        if self.verbose:
            print("training")
        for epoch in range(n_epochs):
            self.train_running_loss = 0
            with trange(train_steps, disable=self.verbose != 1) as t:
                for batch_idx, (data, target) in zip(t, train_loader):
                    t.set_description('epoch % i' % (epoch + 1))
                    acc, train_loss = self._training_step(data, target, batch_idx)
                    if acc is not None:
                        t.set_postfix(metrics=acc, loss=train_loss)
                    else:
                        t.set_postfix(loss=np.sqrt(train_loss))
            if epoch % validation_freq == (validation_freq - 1):
                if eval_set is not None:
                    eval_loader = DataLoader(dataset=eval_set, batch_size=batch_size, num_workers=n_cpus,
                                             shuffle=False)
                    eval_steps = len(eval_loader)
                    self.valid_running_loss = 0.
                    with trange(eval_steps, disable=self.verbose != 1) as v:
                        for i, (data, target) in zip(v, eval_loader):
                            v.set_description('valid')
                            acc, val_loss = self._validation_step(data, target, i)
                            if acc is not None:
                                v.set_postfix(metrics=acc, loss=val_loss)
                            else:
                                v.set_postfix(loss=np.sqrt(val_loss))
        self.train()

    def predict(self, X_wide=None, X_deep=None, X_test=None):
        if X_test is not None:
            test_set = WideDeepDataset(**X_test)
        else:
            load_dict = {'X_wide': X_wide, 'X_deep': X_deep}
            test_set = WideDeepDataset(**load_dict)
        test_loader = DataLoader(dataset=test_set, batch_size=self.batch_size, num_workers=n_cpus, shuffle=False)
        test_steps = (len(test_loader) // test_loader.batch_size) + 1
        self.eval()
        pred_re = []
        with torch.no_grad():
            with trange(test_steps, disable=self.verbose != 1) as t:
                for i, data in zip(t, test_loader):
                    t.set_description("predict")
                    X = data
                    y_pred = torch.sigmoid(self.forward(X)).data.numpy()
                    pred_re.append(y_pred)
        self.train()
        preds = np.vstack(pred_re).squeeze(1)
        return (preds > 0.5).astype('int')

    def _train_val_split(self, X_wide=None, X_deep=None, X_train=None, X_val=None, val_split=None, target=None):
        if X_val is None and val_split is None:
            if X_train is not None:
                X_wide, X_deep, target = X_train['X_wide'], X_train['X_deep'], X_train['target']

            X_train = {'X_wide': X_wide, 'X_deep': X_deep, 'target': target}
            train_set = WideDeepDataset(**X_train)
            eval_set = None
        else:
            if X_val is not None:
                if X_train is None:
                    X_train = {'X_wide': X_wide, 'X_deep': X_deep, 'target': target}
            else:
                if X_train is not None:
                    X_wide, X_deep, target = X_train['X_wide'], X_train['X_deep'], X_train['target']
                X_tr_wide, X_val_wide, X_tr_deep, X_val_deep, y_tr, y_val = train_test_split(X_wide,
                                                                                             X_deep, target,
                                                                                             test_size=val_split,
                                                                                             random_state=self.seed)
                X_train = {'X_wide': X_tr_wide, 'X_deep': X_tr_deep, 'target': y_tr}
                X_val = {'X_wide': X_val_wide, 'X_deep': X_val_deep, 'target': y_val}

            train_set = WideDeepDataset(**X_train)
            eval_set = WideDeepDataset(**X_val)
        return train_set, eval_set

    def _training_step(self, data, target, batch_idx):
        self.train()
        X = data
        y = target.float()

        self.optimizer.zero_grad()
        y_pred = torch.sigmoid(self.forward(X))
        loss = self.loss_func(y_pred, y.view(-1, 1))
        loss.backward()
        self.optimizer.step()

        self.train_running_loss += loss.item()
        avg_loss = self.train_running_loss / (batch_idx + 1)

        if self.metric is not None:
            acc = self.metric(y_pred, y)
            return acc, avg_loss
        else:
            return None, avg_loss

    def _validation_step(self, data, target, batch_idx: int):

        self.eval()
        with torch.no_grad():
            X = data
            y = target.float()

            y_pred = torch.sigmoid(self.forward(X))
            loss = self.loss_func(y_pred, y.view(-1, 1))
            self.valid_running_loss += loss.item()
            avg_loss = self.valid_running_loss / (batch_idx + 1)

        if self.metric is not None:
            acc = self.metric(y_pred, y)
            return acc, avg_loss
        else:
            return None, avg_loss
