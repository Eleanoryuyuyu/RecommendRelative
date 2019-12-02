import numpy as np
import torch

from sklearn.utils import Bunch
from torch.utils.data import Dataset
from scipy.sparse.csr import csr_matrix as sparse_matrix


class WideDeepDataset(Dataset):

    def __init__(self, X_wide, X_deep, target=None):
        self.X_wide = X_wide
        self.X_deep = X_deep
        self.Y = target

    def __getitem__(self, idx):
        # X_wide and X_deep are assumed to be *always* present
        if isinstance(self.X_wide, sparse_matrix):
            X = Bunch(wide=np.array(self.X_wide[idx].todense()).squeeze())
        else:
            X = Bunch(wide=self.X_wide[idx])
        X.deepdense = self.X_deep[idx]
        if self.Y is not None:
            y = self.Y[idx]
            return X, y
        else:
            return X

    def __len__(self):
        return len(self.X_deep)
