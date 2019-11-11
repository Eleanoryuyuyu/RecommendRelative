import torch
from torch import nn, optim
import numpy as np
from collections import defaultdict
from scipy.sparse import csr
import pandas as pd
from tqdm import tqdm


def vectorize_dic(dic, ix=None, p=None, n=0, g=0):
    """
    dic -- dictionary of feature lists. Keys are the name of features
    ix -- index generator (default None)
    p -- dimension of feature space (number of columns in the sparse matrix) (default None)
    """
    if ix == None:
        ix = dict()

    nz = n * g

    col_ix = np.empty(nz, dtype=int)

    i = 0
    for k, lis in dic.items():
        for t in range(len(lis)):
            ix[str(lis[t]) + str(k)] = ix.get(str(lis[t]) + str(k), 0) + 1
            col_ix[i + t * g] = ix[str(lis[t]) + str(k)]
        i += 1

    row_ix = np.repeat(np.arange(0, n), g)
    data = np.ones(nz)
    if p == None:
        p = len(ix)

    ixx = np.where(col_ix < p)
    return csr.csr_matrix((data[ixx], (row_ix[ixx], col_ix[ixx])), shape=(n, p)), ix


def batcher(X_, y_=None, batch_size=-1):
    n_samples = X_.shape[0]

    if batch_size == -1:
        batch_size = n_samples
    if batch_size < 1:
        raise ValueError('Parameter batch_size={} is unsupported'.format(batch_size))

    for i in range(0, n_samples, batch_size):
        upper_bound = min(i + batch_size, n_samples)
        ret_x = X_[i:upper_bound]
        ret_y = None
        if y_ is not None:
            ret_y = y_[i:i + batch_size]
            yield (ret_x, ret_y)


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


cols = ['user', 'item', 'rating', 'timestamp']
train = pd.read_csv('data/ua.base', delimiter='\t', names=cols)
test = pd.read_csv('data/ua.test', delimiter='\t', names=cols)
x_train, ix = vectorize_dic(dic={'user': train['user'].values, 'item': train['item'].values},
                            n=len(train.index), g=2)
x_test, ix = vectorize_dic(dic={'user': test['user'].values, 'item': test['item'].values},
                           ix=ix, p=x_train.shape[1], n=len(test.index), g=2)

y_train = train['rating'].values
y_test = test['rating'].values
x_train = x_train.todense()
x_test = x_test.todense()
print(x_train.shape)
n, p = x_train.shape
k = 10

batch_size = 64
model = FM(p, k)
loss_fn = nn.MSELoss()
optimer = optim.SGD(model.parameters(), lr=0.001, weight_decay=0.001)
epochs = 100
for epoch in range(epochs):
    loss_all = 0.0
    perm = np.random.permutation(x_train.shape[0])
    model.train()
    for x, y in tqdm(batcher(x_train[perm], y_train[perm], batch_size)):
        x = torch.as_tensor(np.array(x.tolist()), dtype=torch.float)
        y = torch.as_tensor(np.array(y.tolist()), dtype=torch.float)
        x = x.view(-1, p)
        y = y.view(-1, 1)
        preds = model(x)
        loss = loss_fn(preds, y)

        optimer.zero_grad()
        loss_all += loss.item()
        loss.backward()
        optimer.step()
    loss_epoch = loss_all / len(x)

    if (epoch + 1) % 10 == 0:
        print("Epoch:", epoch, "| Loss {:.8f}:".format(loss_epoch))
