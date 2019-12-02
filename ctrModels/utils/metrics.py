import numpy as np
from sklearn.metrics import *

def get_metric(metric_type, y_pred, y_true):
    if metric_type == 'acc':
        correct_count = y_pred.round().eq(y_true.view(-1, 1)).float().sum().item()
        total_count = len(y_pred)
        acc = float(correct_count) / float(total_count)
        return np.round(acc, 4)
    if metric_type == 'auc':
        return roc_auc_score
    else:
        raise NotImplementedError


