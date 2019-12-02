import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score

from ctrModels.layers.DeepModule import DNN
from ctrModels.layers.WideModule import LinearModule
from ctrModels.deep_models.WideDeep import WideDeep
from ctrModels.preprocessing.DataProcess import *

if __name__ == '__main__':
    train_df, test_df = read_data()
    print(test_df.head())
    df_train, df_test, wide_cols, cross_cols, embedding_cols, cont_cols, target = data_process(train_df, test_df)
    X_train_wide, y_train_wide, X_test_wide, y_test_wide = wide_feature_process(wide_cols, cross_cols, target, df_train,
                                                                                df_test)
    print(X_train_wide.shape, X_test_wide.shape)
    X_train_deep, y_train_deep, X_test_deep, y_test_deep, deep_col_idx, embed_input = \
        deep_feature_process(embedding_cols, cont_cols, target, df_train, df_test)
    # print(X_train_deep.shape, X_test_deep.shape)
    # print(deep_col_idx)
    # print(embed_input)
    wide = LinearModule(wide_dim=X_train_wide.shape[1], output_dim=1)
    deepdense = DNN(deep_col_idx=deep_col_idx, embed_input=embed_input, cont_cols=cont_cols,
                    hidden_units=[64, 32], dnn_dropout=0.5)
    model = WideDeep(wide, deepdense)

    model.compile(method='binary')
    X_train = {'X_wide': X_train_wide, 'X_deep': X_train_deep, 'target': y_train_wide}
    X_val = {'X_wide': X_test_wide, 'X_deep': X_test_deep, 'target': y_test_wide}
    model.fit(X_train=X_train, X_val=X_val, n_epochs=10, batch_size=256)
    X_test = {'X_wide': X_test_wide, 'X_deep': X_test_deep}
    pred_val = model.predict(X_test=X_test)
    acc = accuracy_score(y_test_wide[:256], pred_val)
    print(acc)


