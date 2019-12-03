import numpy as np
import pandas as pd
import os
import argparse
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, StandardScaler


def read_data():
    COLUMNS = ["age", "workclass", "fnlwgt", "education", "education_num",
               "marital_status", "occupation", "relationship", "race", "gender",
               "capital_gain", "capital_loss", "hours_per_week", "native_country",
               "income_bracket"]

    df_train = pd.read_csv("../data/adult.data",
                           names=COLUMNS, skipinitialspace=True)

    df_test = pd.read_csv("../data/adult.test",
                          names=COLUMNS, skipinitialspace=True, skiprows=1)

    return df_train, df_test

def cross_columns(x_cols):
    """simple helper to build the crossed columns in a pandas dataframe
    """
    crossed_columns = dict()
    colnames = ['_'.join(x_c) for x_c in x_cols]
    for cname, x_c in zip(colnames, x_cols):
        crossed_columns[cname] = x_c
    return crossed_columns


def val2idx(df, cols):
    """helper to index categorical columns before embeddings.
    """
    val_types = dict()
    for c in cols:
        val_types[c] = df[c].unique()

    val_to_idx = dict()
    for k, v in val_types.items():
        val_to_idx[k] = {o: i for i, o in enumerate(val_types[k])}

    for k, v in val_to_idx.items():
        df[k] = df[k].apply(lambda x: v[x])

    unique_vals = dict()
    for c in cols:
        unique_vals[c] = df[c].nunique()

    return df, val_to_idx


def onehot(x):
    return np.array(OneHotEncoder().fit_transform(x).todense())


def data_process(df_train, df_test):
    df_train['income_label'] = (
        df_train["income_bracket"].apply(lambda x: ">50K" in x)).astype(int)
    df_test['income_label'] = (
        df_test["income_bracket"].apply(lambda x: ">50K" in x)).astype(int)
    age_groups = [0, 25, 65, 90]
    age_labels = range(len(age_groups) - 1)
    df_train['age_group'] = pd.cut(
        df_train['age'], age_groups, labels=age_labels)
    df_test['age_group'] = pd.cut(
        df_test['age'], age_groups, labels=age_labels)
    df_train.drop('income_bracket', axis=1, inplace=True)
    df_test.drop('income_bracket', axis=1, inplace=True)


    # columns for wide model
    wide_cols = ['workclass', 'education', 'marital_status', 'occupation',
                 'relationship', 'race', 'gender', 'native_country', 'age_group']
    cross_cols = (['education', 'occupation'], ['native_country', 'occupation'])

    # columns for deep model
    embedding_cols = ['workclass', 'education', 'marital_status', 'occupation',
                      'relationship', 'race', 'gender', 'native_country']
    cont_cols = ['age', 'capital_gain', 'capital_loss', 'hours_per_week']

    # target for logistic
    target = 'income_label'

    return df_train, df_test, wide_cols, cross_cols, embedding_cols, cont_cols, target

def wide_feature_process(wide_cols, cross_cols, target, df_train, df_test):
    df_train['IS_TRAIN'] = 1
    df_test['IS_TRAIN'] = 0
    df_wide = pd.concat([df_train, df_test], ignore_index=True)
    crossed_cols = cross_columns(cross_cols)
    for k, v in crossed_cols.items():
        df_wide[k] = df_wide[v].apply(lambda x: '-'.join(x), axis=1)
    wide_cols += list(crossed_cols.keys())
    df_wide = df_wide[wide_cols + [target] + ['IS_TRAIN']]
    df_wide = pd.get_dummies(df_wide, columns=[x for x in wide_cols])
    train = df_wide[df_wide.IS_TRAIN == 1].drop('IS_TRAIN', axis=1)
    test = df_wide[df_wide.IS_TRAIN == 0].drop('IS_TRAIN', axis=1)
    assert all(train.columns == test.columns)

    cols = [c for c in train.columns if c != target]
    X_train = train[cols].values
    y_train = train[target].values.reshape(-1, 1)
    X_test = test[cols].values
    y_test = test[target].values.reshape(-1, 1)
    return X_train, y_train, X_test, y_test
def deep_feature_process(embedding_cols, cont_cols, target, df_train,df_test):
    df_train['IS_TRAIN'] = 1
    df_test['IS_TRAIN'] = 0
    df_deep = pd.concat([df_train, df_test], ignore_index=True)
    deep_cols = embedding_cols + cont_cols
    df_deep = df_deep[deep_cols + [target, 'IS_TRAIN']]
    scaler = StandardScaler()
    df_deep[cont_cols] = pd.DataFrame(scaler.fit_transform(df_deep[cont_cols]), columns=cont_cols)
    df_deep, unique_vals = val2idx(df_deep, embedding_cols)
    train = df_deep[df_deep.IS_TRAIN == 1].drop('IS_TRAIN', axis=1)
    test = df_deep[df_deep.IS_TRAIN == 0].drop('IS_TRAIN', axis=1)
    embed_input = []
    for k, v in unique_vals.items():
        embed_input.append((k, len(v)))
    X_train_deep = train[deep_cols].values
    y_train_deep = np.array(train[target].values).reshape(-1, 1)
    X_test_deep = test[deep_cols].values
    y_test_deep = np.array(test[target].values).reshape(-1, 1)
    deep_col_idx = {k: idx for idx, k in enumerate(deep_cols)}
    return X_train_deep, y_train_deep, X_test_deep, y_test_deep, deep_col_idx, embed_input
def deepfm_feature_process(embedding_cols, target, df_train,df_test):
    df_train['IS_TRAIN'] = 1
    df_test['IS_TRAIN'] = 0
    df_deep = pd.concat([df_train, df_test], ignore_index=True)
    df_deep = df_deep[embedding_cols + [target, 'IS_TRAIN']]
    df_deep, unique_vals = val2idx(df_deep, embedding_cols)
    train = df_deep[df_deep.IS_TRAIN == 1].drop('IS_TRAIN', axis=1)
    test = df_deep[df_deep.IS_TRAIN == 0].drop('IS_TRAIN', axis=1)
    embed_input = []
    for k, v in unique_vals.items():
        embed_input.append((k, len(v)))
    X_train = train[embedding_cols].values
    y_train = np.array(train[target].values).reshape(-1, 1)
    X_test = test[embedding_cols].values
    y_test = np.array(test[target].values).reshape(-1, 1)
    deep_col_idx = {k: idx for idx, k in enumerate(embedding_cols)}
    return X_train, y_train, X_test, y_test, deep_col_idx, embed_input









