
from ctrModels.deep_models.DeepFM import DeepFM
from ctrModels.preprocessing.DataProcess import *

if __name__ == '__main__':
    train_df, test_df = read_data()
    print(test_df.head())
    df_train, df_test, wide_cols, cross_cols, embedding_cols, cont_cols, target = data_process(train_df, test_df)
    X_train, y_train, X_test, y_test, deep_col_idx, embed_input = \
        deepfm_feature_process(embedding_cols, target, df_train, df_test)
    train_data = {'data': X_train, 'target': y_train}
    eval_data = {'data': X_test, 'target': y_test}

    model = DeepFM(embed_cols=embed_input, embed_dim=10, deep_col_idx=deep_col_idx, hidden_units=[64, 32],
                   dnn_dropout=0.9)
    model.compile(method='binary', optimizer='adam', loss_func='binary_crossentropy', metric='acc')

    model.fit(train_data=train_data, eval_data=eval_data, epochs=50, batch_size=256, validation_freq=5)



