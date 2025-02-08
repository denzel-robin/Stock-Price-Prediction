import numpy as np 
import pandas_ta as ta 
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

def train_test_split(dataset, tstart, tend, columns=['High']):
    train = dataset.loc[f"{tstart}":f"{tend}", columns].values
    test = dataset.loc[f"{tend+1}":, columns].values
    return train, test

def split_sequence(sequence, n_steps):
    x,y = list(), list()
    for i in range(len(sequence)):
        end_ix = i + n_steps
        if end_ix > len(sequence) - 1:
            break
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
        x.append(seq_x)
        y.append(seq_y)
    return np.array(x), np.array(y)

def return_rmse(test, predicted):
    rmse = np.sqrt(mean_squared_error(test, predicted))
    print(f"The root mean squared error is {rmse}.")

def process_and_split_multivariate_data(dataset, tstart, tend, mv_features):
    multi_variate_df = dataset.copy()

    multi_variate_df['RSI'] = ta.rsi(multi_variate_df.Close, length=15)
    multi_variate_df['EMAF'] = ta.rsi(multi_variate_df.Close, length=15)
    multi_variate_df['EMAM'] = ta.rsi(multi_variate_df.Close, length=15)
    multi_variate_df['EMAS'] = ta.rsi(multi_variate_df.Close, length=15)
    
    multi_variate_df['Target'] = multi_variate_df['Adj Close'] - dataset.Open
    multi_variate_df['Target'] = multi_variate_df['Target'].shift(-1)
    multi_variate_df.dropna(inplace=True)

    multi_variate_df.drop(['Volume', 'Close'], axis=1, inplace=True)

    multi_variate_df.loc[f"{tstart}":f"{tend}", ['High', 'RSI']].plot(figsize=(16, 4), legend=True)

    multi_variate_df.loc[f"{tstart}":f"{tend}", ['High', 'EMAF', 'EMAM', 'EMAS']].plot(figsize=(16, 4),legend=True)

    feat_columns = ['Open', 'High', 'RSI', 'EMAF', 'EMAM', 'EMAS']
    label_col = ['Target']

    mv_training_set, mv_test_set = train_test_split(multi_variate_df, tstart, tend, feat_columns + label_col)

    x_train = mv_training_set[:, :-1]
    y_train = mv_training_set[:, -1]

    x_test = mv_test_set[:, :-1]
    y_test = mv_test_set[:, -1]

    mv_sc = MinMaxScaler(feature_range=(0, 1))
    x_train = mv_sc.fit_transform(x_train).reshape(-1, 1, mv_features)
    x_test = mv_sc.transform(x_test).reshape(-1, 1, mv_features)

    return x_train, y_train, x_test, y_test, mv_sc
