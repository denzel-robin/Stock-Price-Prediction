from utils import *
from train import *
import yfinance as yf 
import numpy as np 
from projectpro import model_snapshot, checkpoint


dataset = yf.download('NVDA', start='2012-01-01', end=datetime.now().strftime('%Y-%m-%d'))

checkpoint('34db30')
print("Data Loaded")

tstart = 2012
tend = 2023

training_set, test_set = train_test_split(dataset, tstart, tend)

sc = MinMaxScaler(feature_range=(0,1))
training_set = training_set.reshape(-1,1)
training_set_scaled = sc.fit_transform(training_set)

n_steps = 10
features = 1
x_train,y_train = split_sequence(training_set_scaled, n_steps)

x_train = x_train.reshape(x_train.shape[0],x_train.shape[1], features)

model_rnn = train_rnn_model(x_train, y_train, n_steps, features, sc, test_set, dataset, epochs=10, batch_size=36, verbose=1, steps_in_future=25, save_model_path="output/model_rnn.h5")
model_snapshot("34db30")

model_lstm = train_lstm_model(x_train, y_train, n_steps, features, sc, test_set, dataset, epochs=10, batch_size=36, verbose=1, steps_in_future=25, save_model_path="output/model_lstm.h5")

# mv_features = 6
#
# x_train, y_train, x_test, y_test, mv_sc = process_and_split_multivariate_data(dataset, tstart, tend, mv_features)
#
# model_mv = train_multivariate_lstm(x_train,y_train,x_test,y_test, mv_features,mv_sc, save_model_path="output/model_mv_lstm.h5")
# model_snapshot("34db30")
