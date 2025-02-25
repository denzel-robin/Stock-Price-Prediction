from datetime import datetime
import tensorflow as tf 
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
import keras
import yfinance as yf
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.preprocessing import MinMaxScaler
from pandas_datareader.data import DataReader
from pandas_datareader import data as pdr

from keras.layers import LSTM, SimpleRNN, Dense, Input
from keras.models import Sequential

from utils import split_sequence 

n_steps = 10
features = 1

def sequence_generation(dataset: pd.DataFrame, sc: MinMaxScaler, model: Sequential, steps_future: int, test_set):
    high_dataset = dataset.iloc[len(dataset) - len(test_set) - n_steps:]["High"]
    high_dataset = sc.transform(high_dataset.values.reshape(-1,1))
    inputs = high_dataset[:n_steps]

    for _ in range(steps_future):
        curr_pred = model.predict(inputs[-n_steps:].reshape(-1, n_steps, features), verbose=0)
        inputs = np.append(inputs, curr_pred, axis=0)

    return sc.inverse_transform(inputs[n_steps:])

def plot_test_vs_predicted_with_date(dates, y_test_original, predicted_stock_price):
    plt.figure(figsize=(12, 6))
    plt.plot(dates, y_test_original, color='blue', label='Actual Stock Price')
    plt.plot(dates, predicted_stock_price, color='red', label='Predicted Stock Price')
    plt.title('Actual vs Predicted Stock Prices')
    plt.xlabel('Date')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
    plt.show()

def train_rnn_model(x_train, y_train, n_steps, features, sc, test_set, dataset, epochs=10, batch_size=32, verbose=1, steps_in_future=25, save_model_path = None):
    model = Sequential([
        Input(shape=(n_steps, features)),
        SimpleRNN(125)
        ])
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mse')
    history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, verbose=verbose)

    inputs = sc.transform(test_set.reshape(-1, 1))

    x_test, y_test = split_sequence(inputs, n_steps)

    x_test = x_test.reshape(-1, n_steps, features)

    predicted_stock_price = model.predict(x_test)
    predicted_stock_price = sc.inverse_transform(predicted_stock_price)

    y_test_original = sc.inverse_transform(y_test.reshape(-1, 1))
    dates = dataset.iloc[-len(y_test):].index

    plot_test_vs_predicted_with_date(dates, y_test_original, predicted_stock_price)
    
    rmse = np.sqrt(mean_squared_error(y_test, predicted_stock_price))
    print(f"The root mean squared error is {rmse}.")

    # results = sequence_generation(dataset, sc, model, steps_in_future, test_set)
    # print("Generated sequence of future predictions: ")
    # print(results)

    if save_model_path:
        model.save(save_model_path)
        print("Model saved successfully.")

    return model

def train_lstm_model(X_train, y_train, n_steps, features, sc, test_set, dataset, epochs, batch_size, verbose, steps_in_future, save_model_path=None):
    model = Sequential()
    model.add(LSTM(units=125, input_shape=(n_steps, features)))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mse')
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=verbose)
    
    inputs = sc.transform(test_set.reshape(-1, 1))

    X_test, y_test = split_sequence(inputs, n_steps)
    X_test = X_test.reshape(-1, n_steps, features)

    predicted_stock_price = model.predict(X_test)
    predicted_stock_price = sc.inverse_transform(predicted_stock_price)
    
    y_test_original = sc.inverse_transform(y_test.reshape(-1, 1))
    dates = dataset.iloc[-len(y_test):].index

    plot_test_vs_predicted_with_date(dates, y_test_original, predicted_stock_price)
    
    rmse = np.sqrt(mean_squared_error(y_test, predicted_stock_price))
    print(f"The root mean squared error is {rmse}.")
    
    
    # results = sequence_generation(dataset, sc, model, steps_in_future, test_set)
    # print("Generated sequence of future predictions:")
    # print(results)
    
    if save_model_path:
        model.save(save_model_path)
        print("Model saved successfully.")
    
    return model

def train_multivariate_lstm(x_train, y_train, x_test, y_test, mv_features, mv_sc, save_model_path=None):
    model_mv = Sequential()
    model_mv.add(LSTM(units=125, input_shape=(1, mv_features)))
    model_mn.add(Dense(units=1))

    model_mv.compile(optimizer="RMSprop", loss="mse")

    history = model_mv.fit(x_train, y_train, epochs=20, batch_size=32, verbose=0)

    predictions = model_mv.predict(x_test)

    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    print(f"The root mean squared error is {rmse}.")

    if save_model_path:
        model_mv.save(save_model_path)
        print("Model saved successfully.")

    return model_mv
