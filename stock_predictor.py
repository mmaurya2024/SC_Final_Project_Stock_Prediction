"""
Stock Price Predictor
=====================
This module defines a StockPredictor class that uses a Long Short-Term Memory
(LSTM) model and a K-Nearest Neighbors (KNN) model to predict future stock
prices based on historical data fetched using the yfinance API.
"""

import yfinance as yf
import numpy as np
import pandas as pd
import math
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import LSTM, Dense
import matplotlib.pyplot as plt
from joblib import dump


class StockPredictor:
    """
    A class used to predict stock prices using LSTM and KNN models.

    Attributes
    ----------
    ticker : str
        Stock ticker symbol (default is "AAPL").
    start_date : str
        Starting date for fetching historical data (default is '2001-01-19').
    end_date : str
        Ending date for fetching historical data (default is '2024-05-10').
    interval : int
        The sequence length for LSTM and KNN models (default is 60).
    lstm_model : keras.models.Sequential
        The LSTM model for predictions.
    knn_model : sklearn.neighbors.KNeighborsRegressor
        The KNN model for predictions.
    scaler : sklearn.preprocessing.MinMaxScaler
        Scaler used for normalizing data.
    train_data : np.ndarray
        Training data for the models.
    test_data : np.ndarray
        Testing data for the models.
    df : pd.DataFrame
        DataFrame holding the historical stock data.
    lstm_predictions : np.ndarray
        LSTM model predictions for the testing data.
    knn_predictions : np.ndarray
        KNN model predictions for the testing data.
    """

    def __init__(self, ticker="AAPL", start_date='2001-01-19', end_date='2024-05-10'):
        """
        Initializes the StockPredictor class with given stock ticker,
        start and end dates.

        Parameters
        ----------
        ticker : str, optional
            Stock ticker symbol (default is "AAPL").
        start_date : str, optional
            Starting date for fetching historical data (default is '2001-01-19').
        end_date : str, optional
            Ending date for fetching historical data (default is '2024-05-10').
        """
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.interval = 60
        self.lstm_model = None
        self.knn_model = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.train_data = None
        self.test_data = None
        self.df = None
        self.lstm_predictions = None
        self.knn_predictions = None

    def fetch_data(self):
        """
        Fetches historical stock data using yfinance and stores it in a DataFrame.
        """
        ticker_data = yf.Ticker(self.ticker)
        self.df = ticker_data.history(start=self.start_date, end=self.end_date, actions=False)
        self.df = self.df.drop(['Open', 'High', 'Volume', 'Low'], axis=1)

    def plot_historical_data(self):
        """
        Plots historical stock data over multiple time ranges.
        """
        plt.figure(figsize=(20, 7))
        plt.title(f"Price of {self.ticker} over the years")
        plt.plot(self.df['2019-09-18':self.end_date], label="Historical Prices")
        plt.ylabel("Price in USD")
        plt.xlabel("Time")
        plt.legend()
        plt.show()

        plt.figure(figsize=(20, 7))
        plt.title(f"Price of {self.ticker} in 2024")
        plt.plot(self.df['2021-01-01':self.end_date], label="Historical Prices")
        plt.ylabel("Price in USD")
        plt.xlabel("Time")
        plt.legend()
        plt.show()

    def preprocess_data(self):
        """
        Preprocesses historical stock data by scaling and splitting into train and test datasets.

        Returns
        -------
        int
            Length of the training data.
        """
        data = self.df.values
        train_len = math.ceil(len(data) * 0.92)
        scaled_data = self.scaler.fit_transform(data)

        self.train_data = scaled_data[:train_len, :]
        self.test_data = scaled_data[train_len - self.interval:, :]
        return train_len

    def create_sequences(self, data):
        """
        Creates input and output sequences for the LSTM model.

        Parameters
        ----------
        data : np.ndarray
            Scaled data used to create sequences.

        Returns
        -------
        tuple
            Tuple containing input sequences (x) and output values (y).
        """
        x, y = [], []
        for i in range(self.interval, len(data)):
            x.append(data[i - self.interval:i, 0])
            y.append(data[i, 0])
        return np.array(x), np.array(y)

    def build_lstm_model(self):
        """
        Builds the LSTM model architecture.
        """
        self.lstm_model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(self.interval, 1)),
            LSTM(50),
            Dense(50),
            Dense(1)
        ])
        self.lstm_model.compile(optimizer="adam", loss="mean_squared_error")

    def build_knn_model(self, n_neighbors=5):
        """
        Builds the KNN model architecture.

        Parameters
        ----------
        n_neighbors : int, optional
            Number of neighbors to use in KNN (default is 5).
        """
        self.knn_model = KNeighborsRegressor(n_neighbors=n_neighbors)

    def train_lstm_model(self, x_train, y_train):
        """
        Trains the LSTM model.

        Parameters
        ----------
        x_train : np.ndarray
            Input training sequences.
        y_train : np.ndarray
            Output training values.
        """
        self.lstm_model.fit(x_train, y_train, batch_size=64, epochs=1)

    def train_knn_model(self, x_train, y_train):
        """
        Trains the KNN model.

        Parameters
        ----------
        x_train : np.ndarray
            Input training sequences.
        y_train : np.ndarray
            Output training values.
        """
        x_train_flat = x_train.reshape((x_train.shape[0], -1))
        self.knn_model.fit(x_train_flat, y_train)

    def make_lstm_predictions(self):
        """
        Makes predictions using the trained LSTM model on test data.

        Returns
        -------
        np.ndarray
            Actual output values for the test data.
        """
        x_test, y_test = self.create_sequences(self.test_data)
        x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))
        predictions = self.lstm_model.predict(x_test)
        self.lstm_predictions = self.scaler.inverse_transform(predictions)

        # Calculate RMSE error for LSTM
        lstm_rmse_error = mean_squared_error(y_test, self.lstm_predictions, squared=False)
        print("LSTM RMSE Error:", lstm_rmse_error)

        return y_test

    def make_knn_predictions(self):
        """
        Makes predictions using the trained KNN model on test data.

        Returns
        -------
        np.ndarray
            Actual output values for the test data.
        """
        x_test, y_test = self.create_sequences(self.test_data)
        x_test_flat = x_test.reshape((x_test.shape[0], -1))
        predictions = self.knn_model.predict(x_test_flat)
        predictions = predictions.reshape(-1, 1)
        self.knn_predictions = self.scaler.inverse_transform(predictions)

        # Calculate RMSE error for KNN
        knn_rmse_error = mean_squared_error(y_test, self.knn_predictions, squared=False)
        print("KNN RMSE Error:", knn_rmse_error)

        return y_test

    def plot_comparison_results(self, train_len):
        """
        Plots the actual and predicted stock prices for both LSTM and KNN models.

        Parameters
        ----------
        train_len : int
            Length of the training data.
        """
        train_data = self.df[:train_len]
        valid_data = self.df[train_len:]
        valid_data = valid_data.iloc[:len(self.lstm_predictions)]
        valid_data['LSTM_Predictions'] = self.lstm_predictions
        valid_data['KNN_Predictions'] = self.knn_predictions

        plt.figure(figsize=(20, 7))
        plt.title("Model Predictions vs Actual Price")
        plt.xlabel("Date", fontsize=18)
        plt.ylabel("Price in USD", fontsize=18)
        plt.plot(train_data['Close'], label='Train')
        plt.plot(valid_data['Close'], label='Actual Price')
        plt.plot(valid_data['LSTM_Predictions'], label='LSTM Prediction')
        plt.plot(valid_data['KNN_Predictions'], label='KNN Prediction')
        plt.legend(loc='upper left', fontsize=15)
        plt.show()

    def forecast_next_days_lstm(self, days=3):
        """
        Forecasts stock prices for the next specified number of days using LSTM.

        Parameters
        ----------
        days : int, optional
            Number of future days to forecast (default is 3).
        """
        last_sequence = self.test_data[-self.interval:, 0].reshape(1, self.interval, 1)
        future_predictions = []

        for _ in range(days):
            next_prediction = self.lstm_model.predict(last_sequence)
            next_prediction_inverse = self.scaler.inverse_transform(next_prediction)
            future_predictions.append(next_prediction_inverse[0][0])
            last_sequence = np.append(last_sequence[:, 1:, :], next_prediction.reshape(1, 1, 1), axis=1)

        for i, value in enumerate(future_predictions):
            print(f"LSTM Opening Price of {self.ticker} Day {i + 1}: {value}")

    def forecast_next_days_knn(self, days=3):
        """
        Forecasts stock prices for the next specified number of days using KNN.

        Parameters
        ----------
        days : int, optional
            Number of future days to forecast (default is 3).
        """
        last_sequence = self.test_data[-self.interval:, 0].reshape(1, -1)
        future_predictions = []

        for _ in range(days):
            next_prediction = self.knn_model.predict(last_sequence)
            next_prediction_inverse = self.scaler.inverse_transform(next_prediction.reshape(-1, 1))
            future_predictions.append(next_prediction_inverse[0][0])
            last_sequence = np.append(last_sequence[:, 1:], next_prediction.reshape(1, 1), axis=1)

        for i, value in enumerate(future_predictions):
            print(f"KNN Opening Price of {self.ticker} Day {i + 1}: {value}")

    def save_models(self, lstm_filename="LSTM_Model.h5", knn_filename="KNN_Model.pkl"):
        """
        Saves the trained models to files.

        Parameters
        ----------
        lstm_filename : str, optional
            Filename to save the LSTM model (default is "LSTM_Model.h5").
        knn_filename : str, optional
            Filename to save the KNN model (default is "KNN_Model.pkl").
        """
        self.lstm_model.save(lstm_filename)
        dump(self.knn_model, knn_filename)

    def run(self):
        """
        Executes the entire stock price prediction process.
        """
        self.fetch_data()
        self.plot_historical_data()
        train_len = self.preprocess_data()

        # Train and predict using LSTM
        x_train, y_train = self.create_sequences(self.train_data)
        x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
        self.build_lstm_model()
        self.train_lstm_model(x_train, y_train)
        self.make_lstm_predictions()

        # Train and predict using KNN
        self.build_knn_model(n_neighbors=5)
        self.train_knn_model(x_train, y_train)
        self.make_knn_predictions()

        # Plot results for comparison
        self.plot_comparison_results(train_len)

        # Forecast future days
        self.forecast_next_days_lstm(3)
        self.forecast_next_days_knn(3)

        # Save models
        self.save_models()
