# Import necessary libraries

import yfinance as yf
"""
yfinance is a Python library that provides a simple and convenient way to download historical market data from Yahoo Finance.
"""

from sklearn.preprocessing import MinMaxScaler
"""
MinMaxScaler is a class in the scikit-learn library that scales features to a specified range. It is commonly used for feature scaling in machine learning models.
"""

import math
"""
The math module provides mathematical functions and constants. It is a standard library in Python.
"""

import numpy as np
"""
NumPy is a powerful library for numerical computing in Python. It provides support for large, multi-dimensional arrays and matrices, along with a collection of mathematical functions to operate on these arrays.
"""

import tensorflow as tf
"""
TensorFlow is an open-source machine learning framework developed by Google. It provides a comprehensive set of tools and libraries for building and deploying machine learning models.
"""

from tensorflow import keras
"""
Keras is a high-level neural networks API written in Python and integrated with TensorFlow. It provides a user-friendly interface for building and training deep learning models.
"""

from tensorflow.keras import layers
"""
The layers module in Keras provides a collection of pre-built layers that can be used to construct neural networks. These layers can be easily stacked together to create complex models.
"""

import matplotlib.pyplot as plt
"""
Matplotlib is a plotting library for Python. It provides a wide variety of functions and classes for creating different types of plots, such as line plots, scatter plots, bar plots, etc.
"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.metrics import mean_squared_error
"""
mean_squared_error is a function in the scikit-learn library that calculates the mean squared error between the predicted and actual values. It is commonly used as a metric for evaluating regression models.
"""

def get_ticker_data(ticker):
    """
    Retrieve the stock data for the given ticker.

    Args:
        ticker (str): The stock ticker symbol.

    Returns:
        yfinance.Ticker: The ticker object for the given stock.
    """
    # Get the ticker object for Apple Inc.
    Apple_Inc = yf.Ticker(ticker)

    # Retrieve the company information
    Apple_Inc.info

    # Assign the ticker object to a variable
    return Apple_Inc

def get_stock_history(ticker, start_date, end_date, actions=False):
    """
    Retrieve the stock history for the given ticker and date range.

    Args:
        ticker (yfinance.Ticker): The ticker object for the stock.
        start_date (str): The start date of the stock history in the format 'YYYY-MM-DD'.
        end_date (str): The end date of the stock history in the format 'YYYY-MM-DD'.
        actions (bool): Whether to include the actions data or not.

    Returns:
        pandas.DataFrame: The stock history dataframe.
    """
    # Retrieve the stock history from the given date range, excluding the actions data
    df = ticker.history(start=start_date, end=end_date, actions=actions)
    return df

def preprocess_data(df):
    """
    Preprocess the stock data by scaling it and splitting it into training and testing sets.

    Args:
        df (pandas.DataFrame): The stock history dataframe.

    Returns:
        tuple: A tuple containing the preprocessed training and testing data, the scaler, and the length of the training data.
    """
    # Drop the unnecessary columns (Open, High, Volume, Low)
    df = df.drop(['Open', 'High', 'Volume', 'Low'], axis=1)

    # Convert the dataframe to a numpy array
    data = df.values

    # Get the length of the data
    print(len(data))

    # Calculate the length of the training data (92% of the total data)
    train_len = math.ceil(len(data) * 0.92)

    # Create a MinMaxScaler object to scale the data between 0 and 1
    min_max_scalar = MinMaxScaler(feature_range=(0, 1))

    # Scale the data
    scaled_data = min_max_scalar.fit_transform(data)

    # Get the length of the scaled data
    print(len(scaled_data))

    # Extract the training data
    train_data = scaled_data[0:train_len, :]
    test_data = scaled_data[train_len:, :]

    return train_data, test_data, min_max_scalar, train_len

def create_sequences(data, seq_length):
    """
    Create input and output sequences for the LSTM model.

    Args:
        data (numpy.ndarray): The preprocessed data.
        seq_length (int): The length of the input sequence.

    Returns:
        tuple: A tuple containing the input (X) and output (y) sequences.
    """
    x = []
    y = []
    for i in range(seq_length, len(data)):
        x.append(data[i - seq_length:i, 0])
        y.append(data[i, 0])
    return np.array(x), np.array(y)

def train_lstm_model(x_train, y_train):
    """
    Train the LSTM model.

    Args:
        x_train (numpy.ndarray): The input training sequences.
        y_train (numpy.ndarray): The output training sequences.

    Returns:
        keras.models.Sequential: The trained LSTM model.
    """
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(LSTM(units=50))
    model.add(Dense(50))
    model.add(Dense(1))
    model.compile(optimizer="adam", loss="mean_squared_error")
    model.fit(x_train, y_train, batch_size=64, epochs=1)
    return model

def make_predictions(model, x_test, min_max_scalar):
    """
    Make predictions using the trained LSTM model.

    Args:
        model (keras.models.Sequential): The trained LSTM model.
        x_test (numpy.ndarray): The input test sequences.
        min_max_scalar (sklearn.preprocessing.MinMaxScaler): The scaler used for data normalization.

    Returns:
        numpy.ndarray: The predicted stock prices.
    """
    predictions = model.predict(x_test)
    predictions = min_max_scalar.inverse_transform(predictions)
    return predictions

def plot_results(train_data, valid_data, predictions):
    """
    Plot the actual and predicted stock prices.

    Args:
        train_data (pandas.DataFrame): The training data.
        valid_data (pandas.DataFrame): The validation data.
        predictions (numpy.ndarray): The predicted stock prices.
    """
    plt.figure(figsize=(20, 7))
    plt.title("Model Prediction vs Actual Price")
    plt.xlabel("Date", fontsize=18)
    plt.ylabel("Price in INR(in Million)", fontsize=18)
    plt.plot(train_data['Close'])
    plt.plot(valid_data['Close'])
    plt.plot(valid_data['predictions'])
    plt.legend(['Train', 'Actual Price', 'Model Prediction'], loc='upper left', fontsize=15)
    plt.show()


def main():
    """
    The main function that runs the entire script.
    """
    # Get the ticker object for Apple Inc.
    ticker = get_ticker_data("AAPL")

    # Retrieve the stock history from 2001-01-19 to 2024-05-06, excluding the actions data
    df = get_stock_history(ticker, '2001-01-19', '2024-05-06', actions=False)

    # Preprocess the data
    train_data, test_data, min_max_scalar, train_len = preprocess_data(df)

    # Create sequences for the LSTM model
    x_train, y_train = create_sequences(train_data, 60)
    x_test, y_test = create_sequences(test_data, 60)

    # Train the LSTM model
    model = train_lstm_model(x_train, y_train)

    # Make predictions on the test data
    predictions = make_predictions(model, x_test, min_max_scalar)

    # Split the original dataframe into training and validation sets
    train_data = df[0:train_len]
    valid_data = df[train_len:]
    valid_data = valid_data.iloc[:len(predictions)]
    valid_data['predictions'] = predictions

    # Plot the actual and predicted prices
    plot_results(train_data, valid_data, predictions)

    # Prepare the data for future prediction
    df_test = get_stock_history(ticker, '2001-01-19', '2024-05-06', actions=False)
    df_test = df_test.drop(['Open', 'High', 'Volume', 'Low'], axis=1)
    test_value = df_test[-60:].values
    test_value = min_max_scalar.transform(test_value)

    # Make predictions for the next 3 days
    future_predictions = []
    for _ in range(3):
        test = test_value[-60:].reshape(1, 60, 1)
        tomorrow_prediction = model.predict(test)
        tomorrow_prediction = min_max_scalar.inverse_transform(tomorrow_prediction)
        future_predictions.append(tomorrow_prediction[0][0])
        test_value = np.append(test_value[1:], tomorrow_prediction[0][0])

    print("Future Predictions:", future_predictions)

    # Save the trained model
    model.save("Apple_Inc.h5")

if __name__ == "__main__":
    main()
