# Stock Price Predictor

This project uses a Long Short-Term Memory (LSTM) model and a K-Nearest Neighbors (KNN) model to predict future stock prices based on historical data fetched using the yfinance API. It encapsulates the entire process in an object-oriented manner. The project includes two main files:

1. **`stock_predictor.py`**: Contains the `StockPredictor` class definition.
2. **`main.py`**: Imports the `StockPredictor` class and executes its prediction process.

## Features

- Fetches historical stock data using `yfinance`.
- Predicts future stock prices using two models:
  - Long Short-Term Memory (LSTM)
  - K-Nearest Neighbors (KNN)
- Compares the performance of both models using graphs and numerical values.
- Saves the trained models for later use.

## Models Used

### Long Short-Term Memory (LSTM)

LSTM is a type of recurrent neural network (RNN) capable of learning long-term dependencies. Here, it is used to predict future stock prices based on past historical data.

- **Architecture**:
  - Two LSTM layers
  - Dense layers for prediction

```python
def build_lstm_model(self):
    self.lstm_model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(self.interval, 1)),
        LSTM(50),
        Dense(50),
        Dense(1)
    ])
    self.lstm_model.compile(optimizer="adam", loss="mean_squared_error")

K-Nearest Neighbors (KNN)
KNN is a simple, non-parametric model used for classification and regression. Here, it is used to predict future stock prices based on the historical data points that are closest to the target.

Architecture:
Uses the KNeighborsRegressor from sklearn with customizable n_neighbors.

def build_knn_model(self, n_neighbors=5):
    self.knn_model = KNeighborsRegressor(n_neighbors=n_neighbors)

Dependencies
The project requires the following Python libraries:

yfinance
numpy
pandas
sklearn
keras
matplotlib
joblib
You can install all dependencies using pip:

pip install -r requirements.txt
