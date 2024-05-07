# SC_Final_Project_Stock_Prediction

This project aims to predict the future stock prices of any company using a Long Short-Term Memory (LSTM) neural network model.

## Table of Contents
- [Introduction](#introduction)
- [Data](#data)
- [Methodology](#methodology)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [Usage](#usage)
- [Dependencies](#dependencies)
- [Contributing](#contributing)
- [License](#license)

## Introduction
Stock price prediction is a challenging task due to the complex and volatile nature of financial markets. In this project, we utilize an LSTM model to capture the temporal patterns in the historical stock data and make predictions about future stock prices.

## Data
The data used in this project is obtained from Yahoo Finance, covering the stock prices of a company specified by the ticker symbol. The dataset includes the following features:
- Open
- High
- Low
- Close
- Volume

For this analysis, we focus on the 'Close' column, which represents the closing price of the stock.

## Methodology
The project follows these steps:
1. Data preprocessing: The data is scaled using the MinMaxScaler to normalize the values between 0 and 1.
2. Data splitting: The dataset is split into training and testing sets, with 80% of the data used for training and the remaining 20% for testing.
3. LSTM model creation: An LSTM model is built using the Keras library, with multiple LSTM layers, Dropout layers, and Dense layers.
4. Model training: The LSTM model is trained on the training data using the Adam optimizer and mean squared error loss function.
5. Model evaluation: The trained model is evaluated on the testing data, and the performance metrics (e.g., Mean Squared Error, R-squared) are reported.
6. Future prediction: The trained model is used to predict the stock prices for the next 3 days.

## Model Architecture
The LSTM model used in this project has the following architecture:
```python
model = Sequential()
model.add(LSTM(128, input_shape=(seq_length, 1), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(64))
model.add(Dropout(0.2))
model.add(Dense(32, activation='relu'))
model.add(Dense(1))
model.compile(optimizer=Adam(lr=0.001), loss='mean_squared_error')
```

## Results
The performance of the LSTM model on the testing data is as follows:
- Mean Squared Error: XX.XX
- R-squared: XX.XX

The model is then used to predict the stock prices for the next 3 days, and the results are displayed.

## Usage
To use this project, follow these steps:
1. Clone the repository: `git clone https://github.com/your-username/stock-prediction.git`
2. Install the required dependencies (see the [Dependencies](#dependencies) section).
3. Specify the ticker symbol of the company you want to predict in the code.
4. Run the main script: `python main.py`

The predicted stock prices for the next 3 days will be displayed.

## Dependencies
- Python 3.x
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- TensorFlow
- Keras

## Contributing
Contributions to this project are welcome. If you find any issues or have suggestions for improvements, please feel free to open an issue or submit a pull request.

## License
This project is licensed under the [MIT License](LICENSE).
