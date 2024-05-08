# Stock Price Prediction using LSTM

This project aims to predict future stock prices using a Long Short-Term Memory (LSTM) neural network model. The model is trained on historical stock data and can make predictions for the next few days.

## Table of Contents

- [Introduction](#introduction)
- [Dependencies](#dependencies)
- [Data](#data)
- [Methodology](#methodology)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Introduction

Stock price prediction is a challenging task due to the complex and dynamic nature of financial markets. This project utilizes an LSTM model, a type of recurrent neural network, to capture the temporal patterns and dependencies in the stock data. By analyzing the historical stock prices, the model learns to make accurate predictions for future prices.

## Dependencies

The following dependencies are required to run this project:

- Python 3.x
- yfinance
- scikit-learn
- NumPy
- TensorFlow
- Keras
- Matplotlib

## Data

The project uses historical stock data fetched from Yahoo Finance using the `yfinance` library. The data includes the following features:

- Open
- High
- Low
- Close
- Volume

For the analysis, only the 'Close' column, representing the closing price of the stock, is used.

## Methodology

The project follows these steps:

1. **Data Preprocessing**: The data is scaled using the `MinMaxScaler` from scikit-learn to normalize the values between 0 and 1. The dataset is then split into training and testing sets, with 92% of the data used for training and the remaining 8% for testing.

2. **Sequence Creation**: Input and output sequences are created from the preprocessed data for the LSTM model. The input sequences have a length of 60 days, and the output sequences are the corresponding next day's closing price.

3. **LSTM Model Training**: An LSTM model is built using the Keras library, with multiple LSTM layers, Dropout layers, and Dense layers. The model is trained on the input and output sequences using the Adam optimizer and mean squared error loss function.

4. **Model Evaluation**: The trained LSTM model is evaluated on the testing data, and the performance metrics, such as Mean Squared Error (MSE) and R-squared, are calculated.

5. **Future Prediction**: The trained model is used to predict the stock prices for the next three days, using the most recent 60 days of data as input.

6. **Visualization**: The actual and predicted stock prices are visualized using Matplotlib, allowing for visual comparison and analysis.

7. **Model Saving**: The trained LSTM model is saved for future use.

## Model Architecture

The LSTM model used in this project has the following architecture:

```python
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(LSTM(units=50))
model.add(Dense(50))
model.add(Dense(1))
model.compile(optimizer="adam", loss="mean_squared_error")
```

The model consists of two LSTM layers, followed by two Dense layers. The first LSTM layer has 50 units and returns the entire sequence, while the second LSTM layer has 50 units and does not return the sequence. The first Dense layer has 50 units, and the final Dense layer has a single unit for the output prediction.

## Results

The performance of the LSTM model on the testing data is evaluated using the Mean Squared Error (MSE) and R-squared metrics. The specific values for these metrics will depend on the dataset and the model's training process.

The model is then used to predict the stock prices for the next three days, and the results are printed to the console.

## Usage

To use this project, follow these steps:

1. Clone the repository:

```
git clone https://github.com/your-username/stock-price-prediction.git
```

2. Install the required dependencies (see the [Dependencies](#dependencies) section).

3. Navigate to the project directory:

```
cd stock-price-prediction
```

4. Run the main script:

```
python main.py
```

The script will fetch the historical stock data for Apple Inc. (AAPL) by default. You can modify the ticker symbol in the `main()` function to predict prices for a different stock.

The script will preprocess the data, train the LSTM model, make predictions on the test data, and visualize the actual and predicted prices. Additionally, it will predict the stock prices for the next three days and print them to the console.

## Contributing

Contributions to this project are welcome. If you find any issues or have suggestions for improvements, please open an issue or submit a pull request.

## License

This project is licensed under the [MIT License](LICENSE).

Citations:
[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/10527565/b9bd3547-7c34-47f0-9420-a40716833374/paste.txt
