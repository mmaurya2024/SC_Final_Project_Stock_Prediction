"""
Main File to Run Stock Price Prediction
=======================================
This file imports the StockPredictor class and executes its prediction process.
"""

from stock_predictor import StockPredictor

if __name__ == "__main__":
    predictor = StockPredictor(ticker="MSFT", start_date='2001-01-19', end_date='2024-05-10')
    predictor.run()
