import requests
import numpy as np
import pandas as pd
import os
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt

# Load the API key from environment variables for security
ALPHA_VANTAGE_API_KEY = os.getenv('ALPHA_VANTAGE_API_KEY', 'I9EDHEVIIKBSN1LT')

# Function to fetch historical stock data
def fetch_stock_data(symbol):
    url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY_ADJUSTED&symbol={symbol}&outputsize=full&apikey={ALPHA_VANTAGE_API_KEY}"
    response = requests.get(url)
    data = response.json()
    if "Time Series (Daily)" in data:
        df = pd.DataFrame.from_dict(data["Time Series (Daily)"], orient='index', dtype='float')
        df = df.rename(columns={
            '1. open': 'Open',
            '2. high': 'High',
            '3. low': 'Low',
            '4. close': 'Close',
            '5. adjusted close': 'Adj Close',
            '6. volume': 'Volume'
        })
        df.index = pd.to_datetime(df.index)
        df.sort_index(inplace=True)
        return df
    else:
        print(f"Error fetching data: {data}")
        return None

# Function to calculate technical indicators
def calculate_indicators(df):
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
    df['RSI'] = calculate_rsi(df['Close'], 14)
    df['MACD'], df['MACD_signal'], df['MACD_hist'] = calculate_macd(df['Close'])
    df['Upper_BB'], df['Lower_BB'] = calculate_bollinger_bands(df['Close'])
    return df

def calculate_rsi(series, period):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_macd(series, fast=12, slow=26, signal=9):
    exp1 = series.ewm(span=fast, adjust=False).mean()
    exp2 = series.ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    macd_signal = macd.ewm(span=signal, adjust=False).mean()
    macd_hist = macd - macd_signal
    return macd, macd_signal, macd_hist

def calculate_bollinger_bands(series, window=20, std_dev=2):
    rolling_mean = series.rolling(window).mean()
    rolling_std = series.rolling(window).std()
    upper_band = rolling_mean + (rolling_std * std_dev)
    lower_band = rolling_mean - (rolling_std * std_dev)
    return upper_band, lower_band

# Function to predict the next day's closing price using linear regression
def predict_next_day(df):
    df = df.dropna()  # Ensure no NaN values
    X = np.arange(len(df)).reshape(-1, 1)
    y = df['Close'].values
    model = LinearRegression().fit(X, y)
    next_day_prediction = model.predict([[len(df)]])
    return next_day_prediction[0]

# Function to predict 30, 60, and 90 days using ARIMA
def predict_future(df, days):
    model = ARIMA(df['Close'], order=(5, 1, 0))  # Simple ARIMA model
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=days)
    return forecast

# Function to generate a summary of the analysis
def generate_summary(symbol, next_day, forecast_30, forecast_60, forecast_90):
    summary = (
        f"Stock Analysis for {symbol}:\n"
        f"Predicted price for the next day: ${next_day:.2f}\n"
        f"Predicted price for 30 days: ${forecast_30[-1]:.2f}\n"
        f"Predicted price for 60 days: ${forecast_60[-1]:.2f}\n"
        f"Predicted price for 90 days: ${forecast_90[-1]:.2f}\n\n"
        f"Summary:\n"
        f"The stock is currently showing {'upward' if next_day > df['Close'].iloc[-1] else 'downward'} momentum.\n"
        f"Based on the moving averages and RSI, the stock is considered "
        f"{'overbought' if df['RSI'].iloc[-1] > 70 else 'oversold' if df['RSI'].iloc[-1] < 30 else 'neutral'}.\n"
        f"Bollinger Bands indicate the stock is {'near the upper' if df['Close'].iloc[-1] >= df['Upper_BB'].iloc[-1] else 'near the lower' if df['Close'].iloc[-1] <= df['Lower_BB'].iloc[-1] else 'within'} the band."
    )
    print(summary)

# Function to plot the predictions along with historical data
def plot_predictions(df, symbol, forecast_30, forecast_60, forecast_90):
    plt.figure(figsize=(14,7))
    plt.plot(df['Close'], label='Historical Prices')
    plt.plot(pd.date_range(df.index[-1], periods=31, freq='D')[1:], forecast_30, label='30-Day Prediction', linestyle='--')
    plt.plot(pd.date_range(df.index[-1], periods=61, freq='D')[31:], forecast_60[30:], label='60-Day Prediction', linestyle='--')
    plt.plot(pd.date_range(df.index[-1], periods=91, freq='D')[61:], forecast_90[60:], label='90-Day Prediction', linestyle='--')
    plt.title(f'Stock Price Prediction for {symbol}')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    symbol = input("Enter the stock symbol (e.g., AAPL): ")
    df = fetch_stock_data(symbol)
    if df is not None:
        df = calculate_indicators(df)
        
        # Predict next day
        next_day_price = predict_next_day(df)
        print(f"Predicted price for next day: {next_day_price:.2f}")

        # Predict 30, 60, 90 days
        forecast_30 = predict_future(df, 30)
        forecast_60 = predict_future(df, 60)
        forecast_90 = predict_future(df, 90)

        # Generate Summary
        generate_summary(symbol, next_day_price, forecast_30, forecast_60, forecast_90)

        # Plot Predictions
        plot_predictions(df, symbol, forecast_30, forecast_60, forecast_90)
