import numpy as np
import pandas as pd
from flask import Flask, render_template, request
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import requests

app = Flask(__name__)

# Fetch or generate realistic data
def fetch_real_data(symbol='AAPL', start_date='2019-01-01'):
    api_key = 'your_alpha_vantage_api_key'
    url = f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={symbol}&outputsize=full&apikey={api_key}'
    r = requests.get(url)
    data = r.json()
    
    if "Time Series (Daily)" in data:
        df = pd.DataFrame.from_dict(data["Time Series (Daily)"], orient='index').sort_index()
        df = df.rename(columns={
            '1. open': 'Open',
            '2. high': 'High',
            '3. low': 'Low',
            '4. close': 'Close',
            '5. volume': 'Volume'
        }).astype(float)
        df.index = pd.to_datetime(df.index)
        df = df.dropna()  # Ensure no NaN values in the data
        return df
    else:
        raise ValueError("Failed to fetch data from API. Please check your API key and symbol.")

# Calculate MACD
def calculate_macd(df):
    df['EMA12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['EMA26'] = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = df['EMA12'] - df['EMA26']
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
    return df.dropna()

# Calculate RSI
def calculate_rsi(df):
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    return df.dropna()

# Calculate the largest daily gain as a percentage
def calculate_largest_daily_gain(df):
    df['Daily_Change'] = df['Close'].pct_change()
    max_gain = df['Daily_Change'].max()  # Largest daily percentage gain
    return max_gain

# Simple Prediction Method for Each Indicator
def predict_based_on_volume(df):
    return df['Close'].iloc[-1] * (1 + np.log(df['Volume'].iloc[-1] / df['Volume'].mean()))

def predict_based_on_macd(df):
    macd_diff = df['MACD'].iloc[-1] - df['Signal_Line'].iloc[-1]
    return df['Close'].iloc[-1] * (1 + macd_diff / 100)

def predict_based_on_rsi(df):
    rsi_value = df['RSI'].iloc[-1]
    if rsi_value > 70:
        return df['Close'].iloc[-1] * 0.98  # Price expected to drop slightly
    elif rsi_value < 30:
        return df['Close'].iloc[-1] * 1.02  # Price expected to rise slightly
    else:
        return df['Close'].iloc[-1]

# Function to make average prediction
def make_average_prediction(df):
    predictions = []
    try:
        predictions.append(predict_based_on_volume(df))
    except:
        pass
    try:
        predictions.append(predict_based_on_macd(df))
    except:
        pass
    try:
        predictions.append(predict_based_on_rsi(df))
    except:
        pass
    return np.mean(predictions) if predictions else np.nan

# Function to generate predictions for different time frames, regulated by max daily gain
def generate_predictions(df, days=90):
    max_daily_gain = calculate_largest_daily_gain(df)
    current_price = df['Close'].iloc[-1]
    predictions = []
    
    for i in range(1, days + 1):
        prediction = make_average_prediction(df)
        
        # Regulate the prediction by the maximum daily gain
        if i == 1:
            # For the next day, apply the max gain directly
            prediction = min(prediction, current_price * (1 + max_daily_gain))
        else:
            # For future days, compound the max gain
            prediction = min(prediction, predictions[-1] * (1 + max_daily_gain))
        
        if np.isnan(prediction):
            break  # Stop if the prediction cannot be made
        
        predictions.append(prediction)
        
        # Simulate moving forward one day
        df = df.append(pd.Series([prediction], index=['Close']).T, ignore_index=True)
        df.index = pd.date_range(start=df.index[0], periods=len(df), freq='B')
        df = calculate_macd(df)
        df = calculate_rsi(df)
    return predictions

# Function to plot predictions
def plot_predictions(df, predictions, symbol, save_chart=False, chart_path="static/stock_chart.png"):
    plt.figure(figsize=(14,7))
    plt.plot(df['Close'], label='Historical Prices')
    
    # Future dates
    future_dates = pd.date_range(df.index[-1] + timedelta(days=1), periods=len(predictions), freq='B')
    plt.plot(future_dates, predictions, linestyle='--', label='Predictions')
    
    plt.title(f'Stock Price Prediction for {symbol}')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    
    if save_chart:
        plt.savefig(chart_path)
        print(f"Chart saved at {chart_path}")  # Debugging statement
    
    plt.show()

@app.route('/', methods=['GET', 'POST'])
def index():
    symbol = request.form.get("symbol", "AAPL")  # Default to AAPL if no symbol is provided
    print(f"Received symbol: {symbol}")

    df = fetch_real_data(symbol)  # Fetch the latest data
    
    if df is None or df.empty:
        return "Error fetching stock data or insufficient data. Please try again later."
    
    current_price = df['Close'].iloc[-1]
    
    try:
        df = calculate_macd(df)
        df = calculate_rsi(df)
        predictions = generate_predictions(df, 90)
        chart_path = f"static/{symbol}_chart.png"
        plot_predictions(df, predictions, symbol, save_chart=True, chart_path=chart_path)
    except ValueError as e:
        return f"Prediction Error: {str(e)}"
    except Exception as e:
        return f"Unexpected error occurred: {str(e)}"
    
    chart_url = f"{symbol}_chart.png"
    
    # Create a summary including the predictions
    if predictions:
        high_prediction = max(predictions)
        low_prediction = min(predictions)
        summary = f"Stock Analysis for {symbol}: Current price: ${current_price:.2f}\n" \
                  f"Predicted price range for the next day: ${predictions[0]:.2f}\n" \
                  f"Predicted price range for 30 days: ${predictions[29]:.2f}\n" \
                  f"Predicted price range for 60 days: ${predictions[59]:.2f}\n" \
                  f"Predicted price range for 90 days: ${predictions[89]:.2f}\n" \
                  f"Overall high prediction: ${high_prediction:.2f}, low prediction: ${low_prediction:.2f}"
    else:
        summary = "Predictions could not be generated due to insufficient data or an error."

    return render_template(
        'index.html',
        summary=summary,
        symbol=symbol,
        chart_url=chart_url
    )

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

