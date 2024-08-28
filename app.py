import requests
import numpy as np
import pandas as pd
from flask import Flask, render_template, request
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from datetime import datetime  # Import for current date

app = Flask(__name__)

# Load the API key from config.txt
def load_api_key():
    with open('config.txt') as f:
        for line in f:
            if 'ALPHA_VANTAGE_API_KEY' in line:
                return line.split('=')[1].strip()
    return None

ALPHA_VANTAGE_API_KEY = load_api_key()

# Function to fetch historical stock data
def fetch_stock_data(symbol, api_key):
    url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={symbol}&outputsize=full&apikey={api_key}"
    response = requests.get(url)
    data = response.json()

    # Check if the data is available
    if "Time Series (Daily)" in data:
        df = pd.DataFrame.from_dict(data["Time Series (Daily)"], orient='index', dtype='float')
        df = df.rename(columns={
            '1. open': 'Open',
            '2. high': 'High',
            '3. low': 'Low',
            '4. close': 'Close',
            '5. volume': 'Volume'
        })
        df.index = pd.to_datetime(df.index)
        df.sort_index(inplace=True)
        df = df.asfreq('B')  # Set frequency to business days

        # Filter the data to include only the last 5 years
        start_date = pd.Timestamp.today() - pd.DateOffset(years=5)
        df = df[df.index >= start_date]

        # Debugging print to check data
        print(f"Fetched Data for {symbol} from {start_date.date()} to {df.index[-1].date()}:\n", df.head())
        return df
    else:
        print(f"Error fetching data: {data}")
        return None

# Function to calculate indicators for a specific combination
def calculate_indicators(df, combination):
    if combination == 1:
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
        df['RSI'] = calculate_rsi(df['Close'], 14)
        df['MACD'], df['MACD_signal'], df['MACD_hist'] = calculate_macd(df['Close'])
        df['Upper_BB'], df['Lower_BB'] = calculate_bollinger_bands(df['Close'])
    elif combination == 2:
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
        df['MACD'], df['MACD_signal'], df['MACD_hist'] = calculate_macd(df['Close'])
        df['Stochastic'] = calculate_stochastic(df)
    elif combination == 3:
        df['RSI'] = calculate_rsi(df['Close'], 14)
        df['MACD'], df['MACD_signal'], df['MACD_hist'] = calculate_macd(df['Close'])
        df['OBV'] = calculate_obv(df)
    elif combination == 4:
        df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
        df['Upper_BB'], df['Lower_BB'] = calculate_bollinger_bands(df['Close'])
        df['Stochastic'] = calculate_stochastic(df)
    
    df.dropna(inplace=True)  # Drop NaN values to ensure model compatibility
    return df

# RSI Calculation
def calculate_rsi(series, period):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

# MACD Calculation
def calculate_macd(series, fast=12, slow=26, signal=9):
    exp1 = series.ewm(span=fast, adjust=False).mean()
    exp2 = series.ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    macd_signal = macd.ewm(span=signal, adjust=False).mean()
    macd_hist = macd - macd_signal
    return macd, macd_signal, macd_hist

# Bollinger Bands Calculation
def calculate_bollinger_bands(series, window=20, std_dev=2):
    rolling_mean = series.rolling(window).mean()
    rolling_std = series.rolling(window).std()
    upper_band = rolling_mean + (rolling_std * std_dev)
    lower_band = rolling_mean - (rolling_std * std_dev)
    return upper_band, lower_band

# Stochastic Calculation
def calculate_stochastic(df, k_period=14, d_period=3):
    df['L14'] = df['Low'].rolling(window=k_period).min()
    df['H14'] = df['High'].rolling(window=k_period).max()
    df['%K'] = 100 * ((df['Close'] - df['L14']) / (df['H14'] - df['L14']))
    df['%D'] = df['%K'].rolling(window=d_period).mean()
    return df['%D']

# OBV Calculation
def calculate_obv(df):
    obv = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()
    return obv

# Enhance features by adding more indicators
def enhance_features(df):
    # Example of additional features
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['EMA_100'] = df['Close'].ewm(span=100, adjust=False).mean()
    df['ATR'] = df['High'] - df['Low']  # Simple example of an Average True Range
    df['Volatility'] = df['Close'].rolling(window=50).std()  # Rolling volatility
    
    df.dropna(inplace=True)  # Drop NaN values
    return df

# Function to predict stock prices using a gradient boosting model
def predict_future(df, days):
    df['Date'] = pd.to_datetime(df.index)
    df['Date'] = df['Date'].map(pd.Timestamp.toordinal)

    # Enhance features
    df = enhance_features(df)

    # Define feature columns and target
    feature_columns = [col for col in df.columns if col not in ['Open', 'High', 'Low', 'Volume', 'Close']]
    features = df[feature_columns]
    target = df['Close']

    # Scale the features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # Train a Gradient Boosting model with tuned hyperparameters
    model = GradientBoostingRegressor(n_estimators=500, max_depth=4, learning_rate=0.01, random_state=42)
    model.fit(features_scaled, target)

    # Predict the future
    last_row = pd.DataFrame([features.iloc[-1].values], columns=feature_columns)
    last_row_scaled = scaler.transform(last_row)
    predictions = []

    for i in range(1, days + 1):
        future_row = last_row.copy()
        future_row['Date'] += i  # Increment the date
        future_row_scaled = scaler.transform(future_row)
        pred = model.predict(future_row_scaled)
        predictions.append(pred[0])

    return predictions

# Function to generate a summary of the analysis
def generate_summary(symbol, next_day, forecast_30, forecast_60, forecast_90, df):
    current_date = datetime.now().strftime("%Y-%m-%d")  # Get the current date
    summary = f"Stock Analysis for {symbol} on {current_date}:\n"  # Include the date in the summary
    summary += f"Predicted price for the next day: ${next_day:.2f}\n"
    summary += f"Predicted price for 30 days: ${forecast_30[-1]:.2f}\n"
    summary += f"Predicted price for 60 days: ${forecast_60[-1]:.2f}\n"
    summary += f"Predicted price for 90 days: ${forecast_90[-1]:.2f}\n\n"
    
    summary += "Summary:\n"
    summary += f"The stock is currently showing {'upward' if next_day > df['Close'].iloc[-1] else 'downward'} momentum.\n"
    
    if 'RSI' in df.columns:
        summary += f"Based on the RSI, the stock is considered "
        summary += f"{'overbought' if df['RSI'].iloc[-1] > 70 else 'oversold' if df['RSI'].iloc[-1] < 30 else 'neutral'}.\n"
    
    if 'Upper_BB' in df.columns and 'Lower_BB' in df.columns:
        summary += f"Bollinger Bands indicate the stock is "
        summary += f"{'near the upper' if df['Close'].iloc[-1] >= df['Upper_BB'].iloc[-1] else 'near the lower' if df['Close'].iloc[-1] <= df['Lower_BB'].iloc[-1] else 'within'} the band.\n"

    print(summary)
    return summary

# Function to plot the predictions along with historical data and save the chart
def plot_predictions(df, symbol, forecast_30, forecast_60, forecast_90, save_chart=False, chart_path="static/stock_chart.png"):
    plt.figure(figsize=(14,7))
    plt.plot(df['Close'], label='Historical Prices')
    plt.plot(pd.date_range(df.index[-1], periods=31, freq='B')[1:], forecast_30, label='30-Day Prediction', linestyle='--')
    plt.plot(pd.date_range(df.index[-1], periods=61, freq='B')[31:], forecast_60[30:], label='60-Day Prediction', linestyle='--')
    plt.plot(pd.date_range(df.index[-1], periods=91, freq='B')[61:], forecast_90[60:], label='90-Day Prediction', linestyle='--')
    plt.title(f'Stock Price Prediction for {symbol}')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    
    if save_chart:
        plt.savefig(chart_path)
        print(f"Stock chart saved as {chart_path}")
    
    plt.show()

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        symbol = request.form['symbol'].upper()  # Get the stock symbol from the form
    else:
        symbol = "TSLA"  # Default stock symbol

    df = fetch_stock_data(symbol, ALPHA_VANTAGE_API_KEY)
    
    if df is None:
        return "Error fetching stock data. Please try again later."
    
    # Analyze using a specific indicator combination
    combination = 1  # Choose an indicator combination
    df_combined = calculate_indicators(df, combination)
    
    next_day_price = predict_future(df_combined, 1)[-1]
    forecast_30 = predict_future(df_combined, 30)
    forecast_60 = predict_future(df_combined, 60)
    forecast_90 = predict_future(df_combined, 90)
    
    # Generate Summary
    summary = generate_summary(symbol, next_day_price, forecast_30, forecast_60, forecast_90, df_combined)
    
    # Plot Predictions and save the chart
    plot_predictions(df_combined, symbol, forecast_30, forecast_60, forecast_90, save_chart=True, chart_path=f"static/{symbol}_chart_combination_{combination}.png")
    
    # Return the summary in the HTML template
    return render_template('index.html', summary=summary, symbol=symbol, combination=combination)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
