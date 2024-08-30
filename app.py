import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from fbprophet import Prophet
from statsmodels.tsa.arima.model import ARIMA
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import matplotlib.pyplot as plt
from flask import Flask, render_template, request
from datetime import datetime

app = Flask(__name__)

# Function to generate synthetic data for demonstration purposes
def generate_synthetic_data(symbol='AAPL', periods=1000):
    dates = pd.date_range(start='2019-01-01', periods=periods, freq='B')
    prices = np.cumsum(np.random.randn(periods)) + 100
    return pd.DataFrame(data={'Date': dates, 'Close': prices}).set_index('Date')

# Function to predict using LSTM
def predict_lstm(df, days=90):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df['Close'].values.reshape(-1, 1))

    def create_dataset(data, time_step=60):
        X, y = [], []
        for i in range(len(data)-time_step-1):
            X.append(data[i:(i+time_step), 0])
            y.append(data[i + time_step, 0])
        return np.array(X), np.array(y)

    time_step = 60
    X, y = create_dataset(scaled_data, time_step)
    X = X.reshape(X.shape[0], X.shape[1], 1)

    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(units=25))
    model.add(Dense(units=1))

    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X, y, batch_size=32, epochs=20, verbose=0)

    last_X = X[-1].reshape(1, time_step, 1)
    predictions = []

    for _ in range(days):
        pred = model.predict(last_X)
        predictions.append(scaler.inverse_transform(pred)[0, 0])
        last_X = np.roll(last_X, -1, axis=1)
        last_X[0, -1, 0] = pred

    return predictions

# Function to predict using Prophet
def predict_prophet(df, days=90):
    df_prophet = df.reset_index().rename(columns={'Date': 'ds', 'Close': 'y'})
    model = Prophet(daily_seasonality=True)
    model.fit(df_prophet)
    future = model.make_future_dataframe(periods=days)
    forecast = model.predict(future)
    return forecast['yhat'][-days:].values

# Function to predict using ARIMA
def predict_arima(df, days=90):
    model = ARIMA(df['Close'], order=(5, 1, 0))
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=days)
    return forecast.values

# Function to predict using XGBoost
def predict_xgboost(df, days=90):
    df['Date'] = pd.to_datetime(df.index)
    df['Date'] = df['Date'].map(pd.Timestamp.toordinal)
    feature_columns = ['Date']
    target = df['Close']
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(df[feature_columns])

    model = XGBRegressor(n_estimators=1000)
    model.fit(features_scaled, target)

    last_row = pd.DataFrame([features_scaled[-1]], columns=feature_columns)
    predictions = []

    for i in range(1, days + 1):
        future_row = last_row.copy()
        future_row['Date'] += i
        pred = model.predict(future_row)
        predictions.append(pred[0])

    return predictions

# Function to predict using LightGBM
def predict_lightgbm(df, days=90):
    df['Date'] = pd.to_datetime(df.index)
    df['Date'] = df['Date'].map(pd.Timestamp.toordinal)
    feature_columns = ['Date']
    target = df['Close']
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(df[feature_columns])

    model = LGBMRegressor(n_estimators=1000)
    model.fit(features_scaled, target)

    last_row = pd.DataFrame([features_scaled[-1]], columns=feature_columns)
    predictions = []

    for i in range(1, days + 1):
        future_row = last_row.copy()
        future_row['Date'] += i
        pred = model.predict(future_row)
        predictions.append(pred[0])

    return predictions

# Fallback prediction system
def predict_stock_prices(df, days=90):
    try:
        return predict_lstm(df, days)
    except:
        print("LSTM failed, trying Prophet...")
    try:
        return predict_prophet(df, days)
    except:
        print("Prophet failed, trying ARIMA...")
    try:
        return predict_arima(df, days)
    except:
        print("ARIMA failed, trying XGBoost...")
    try:
        return predict_xgboost(df, days)
    except:
        print("XGBoost failed, trying LightGBM...")
    try:
        return predict_lightgbm(df, days)
    except:
        raise ValueError("All prediction models failed.")

# Function to plot predictions
def plot_predictions(df, predictions, symbol, save_chart=False, chart_path="static/stock_chart.png"):
    plt.figure(figsize=(14,7))
    plt.plot(df['Close'], label='Historical Prices')
    future_dates = pd.date_range(df.index[-1], periods=len(predictions) + 1, freq='B')[1:]
    plt.plot(future_dates, predictions, label='Predictions', linestyle='--')
    plt.title(f'Stock Price Prediction for {symbol}')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    
    if save_chart:
        plt.savefig(chart_path)
    
    plt.show()

@app.route('/', methods=['GET', 'POST'])
def index():
    symbol = "AAPL"
    df = generate_synthetic_data(symbol)

    if df is None or df.empty:
        return "Error fetching stock data or insufficient data. Please try again later."
    
    try:
        predictions = predict_stock_prices(df, 90)
    except ValueError as e:
        return f"Prediction Error: {str(e)}"
    
    plot_predictions(df, predictions, symbol, save_chart=True, chart_path=f"static/{symbol}_chart_combination.png")
    
    return render_template('index.html', summary=f"Predictions for {symbol}", symbol=symbol)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
