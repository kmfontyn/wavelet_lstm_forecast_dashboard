# streamlit_forecast_app.py
import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import yfinance as yf
import pywt
from numpy.linalg import svd
from statsmodels.tsa.stattools import pacf
from streamlit_autorefresh import st_autorefresh

# Auto-refresh every 5 minutes
st_autorefresh(interval=5 * 60 * 1000, key="datarefresh")

st.set_page_config(layout="wide")
st.title("📈 Ticker Forecast Dashboard")

# ---- Helper Functions ----
def load_and_prepare_data(ticker, period="60d", interval="5m"):
    data = yf.download(ticker, period=period, interval=interval)
    data = data[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)
    close_pacf = pacf(data['Close'].dropna(), nlags=40)
    optimal_lag = np.argmax(close_pacf < 0.05)
    optimal_lag = optimal_lag if optimal_lag != 0 else 10

    def create_sequences(data, time_steps):
        X, y = [], []
        for i in range(len(data) - time_steps):
            X.append(data[i:i + time_steps])
            y.append(data[i + time_steps, 3])
        return np.array(X), np.array(y)

    X, y = create_sequences(data_scaled, optimal_lag)
    return data, data_scaled, scaler, optimal_lag, X, y

def wavelet_denoise(data, lag):
    denoised = []
    for i in range(data.shape[1]):
        coeffs = pywt.wavedec(data[:, i], 'sym4', level=3)
        for j in range(1, len(coeffs)):
            coeffs[j] = np.zeros_like(coeffs[j])
        smoothed = pywt.waverec(coeffs, 'sym4')[:len(data)]
        denoised.append(smoothed)
    denoised_data = np.stack(denoised, axis=1)

    def create_sequences(data, time_steps):
        X, y = [], []
        for i in range(len(data) - time_steps):
            X.append(data[i:i + time_steps])
            y.append(data[i + time_steps, 3])
        return np.array(X), np.array(y)

    return create_sequences(denoised_data, lag)

def ssa_denoise(data, lag):
    denoised = []
    for i in range(data.shape[1]):
        series = data[:, i]
        N, window = len(series), 60
        K = N - window + 1
        X = np.column_stack([series[j:j+K] for j in range(window)])
        U, s, Vt = svd(X, full_matrices=False)
        X_hat = sum(s[r] * np.outer(U[:, r], Vt[r, :]) for r in range(10))
        smoothed = np.mean(X_hat, axis=1)[:N]
        denoised.append(smoothed)
    data_ssa = np.stack(denoised, axis=1)

    def create_sequences(data, time_steps):
        X, y = [], []
        for i in range(len(data) - time_steps):
            X.append(data[i:i + time_steps])
            y.append(data[i + time_steps, 3])
        return np.array(X), np.array(y)

    return create_sequences(data_ssa, lag)

def build_lstm(X, y):
    model = Sequential([
        LSTM(150, return_sequences=True, input_shape=(X.shape[1], X.shape[2])),
        LSTM(50),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, epochs=10, batch_size=32, verbose=0)
    return model

def evaluate(model, X, y, scaler, actual_close):
    y_pred = model.predict(X)
    y_pred = scaler.inverse_transform(np.hstack([np.zeros((len(y_pred), 3)), y_pred, np.zeros((len(y_pred), 1))]))[:, 3]
    y_true = scaler.inverse_transform(np.hstack([np.zeros((len(y), 3)), y.reshape(-1,1), np.zeros((len(y), 1))]))[:, 3]
    residuals = y_true - y_pred
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    sdape = np.std(np.abs((y_true - y_pred) / y_true)) * 100
    forecast = y_pred[-1]
    accuracy = 100 - abs(forecast - actual_close) / actual_close * 100
    confidence = max(0, min(100 * (1 - np.std(residuals) / (np.mean(y_true) + 1e-8)), 100))
    return y_pred, y_true, dict(RMSE=rmse, MAE=mae, MAPE=mape, SDAPE=sdape, Forecast=forecast, Accuracy=accuracy, Confidence=confidence)

# ---- Interface ----
ticker_input = st.text_input("Enter tickers (comma-separated):")
if ticker_input:
    tickers = [t.strip().upper() for t in ticker_input.split(',') if t.strip()]
    selected = st.selectbox("Select ticker to view", tickers)

    data, scaled, scaler, lag, X, y = load_and_prepare_data(selected)
    close_price = data['Close'].iloc[-1]
    X_wt, y_wt = wavelet_denoise(scaled, lag)
    X_ssa, y_ssa = ssa_denoise(scaled, lag)

    model_lstm = build_lstm(X, y)
    model_wt = build_lstm(X_wt, y_wt)
    model_ssa = build_lstm(X_ssa, y_ssa)

    pred_lstm, true_lstm, eval_lstm = evaluate(model_lstm, X, y, scaler, close_price)
    pred_wt, true_wt, eval_wt = evaluate(model_wt, X_wt, y_wt, scaler, close_price)
    pred_ssa, true_ssa, eval_ssa = evaluate(model_ssa, X_ssa, y_ssa, scaler, close_price)

    st.header(f"{selected} Forecast Results")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("SSA-LSTM")
        fig, ax = plt.subplots()
        ax.plot(true_ssa, label='True')
        ax.plot(pred_ssa, label='Predicted')
        ax.legend()
        st.pyplot(fig)

    with col2:
        st.subheader("WT-LSTM")
        fig, ax = plt.subplots()
        ax.plot(true_wt, label='True')
        ax.plot(pred_wt, label='Predicted')
        ax.legend()
        st.pyplot(fig)

    st.subheader("Vanilla LSTM")
    fig, ax = plt.subplots()
    ax.plot(true_lstm, label='True')
    ax.plot(pred_lstm, label='Predicted')
    ax.legend()
    st.pyplot(fig)

    st.subheader("Evaluation Metrics")
    df_eval = pd.DataFrame({
        'Model': ['WT-LSTM', 'SSA-LSTM', 'LSTM'],
        'RMSE': [eval_wt['RMSE'], eval_ssa['RMSE'], eval_lstm['RMSE']],
        'MAE': [eval_wt['MAE'], eval_ssa['MAE'], eval_lstm['MAE']],
        'MAPE (%)': [eval_wt['MAPE'], eval_ssa['MAPE'], eval_lstm['MAPE']],
        'SDAPE (%)': [eval_wt['SDAPE'], eval_ssa['SDAPE'], eval_lstm['SDAPE']],
        'Current Close': [close_price] * 3,
        'Next Forecast': [eval_wt['Forecast'], eval_ssa['Forecast'], eval_lstm['Forecast']],
        'Forecast Confidence (%)': [eval_wt['Confidence'], eval_ssa['Confidence'], eval_lstm['Confidence']],
        'Accuracy (%)': [eval_wt['Accuracy'], eval_ssa['Accuracy'], eval_lstm['Accuracy']]
    })
    st.dataframe(df_eval.round(3))
