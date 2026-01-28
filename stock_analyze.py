import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
from datetime import datetime, timedelta

# --- API 設定 ---
FINNHUB_API_KEY = "d5t2rvhr01qt62ngu1kgd5t2rvhr01qt62ngu1l0"

st.set_page_config(page_title="AI 股市預測專家", layout="wide")

# --- 1. 數據獲取 ---
@st.cache_data(ttl=3600)
def get_stock_data(symbol):
    try:
        df = yf.download(symbol, period="3mo", interval="1d", progress=False)
        if df.empty: return None
        df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
        return df.reset_index()
    except:
        return None

# --- 2. 模擬未來 10 天預測邏輯 ---
def predict_future_prices(df, sentiment_score, days=10):
    # 基於最後一天的收盤價
    last_price = df['Close'].iloc[-1]
    last_date = df['Date'].iloc[-1]
    
    # 計算近期波動率作為預測基礎
    volatility = df['Close'].pct_change().std() 
    # 情緒影響因子 (將 0~1 的情緒轉化為 -1% ~ +1% 的每日偏移)
    bias = (sentiment_score - 0.5) * 0.02 
    
    future_dates = [last_date + timedelta(days=i) for i in range(1, days + 1)]
    future_prices = []
    
    current_price = last_price
    for _ in range(days):
        # 簡單隨機漫步模型 + 情緒偏差
        change_pct = np.random.normal(bias, volatility)
        current_price *= (1 + change_pct)
        future_prices.append(current_price)
        
    return pd.DataFrame({'Date': future_dates, 'Close': future_prices})

# --- 3. Finnhub 情緒抓取 ---
@st.cache_data(ttl=3600)
def get_finnhub_sentiment(symbol):
    clean_symbol = symbol.split('.')[0]
    url = f"https://finnhub.io/api/v1/news-sentiment?symbol={clean_symbol}&token={FINNHUB_API_KEY}"
    try:
        res = requests.get(url).json()
        return res
    except: return None

# --- UI 介面 ---
st.title("📈 AI 股市趨勢分析與 10 日走勢預測")

target_stock = st.sidebar.text_input("輸入股票代碼 (例: 2330.TW)", "2330.TW").upper()
forecast_days = st.sidebar.slider("預測天數", 5, 10, 7)

df = get_stock_data(target_stock)
sentiment_data = get_finnhub_sentiment(target_stock)
sent_score = sentiment_data['sentiment'].get('bullishPercent', 0.5) if sentiment_data and 'sentiment' in sentiment_data else 0.5

if df is not None:
    # 執行預測
    future_df = predict_future_prices(df, sent_score, days=forecast_days)
    
    # 繪製圖表
    st.subheader(f"📊 {target_stock} 歷史走勢與 AI 預期路徑")
    
    fig = go.Figure()

    # 歷史 K 線
    fig.add_trace(go.Candlestick(
        x=df['Date'], open=df['Open'], high=df['High'],
        low=df['Low'], close=df['Close'], name="歷史數據"
    ))

    # 預測走勢 (虛線)
    # 連接歷史最後一天與預測第一天
    connect_df = pd.concat([df.tail(1)[['Date', 'Close']], future_df])
    
    fig.add_trace(go.Scatter(
        x=connect_df['Date'], y=connect_df['Close'],
        mode='lines+markers',
        line=dict(color='orange', width=3, dash='dot'),
        name=f"AI 預測未來 {forecast_days} 日"
    ))

    fig.update_layout(xaxis_rangeslider_visible=False, height=600, template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)

    # --- 分析面板 ---
    col1, col2 = st.columns(2)
    with col1:
        st.write("### 📉 數據摘要")
        st.write(f"當前價格: `{df['Close'].iloc[-1]:.2f}`")
        st.write(f"預計 {forecast_days} 日後價格: `{future_df['Close'].iloc[-1]:.2f}`")
        
    with col2:
        st.write("### 🧠 AI 預測依據")
        st.write(f"市場情緒權重: `{sent_score:.2f}`")
        st.write(f"技術面波動率: `{df['Close'].pct_change().std():.4f}`")

else:
    st.error("無法獲取數據，請檢查代碼格式。")
