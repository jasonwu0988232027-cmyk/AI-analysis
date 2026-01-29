import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
from datetime import datetime, timedelta

# 必須在最前面
st.set_page_config(page_title="AI 股市分析系統", layout="wide")

FINNHUB_API_KEY = "d5t2rvhr01qt62ngu1kgd5t2rvhr01qt62ngu1l0"

# --- 數據抓取：增加異常處理與快取 ---
@st.cache_data(ttl=3600)
def get_stock_data(symbol):
    try:
        # 下載 1 年數據用於計算 SMA50
        df = yf.download(symbol, period="1y", interval="1d", progress=False, timeout=10)
        if df.empty: return None
        df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
        return df.reset_index()
    except:
        return None

@st.cache_data(ttl=3600)
def get_sentiment(symbol):
    clean_symbol = symbol.split('.')[0]
    url = f"https://finnhub.io/api/v1/news-sentiment?symbol={clean_symbol}&token={FINNHUB_API_KEY}"
    try:
        r = requests.get(url, timeout=5)
        return r.json() if r.status_code == 200 else None
    except:
        return None

# --- 技術指標計算 ---
def apply_indicators(df):
    d = df.copy()
    # RSI
    delta = d['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    d['RSI'] = 100 - (100 / (1 + (gain / loss)))
    # MA
    d['SMA_20'] = d['Close'].rolling(20).mean()
    d['SMA_50'] = d['Close'].rolling(50).mean()
    # KD
    l14, h14 = d['Low'].rolling(14).min(), d['High'].rolling(14).max()
    d['K'] = 100 * ((d['Close'] - l14) / (h14 - l14))
    d['D'] = d['K'].rolling(3).mean()
    return d.bfill().ffill()

# --- 主介面 ---
st.title("🚀 AI 股市全方位預測")

target = st.sidebar.text_input("股票代碼", "2330.TW").upper()
days = st.sidebar.slider("預測天數", 5, 10, 7)
# 預設不勾選，避免一開始就卡住
load_extra = st.sidebar.checkbox("加載公司基本面 (易卡頓)")

with st.spinner('AI 正在計算中...'):
    raw_df = get_stock_data(target)
    
    if raw_df is not None:
        df = apply_indicators(raw_df)
        sent_res = get_sentiment(target)
        score = sent_res['sentiment'].get('bullishPercent', 0.5) if sent_res and 'sentiment' in sent_res else 0.5
        
        # 10日預測邏輯
        last_p = df['Close'].iloc[-1]
        vol = df['Close'].pct_change().std()
        bias = (score - 0.5) * 0.02 + ((50 - df['RSI'].iloc[-1])/500)
        
        pred_dates = [df['Date'].iloc[-1] + timedelta(days=i) for i in range(1, days+1)]
        pred_prices = [last_p * (1 + np.random.normal(bias, vol)) for _ in range(days)]
        for i in range(1, len(pred_prices)): pred_prices[i] = pred_prices[i-1] * (1 + np.random.normal(bias*0.5, vol))
        
        # 圖表
        fig = go.Figure()
        d_show = df.tail(60)
        fig.add_trace(go.Candlestick(x=d_show['Date'], open=d_show['Open'], high=d_show['High'], low=d_show['Low'], close=d_show['Close'], name="歷史"))
        fig.add_trace(go.Scatter(x=[df['Date'].iloc[-1]] + pred_dates, y=[last_p] + pred_prices, line=dict(color='orange', dash='dot'), name="AI預測"))
        fig.update_layout(height=500, template="plotly_dark", xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)
        
        # 指標卡
        c1, c2, c3 = st.columns(3)
        c1.metric("目前價格", f"{last_p:.2f}")
        c2.metric("預期價格", f"{pred_prices[-1]:.2f}", f"{((pred_prices[-1]-last_p)/last_p)*100:+.2f}%")
        c3.metric("市場情緒", f"{score*100:.1f}%")

        if load_extra:
            st.warning("正在嘗試從 Yahoo 獲取數據，若長時間沒反應請取消勾選...")
            info = yf.Ticker(target).info
            st.write(f"**公司產業**：{info.get('industry', '未知')}")
            st.write(f"**本益比**：{info.get('trailingPE', 'N/A')}")
    else:
        st.error("無法讀取數據，請確認代碼（台股需 .TW，美股直接輸入）")
