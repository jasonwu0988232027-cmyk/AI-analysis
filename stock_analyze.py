import streamlit as st
import pandas as pd
import requests
import plotly.graph_objects as go
from datetime import datetime, timedelta

# --- 設定 Finnhub API Key ---
FINNHUB_API_KEY = "你的_FINNHUB_API_KEY" # <--- 請換成你的 Key

st.set_page_config(page_title="專業級 AI 股市分析", layout="wide")
st.title("🏛️ 官方 API 驅動：股市行業情緒與預測")

# --- 1. 獲取股價數據 (替代 yfinance) ---
@st.cache_data(ttl=600)
def get_stock_candles(symbol):
    # Finnhub 使用的是 Unix Timestamp
    end = int(datetime.now().timestamp())
    start = int((datetime.now() - timedelta(days=30)).timestamp())
    
    # 台股需轉換格式，例如 2330.TW -> 2330.TW (Finnhub 支援美股與部分國際股市)
    # 注意：Finnhub 免費版對台股支援度視地區而定，建議先測試美股如 AAPL
    url = f"https://finnhub.io/api/v1/stock/candle?symbol={symbol}&resolution=D&from={start}&to={end}&token={FINNHUB_API_KEY}"
    res = requests.get(url).json()
    
    if res.get('s') == 'ok':
        df = pd.DataFrame({
            'Date': pd.to_datetime(res['t'], unit='s'),
            'Close': res['c'],
            'Open': res['o'],
            'High': res['h'],
            'Low': res['l']
        })
        return df
    return pd.DataFrame()

# --- 2. 獲取新聞情緒分析 (內建 AI 判斷) ---
@st.cache_data(ttl=3600)
def get_sentiment(symbol):
    url = f"https://finnhub.io/api/v1/news-sentiment?symbol={symbol}&token={FINNHUB_API_KEY}"
    res = requests.get(url).json()
    return res

# --- 側邊欄 ---
st.sidebar.header("搜尋設定")
# Finnhub 免費版對美股(AAPL, TSLA)支援最完美，台股格式通常為 2330.TW
stock_symbol = st.sidebar.text_input("輸入股票代碼", "AAPL") 

# --- 主畫面 ---
col1, col2 = st.columns([2, 1])

with col1:
    df = get_stock_candles(stock_symbol)
    if not df.empty:
        st.subheader(f"📈 {stock_symbol} 價格走勢 (K線圖)")
        fig = go.Figure(data=[go.Candlestick(x=df['Date'],
                open=df['Open'], high=df['High'],
                low=df['Low'], close=df['Close'])])
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.error("無法獲取數據，請確認 API Key 或代碼是否正確。")

with col2:
    st.subheader("🤖 官方情緒指標")
    sentiment = get_sentiment(stock_symbol)
    
    if 'sentiment' in sentiment:
        # Finnhub 提供的看漲看跌比例
        bullish = sentiment['sentiment'].get('bullishPercent', 0)
        st.metric("市場看漲情緒", f"{bullish*100:.1f}%")
        
        # 繪製情緒圓餅圖
        st.write("近期新聞情緒分布：")
        st.json({
            "看漲新聞比率": bullish,
            "行業平均情緒": sentiment.get('sectorAverageBullishPercent', 0)
        })
    else:
        st.info("該代碼目前無足夠新聞進行情緒分析。")

# --- AI 行業變動分析 ---
st.divider()
st.subheader("📋 AI 5日行業趨勢預測")
if st.button("綜合分析技術面 + 消息面"):
    if not df.empty and 'sentiment' in sentiment:
        # 這裡結合真實數據生成判斷
        price_change = ((df['Close'].iloc[-1] / df['Close'].iloc[0]) - 1) * 100
        sent_score = sentiment['sentiment'].get('bullishPercent', 0)
        
        analysis = f"""
        **分析報告：**
        1. **技術面**：過去30天股價變動約 {price_change:.2f}%。
        2. **消息面**：Finnhub AI 監測到市場看漲情緒為 {sent_score*100:.1f}%。
        3. **綜合預測**：由於情緒{'高於' if sent_score > 0.5 else '低於'}中值，且股價走勢{'穩定' if price_change > 0 else '疲軟'}，
           預計未來 5 天該行業將會{'延續漲勢' if sent_score > 0.6 else '進入高檔震盪'}。
        """
        st.success(analysis)
    else:
        st.warning("數據不足，無法生成報告。")
