import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
from datetime import datetime, timedelta

# --- 頁面配置（必須在最前面）---
st.set_page_config(page_title="AI 股市預測專家 Pro", layout="wide")

# --- API 設定 ---
FINNHUB_API_KEY = "d5t2rvhr01qt62ngu1kgd5t2rvhr01qt62ngu1l0"

# ==================== 1. 數據獲取模組 ====================

@st.cache_data(ttl=3600)
def get_stock_data(symbol, period="1y"):
    """獲取歷史股價數據"""
    try:
        # 使用 download 較 history 穩定且快
        df = yf.download(symbol, period=period, interval="1d", progress=False)
        if df.empty: 
            return None
        # 處理 yfinance 可能產生的多層索引
        df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
        return df.reset_index()
    except Exception as e:
        st.error(f"股價獲取失敗: {str(e)}")
        return None

@st.cache_data(ttl=86400)  # 基本面一天更新一次即可
def get_fundamental_data(symbol):
    """獲取基本面數據（這部分最耗時，故使用長效快取）"""
    try:
        ticker = yf.Ticker(symbol)
        # 僅在必要時調用 info
        info = ticker.info
        if not info: return None
        
        return {
            'PE Ratio': info.get('trailingPE', 'N/A'),
            'Forward PE': info.get('forwardPE', 'N/A'),
            'Price to Book': info.get('priceToBook', 'N/A'),
            'Dividend Yield': info.get('dividendYield', 'N/A'),
            'ROE': info.get('returnOnEquity', 'N/A'),
            'Market Cap': info.get('marketCap', 'N/A'),
        }
    except:
        return None

@st.cache_data(ttl=3600)
def get_finnhub_sentiment(symbol):
    """從 Finnhub 獲取 AI 情緒分析"""
    clean_symbol = symbol.split('.')[0]
    url = f"https://finnhub.io/api/v1/news-sentiment?symbol={clean_symbol}&token={FINNHUB_API_KEY}"
    try:
        res = requests.get(url, timeout=5).json()
        return res
    except:
        return None

# ==================== 2. 技術指標計算 ====================

def calculate_indicators(df):
    """計算 RSI, MACD, 布林通道, KD"""
    df_copy = df.copy()
    
    # RSI
    delta = df_copy['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    df_copy['RSI'] = 100 - (100 / (1 + (gain / loss)))
    
    # MA & MACD
    df_copy['SMA_20'] = df_copy['Close'].rolling(window=20).mean()
    df_copy['SMA_50'] = df_copy['Close'].rolling(window=50).mean()
    ema12 = df_copy['Close'].ewm(span=12, adjust=False).mean()
    ema26 = df_copy['Close'].ewm(span=26, adjust=False).mean()
    df_copy['MACD'] = ema12 - ema26
    df_copy['MACD_Signal'] = df_copy['MACD'].ewm(span=9, adjust=False).mean()
    df_copy['MACD_Diff'] = df_copy['MACD'] - df_copy['MACD_Signal']
    
    # 布林通道
    std = df_copy['Close'].rolling(window=20).std()
    df_copy['BB_High'] = df_copy['SMA_20'] + (std * 2)
    df_copy['BB_Low'] = df_copy['SMA_20'] - (std * 2)
    
    # KD
    low_14 = df_copy['Low'].rolling(window=14).min()
    high_14 = df_copy['High'].rolling(window=14).max()
    df_copy['K'] = 100 * ((df_copy['Close'] - low_14) / (high_14 - low_14))
    df_copy['D'] = df_copy['K'].rolling(window=3).mean()
    
    # 填充缺失值 (相容新版 Pandas)
    return df_copy.bfill().ffill()

# ==================== 3. AI 預測模型 ====================

def predict_future(df, sentiment_score, days=10):
    """蒙地卡羅隨機漫步 + 技術面權重預測"""
    last_price = df['Close'].iloc[-1]
    last_date = df['Date'].iloc[-1]
    volatility = df['Close'].pct_change().std()
    
    # 根據技術面給予偏移量 (Bias)
    rsi = df['RSI'].iloc[-1]
    rsi_bias = (50 - rsi) / 100 * 0.01  # RSI 低於 50 視為反彈機會
    sent_bias = (sentiment_score - 0.5) * 0.02 # Finnhub 情緒影響
    
    total_bias = rsi_bias + sent_bias
    
    np.random.seed(42)
    future_dates = [last_date + timedelta(days=i) for i in range(1, days + 1)]
    future_prices = []
    
    curr = last_price
    for i in range(days):
        # 模擬每日變動
        change = np.random.normal(total_bias * (0.9**i), volatility)
        curr *= (1 + change)
        future_prices.append(curr)
    
    return pd.DataFrame({'Date': future_dates, 'Close': future_prices})

# ==================== 4. 主程式介面 ====================

def main():
    st.title("🏛️ AI 股市全方位預測系統")
    
    # --- 側邊欄控制 ---
    st.sidebar.header("🔍 股票搜尋")
    target_stock = st.sidebar.text_input("輸入代碼 (例如: 2330.TW, TSLA)", "2330.TW").upper()
    forecast_days = st.sidebar.slider("預測未來天數", 5, 10, 7)
    show_fundamentals = st.sidebar.toggle("加載基本面數據 (可能較慢)", value=False)

    # --- 數據加載流程 ---
    with st.spinner('數據同步中...'):
        df_raw = get_stock_data(target_stock)
        
        if df_raw is not None:
            df = calculate_indicators(df_raw)
            sentiment_data = get_finnhub_sentiment(target_stock)
            sent_score = sentiment_data['sentiment'].get('bullishPercent', 0.5) if sentiment_data and 'sentiment' in sentiment_data else 0.5
            
            # 預測未來
            future_df = predict_future(df, sent_score, days=forecast_days)
            
            # --- 繪製主圖表 ---
            st.subheader(f"📈 {target_stock} 歷史走勢與 AI 預期")
            fig = go.Figure()
            
            # 歷史 K 線 (僅顯示最近 100 天)
            d_plot = df.tail(100)
            fig.add_trace(go.Candlestick(x=d_plot['Date'], open=d_plot['Open'], high=d_plot['High'], 
                                         low=d_plot['Low'], close=d_plot['Close'], name="歷史K線"))
            
            # 布林通道
            fig.add_trace(go.Scatter(x=d_plot['Date'], y=d_plot['BB_High'], line=dict(color='rgba(200,200,200,0.3)'), name="布林上軌"))
            fig.add_trace(go.Scatter(x=d_plot['Date'], y=d_plot['BB_Low'], fill='tonexty', line=dict(color='rgba(200,200,200,0.3)'), name="布林下軌"))
            
            # AI 預測線
            connect_df = pd.concat([df.tail(1)[['Date', 'Close']], future_df])
            fig.add_trace(go.Scatter(x=connect_df['Date'], y=connect_df['Close'], mode='lines+markers',
                                     line=dict(color='orange', width=3, dash='dot'), name="AI 預估走向"))
            
            fig.update_layout(height=600, template="plotly_dark", xaxis_rangeslider_visible=False)
            st.plotly_chart(fig, use_container_width=True)

            # --- 分析面板 ---
            c1, c2 = st.columns(2)
            with c1:
                st.info("### 📊 關鍵指標預估")
                curr_p = df['Close'].iloc[-1]
                pred_p = future_df['Close'].iloc[-1]
                diff = ((pred_p - curr_p) / curr_p) * 100
                st.metric("當前收盤", f"${curr_p:.2f}")
                st.metric(f"未來 {forecast_days} 日目標", f"${pred_p:.2f}", f"{diff:+.2f}%")
            
            with c2:
                st.success("### 🧠 AI 綜合評判")
                rsi_val = df['RSI'].iloc[-1]
                msg = "📈 **多頭格局**：情緒偏好且技術指標未過熱。" if sent_score > 0.5 and rsi_val < 70 else "📉 **警惕回調**：技術面進入超買區或情緒轉弱。"
                st.markdown(f"**市場情緒指數**: `{sent_score:.2f}`\n\n**當前 RSI**: `{rsi_val:.1f}`\n\n**AI 建議**: {msg}")

            # --- 基本面面板 (勾選才執行) ---
            if show_fundamentals:
                st.divider()
                st.subheader("💼 公司基本面核心數據")
                f_data = get_fundamental_data(target_stock)
                if f_data:
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("本益比 (PE)", f_data['PE Ratio'])
                    m2.metric("ROE", f"{f_data['ROE']*100:.2f}%" if f_data['ROE'] != 'N/A' else 'N/A')
                    m3.metric("殖利率", f"{f_data['Dividend Yield']*100:.2f}%" if f_data['Dividend Yield'] != 'N/A' else 'N/A')
                    m4.metric("市值", f"{f_data['Market Cap']/1e9:.1f}B")
                else:
                    st.warning("無法載入基本面，請稍後再試。")

        else:
            st.error("❌ 找不到該股票數據，請確保輸入正確（台股請加 .TW）。")

if __name__ == "__main__":
    main()
