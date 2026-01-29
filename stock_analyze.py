import streamlit as st

# --- 頁面配置（必須在最前面）---
st.set_page_config(page_title="AI 股市預測專家 Lite", layout="wide")

import yfinance as yf
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
from datetime import datetime, timedelta

# --- API 設定 ---
FINNHUB_API_KEY = "d5t2rvhr01qt62ngu1kgd5t2rvhr01qt62ngu1l0"

# ==================== 1. 數據獲取 ====================

@st.cache_data(ttl=3600)
def get_stock_data(symbol, period="1y"):
    """獲取股票數據"""
    try:
        df = yf.download(symbol, period=period, interval="1d", progress=False)
        if df.empty: 
            return None
        df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
        df = df.reset_index()
        return df
    except Exception as e:
        st.error(f"獲取數據失敗: {str(e)}")
        return None

@st.cache_data(ttl=3600)
def get_fundamental_data(symbol):
    """獲取基本面數據"""
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        
        fundamentals = {
            'PE Ratio': info.get('trailingPE', 'N/A'),
            'Forward PE': info.get('forwardPE', 'N/A'),
            'Price to Book': info.get('priceToBook', 'N/A'),
            'Dividend Yield': info.get('dividendYield', 'N/A'),
            'ROE': info.get('returnOnEquity', 'N/A'),
            'Debt to Equity': info.get('debtToEquity', 'N/A'),
            'Market Cap': info.get('marketCap', 'N/A'),
        }
        
        return fundamentals, info
    except:
        return None, None

# ==================== 2. 技術指標計算（簡化版）====================

def calculate_technical_indicators(df):
    """計算基本技術指標"""
    df_copy = df.copy()
    
    # RSI
    delta = df_copy['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df_copy['RSI'] = 100 - (100 / (1 + rs))
    
    # 移動平均
    df_copy['SMA_20'] = df_copy['Close'].rolling(window=20).mean()
    df_copy['SMA_50'] = df_copy['Close'].rolling(window=50).mean()
    df_copy['EMA_12'] = df_copy['Close'].ewm(span=12, adjust=False).mean()
    df_copy['EMA_26'] = df_copy['Close'].ewm(span=26, adjust=False).mean()
    
    # MACD
    df_copy['MACD'] = df_copy['EMA_12'] - df_copy['EMA_26']
    df_copy['MACD_Signal'] = df_copy['MACD'].ewm(span=9, adjust=False).mean()
    df_copy['MACD_Diff'] = df_copy['MACD'] - df_copy['MACD_Signal']
    
    # 布林通道
    df_copy['BB_Mid'] = df_copy['Close'].rolling(window=20).mean()
    bb_std = df_copy['Close'].rolling(window=20).std()
    df_copy['BB_High'] = df_copy['BB_Mid'] + (bb_std * 2)
    df_copy['BB_Low'] = df_copy['BB_Mid'] - (bb_std * 2)
    df_copy['BB_Width'] = (df_copy['BB_High'] - df_copy['BB_Low']) / df_copy['BB_Mid']
    
    # KD
    low_14 = df_copy['Low'].rolling(window=14).min()
    high_14 = df_copy['High'].rolling(window=14).max()
    df_copy['K'] = 100 * ((df_copy['Close'] - low_14) / (high_14 - low_14))
    df_copy['D'] = df_copy['K'].rolling(window=3).mean()
    
    # ATR
    high_low = df_copy['High'] - df_copy['Low']
    high_close = np.abs(df_copy['High'] - df_copy['Close'].shift())
    low_close = np.abs(df_copy['Low'] - df_copy['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    df_copy['ATR'] = true_range.rolling(14).mean()
    
    # 填充 NaN
    df_copy = df_copy.fillna(method='bfill').fillna(method='ffill')
    
    return df_copy

# ==================== 3. 預測模型 ====================

def predict_future(df, sentiment_score, days=10):
    """技術分析預測"""
    last_price = df['Close'].iloc[-1]
    last_date = df['Date'].iloc[-1]
    
    # 技術指標
    volatility = df['Close'].pct_change().std()
    recent_trend = (df['Close'].iloc[-1] - df['Close'].iloc[-5]) / df['Close'].iloc[-5]
    
    rsi = df['RSI'].iloc[-1] if 'RSI' in df.columns else 50
    macd_diff = df['MACD_Diff'].iloc[-1] if 'MACD_Diff' in df.columns else 0
    
    # 綜合因子
    rsi_bias = (50 - rsi) / 100 * 0.01
    macd_bias = np.sign(macd_diff) * 0.005
    sentiment_bias = (sentiment_score - 0.5) * 0.015
    trend_bias = recent_trend * 0.3
    
    total_bias = sentiment_bias + trend_bias + rsi_bias + macd_bias
    
    # 預測
    np.random.seed(42)
    future_dates = [last_date + timedelta(days=i) for i in range(1, days + 1)]
    future_prices = []
    
    current_price = last_price
    for i in range(days):
        decay_factor = 0.95 ** i
        adjusted_bias = total_bias * decay_factor
        change_pct = np.random.normal(adjusted_bias, volatility)
        current_price *= (1 + change_pct)
        future_prices.append(current_price)
    
    np.random.seed(None)
    
    return pd.DataFrame({'Date': future_dates, 'Close': future_prices})

# ==================== 4. 分析報告 ====================

def generate_analysis(df, future_df, sentiment_score):
    """生成分析報告"""
    analysis = []
    latest = df.iloc[-1]
    
    # 價格預測
    current_price = df['Close'].iloc[-1]
    predicted_price = future_df['Close'].iloc[-1]
    change_pct = ((predicted_price - current_price) / current_price) * 100
    
    if change_pct > 0:
        analysis.append(f"### 📈 預測上漲 {change_pct:.2f}%")
    else:
        analysis.append(f"### 📉 預測下跌 {abs(change_pct):.2f}%")
    
    # RSI
    rsi = latest['RSI']
    if rsi > 70:
        analysis.append("⚠️ **RSI 超買** (>70)：可能回調")
    elif rsi < 30:
        analysis.append("✅ **RSI 超賣** (<30)：可能反彈")
    else:
        analysis.append(f"📊 **RSI 正常** ({rsi:.1f})")
    
    # MACD
    if latest['MACD_Diff'] > 0:
        analysis.append("📈 **MACD 金叉**：多頭訊號")
    else:
        analysis.append("📉 **MACD 死叉**：空頭訊號")
    
    # 布林通道
    close = latest['Close']
    if close > latest['BB_High']:
        analysis.append("⚠️ **突破布林上軌**：強勢但需警惕")
    elif close < latest['BB_Low']:
        analysis.append("💚 **跌破布林下軌**：超賣訊號")
    
    # 市場情緒
    if sentiment_score > 0.6:
        analysis.append(f"🟢 **市場情緒偏多** ({sentiment_score:.2f})")
    elif sentiment_score < 0.4:
        analysis.append(f"🔴 **市場情緒偏空** ({sentiment_score:.2f})")
    else:
        analysis.append(f"🟡 **市場情緒中性** ({sentiment_score:.2f})")
    
    return "\n\n".join(analysis)

@st.cache_data(ttl=3600)
def get_finnhub_sentiment(symbol):
    """獲取市場情緒"""
    clean_symbol = symbol.split('.')[0]
    url = f"https://finnhub.io/api/v1/news-sentiment?symbol={clean_symbol}&token={FINNHUB_API_KEY}"
    try:
        res = requests.get(url, timeout=5).json()
        return res
    except:
        return None

# ==================== 5. 主程式 ====================

def main():
    st.title("📈 AI 股市預測專家 Lite")
    st.markdown("*輕量級股市分析工具 - 快速部署版*")
    
    # 側邊欄
    st.sidebar.header("⚙️ 設定")
    target_stock = st.sidebar.text_input("股票代碼", "2330.TW").upper()
    forecast_days = st.sidebar.slider("預測天數", 5, 30, 10)
    show_fundamentals = st.sidebar.checkbox("顯示基本面", value=True)
    
    # 獲取數據
    with st.spinner(f'正在獲取 {target_stock} 數據...'):
        df = get_stock_data(target_stock)
        if df is None:
            st.error("❌ 無法獲取數據，請檢查股票代碼")
            return
        
        df = calculate_technical_indicators(df)
        sentiment_data = get_finnhub_sentiment(target_stock)
        sent_score = sentiment_data['sentiment'].get('bullishPercent', 0.5) if sentiment_data and 'sentiment' in sentiment_data else 0.5
        
        if show_fundamentals:
            fundamentals, info = get_fundamental_data(target_stock)
    
    # 預測
    future_df = predict_future(df, sent_score, days=forecast_days)
    
    # ===== 主圖表 =====
    st.subheader(f"📊 {target_stock} 技術分析")
    
    fig = go.Figure()
    
    # K線
    fig.add_trace(go.Candlestick(
        x=df['Date'][-90:],
        open=df['Open'][-90:],
        high=df['High'][-90:],
        low=df['Low'][-90:],
        close=df['Close'][-90:],
        name="K線"
    ))
    
    # 移動平均
    fig.add_trace(go.Scatter(
        x=df['Date'][-90:], 
        y=df['SMA_20'][-90:], 
        name='SMA 20', 
        line=dict(color='orange', width=1)
    ))
    fig.add_trace(go.Scatter(
        x=df['Date'][-90:], 
        y=df['SMA_50'][-90:], 
        name='SMA 50', 
        line=dict(color='blue', width=1)
    ))
    
    # 布林通道
    fig.add_trace(go.Scatter(
        x=df['Date'][-90:], 
        y=df['BB_High'][-90:], 
        name='布林上軌', 
        line=dict(color='gray', width=1, dash='dot')
    ))
    fig.add_trace(go.Scatter(
        x=df['Date'][-90:], 
        y=df['BB_Low'][-90:], 
        name='布林下軌', 
        line=dict(color='gray', width=1, dash='dot'),
        fill='tonexty'
    ))
    
    # 預測線
    connect_df = pd.concat([df.tail(1)[['Date', 'Close']], future_df])
    fig.add_trace(go.Scatter(
        x=connect_df['Date'],
        y=connect_df['Close'],
        mode='lines+markers',
        line=dict(color='red', width=3, dash='dot'),
        marker=dict(size=8),
        name='AI 預測'
    ))
    
    fig.update_layout(
        xaxis_rangeslider_visible=False,
        height=600,
        template="plotly_dark",
        hovermode='x unified'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # ===== 分析面板 =====
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📊 數據摘要")
        
        current_price = df['Close'].iloc[-1]
        predicted_price = future_df['Close'].iloc[-1]
        change_pct = ((predicted_price - current_price) / current_price) * 100
        
        st.metric("當前價格", f"${current_price:.2f}")
        st.metric(
            f"{forecast_days} 日預測",
            f"${predicted_price:.2f}",
            f"{change_pct:+.2f}%"
        )
        
        st.markdown("**技術指標：**")
        latest = df.iloc[-1]
        st.write(f"- RSI: `{latest['RSI']:.1f}`")
        st.write(f"- MACD: `{latest['MACD']:.2f}`")
        st.write(f"- K值: `{latest['K']:.1f}`")
        st.write(f"- D值: `{latest['D']:.1f}`")
    
    with col2:
        st.markdown("### 🧠 技術分析")
        analysis = generate_analysis(df, future_df, sent_score)
        st.markdown(analysis)
    
    # ===== 技術指標圖表 =====
    with st.expander("📈 查看詳細技術指標"):
        tab1, tab2 = st.tabs(["RSI & MACD", "KD & 成交量"])
        
        with tab1:
            # RSI
            fig_rsi = go.Figure()
            fig_rsi.add_trace(go.Scatter(
                x=df['Date'][-90:], 
                y=df['RSI'][-90:], 
                name='RSI',
                line=dict(color='purple')
            ))
            fig_rsi.add_hline(y=70, line_dash="dash", line_color="red")
            fig_rsi.add_hline(y=30, line_dash="dash", line_color="green")
            fig_rsi.update_layout(height=250, template="plotly_dark")
            st.plotly_chart(fig_rsi, use_container_width=True)
            
            # MACD
            fig_macd = go.Figure()
            fig_macd.add_trace(go.Scatter(
                x=df['Date'][-90:], 
                y=df['MACD'][-90:], 
                name='MACD',
                line=dict(color='blue')
            ))
            fig_macd.add_trace(go.Scatter(
                x=df['Date'][-90:], 
                y=df['MACD_Signal'][-90:], 
                name='Signal',
                line=dict(color='orange')
            ))
            fig_macd.add_trace(go.Bar(
                x=df['Date'][-90:], 
                y=df['MACD_Diff'][-90:], 
                name='Histogram'
            ))
            fig_macd.update_layout(height=250, template="plotly_dark")
            st.plotly_chart(fig_macd, use_container_width=True)
        
        with tab2:
            # KD
            fig_kd = go.Figure()
            fig_kd.add_trace(go.Scatter(
                x=df['Date'][-90:], 
                y=df['K'][-90:], 
                name='K',
                line=dict(color='blue')
            ))
            fig_kd.add_trace(go.Scatter(
                x=df['Date'][-90:], 
                y=df['D'][-90:], 
                name='D',
                line=dict(color='orange')
            ))
            fig_kd.add_hline(y=80, line_dash="dash", line_color="red")
            fig_kd.add_hline(y=20, line_dash="dash", line_color="green")
            fig_kd.update_layout(height=250, template="plotly_dark")
            st.plotly_chart(fig_kd, use_container_width=True)
            
            # 成交量
            fig_vol = go.Figure()
            colors = ['red' if df['Close'].iloc[i] < df['Open'].iloc[i] else 'green' 
                     for i in range(-90, 0)]
            fig_vol.add_trace(go.Bar(
                x=df['Date'][-90:], 
                y=df['Volume'][-90:],
                marker_color=colors,
                name='成交量'
            ))
            fig_vol.update_layout(height=250, template="plotly_dark")
            st.plotly_chart(fig_vol, use_container_width=True)
    
    # ===== 基本面 =====
    if show_fundamentals and fundamentals:
        with st.expander("💼 基本面數據"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                pe = fundamentals['PE Ratio']
                st.metric("本益比", f"{pe:.2f}" if isinstance(pe, (int, float)) else "N/A")
                
            with col2:
                roe = fundamentals['ROE']
                st.metric("ROE", f"{roe*100:.2f}%" if isinstance(roe, (int, float)) else "N/A")
            
            with col3:
                dy = fundamentals['Dividend Yield']
                st.metric("殖利率", f"{dy*100:.2f}%" if isinstance(dy, (int, float)) else "N/A")
    
    # ===== 預測數據表 =====
    with st.expander("📅 預測明細"):
        display_df = future_df.copy()
        display_df['Date'] = display_df['Date'].dt.strftime('%Y-%m-%d')
        display_df['價格'] = display_df['Close'].apply(lambda x: f"${x:.2f}")
        display_df['變化%'] = display_df['Close'].pct_change().fillna(0).apply(lambda x: f"{x*100:+.2f}%")
        st.dataframe(display_df[['Date', '價格', '變化%']], use_container_width=True)
    
    st.markdown("---")
    st.caption("⚠️ **免責聲明**：本系統僅供學習參考，不構成投資建議。")

if __name__ == "__main__":
    main()
