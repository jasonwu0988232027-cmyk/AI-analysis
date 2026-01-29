import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# ==================== 模組檢測與導入 ====================
# 延遲導入，避免啟動時卡頓
TA_AVAILABLE = False
SKLEARN_AVAILABLE = False
TF_AVAILABLE = False

def lazy_import_ta():
    """延遲導入 ta 套件"""
    global TA_AVAILABLE
    if not TA_AVAILABLE:
        try:
            import ta
            TA_AVAILABLE = True
            return True
        except ImportError:
            return False
    return True

def lazy_import_sklearn():
    """延遲導入 sklearn"""
    global SKLEARN_AVAILABLE
    if not SKLEARN_AVAILABLE:
        try:
            from sklearn.preprocessing import MinMaxScaler
            from sklearn.metrics import mean_absolute_error
            SKLEARN_AVAILABLE = True
            return True
        except ImportError:
            return False
    return True

def lazy_import_tensorflow():
    """延遲導入 TensorFlow"""
    global TF_AVAILABLE
    if not TF_AVAILABLE:
        try:
            import tensorflow as tf
            tf.get_logger().setLevel('ERROR')  # 禁用 TF 警告
            TF_AVAILABLE = True
            return True
        except ImportError:
            return False
    return True

# ==================== 頁面配置 ====================
st.set_page_config(
    page_title="AI 股市預測專家 Pro",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API 設定
FINNHUB_API_KEY = "d5t2rvhr01qt62ngu1kgd5t2rvhr01qt62ngu1l0"

# ==================== 1. 數據獲取（優化版）====================

@st.cache_data(ttl=3600, show_spinner=False)
def get_stock_data(symbol, period="6mo"):  # 減少預設期間到6個月
    """獲取股票數據 - 優化版"""
    try:
        df = yf.download(symbol, period=period, interval="1d", progress=False)
        if df.empty: 
            return None
        df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
        return df.reset_index()
    except Exception as e:
        st.error(f"❌ 數據獲取失敗: {str(e)}")
        return None

@st.cache_data(ttl=7200, show_spinner=False)
def get_fundamental_data(symbol):
    """獲取基本面數據 - 快取時間更長"""
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        
        fundamentals = {
            'PE Ratio': info.get('trailingPE', 'N/A'),
            'Price to Book': info.get('priceToBook', 'N/A'),
            'Dividend Yield': info.get('dividendYield', 'N/A'),
            'ROE': info.get('returnOnEquity', 'N/A'),
            'Debt to Equity': info.get('debtToEquity', 'N/A'),
            'Market Cap': info.get('marketCap', 'N/A'),
        }
        return fundamentals
    except:
        return None

@st.cache_data(ttl=3600, show_spinner=False)
def get_sentiment(symbol):
    """獲取市場情緒"""
    clean_symbol = symbol.split('.')[0]
    url = f"https://finnhub.io/api/v1/news-sentiment?symbol={clean_symbol}&token={FINNHUB_API_KEY}"
    try:
        res = requests.get(url, timeout=3).json()
        if res and 'sentiment' in res:
            return res['sentiment'].get('bullishPercent', 0.5)
    except:
        pass
    return 0.5

# ==================== 2. 技術指標計算（簡化版）====================

@st.cache_data(show_spinner=False)
def calculate_indicators(df):
    """計算技術指標 - 精簡高效版"""
    df = df.copy()
    
    # RSI
    delta = df['Close'].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = -delta.clip(upper=0).rolling(14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # 移動平均
    df['SMA_20'] = df['Close'].rolling(20).mean()
    df['SMA_50'] = df['Close'].rolling(50).mean()
    df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()
    
    # MACD
    df['MACD'] = df['EMA_12'] - df['EMA_26']
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Diff'] = df['MACD'] - df['MACD_Signal']
    
    # 布林通道
    df['BB_Mid'] = df['Close'].rolling(20).mean()
    bb_std = df['Close'].rolling(20).std()
    df['BB_High'] = df['BB_Mid'] + (bb_std * 2)
    df['BB_Low'] = df['BB_Mid'] - (bb_std * 2)
    
    # KD
    low_14 = df['Low'].rolling(14).min()
    high_14 = df['High'].rolling(14).max()
    df['K'] = 100 * ((df['Close'] - low_14) / (high_14 - low_14))
    df['D'] = df['K'].rolling(3).mean()
    
    return df.fillna(method='bfill')

# ==================== 3. 預測模型（優化版）====================

def predict_price(df, sentiment, days=10):
    """技術分析預測 - 優化版"""
    last_price = df['Close'].iloc[-1]
    last_date = df['Date'].iloc[-1]
    
    # 計算指標
    volatility = df['Close'].pct_change().std()
    trend = (df['Close'].iloc[-1] / df['Close'].iloc[-5]) - 1
    rsi = df['RSI'].iloc[-1]
    macd = df['MACD_Diff'].iloc[-1]
    
    # 綜合因子
    bias = (
        (sentiment - 0.5) * 0.01 +
        trend * 0.3 +
        (50 - rsi) / 1000 +
        np.sign(macd) * 0.005
    )
    
    # 生成預測
    np.random.seed(42)
    dates = pd.date_range(last_date + timedelta(1), periods=days)
    prices = [last_price]
    
    for i in range(days):
        change = np.random.normal(bias * (0.95 ** i), volatility)
        prices.append(prices[-1] * (1 + change))
    
    return pd.DataFrame({'Date': dates, 'Close': prices[1:]})

# LSTM 相關函數（僅在需要時加載）
def train_lstm_model(df, epochs=30):
    """訓練 LSTM 模型 - 簡化版"""
    if not lazy_import_sklearn() or not lazy_import_tensorflow():
        raise ImportError("需要 scikit-learn 和 TensorFlow")
    
    from sklearn.preprocessing import MinMaxScaler
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    
    # 準備數據
    data = df[['Close']].values
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(data)
    
    lookback = 30  # 減少回看窗口
    X, y = [], []
    for i in range(lookback, len(scaled)):
        X.append(scaled[i-lookback:i, 0])
        y.append(scaled[i, 0])
    
    X = np.array(X).reshape(-1, lookback, 1)
    y = np.array(y)
    
    # 分割數據
    split = int(len(X) * 0.8)
    X_train, y_train = X[:split], y[:split]
    
    # 簡化模型
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(lookback, 1)),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(1)
    ])
    
    model.compile(optimizer='adam', loss='mse')
    model.fit(X_train, y_train, epochs=epochs, batch_size=32, verbose=0)
    
    return model, scaler, lookback

def predict_lstm(model, df, scaler, lookback, days=10):
    """LSTM 預測"""
    data = df[['Close']].values
    scaled = scaler.transform(data)
    
    predictions = []
    current = scaled[-lookback:].reshape(1, lookback, 1)
    
    for _ in range(days):
        pred = model.predict(current, verbose=0)[0, 0]
        predictions.append(pred)
        current = np.append(current[:, 1:, :], [[[pred]]], axis=1)
    
    predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
    dates = pd.date_range(df['Date'].iloc[-1] + timedelta(1), periods=days)
    
    return pd.DataFrame({'Date': dates, 'Close': predictions.flatten()})

# ==================== 4. 分析報告 ====================

def generate_analysis(df, future_df, sentiment):
    """生成簡潔分析報告"""
    latest = df.iloc[-1]
    change = ((future_df['Close'].iloc[-1] / df['Close'].iloc[-1]) - 1) * 100
    
    analysis = []
    
    # 預測方向
    if change > 0:
        analysis.append(f"### 📈 預測上漲 {change:.2f}%")
    else:
        analysis.append(f"### 📉 預測下跌 {abs(change):.2f}%")
    
    # RSI
    rsi = latest['RSI']
    if rsi > 70:
        analysis.append("⚠️ RSI 超買 (>70)")
    elif rsi < 30:
        analysis.append("✅ RSI 超賣 (<30)")
    else:
        analysis.append(f"📊 RSI 正常 ({rsi:.1f})")
    
    # MACD
    if latest['MACD_Diff'] > 0:
        analysis.append("📈 MACD 金叉")
    else:
        analysis.append("📉 MACD 死叉")
    
    # 情緒
    if sentiment > 0.6:
        analysis.append(f"🟢 情緒偏多 ({sentiment:.2f})")
    elif sentiment < 0.4:
        analysis.append(f"🔴 情緒偏空 ({sentiment:.2f})")
    
    return "\n\n".join(analysis)

# ==================== 5. 圖表生成（優化版）====================

def create_main_chart(df, future_df, show_last_days=60):
    """創建主圖表 - 只顯示最近N天"""
    fig = go.Figure()
    
    # 限制顯示範圍
    df_display = df.tail(show_last_days)
    
    # K線
    fig.add_trace(go.Candlestick(
        x=df_display['Date'],
        open=df_display['Open'],
        high=df_display['High'],
        low=df_display['Low'],
        close=df_display['Close'],
        name="K線",
        increasing_line_color='red',
        decreasing_line_color='green'
    ))
    
    # 均線
    fig.add_trace(go.Scatter(
        x=df_display['Date'],
        y=df_display['SMA_20'],
        name='MA20',
        line=dict(color='orange', width=1)
    ))
    
    fig.add_trace(go.Scatter(
        x=df_display['Date'],
        y=df_display['SMA_50'],
        name='MA50',
        line=dict(color='blue', width=1)
    ))
    
    # 預測線
    connect = pd.concat([df.tail(1)[['Date', 'Close']], future_df])
    fig.add_trace(go.Scatter(
        x=connect['Date'],
        y=connect['Close'],
        mode='lines+markers',
        line=dict(color='red', width=3, dash='dot'),
        name='預測'
    ))
    
    fig.update_layout(
        xaxis_rangeslider_visible=False,
        height=500,
        template="plotly_dark",
        hovermode='x unified',
        showlegend=True
    )
    
    return fig

def create_indicator_chart(df, indicator, name, show_last_days=60):
    """創建指標圖表"""
    fig = go.Figure()
    df_display = df.tail(show_last_days)
    
    fig.add_trace(go.Scatter(
        x=df_display['Date'],
        y=df_display[indicator],
        name=name,
        line=dict(width=2)
    ))
    
    # 添加參考線
    if indicator == 'RSI':
        fig.add_hline(y=70, line_dash="dash", line_color="red", opacity=0.5)
        fig.add_hline(y=30, line_dash="dash", line_color="green", opacity=0.5)
    
    fig.update_layout(
        height=200,
        template="plotly_dark",
        showlegend=False,
        margin=dict(t=20, b=20)
    )
    
    return fig

# ==================== 6. 主程式 ====================

def main():
    # 標題
    st.title("📈 AI 股市預測專家 Pro")
    st.markdown("*優化版 - 更快速、更穩定*")
    
    # ===== 側邊欄 =====
    with st.sidebar:
        st.header("⚙️ 設定")
        
        symbol = st.text_input("股票代碼", "2330.TW", help="例: 2330.TW, AAPL").upper()
        forecast_days = st.slider("預測天數", 5, 20, 10)
        
        st.subheader("模型選擇")
        model_type = st.radio(
            "預測模型",
            ["技術分析", "LSTM 深度學習"],
            help="技術分析較快，LSTM 更準但需時間"
        )
        
        if model_type == "LSTM 深度學習":
            epochs = st.slider("訓練輪數", 10, 50, 20)
        else:
            epochs = 20
        
        show_indicators = st.checkbox("顯示技術指標", value=True)
        show_fundamentals = st.checkbox("顯示基本面", value=False)
    
    # ===== 獲取數據 =====
    with st.spinner(f'📊 正在獲取 {symbol} 數據...'):
        df = get_stock_data(symbol)
        
        if df is None:
            st.error("❌ 無法獲取數據，請檢查股票代碼")
            st.stop()
        
        # 計算指標
        df = calculate_indicators(df)
        sentiment = get_sentiment(symbol)
        
        # 獲取基本面（如果需要）
        fundamentals = None
        if show_fundamentals:
            fundamentals = get_fundamental_data(symbol)
    
    # ===== 執行預測 =====
    try:
        if model_type == "LSTM 深度學習":
            with st.spinner('🤖 正在訓練 LSTM 模型...'):
                model, scaler, lookback = train_lstm_model(df, epochs=epochs)
                future_df = predict_lstm(model, df, scaler, lookback, days=forecast_days)
            model_name = "LSTM"
        else:
            future_df = predict_price(df, sentiment, days=forecast_days)
            model_name = "技術分析"
    except Exception as e:
        st.warning(f"⚠️ {model_type} 失敗，使用技術分析: {str(e)}")
        future_df = predict_price(df, sentiment, days=forecast_days)
        model_name = "技術分析"
    
    # ===== 主要展示區 =====
    st.subheader(f"📊 {symbol} 走勢與預測")
    
    # 主圖表
    fig_main = create_main_chart(df, future_df)
    st.plotly_chart(fig_main, use_container_width=True)
    
    # ===== 數據摘要 =====
    col1, col2, col3, col4 = st.columns(4)
    
    current = df['Close'].iloc[-1]
    predicted = future_df['Close'].iloc[-1]
    change = ((predicted - current) / current) * 100
    
    col1.metric("當前價格", f"${current:.2f}")
    col2.metric(f"{forecast_days}日預測", f"${predicted:.2f}", f"{change:+.2f}%")
    col3.metric("RSI", f"{df['RSI'].iloc[-1]:.1f}")
    col4.metric("市場情緒", f"{sentiment:.2f}")
    
    # ===== 分析與指標 =====
    tab1, tab2, tab3 = st.tabs(["🎯 分析", "📈 指標", "💼 基本面"])
    
    with tab1:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("### 技術分析")
            analysis = generate_analysis(df, future_df, sentiment)
            st.markdown(analysis)
        
        with col2:
            st.markdown("### 預測明細")
            display_df = future_df.copy()
            display_df['Date'] = display_df['Date'].dt.strftime('%Y-%m-%d')
            display_df['價格'] = display_df['Close'].apply(lambda x: f"${x:.2f}")
            display_df['變化'] = display_df['Close'].pct_change().fillna(0).apply(lambda x: f"{x*100:+.2f}%")
            st.dataframe(display_df[['Date', '價格', '變化']], hide_index=True)
    
    with tab2:
        if show_indicators:
            st.markdown("### 技術指標")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**RSI**")
                st.plotly_chart(create_indicator_chart(df, 'RSI', 'RSI'), use_container_width=True)
            
            with col2:
                st.markdown("**MACD**")
                fig_macd = go.Figure()
                df_display = df.tail(60)
                fig_macd.add_trace(go.Scatter(x=df_display['Date'], y=df_display['MACD'], name='MACD'))
                fig_macd.add_trace(go.Scatter(x=df_display['Date'], y=df_display['MACD_Signal'], name='Signal'))
                fig_macd.update_layout(height=200, template="plotly_dark", margin=dict(t=20, b=20))
                st.plotly_chart(fig_macd, use_container_width=True)
        else:
            st.info("在側邊欄啟用技術指標顯示")
    
    with tab3:
        if fundamentals:
            st.markdown("### 基本面數據")
            
            col1, col2, col3 = st.columns(3)
            
            pe = fundamentals.get('PE Ratio', 'N/A')
            roe = fundamentals.get('ROE', 'N/A')
            dy = fundamentals.get('Dividend Yield', 'N/A')
            
            col1.metric("本益比", f"{pe:.2f}" if isinstance(pe, (int, float)) else "N/A")
            col2.metric("ROE", f"{roe*100:.2f}%" if isinstance(roe, (int, float)) else "N/A")
            col3.metric("殖利率", f"{dy*100:.2f}%" if isinstance(dy, (int, float)) else "N/A")
        else:
            st.info("在側邊欄啟用基本面顯示")
    
    # ===== 頁腳 =====
    st.markdown("---")
    st.caption(f"⚙️ 使用模型: {model_name} | ⚠️ 本系統僅供參考，不構成投資建議")

if __name__ == "__main__":
    main()
