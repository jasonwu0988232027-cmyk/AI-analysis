import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
from datetime import datetime, timedelta
try:
    import ta
    TA_AVAILABLE = True
except ImportError:
    TA_AVAILABLE = False
    st.warning("⚠️ 技術指標套件 'ta' 未安裝，部分功能將受限。請執行: pip install ta")

try:
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

import warnings
warnings.filterwarnings('ignore')

# --- API 設定 ---
FINNHUB_API_KEY = "d5t2rvhr01qt62ngu1kgd5t2rvhr01qt62ngu1l0"
st.set_page_config(page_title="AI 股市預測專家 Pro", layout="wide", initial_sidebar_state="expanded")

# --- 全局變數 ---
LOOKBACK_DAYS = 60  # LSTM 訓練窗口

# ==================== 1. 數據獲取與處理 ====================

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
            'PEG Ratio': info.get('pegRatio', 'N/A'),
            'Price to Book': info.get('priceToBook', 'N/A'),
            'Dividend Yield': info.get('dividendYield', 'N/A'),
            'Profit Margins': info.get('profitMargins', 'N/A'),
            'ROE': info.get('returnOnEquity', 'N/A'),
            'ROA': info.get('returnOnAssets', 'N/A'),
            'Debt to Equity': info.get('debtToEquity', 'N/A'),
            'Current Ratio': info.get('currentRatio', 'N/A'),
            'Market Cap': info.get('marketCap', 'N/A'),
            'Revenue Growth': info.get('revenueGrowth', 'N/A'),
            'Earnings Growth': info.get('earningsGrowth', 'N/A'),
        }
        
        return fundamentals, info
    except Exception as e:
        return None, None

# ==================== 2. 技術指標計算 ====================

def calculate_technical_indicators(df):
    """計算各種技術指標"""
    if not TA_AVAILABLE:
        # 如果 ta 套件不可用，使用簡化版計算
        return calculate_basic_indicators(df)
    
    df_copy = df.copy()
    
    try:
        # MACD
        macd = ta.trend.MACD(df_copy['Close'])
        df_copy['MACD'] = macd.macd()
        df_copy['MACD_Signal'] = macd.macd_signal()
        df_copy['MACD_Diff'] = macd.macd_diff()
        
        # RSI
        df_copy['RSI'] = ta.momentum.RSIIndicator(df_copy['Close'], window=14).rsi()
        
        # 布林通道
        bollinger = ta.volatility.BollingerBands(df_copy['Close'])
        df_copy['BB_High'] = bollinger.bollinger_hband()
        df_copy['BB_Mid'] = bollinger.bollinger_mavg()
        df_copy['BB_Low'] = bollinger.bollinger_lband()
        df_copy['BB_Width'] = (df_copy['BB_High'] - df_copy['BB_Low']) / df_copy['BB_Mid']
        
        # 移動平均線
        df_copy['SMA_20'] = ta.trend.SMAIndicator(df_copy['Close'], window=20).sma_indicator()
        df_copy['SMA_50'] = ta.trend.SMAIndicator(df_copy['Close'], window=50).sma_indicator()
        df_copy['EMA_12'] = ta.trend.EMAIndicator(df_copy['Close'], window=12).ema_indicator()
        df_copy['EMA_26'] = ta.trend.EMAIndicator(df_copy['Close'], window=26).ema_indicator()
        
        # KD 指標
        stoch = ta.momentum.StochasticOscillator(df_copy['High'], df_copy['Low'], df_copy['Close'])
        df_copy['K'] = stoch.stoch()
        df_copy['D'] = stoch.stoch_signal()
        
        # ATR (平均真實波幅)
        df_copy['ATR'] = ta.volatility.AverageTrueRange(df_copy['High'], df_copy['Low'], df_copy['Close']).average_true_range()
        
        # OBV (能量潮)
        df_copy['OBV'] = ta.volume.OnBalanceVolumeIndicator(df_copy['Close'], df_copy['Volume']).on_balance_volume()
        
        # ADX (趨勢強度)
        df_copy['ADX'] = ta.trend.ADXIndicator(df_copy['High'], df_copy['Low'], df_copy['Close']).adx()
        
        # 威廉指標
        df_copy['Williams_R'] = ta.momentum.WilliamsRIndicator(df_copy['High'], df_copy['Low'], df_copy['Close']).williams_r()
        
    except Exception as e:
        st.error(f"技術指標計算錯誤: {str(e)}")
        return calculate_basic_indicators(df_copy)
    
    # 填充 NaN 值
    df_copy = df_copy.fillna(method='bfill').fillna(method='ffill')
    
    return df_copy

def calculate_basic_indicators(df):
    """計算基本技術指標（不依賴 ta 套件）"""
    df_copy = df.copy()
    
    # 簡單移動平均
    df_copy['SMA_20'] = df_copy['Close'].rolling(window=20).mean()
    df_copy['SMA_50'] = df_copy['Close'].rolling(window=50).mean()
    
    # 簡單 RSI
    delta = df_copy['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df_copy['RSI'] = 100 - (100 / (1 + rs))
    
    # 簡單布林通道
    df_copy['BB_Mid'] = df_copy['Close'].rolling(window=20).mean()
    bb_std = df_copy['Close'].rolling(window=20).std()
    df_copy['BB_High'] = df_copy['BB_Mid'] + (bb_std * 2)
    df_copy['BB_Low'] = df_copy['BB_Mid'] - (bb_std * 2)
    df_copy['BB_Width'] = (df_copy['BB_High'] - df_copy['BB_Low']) / df_copy['BB_Mid']
    
    # EMA
    df_copy['EMA_12'] = df_copy['Close'].ewm(span=12, adjust=False).mean()
    df_copy['EMA_26'] = df_copy['Close'].ewm(span=26, adjust=False).mean()
    
    # MACD
    df_copy['MACD'] = df_copy['EMA_12'] - df_copy['EMA_26']
    df_copy['MACD_Signal'] = df_copy['MACD'].ewm(span=9, adjust=False).mean()
    df_copy['MACD_Diff'] = df_copy['MACD'] - df_copy['MACD_Signal']
    
    # KD 指標
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
    
    # OBV
    df_copy['OBV'] = (np.sign(df_copy['Close'].diff()) * df_copy['Volume']).fillna(0).cumsum()
    
    # ADX (簡化版)
    df_copy['ADX'] = df_copy['ATR'].rolling(window=14).mean() / df_copy['Close'] * 100
    
    # Williams %R
    df_copy['Williams_R'] = -100 * ((high_14 - df_copy['Close']) / (high_14 - low_14))
    
    # 填充 NaN
    df_copy = df_copy.fillna(method='bfill').fillna(method='ffill')
    
    return df_copy

# ==================== 3. LSTM 模型 ====================

def prepare_lstm_data(df, lookback=60):
    """準備 LSTM 訓練數據"""
    if not SKLEARN_AVAILABLE:
        raise ImportError("scikit-learn 未安裝，無法使用 LSTM 功能")
    
    # 選擇特徵
    feature_columns = ['Close', 'Volume', 'MACD', 'RSI', 'BB_Width', 'ATR', 'OBV', 'ADX']
    
    # 確保所有特徵都存在
    available_features = [col for col in feature_columns if col in df.columns]
    data = df[available_features].values
    
    # 標準化
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)
    
    # 創建序列
    X, y = [], []
    for i in range(lookback, len(scaled_data)):
        X.append(scaled_data[i-lookback:i])
        y.append(scaled_data[i, 0])  # 預測收盤價
    
    X, y = np.array(X), np.array(y)
    
    return X, y, scaler, available_features

def build_lstm_model(input_shape):
    """構建 LSTM 模型"""
    if not TF_AVAILABLE:
        raise ImportError("TensorFlow 未安裝，無法使用 LSTM 功能")
    
    model = Sequential([
        Input(shape=input_shape),
        LSTM(128, return_sequences=True),
        Dropout(0.2),
        LSTM(64, return_sequences=True),
        Dropout(0.2),
        LSTM(32, return_sequences=False),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(1)
    ])
    
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

@st.cache_resource
def train_lstm_model(df, lookback=60, epochs=50):
    """訓練 LSTM 模型"""
    if not TF_AVAILABLE or not SKLEARN_AVAILABLE:
        raise ImportError("需要安裝 TensorFlow 和 scikit-learn 才能使用 LSTM 功能")
    
    X, y, scaler, features = prepare_lstm_data(df, lookback)
    
    # 分割訓練集和測試集
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # 構建並訓練模型
    model = build_lstm_model((lookback, X.shape[2]))
    
    with st.spinner('🤖 正在訓練 LSTM 模型...'):
        history = model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=32,
            validation_split=0.1,
            verbose=0,
            callbacks=[
                keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
            ]
        )
    
    # 評估模型
    train_pred = model.predict(X_train, verbose=0)
    test_pred = model.predict(X_test, verbose=0)
    
    train_mae = mean_absolute_error(y_train, train_pred)
    test_mae = mean_absolute_error(y_test, test_pred)
    
    metrics = {
        'train_mae': train_mae,
        'test_mae': test_mae,
        'history': history.history
    }
    
    return model, scaler, features, metrics, (X_test, y_test, test_pred)

def predict_lstm(model, df, scaler, features, lookback=60, days=10):
    """使用 LSTM 進行預測"""
    if not SKLEARN_AVAILABLE:
        raise ImportError("scikit-learn 未安裝，無法使用 LSTM 功能")
    
    # 準備最後 lookback 天的數據
    feature_columns = features
    last_data = df[feature_columns].tail(lookback).values
    scaled_last = scaler.transform(last_data)
    
    # 預測未來
    predictions = []
    current_sequence = scaled_last.copy()
    
    for _ in range(days):
        # 預測下一天
        pred_input = current_sequence.reshape(1, lookback, len(features))
        next_pred = model.predict(pred_input, verbose=0)[0, 0]
        predictions.append(next_pred)
        
        # 更新序列（簡化：只更新價格，其他特徵保持最後值）
        next_row = current_sequence[-1].copy()
        next_row[0] = next_pred
        current_sequence = np.vstack([current_sequence[1:], next_row])
    
    # 反標準化（只取價格列）
    predictions = np.array(predictions).reshape(-1, 1)
    # 創建完整特徵數組進行反轉換
    full_predictions = np.zeros((len(predictions), len(features)))
    full_predictions[:, 0] = predictions.flatten()
    predictions_rescaled = scaler.inverse_transform(full_predictions)[:, 0]
    
    # 生成日期
    last_date = df['Date'].iloc[-1]
    future_dates = [last_date + timedelta(days=i) for i in range(1, days + 1)]
    
    return pd.DataFrame({'Date': future_dates, 'Close': predictions_rescaled})

# ==================== 4. 傳統預測方法（改進版）====================

def predict_traditional(df, sentiment_score, days=10):
    """改進的傳統預測方法"""
    last_price = df['Close'].iloc[-1]
    last_date = df['Date'].iloc[-1]
    
    # 技術指標
    volatility = df['Close'].pct_change().std()
    recent_trend = (df['Close'].iloc[-1] - df['Close'].iloc[-5]) / df['Close'].iloc[-5]
    
    rsi = df['RSI'].iloc[-1] if 'RSI' in df.columns else 50
    macd_diff = df['MACD_Diff'].iloc[-1] if 'MACD_Diff' in df.columns else 0
    
    # 綜合因子
    rsi_bias = (50 - rsi) / 100 * 0.01  # RSI 偏離中性值的影響
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

# ==================== 5. 回測功能 ====================

def backtest_model(df, model_type='lstm', lookback=60, test_days=30):
    """回測模型準確度"""
    results = []
    
    # 選擇測試期間
    test_start_idx = len(df) - test_days - lookback
    
    for i in range(test_start_idx, len(df) - 10):
        train_df = df.iloc[:i+lookback]
        actual_prices = df.iloc[i+lookback:i+lookback+10]['Close'].values
        actual_dates = df.iloc[i+lookback:i+lookback+10]['Date'].values
        
        if len(actual_prices) < 10:
            break
        
        # 進行預測
        if model_type == 'lstm' and 'RSI' in train_df.columns:
            try:
                model, scaler, features, _, _ = train_lstm_model(train_df, lookback=lookback, epochs=20)
                pred_df = predict_lstm(model, train_df, scaler, features, lookback=lookback, days=10)
                predicted_prices = pred_df['Close'].values
            except:
                continue
        else:
            pred_df = predict_traditional(train_df, 0.5, days=10)
            predicted_prices = pred_df['Close'].values
        
        # 計算誤差
        mae = mean_absolute_error(actual_prices, predicted_prices)
        mape = np.mean(np.abs((actual_prices - predicted_prices) / actual_prices)) * 100
        
        results.append({
            'date': actual_dates[0],
            'mae': mae,
            'mape': mape,
            'actual': actual_prices,
            'predicted': predicted_prices
        })
    
    return results

# ==================== 6. 情緒分析 ====================

@st.cache_data(ttl=3600)
def get_finnhub_sentiment(symbol):
    """獲取 Finnhub 市場情緒"""
    clean_symbol = symbol.split('.')[0]
    url = f"https://finnhub.io/api/v1/news-sentiment?symbol={clean_symbol}&token={FINNHUB_API_KEY}"
    try:
        res = requests.get(url, timeout=5).json()
        return res
    except:
        return None

# ==================== 7. 分析報告生成 ====================

def generate_technical_analysis(df):
    """生成技術分析報告"""
    analysis = []
    
    latest = df.iloc[-1]
    
    # RSI 分析
    rsi = latest['RSI']
    if rsi > 70:
        analysis.append("📊 **RSI 超買** (>70)：可能面臨回調壓力")
    elif rsi < 30:
        analysis.append("📊 **RSI 超賣** (<30)：可能出現反彈機會")
    else:
        analysis.append(f"📊 **RSI 正常** ({rsi:.1f})：處於健康區間")
    
    # MACD 分析
    macd_diff = latest['MACD_Diff']
    if macd_diff > 0:
        analysis.append("📈 **MACD 金叉**：多頭訊號")
    else:
        analysis.append("📉 **MACD 死叉**：空頭訊號")
    
    # 布林通道分析
    close = latest['Close']
    bb_high = latest['BB_High']
    bb_low = latest['BB_Low']
    
    if close > bb_high:
        analysis.append("⚠️ **突破布林上軌**：強勢但需警惕過熱")
    elif close < bb_low:
        analysis.append("⚠️ **跌破布林下軌**：超賣但需確認止跌")
    else:
        analysis.append("✅ **布林通道內運行**：正常波動範圍")
    
    # KD 指標
    k = latest['K']
    d = latest['D']
    if k > 80 and d > 80:
        analysis.append("🔴 **KD 高檔鈍化**：短期超買")
    elif k < 20 and d < 20:
        analysis.append("🟢 **KD 低檔鈍化**：短期超賣")
    
    # ADX 趨勢強度
    adx = latest['ADX']
    if adx > 25:
        analysis.append(f"💪 **趨勢強勁** (ADX={adx:.1f})：明顯趨勢")
    else:
        analysis.append(f"😐 **盤整格局** (ADX={adx:.1f})：趨勢不明")
    
    return "\n".join(analysis)

def generate_fundamental_analysis(fundamentals):
    """生成基本面分析"""
    analysis = []
    
    # PE Ratio
    pe = fundamentals.get('PE Ratio', 'N/A')
    if pe != 'N/A' and isinstance(pe, (int, float)):
        if pe < 15:
            analysis.append(f"💰 **本益比低** (PE={pe:.1f})：可能被低估")
        elif pe > 30:
            analysis.append(f"💸 **本益比高** (PE={pe:.1f})：估值偏高，需關注成長性")
        else:
            analysis.append(f"✅ **本益比合理** (PE={pe:.1f})")
    
    # ROE
    roe = fundamentals.get('ROE', 'N/A')
    if roe != 'N/A' and isinstance(roe, (int, float)):
        roe_pct = roe * 100
        if roe_pct > 15:
            analysis.append(f"🎯 **股東權益報酬率優異** (ROE={roe_pct:.1f}%)")
        elif roe_pct < 10:
            analysis.append(f"⚠️ **股東權益報酬率偏低** (ROE={roe_pct:.1f}%)")
    
    # Debt to Equity
    de = fundamentals.get('Debt to Equity', 'N/A')
    if de != 'N/A' and isinstance(de, (int, float)):
        if de < 0.5:
            analysis.append(f"💪 **財務槓桿健康** (負債權益比={de:.2f})")
        elif de > 2:
            analysis.append(f"⚠️ **負債比例較高** (負債權益比={de:.2f})")
    
    return "\n".join(analysis) if analysis else "基本面數據不足"

# ==================== 8. 主程式 UI ====================

def main():
    st.title("📈 AI 股市趨勢分析與預測系統 Pro")
    st.markdown("*整合技術分析、機器學習、基本面分析的專業投資工具*")
    
    # ===== 側邊欄 =====
    st.sidebar.header("⚙️ 設定參數")
    
    # 股票選擇
    analysis_mode = st.sidebar.radio("分析模式", ["單一股票分析", "多股票比較"])
    
    if analysis_mode == "單一股票分析":
        target_stocks = [st.sidebar.text_input("股票代碼", "2330.TW").upper()]
    else:
        stock_input = st.sidebar.text_area(
            "股票代碼 (每行一個)",
            "2330.TW\n2317.TW\n2454.TW"
        )
        target_stocks = [s.strip().upper() for s in stock_input.split('\n') if s.strip()]
    
    # 預測參數
    st.sidebar.subheader("📊 預測設定")
    forecast_days = st.sidebar.slider("預測天數", 5, 30, 10)
    model_choice = st.sidebar.selectbox("預測模型", ["LSTM 深度學習", "傳統技術分析", "混合模型"])
    
    # 進階選項
    with st.sidebar.expander("🔬 進階選項"):
        show_backtest = st.checkbox("啟用回測驗證", value=False)
        show_fundamentals = st.checkbox("顯示基本面分析", value=True)
        lstm_epochs = st.slider("LSTM 訓練輪數", 20, 100, 50)
    
    # ===== 主要內容 =====
    
    if analysis_mode == "單一股票分析":
        # ===== 單一股票分析 =====
        symbol = target_stocks[0]
        
        # 獲取數據
        with st.spinner(f'正在獲取 {symbol} 數據...'):
            df = get_stock_data(symbol, period="1y")
            if df is None:
                st.error("❌ 無法獲取數據，請檢查股票代碼")
                return
            
            df = calculate_technical_indicators(df)
            sentiment_data = get_finnhub_sentiment(symbol)
            sent_score = sentiment_data['sentiment'].get('bullishPercent', 0.5) if sentiment_data and 'sentiment' in sentiment_data else 0.5
            
            if show_fundamentals:
                fundamentals, info = get_fundamental_data(symbol)
            else:
                fundamentals = None
        
        # 模型預測
        if model_choice == "LSTM 深度學習":
            if not TF_AVAILABLE or not SKLEARN_AVAILABLE:
                st.error("❌ LSTM 需要安裝 TensorFlow 和 scikit-learn。請執行：pip install tensorflow scikit-learn")
                st.info("⏳ 自動切換至傳統技術分析方法...")
                future_df = predict_traditional(df, sent_score, days=forecast_days)
                model_name = "Traditional"
                metrics = None
            else:
                try:
                    model, scaler, features, metrics, test_data = train_lstm_model(df, epochs=lstm_epochs)
                    future_df = predict_lstm(model, df, scaler, features, days=forecast_days)
                    model_name = "LSTM"
                except Exception as e:
                    st.warning(f"⚠️ LSTM 訓練失敗，切換至傳統方法: {str(e)}")
                    future_df = predict_traditional(df, sent_score, days=forecast_days)
                    model_name = "Traditional"
                    metrics = None
        elif model_choice == "傳統技術分析":
            future_df = predict_traditional(df, sent_score, days=forecast_days)
            model_name = "Traditional"
            metrics = None
        else:  # 混合模型
            if not TF_AVAILABLE or not SKLEARN_AVAILABLE:
                st.warning("⚠️ 混合模型需要 LSTM 支援，自動切換至傳統方法")
                future_df = predict_traditional(df, sent_score, days=forecast_days)
                model_name = "Traditional"
                metrics = None
            else:
                try:
                    model, scaler, features, metrics, test_data = train_lstm_model(df, epochs=lstm_epochs)
                    lstm_pred = predict_lstm(model, df, scaler, features, days=forecast_days)
                    trad_pred = predict_traditional(df, sent_score, days=forecast_days)
                    # 混合：70% LSTM + 30% 傳統
                    future_df = lstm_pred.copy()
                    future_df['Close'] = 0.7 * lstm_pred['Close'] + 0.3 * trad_pred['Close']
                    model_name = "Hybrid"
                except Exception as e:
                    st.warning(f"⚠️ 混合模型建立失敗: {str(e)}")
                    future_df = predict_traditional(df, sent_score, days=forecast_days)
                    model_name = "Traditional"
                    metrics = None
        
        # ===== 圖表展示 =====
        st.subheader(f"📊 {symbol} 技術分析與預測")
        
        # 主圖表
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
        
        # 移動平均線
        fig.add_trace(go.Scatter(x=df['Date'][-90:], y=df['SMA_20'][-90:], name='SMA 20', line=dict(color='orange', width=1)))
        fig.add_trace(go.Scatter(x=df['Date'][-90:], y=df['SMA_50'][-90:], name='SMA 50', line=dict(color='blue', width=1)))
        
        # 布林通道
        fig.add_trace(go.Scatter(x=df['Date'][-90:], y=df['BB_High'][-90:], name='布林上軌', line=dict(color='gray', width=1, dash='dot')))
        fig.add_trace(go.Scatter(x=df['Date'][-90:], y=df['BB_Low'][-90:], name='布林下軌', line=dict(color='gray', width=1, dash='dot'), fill='tonexty'))
        
        # 預測線
        connect_df = pd.concat([df.tail(1)[['Date', 'Close']], future_df])
        fig.add_trace(go.Scatter(
            x=connect_df['Date'],
            y=connect_df['Close'],
            mode='lines+markers',
            line=dict(color='red', width=3, dash='dot'),
            marker=dict(size=8),
            name=f'{model_name} 預測'
        ))
        
        fig.update_layout(
            xaxis_rangeslider_visible=False,
            height=600,
            template="plotly_dark",
            hovermode='x unified',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # ===== 技術指標面板 =====
        tab1, tab2, tab3, tab4 = st.tabs(["📈 技術指標", "🧠 預測分析", "📊 基本面", "🔍 回測驗證"])
        
        with tab1:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### RSI & MACD")
                
                # RSI 圖
                fig_rsi = go.Figure()
                fig_rsi.add_trace(go.Scatter(x=df['Date'][-90:], y=df['RSI'][-90:], name='RSI', line=dict(color='purple')))
                fig_rsi.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="超買")
                fig_rsi.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="超賣")
                fig_rsi.update_layout(height=250, template="plotly_dark", showlegend=False)
                st.plotly_chart(fig_rsi, use_container_width=True)
                
                # MACD 圖
                fig_macd = go.Figure()
                fig_macd.add_trace(go.Scatter(x=df['Date'][-90:], y=df['MACD'][-90:], name='MACD', line=dict(color='blue')))
                fig_macd.add_trace(go.Scatter(x=df['Date'][-90:], y=df['MACD_Signal'][-90:], name='Signal', line=dict(color='orange')))
                fig_macd.add_trace(go.Bar(x=df['Date'][-90:], y=df['MACD_Diff'][-90:], name='Histogram'))
                fig_macd.update_layout(height=250, template="plotly_dark")
                st.plotly_chart(fig_macd, use_container_width=True)
            
            with col2:
                st.markdown("### KD & 成交量")
                
                # KD 圖
                fig_kd = go.Figure()
                fig_kd.add_trace(go.Scatter(x=df['Date'][-90:], y=df['K'][-90:], name='K', line=dict(color='blue')))
                fig_kd.add_trace(go.Scatter(x=df['Date'][-90:], y=df['D'][-90:], name='D', line=dict(color='orange')))
                fig_kd.add_hline(y=80, line_dash="dash", line_color="red")
                fig_kd.add_hline(y=20, line_dash="dash", line_color="green")
                fig_kd.update_layout(height=250, template="plotly_dark")
                st.plotly_chart(fig_kd, use_container_width=True)
                
                # 成交量圖
                fig_vol = go.Figure()
                colors = ['red' if df['Close'].iloc[i] < df['Open'].iloc[i] else 'green' for i in range(-90, 0)]
                fig_vol.add_trace(go.Bar(x=df['Date'][-90:], y=df['Volume'][-90:], marker_color=colors, name='成交量'))
                fig_vol.update_layout(height=250, template="plotly_dark", showlegend=False)
                st.plotly_chart(fig_vol, use_container_width=True)
        
        with tab2:
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.markdown("### 📉 預測摘要")
                
                current_price = df['Close'].iloc[-1]
                predicted_price = future_df['Close'].iloc[-1]
                change_pct = ((predicted_price - current_price) / current_price) * 100
                
                st.metric("當前價格", f"${current_price:.2f}")
                st.metric(
                    f"{forecast_days} 日後預測",
                    f"${predicted_price:.2f}",
                    f"{change_pct:+.2f}%"
                )
                
                st.markdown("**模型資訊：**")
                st.write(f"- 使用模型：`{model_name}`")
                st.write(f"- 市場情緒：`{sent_score:.2f}`")
                
                if metrics:
                    st.write(f"- 訓練誤差：`{metrics['train_mae']:.4f}`")
                    st.write(f"- 測試誤差：`{metrics['test_mae']:.4f}`")
            
            with col2:
                st.markdown("### 🎯 技術面分析")
                tech_analysis = generate_technical_analysis(df)
                st.markdown(tech_analysis)
        
        with tab3:
            if fundamentals:
                st.markdown("### 💼 基本面數據")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("本益比 (PE)", f"{fundamentals['PE Ratio']:.2f}" if isinstance(fundamentals['PE Ratio'], (int, float)) else "N/A")
                    st.metric("股價淨值比 (PB)", f"{fundamentals['Price to Book']:.2f}" if isinstance(fundamentals['Price to Book'], (int, float)) else "N/A")
                    st.metric("殖利率", f"{fundamentals['Dividend Yield']*100:.2f}%" if isinstance(fundamentals['Dividend Yield'], (int, float)) else "N/A")
                
                with col2:
                    st.metric("股東權益報酬率 (ROE)", f"{fundamentals['ROE']*100:.2f}%" if isinstance(fundamentals['ROE'], (int, float)) else "N/A")
                    st.metric("資產報酬率 (ROA)", f"{fundamentals['ROA']*100:.2f}%" if isinstance(fundamentals['ROA'], (int, float)) else "N/A")
                    st.metric("利潤率", f"{fundamentals['Profit Margins']*100:.2f}%" if isinstance(fundamentals['Profit Margins'], (int, float)) else "N/A")
                
                with col3:
                    st.metric("負債權益比", f"{fundamentals['Debt to Equity']:.2f}" if isinstance(fundamentals['Debt to Equity'], (int, float)) else "N/A")
                    st.metric("流動比率", f"{fundamentals['Current Ratio']:.2f}" if isinstance(fundamentals['Current Ratio'], (int, float)) else "N/A")
                    st.metric("營收成長", f"{fundamentals['Revenue Growth']*100:.2f}%" if isinstance(fundamentals['Revenue Growth'], (int, float)) else "N/A")
                
                st.markdown("### 📝 基本面評析")
                fund_analysis = generate_fundamental_analysis(fundamentals)
                st.markdown(fund_analysis)
            else:
                st.info("未啟用基本面分析或數據不可用")
        
        with tab4:
            if show_backtest:
                st.markdown("### 🔬 回測驗證結果")
                
                with st.spinner('正在執行回測...'):
                    backtest_results = backtest_model(df, model_type='lstm' if model_choice == "LSTM 深度學習" else 'traditional')
                
                if backtest_results:
                    mae_list = [r['mape'] for r in backtest_results]
                    avg_mape = np.mean(mae_list)
                    
                    col1, col2, col3 = st.columns(3)
                    col1.metric("平均誤差 (MAPE)", f"{avg_mape:.2f}%")
                    col2.metric("最佳預測", f"{min(mae_list):.2f}%")
                    col3.metric("最差預測", f"{max(mae_list):.2f}%")
                    
                    # 誤差分布圖
                    fig_backtest = go.Figure()
                    fig_backtest.add_trace(go.Scatter(
                        x=[r['date'] for r in backtest_results],
                        y=mae_list,
                        mode='lines+markers',
                        name='MAPE'
                    ))
                    fig_backtest.update_layout(
                        title="回測誤差率變化",
                        yaxis_title="MAPE (%)",
                        template="plotly_dark",
                        height=400
                    )
                    st.plotly_chart(fig_backtest, use_container_width=True)
                    
                    st.success(f"✅ 回測完成！平均預測誤差：{avg_mape:.2f}%")
                else:
                    st.warning("回測數據不足")
            else:
                st.info("請在側邊欄啟用回測功能")
    
    else:
        # ===== 多股票比較 =====
        st.subheader("📊 多股票比較分析")
        
        comparison_data = {}
        
        for symbol in target_stocks:
            with st.spinner(f'正在獲取 {symbol} 數據...'):
                df = get_stock_data(symbol, period="6mo")
                if df is not None:
                    df = calculate_technical_indicators(df)
                    comparison_data[symbol] = df
        
        if len(comparison_data) > 0:
            # 價格走勢比較
            fig_compare = go.Figure()
            
            for symbol, df in comparison_data.items():
                # 正規化價格（以第一天為基準100）
                normalized_price = (df['Close'] / df['Close'].iloc[0]) * 100
                fig_compare.add_trace(go.Scatter(
                    x=df['Date'],
                    y=normalized_price,
                    name=symbol,
                    mode='lines'
                ))
            
            fig_compare.update_layout(
                title="股價走勢比較（正規化至100）",
                yaxis_title="相對表現",
                template="plotly_dark",
                height=500,
                hovermode='x unified'
            )
            st.plotly_chart(fig_compare, use_container_width=True)
            
            # 技術指標比較表
            st.markdown("### 📊 技術指標對比")
            
            comparison_table = []
            for symbol, df in comparison_data.items():
                latest = df.iloc[-1]
                comparison_table.append({
                    '股票代碼': symbol,
                    '當前價格': f"${latest['Close']:.2f}",
                    'RSI': f"{latest['RSI']:.1f}",
                    'MACD': f"{latest['MACD']:.2f}",
                    'K值': f"{latest['K']:.1f}",
                    'D值': f"{latest['D']:.1f}",
                    'ADX': f"{latest['ADX']:.1f}",
                    '20日均線': f"${latest['SMA_20']:.2f}",
                })
            
            st.dataframe(pd.DataFrame(comparison_table), use_container_width=True)
            
            # 績效比較
            st.markdown("### 📈 績效統計")
            
            perf_table = []
            for symbol, df in comparison_data.items():
                returns = df['Close'].pct_change()
                perf_table.append({
                    '股票代碼': symbol,
                    '近5日報酬': f"{((df['Close'].iloc[-1] / df['Close'].iloc[-6]) - 1) * 100:+.2f}%",
                    '近20日報酬': f"{((df['Close'].iloc[-1] / df['Close'].iloc[-21]) - 1) * 100:+.2f}%",
                    '波動率': f"{returns.std() * np.sqrt(252) * 100:.2f}%",
                    '最大回撤': f"{(df['Close'] / df['Close'].cummax() - 1).min() * 100:.2f}%",
                })
            
            st.dataframe(pd.DataFrame(perf_table), use_container_width=True)
        else:
            st.error("無法獲取任何股票數據")
    
    # ===== 頁腳 =====
    st.markdown("---")
    st.caption("⚠️ **免責聲明**：本系統僅供學習與研究使用，不構成投資建議。模型預測存在不確定性，實際投資請審慎評估風險。")

if __name__ == "__main__":
    main()
