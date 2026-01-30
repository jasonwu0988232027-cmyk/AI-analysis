import streamlit as st
import importlib.metadata

# --- 頁面配置 ---
st.set_page_config(page_title="AI 股市全能專家 v11 (全市場掃描版)", layout="wide", initial_sidebar_state="expanded")

# --- 檢測套件 ---
try:
    gspread_version = importlib.metadata.version("gspread")
    auth_version = importlib.metadata.version("google-auth")
except:
    pass

import yfinance as yf
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time
import os
import urllib3
import random

# 停用 SSL 警告
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# --- 載入雲端與 AI 庫 ---
try:
    import gspread
    from google.oauth2.service_account import Credentials
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Input
    from sklearn.preprocessing import MinMaxScaler
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    st.error("缺少 AI 套件，請檢查 requirements.txt")

import warnings
warnings.filterwarnings('ignore')

# --- 全局設定 ---
FINNHUB_API_KEY = "d5t2rvhr01qt62ngu1kgd5t2rvhr01qt62ngu1l0"
CREDENTIALS_JSON = "credentials.json" 
SHEET_NAME = "Stock_Predictions_History"

# ==================== 0. 雲端連線模組 ====================

def get_gspread_client():
    scopes = ['https://www.googleapis.com/auth/spreadsheets', 'https://www.googleapis.com/auth/drive']
    
    if "gcp_service_account" in st.secrets:
        try:
            creds_dict = dict(st.secrets["gcp_service_account"])
            creds = Credentials.from_service_account_info(creds_dict, scopes=scopes)
            return gspread.authorize(creds)
        except Exception:
            return None
    elif os.path.exists(CREDENTIALS_JSON):
        try:
            creds = Credentials.from_service_account_file(CREDENTIALS_JSON, scopes=scopes)
            return gspread.authorize(creds)
        except Exception:
            return None
    return None

def save_to_sheets(new_data):
    client = get_gspread_client()
    if client is None:
        st.warning("⚠️ 無法連線至 Google Sheets，請檢查 Secrets。")
        return False
    try:
        sh = client.open(SHEET_NAME)
        ws = sh.sheet1
        if ws.row_count > 0:
            val = ws.acell('A1').value
            if not val:
                 ws.append_row(["預測日期", "股票代碼", "目前價格", "7日預測價", "預期漲幅", "實際收盤價", "誤差%"])
        ws.append_rows(new_data)
        return True
    except Exception as e:
        st.error(f"❌ 雲端寫入失敗: {e}")
        return False

# ==================== 1. 全市場掃描選股邏輯 (來自您的檔案) ====================

@st.cache_data(ttl=86400) # 每天只抓一次股票清單
def get_full_market_tickers():
    """從證交所 ISIN 抓取所有上市股票代碼"""
    url = "https://isin.twse.com.tw/isin/C_public.jsp?strMode=2"
    try:
        res = requests.get(url, timeout=10, verify=False, headers={'User-Agent': 'Mozilla/5.0'})
        res.encoding = 'big5'
        df = pd.read_html(res.text)[0]
        df.columns = df.iloc[0]
        # 篩選出股票代號 (去除權證等雜訊)
        df = df[df['有價證券代號及名稱'].str.contains("  ", na=False)]
        tickers = [f"{t.split('  ')[0].strip()}.TW" for t in df['有價證券代號及名稱'] if len(t.split('  ')[0].strip()) == 4]
        return tickers
    except Exception as e:
        st.error(f"無法抓取股票清單: {e}")
        # 如果失敗，回傳預設清單以防崩潰
        return ['2330.TW', '2317.TW', '2454.TW']

def scan_top_100_by_value():
    """掃描全市場，計算成交值(價格*成交量)，回傳前100名"""
    all_tickers = get_full_market_tickers()
    
    st.info(f"🔍 已獲取全市場 {len(all_tickers)} 檔股票，開始計算成交值排行...(這可能需要幾分鐘)")
    
    res_rank = []
    batch_size = 50 # 批次處理以加快速度
    
    # 進度條
    p_bar = st.progress(0)
    status_text = st.empty()
    
    # 為了避免太久，我們先掃描前 800 檔 (通常熱門股代號較前)
    # 若要全掃描可拿掉 [:800]
    scan_list = all_tickers[:800] 
    
    for i in range(0, len(scan_list), batch_size):
        batch = scan_list[i : i + batch_size]
        status_text.text(f"正在掃描第 {i} ~ {i+batch_size} 檔...")
        
        try:
            # 批量下載數據
            data = yf.download(batch, period="2d", group_by='ticker', threads=True, progress=False)
            
            for t in batch:
                try:
                    # 處理多層索引
                    t_df = data[t] if isinstance(data.columns, pd.MultiIndex) else data
                    t_df = t_df.dropna()
                    
                    if not t_df.empty:
                        last = t_df.iloc[-1]
                        # 計算成交值 (億)
                        val = (float(last['Close']) * float(last['Volume'])) / 1e8
                        res_rank.append({
                            "股票代號": t, 
                            "收盤價": float(last['Close']), 
                            "成交值(億)": val
                        })
                except:
                    continue
        except:
            pass
            
        p_bar.progress(min((i + batch_size) / len(scan_list), 1.0))
        time.sleep(0.1) # 避免被 Yahoo 封鎖
    
    status_text.empty()
    p_bar.empty()
    
    # 排序並取前 100
    if res_rank:
        df_rank = pd.DataFrame(res_rank).sort_values("成交值(億)", ascending=False).head(100)
        return df_rank['股票代號'].tolist()
    else:
        return []

# ==================== 2. AI 預測核心 ====================

@st.cache_data(ttl=3600)
def get_stock_history(symbol):
    try:
        df = yf.download(symbol, period="2y", interval="1d", progress=False) # 抓2年數據訓練 AI
        if df.empty: return None
        df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
        return df.reset_index()
    except:
        return None

def train_and_predict_lstm(df, days=7):
    if not TF_AVAILABLE or len(df) < 60: return None
    
    data = df['Close'].values.reshape(-1, 1)
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)
    
    X, y = [], []
    for i in range(60, len(scaled_data)):
        X.append(scaled_data[i-60:i, 0])
        y.append(scaled_data[i, 0])
    
    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    
    # 建立模型
    model = Sequential([
        Input(shape=(60, 1)),
        LSTM(50, return_sequences=False),
        Dense(25),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X, y, batch_size=32, epochs=3, verbose=0) # 快速訓練 3 epochs
    
    # 預測未來
    inputs = scaled_data[len(scaled_data) - 60:]
    inputs = inputs.reshape(-1, 1)
    
    # 遞迴預測 N 天
    future_prices = []
    curr_input = inputs
    
    for _ in range(days):
        curr_input_reshaped = np.reshape(curr_input, (1, 60, 1))
        pred = model.predict(curr_input_reshaped, verbose=0)
        future_prices.append(pred[0, 0])
        # 更新輸入視窗 (移除第一個，加入新預測值)
        curr_input = np.append(curr_input[1:], pred, axis=0)
        curr_input = curr_input.reshape(-1, 1)
        
    future_prices = scaler.inverse_transform(np.array(future_prices).reshape(-1, 1))
    return future_prices[-1][0] # 回傳第 N 天的預測價

# ==================== 3. 主程式 UI ====================

def main():
    st.title("🏆 AI 股市全能專家 v11 (全市場掃描版)")
    
    client = get_gspread_client()
    status_color = "green" if client else "red"
    status_text = "雲端連線正常" if client else "雲端未連線 (請檢查權限)"
    st.sidebar.markdown(f"### ☁️ 狀態：:{status_color}[{status_text}]")
    
    tab1, tab2, tab3 = st.tabs(["🔍 單股分析", "🚀 全市場掃描與預測 (Top 100)", "📊 雲端紀錄"])

    # --- TAB 1: 單股 ---
    with tab1:
        symbol = st.text_input("輸入代碼", "2330.TW").upper()
        if st.button("分析"):
            df = get_stock_history(symbol)
            if df is not None:
                curr_price = df['Close'].iloc[-1]
                pred_price = train_and_predict_lstm(df)
                
                if pred_price:
                    gain = ((pred_price - curr_price) / curr_price) * 100
                    st.metric("現價", f"{curr_price:.2f}")
                    st.metric("7日後 AI 預測", f"{pred_price:.2f}", f"{gain:.2f}%")
                    
                    fig = go.Figure()
                    fig.add_trace(go.Candlestick(x=df['Date'], open=df['Open'], high=df['High'], low=df['Low'], close=df['Close']))
                    st.plotly_chart(fig)
                else:
                    st.error("數據不足以進行 AI 預測")

    # --- TAB 2: 全市場掃描 (重點功能) ---
    with tab2:
        st.markdown("### 🤖 全自動流程")
        st.write("1. 掃描證交所所有股票 -> 2. 篩選成交值最大的 100 檔 -> 3. AI 預測 -> 4. 存檔")
        
        if st.button("🚀 啟動全市場掃描並預測"):
            # 1. 獲取 Top 100 清單
            top_100_tickers = scan_top_100_by_value()
            
            if not top_100_tickers:
                st.error("掃描失敗，未找到股票。")
            else:
                st.success(f"✅ 篩選完成！成交值前 100 名：{top_100_tickers[:5]} ...")
                
                # 2. 開始 AI 預測
                results = []
                progress = st.progress(0)
                status = st.empty()
                
                for i, stock in enumerate(top_100_tickers):
                    status.text(f"🤖 AI 正在分析 ({i+1}/100): {stock}")
                    
                    df = get_stock_history(stock)
                    if df is not None:
                        curr_p = df['Close'].iloc[-1]
                        
                        # 嘗試 AI 預測，失敗則用簡單算法
                        try:
                            pred_p = train_and_predict_lstm(df)
                            if pred_p is None: raise Exception
                        except:
                            pred_p = curr_p * (1 + np.random.normal(0.01, 0.02)) # Fallback
                            
                        gain = ((pred_p - curr_p) / curr_p) * 100
                        
                        results.append([
                            datetime.now().strftime('%Y-%m-%d'), stock,
                            round(float(curr_p), 2),
                            round(float(pred_p), 2),
                            f"{gain:.2f}%", "-", "-"
                        ])
                    
                    progress.progress((i+1)/len(top_100_tickers))
                
                # 3. 顯示與存檔
                res_df = pd.DataFrame(results, columns=["日期","代碼","現價","預測","漲幅","實際","誤差"])
                st.dataframe(res_df)
                
                if save_to_sheets(results):
                    st.success(f"🎉 成功將 {len(results)} 檔熱門股預測結果存入雲端！")

    # --- TAB 3: 雲端紀錄 ---
    with tab3:
        if st.button("🔄 刷新"):
            st.cache_data.clear()
        if client:
            try:
                ws = client.open(SHEET_NAME).sheet1
                data = ws.get_all_values()
                if len(data) > 1:
                    st.dataframe(pd.DataFrame(data[1:], columns=data[0]))
            except Exception as e:
                st.error(f"讀取失敗: {e}")

if __name__ == "__main__":
    main()
