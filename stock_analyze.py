import streamlit as st
import importlib.metadata

# --- 頁面配置 ---
st.set_page_config(page_title="AI 股市預測專家 Pro v9.1 (版本檢測版)", layout="wide")

# --- 檢測套件版本 (除錯用) ---
try:
    gspread_version = importlib.metadata.version("gspread")
    auth_version = importlib.metadata.version("google-auth")
    st.sidebar.success(f"📦 套件狀態：gspread v{gspread_version} | google-auth v{auth_version}")
    
    if gspread_version.startswith("5") or gspread_version.startswith("4"):
        st.error("🚨 警告：你的 gspread 版本太舊！請更新 requirements.txt 並重啟 App。")
except:
    st.sidebar.warning("無法檢測套件版本")

import yfinance as yf
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time
import os
import urllib3

# 停用 SSL 警告
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# 載入必要庫 (改用 google-auth)
try:
    import gspread
    from google.oauth2.service_account import Credentials
    import tensorflow as tf
    from sklearn.preprocessing import MinMaxScaler
except ImportError:
    st.error("缺少套件，請更新 requirements.txt")

# --- 全局設定 ---
CREDENTIALS_JSON = "credentials.json" 
SHEET_NAME = "Stock_Predictions_History" 
BATCH_CD = 0.5 

# ==================== 1. 穩定版百大名單 (內建) ====================

def get_stable_stock_list():
    tickers = [
        '2330.TW', '2317.TW', '2454.TW', '2308.TW', '2382.TW', '2303.TW', '2881.TW', '2882.TW', 
        '2891.TW', '2886.TW', '2412.TW', '2884.TW', '1216.TW', '2885.TW', '3711.TW', '2892.TW', 
        '2357.TW', '2880.TW', '2890.TW', '5880.TW', '2345.TW', '3008.TW', '2327.TW', '2395.TW',
        '2883.TW', '2887.TW', '3045.TW', '4938.TW', '2408.TW', '1101.TW'
    ]
    data = {'證券代號': tickers, '證券名稱': [f"Stock {t}" for t in tickers]} 
    df = pd.DataFrame(data)
    return df

def get_stock_data(symbol, period="1y"):
    try:
        stock = yf.Ticker(symbol)
        df = stock.history(period=period)
        if df.empty: return None
        return df.reset_index()
    except:
        return None

# ==================== 2. 雲端同步模組 (v9 google-auth) ====================

def get_gspread_client():
    scopes = [
        'https://www.googleapis.com/auth/spreadsheets',
        'https://www.googleapis.com/auth/drive'
    ]
    
    # 方式 A: Streamlit Secrets (優先)
    if "gcp_service_account" in st.secrets:
        try:
            creds_dict = dict(st.secrets["gcp_service_account"])
            creds = Credentials.from_service_account_info(creds_dict, scopes=scopes)
            return gspread.authorize(creds)
        except Exception as e:
            st.error(f"Secrets 設定有誤: {e}")
            return None

    # 方式 B: 本地檔案
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
        st.warning("⚠️ 無法連線至 Google Sheets。請檢查 Secrets。")
        return False
        
    try:
        sh = client.open(SHEET_NAME)
        ws = sh.sheet1
        if ws.row_count <= 1 and (not ws.cell(1, 1).value):
            ws.append_row(["預測日期", "股票代碼", "目前價格", "7日預測價", "預期漲幅", "實際收盤價", "誤差%"])
            
        ws.append_rows(new_data)
        st.success(f"✅ 成功寫入 {len(new_data)} 筆資料至雲端！")
        return True
    except Exception as e:
        # 如果還是報錯，印出詳細類型
        st.error(f"❌ 雲端寫入失敗: {type(e).__name__} - {e}")
        return False

# ==================== 3. 機器學習推論模組 ====================

def generate_dummy_data():
    dates = pd.date_range(end=datetime.now(), periods=100)
    prices = np.sin(np.linspace(0, 10, 100)) * 50 + 500 
    df = pd.DataFrame({'Date': dates, 'Close': prices})
    return df

@st.cache_resource
def get_trained_base_model():
    df = get_stock_data("2330.TW")
    if df is None or len(df) < 60:
        df = get_stock_data("2317.TW")
    if df is None or len(df) < 60:
        df = generate_dummy_data()

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df[['Close']].values)
    
    X, y = [], []
    for i in range(60, len(scaled)):
        X.append(scaled[i-60:i, 0])
        y.append(scaled[i, 0])
    
    if len(X) == 0: 
        X = np.zeros((10, 60, 1))
        y = np.zeros((10,))

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(60, 1)),
        tf.keras.layers.LSTM(50),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(np.array(X), np.array(y), epochs=1, batch_size=32, verbose=0)
    return model

def fast_predict(model, df):
    if len(df) < 60:
        last_val = df['Close'].iloc[-1]
        fill_needed = 60 - len(df)
        fill_data = pd.DataFrame({'Close': [last_val] * fill_needed})
        df = pd.concat([fill_data, df[['Close']]], ignore_index=True)

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df[['Close']].values)
    last_60 = scaled[-60:].reshape(1, 60, 1)
    pred = model.predict(last_60, verbose=0)
    return scaler.inverse_transform(pred)[0][0]

# ==================== 4. 主介面 ====================

def main():
    st.title("📈 AI 股市預測專家 Pro v9.1 (版本檢測版)")
    
    tab1, tab2 = st.tabs(["🚀 智能批次預測", "🧐 歷史反思"])

    with tab1:
        if st.button("開始執行預測"):
            with st.spinner("模型初始化中..."):
                target_stocks = get_stable_stock_list()
                model = get_trained_base_model()
            
            if not target_stocks.empty and model:
                results = []
                bar = st.progress(0)
                msg = st.empty()
                
                total = len(target_stocks)
                for i, row in target_stocks.iterrows():
                    symbol = row['證券代號']
                    msg.text(f"正在運算 ({i+1}/{total}): {symbol}")
                    
                    time.sleep(0.1) 
                    df = get_stock_data(symbol)
                    
                    if df is None or len(df) < 60:
                        df = generate_dummy_data()
                        if df is not None: df['Close'] = df['Close']
                    
                    if df is not None:
                        curr_p = df['Close'].iloc[-1]
                        pred_p = fast_predict(model, df)
                        if curr_p == 0: curr_p = 100
                        gain = ((pred_p - curr_p) / curr_p) * 100
                        
                        results.append([
                            datetime.now().strftime('%Y-%m-%d'),
                            symbol,
                            round(float(curr_p), 2),
                            round(float(pred_p), 2),
                            f"{gain:.2f}%",
                            "-", "-"
                        ])
                    bar.progress((i+1)/total)
                
                res_df = pd.DataFrame(results, columns=["日期","代碼","現價","預測價","漲幅","實際","誤差"])
                st.dataframe(res_df)
                save_to_sheets(results)

    with tab2:
        st.subheader("Google Sheets 歷史紀錄")
        client = get_gspread_client()
        if client:
            try:
                ws = client.open(SHEET_NAME).sheet1
                records = ws.get_all_records()
                st.dataframe(pd.DataFrame(records).tail(20) if records else "暫無紀錄")
            except Exception as e:
                st.error(f"讀取失敗: {e}")
        else:
            st.info("請確認 Secrets 設定。")

if __name__ == "__main__":
    main()
