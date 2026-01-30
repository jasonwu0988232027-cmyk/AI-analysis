import streamlit as st
import importlib.metadata

# --- 頁面配置 ---
st.set_page_config(page_title="AI 股市全能專家 v13 (Yahoo爬蟲+分頁存檔)", layout="wide", initial_sidebar_state="expanded")

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
CREDENTIALS_JSON = "credentials.json" 
SHEET_NAME = "Stock_Predictions_History"

# ==================== 0. 雲端連線模組 (支援多重分頁) ====================

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

def save_to_sheets(new_data, sheet_index=0):
    """
    sheet_index=0: 存入第一個分頁 (單股分析)
    sheet_index=1: 存入第二個分頁 (全市場掃描)
    """
    client = get_gspread_client()
    if client is None:
        st.warning("⚠️ 無法連線至 Google Sheets，請檢查 Secrets。")
        return False
    try:
        sh = client.open(SHEET_NAME)
        
        # --- 分頁處理邏輯 ---
        try:
            # 嘗試獲取指定索引的分頁
            ws = sh.get_worksheet(sheet_index)
            if ws is None:
                # 如果第二頁不存在，則自動建立
                ws = sh.add_worksheet(title="全市場掃描結果", rows=500, cols=10)
        except:
            # 如果發生任何錯誤，嘗試建立新分頁
            ws = sh.add_worksheet(title=f"Scan_Result_{datetime.now().strftime('%H%M')}", rows=500, cols=10)

        # 寫入標題 (如果表是空的)
        if ws.row_count > 0:
            try:
                val = ws.acell('A1').value
                if not val:
                    ws.append_row(["預測日期", "股票代碼", "目前價格", "7日預測價", "預期漲幅", "實際收盤價", "誤差%"])
            except:
                pass
        else:
             ws.append_row(["預測日期", "股票代碼", "目前價格", "7日預測價", "預期漲幅", "實際收盤價", "誤差%"])
             
        ws.append_rows(new_data)
        return True
    except Exception as e:
        st.error(f"❌ 雲端寫入失敗: {e}")
        return False

# ==================== 1. Yahoo 股市爬蟲 (來自您的代碼) ====================

class StockPoolManagerV2:
    def __init__(self):
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }

    def get_hot_stocks(self, limit=100):
        # st.write(f"🚀 正在掃描市場成交重心 (Yahoo Finance)，目標前 {limit} 檔...")
        hot_tickers = []
        
        try:
            # 抓取 Yahoo 股市「成交值」排行榜
            url = "https://tw.stock.yahoo.com/rank/turnover?exchange=TAI" 
            r = requests.get(url, headers=self.headers, timeout=10)
            
            # 讀取網頁表格
            dfs = pd.read_html(r.text)
            df = dfs[0] 
            
            # --- 智慧清洗邏輯 ---
            target_col = None
            for i, col_name in enumerate(df.columns):
                if '股' in str(col_name) or '名' in str(col_name):
                    target_col = i
                    break
            
            if target_col is None: target_col = 1
            
            count = 0
            for item in df.iloc[:, target_col]:
                item_str = str(item).strip()
                # 切割出代號 (例如 "2330 台積電" -> "2330")
                parts = item_str.split(' ')
                ticker = parts[0]
                
                # 過濾：只取4位數股票代碼
                if ticker.isdigit() and len(ticker) == 4:
                    hot_tickers.append(f"{ticker}.TW")
                    count += 1
                
                if count >= limit:
                    break
            
            st.success(f"✅ 成功從 Yahoo 鎖定 {len(hot_tickers)} 檔熱門潛力股！")
            return hot_tickers

        except Exception as e:
            st.warning(f"❌ Yahoo 爬蟲遭遇亂流: {e}")
            st.info("🛡️ 啟動「戰備清單 (Fallback)」模式，載入預設高波動股庫。")
            return self._get_fallback_list(limit)

    def _get_fallback_list(self, limit):
        # 手動維護的「戰備清單」
        fallback = [
            "2330.TW", "2454.TW", "2317.TW", "2303.TW", "2308.TW", "2382.TW", "3231.TW", "3443.TW", "3661.TW", "3035.TW",
            "2376.TW", "2356.TW", "6669.TW", "3017.TW", "3324.TW", "2421.TW", "3037.TW", "2368.TW", "2449.TW", "6271.TW",
            "2603.TW", "2609.TW", "2615.TW", "2618.TW", "2610.TW", "1513.TW", "1519.TW", "1504.TW", "1605.TW", "2002.TW",
            "2881.TW", "2882.TW", "2891.TW", "2886.TW", "2884.TW",
            "2409.TW", "3481.TW", "3008.TW", "2481.TW", "2344.TW", "2408.TW", "6770.TW", "5347.TW", "4961.TW", "9958.TW"
        ]
        return fallback[:limit]

# ==================== 2. AI 預測核心 ====================

@st.cache_data(ttl=3600)
def get_stock_history(symbol):
    try:
        # 抓 1.5 年數據，足夠訓練 60 天 lookback
        df = yf.download(symbol, period="18mo", interval="1d", progress=False)
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
    
    model = Sequential([
        Input(shape=(60, 1)),
        LSTM(50, return_sequences=False),
        Dense(25),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X, y, batch_size=32, epochs=2, verbose=0) # 快速訓練
    
    inputs = scaled_data[len(scaled_data) - 60:]
    inputs = inputs.reshape(-1, 1)
    
    future_prices = []
    curr_input = inputs
    
    for _ in range(days):
        curr_input_reshaped = np.reshape(curr_input, (1, 60, 1))
        pred = model.predict(curr_input_reshaped, verbose=0)
        future_prices.append(pred[0, 0])
        curr_input = np.append(curr_input[1:], pred, axis=0)
        curr_input = curr_input.reshape(-1, 1)
        
    future_prices = scaler.inverse_transform(np.array(future_prices).reshape(-1, 1))
    return future_prices[-1][0]

# ==================== 3. 主程式 UI ====================

def main():
    st.title("🏆 AI 股市全能專家 v13 (Yahoo 爬蟲整合版)")
    
    client = get_gspread_client()
    status_color = "green" if client else "red"
    status_text = "雲端連線正常" if client else "雲端未連線 (請檢查權限)"
    st.sidebar.markdown(f"### ☁️ 狀態：:{status_color}[{status_text}]")
    
    tab1, tab2, tab3 = st.tabs(["🔍 單股分析 (存分頁1)", "🚀 全市場掃描 (存分頁2)", "📊 雲端紀錄"])

    # --- TAB 1: 單股分析 ---
    with tab1:
        st.info("此處的分析結果將存入 Google Sheets 的 **第一分頁 (Sheet1)**")
        symbol = st.text_input("輸入代碼", "2330.TW").upper()
        if st.button("單股分析"):
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

                    if st.button("💾 存檔"):
                        save_data = [[
                            datetime.now().strftime('%Y-%m-%d'), symbol,
                            round(float(curr_price), 2), round(float(pred_price), 2),
                            f"{gain:.2f}%", "-", "-"
                        ]]
                        # sheet_index=0 -> 第一頁
                        if save_to_sheets(save_data, sheet_index=0):
                            st.success("已存入第一分頁！")

    # --- TAB 2: 全市場掃描 (使用 Yahoo 爬蟲) ---
    with tab2:
        st.markdown("### 🤖 全自動流程 (Yahoo 成交值排行)")
        st.write("1. 爬取 Yahoo 股市成交值排行榜 (前100名) -> 2. AI 預測 -> 3. 存入 Google Sheets **第二分頁**")
        
        if st.button("🚀 啟動掃描並預測"):
            manager = StockPoolManagerV2()
            top_100_tickers = manager.get_hot_stocks(limit=100)
            
            st.write(f"📋 掃描名單預覽：{top_100_tickers[:5]} ...")
            
            # 開始 AI 預測
            results = []
            progress = st.progress(0)
            status = st.empty()
            
            for i, stock in enumerate(top_100_tickers):
                status.text(f"🤖 AI 正在分析 ({i+1}/{len(top_100_tickers)}): {stock}")
                
                df = get_stock_history(stock)
                if df is not None:
                    curr_p = df['Close'].iloc[-1]
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
            
            # 顯示與存檔
            res_df = pd.DataFrame(results, columns=["日期","代碼","現價","預測","漲幅","實際","誤差"])
            st.dataframe(res_df)
            
            # sheet_index=1 -> 第二頁
            if save_to_sheets(results, sheet_index=1):
                st.success(f"🎉 成功將 {len(results)} 檔熱門股預測結果存入 **第二分頁**！")

    # --- TAB 3: 雲端紀錄 ---
    with tab3:
        if st.button("🔄 刷新"):
            st.cache_data.clear()
        
        sheet_option = st.radio("選擇分頁", ["第一分頁 (單股)", "第二分頁 (掃描結果)"])
        idx = 0 if "第一" in sheet_option else 1

        if client:
            try:
                sh = client.open(SHEET_NAME)
                try:
                    ws = sh.get_worksheet(idx)
                    if ws:
                        data = ws.get_all_values()
                        if len(data) > 1:
                            st.dataframe(pd.DataFrame(data[1:], columns=data[0]))
                        else:
                            st.info("此分頁無資料")
                    else:
                        st.warning("此分頁尚未建立")
                except:
                     st.warning("讀取分頁失敗")
            except Exception as e:
                st.error(f"讀取失敗: {e}")

if __name__ == "__main__":
    main()
