import streamlit as st

# --- 頁面配置 ---
st.set_page_config(page_title="AI 股市預測專家 Pro v3", layout="wide")

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

# 載入必要庫
try:
    import gspread
    from oauth2client.service_account import ServiceAccountCredentials
    import tensorflow as tf
    from sklearn.preprocessing import MinMaxScaler
except ImportError:
    st.error("缺少套件，請執行：pip install gspread oauth2client tensorflow scikit-learn urllib3 certifi")

# --- 全局設定 ---
CREDENTIALS_JSON = "credentials.json" 
SHEET_NAME = "Stock_Predictions_History" 
BATCH_CD = 1.2 # 防止被 yfinance 封鎖的延遲

# ==================== 1. 數據獲取 (含自動日期邏輯) ====================

def get_top_100_value_stocks():
    """自動判斷時間：盤中抓昨日，盤後抓今日，假日自動往前找"""
    now = datetime.now()
    # 證交所通常在 14:30 後才完成當日結算，建議設 15:00 較穩
    if now.hour < 15:
        target_date = now - timedelta(days=1)
    else:
        target_date = now

    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
    
    attempts = 0
    data = {}
    while attempts < 7: # 最多往前找 7 天
        date_str = target_date.strftime('%Y%m%d')
        url = f"https://www.twse.com.tw/exchangeReport/MI_INDEX?response=json&date={date_str}&type=ALLBUT0999"
        
        try:
            res = requests.get(url, headers=headers, timeout=15, verify=False)
            temp_data = res.json()
            if temp_data.get('stat') == "OK":
                data = temp_data
                break
        except:
            pass
        
        target_date -= timedelta(days=1)
        attempts += 1

    if not data:
        st.error("無法從證交所獲取數據，請檢查網路或 API 狀態。")
        return pd.DataFrame()

    # 判斷資料欄位 (data9 或 data8)
    target_key = 'data9' if 'data9' in data else 'data8'
    fields_key = 'fields9' if 'fields9' in data else 'fields8'
    
    df = pd.DataFrame(data[target_key], columns=data[fields_key])
    df['成交金額'] = df['成交金額'].str.replace(',', '').astype(float)
    df['證券代號'] = df['證券代號'] + ".TW"
    
    st.info(f"📅 數據來源日期: {target_date.strftime('%Y-%m-%d')}")
    return df.nlargest(100, '成交金額')[['證券代號', '證券名稱', '收盤價']]

def get_stock_data(symbol, period="1y"):
    try:
        df = yf.download(symbol, period=period, interval="1d", progress=False)
        if df.empty: return None
        df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
        return df.reset_index()
    except:
        return None

# ==================== 2. 雲端同步模組 ====================

def get_gspread_client():
    if not os.path.exists(CREDENTIALS_JSON):
        st.warning(f"⚠️ 找不到 {CREDENTIALS_JSON}。")
        return None
    try:
        scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
        creds = ServiceAccountCredentials.from_json_keyfile_name(CREDENTIALS_JSON, scope)
        return gspread.authorize(creds)
    except Exception as e:
        st.error(f"❌ Google 連接失敗: {e}")
        return None

def save_to_sheets(new_data):
    client = get_gspread_client()
    if client:
        try:
            sh = client.open(SHEET_NAME)
            ws = sh.sheet1
            if ws.row_count <= 1 and (not ws.cell(1, 1).value):
                ws.append_row(["預測日期", "股票代碼", "目前價格", "7日預測價", "預期漲幅", "實際收盤價", "誤差%"])
            ws.append_rows(new_data)
            return True
        except Exception as e:
            st.error(f"❌ 寫入雲端失敗: {e}")
    return False

# ==================== 3. 預訓練推論模組 ====================

@st.cache_resource
def get_trained_base_model():
    """建立基準模型（以 2330 為訓練樣板）"""
    df = get_stock_data("2330.TW")
    if df is None: return None
    
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df[['Close']].values)
    
    X, y = [], []
    for i in range(60, len(scaled)):
        X.append(scaled[i-60:i, 0])
        y.append(scaled[i, 0])
    
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(60, 1)),
        tf.keras.layers.LSTM(50),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(np.array(X), np.array(y), epochs=3, batch_size=32, verbose=0)
    return model

def fast_predict(model, df):
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df[['Close']].values)
    last_60 = scaled[-60:].reshape(1, 60, 1)
    pred = model.predict(last_60, verbose=0)
    return scaler.inverse_transform(pred)[0][0]

# ==================== 4. 主介面邏輯 ====================

def main():
    st.title("📈 AI 台股百大趨勢自動化系統")
    
    tab1, tab2 = st.tabs(["🚀 啟動百大分析", "🧐 預測對錯反思"])

    with tab1:
        st.markdown("### 自動篩選台股成交值 Top 100 並存入雲端")
        if st.button("執行批次預測"):
            with st.spinner("正在初始化..."):
                top_100 = get_top_100_value_stocks()
                model = get_trained_base_model()
            
            if not top_100.empty and model:
                results = []
                bar = st.progress(0)
                msg = st.empty()
                
                for i, row in top_100.iterrows():
                    symbol = row['證券代號']
                    msg.text(f"分析中 ({i+1}/100): {symbol}")
                    
                    time.sleep(BATCH_CD)
                    df = get_stock_data(symbol)
                    
                    if df is not None and len(df) >= 60:
                        curr_p = df['Close'].iloc[-1]
                        pred_p = fast_predict(model, df)
                        gain = ((pred_p - curr_p) / curr_p) * 100
                        
                        results.append([
                            datetime.now().strftime('%Y-%m-%d'),
                            symbol,
                            round(float(curr_p), 2),
                            round(float(pred_p), 2),
                            f"{gain:.2f}%",
                            "-", "-"
                        ])
                    bar.progress((i+1)/100)
                
                if save_to_sheets(results):
                    st.success("✅ 分析完成！結果已同步至 Google Sheets。")
                    st.dataframe(pd.DataFrame(results, columns=["日期","代碼","現價","預測價","漲幅","實際","誤差"]))

    with tab2:
        st.subheader("歷史預測反思")
        client = get_gspread_client()
        if client:
            try:
                ws = client.open(SHEET_NAME).sheet1
                df_h = pd.DataFrame(ws.get_all_records())
                if not df_h.empty:
                    st.write("### 最近 20 筆預測紀錄")
                    st.dataframe(df_h.tail(20))
                else:
                    st.info("尚無紀錄。")
            except:
                st.error("讀取失敗。")

if __name__ == "__main__":
    main()
