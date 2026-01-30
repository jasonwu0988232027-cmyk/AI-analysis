import streamlit as st

# --- 頁面配置 ---
st.set_page_config(page_title="AI 股市預測專家 Pro v6 (穩定版)", layout="wide")

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
BATCH_CD = 1.0 # 稍微加快速度

# ==================== 1. 穩定版百大名單 (不依賴證交所 API) ====================

def get_stable_stock_list():
    """直接回傳內建的熱門台股名單，保證不缺資料"""
    # 這是台灣 50 + 熱門電子/金融/傳產股的綜合名單
    tickers = [
        '2330.TW', '2317.TW', '2454.TW', '2308.TW', '2382.TW', '2303.TW', '2881.TW', '2882.TW', '2891.TW', '2886.TW',
        '2412.TW', '2884.TW', '1216.TW', '2885.TW', '3711.TW', '2892.TW', '2357.TW', '2880.TW', '2890.TW', '5880.TW',
        '2345.TW', '3008.TW', '2327.TW', '2395.TW', '2883.TW', '2887.TW', '3045.TW', '4938.TW', '2408.TW', '1101.TW',
        '2002.TW', '3037.TW', '2379.TW', '3034.TW', '2603.TW', '2609.TW', '2615.TW', '3231.TW', '2356.TW', '2301.TW',
        '2801.TW', '2888.TW', '6669.TW', '6415.TW', '3035.TW', '3017.TW', '4904.TW', '5871.TW', '2912.TW', '9910.TW',
        '1301.TW', '1303.TW', '1326.TW', '6505.TW', '2353.TW', '2409.TW', '3481.TW', '6770.TW', '1513.TW', '1519.TW',
        '1605.TW', '2371.TW', '2383.TW', '2388.TW', '2451.TW', '2474.TW', '3019.TW', '3042.TW', '3044.TW', '3189.TW',
        '3293.TW', '3529.TW', '3532.TW', '3533.TW', '3653.TW', '3661.TW', '3702.TW', '4919.TW', '4958.TW', '4961.TW'
    ]
    
    # 建立 DataFrame 結構
    data = {'證券代號': tickers, '證券名稱': [f"股票 {t}" for t in tickers]} 
    df = pd.DataFrame(data)
    return df

def get_stock_data(symbol, period="1y"):
    """獲取單股歷史數據"""
    try:
        df = yf.download(symbol, period=period, interval="1d", progress=False)
        if df.empty: return None
        df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
        return df.reset_index()
    except:
        return None

# ==================== 2. 雲端同步模組 (容錯版) ====================

def get_gspread_client():
    if not os.path.exists(CREDENTIALS_JSON):
        # 這裡不報錯，改用回傳 None，讓程式知道沒鑰匙就好
        return None
    try:
        scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
        creds = ServiceAccountCredentials.from_json_keyfile_name(CREDENTIALS_JSON, scope)
        return gspread.authorize(creds)
    except Exception as e:
        st.error(f"❌ Google Sheets 連接失敗: {e}")
        return None

def save_to_sheets(new_data):
    client = get_gspread_client()
    if client is None:
        st.warning("⚠️ 未偵測到憑證 (credentials.json)，本次預測結果將**不會**上傳至雲端，僅顯示於下方。")
        return False
        
    try:
        sh = client.open(SHEET_NAME)
        ws = sh.sheet1
        if ws.row_count <= 1 and (not ws.cell(1, 1).value):
            ws.append_row(["預測日期", "股票代碼", "目前價格", "7日預測價", "預期漲幅", "實際收盤價", "誤差%"])
        ws.append_rows(new_data)
        st.success("✅ 雲端存檔成功！")
        return True
    except Exception as e:
        st.error(f"❌ 雲端寫入失敗: {e}")
        return False

# ==================== 3. 機器學習推論模組 ====================

@st.cache_resource
def get_trained_base_model():
    """建立基礎基準模型"""
    df = get_stock_data("2330.TW")
    if df is None: 
        st.error("無法下載台積電數據作為基準，請檢查網路。")
        return None
    
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

# ==================== 4. 主介面 ====================

def main():
    st.title("📈 AI 股市趨勢分析 Pro (v6 穩定版)")
    
    tab1, tab2 = st.tabs(["🚀 熱門股批量預測", "🧐 歷史反思"])

    with tab1:
        st.info("系統採用「內建熱門股名單 (80+)」，不再受證交所連線限制，保證執行順暢。")
        
        if st.button("開始執行預測"):
            with st.spinner("模型初始化中..."):
                # 直接使用穩定名單，不再去證交所冒險
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
                    
                    # 速度稍微加快，因為內建名單很穩
                    time.sleep(0.5) 
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
                    bar.progress((i+1)/total)
                
                # 顯示結果 DataFrame
                res_df = pd.DataFrame(results, columns=["日期","代碼","現價","預測價","漲幅","實際","誤差"])
                st.dataframe(res_df)
                
                # 嘗試存檔 (如果沒鑰匙，會自動跳過並顯示警告，不會報錯)
                save_to_sheets(results)

    with tab2:
        st.subheader("Google Sheets 歷史紀錄")
        client = get_gspread_client()
        if client:
            try:
                ws = client.open(SHEET_NAME).sheet1
                records = ws.get_all_records()
                if records:
                    st.dataframe(pd.DataFrame(records).tail(20))
                else:
                    st.info("暫無紀錄")
            except Exception as e:
                st.error(f"讀取失敗: {e}")
        else:
            st.info("請上傳 credentials.json 以啟用歷史回測功能。")

if __name__ == "__main__":
    main()
