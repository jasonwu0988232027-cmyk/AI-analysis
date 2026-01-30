import streamlit as st

# --- 頁面配置 ---
st.set_page_config(page_title="AI 股市預測專家 Pro v5 (備援版)", layout="wide")

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
BATCH_CD = 1.2 

# ==================== 1. 備援名單 (當證交所 API 掛掉時使用) ====================

def get_fallback_stocks():
    """提供台灣 30 大權值股作為備用名單"""
    st.warning("⚠️ 檢測到證交所 API 連線困難，已自動切換至「備援權值股名單」繼續執行。")
    data = {
        '證券代號': [
            '2330.TW', '2317.TW', '2454.TW', '2308.TW', '2382.TW', 
            '2303.TW', '2881.TW', '2882.TW', '2891.TW', '2886.TW',
            '2412.TW', '2884.TW', '1216.TW', '2885.TW', '3711.TW',
            '2892.TW', '2357.TW', '2880.TW', '2890.TW', '5880.TW',
            '2345.TW', '3008.TW', '2327.TW', '2395.TW', '2883.TW',
            '2887.TW', '3045.TW', '4938.TW', '2408.TW', '1101.TW'
        ],
        '證券名稱': [
            '台積電', '鴻海', '聯發科', '台達電', '廣達', 
            '聯電', '富邦金', '國泰金', '中信金', '兆豐金',
            '中華電', '玉山金', '統一', '元大金', '日月光',
            '第一金', '華碩', '華南金', '永豐金', '合庫金',
            '智邦', '大立光', '國巨', '研華', '開發金',
            '台新金', '台灣大', '和碩', '南亞科', '台泥'
        ]
    }
    # 嘗試抓取現價填入
    df = pd.DataFrame(data)
    current_prices = []
    for symbol in df['證券代號']:
        try:
            stock = yf.Ticker(symbol)
            hist = stock.history(period="1d")
            if not hist.empty:
                current_prices.append(hist['Close'].iloc[-1])
            else:
                current_prices.append(0)
        except:
            current_prices.append(0)
    df['收盤價'] = current_prices
    return df

# ==================== 2. 數據獲取 (含自動備援) ====================

def get_top_100_value_stocks():
    """嘗試抓取證交所數據，失敗則調用備援名單"""
    now = datetime.now()
    if now.hour < 15:
        target_date = now - timedelta(days=1)
    else:
        target_date = now

    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
    
    attempts = 0
    # 搜尋最近 7 天
    while attempts < 7:
        date_str = target_date.strftime('%Y%m%d')
        url = f"https://www.twse.com.tw/exchangeReport/MI_INDEX?response=json&date={date_str}&type=ALLBUT0999"
        
        try:
            # 降低 timeout 防止卡死
            res = requests.get(url, headers=headers, timeout=5, verify=False)
            data = res.json()
            
            target_key = 'data9' if 'data9' in data else 'data8'
            
            if data.get('stat') == "OK" and target_key in data:
                fields_key = 'fields9' if 'fields9' in data else 'fields8'
                df = pd.DataFrame(data[target_key], columns=data[fields_key])
                df['成交金額'] = df['成交金額'].str.replace(',', '').astype(float)
                df['證券代號'] = df['證券代號'] + ".TW"
                st.success(f"📅 成功連線證交所 (資料日期: {target_date.strftime('%Y-%m-%d')})")
                return df.nlargest(100, '成交金額')[['證券代號', '證券名稱', '收盤價']]
            
        except Exception:
            pass 
        
        target_date -= timedelta(days=1)
        attempts += 1

    # 如果跑完迴圈還是沒資料，回傳備用名單
    return get_fallback_stocks()

def get_stock_data(symbol, period="1y"):
    """獲取單股歷史數據"""
    try:
        df = yf.download(symbol, period=period, interval="1d", progress=False)
        if df.empty: return None
        df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
        return df.reset_index()
    except:
        return None

# ==================== 3. 雲端與模型模組 ====================

def get_gspread_client():
    if not os.path.exists(CREDENTIALS_JSON):
        st.warning(f"⚠️ 未找到 {CREDENTIALS_JSON}，請上傳憑證。")
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
    if client:
        try:
            sh = client.open(SHEET_NAME)
            ws = sh.sheet1
            if ws.row_count <= 1 and (not ws.cell(1, 1).value):
                ws.append_row(["預測日期", "股票代碼", "目前價格", "7日預測價", "預期漲幅", "實際收盤價", "誤差%"])
            ws.append_rows(new_data)
            return True
        except Exception as e:
            st.error(f"❌ 雲端寫入失敗: {e}")
    return False

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
    st.title("📈 AI 股市趨勢分析 Pro (防崩潰版)")
    
    tab1, tab2 = st.tabs(["🚀 自動預測執行", "🧐 歷史反思"])

    with tab1:
        st.write("系統將優先抓取證交所即時百大排名，若連線失敗將自動切換至權值股名單。")
        if st.button("開始執行"):
            with st.spinner("系統初始化中..."):
                target_stocks = get_top_100_value_stocks()
                model = get_trained_base_model()
            
            if not target_stocks.empty and model:
                results = []
                bar = st.progress(0)
                msg = st.empty()
                
                total = len(target_stocks)
                for i, row in target_stocks.iterrows():
                    symbol = row['證券代號']
                    msg.text(f"分析進度 ({i+1}/{total}): {symbol}")
                    
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
                    bar.progress((i+1)/total)
                
                if save_to_sheets(results):
                    st.success(f"🎉 已完成 {len(results)} 檔股票預測並存檔！")
                    st.dataframe(pd.DataFrame(results, columns=["日期","代碼","現價","預測價","漲幅","實際","誤差"]))

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

if __name__ == "__main__":
    main()
