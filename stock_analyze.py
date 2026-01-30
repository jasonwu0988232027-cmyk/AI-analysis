import streamlit as st

# --- 頁面配置 ---
st.set_page_config(page_title="AI 股市預測專家 Pro v4", layout="wide")

import yfinance as yf
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time
import os
import urllib3

# 停用 SSL 警告 (針對證交所 API)
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

# ==================== 1. 數據獲取 (解決 KeyError 與 盤中抓不到問題) ====================

def get_top_100_value_stocks():
    """自動判斷時間：確保抓到最近的有資料交易日"""
    now = datetime.now()
    
    # 如果還沒到下午 3:00 (結算完成時間)，先從昨天開始找
    if now.hour < 15:
        target_date = now - timedelta(days=1)
    else:
        target_date = now

    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
    
    attempts = 0
    final_df = pd.DataFrame()
    
    # 往前搜尋最多 10 天，直到抓到真正的數據
    while attempts < 10:
        date_str = target_date.strftime('%Y%m%d')
        url = f"https://www.twse.com.tw/exchangeReport/MI_INDEX?response=json&date={date_str}&type=ALLBUT0999"
        
        try:
            res = requests.get(url, headers=headers, timeout=15, verify=False)
            data = res.json()
            
            # 判斷標籤是否存在 (修正 KeyError 的核心)
            target_key = 'data9' if 'data9' in data else 'data8'
            
            if data.get('stat') == "OK" and target_key in data:
                fields_key = 'fields9' if 'fields9' in data else 'fields8'
                df = pd.DataFrame(data[target_key], columns=data[fields_key])
                
                # 資料清洗：去除數字中的逗號
                df['成交金額'] = df['成交金額'].str.replace(',', '').astype(float)
                df['證券代號'] = df['證券代號'] + ".TW"
                
                st.info(f"📅 數據抓取成功！來源日期: {target_date.strftime('%Y-%m-%d')}")
                return df.nlargest(100, '成交金額')[['證券代號', '證券名稱', '收盤價']]
            
        except Exception:
            pass # 略過錯誤，嘗試前一天
        
        target_date -= timedelta(days=1)
        attempts += 1

    st.error("❌ 無法獲取台股百大排名，請檢查網路或確認是否為連續長假。")
    return pd.DataFrame()

def get_stock_data(symbol, period="1y"):
    """獲取單股歷史數據"""
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
        st.warning(f"⚠️ 找不到憑證檔案 {CREDENTIALS_JSON}。")
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
            # 初始化標題
            if ws.row_count <= 1 and (not ws.cell(1, 1).value):
                ws.append_row(["預測日期", "股票代碼", "目前價格", "7日預測價", "預期漲幅", "實際收盤價", "誤差%"])
            ws.append_rows(new_data)
            return True
        except Exception as e:
            st.error(f"❌ 雲端寫入失敗: {e}")
    return False

# ==================== 3. 機器學習推論模組 ====================

@st.cache_resource
def get_trained_base_model():
    """建立基礎基準模型（使用 2330 作為權重範本）"""
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
    """基準模型快速推論"""
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df[['Close']].values)
    last_60 = scaled[-60:].reshape(1, 60, 1)
    pred = model.predict(last_60, verbose=0)
    return scaler.inverse_transform(pred)[0][0]

# ==================== 4. 主介面邏輯 ====================

def main():
    st.title("📈 AI 股市趨勢分析與雲端存檔系統")
    
    tab1, tab2 = st.tabs(["🚀 百大自動預測", "🧐 歷史反思與學習"])

    with tab1:
        st.markdown("### 自動抓取成交值前 100 名並進行 7 日預測")
        if st.button("執行全自動批次分析"):
            with st.spinner("🔍 正在檢索證交所數據..."):
                top_100 = get_top_100_value_stocks()
                model = get_trained_base_model()
            
            if not top_100.empty and model:
                results = []
                bar = st.progress(0)
                msg = st.empty()
                
                for i, row in top_100.iterrows():
                    symbol = row['證券代號']
                    msg.text(f"分析進度 ({i+1}/100): {symbol}")
                    
                    time.sleep(BATCH_CD) # CD 緩衝
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
                            "-", "-" # 預留給 7 天後的實際值
                        ])
                    bar.progress((i+1)/100)
                
                if save_to_sheets(results):
                    st.success("🎉 分析完成！所有數據已同步至 Google 雲端試算表。")
                    st.dataframe(pd.DataFrame(results, columns=["日期","代碼","現價","預測價","漲幅","實際","誤差"]))

    with tab2:
        st.subheader("Google Sheets 歷史紀錄反思")
        client = get_gspread_client()
        if client:
            try:
                ws = client.open(SHEET_NAME).sheet1
                records = ws.get_all_records()
                if records:
                    st.write("📊 雲端儲存的最近預測：")
                    st.dataframe(pd.DataFrame(records).tail(20))
                else:
                    st.info("雲端目前尚無預測紀錄。")
            except Exception as e:
                st.error(f"讀取雲端紀錄失敗: {e}")

if __name__ == "__main__":
    main()
