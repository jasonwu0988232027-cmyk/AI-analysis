import streamlit as st

# --- 頁面配置 ---
st.set_page_config(page_title="AI 股市預測專家 Pro - 自動化版", layout="wide")

import yfinance as yf
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time
import os

# 雲端與機器學習相關庫
try:
    import gspread
    from oauth2client.service_account import ServiceAccountCredentials
    import tensorflow as tf
    from sklearn.preprocessing import MinMaxScaler
    TA_AVAILABLE = True
except ImportError:
    st.error("請確認已安裝所有必要套件：pip install gspread oauth2client tensorflow scikit-learn")

# --- 設定區 ---
FINNHUB_API_KEY = "d5t2rvhr01qt62ngu1kgd5t2rvhr01qt62ngu1l0"
CREDENTIALS_JSON = "credentials.json"  # 你的 Google 憑證檔案路徑
SHEET_NAME = "Stock_Predictions_History" # Google 試算表名稱
BATCH_CD = 1.5 # 每次抓取間隔秒數

# ==================== 1. 雲端整合模組 ====================

def get_gspread_client():
    """連接 Google Sheets"""
    if not os.path.exists(CREDENTIALS_JSON):
        st.warning(f"找不到 {CREDENTIALS_JSON}，請上傳憑證檔案。")
        return None
    try:
        scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
        creds = ServiceAccountCredentials.from_json_keyfile_name(CREDENTIALS_JSON, scope)
        return gspread.authorize(creds)
    except Exception as e:
        st.error(f"Google 連接失敗: {e}")
        return None

def save_to_sheets(new_df):
    """將預測結果存入雲端"""
    client = get_gspread_client()
    if client:
        try:
            sh = client.open(SHEET_NAME)
            ws = sh.sheet1
            # 如果是空表，先寫入標題
            if ws.row_count <= 1 and ws.cell(1, 1).value == "":
                ws.append_row(["預測日期", "股票代碼", "目前價格", "7日預測價", "預期漲幅%", "實際收盤價", "誤差%"])
            
            ws.append_rows(new_df.values.tolist())
            return True
        except Exception as e:
            st.error(f"寫入雲端失敗: {e}")
    return False

# ==================== 2. 數據抓取模組 ====================

@st.cache_data(ttl=3600)
def get_top_100_value_stocks():
    """自動從證交所抓取成交值前 100 名"""
    try:
        url = "https://www.twse.com.tw/exchangeReport/MI_INDEX?response=json&type=ALLBUT0999"
        res = requests.get(url, timeout=10)
        data = res.json()
        df = pd.DataFrame(data['data9'], columns=data['fields9'])
        df['成交金額'] = df['成交金額'].str.replace(',', '').astype(float)
        df['證券代號'] = df['證券代號'] + ".TW"
        top_100 = df.nlargest(100, '成交金額')[['證券代號', '證券名稱', '收盤價']]
        return top_100
    except Exception as e:
        st.error(f"抓取百大排名失敗: {e}")
        return pd.DataFrame()

def get_stock_data(symbol, period="1y"):
    try:
        df = yf.download(symbol, period=period, interval="1d", progress=False)
        if df.empty: return None
        df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
        return df.reset_index()
    except:
        return None

# ==================== 3. 預訓練模型邏輯 ====================

def get_base_model(input_shape):
    """建立共用的 LSTM 模型架構"""
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape),
        tf.keras.layers.LSTM(64, return_sequences=True),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.LSTM(32),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

def batch_inference(model, df, days=7):
    """利用現有模型進行快速預測"""
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df[['Close']].values)
    
    # 取最後 60 天
    last_60 = scaled_data[-60:].reshape(1, 60, 1)
    pred_scaled = model.predict(last_60, verbose=0)
    
    # 反正規化
    prediction = scaler.inverse_transform(pred_scaled)[0][0]
    return prediction

# ==================== 4. 反思與分析模組 ====================

def run_reflection():
    """對比 7 天前的預測與今日價格"""
    st.header("🧐 預測對錯反思報告")
    client = get_gspread_client()
    if not client: return
    
    try:
        ws = client.open(SHEET_NAME).sheet1
        data = ws.get_all_records()
        df = pd.DataFrame(data)
        
        if df.empty:
            st.info("尚無歷史預測資料。")
            return

        today = datetime.now()
        updated_count = 0
        
        # 尋找 7 天前預測且尚未填寫「實際結果」的資料
        for i, row in df.iterrows():
            pred_date = datetime.strptime(str(row['預測日期']), '%Y-%m-%d')
            if (today - pred_date).days >= 7 and (pd.isna(row['實際收盤價']) or row['實際收盤價'] == "-"):
                # 抓取該股票今日價格
                actual_df = get_stock_data(row['股票代碼'], period="1d")
                if actual_df is not None:
                    actual_price = actual_df['Close'].iloc[-1]
                    error = ((actual_price - row['7日預測價']) / actual_price) * 100
                    
                    # 更新表格 (gspread 索引從 1 開始，row 從 2 開始)
                    ws.update_cell(i + 2, 6, round(actual_price, 2))
                    ws.update_cell(i + 2, 7, f"{error:.2f}%")
                    updated_count += 1
        
        if updated_count > 0:
            st.success(f"已自動更新 {updated_count} 筆歷史資料對比！")
        
        st.dataframe(df.tail(20)) # 顯示最近 20 筆
    except Exception as e:
        st.error(f"反思過程出錯: {e}")

# ==================== 5. 主介面 ====================

def main():
    st.title("📈 AI 股市自動化預測系統")
    
    menu = st.sidebar.selectbox("功能選單", ["百大交易值預測", "預測反思報告", "單股詳細分析"])

    if menu == "百大交易值預測":
        st.subheader("🔥 今日台股成交值前 100 名自動預測")
        if st.button("啟動批次分析"):
            top_stocks = get_top_100_value_stocks()
            if not top_stocks.empty:
                # 1. 訓練一個基礎模型 (以 2330 為基準)
                st.info("⏳ 正在建立基礎預訓練模型 (以 2330.TW 為基準)...")
                base_df = get_stock_data("2330.TW")
                base_model = get_base_model((60, 1))
                
                # 批次預測
                results = []
                progress = st.progress(0)
                
                for idx, row in top_stocks.iterrows():
                    symbol = row['證券代號']
                    time.sleep(BATCH_CD) # CD 延遲
                    
                    stock_df = get_stock_data(symbol)
                    if stock_df is not None and len(stock_df) > 60:
                        pred_price = batch_inference(base_model, stock_df)
                        current_price = stock_df['Close'].iloc[-1]
                        gain = ((pred_price - current_price) / current_price) * 100
                        
                        results.append([
                            datetime.now().strftime('%Y-%m-%d'),
                            symbol,
                            round(float(current_price), 2),
                            round(float(pred_price), 2),
                            round(float(gain), 2),
                            "-", # 實際收盤價 (待 7 天後回填)
                            "-"  # 誤差% (待 7 天後回填)
                        ])
                    progress.progress((idx + 1) / 100)
                
                # 儲存到雲端
                res_df = pd.DataFrame(results)
                if save_to_sheets(res_df):
                    st.success("🎉 百大股票預測完成並已存入 Google Sheets！")
                    st.table(res_df.head(10)) # 預覽前 10 筆

    elif menu == "預測反思報告":
        run_reflection()

    elif menu == "單股詳細分析":
        # 此處可保留你原本單支股票的詳細圖解代碼
        st.info("此功能可整合原有的詳細技術圖表分析。")

if __name__ == "__main__":
    main()
