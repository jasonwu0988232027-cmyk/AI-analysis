import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import gspread
import requests
from bs4 import BeautifulSoup
from google.oauth2.service_account import Credentials
from datetime import datetime
import time
import os
import random
import urllib3

# --- 基礎配置 ---
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
st.set_page_config(page_title="台股量化預測系統 v19.6", layout="wide")

# --- 參數設定 ---
SHEET_NAME = "Stock_Predictions_History"
CREDENTIALS_JSON = "eco-precept-485904-j5-7ef3cdda1b03.json"

# ==================== 1. 雲端連線模組 ====================

def get_gspread_client():
    """處理私鑰並建立連線，防止 Header 報錯"""
    scopes = ['https://www.googleapis.com/auth/spreadsheets', 'https://www.googleapis.com/auth/drive']
    try:
        if "gcp_service_account" in st.secrets:
            creds_info = dict(st.secrets["gcp_service_account"])
            # 修正換行符號問題
            creds_info["private_key"] = creds_info["private_key"].replace("\\n", "\n")
            creds = Credentials.from_service_account_info(creds_info, scopes=scopes)
        elif os.path.exists(CREDENTIALS_JSON):
            creds = Credentials.from_service_account_file(CREDENTIALS_JSON, scopes=scopes)
        else:
            return None
        return gspread.authorize(creds)
    except Exception as e:
        st.error(f"❌ 授權失敗: {e}")
        return None

# ==================== 2. 分析核心模組 ====================

def calculate_quant_logic(ticker, df):
    """量化分析：黃金交叉 + PE 積分預測"""
    score = 0
    try:
        # 1. 技術面：MA5/MA20 黃金交叉判定
        ma5 = df['Close'].rolling(5).mean().iloc[-1]
        ma20 = df['Close'].rolling(20).mean().iloc[-1]
        if ma5 > ma20: score += 5
        
        # 2. 基本面：本益比資訊
        info = yf.Ticker(ticker).info
        pe = info.get('forwardPE', 100)
        if pe < 18: score += 2
        
        # 趨勢與波動率計算
        volatility = df['Close'].pct_change().std()
        last_price = float(df['Close'].iloc[-1])
        trend = (score - 3.5) * 0.001 
        
        preds = []
        curr_p = last_price
        for i in range(1, 6):
            move = trend + np.random.normal(0, volatility * 0.4)
            curr_p *= (1 + move)
            preds.append(round(curr_p, 2))
        return preds
    except:
        return [round(float(df['Close'].iloc[-1]) * 1.01, 2)] * 5

# ==================== 3. 主執行流程 ====================

st.title("📊 台股量化分析系統 v19.6")

if st.button("🚀 啟動 Top 100 量化預測任務"):
    client = get_gspread_client()
    if client:
        sh = client.open(SHEET_NAME)
        ws = sh.get_worksheet(0)
        
        # --- [修復重點]：初始化表首並防止 get_all_records 報錯 ---
        headers = ["日期", "股票代號", "收盤價格", "交易值指標", "5日預測-1", "5日預測-2", "5日預測-3", "5日預測-4", "5日預測-5", "誤差%"]
        
        # 檢查第一列是否有標題，若無則寫入
        first_row = ws.row_values(1)
        if not first_row:
            ws.insert_row(headers, 1)
            st.info("检测到空白表格，已自動建立 A-J 欄位標題。")
            st.rerun() # 重啟以確保 get_all_records 能讀取到標題

        try:
            # 獲取 Excel 內的股票清單
            data = ws.get_all_records()
            df_sheet = pd.DataFrame(data)
            
            if df_sheet.empty:
                st.warning("⚠️ Excel 中目前沒有股票代碼資料，請先填入 B 欄。")
                st.stop()
                
            tickers = df_sheet['股票代號'].dropna().astype(str).head(100).tolist()
            
            p_bar = st.progress(0)
            status = st.empty()
            
            # 批量下載
            all_hist = yf.download(tickers, period="3mo", group_by='ticker', threads=True, progress=False)
            
            for idx, t in enumerate(tickers):
                try:
                    status.text(f"分析中: {t}")
                    df = all_hist[t].dropna() if isinstance(all_hist.columns, pd.MultiIndex) else all_hist.dropna()
                    if df.empty: continue
                    
                    # 計算預測
                    preds = calculate_quant_logic(t, df)
                    
                    # 寫入 E-J 欄 (列號為 idx + 2，因為第一列是標題)
                    ws.update(f"E{idx+2}:J{idx+2}", [preds + ["-"]])
                    
                    time.sleep(random.uniform(0.5, 1.0))
                except Exception as e:
                    st.warning(f"跳過 {t}: {e}")
                
                p_bar.progress((idx + 1) / len(tickers))
                
            st.success("🎉 全部預測數據已更新完成！")
            
        except gspread.exceptions.GSpreadException as e:
            st.error(f"Excel 格式錯誤: {e}。請確保第一列包含標題『股票代號』。")
