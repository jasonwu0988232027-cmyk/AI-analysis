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
st.set_page_config(page_title="台股量化預測系統 v19.7", layout="wide")

# --- 參數設定 ---
SHEET_NAME = "Stock_Predictions_History"
CREDENTIALS_JSON = "eco-precept-485904-j5-7ef3cdda1b03.json"

# ==================== 1. 雲端連線模組 ====================

def get_gspread_client():
    """建立授權連線並修正私鑰格式"""
    scopes = ['https://www.googleapis.com/auth/spreadsheets', 'https://www.googleapis.com/auth/drive']
    try:
        if "gcp_service_account" in st.secrets:
            creds_info = dict(st.secrets["gcp_service_account"])
            # 強制處理換行符號
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

# ==================== 2. 量化核心演算 ====================

def calculate_quant_analysis(ticker, df):
    """
    量化演算邏輯：黃金交叉 + 本益比權重
    """
    score = 0
    try:
        # 技術面：MA5 與 MA20 黃金交叉判定
        ma5 = df['Close'].rolling(5).mean().iloc[-1]
        ma20 = df['Close'].rolling(20).mean().iloc[-1]
        if ma5 > ma20: score += 5
        
        # 基本面：本益比 (PE) 判定
        info = yf.Ticker(ticker).info
        pe = info.get('forwardPE', 100)
        if pe < 18: score += 2
        
        # 產出 5 日預測
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

st.title("📊 台股量化分析系統 v19.7")

if st.button("🚀 啟動 Top 100 量化預測任務"):
    client = get_gspread_client()
    if client:
        sh = client.open(SHEET_NAME)
        ws = sh.get_worksheet(0)
        
        # --- [修正重點]：使用 get_all_values 避開標題重複錯誤 ---
        raw_data = ws.get_all_values()
        
        if not raw_data:
            # 完全空白則建立標題
            headers = ["日期", "股票代號", "收盤價格", "交易值指標", "5日預測-1", "5日預測-2", "5日預測-3", "5日預測-4", "5日預測-5", "誤差%"]
            ws.insert_row(headers, 1)
            st.info("已建立 A-J 標題欄位，請在 B 欄填入代碼後重新執行。")
            st.stop()

        # 使用 Pandas 處理資料，並強制指定第一列為 Header
        df_sheet = pd.DataFrame(raw_data[1:], columns=raw_data[0])
        
        # 移除標題名稱為空欄位的 Columns
        df_sheet = df_sheet.loc[:, df_sheet.columns != '']
        
        if '股票代號' not in df_sheet.columns:
            st.error("❌ 找不到『股票代號』欄位。請確保 B1 儲存格內容為『股票代號』。")
            st.stop()
            
        tickers = df_sheet['股票代號'].replace('', np.nan).dropna().head(100).tolist()
        
        if not tickers:
            st.warning("⚠️ B 欄中目前沒有股票代碼。")
            st.stop()

        st.info(f"正在分析 {len(tickers)} 檔股票之趨勢...")
        p_bar = st.progress(0)
        
        # 批量下載
        all_hist = yf.download(tickers, period="3mo", group_by='ticker', threads=True, progress=False)
        
        for idx, t in enumerate(tickers):
            try:
                df = all_hist[t].dropna() if isinstance(all_hist.columns, pd.MultiIndex) else all_hist.dropna()
                if df.empty: continue
                
                preds = calculate_quant_analysis(t, df)
                
                # 寫入 E-J 欄位 (對應 Excel 索引)
                ws.update(f"E{idx+2}:J{idx+2}", [preds + ["-"]])
                
                time.sleep(random.uniform(0.5, 1.0))
            except Exception as e:
                st.warning(f"跳過 {t}: {e}")
                
            p_bar.progress((idx + 1) / len(tickers))
            
        st.success("🎉 全部量化分析數據已同步至 A-J 欄！")
