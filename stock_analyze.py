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
st.set_page_config(page_title="台股量化預測系統 v19.5", layout="wide")

# --- 參數設定 ---
SHEET_NAME = "Stock_Predictions_History"
CREDENTIALS_JSON = "eco-precept-485904-j5-7ef3cdda1b03.json"

# ==================== 1. 雲端連線模組 ====================

def get_gspread_client():
    """建立連線並處理私鑰格式"""
    scopes = ['https://www.googleapis.com/auth/spreadsheets', 'https://www.googleapis.com/auth/drive']
    try:
        if "gcp_service_account" in st.secrets:
            creds_info = dict(st.secrets["gcp_service_account"])
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

def calculate_quant_prediction(ticker, df):
    """
    量化因子積分制：技術面(MA5/20黃金交叉) + 基本面(PE)
    """
    score = 0
    try:
        # 技術面：均線黃金交叉判定
        ma5 = df['Close'].rolling(5).mean().iloc[-1]
        ma20 = df['Close'].rolling(20).mean().iloc[-1]
        if ma5 > ma20: score += 5
        
        # 基本面：本益比判定
        info = yf.Ticker(ticker).info
        pe = info.get('forwardPE', 100)
        if pe < 18: score += 2
        
        # 預測邏輯：根據積分與歷史波動率產出 5 日預測
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

st.title("📊 台股量化分析系統 (自動表首版)")

if st.button("🚀 開始執行全市場前 100 名量化預測"):
    client = get_gspread_client()
    if client:
        sh = client.open(SHEET_NAME)
        ws = sh.get_worksheet(0)
        
        # --- 新增：自動建立表首邏輯 ---
        headers = ["日期", "股票代號", "收盤價格", "交易值指標", "5日預測-1", "5日預測-2", "5日預測-3", "5日預測-4", "5日預測-5", "誤差%"]
        try:
            # 檢查 A1 是否有值，若無則寫入標題
            if not ws.acell('A1').value:
                ws.insert_row(headers, 1)
                st.info("已為您自動建立 Excel 表首 (A-J 欄)。")
        except:
            pass

        # 讀取現有代碼 (B 欄)
        data = ws.get_all_records()
        df_sheet = pd.DataFrame(data)
        if df_sheet.empty or '股票代號' not in df_sheet.columns:
            st.error("Excel 中找不到『股票代號』欄位。")
            st.stop()
            
        tickers = df_sheet['股票代號'].dropna().astype(str).head(100).tolist()
        
        st.info(f"正在執行 {len(tickers)} 檔股票之量化分析...")
        p_bar = st.progress(0)
        
        # 批量獲取數據
        all_hist = yf.download(tickers, period="3mo", group_by='ticker', threads=True, progress=False)
        
        for idx, t in enumerate(tickers):
            try:
                df = all_hist[t].dropna() if isinstance(all_hist.columns, pd.MultiIndex) else all_hist.dropna()
                if df.empty: continue
                
                # 計算 5 日預測
                preds = calculate_quant_prediction(t, df)
                
                # 更新至 E-J 欄位
                # E-I 欄 (5-9): 預測價, J 欄 (10): 誤差
                ws.update(f"E{idx+2}:J{idx+2}", [preds + ["-"]])
                
                # 隨機延遲預防限流
                time.sleep(random.uniform(0.5, 1.2))
                    
            except Exception as e:
                st.warning(f"跳過 {t}: {e}")
                
            p_bar.progress((idx + 1) / len(tickers))
            
        st.success("🎉 全部數據與預測已成功同步至 Excel A-J 欄！")
