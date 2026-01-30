import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import gspread
import requests
from google.oauth2.service_account import Credentials
from datetime import datetime, timedelta
import time
import os
import random
import urllib3

# --- 基礎配置 ---
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
st.set_page_config(page_title="台股量化預測與對帳系統 v20.0", layout="wide")

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

# ==================== 2. 量化與誤差計算邏輯 ====================

def calculate_quant_logic(ticker, df):
    """量化積分預測：MA黃金交叉 + PE權重"""
    score = 0
    try:
        # 技術面：MA5/MA20 黃金交叉
        ma5 = df['Close'].rolling(5).mean().iloc[-1]
        ma20 = df['Close'].rolling(20).mean().iloc[-1]
        if ma5 > ma20: score += 5
        
        # 基本面：本益比
        info = yf.Ticker(ticker).info
        if info.get('forwardPE', 100) < 18: score += 2
        
        # 預測未來 5 日
        vol = df['Close'].pct_change().std()
        last_p = float(df['Close'].iloc[-1])
        trend = (score - 3.5) * 0.001
        
        preds = []
        curr = last_p
        for i in range(1, 6):
            move = trend + np.random.normal(0, vol * 0.4)
            curr *= (1 + move)
            preds.append(round(curr, 2))
        return preds
    except:
        return [round(float(df['Close'].iloc[-1]) * 1.01, 2)] * 5

def update_error_analysis(ws):
    """
    自動計算 J 欄誤差%
    對比『5日預測-5』與『5天後的實際價格』
    """
    raw_data = ws.get_all_values()
    if len(raw_data) <= 1: return "無資料可對帳"
    
    headers = raw_data[0]
    df = pd.DataFrame(raw_data[1:], columns=headers)
    
    updated_count = 0
    now = datetime.now()
    
    for idx, row in df.iterrows():
        # 若已計算過誤差或無預測資料則跳過
        if row['誤差%'] != "-" and row['誤差%'] != "": continue
        
        try:
            pred_date = datetime.strptime(row['日期'], '%Y-%m-%d')
            target_date = pred_date + timedelta(days=7) # 考慮假日，預測第5天約在1週後
            
            # 若時間未到則跳過
            if now < target_date: continue
            
            ticker = row['股票代號']
            pred_5 = float(row['5日預測-5'])
            
            # 抓取目標日期的實際價格
            hist = yf.download(ticker, start=target_date.strftime('%Y-%m-%d'), 
                               end=(target_date + timedelta(days=3)).strftime('%Y-%m-%d'), 
                               progress=False)
            
            if not hist.empty:
                actual_p = float(hist['Close'].iloc[0])
                error_val = ((actual_p - pred_5) / pred_5) * 100
                # 更新 J 欄 (第10欄)
                ws.update_cell(idx + 2, 10, f"{error_val:.2f}%")
                updated_count += 1
        except: continue
        
    return f"已成功更新 {updated_count} 筆資料的誤差分析"

# ==================== 3. 主執行流程 ====================

st.title("🏆 台股量化預測與自動對帳系統 v20.0")

tab1, tab2 = st.tabs(["🚀 執行預測", "🔄 誤差對帳"])

with tab1:
    if st.button("啟動 Top 100 量化預測任務"):
        client = get_gspread_client()
        if client:
            sh = client.open(SHEET_NAME)
            ws = sh.get_worksheet(0)
            
            # --- 增加表首：自動初始化 A-J 欄 ---
            headers = ["日期", "股票代號", "收盤價格", "交易值指標", "5日預測-1", "5日預測-2", "5日預測-3", "5日預測-4", "5日預測-5", "誤差%"]
            if not ws.row_values(1):
                ws.insert_row(headers, 1)
                st.info("已自動初始化 Excel 表首。")

            # 讀取資料
            raw = ws.get_all_values()
            df_sheet = pd.DataFrame(raw[1:], columns=raw[0]).loc[:, lambda x: x.columns != '']
            
            if '股票代號' not in df_sheet.columns:
                st.error("請確保 B1 欄位名稱為『股票代號』")
                st.stop()
                
            tickers = df_sheet['股票代號'].replace('', np.nan).dropna().head(100).tolist()
            st.info(f"正在執行 {len(tickers)} 檔量化分析...")
            
            all_hist = yf.download(tickers, period="3mo", group_by='ticker', threads=True, progress=False)
            
            for idx, t in enumerate(tickers):
                try:
                    df = all_hist[t].dropna() if isinstance(all_hist.columns, pd.MultiIndex) else all_hist.dropna()
                    if df.empty: continue
                    
                    preds = calculate_quant_logic(t, df)
                    # 寫入 E-J 欄
                    ws.update(f"E{idx+2}:J{idx+2}", [preds + ["-"]])
                except: continue
            st.success("🎉 預測更新完成！")

with tab2:
    st.markdown("### 📊 自動回填實際價格與計算誤差")
    st.write("系統將檢查一週前的預測，並自動從市場抓取實際收盤價計算誤差。")
    if st.button("執行誤差對帳任務"):
        client = get_gspread_client()
        if client:
            sh = client.open(SHEET_NAME)
            ws = sh.get_worksheet(0)
            msg = update_error_analysis(ws)
            st.success(msg)
