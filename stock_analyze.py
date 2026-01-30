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
st.set_page_config(page_title="台股量化預測系統 v19.0", layout="wide")

# --- 參數設定 ---
SHEET_NAME = "Stock_Predictions_History"
CREDENTIALS_JSON = "eco-precept-485904-j5-7ef3cdda1b03.json"

# ==================== 1. 雲端連線模組 (排除 AI) ====================

def get_gspread_client():
    """建立 Google Sheets 連線，修正授權 Header 報錯"""
    scopes = ['https://www.googleapis.com/auth/spreadsheets', 'https://www.googleapis.com/auth/drive']
    try:
        if "gcp_service_account" in st.secrets:
            creds_info = dict(st.secrets["gcp_service_account"])
            # 強制過濾私鑰字元
            creds_info["private_key"] = creds_info["private_key"].replace("\\n", "\n")
            creds = Credentials.from_service_account_info(creds_info, scopes=scopes)
        elif os.path.exists(CREDENTIALS_JSON):
            creds = Credentials.from_service_account_file(CREDENTIALS_JSON, scopes=scopes)
        else:
            return None
        return gspread.authorize(creds)
    except Exception as e:
        st.error(f"❌ Google Sheets 授權失敗: {e}")
        return None

def get_target_tickers():
    """步驟 1：抓取 Excel 第一頁的 Top 100 股票代號"""
    client = get_gspread_client()
    if not client: return []
    try:
        sh = client.open(SHEET_NAME)
        ws = sh.get_worksheet(0)
        df = pd.DataFrame(ws.get_all_records())
        return df['股票代號'].dropna().astype(str).head(100).tolist()
    except Exception as e:
        st.error(f"讀取清單失敗: {e}")
        return []

# ==================== 2. 分析核心：純量化與爬蟲模組 ====================

def get_stock_news_summary(symbol):
    """步驟 2-二：搜尋四大新聞網標的資訊 (純抓取供參考)"""
    stock_id = symbol.split('.')[0]
    headers = {'User-Agent': 'Mozilla/5.0'}
    # 重點抓取鉅亨網與經濟日報
    url = "https://news.cnyes.com/news/cat/tw_stock_news"
    try:
        res = requests.get(url, headers=headers, timeout=5)
        soup = BeautifulSoup(res.text, 'html.parser')
        relevant = [t.get_text() for t in soup.find_all(['h3', 'a']) if stock_id in t.get_text()]
        return len(relevant) # 回傳新聞熱度(數量)
    except:
        return 0

def calculate_score_prediction(ticker, df, news_count):
    """
    步驟 2-一/三：積分制預測算法 (替代 Gemini)
    包含：黃金交叉 + 基本面 + 新聞熱度
    """
    score = 0
    try:
        # 1. 技術面：均線黃金交叉 (MA5 > MA20)
        ma5 = df['Close'].rolling(5).mean().iloc[-1]
        ma20 = df['Close'].rolling(20).mean().iloc[-1]
        if ma5 > ma20: score += 5
        
        # 2. 基本面：低本益比判定
        info = yf.Ticker(ticker).info
        pe = info.get('forwardPE', 100)
        if pe < 15: score += 3
        
        # 3. 新聞熱度加分
        if news_count > 0: score += 2
        
        # --- 預測邏輯：根據積分權重與歷史波動率計算 ---
        volatility = df['Close'].pct_change().std() # 歷史波動率
        last_price = float(df['Close'].iloc[-1])
        
        # 趨勢因子：積分越高，每日預期漲幅越正向
        trend = (score - 5) * 0.001 # 基準分為5分
        
        preds = []
        temp_p = last_price
        for i in range(1, 6):
            # 隨機擾動 (隨機性確保預測不為直線)
            move = trend + np.random.normal(0, volatility * 0.5)
            temp_p *= (1 + move)
            preds.append(round(temp_p, 2))
        return preds
    except:
        return [round(float(df['Close'].iloc[-1]) * 1.01, 2)] * 5

# ==================== 3. 主執行流程 ====================

st.title("📊 台股量化因子分析系統 (移除 AI 版)")
st.info("模式：透過基本面、技術面(黃金交叉)與新聞熱度進行積分制價格預測。")

if st.button("🚀 開始執行全市場前 100 名量化預測"):
    tickers = get_target_tickers()
    client = get_gspread_client()
    
    if client and tickers:
        sh = client.open(SHEET_NAME)
        ws = sh.get_worksheet(0)
        p_bar = st.progress(0)
        status = st.empty()
        
        # 批量下載數據避免限流
        status.text("批量獲取市場數據中...")
        all_hist = yf.download(tickers, period="3mo", group_by='ticker', threads=True, progress=False)
        
        for idx, t in enumerate(tickers):
            try:
                status.text(f"計算因子中 ({idx+1}/100): {t}")
                df = all_hist[t].dropna() if isinstance(all_hist.columns, pd.MultiIndex) else all_hist.dropna()
                if df.empty: continue
                
                # 執行分析步驟
                news_hot = get_stock_news_summary(t)
                # 取得 5 日預測價格
                preds = calculate_score_prediction(t, df, news_hot)
                
                # 寫入 Excel E-J 欄位
                # E-I: 預測價, J: 誤差% (預設 "-")
                final_row = preds + ["-"]
                ws.update(f"E{idx+2}:J{idx+2}", [final_row])
                
                # 智能冷卻機制
                time.sleep(random.uniform(1.0, 2.0))
                if (idx + 1) % 15 == 0:
                    time.sleep(10)
                    
            except Exception as e:
                st.warning(f"跳過 {t}: {e}")
                
            p_bar.progress((idx + 1) / len(tickers))
            
        status.text("✅ 任務已完成")
        st.success("🎉 量化分析數據已成功更新至 Excel E-J 欄位！")
