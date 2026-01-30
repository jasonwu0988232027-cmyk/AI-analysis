import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import gspread
import google.generativeai as genai
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
st.set_page_config(page_title="AI 股市預測系統 v16.8", layout="wide")

# --- 參數與金鑰設定 ---
SHEET_NAME = "Stock_Predictions_History"
CREDENTIALS_JSON = "eco-precept-485904-j5-7ef3cdda1b03.json"
DEFAULT_GEMINI_KEY = "AIzaSyDE4yDZMnniFaYLQd-LK7WSQpHh-6JRA3Q"
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY", DEFAULT_GEMINI_KEY)

# 修正模型呼叫：確保使用最新支援的型號名稱
try:
    genai.configure(api_key=GEMINI_API_KEY)
    # 2026 年建議使用此路徑或確認模型清單
    ai_model = genai.GenerativeModel('models/gemini-1.5-flash-latest')
except Exception as e:
    st.error(f"Gemini 初始化失敗: {e}")

# ==================== 1. 雲端連線模組 ====================

def get_gspread_client():
    """處理私鑰格式並建立連線"""
    scopes = ['https://www.googleapis.com/auth/spreadsheets', 'https://www.googleapis.com/auth/drive']
    try:
        if "gcp_service_account" in st.secrets:
            creds_info = dict(st.secrets["gcp_service_account"])
            # 修正 Metadata 報錯：強制轉義換行符號
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

def get_top_100_tickers():
    """步驟 1：讀取第一頁 B 欄的股票代號"""
    client = get_gspread_client()
    if not client: return []
    try:
        sh = client.open(SHEET_NAME)
        ws = sh.get_worksheet(0)
        data = ws.get_all_records()
        df = pd.DataFrame(data)
        # 抓取「股票代號」欄位的前 100 筆
        return df['股票代號'].dropna().astype(str).head(100).tolist()
    except Exception as e:
        st.error(f"讀取清單失敗: {e}")
        return []

# ==================== 2. 分析與預測邏輯 ====================

def fetch_stock_news(symbol):
    """步驟 2-二：搜尋四大新聞網相關報導"""
    stock_id = symbol.split('.')[0]
    headers = {'User-Agent': 'Mozilla/5.0'}
    sources = [
        "https://news.cnyes.com/news/cat/tw_stock_news",
        "https://money.udn.com/money/index",
        "https://www.ftnn.com.tw/category/6",
        "https://news.wearn.com/index.html"
    ]
    news_text = ""
    try:
        # 爬取新聞標題
        res = requests.get(sources[0], headers=headers, timeout=5)
        soup = BeautifulSoup(res.text, 'html.parser')
        relevant = [t.get_text() for t in soup.find_all(['h3', 'a']) if stock_id in t.get_text()]
        news_text = " ".join(relevant[:5])
    except: pass
    return news_text if news_text else "無即時重大新聞"

def get_technical_analysis(ticker, df):
    """步驟 2-一/三：積分制分析"""
    score = 0
    try:
        # 黃金交叉判定
        ma5 = df['Close'].rolling(5).mean().iloc[-1]
        ma20 = df['Close'].rolling(20).mean().iloc[-1]
        if ma5 > ma20: score += 2
        
        # 基本面：本益比
        info = yf.Ticker(ticker).info
        if info.get('forwardPE', 100) < 18: score += 1
    except: pass
    return score

# ==================== 3. 主執行程序 ====================

st.title("🤖 AI 股市深度分析系統 v16.8")

if st.button("🚀 開始分析第一頁之 Top 100 標的"):
    tickers = get_top_100_tickers()
    client = get_gspread_client()
    
    if client and tickers:
        sh = client.open(SHEET_NAME)
        ws = sh.get_worksheet(0)
        p_bar = st.progress(0)
        status = st.empty()
        
        # 批量獲取歷史數據減少 API 請求次數
        status.text("批量獲取市場數據中...")
        all_hist = yf.download(tickers, period="3mo", group_by='ticker', threads=True, progress=False)
        
        for idx, t in enumerate(tickers):
            try:
                status.text(f"分析中 ({idx+1}/100): {t}")
                df = all_hist[t].dropna() if isinstance(all_hist.columns, pd.MultiIndex) else all_hist.dropna()
                if df.empty: continue
                
                curr_p = round(float(df['Close'].iloc[-1]), 2)
                tech_score = get_technical_analysis(t, df)
                news_content = fetch_stock_news(t)
                
                # Gemini 預測
                prompt = f"股票{t}，價{curr_p}，分析分{tech_score}。新聞：{news_content}。預測未來5日價格。僅回傳5個數字(逗號分隔)。"
                response = ai_model.generate_content(prompt)
                preds = [float(p.strip()) for p in response.text.strip().split(',')]
                
                # 寫入 A-J 欄位（E-I為預測價，J為誤差）
                ws.update(f"E{idx+2}:J{idx+2}", [preds + ["-"]])
                
                # 冷卻預防封鎖
                time.sleep(random.uniform(1.2, 2.5))
                if (idx + 1) % 10 == 0:
                    time.sleep(15)
                    
            except Exception as e:
                st.warning(f"跳過 {t}: {e}")
                
            p_bar.progress((idx + 1) / len(tickers))
            
        status.text("✅ 任務已完成")
        st.success("🎉 分析與預測已成功寫入 Excel E-J 欄位！")
