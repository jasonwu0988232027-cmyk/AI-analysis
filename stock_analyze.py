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
st.set_page_config(page_title="AI 股市深度預測系統 v16.9", layout="wide")

# --- 參數與金鑰設定 ---
SHEET_NAME = "Stock_Predictions_History"
CREDENTIALS_JSON = "eco-precept-485904-j5-7ef3cdda1b03.json"
DEFAULT_GEMINI_KEY = "AIzaSyDE4yDZMnniFaYLQd-LK7WSQpHh-6JRA3Q"
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY", DEFAULT_GEMINI_KEY)

# 初始化 Gemini AI：解決 404 模型找不到問題
try:
    genai.configure(api_key=GEMINI_API_KEY)
    # 使用通用的模型名稱，SDK 會自動處理版本對應
    ai_model = genai.GenerativeModel('gemini-1.5-flash')
except Exception as e:
    st.error(f"Gemini 初始化失敗: {e}")

# ==================== 1. 雲端連線模組 ====================

def get_gspread_client():
    """處理私鑰格式，修正 Illegal header value 報錯"""
    scopes = ['https://www.googleapis.com/auth/spreadsheets', 'https://www.googleapis.com/auth/drive']
    try:
        if "gcp_service_account" in st.secrets:
            creds_info = dict(st.secrets["gcp_service_account"])
            # 強制將轉義的 \n 換回真正的換行符號
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
    """步驟 1：從共用 Excel 第一頁抓取前 100 支股票代號"""
    client = get_gspread_client()
    if not client: return []
    try:
        sh = client.open(SHEET_NAME)
        ws = sh.get_worksheet(0)
        df = pd.DataFrame(ws.get_all_records())
        # 確保讀取標題為「股票代號」的欄位
        return df['股票代號'].dropna().astype(str).head(100).tolist()
    except Exception as e:
        st.error(f"讀取 Excel 清單失敗: {e}")
        return []

# ==================== 2. 分析核心：基本面、技術面與新聞爬蟲 ====================

def fetch_multi_news(symbol):
    """步驟 2-二：爬蟲四大新聞網搜尋相關新聞"""
    stock_id = symbol.split('.')[0]
    headers = {'User-Agent': 'Mozilla/5.0'}
    # 針對重點新聞來源進行掃描
    news_urls = [
        "https://news.cnyes.com/news/cat/tw_stock_news", # 鉅亨網
        "https://money.udn.com/money/index"             # 經濟日報
    ]
    summary = ""
    try:
        res = requests.get(news_urls[0], headers=headers, timeout=5)
        soup = BeautifulSoup(res.text, 'html.parser')
        # 抓取包含股票代碼的標題文字
        titles = [t.get_text() for t in soup.find_all(['h3', 'a']) if stock_id in t.get_text()]
        summary = " ".join(titles[:5])
    except: pass
    return summary if summary else "查無近期即時新聞"

def get_market_score(ticker, df):
    """步驟 2-一/三：積分制分析 (包含黃金交叉)"""
    score = 0
    try:
        # 技術面：MA5 與 MA20 黃金交叉判定
        ma5 = df['Close'].rolling(5).mean().iloc[-1]
        ma20 = df['Close'].rolling(20).mean().iloc[-1]
        if ma5 > ma20: score += 2 
        
        # 基本面：本益比 (PE Ratio) 加分
        info = yf.Ticker(ticker).info
        if info.get('forwardPE', 100) < 18: score += 1
    except: pass
    return score

# ==================== 3. 主程序執行流程 ====================

st.title("🤖 AI 股市深度分析與預測系統 v16.9")

if st.button("🚀 開始全市場 Top 100 分析預測"):
    tickers = get_top_100_tickers()
    client = get_gspread_client()
    
    if client and tickers:
        sh = client.open(SHEET_NAME)
        ws = sh.get_worksheet(0)
        p_bar = st.progress(0)
        status = st.empty()
        
        # 批量獲取歷史數據減少 API 請求頻率
        status.text("正在執行批量數據同步 (Batch Download)...")
        all_hist = yf.download(tickers, period="3mo", group_by='ticker', threads=True, progress=False)
        
        for idx, t in enumerate(tickers):
            try:
                status.text(f"分析中 ({idx+1}/100): {t}")
                
                # 提取個股歷史數據
                df = all_hist[t].dropna() if isinstance(all_hist.columns, pd.MultiIndex) else all_hist.dropna()
                if df.empty: continue
                
                curr_price = round(float(df['Close'].iloc[-1]), 2)
                tech_score = get_market_score(t, df)
                news_text = fetch_multi_news(t)
                
                # 步驟 2-二：由 Gemini 分析並給出 5 日預測價
                prompt = f"""
                分析股票 {t}。當前價 {curr_price}。技術基本分 {tech_score}。新聞內容：{news_text}。
                請預測未來 5 個交易日的收盤價。
                請僅回傳 5 個數字，並用逗號分隔，例如: 100.5,101.2,102,101.8,103
                """
                response = ai_model.generate_content(prompt)
                # 解析預測價格並處理可能出現的非數值字元
                raw_preds = response.text.strip().split(',')
                pred_row = [float(p.strip()) for p in raw_preds[:5]]
                
                # 步驟 3：寫入 Excel E-J 欄位
                # E-I: 預測價, J: 誤差% (設為待定 "-")
                final_row = pred_row + ["-"]
                ws.update(f"E{idx+2}:J{idx+2}", [final_row])
                
                # 智能冷卻機制防止 Too Many Requests
                time.sleep(random.uniform(1.2, 2.0))
                if (idx + 1) % 10 == 0:
                    status.text("執行冷卻機制中 (15秒)...")
                    time.sleep(15)
                    
            except Exception as e:
                st.warning(f"跳過 {t}: {e}")
                
            p_bar.progress((idx + 1) / len(tickers))
            
        status.text("✅ 預測任務已完成")
        st.success("🎉 分析與 5 日預測數據已成功寫入 Excel E-J 欄位！")
