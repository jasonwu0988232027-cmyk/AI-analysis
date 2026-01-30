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
st.set_page_config(page_title="AI 股市全能專家 v16.7", layout="wide")

# --- 參數與金鑰設定 ---
SHEET_NAME = "Stock_Predictions_History"
CREDENTIALS_JSON = "eco-precept-485904-j5-7ef3cdda1b03.json"

# 使用您提供的預設 API KEY
DEFAULT_GEMINI_KEY = "AIzaSyDE4yDZMnniFaYLQd-LK7WSQpHh-6JRA3Q"
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY", DEFAULT_GEMINI_KEY)

# 初始化 Gemini AI
try:
    genai.configure(api_key=GEMINI_API_KEY)
    ai_model = genai.GenerativeModel('gemini-1.5-flash')
except Exception as e:
    st.error(f"Gemini 初始化失敗: {e}")

# ==================== 1. 雲端連線模組 (修正授權錯誤) ====================

def get_gspread_client():
    """處理私鑰格式並建立 Google Sheets 連線"""
    scopes = ['https://www.googleapis.com/auth/spreadsheets', 'https://www.googleapis.com/auth/drive']
    try:
        if "gcp_service_account" in st.secrets:
            creds_info = dict(st.secrets["gcp_service_account"])
            # 強制轉義換行符號防止 Metadata 報錯
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
    """步驟 1：從共用 EXCEL 第一頁抓取前 100 支股票"""
    client = get_gspread_client()
    if not client: return []
    try:
        sh = client.open(SHEET_NAME)
        ws = sh.get_worksheet(0) # 讀取第一頁
        data = ws.get_all_records()
        df = pd.DataFrame(data)
        # 抓取標題為 "股票代號" 的 B 欄數據
        return df['股票代號'].dropna().astype(str).head(100).tolist()
    except Exception as e:
        st.error(f"無法讀取股票清單: {e}")
        return []

# ==================== 2. 多維度分析與爬蟲模組 ====================

def fetch_stock_news(symbol):
    """步驟 2-二：爬蟲四大新聞網搜尋相關新聞"""
    stock_id = symbol.split('.')[0]
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
    # 定義爬取目標
    sources = [
        "https://www.ftnn.com.tw/category/6",           # FTNN
        "https://news.wearn.com/index.html",            # 聚財網
        "https://news.cnyes.com/news/cat/tw_stock_news",# 鉅亨網
        "https://money.udn.com/money/index"             # 經濟日報
    ]
    news_text = ""
    # 隨機挑選 1-2 個來源以防被封鎖 IP
    for url in random.sample(sources, 2):
        try:
            res = requests.get(url, headers=headers, timeout=5)
            soup = BeautifulSoup(res.text, 'html.parser')
            # 抓取標題中包含股票代碼的文字
            titles = [t.get_text() for t in soup.find_all(['h3', 'a', 'h2']) if stock_id in t.get_text()]
            news_text += " ".join(titles[:3]) + " "
        except: continue
    return news_text if news_text else "查無近期相關重大新聞"

def calculate_technical_score(ticker, df):
    """步驟 2-一：抓取基本面、技術面並實作積分制"""
    score = 0
    try:
        # 技術面：黃金交叉判定 (MA5 上穿 MA20)
        ma5 = df['Close'].rolling(5).mean().iloc[-1]
        ma20 = df['Close'].rolling(20).mean().iloc[-1]
        ma5_prev = df['Close'].rolling(5).mean().iloc[-2]
        ma20_prev = df['Close'].rolling(20).mean().iloc[-2]
        
        if ma5 > ma20 and ma5_prev <= ma20_prev:
            score += 5  # 強力黃金交叉加分
        elif ma5 > ma20:
            score += 2  # 多頭排列加分
            
        # 基本面：本益比 (PE Ratio)
        info = yf.Ticker(ticker).info
        pe = info.get('forwardPE', 100)
        if pe < 15: score += 3 # 低本益比加分
        elif pe < 25: score += 1
    except: pass
    return score

# ==================== 3. 主執行程序 (抗封鎖與預測) ====================

st.title("🛡️ AI 股市深度預測系統 v16.7")
st.markdown(f"**使用 Gemini API Key:** `{GEMINI_API_KEY[:8]}...`")

if st.button("🚀 開始全自動 Top 100 積分預測分析"):
    tickers = get_top_100_tickers()
    client = get_gspread_client()
    
    if client and tickers:
        sh = client.open(SHEET_NAME)
        ws = sh.get_worksheet(0)
        p_bar = st.progress(0)
        status_msg = st.empty()
        
        # 步驟 1：批量獲取全市場數據
        status_msg.text("正在同步批量市場歷史數據...")
        all_hist = yf.download(tickers, period="3mo", group_by='ticker', threads=True, progress=False)
        
        for idx, t in enumerate(tickers):
            try:
                status_msg.text(f"分析中 ({idx+1}/100): {t}")
                
                # 獲取個別股票數據
                df = all_hist[t].dropna() if isinstance(all_hist.columns, pd.MultiIndex) else all_hist.dropna()
                if df.empty: continue
                
                # 執行分析與爬蟲
                curr_price = round(float(df['Close'].iloc[-1]), 2)
                tech_score = calculate_technical_score(t, df)
                news_content = fetch_stock_news(t)
                
                # 步驟 2-二：丟給 Gemini 分析積分與預測走勢
                prompt = f"""
                分析股票 {t}。現價 {curr_price}。技術基本分 {tech_score}。新聞內容：{news_content}。
                請根據這些資訊給出未來 5 個交易日的預期收盤價。
                請嚴格遵守格式回答 5 個數字，以逗號分隔，不要有任何其他文字：
                數字1,數字2,數字3,數字4,數字5
                """
                response = ai_model.generate_content(prompt)
                # 解析預測價格
                preds = [float(p.strip()) for p in response.text.strip().split(',')]
                
                # 步驟 3：寫入 E-J 欄位
                # E-I 欄: 5日預測價格, J 欄: 誤差% (設為待定)
                ws.update(f"E{idx+2}:J{idx+2}", [preds + ["-"]])
                
                # 智能冷卻機制防止 Too Many Requests
                time.sleep(random.uniform(1.0, 2.0))
                if (idx + 1) % 10 == 0:
                    status_msg.text("執行分段冷卻中 (15秒)...")
                    time.sleep(15)
                    
            except Exception as e:
                st.warning(f"跳過 {t}: {e}")
                
            p_bar.progress((idx + 1) / len(tickers))
            
        status_msg.text("✅ 全部任務執行完畢")
        st.success("🎉 Top 100 標的基本面、技術面、新聞分析與 5 日預測已同步至雲端！")
