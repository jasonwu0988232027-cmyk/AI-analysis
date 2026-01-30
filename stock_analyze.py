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
st.set_page_config(page_title="Gemini AI 股市分析 v17.0", layout="wide")

# --- 參數與金鑰設定 ---
SHEET_NAME = "Stock_Predictions_History"
CREDENTIALS_JSON = "eco-precept-485904-j5-7ef3cdda1b03.json"
# 預設金鑰設定
DEFAULT_GEMINI_KEY = "AIzaSyDE4yDZMnniFaYLQd-LK7WSQpHh-6JRA3Q"
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY", DEFAULT_GEMINI_KEY)

# 修正模型呼叫邏輯：解決 404 找不到模型問題
try:
    genai.configure(api_key=GEMINI_API_KEY)
    # 使用通用名稱，SDK 會自動解析為 models/gemini-1.5-flash
    ai_model = genai.GenerativeModel('gemini-1.5-flash')
except Exception as e:
    st.error(f"Gemini 初始化失敗: {e}")

# ==================== 1. 雲端連線模組 ====================

def get_gspread_client():
    """修正 Illegal header value 報錯並建立連線"""
    scopes = ['https://www.googleapis.com/auth/spreadsheets', 'https://www.googleapis.com/auth/drive']
    try:
        if "gcp_service_account" in st.secrets:
            creds_info = dict(st.secrets["gcp_service_account"])
            # 強制將轉義的 \\n 換回真正的換行符號，防止 Header 驗證失敗
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
    """步驟 1：從共用 Excel 第一頁抓取當日交易值前 100 股票"""
    client = get_gspread_client()
    if not client: return []
    try:
        sh = client.open(SHEET_NAME)
        ws = sh.get_worksheet(0) # 讀取第一頁
        data = ws.get_all_records()
        df = pd.DataFrame(data)
        # 抓取標題為「股票代號」的欄位數據
        return df['股票代號'].dropna().astype(str).head(100).tolist()
    except Exception as e:
        st.error(f"讀取 Excel 資料失敗: {e}")
        return []

# ==================== 2. 分析核心模組 ====================

def fetch_web_news(symbol):
    """步驟 2-二：爬蟲四大新聞網搜尋相關新聞"""
    stock_id = symbol.split('.')[0]
    headers = {'User-Agent': 'Mozilla/5.0'}
    # 重點抓取鉅亨網與經濟日報
    urls = [
        f"https://news.cnyes.com/news/cat/tw_stock_news",
        f"https://money.udn.com/money/index"
    ]
    news_text = ""
    try:
        # 隨機選擇來源以降低被封鎖風險
        res = requests.get(random.choice(urls), headers=headers, timeout=5)
        soup = BeautifulSoup(res.text, 'html.parser')
        # 擷取包含代碼的標題
        titles = [t.get_text() for t in soup.find_all(['h3', 'a']) if stock_id in t.get_text()]
        news_text = " ".join(titles[:5])
    except: pass
    return news_text if news_text else "查無近期即時新聞"

def get_technical_factor_score(ticker, df):
    """步驟 2-一/三：積分制分析 (包含基本面、技術面、黃金交叉)"""
    score = 0
    try:
        # 技術面：MA5 與 MA20 黃金交叉判定
        ma5 = df['Close'].rolling(5).mean().iloc[-1]
        ma20 = df['Close'].rolling(20).mean().iloc[-1]
        if ma5 > ma20: score += 2  # 黃金交叉加分
        
        # 基本面：本益比 (PE Ratio)
        info = yf.Ticker(ticker).info
        if info.get('forwardPE', 100) < 18: score += 1
    except: pass
    return score

# ==================== 3. 主程序執行流程 ====================

st.title("🤖 AI 股市深度分析系統 v17.0")

if st.button("🚀 開始執行 Top 100 多因子預測"):
    tickers = get_target_tickers()
    client = get_gspread_client()
    
    if client and tickers:
        sh = client.open(SHEET_NAME)
        ws = sh.get_worksheet(0)
        p_bar = st.progress(0)
        status = st.empty()
        
        # 批量獲取數據以減少 API 請求頻率
        status.text("正在批量執行全市場數據下載...")
        all_hist = yf.download(tickers, period="3mo", group_by='ticker', threads=True, progress=False)
        
        for idx, t in enumerate(tickers):
            try:
                status.text(f"分析中 ({idx+1}/100): {t}")
                
                # 提取個股歷史數據
                df = all_hist[t].dropna() if isinstance(all_hist.columns, pd.MultiIndex) else all_hist.dropna()
                if df.empty: continue
                
                curr_price = round(float(df['Close'].iloc[-1]), 2)
                tech_score = get_technical_factor_score(t, df)
                news_content = fetch_web_news(t)
                
                # 步驟 2-二：AI 積分制分析與預測
                prompt = f"""
                分析股票 {t}。現價 {curr_price}。技術基本分 {tech_score}。新聞內容：{news_content}。
                請根據黃金交叉、基本面與新聞給出積分，並預測未來 5 個交易日的收盤價。
                請僅回答 5 個數字並以逗號分隔，例如: 100,101,102,101,103
                """
                response = ai_model.generate_content(prompt)
                # 解析 AI 回傳的預測價
                pred_row = [float(p.strip()) for p in response.text.strip().split(',')[:5]]
                
                # 步驟 3：寫入 Excel E-J 欄位
                # E-I: 預測1-5日, J: 誤差% (設為待定 "-")
                final_data = pred_row + ["-"]
                ws.update(f"E{idx+2}:J{idx+2}", [final_data])
                
                # 智能冷卻機制
                time.sleep(random.uniform(1.2, 2.5))
                if (idx + 1) % 10 == 0:
                    status.text("分段冷卻中 (15秒)...")
                    time.sleep(15)
                    
            except Exception as e:
                st.warning(f"跳過 {t}: {e}")
                
            p_bar.progress((idx + 1) / len(tickers))
            
        status.text("✅ 任務執行完成")
        st.success("🎉 分析與預測數據已成功同步至 Excel E-J 欄！")
