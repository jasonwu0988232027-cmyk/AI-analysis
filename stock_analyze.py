import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import gspread
import google.generativeai as genai
import requests
from bs4 import BeautifulSoup
from google.oauth2.service_account import Credentials
from datetime import datetime, timedelta
import time
import os
import random
import urllib3

# --- 基礎配置 ---
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
st.set_page_config(page_title="Gemini AI 股市預測專家 v16.2", layout="wide")

# Google Sheets 與 AI 參數設定
SHEET_NAME = "Stock_Predictions_History"
CREDENTIALS_JSON = "eco-precept-485904-j5-7ef3cdda1b03.json"

# 請在 Streamlit Secrets 設定中填入您的 GEMINI_API_KEY
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY", "您的_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)
ai_model = genai.GenerativeModel('gemini-1.5-flash')

# ==================== 1. 雲端連線模組 (修正授權錯誤) ====================

def get_gspread_client():
    """修正 Illegal header value 報錯"""
    scopes = ['https://www.googleapis.com/auth/spreadsheets', 'https://www.googleapis.com/auth/drive']
    try:
        if "gcp_service_account" in st.secrets:
            creds_info = dict(st.secrets["gcp_service_account"])
            # 強制處理換行符號，避免傳輸外掛報錯
            creds_info["private_key"] = creds_info["private_key"].replace("\\n", "\n")
            creds = Credentials.from_service_account_info(creds_info, scopes=scopes)
        elif os.path.exists(CREDENTIALS_JSON):
            creds = Credentials.from_service_account_file(CREDENTIALS_JSON, scopes=scopes)
        else:
            return None
        return gspread.authorize(creds)
    except Exception as e:
        st.error(f"❌ 授權連線失敗: {e}")
        return None

def get_target_tickers():
    """步驟 1：讀取 Excel 第一頁的前 100 支股票"""
    client = get_gspread_client()
    if not client: return []
    try:
        sh = client.open(SHEET_NAME)
        ws = sh.get_worksheet(0)
        # 讀取整頁並尋找標題為 "股票代號" 的欄位
        data = ws.get_all_records()
        df = pd.DataFrame(data)
        return df['股票代號'].dropna().astype(str).head(100).tolist()
    except Exception as e:
        st.error(f"讀取 Excel 資料失敗: {e}")
        return []

# ==================== 2. 分析與新聞模組 ====================

def fetch_news_text(stock_code):
    """步驟 2-二：搜尋四大新聞網相關報導"""
    symbol = stock_code.split('.')[0]
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
    # 針對重點新聞網進行聚合
    news_urls = [
        f"https://news.cnyes.com/news/cat/tw_stock_news",
        f"https://money.udn.com/money/index",
        f"https://www.ftnn.com.tw/category/6",
        f"https://news.wearn.com/index.html"
    ]
    news_summary = ""
    try:
        # 為避免過度請求，隨機選取一個新聞源進行深度掃描
        target_url = random.choice(news_urls)
        res = requests.get(target_url, headers=headers, timeout=5)
        soup = BeautifulSoup(res.text, 'html.parser')
        # 抓取包含代碼的標題文字
        relevant_titles = [t.get_text() for t in soup.find_all(['h3', 'a']) if symbol in t.get_text()]
        news_summary = " ".join(relevant_titles[:5])
    except: pass
    return news_summary if news_summary else "目前無即時重大新聞"

def get_base_score(ticker_name, ticker_df):
    """步驟 2-三：技術面與基本面積分制"""
    score = 0
    try:
        # 技術面：MA5 與 MA20 黃金交叉
        ma5 = ticker_df['Close'].rolling(5).mean().iloc[-1]
        ma20 = ticker_df['Close'].rolling(20).mean().iloc[-1]
        if ma5 > ma20: score += 2 
        
        # 基本面：本益比資訊
        info = yf.Ticker(ticker_name).info
        if info.get('forwardPE', 100) < 16: score += 1
    except: pass
    return score

# ==================== 3. 主程序執行邏輯 (抗封鎖版) ====================

st.title("🛡️ AI 股市深度預測系統 v16.2")
st.info("模式：批量下載數據 + 積分制 AI 分析")

if st.button("🚀 開始全自動預測任務"):
    tickers = get_target_tickers()
    client = get_gspread_client()
    
    if client and tickers:
        sh = client.open(SHEET_NAME)
        ws = sh.get_worksheet(0)
        p_bar = st.progress(0)
        status_text = st.empty()
        
        # 核心優化：批量下載 100 支股票歷史數據，極大化減少請求頻率
        status_text.text("正在執行批量數據下載 (1/2)...")
        all_hist_data = yf.download(tickers, period="3mo", group_by='ticker', threads=True, progress=False)
        
        results_for_excel = []
        
        for idx, t in enumerate(tickers):
            try:
                status_text.text(f"正在分析 ({idx+1}/{len(tickers)}): {t}")
                
                # 從批量數據中提取個別 DataFrame
                if isinstance(all_hist_data.columns, pd.MultiIndex):
                    df = all_hist_data[t].dropna()
                else:
                    df = all_hist_data.dropna()
                
                if df.empty: continue
                
                # 執行分析與新聞爬蟲
                current_p = round(float(df['Close'].iloc[-1]), 2)
                base_score = get_base_score(t, df)
                news_content = fetch_news_text(t)
                
                # AI 整合預測 (Gemini)
                prompt = f"股票{t}，現價{current_p}，技術分{base_score}，新聞：{news_content}。請預測未來5日價格，格式：分數,價1,價2,價3,價4,價5"
                response = ai_model.generate_content(prompt)
                ai_preds = response.text.strip().split(',')
                
                # 取出 5 日價格並填入 J 欄誤差預留位
                pred_row = [float(p) for p in ai_preds[1:6]] + ["-"]
                
                # 即時更新 Excel 工作表 E-J 欄
                ws.update(f"E{idx+2}:J{idx+2}", [pred_row])
                
                # 智能冷卻：隨機休息 1~2 秒預防封鎖
                time.sleep(random.uniform(1.0, 2.0))
                
                # 每 10 支標的執行長休息 (15秒)
                if (idx + 1) % 10 == 0:
                    status_text.text(f"已完成 {idx+1} 檔，冷卻中避免觸發 Too Many Requests...")
                    time.sleep(15)
                    
            except Exception as e:
                st.warning(f"跳過 {t}: {e}")
                if "Too Many Requests" in str(e):
                    st.error("偵測到 IP 被限流，請暫停 15 分鐘後再試。")
                    break
            
            p_bar.progress((idx + 1) / len(tickers))
            
        status_text.text("✅ 任務執行完畢")
        st.success(f"🎉 已成功更新 {len(tickers)} 檔預測數據至雲端 A-J 欄位！")
