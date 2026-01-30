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

# --- 基礎配置 ---
st.set_page_config(page_title="Gemini AI 股市預測系統", layout="wide")

# 請在 Streamlit Secrets 中設定 GEMINI_API_KEY
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY", "您的_GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-1.5-flash')

SHEET_NAME = "Stock_Predictions_History"
CREDENTIALS_JSON = "eco-precept-485904-j5-7ef3cdda1b03.json" # 使用您上傳的金鑰

# ==================== 1. 雲端連線與資料讀取 ====================

def get_gspread_client():
    scopes = ['https://www.googleapis.com/auth/spreadsheets', 'https://www.googleapis.com/auth/drive']
    if "gcp_service_account" in st.secrets:
        creds = Credentials.from_service_account_info(dict(st.secrets["gcp_service_account"]), scopes=scopes)
    elif os.path.exists(CREDENTIALS_JSON):
        creds = Credentials.from_service_account_file(CREDENTIALS_JSON, scopes=scopes)
    else:
        return None
    return gspread.authorize(creds)

def get_top_100_from_sheet():
    """從第一分頁讀取前 100 支交易值指標股票"""
    client = get_gspread_client()
    if not client: return []
    sh = client.open(SHEET_NAME)
    ws = sh.get_worksheet(0) # 第一頁
    data = ws.get_all_records()
    df = pd.DataFrame(data)
    # 假設欄位名稱為 "股票代號"
    return df['股票代號'].head(100).tolist()

# ==================== 2. 多維度分析模組 ====================

def get_technical_score(df):
    """技術面積分：黃金交叉、RSI"""
    score = 0
    # 計算均線
    ma5 = df['Close'].rolling(window=5).mean()
    ma20 = df['Close'].rolling(window=20).mean()
    
    # 黃金交叉判定
    if ma5.iloc[-1] > ma20.iloc[-1] and ma5.iloc[-2] <= ma20.iloc[-2]:
        score += 5  # 強力訊號
    elif ma5.iloc[-1] > ma20.iloc[-1]:
        score += 2  # 多頭排列
        
    # RSI 判定
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    
    if rsi.iloc[-1] < 30: score += 3 # 超跌反彈
    elif rsi.iloc[-1] > 70: score -= 2 # 超買警戒
    
    return score, rsi.iloc[-1]

def get_fundamental_info(ticker_obj):
    """基本面數據獲取"""
    info = ticker_obj.info
    score = 0
    pe = info.get('trailingPE', 100)
    # 低本益比加分
    if pe < 15: score += 3
    elif pe < 25: score += 1
    
    return score, pe

# ==================== 3. 新聞爬蟲與 Gemini 分析 ====================

def crawl_news(stock_code):
    """針對指定標的搜尋新聞 (簡化版以防被封)"""
    symbol = stock_code.split('.')[0]
    headers = {'User-Agent': 'Mozilla/5.0'}
    news_text = ""
    
    # 範例：鉅亨網搜尋 (模擬)
    urls = [f"https://news.cnyes.com/news/cat/tw_stock_news"] 
    # 實際運作時可針對關鍵字串接搜尋 URL
    try:
        for url in urls[:1]: # 限制請求數
            res = requests.get(url, headers=headers, timeout=5)
            soup = BeautifulSoup(res.text, 'html.parser')
            # 抓取包含股票代碼或名稱的標題
            news_text += " ".join([t.get_text() for t in soup.find_all('h3') if symbol in t.get_text()][:5])
    except: pass
    return news_text if news_text else "查無近期重大新聞"

def get_ai_sentiment(news_text):
    """將新聞丟給 Gemini 進行情緒評分與預測建議"""
    prompt = f"""
    分析以下股票新聞文本，請給出：
    1. 情緒積分 (-5 到 5，5 最利多)
    2. 預期未來 5 日走勢方向。
    新聞內容：{news_text}
    請僅回答：分數,方向 (例如: 3,看多)
    """
    try:
        response = model.generate_content(prompt)
        res = response.text.strip().split(',')
        return int(res[0]), res[1]
    except:
        return 0, "中性"

# ==================== 4. 主程式邏輯 ====================

st.title("🤖 Gemini AI 多因子股票預測系統")

if st.button("🚀 開始分析 Top 100 標的"):
    tickers = get_top_100_from_sheet()
    if not tickers:
        st.error("無法從 Excel 讀取股票代號，請檢查第一頁 B 欄。")
    else:
        client = get_gspread_client()
        sh = client.open(SHEET_NAME)
        ws = sh.get_worksheet(0)
        
        progress = st.progress(0)
        results_to_update = []
        
        for idx, ticker in enumerate(tickers):
            try:
                # 1. 抓取數據
                t_obj = yf.Ticker(ticker)
                df = t_obj.history(period="1mo")
                if df.empty: continue
                
                # 2. 基本面 & 技術面
                f_score, pe = get_fundamental_info(t_obj)
                t_score, rsi = get_technical_score(df)
                
                # 3. 新聞分析
                news = crawl_news(ticker)
                ai_score, ai_dir = get_ai_sentiment(news)
                
                # 4. 總積分與 5 日預測
                total_score = f_score + t_score + ai_score
                last_price = df['Close'].iloc[-1]
                
                # 簡單預測模型：根據積分調整波動率
                preds = []
                for i in range(1, 6):
                    move = (total_score * 0.005) + np.random.normal(0, 0.01)
                    pred_price = last_price * (1 + move * i)
                    preds.append(round(pred_price, 2))
                
                # 5. 準備更新至 E-J 欄
                # E-I: 預測 1-5, J: 誤差 (新預測為待定)
                row_data = preds + ["-"] 
                results_to_update.append({"range": f"E{idx+2}:J{idx+2}", "values": [row_data]})
                
                st.write(f"✅ {ticker} 分析完成 | 積分: {total_score} | 預測: {preds[4]}")
                
            except Exception as e:
                st.warning(f"跳過 {ticker}: {e}")
            
            progress.progress((idx + 1) / len(tickers))
            time.sleep(1) # 防封鎖延遲
        
        # 批量更新 Excel
        if results_to_update:
            for item in results_to_update:
                ws.update(item['range'], item['values'])
            st.success("🎉 所有預測已更新至 Excel E-J 欄！")
