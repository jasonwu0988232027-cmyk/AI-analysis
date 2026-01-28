import streamlit as st
import pandas as pd
import requests
from datetime import datetime, timedelta

# --- 請確保這裡填入正確的 Key ---
FINNHUB_API_KEY = "你的_FINNHUB_API_KEY" 

def get_stock_candles(symbol):
    end = int(datetime.now().timestamp())
    start = int((datetime.now() - timedelta(days=30)).timestamp())
    
    url = f"https://finnhub.io/api/v1/stock/candle?symbol={symbol}&resolution=D&from={start}&to={end}&token={FINNHUB_API_KEY}"
    
    try:
        response = requests.get(url)
        # 檢查 HTTP 狀態碼
        if response.status_code == 401:
            st.error("❌ API Key 錯誤：請檢查你的 Finnhub Key 是否填寫正確。")
            return pd.DataFrame()
        elif response.status_code == 403:
            st.error("❌ 權限不足：免費版 API 可能不支援此市場（如部分台股）或請求過快。")
            return pd.DataFrame()
        
        res = response.json()
        
        # 檢查數據內容
        if res.get('s') == 'ok':
            df = pd.DataFrame({
                'Date': pd.to_datetime(res['t'], unit='s'),
                'Close': res['c'], 'Open': res['o'], 'High': res['h'], 'Low': res['l']
            })
            return df
        elif res.get('s') == 'no_data':
            st.warning(f"⚠️ 查無數據：代碼 '{symbol}' 在此時間範圍內無交易資料。")
        else:
            st.info(f"💡 伺服器回應：{res.get('error', '未知原因')}")
            
    except Exception as e:
        st.error(f"⚠️ 連線失敗: {e}")
        
    return pd.DataFrame()

# --- 測試建議 ---
st.sidebar.info("測試建議：\n1. 先輸入 AAPL 測試 API 是否正常。\n2. 台股請試試 2330.TW 或 2454.TW。")
