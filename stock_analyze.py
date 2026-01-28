import streamlit as st
import yfinance as yf
import pandas as pd
import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta

# --- 頁面設定 ---
st.set_page_config(page_title="AI 股市智囊團", layout="wide")
st.title("📈 AI 股市行業變動與個股預測")

# --- 1. 爬蟲函數 ---
def get_yahoo_news(stock_id=None):
    if stock_id:
        url = f"https://tw.stock.yahoo.com/quote/{stock_id}/news"
    else:
        url = "https://tw.stock.yahoo.com/news/"
    
    headers = {'User-Agent': 'Mozilla/5.0'}
    try:
        res = requests.get(url, headers=headers)
        soup = BeautifulSoup(res.text, 'html.parser')
        # 抓取新聞標題
        titles = soup.find_all('h3', class_='Mt(0) Mb(8px)', limit=10)
        return [t.get_text() for t in titles]
    except:
        return ["無法取得新聞數據"]

# --- 2. 模擬 AI 情緒分析 (可替換為 OpenAI/Gemini API) ---
def analyze_sentiment(news_list):
    # 這裡建議串接 OpenAI API，目前以模擬邏輯演示
    results = []
    for news in news_list:
        # 模擬分數: 隨機模擬 AI 判斷（實際應用時請調用 LLM）
        score = 0.5 if "漲" in news or "旺" in news else (-0.5 if "跌" in news or "壓力" in news else 0.1)
        results.append({"標題": news, "情緒分數": score})
    return pd.DataFrame(results)

# --- 側邊欄：使用者輸入 ---
st.sidebar.header("搜尋參數")
stock_input = st.sidebar.text_input("請輸入股票代碼 (例如: 2330.TW)", "2330.TW")
days_to_look = st.sidebar.slider("歷史數據天數", 5, 30, 14)

# --- 主畫面佈局 ---
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader(f"📊 {stock_input} 近期走勢")
    stock_data = yf.Ticker(stock_input)
    df = stock_data.history(period=f"{days_to_look}d")
    if not df.empty:
        st.line_chart(df['Close'])
        st.write("最新收盤價：", round(df['Close'].iloc[-1], 2))
    else:
        st.error("找不到該股票數據，請確保代碼正確（如台股需加 .TW）")

with col2:
    st.subheader("📰 最新消息與情緒分析")
    news_data = get_yahoo_news(stock_input.split('.')[0])
    sentiment_df = analyze_sentiment(news_data)
    
    # 顯示情緒圖表
    avg_score = sentiment_df['情緒分數'].mean()
    st.metric("平均市場情緒", f"{avg_score:.2f}", delta="偏多" if avg_score > 0 else "偏空")
    st.dataframe(sentiment_df)

# --- AI 綜合預測 ---
st.divider()
st.subheader("🤖 AI 綜合預測報告")
if st.button("生成分析報告"):
    with st.spinner('AI 正在分析技術面與消息面...'):
        # 這裡組合 Prompt 傳給 AI
        tech_info = f"過去{days_to_look}天平均股價: {df['Close'].mean():.2f}"
        news_info = ", ".join(news_data)
        
        # 模擬 AI 回傳內容
        st.info(f"""
        **【AI 判斷結果】**
        1. **消息面**：當前新聞情緒分數為 {avg_score:.2f}，市場對該行業/個股抱持{'樂觀' if avg_score > 0 else '謹慎'}態度。
        2. **技術面**：參考近期收盤價，目前處於{'上升' if df['Close'].iloc[-1] > df['Close'].iloc[0] else '修正'}階段。
        3. **5日預測**：結合{stock_input}的新聞與股價，預計未來 5 天將呈現{'震盪向上' if avg_score > 0 else '壓力測試'}，重點關注行業出口數據。
        """)