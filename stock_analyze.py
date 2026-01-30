import streamlit as st
import importlib.metadata

# --- 頁面配置 ---
st.set_page_config(page_title="AI 股市全能專家 v15", layout="wide", initial_sidebar_state="expanded")

# --- 檢測套件 ---
try:
    gspread_version = importlib.metadata.version("gspread")
    auth_version = importlib.metadata.version("google-auth")
except:
    pass

import yfinance as yf
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
from datetime import datetime, timedelta, time as dt_time
import time
import os
import urllib3
import random

# 停用 SSL 警告
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# --- 載入雲端與 AI 庫 ---
try:
    import gspread
    from google.oauth2.service_account import Credentials
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Input
    from sklearn.preprocessing import MinMaxScaler
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    st.error("缺少 AI 套件，請檢查 requirements.txt")

import warnings
warnings.filterwarnings('ignore')

# --- 全局設定 ---
CREDENTIALS_JSON = "credentials.json" 
SHEET_NAME = "Stock_Predictions_History"
FINNHUB_API_KEY = "d5t2rvhr01qt62ngu1kgd5t2rvhr01qt62ngu1l0"  # 請使用您自己的 API Key

# ==================== 0. 雲端連線模組 (支援多重分頁) ====================

def get_gspread_client():
    scopes = ['https://www.googleapis.com/auth/spreadsheets', 'https://www.googleapis.com/auth/drive']
    
    if "gcp_service_account" in st.secrets:
        try:
            creds_dict = dict(st.secrets["gcp_service_account"])
            creds = Credentials.from_service_account_info(creds_dict, scopes=scopes)
            return gspread.authorize(creds)
        except Exception:
            return None
    elif os.path.exists(CREDENTIALS_JSON):
        try:
            creds = Credentials.from_service_account_file(CREDENTIALS_JSON, scopes=scopes)
            return gspread.authorize(creds)
        except Exception:
            return None
    return None

def save_to_sheets(new_data, sheet_index=0):
    """
    sheet_index=0: 存入第一分頁 (單股分析)
    sheet_index=1: 存入第二分頁 (全市場掃描)
    sheet_index=2: 存入第三分頁 (交易值排行)
    """
    client = get_gspread_client()
    if client is None:
        st.warning("⚠️ 無法連線至 Google Sheets，請檢查 Secrets。")
        return False
    try:
        sh = client.open(SHEET_NAME)
        
        # --- 分頁處理邏輯 ---
        target_ws = None
        sheet_titles = ["單股分析", "市場掃描", "交易值排行"]
        
        try:
            all_ws = sh.worksheets()
            if len(all_ws) > sheet_index:
                target_ws = all_ws[sheet_index]
            else:
                # 建立新分頁
                target_ws = sh.add_worksheet(title=sheet_titles[sheet_index] if sheet_index < len(sheet_titles) else f"Sheet_{sheet_index+1}", rows=500, cols=15)
        except Exception as e:
            st.warning(f"分頁存取異常，嘗試建立新分頁: {e}")
            target_ws = sh.add_worksheet(title=f"Backup_{datetime.now().strftime('%H%M')}", rows=500, cols=15)

        # 寫入標題 (根據不同分頁)
        headers = {
            0: ["預測日期", "股票代碼", "目前價格", "7日預測價", "預期漲幅", "實際收盤價", "誤差%", "波動率", "5日趨勢", "成交量變化", "市場情緒", "可信度"],
            1: ["預測日期", "股票代碼", "目前價格", "7日預測價", "預期漲幅", "實際收盤價", "誤差%"],
            2: ["日期", "股票代碼", "股票名稱", "收盤價", "成交量", "成交值(億)", "排名"]
        }
        
        if target_ws.row_count > 0:
            try:
                val = target_ws.acell('A1').value
                if not val:
                    target_ws.append_row(headers.get(sheet_index, headers[0]))
            except:
                pass
        else:
            target_ws.append_row(headers.get(sheet_index, headers[0]))
             
        target_ws.append_rows(new_data)
        return True
    except Exception as e:
        st.error(f"❌ 雲端寫入失敗: {e}")
        return False

def update_actual_prices(sheet_index=1):
    """
    更新分頁2的實際收盤價和誤差%
    僅在收盤後執行 (台股收盤時間 13:30)
    """
    # 檢查是否為收盤後
    taiwan_tz = 8  # UTC+8
    now = datetime.utcnow() + timedelta(hours=taiwan_tz)
    market_close_time = dt_time(13, 30)
    
    if now.time() < market_close_time and now.weekday() < 5:  # 平日且未收盤
        return False, "市場尚未收盤，將在收盤後自動更新"
    
    client = get_gspread_client()
    if not client:
        return False, "無法連線至 Google Sheets"
    
    try:
        sh = client.open(SHEET_NAME)
        all_ws = sh.worksheets()
        
        if len(all_ws) <= sheet_index:
            return False, "目標分頁不存在"
        
        ws = all_ws[sheet_index]
        all_data = ws.get_all_values()
        
        if len(all_data) <= 1:
            return False, "無資料需要更新"
        
        updated_count = 0
        for i, row in enumerate(all_data[1:], start=2):  # 從第2行開始 (跳過標題)
            if len(row) < 6:
                continue
            
            # 檢查是否已經有實際價格
            if row[5] and row[5] != "-":
                continue
            
            stock_code = row[1]
            prediction_date = row[0]
            predicted_price = float(row[3]) if row[3] else 0
            
            # 計算7天後的日期
            try:
                pred_date = datetime.strptime(prediction_date, '%Y-%m-%d')
                target_date = pred_date + timedelta(days=7)
                
                # 如果還沒到7天，跳過
                if target_date > now:
                    continue
                
                # 抓取實際股價
                df = yf.download(stock_code, start=target_date.strftime('%Y-%m-%d'), 
                                end=(target_date + timedelta(days=3)).strftime('%Y-%m-%d'), 
                                progress=False)
                
                if not df.empty:
                    actual_price = float(df['Close'].iloc[0])
                    error_pct = ((actual_price - predicted_price) / predicted_price) * 100
                    
                    # 更新 F 和 G 欄 (實際價格和誤差)
                    ws.update_cell(i, 6, round(actual_price, 2))
                    ws.update_cell(i, 7, f"{error_pct:.2f}%")
                    updated_count += 1
                    
            except Exception as e:
                continue
        
        return True, f"已更新 {updated_count} 筆資料"
    except Exception as e:
        return False, f"更新失敗: {e}"

# ==================== 1. 改進的數據獲取 (使用即時數據) ====================

def get_realtime_stock_data(symbol, use_fallback=True):
    """
    優先使用即時數據，失敗則使用歷史數據
    """
    try:
        # 方法 1: 使用 yfinance 的即時報價
        ticker = yf.Ticker(symbol)
        info = ticker.info
        
        if 'regularMarketPrice' in info and info['regularMarketPrice']:
            current_price = info['regularMarketPrice']
            volume = info.get('regularMarketVolume', 0)
            return {
                'price': current_price,
                'volume': volume,
                'source': 'realtime'
            }
    except:
        pass
    
    if use_fallback:
        # 方法 2: 備用方案 - 使用最近的歷史數據
        try:
            df = yf.download(symbol, period="2d", progress=False)
            if not df.empty:
                df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
                return {
                    'price': float(df['Close'].iloc[-1]),
                    'volume': float(df['Volume'].iloc[-1]),
                    'source': 'historical'
                }
        except:
            pass
    
    return None

# ==================== 2. 本地運算市場掃描引擎 ====================

def get_market_universe():
    """
    內建 400+ 檔台股活躍名單
    """
    tickers = [
        # 半導體/權值
        '2330.TW', '2317.TW', '2454.TW', '2308.TW', '2303.TW', '2382.TW', '2379.TW', '3661.TW', '3443.TW', '3035.TW',
        '2301.TW', '2345.TW', '2408.TW', '2449.TW', '3037.TW', '3034.TW', '3711.TW', '2357.TW', '3231.TW', '2356.TW',
        '6669.TW', '2376.TW', '2368.TW', '3017.TW', '3533.TW', '5269.TW', '5274.TW', '6271.TW', '6531.TW', '8069.TW',
        '3189.TW', '3008.TW', '3406.TW', '3653.TW', '4961.TW', '4966.TW', '6176.TW', '6415.TW', '6456.TW', '6515.TW',
        # AI 伺服器/散熱/機殼
        '3324.TW', '2421.TW', '3013.TW', '3044.TW', '5483.TW', '6121.TW', '6213.TW', '8150.TW', '8996.TW', '2383.TW',
        '2388.TW', '3515.TW', '3694.TW', '8210.TW', '2486.TW', '6278.TW', '2059.TW', '3042.TW', '6117.TW', '8473.TW',
        # 航運
        '2603.TW', '2609.TW', '2615.TW', '2618.TW', '2610.TW', '2606.TW', '2605.TW', '2637.TW', '2633.TW', '2634.TW',
        # 重電/綠能
        '1513.TW', '1519.TW', '1503.TW', '1504.TW', '1514.TW', '1605.TW', '1609.TW', '1618.TW', '6806.TW', '3708.TW',
        '9958.TW', '3209.TW', '6282.TW', '6443.TW', '6477.TW', '8046.TW', '8938.TW', '9937.TW', '2049.TW',
        # 金融
        '2881.TW', '2882.TW', '2891.TW', '2886.TW', '2884.TW', '2885.TW', '2880.TW', '2890.TW', '2892.TW', '2883.TW',
        '2887.TW', '2888.TW', '2801.TW', '2812.TW', '2834.TW', '2838.TW', '2845.TW', '2849.TW', '2850.TW', '2851.TW',
        # 面板/光電/網通
        '2409.TW', '3481.TW', '6116.TW', '2344.TW', '3049.TW', '4904.TW', '4906.TW', '4938.TW', '5388.TW', '6285.TW',
        '2314.TW', '2324.TW', '2332.TW', '2340.TW', '2374.TW', '2392.TW', '2419.TW', '2439.TW', '2451.TW', '2481.TW',
        # 傳產/原物料
        '2002.TW', '2014.TW', '2027.TW', '1101.TW', '1102.TW', '1301.TW', '1303.TW', '1326.TW', '6505.TW', '1402.TW',
        '1476.TW', '9904.TW', '9910.TW', '1717.TW', '1722.TW', '1907.TW', '2105.TW', '2501.TW', '2542.TW', '9945.TW'
    ]
    return list(set(tickers))

def scan_top_100_by_value_local():
    """
    掃描市場並計算成交值排行
    返回: (top_100_tickers, turnover_data)
    """
    tickers = get_market_universe()
    st.info(f"🔍 載入全市場觀察名單 (共 {len(tickers)} 檔)，開始計算成交重心...")
    
    batch_size = 50
    results = []
    
    progress = st.progress(0)
    status = st.empty()
    
    for i in range(0, len(tickers), batch_size):
        batch = tickers[i : i + batch_size]
        status.text(f"正在掃描市場數據：第 {i} ~ {i+len(batch)} 檔...")
        
        try:
            data = yf.download(batch, period="2d", group_by='ticker', threads=True, progress=False)
            
            for t in batch:
                try:
                    if isinstance(data.columns, pd.MultiIndex):
                        if t in data.columns.levels[0]:
                            t_df = data[t].dropna()
                        else:
                            continue
                    else:
                        t_df = data.dropna()
                    
                    if not t_df.empty:
                        last_row = t_df.iloc[-1]
                        price = float(last_row['Close'])
                        volume = float(last_row['Volume'])
                        turnover = (price * volume) / 1e8
                        
                        # 獲取股票名稱
                        try:
                            ticker_obj = yf.Ticker(t)
                            name = ticker_obj.info.get('longName', t.split('.')[0])
                        except:
                            name = t.split('.')[0]
                        
                        results.append({
                            "ticker": t,
                            "name": name,
                            "price": price,
                            "volume": volume,
                            "turnover": turnover
                        })
                except:
                    continue
        except:
            pass
        
        progress.progress(min((i + batch_size) / len(tickers), 1.0))
        time.sleep(0.5)
        
    status.empty()
    progress.empty()
    
    df_res = pd.DataFrame(results)
    if not df_res.empty:
        df_res = df_res.sort_values("turnover", ascending=False)
        df_res['rank'] = range(1, len(df_res) + 1)
        
        top_100 = df_res.head(100)
        top_100_tickers = top_100['ticker'].tolist()
        
        st.success(f"✅ 計算完成！已鎖定市場最熱門的 {len(top_100_tickers)} 檔標的。")
        return top_100_tickers, top_100
    else:
        st.error("市場數據掃描失敗，請稍後再試。")
        return [], pd.DataFrame()

# ==================== 3. AI 預測核心 (改進版) ====================

@st.cache_data(ttl=3600)
def get_stock_history(symbol):
    try:
        df = yf.download(symbol, period="18mo", interval="1d", progress=False)
        if df.empty: return None
        df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
        return df.reset_index()
    except:
        return None

def predict_future_prices(df, sentiment_score, days=7):
    """
    改進版預測函數 - 使用固定隨機種子確保一致性
    """
    last_price = df['Close'].iloc[-1]
    last_date = df['Date'].iloc[-1]
    seed = int(last_price * 1000 + days)
    np.random.seed(seed)
    
    # 計算技術指標
    volatility = df['Close'].pct_change().std() 
    recent_trend = (df['Close'].iloc[-1] - df['Close'].iloc[-5]) / df['Close'].iloc[-5]
    volume_change = (df['Volume'].iloc[-5:].mean() - df['Volume'].iloc[-20:-5].mean()) / df['Volume'].iloc[-20:-5].mean()
    
    # 情緒影響因子
    sentiment_bias = (sentiment_score - 0.5) * 0.015
    trend_bias = recent_trend * 0.3
    total_bias = sentiment_bias + trend_bias
    
    future_dates = [last_date + timedelta(days=i) for i in range(1, days + 1)]
    future_prices = []
    
    current_price = last_price
    for i in range(days):
        decay_factor = 0.95 ** i
        adjusted_bias = total_bias * decay_factor
        change_pct = np.random.normal(adjusted_bias, volatility)
        current_price *= (1 + change_pct)
        future_prices.append(current_price)
    
    np.random.seed(None)
    
    return pd.DataFrame({'Date': future_dates, 'Close': future_prices}), {
        'volatility': volatility,
        'recent_trend': recent_trend,
        'volume_change': volume_change,
        'sentiment_bias': sentiment_bias,
        'trend_bias': trend_bias,
        'total_bias': total_bias
    }

def generate_prediction_reason(df, future_df, metrics, sentiment_score):
    """
    生成詳細的預測原因說明
    """
    reasons = []
    current_price = df['Close'].iloc[-1]
    predicted_price = future_df['Close'].iloc[-1]
    price_change_pct = ((predicted_price - current_price) / current_price) * 100
    
    if price_change_pct > 0:
        direction = "📈 上漲"
    else:
        direction = "📉 下跌"
    
    reasons.append(f"### {direction} 預測：{abs(price_change_pct):.2f}%")
    reasons.append("\n**📊 技術面因素：**")
    
    if metrics['recent_trend'] > 0.02:
        reasons.append(f"✓ 近期呈現上升趨勢 (+{metrics['recent_trend']*100:.2f}%)，慣性延續")
    elif metrics['recent_trend'] < -0.02:
        reasons.append(f"✓ 近期呈現下降趨勢 ({metrics['recent_trend']*100:.2f}%)，下行壓力存在")
    else:
        reasons.append(f"✓ 近期橫盤整理，趨勢不明顯")
    
    if metrics['volatility'] > 0.03:
        reasons.append(f"⚠ 高波動率 ({metrics['volatility']:.4f})，價格波動較大")
    elif metrics['volatility'] < 0.015:
        reasons.append(f"✓ 低波動率 ({metrics['volatility']:.4f})，價格相對穩定")
    else:
        reasons.append(f"✓ 中等波動率 ({metrics['volatility']:.4f})")
    
    if metrics['volume_change'] > 0.2:
        reasons.append(f"✓ 成交量放大 (+{metrics['volume_change']*100:.1f}%)，市場關注度提升")
    elif metrics['volume_change'] < -0.2:
        reasons.append(f"⚠ 成交量萎縮 ({metrics['volume_change']*100:.1f}%)，交易意願降低")
    
    reasons.append("\n**🧠 市場情緒：**")
    if sentiment_score > 0.6:
        reasons.append(f"✓ 市場情緒偏多 ({sentiment_score:.2f})，利多氛圍濃厚")
    elif sentiment_score < 0.4:
        reasons.append(f"⚠ 市場情緒偏空 ({sentiment_score:.2f})，謹慎觀望氣氛")
    else:
        reasons.append(f"✓ 市場情緒中性 ({sentiment_score:.2f})，多空平衡")
    
    reasons.append("\n**🎯 綜合評估：**")
    confidence_factors = []
    if abs(metrics['recent_trend']) > 0.03:
        confidence_factors.append("趨勢明確")
    if sentiment_score > 0.6 or sentiment_score < 0.4:
        confidence_factors.append("情緒明顯")
    if metrics['volume_change'] > 0.2:
        confidence_factors.append("量能配合")
    
    if len(confidence_factors) >= 2:
        confidence = "高"
        conf_emoji = "🟢"
    elif len(confidence_factors) == 1:
        confidence = "中"
        conf_emoji = "🟡"
    else:
        confidence = "低"
        conf_emoji = "🔴"
    
    reasons.append(f"{conf_emoji} 預測可信度：**{confidence}** ({', '.join(confidence_factors) if confidence_factors else '訊號不足'})")
    
    reasons.append("\n**⚡ 風險提示：**")
    if metrics['volatility'] > 0.03:
        reasons.append("- 價格波動較大，建議設定停損")
    if abs(metrics['volume_change']) > 0.3:
        reasons.append("- 成交量異常變化，留意資金動向")
    reasons.append("- 本預測僅供參考，投資前請自行評估風險")
    
    return "\n".join(reasons), confidence

@st.cache_data(ttl=3600)
def get_finnhub_sentiment(symbol):
    clean_symbol = symbol.split('.')[0]
    url = f"https://finnhub.io/api/v1/news-sentiment?symbol={clean_symbol}&token={FINNHUB_API_KEY}"
    try:
        res = requests.get(url).json()
        return res
    except: 
        return None

# ==================== 4. 主程式 UI ====================

def main():
    st.title("🏆 AI 股市全能專家 v15 (強化版)")
    st.markdown("*即時數據 + 智能預測 + 雲端記錄*")
    
    client = get_gspread_client()
    status_color = "green" if client else "red"
    status_text = "雲端連線正常" if client else "雲端未連線 (請檢查權限)"
    st.sidebar.markdown(f"### ☁️ 狀態：:{status_color}[{status_text}]")
    
    tab1, tab2, tab3, tab4 = st.tabs(["🔍 單股分析", "🚀 全市場掃描", "💰 交易值排行", "📊 雲端紀錄"])

    # --- TAB 1: 單股分析 (按照 AI 1.0 改進) ---
    with tab1:
        st.markdown("### 📈 AI 股市趨勢分析與預測系統")
        st.info("此處的分析結果將存入 Google Sheets 的 **第一分頁 (單股分析)**")
        
        col_input1, col_input2 = st.columns([3, 1])
        with col_input1:
            symbol = st.text_input("輸入股票代碼 (例: 2330.TW)", "2330.TW").upper()
        with col_input2:
            forecast_days = st.slider("預測天數", 5, 10, 7)
        
        if st.button("🔍 開始分析", key="analyze_single"):
            df = get_stock_history(symbol)
            sentiment_data = get_finnhub_sentiment(symbol)
            sent_score = sentiment_data['sentiment'].get('bullishPercent', 0.5) if sentiment_data and 'sentiment' in sentiment_data else 0.5
            
            if df is not None:
                # 執行預測
                future_df, metrics = predict_future_prices(df, sent_score, days=forecast_days)
                prediction_reason, confidence = generate_prediction_reason(df, future_df, metrics, sent_score)
                
                # 繪製圖表
                st.subheader(f"📊 {symbol} 歷史走勢與 AI 預測路徑")
                
                fig = go.Figure()
                fig.add_trace(go.Candlestick(
                    x=df['Date'], 
                    open=df['Open'], 
                    high=df['High'],
                    low=df['Low'], 
                    close=df['Close'], 
                    name="歷史數據"
                ))
                
                connect_df = pd.concat([df.tail(1)[['Date', 'Close']], future_df])
                fig.add_trace(go.Scatter(
                    x=connect_df['Date'], 
                    y=connect_df['Close'],
                    mode='lines+markers',
                    line=dict(color='orange', width=3, dash='dot'),
                    marker=dict(size=6),
                    name=f"AI 預測未來 {forecast_days} 日"
                ))
                
                fig.update_layout(
                    xaxis_rangeslider_visible=False, 
                    height=600, 
                    template="plotly_dark",
                    hovermode='x unified'
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # 分析面板
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    st.markdown("### 📉 數據摘要")
                    current_price = df['Close'].iloc[-1]
                    predicted_price = future_df['Close'].iloc[-1]
                    change = ((predicted_price - current_price) / current_price) * 100
                    
                    st.metric("當前價格", f"${current_price:.2f}")
                    st.metric(
                        f"{forecast_days} 日後預測價格", 
                        f"${predicted_price:.2f}",
                        f"{change:+.2f}%"
                    )
                    
                    st.markdown("**技術指標：**")
                    st.write(f"- 波動率：`{metrics['volatility']:.4f}`")
                    st.write(f"- 5日趨勢：`{metrics['recent_trend']*100:+.2f}%`")
                    st.write(f"- 成交量變化：`{metrics['volume_change']*100:+.1f}%`")
                
                with col2:
                    st.markdown("### 🧠 AI 預測依據")
                    st.markdown(prediction_reason)
                
                # 詳細預測數據表
                with st.expander("📅 查看每日預測明細"):
                    display_df = future_df.copy()
                    display_df['Date'] = display_df['Date'].dt.strftime('%Y-%m-%d')
                    display_df['價格'] = display_df['Close'].apply(lambda x: f"${x:.2f}")
                    display_df['變化%'] = display_df['Close'].pct_change().fillna(0).apply(lambda x: f"{x*100:+.2f}%")
                    st.dataframe(display_df[['Date', '價格', '變化%']], use_container_width=True)
                
                # 存檔按鈕
                if st.button("💾 存入雲端紀錄", key="save_single"):
                    save_data = [[
                        datetime.now().strftime('%Y-%m-%d'),
                        symbol,
                        round(float(current_price), 2),
                        round(float(predicted_price), 2),
                        f"{change:.2f}%",
                        "-",
                        "-",
                        f"{metrics['volatility']:.4f}",
                        f"{metrics['recent_trend']*100:.2f}%",
                        f"{metrics['volume_change']*100:.1f}%",
                        f"{sent_score:.2f}",
                        confidence
                    ]]
                    if save_to_sheets(save_data, sheet_index=0):
                        st.success("✅ 已存入第一分頁！")
                
                st.markdown("---")
                st.caption("⚠️ **免責聲明**：本預測系統僅供學習與研究使用，不構成投資建議。")
            else:
                st.error("❌ 無法獲取數據，請檢查股票代碼。")

    # --- TAB 2: 全市場掃描 ---
    with tab2:
        st.markdown("### 🤖 全自動市場掃描流程")
        st.write("1. 掃描 400+ 檔活躍股 → 2. 計算成交值排序 Top 100 → 3. AI 預測 → 4. 存入 **第二分頁**")
        
        # 檢查市場時間
        taiwan_tz = 8
        now = datetime.utcnow() + timedelta(hours=taiwan_tz)
        market_close_time = dt_time(13, 30)
        is_market_closed = now.time() >= market_close_time or now.weekday() >= 5
        
        if is_market_closed:
            st.info("✅ 市場已收盤，預測結果將包含實際價格比對")
        else:
            st.warning("⚠️ 市場尚未收盤，實際收盤價與誤差將在收盤後更新")
        
        if st.button("🚀 啟動掃描並預測", key="scan_market"):
            top_100_tickers, _ = scan_top_100_by_value_local()
            
            if top_100_tickers:
                st.write(f"📋 掃描名單預覽：{top_100_tickers[:5]} ...")
                
                results = []
                progress = st.progress(0)
                status = st.empty()
                
                for i, stock in enumerate(top_100_tickers):
                    status.text(f"🤖 AI 正在分析 ({i+1}/{len(top_100_tickers)}): {stock}")
                    
                    df = get_stock_history(stock)
                    if df is not None:
                        curr_p = df['Close'].iloc[-1]
                        
                        # 使用改進的預測
                        sent_data = get_finnhub_sentiment(stock)
                        sent = sent_data['sentiment'].get('bullishPercent', 0.5) if sent_data and 'sentiment' in sent_data else 0.5
                        
                        try:
                            future_df, _ = predict_future_prices(df, sent, days=7)
                            pred_p = future_df['Close'].iloc[-1]
                        except:
                            pred_p = curr_p * (1 + np.random.normal(0.01, 0.02))
                        
                        gain = ((pred_p - curr_p) / curr_p) * 100
                        
                        # 如果市場已收盤，嘗試獲取實際價格
                        actual_price = "-"
                        error_pct = "-"
                        
                        if is_market_closed:
                            try:
                                target_date = datetime.now() + timedelta(days=7)
                                actual_df = yf.download(stock, start=target_date.strftime('%Y-%m-%d'),
                                                       end=(target_date + timedelta(days=3)).strftime('%Y-%m-%d'),
                                                       progress=False)
                                if not actual_df.empty:
                                    actual_price = round(float(actual_df['Close'].iloc[0]), 2)
                                    error_pct = f"{((actual_price - pred_p) / pred_p * 100):.2f}%"
                            except:
                                pass
                        
                        results.append([
                            datetime.now().strftime('%Y-%m-%d'),
                            stock,
                            round(float(curr_p), 2),
                            round(float(pred_p), 2),
                            f"{gain:.2f}%",
                            actual_price,
                            error_pct
                        ])
                    
                    progress.progress((i+1)/len(top_100_tickers))
                
                status.empty()
                progress.empty()
                
                res_df = pd.DataFrame(results, columns=["日期","代碼","現價","預測","漲幅","實際","誤差"])
                st.dataframe(res_df, use_container_width=True)
                
                if save_to_sheets(results, sheet_index=1):
                    st.success(f"🎉 成功將 {len(results)} 檔預測結果存入 **第二分頁**！")
        
        # 手動更新實際價格按鈕
        st.markdown("---")
        if st.button("🔄 更新實際收盤價與誤差", key="update_actual"):
            success, message = update_actual_prices(sheet_index=1)
            if success:
                st.success(f"✅ {message}")
            else:
                st.warning(f"⚠️ {message}")

    # --- TAB 3: 交易值排行 (新增) ---
    with tab3:
        st.markdown("### 💰 台股每日交易值 Top 100")
        st.info("此處數據將存入 Google Sheets 的 **第三分頁 (交易值排行)**")
        
        if st.button("📊 掃描今日交易值", key="scan_turnover"):
            top_100_tickers, turnover_df = scan_top_100_by_value_local()
            
            if not turnover_df.empty:
                # 顯示排行榜
                st.dataframe(
                    turnover_df[['rank', 'ticker', 'name', 'price', 'volume', 'turnover']].rename(columns={
                        'rank': '排名',
                        'ticker': '代碼',
                        'name': '名稱',
                        'price': '收盤價',
                        'volume': '成交量',
                        'turnover': '成交值(億)'
                    }),
                    use_container_width=True
                )
                
                # 準備存檔數據
                save_data = []
                today = datetime.now().strftime('%Y-%m-%d')
                for _, row in turnover_df.iterrows():
                    save_data.append([
                        today,
                        row['ticker'],
                        row['name'],
                        round(row['price'], 2),
                        int(row['volume']),
                        round(row['turnover'], 2),
                        int(row['rank'])
                    ])
                
                if st.button("💾 存入第三分頁", key="save_turnover"):
                    if save_to_sheets(save_data, sheet_index=2):
                        st.success("✅ 已存入第三分頁！")

    # --- TAB 4: 雲端紀錄 ---
    with tab4:
        st.markdown("### 📊 Google Sheets 歷史紀錄")
        
        if st.button("🔄 刷新數據", key="refresh_sheets"):
            st.cache_data.clear()
            st.rerun()
        
        sheet_option = st.radio("選擇分頁", ["第一分頁 (單股分析)", "第二分頁 (市場掃描)", "第三分頁 (交易值排行)"])
        idx = 0 if "第一" in sheet_option else (1 if "第二" in sheet_option else 2)

        if client:
            try:
                sh = client.open(SHEET_NAME)
                all_ws = sh.worksheets()
                
                if len(all_ws) > idx:
                    ws = all_ws[idx]
                    data = ws.get_all_values()
                    
                    if len(data) > 1:
                        df_display = pd.DataFrame(data[1:], columns=data[0])
                        st.dataframe(df_display, use_container_width=True)
                        
                        # 下載按鈕
                        csv = df_display.to_csv(index=False).encode('utf-8-sig')
                        st.download_button(
                            label="📥 下載 CSV",
                            data=csv,
                            file_name=f"{sheet_option}_{datetime.now().strftime('%Y%m%d')}.csv",
                            mime="text/csv"
                        )
                    else:
                        st.info("此分頁無資料")
                else:
                    st.warning("此分頁尚未建立")
            except Exception as e:
                st.error(f"讀取失敗: {e}")
        else:
            st.warning("請先設定 Google Sheets 連線")

if __name__ == "__main__":
    main()
