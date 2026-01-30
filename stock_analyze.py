import time
import random

# --- 優化後的執行流程 ---
if st.button("🚀 執行 Top 100 預測任務"):
    tickers = get_target_tickers()
    client = get_gspread_client()
    
    if client and tickers:
        sh = client.open(SHEET_NAME)
        ws = sh.get_worksheet(0)
        p_bar = st.progress(0)
        
        # 1. 改用「批量下載」歷史數據，減少請求次數
        st.info("正在批量獲取市場歷史數據...")
        all_data = yf.download(tickers, period="2mo", group_by='ticker', threads=True, progress=False)
        
        for idx, t in enumerate(tickers):
            try:
                # 2. 從批量數據中提取，避免重複請求
                if isinstance(all_data.columns, pd.MultiIndex):
                    df = all_data[t].dropna()
                else:
                    df = all_data.dropna()
                
                if df.empty: 
                    continue
                    
                current_p = round(float(df['Close'].iloc[-1]), 2)
                
                # 3. 執行分析與新聞爬蟲 (加入隨機延遲預防封鎖)
                tech_fund_score = get_analysis_score(t, df)
                news_txt = fetch_news_text(t)
                
                # AI 預測
                pred_prices = ai_predict_logic(t, current_p, tech_fund_score, news_txt)
                
                # 4. 寫入 Excel
                update_values = pred_prices + ["-"]
                ws.update(f"E{idx+2}:J{idx+2}", [update_values])
                
                st.write(f"✅ {t} 分析完成")
                
                # --- 關鍵修正：智能冷卻機制 ---
                # 每支股票間隔 1~3 秒隨機休息
                time.sleep(random.uniform(1.0, 3.0)) 
                
                # 每 10 支股票進行一次長時間大休息 (30秒)，重置伺服器計數
                if (idx + 1) % 10 == 0:
                    st.warning(f"已完成 {idx+1} 檔，冷卻中避免被封鎖...")
                    time.sleep(20) 
                    
            except Exception as e:
                if "Too Many Requests" in str(e):
                    st.error("🚨 偵測到頻繁請求封鎖！強制休息 60 秒...")
                    time.sleep(60) # 遇到封鎖立即長休
                else:
                    st.warning(f"跳過 {t}: {e}")
                
            p_bar.progress((idx + 1) / len(tickers))
            
        st.success("🎉 全部 Top 100 標的預測更新完成！")
