import streamlit as st
import pandas as pd
import ta
import plotly.graph_objects as go
from vnstock import Vnstock
from datetime import date, timedelta, datetime, timezone
import google.generativeai as genai
import requests
import yfinance as yf

# --- 1. CẤU HÌNH TRANG ---
st.set_page_config(
    page_title="Pro Stock Analyst AI 4.0",
    layout="wide",
    page_icon="🚀",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .stMetric {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 5px;
    }
    [data-testid="stMetricValue"] {
        font-size: 20px;
    }
</style>
""", unsafe_allow_html=True)

# --- 2. CÁC HÀM XỬ LÝ DỮ LIỆU ---

@st.cache_data(ttl=300)
def load_data(symbol, timeframe):
    end_date = date.today().strftime("%Y-%m-%d")
    start_date = (date.today() - timedelta(days=730)).strftime("%Y-%m-%d") 
    
    try:
        resolution = '1W' if timeframe == 'Tuần' else '1D'
        stock = Vnstock().stock(symbol=symbol, source='KBS')
        df = stock.quote.history(start=start_date, end=end_date, interval=resolution)
        
        if df is None or df.empty: return None
            
        df['time'] = pd.to_datetime(df['time'])
        df.set_index('time', inplace=True)
        
        mapping = {'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'}
        df.rename(columns=mapping, inplace=True)
        
        for col in mapping.values():
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
        return df.dropna()
    except Exception as e:
        st.error(f"Lỗi API dữ liệu {symbol}: {e}")
        return None

@st.cache_data(ttl=300)
def load_vnindex_data(timeframe):
    """Kéo dữ liệu Ngày (1D) từ KBS và tự động gom thành nến Tuần (1W) bằng Pandas"""
    end_date = date.today().strftime("%Y-%m-%d")
    start_date = (date.today() - timedelta(days=730)).strftime("%Y-%m-%d") 
    
    try:
        stock = Vnstock().stock(symbol='VNINDEX', source='KBS')
        df_index = stock.quote.history(start=start_date, end=end_date, interval='1D')
        
        if df_index is not None and not df_index.empty:
            df_index['time'] = pd.to_datetime(df_index['time'])
            df_index.set_index('time', inplace=True)
            
            mapping = {'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'}
            df_index.rename(columns=mapping, inplace=True)
            df_index['Close'] = pd.to_numeric(df_index['Close'], errors='coerce')
            df_index = df_index.dropna()
            
            if timeframe == 'Tuần':
                df_index = df_index.resample('W').agg({
                    'Open': 'first',   
                    'High': 'max',     
                    'Low': 'min',      
                    'Close': 'last',   
                    'Volume': 'sum'    
                }).dropna()
                
            return df_index
    except Exception as e:
        pass
        
    return None

@st.cache_data(ttl=86400) 
def load_fundamental_data(symbol):
    """Sử dụng 100% Yahoo Finance: Lấy P/E, P/B, ROE, Vốn hóa, Cổ tức, Nợ/Vốn"""
    try:
        ticker = yf.Ticker(f"{symbol}.VN")
        info = ticker.info
        
        pe = info.get('trailingPE')
        pb = info.get('priceToBook')
        roe = info.get('returnOnEquity')
        market_cap = info.get('marketCap')
        div_yield = info.get('dividendYield')
        debt_to_equity = info.get('debtToEquity')
        
        pe_str = f"{float(pe):.2f}" if pe is not None else "N/A"
        pb_str = f"{float(pb):.2f}" if pb is not None else "N/A"
        roe_str = f"{float(roe) * 100:.2f}" if roe is not None else "N/A"
        market_cap_str = f"{market_cap / 1e9:,.0f} Tỷ" if market_cap is not None else "N/A"
        
        if div_yield is not None:
            dy_val = float(div_yield)
            if dy_val > 1: div_yield_str = f"{dy_val:.2f}%"
            else: div_yield_str = f"{dy_val * 100:.2f}%"
        else:
            div_yield_str = "0.00%"
        
        debt_to_equity_str = f"{float(debt_to_equity):.2f}%" if debt_to_equity is not None else "N/A"
        
        return {
            'pe': pe_str, 'pb': pb_str, 'roe': roe_str,
            'market_cap': market_cap_str, 'div_yield': div_yield_str, 'debt_to_equity': debt_to_equity_str
        }
        
    except Exception as e:
        return {
            'pe': 'N/A', 'pb': 'N/A', 'roe': 'N/A', 
            'market_cap': 'N/A', 'div_yield': 'N/A', 'debt_to_equity': 'N/A'
        }

def calculate_indicators(df):
    df['RSI'] = ta.momentum.RSIIndicator(close=df['Close'], window=14).rsi()
    df['MA20'] = ta.trend.SMAIndicator(close=df['Close'], window=20).sma_indicator()
    df['MA50'] = ta.trend.SMAIndicator(close=df['Close'], window=50).sma_indicator()
    
    indicator_bb = ta.volatility.BollingerBands(close=df['Close'], window=20, window_dev=2)
    df['BB_Upper'] = indicator_bb.bollinger_hband()
    df['BB_Lower'] = indicator_bb.bollinger_lband()
    return df

# --- 3. HÀM VẼ BIỂU ĐỒ ---
def plot_chart(df, symbol):
    plot_df = df.tail(150)
    fig = go.Figure()

    fig.add_trace(go.Candlestick(
        x=plot_df.index,
        open=plot_df['Open'], high=plot_df['High'],
        low=plot_df['Low'], close=plot_df['Close'],
        name='Giá', increasing_line_color='#26a69a', decreasing_line_color='#ef5350'
    ))
    fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['MA20'], line=dict(color='yellow', width=1.5), name='MA20'))
    fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['MA50'], line=dict(color='purple', width=1.5), name='MA50'))

    fig.update_layout(
        title=f"Biểu đồ kỹ thuật {symbol}", yaxis_title='Giá (VND)', xaxis_rangeslider_visible=False,
        height=600, margin=dict(l=20, r=20, t=40, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig

# --- 4. HÀM GỌI AI PHÂN TÍCH ---
def get_ai_analysis(api_key, symbol, current_price, rsi, ma20, status_ma20, bb_status, avg_vol, vol_today, stock_perf, vnindex_perf, rs_status, pe, pb, roe, market_cap, div_yield, debt_to_equity):
    if not api_key: return "⚠️ Vui lòng nhập API Key để xem phân tích."
    
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-2.5-flash')
    
    prompt = f"""Role: Bạn là Chuyên gia Phân tích Chứng khoán Top 1 tại Việt Nam. Bạn kết hợp xuất sắc cả Phân tích Kỹ thuật (TA) và Phân tích Cơ bản (FA) để ra quyết định an toàn nhất. Không đoán mò.

Task: Phân tích mã {symbol} dựa trên bộ dữ liệu toàn diện sau:

1. DỮ LIỆU KỸ THUẬT (TA) & DÒNG TIỀN:
- Giá hiện tại: {current_price:,.2f}
- RSI(14): {rsi:.2f}
- MA20: {ma20:,.2f} (Giá đang {status_ma20} đường MA20)
- Bollinger Bands: {bb_status}
- Volume: TB 10 phiên là {avg_vol:,.0f}, Hôm nay là {vol_today:,.0f}
- Sức mạnh giá (20 phiên): Mã {symbol} thay đổi {stock_perf:.2f}%, trong khi VN-Index thay đổi {vnindex_perf:.2f}% -> Cổ phiếu này đang {rs_status} thị trường chung.

2. DỮ LIỆU CƠ BẢN (FA) & ĐỘ AN TOÀN:
- Quy mô Vốn hóa: {market_cap}
- Định giá: P/E: {pe} | P/B: {pb}
- Hiệu quả sinh lời: ROE: {roe}%
- Tỷ suất cổ tức (Bảo vệ rủi ro): {div_yield}
- Tỷ lệ Nợ/Vốn CSH (Rủi ro phá sản): {debt_to_equity}

Yêu cầu output format:
📊 **TỔNG QUAN:** [Đánh giá xu hướng kỹ thuật + Sức khỏe tài chính dựa trên Cổ tức & Tỷ lệ Nợ + Định giá đắt/rẻ]
🎯 **KHUYẾN NGHỊ:** [MUA MẠNH / MUA THĂM DÒ / BÁN / QUAN SÁT]
1. Vùng mua an toàn: [Giá A - Giá B]
2. Mục tiêu chốt lời (Target): [Giá C] (Ngắn hạn)
3. Điểm cắt lỗ (Stoploss): [Giá D] (Bắt buộc)
💡 **LÝ DO:** [Giải thích sắc bén sự hội tụ giữa đồ thị và sức khỏe tài chính doanh nghiệp]
"""
    try:
        with st.spinner('🤖 AI đang đánh giá Toàn diện rủi ro (FA + TA)...'):
            response = model.generate_content(prompt)
            return response.text
    except Exception as e:
        return f"Lỗi AI: {e}"

# --- 5. GIAO DIỆN CHÍNH (MAIN) ---
def main():
    vn_tz = timezone(timedelta(hours=7))
    current_time = datetime.now(vn_tz).strftime("%H:%M:%S - %d/%m/%Y")
    
    st.markdown(f"<h2 style='text-align: center; color: #1E88E5;'>🚀 Bot Phân Tích Chứng Khoán Hybrid AI 4.0</h2>", unsafe_allow_html=True)
    st.markdown("---")
    
    # ----------------------------------------------------
    # BƯỚC QUAN TRỌNG: ĐƯA SIDEBAR RA NGOÀI VÀ LÊN TRÊN CÙNG
    # ĐỂ APP LẤY ĐƯỢC MÃ CỔ PHIẾU VÀ KHUNG THỜI GIAN TRƯỚC
    # ----------------------------------------------------
    with st.sidebar:
        st.title("🎛️ Control Panel")
        try: api_key = st.secrets["GEMINI_API_KEY"]
        except KeyError: api_key = ""
            
        symbol = st.text_input("Mã Cổ Phiếu", value="DBC").upper()
        timeframe = st.selectbox("Khung thời gian", ["Ngày", "Tuần"])
        st.info("💡 Mẹo: Chọn 'Tuần' để xem xu hướng dài hạn.")
        st.success("✨ V4.0: Cập nhật Tab Radar quét Siêu Cổ Phiếu")

    # --- KHỞI TẠO 2 TAB GIAO DIỆN ---
    tab1, tab2 = st.tabs(["📊 Phân Tích Chuyên Sâu", "🎯 Radar Quét Cổ Phiếu"])
    
    # --- GIAO DIỆN TAB 1 ---
    with tab1:
        if symbol:
            df = load_data(symbol, timeframe)
            df_vnindex = load_vnindex_data(timeframe)
            fa_data = load_fundamental_data(symbol)
            
            if df is not None and not df.empty:
                df = calculate_indicators(df)
                last_row = df.iloc[-1]
                prev_row = df.iloc[-2]
                
                change = last_row['Close'] - prev_row['Close']
                pct_change = (change / prev_row['Close']) * 100
                
                lookback = 20 if len(df) >= 20 else len(df) - 1
                if lookback > 0:
                    stock_perf = ((last_row['Close'] - df['Close'].iloc[-1 - lookback]) / df['Close'].iloc[-1 - lookback]) * 100
                else:
                    stock_perf = 0.0

                if df_vnindex is not None and len(df_vnindex) > lookback:
                    vnindex_perf = ((df_vnindex['Close'].iloc[-1] - df_vnindex['Close'].iloc[-1 - lookback]) / df_vnindex['Close'].iloc[-1 - lookback]) * 100
                else:
                    vnindex_perf = 0.0
                    
                if stock_perf > vnindex_perf: rs_status = "KHỎE HƠN 💪"
                elif stock_perf < vnindex_perf: rs_status = "YẾU HƠN ⚠️"
                else: rs_status = "TƯƠNG ĐƯƠNG ⚖️"
                
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Giá đóng cửa", f"{last_row['Close']:,.2f}", f"{pct_change:.2f}%")
                m2.metric("Khối lượng", f"{last_row['Volume']:,.0f}")
                m3.metric("RSI (14)", f"{last_row['RSI']:.1f}")
                m4.metric("MA20 Trend", "Tăng" if last_row['Close'] > last_row['MA20'] else "Giảm")
                
                st.divider()

                col_left, col_right = st.columns([2, 1])
                
                with col_left:
                    st.info(f"📈 **Đo lường RS (20 phiên):** Mã **{symbol}** thay đổi **{stock_perf:.2f}%** | VN-Index thay đổi **{vnindex_perf:.2f}%** ➔ Cổ phiếu đang **{rs_status}**")
                    
                    st.subheader(f"📊 Biểu đồ {symbol} ({timeframe})")
                    fig = plot_chart(df, symbol)
                    st.plotly_chart(fig, use_container_width=True)

                with col_right:
                    st.subheader("📋 Chỉ số Kỹ thuật (TA)")
                    tech_data = {
                        "Chỉ số": ["MA20", "MA50", "BB Upper", "BB Lower"],
                        "Giá trị": [f"{last_row['MA20']:,.2f}", f"{last_row['MA50']:,.2f}", f"{last_row['BB_Upper']:,.2f}", f"{last_row['BB_Lower']:,.2f}"]
                    }
                    st.table(pd.DataFrame(tech_data))
                    
                    st.subheader("🏢 Chỉ số Cơ bản & Sức khỏe")
                    fa_df = {
                        "Chỉ số": ["Vốn hóa thị trường", "Tỷ lệ Nợ / Vốn CSH", "Tỷ suất Cổ tức", "P/E", "P/B", "ROE (%)"],
                        "Giá trị": [fa_data['market_cap'], fa_data['debt_to_equity'], fa_data['div_yield'], fa_data['pe'], fa_data['pb'], fa_data['roe']]
                    }
                    st.table(pd.DataFrame(fa_df))
                    
                    status_ma20 = "nằm trên" if last_row['Close'] > last_row['MA20'] else "nằm dưới"
                    if last_row['Close'] >= last_row['BB_Upper']: bb_status = "Chạm/Vượt Band trên (Quá mua)"
                    elif last_row['Close'] <= last_row['BB_Lower']: bb_status = "Chạm/Thủng Band dưới (Quá bán)"
                    else: bb_status = "Dao động bình thường"
                    
                    avg_vol = df['Volume'].tail(10).mean()
                    vol_today = last_row['Volume']
                    
                    st.subheader("🤖 AI Khuyến Nghị V4.0")
                    
                    if 'ai_analysis' not in st.session_state:
                        st.session_state.ai_analysis = ""
                    if 'analyzed_symbol' not in st.session_state:
                        st.session_state.analyzed_symbol = ""

                    if api_key:
                        if st.button("Phân tích chuyên sâu", use_container_width=True):
                            analysis = get_ai_analysis(
                                api_key=api_key, symbol=symbol, current_price=last_row['Close'], 
                                rsi=last_row['RSI'], ma20=last_row['MA20'], status_ma20=status_ma20, 
                                bb_status=bb_status, avg_vol=avg_vol, vol_today=vol_today,
                                stock_perf=stock_perf, vnindex_perf=vnindex_perf, rs_status=rs_status.split()[0],
                                pe=fa_data['pe'], pb=fa_data['pb'], roe=fa_data['roe'],
                                market_cap=fa_data['market_cap'], div_yield=fa_data['div_yield'], debt_to_equity=fa_data['debt_to_equity']
                            )
                            st.session_state.ai_analysis = analysis
                            st.session_state.analyzed_symbol = symbol

                        if st.session_state.ai_analysis and st.session_state.analyzed_symbol == symbol:
                            st.markdown(st.session_state.ai_analysis)
                            
                            st.divider()
                            
                            report_content = f"BÁO CÁO PHÂN TÍCH MÃ {symbol}\n"
                            report_content += f"Ngày phân tích: {current_time}\n"
                            report_content += "-"*40 + "\n"
                            report_content += f"[Kỹ thuật] Giá: {last_row['Close']:,.2f} | RSI: {last_row['RSI']:.2f} | Khối lượng: {vol_today:,.0f}\n"
                            report_content += f"[Cơ bản] Vốn hóa: {fa_data['market_cap']} | P/E: {fa_data['pe']} | P/B: {fa_data['pb']} | ROE: {fa_data['roe']}%\n"
                            report_content += f"[Độ An Toàn] Tỷ suất cổ tức: {fa_data['div_yield']} | Nợ/Vốn CSH: {fa_data['debt_to_equity']}\n"
                            report_content += f"[Sức mạnh Giá] {symbol} thay đổi {stock_perf:.2f}% vs VN-Index {vnindex_perf:.2f}%\n"
                            report_content += "-"*40 + "\n\n"
                            report_content += st.session_state.ai_analysis
                            
                            st.download_button(
                                label="📥 Tải Báo Cáo Nhận Định (TXT)",
                                data=report_content,
                                file_name=f"Bao_cao_AI_RiskControl_{symbol}_{date.today()}.txt",
                                mime="text/plain",
                                use_container_width=True
                            )
                    else:
                        st.warning("Hệ thống chưa thiết lập API Key. Vui lòng kiểm tra lại cài đặt Secrets trên Streamlit.")
            
                st.markdown("---")
                st.caption(f"🕒 *Dữ liệu được cập nhật lần cuối vào lúc: **{current_time}** (Múi giờ Việt Nam)*")
                st.markdown("<h5 style='text-align: center; color: #1E88E5;'>Thiết kế và Lập trình bởi: Hoàng Trung Dũng - Email: dung@hdbn.vip</h5>", unsafe_allow_html=True)
            else:
                st.error(f"Không tìm thấy dữ liệu cho mã {symbol}. Vui lòng kiểm tra lại mã cổ phiếu.")

    # --- GIAO DIỆN TAB 2 ---
    with tab2:
        st.subheader("📡 Radar Quét Tín Hiệu Kỹ Thuật & Dòng Tiền")
        st.markdown("Hệ thống sẽ tự động phân tích hàng loạt mã cổ phiếu để tìm ra các cơ hội đạt chuẩn.")
        
        default_tickers = "SSI, VND, HPG, HSG, VCB, MBB, MSB, TCB, FPT, MWG, DBC, VNM, GEX, DIG, DXG, PVD"
        tickers_input = st.text_input("Nhập danh sách mã cần quét (cách nhau bằng dấu phẩy):", default_tickers)
        
        col_f1, col_f2 = st.columns(2)
        with col_f1:
            filter_rsi = st.selectbox("🎯 Điều kiện RSI:", [
                "Không lọc", 
                "RSI < 30 (Vùng Quá bán - Bắt đáy)", 
                "RSI > 50 (Xu hướng Tích cực)", 
                "RSI > 70 (Vùng Quá mua)"
            ])
        with col_f2:
            filter_ma20 = st.selectbox("📈 Điều kiện Xu hướng (MA20):", [
                "Không lọc", 
                "Giá vừa cắt lên MA20 (Điểm mua sớm)", 
                "Giá nằm trên MA20 (Đang Uptrend)"
            ])

        if st.button("🚀 Kích Hoạt Radar", use_container_width=True):
            tickers = [x.strip().upper() for x in tickers_input.split(",") if x.strip()]
            results = []
            
            progress_text = "Radar đang quét dữ liệu thị trường. Vui lòng đợi..."
            my_bar = st.progress(0, text=progress_text)
            
            for i, t in enumerate(tickers):
                my_bar.progress((i + 1) / len(tickers), text=f"Đang phân tích tín hiệu mã {t}...")
                
                # ĐÃ SỬA: Gọi đúng tên hàm load_data thay vì load_stock_data
                df_scan = load_data(t, "Ngày") 
                
                if df_scan is not None and len(df_scan) > 2:
                    # Chạy qua hàm tính toán RSI, MA20 trước khi đánh giá
                    df_scan = calculate_indicators(df_scan)
                    
                    last_row_scan = df_scan.iloc[-1]
                    prev_row_scan = df_scan.iloc[-2]
                    
                    passed = True
                    
                    if filter_rsi == "RSI < 30 (Vùng Quá bán - Bắt đáy)" and last_row_scan['RSI'] >= 30: passed = False
                    elif filter_rsi == "RSI > 50 (Xu hướng Tích cực)" and last_row_scan['RSI'] <= 50: passed = False
                    elif filter_rsi == "RSI > 70 (Vùng Quá mua)" and last_row_scan['RSI'] <= 70: passed = False
                    
                    if filter_ma20 == "Giá vừa cắt lên MA20 (Điểm mua sớm)":
                        if not (prev_row_scan['Close'] < prev_row_scan['MA20'] and last_row_scan['Close'] > last_row_scan['MA20']):
                            passed = False
                    elif filter_ma20 == "Giá nằm trên MA20 (Đang Uptrend)":
                        if last_row_scan['Close'] < last_row_scan['MA20']:
                            passed = False
                    
                    if passed:
                        results.append({
                            "Mã CP": t,
                            "Giá Đóng Cửa": f"{last_row_scan['Close']:,.2f}",
                            "RSI": round(last_row_scan['RSI'], 2),
                            "MA20": f"{last_row_scan['MA20']:,.2f}",
                            "Khối Lượng": f"{last_row_scan['Volume']:,.0f}"
                        })
                        
            my_bar.empty() 
            
            if results:
                st.success(f"🎉 Rà soát hoàn tất! Có {len(results)} mã đang đạt chuẩn kỹ thuật của anh.")
                st.dataframe(pd.DataFrame(results), use_container_width=True, hide_index=True)
            else:
                st.warning("Khung thị trường hiện tại không có mã nào trong danh sách thỏa mãn bộ lọc này.")

if __name__ == "__main__":
    main()
