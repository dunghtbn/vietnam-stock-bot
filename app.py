import streamlit as st
import pandas as pd
import ta
import plotly.graph_objects as go
from vnstock import Vnstock  # Cú pháp import mới của v3
from datetime import date, timedelta, datetime, timezone
import google.generativeai as genai

# --- 1. CẤU HÌNH TRANG ---
st.set_page_config(
    page_title="Pro Stock Analyst AI 2.0",
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

# --- 2. HÀM XỬ LÝ DỮ LIỆU ---

@st.cache_data(ttl=300)
def load_data(symbol, timeframe):
    """Lấy dữ liệu giá cổ phiếu"""
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
    """Lấy dữ liệu VN-Index để đo sức mạnh RS"""
    end_date = date.today().strftime("%Y-%m-%d")
    start_date = (date.today() - timedelta(days=100)).strftime("%Y-%m-%d") # Lấy dư ra để đủ 20 phiên
    
    try:
        resolution = '1W' if timeframe == 'Tuần' else '1D'
        index = Vnstock().stock(symbol='VNINDEX', source='TCBS')
        df_index = index.quote.history(start=start_date, end=end_date, interval=resolution)
        
        if df_index is not None and not df_index.empty:
            df_index['time'] = pd.to_datetime(df_index['time'])
            df_index.set_index('time', inplace=True)
            df_index.rename(columns={'close': 'Close'}, inplace=True)
            df_index['Close'] = pd.to_numeric(df_index['Close'], errors='coerce')
            return df_index.dropna()
    except Exception:
        return None

@st.cache_data(ttl=86400) # Cache 1 ngày
def load_fundamental_data(symbol):
    """Lấy dữ liệu Cơ bản (P/E, P/B, ROE)"""
    try:
        stock = Vnstock().stock(symbol=symbol, source='TCBS') # VCI hoặc TCBS thường trả FA tốt
        df_overview = stock.company.overview()
        
        if df_overview is not None and not df_overview.empty:
            # Chuyển tên cột về chữ thường để tránh lỗi in hoa/thường
            df_overview.columns = [c.lower() for c in df_overview.columns]
            
            pe = df_overview['pe'].iloc[0] if 'pe' in df_overview.columns else 'N/A'
            pb = df_overview['pb'].iloc[0] if 'pb' in df_overview.columns else 'N/A'
            roe = df_overview['roe'].iloc[0] * 100 if 'roe' in df_overview.columns else 'N/A'
            
            # Format số
            pe = f"{pe:.2f}" if isinstance(pe, (int, float)) else pe
            pb = f"{pb:.2f}" if isinstance(pb, (int, float)) else pb
            roe = f"{roe:.2f}" if isinstance(roe, (int, float)) else roe
            
            return {'pe': pe, 'pb': pb, 'roe': roe}
    except Exception:
        pass
    return {'pe': 'N/A', 'pb': 'N/A', 'roe': 'N/A'}

def calculate_indicators(df):
    """Tính toán các chỉ báo kỹ thuật bằng thư viện 'ta'"""
    df['RSI'] = ta.momentum.RSIIndicator(close=df['Close'], window=14).rsi()
    df['MA20'] = ta.trend.SMAIndicator(close=df['Close'], window=20).sma_indicator()
    df['MA50'] = ta.trend.SMAIndicator(close=df['Close'], window=50).sma_indicator()
    
    indicator_bb = ta.volatility.BollingerBands(close=df['Close'], window=20, window_dev=2)
    df['BB_Upper'] = indicator_bb.bollinger_hband()
    df['BB_Lower'] = indicator_bb.bollinger_lband()
    return df

# --- 3. HÀM VẼ BIỂU ĐỒ (PLOTLY) ---
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

# --- 4. HÀM GỌI AI (NÂNG CẤP V2.0) ---
def get_ai_analysis(api_key, symbol, current_price, rsi, ma20, status_ma20, bb_status, avg_vol, vol_today, stock_perf, vnindex_perf, rs_status, pe, pb, roe):
    if not api_key: return "⚠️ Vui lòng nhập API Key để xem phân tích."
    
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-2.5-flash')
    
    prompt = f"""Role: Bạn là Chuyên gia Phân tích Chứng khoán Top 1 tại Việt Nam. Bạn kết hợp xuất sắc cả Phân tích Kỹ thuật (TA) và Phân tích Cơ bản (FA) để ra quyết định. Không đoán mò.

Task: Phân tích mã {symbol} dựa trên bộ dữ liệu toàn diện sau:

1. DỮ LIỆU KỸ THUẬT (TA) & DÒNG TIỀN:
- Giá hiện tại: {current_price:,.2f}
- RSI(14): {rsi:.2f}
- MA20: {ma20:,.2f} (Giá đang {status_ma20} đường MA20)
- Bollinger Bands: {bb_status}
- Volume: TB 10 phiên là {avg_vol:,.0f}, Hôm nay là {vol_today:,.0f}
- Sức mạnh giá (20 phiên): Mã {symbol} thay đổi {stock_perf:.2f}%, trong khi VN-Index thay đổi {vnindex_perf:.2f}% -> Cổ phiếu này đang {rs_status} thị trường chung.

2. DỮ LIỆU CƠ BẢN (FA) & ĐỊNH GIÁ:
- P/E: {pe}
- P/B: {pb}
- ROE: {roe}%

Yêu cầu output format:
📊 **TỔNG QUAN:** [Đánh giá xu hướng kỹ thuật ngắn hạn + Định giá cơ bản đắt/rẻ + Sức mạnh RS]
🎯 **KHUYẾN NGHỊ:** [MUA MẠNH / MUA THĂM DÒ / BÁN / QUAN SÁT]
1. Vùng mua an toàn: [Giá A - Giá B]
2. Mục tiêu chốt lời (Target): [Giá C] (Ngắn hạn)
3. Điểm cắt lỗ (Stoploss): [Giá D] (Bắt buộc phải có)
💡 **LÝ DO:** [Giải thích sắc bén: Tại sao chọn hành động này dựa trên sự hội tụ giữa đồ thị TA và nền tảng FA]
"""
    try:
        with st.spinner('🤖 AI đang phân tích dữ liệu Hybrid (FA + TA)...'):
            response = model.generate_content(prompt)
            return response.text
    except Exception as e:
        return f"Lỗi AI: {e}"

# --- 5. GIAO DIỆN CHÍNH (MAIN) ---
def main():
    vn_tz = timezone(timedelta(hours=7))
    current_time = datetime.now(vn_tz).strftime("%H:%M:%S - %d/%m/%Y")
    
    st.markdown(f"<h2 style='text-align: center; color: #1E88E5;'>🚀 Bot Phân Tích Chứng Khoán Hybrid AI 2.0</h2>", unsafe_allow_html=True)
    st.markdown("---")
    
    with st.sidebar:
        st.title("🎛️ Control Panel")
        try: api_key = st.secrets["GEMINI_API_KEY"]
        except KeyError: api_key = ""
            
        symbol = st.text_input("Mã Cổ Phiếu", value="FPT").upper()
        timeframe = st.selectbox("Khung thời gian", ["Ngày", "Tuần"])
        st.info("💡 Mẹo: Chọn 'Tuần' để xem xu hướng dài hạn.")
        st.success("✨ V2.0: Đã tích hợp Phân tích Cơ bản & Sức mạnh RS")
    
    if symbol:
        df = load_data(symbol, timeframe)
        df_vnindex = load_vnindex_data(timeframe)
        fa_data = load_fundamental_data(symbol)
        
        if df is not None:
            df = calculate_indicators(df)
            last_row = df.iloc[-1]
            prev_row = df.iloc[-2]
            
            change = last_row['Close'] - prev_row['Close']
            pct_change = (change / prev_row['Close']) * 100
            
            # Tính hiệu suất 20 phiên (Sức mạnh RS)
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
            
            # Hiển thị Metrics Header
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Giá đóng cửa", f"{last_row['Close']:,.2f}", f"{pct_change:.2f}%")
            m2.metric("Khối lượng", f"{last_row['Volume']:,.0f}")
            m3.metric("RSI (14)", f"{last_row['RSI']:.1f}")
            m4.metric("MA20 Trend", "Tăng" if last_row['Close'] > last_row['MA20'] else "Giảm")
            
            st.divider()

            col_left, col_right = st.columns([2, 1])
            
            with col_left:
                # Banner thông báo sức mạnh giá
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
                
                st.subheader("🏢 Chỉ số Cơ bản (FA)")
                fa_df = {
                    "Chỉ số": ["P/E", "P/B", "ROE (%)"],
                    "Giá trị": [fa_data['pe'], fa_data['pb'], fa_data['roe']]
                }
                st.table(pd.DataFrame(fa_df))
                
                status_ma20 = "nằm trên" if last_row['Close'] > last_row['MA20'] else "nằm dưới"
                if last_row['Close'] >= last_row['BB_Upper']: bb_status = "Chạm/Vượt Band trên (Quá mua)"
                elif last_row['Close'] <= last_row['BB_Lower']: bb_status = "Chạm/Thủng Band dưới (Quá bán)"
                else: bb_status = "Dao động bình thường"
                
                avg_vol = df['Volume'].tail(10).mean()
                vol_today = last_row['Volume']
                
                st.subheader("🤖 AI Khuyến Nghị V2.0")
                
                # --- KHỞI TẠO BỘ NHỚ TẠM (SESSION STATE) ---
                if 'ai_analysis' not in st.session_state:
                    st.session_state.ai_analysis = ""
                if 'analyzed_symbol' not in st.session_state:
                    st.session_state.analyzed_symbol = ""

                if api_key:
                    if st.button("Phân tích chuyên sâu", use_container_width=True):
                        # Gọi AI và lưu kết quả vào bộ nhớ tạm
                        analysis = get_ai_analysis(
                            api_key=api_key, symbol=symbol, current_price=last_row['Close'], 
                            rsi=last_row['RSI'], ma20=last_row['MA20'], status_ma20=status_ma20, 
                            bb_status=bb_status, avg_vol=avg_vol, vol_today=vol_today,
                            stock_perf=stock_perf, vnindex_perf=vnindex_perf, rs_status=rs_status.split()[0],
                            pe=fa_data['pe'], pb=fa_data['pb'], roe=fa_data['roe']
                        )
                        st.session_state.ai_analysis = analysis
                        st.session_state.analyzed_symbol = symbol

                    # --- HIỂN THỊ KẾT QUẢ VÀ NÚT TẢI BÁO CÁO ---
                    if st.session_state.ai_analysis and st.session_state.analyzed_symbol == symbol:
                        st.markdown(st.session_state.ai_analysis)
                        
                        st.divider()
                        
                        # Tạo nội dung file báo cáo
                        report_content = f"BÁO CÁO PHÂN TÍCH MÃ {symbol}\n"
                        report_content += f"Ngày phân tích: {current_time}\n"
                        report_content += "-"*40 + "\n"
                        report_content += f"[Thông số Kỹ thuật] Giá: {last_row['Close']:,.2f} | RSI: {last_row['RSI']:.2f} | Khối lượng: {vol_today:,.0f}\n"
                        report_content += f"[Thông số Cơ bản] P/E: {fa_data['pe']} | P/B: {fa_data['pb']} | ROE: {fa_data['roe']}%\n"
                        report_content += f"[Sức mạnh Giá] {symbol} thay đổi {stock_perf:.2f}% vs VN-Index {vnindex_perf:.2f}%\n"
                        report_content += "-"*40 + "\n\n"
                        report_content += st.session_state.ai_analysis
                        
                        # Hiển thị nút Tải xuống
                        st.download_button(
                            label="📥 Tải Báo Cáo Nhận Định (TXT)",
                            data=report_content,
                            file_name=f"Bao_cao_AI_{symbol}_{date.today()}.txt",
                            mime="text/plain",
                            use_container_width=True
                        )
                else:
                    st.warning("Hệ thống chưa thiết lập API Key. Vui lòng kiểm tra lại cài đặt Secrets trên Streamlit.")
            
            st.markdown("---")
            st.caption(f"🕒 *Dữ liệu được cập nhật lần cuối vào lúc: **{current_time}** (Múi giờ Việt Nam)*")
            st.markdown("<h5 style='text-align: center; color: #1E88E5;'>Thiết kế và Lập trình bởi: Hoàng Trung Dũng - Emai: dung@hdbn.vip</h5>", unsafe_allow_html=True)
        else:
            st.error(f"Không tìm thấy dữ liệu cho mã {symbol}")

if __name__ == "__main__":
    main()
