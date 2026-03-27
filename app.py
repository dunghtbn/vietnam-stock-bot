import streamlit as st
import pandas as pd
import ta
import plotly.graph_objects as go
from vnstock import Vnstock
from datetime import date, timedelta, datetime, timezone
import google.generativeai as genai
import requests
import yfinance as yf
from plotly.subplots import make_subplots

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

@st.cache_data(ttl=60)
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

# --- 3. HÀM VẼ BIỂU ĐỒ NÂNG CẤP (CÓ KHỐI LƯỢNG) ---
def plot_chart(df, symbol):
    plot_df = df.tail(150)
    
    # Lấy giá đóng cửa hiện tại (phiên gần nhất)
    current_price = df['Close'].iloc[-1]
    
    # Tạo danh sách màu cho cột Khối lượng: Xanh nếu Giá Đóng >= Giá Mở, Đỏ nếu ngược lại
    colors = ['#26a69a' if row['Close'] >= row['Open'] else '#ef5350' for index, row in plot_df.iterrows()]

    # Chia bố cục biểu đồ thành 2 tầng (80% cho Giá, 20% cho Khối lượng)
    fig = make_subplots(
        rows=2, cols=1, 
        shared_xaxes=True, 
        vertical_spacing=0.03, 
        row_heights=[0.8, 0.2]
    )

    # [TẦNG 1] Thêm Biểu đồ Nến Nhật
    fig.add_trace(go.Candlestick(
        x=plot_df.index,
        open=plot_df['Open'], high=plot_df['High'],
        low=plot_df['Low'], close=plot_df['Close'],
        name='Giá', increasing_line_color='#26a69a', decreasing_line_color='#ef5350'
    ), row=1, col=1)
    
    # Thêm các đường MA vào Tầng 1
    fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['MA20'], line=dict(color='yellow', width=1.5), name='MA20'), row=1, col=1)
    fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['MA50'], line=dict(color='purple', width=1.5), name='MA50'), row=1, col=1)

    # -------------------------------------------------------------------------
    # BỔ SUNG: THÊM ĐƯỜNG KẺ NGANG ĐỨT NÉT THỂ HIỆN GIÁ HIỆN TẠI
    # -------------------------------------------------------------------------
    fig.add_hline(
        y=current_price, 
        line_dash="dash",          
        line_color="#ff9800",      # Chỉnh lại màu cam/vàng cho giống ảnh của bạn
        line_width=1.5,
        annotation_text=f"Giá hiện tại: {current_price:,.2f}", # Đổi tên nhãn
        annotation_position="bottom left",                       # Chuyển nhãn sang lề trái để không đè nến
        annotation_font=dict(color="#ff9800", size=12),
        row=1, col=1
    )
    # -------------------------------------------------------------------------

    # [TẦNG 2] Thêm Biểu đồ Cột Khối lượng
    fig.add_trace(go.Bar(
        x=plot_df.index, 
        y=plot_df['Volume'],
        name='Khối lượng',
        marker_color=colors,
        showlegend=False
    ), row=2, col=1)

    # Căn chỉnh giao diện tổng thể
    fig.update_layout(
        title=f"Biểu đồ Kỹ thuật {symbol}",
        yaxis_title='Giá (VND)',
        yaxis2_title='Khối lượng',
        height=700, # Tăng nhẹ chiều cao tổng thể để có không gian cho tầng khối lượng
        margin=dict(l=20, r=20, t=40, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    # Tắt thanh trượt (Rangeslider) mặc định của Plotly để biểu đồ gọn gàng
    fig.update_xaxes(rangeslider_visible=False)

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

# --- 4.5. HÀM AI SO SÁNH & CHỌN LỌC SIÊU CỔ PHIẾU ---
def get_ai_best_pick(api_key, results_list):
    if not api_key: return "⚠️ Vui lòng nhập API Key để dùng tính năng này."
    
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-2.5-flash')
    
    # Gom thông tin các mã cổ phiếu đạt chuẩn thành một bản báo cáo cho AI
    stocks_info = ""
    for r in results_list:
        stocks_info += f"- Mã {r['Mã CP']}: Giá {r['Giá Đóng Cửa']}, RSI {r['RSI']}, MA20 {r['MA20']}, Khối lượng {r['Khối Lượng']}, ROE {r['ROE (%)']}%, P/E {r['P/E']}\n"
        
    prompt = f"""Role: Bạn là Giám đốc Đầu tư (CIO) quản lý quỹ phòng hộ. 
    Task: Radar của tôi vừa lọc ra được danh sách các cổ phiếu đã vượt qua cả điều kiện kỹ thuật (dòng tiền) và cơ bản (lợi nhuận) sau:
    {stocks_info}
    
    Dựa trên nền tảng kỹ thuật (RSI, Giá so với MA20) kết hợp với định giá & sinh lời (ROE, P/E), hãy chọn ra ĐÚNG 1 MÃ an toàn và có khả năng bùng nổ cao nhất để giải ngân ngay.
    
    Yêu cầu output format:
    🏆 **SIÊU CỔ PHIẾU LỰA CHỌN:** [Tên mã]
    💡 **LÝ DO CHIẾN THẮNG:** [Tại sao mã này vượt trội hơn các mã khác về cả FA lẫn TA]
    🎯 **KẾ HOẠCH HÀNH ĐỘNG:** [Khuyến nghị Vùng giá mua, Mục tiêu chốt lời, Điểm cắt lỗ tuyệt đối]
    """
    try:
        with st.spinner('🤖 Giám đốc AI đang chấm điểm định giá và dòng tiền từng mã...'):
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
        
        # --- THÊM ĐOẠN NÀY ĐỂ TẠO NÚT CẬP NHẬT REAL-TIME ---
        if st.button("🔄 Làm mới dữ liệu (Real-time)", use_container_width=True):
            st.cache_data.clear() # Xóa bộ nhớ tạm
            st.rerun()            # Tải lại ngay lập tức
        # ---------------------------------------------------
        
        st.info("💡 Mẹo: Chọn 'Tuần' để xem xu hướng dài hạn.")
        st.success("✨ V5.0: Nâng cấp Real-time & Radar Đa Lớp")

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
                m1.metric("Giá hiện tại", f"{last_row['Close']:,.2f}", f"{pct_change:.2f}%")
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
        st.subheader("📡 Radar Quét Đa Chiều (TA + FA)")
        st.markdown("Bộ lọc hội tụ: Tìm kiếm các mã đang có dòng tiền vào (TA) và nền tảng kinh doanh sinh lời tốt (FA).")
        
        default_tickers = "SSI, VND, HPG, HSG, NVL, SHS, MSB, VIX, CII, EVF, DBC, VNM, DXG, DIG, PDR, PVD"
        tickers_input = st.text_input("Nhập danh sách mã cần quét (cách nhau bằng dấu phẩy):", default_tickers)
        
        # SỬ DỤNG SESSION_STATE: Để lưu lại kết quả quét, tránh bị mất bảng khi bấm nút AI
        if 'radar_results' not in st.session_state: st.session_state.radar_results = []
        if 'has_run_radar' not in st.session_state: st.session_state.has_run_radar = False
        if 'radar_ai_pick' not in st.session_state: st.session_state.radar_ai_pick = ""
        
        # BỘ LỌC 3 CỘT
        col_f1, col_f2, col_f3 = st.columns(3)
        with col_f1:
            filter_rsi = st.selectbox("🎯 Điều kiện RSI (Kỹ thuật):", [
                "Không lọc", 
                "RSI < 30 (Vùng Quá bán - Bắt đáy)", 
                "RSI > 50 (Xu hướng Tích cực)", 
                "RSI > 70 (Vùng Quá mua)"
            ])
        with col_f2:
            filter_ma20 = st.selectbox("📈 Điều kiện MA20 (Xu hướng):", [
                "Không lọc", 
                "Giá vừa cắt lên MA20 (Điểm mua sớm)", 
                "Giá nằm trên MA20 (Đang Uptrend)"
            ])
        with col_f3:
            filter_roe = st.selectbox("🏢 Điều kiện ROE (Cơ bản):", [
                "Không lọc", 
                "ROE > 10% (Hoạt động Tốt)", 
                "ROE > 15% (Sinh lời Rất Tốt)",
                "ROE > 20% (Doanh nghiệp Xuất sắc)"
            ])

        if st.button("🚀 Kích Hoạt Radar Lọc Cổ Phiếu", use_container_width=True):
            # Xóa dữ liệu cũ mỗi lần quét mới
            st.session_state.radar_results = []
            st.session_state.radar_ai_pick = ""
            st.session_state.has_run_radar = True
            
            tickers = [x.strip().upper() for x in tickers_input.split(",") if x.strip()]
            temp_results = []
            
            my_bar = st.progress(0, text="Radar đang quét dữ liệu thị trường...")
            
            for i, t in enumerate(tickers):
                my_bar.progress((i + 1) / len(tickers), text=f"Đang phân tích tín hiệu & định giá mã {t}...")
                
                df_scan = load_data(t, "Ngày") 
                if df_scan is not None and len(df_scan) > 2:
                    df_scan = calculate_indicators(df_scan)
                    fa_data = load_fundamental_data(t) # Kéo tự động P/E, ROE
                    
                    last_row_scan = df_scan.iloc[-1]
                    prev_row_scan = df_scan.iloc[-2]
                    
                    passed = True
                    
                    # 1. Lọc Kỹ thuật (TA)
                    if filter_rsi == "RSI < 30 (Vùng Quá bán - Bắt đáy)" and last_row_scan['RSI'] >= 30: passed = False
                    elif filter_rsi == "RSI > 50 (Xu hướng Tích cực)" and last_row_scan['RSI'] <= 50: passed = False
                    elif filter_rsi == "RSI > 70 (Vùng Quá mua)" and last_row_scan['RSI'] <= 70: passed = False
                    
                    if filter_ma20 == "Giá vừa cắt lên MA20 (Điểm mua sớm)":
                        if not (prev_row_scan['Close'] < prev_row_scan['MA20'] and last_row_scan['Close'] > last_row_scan['MA20']): passed = False
                    elif filter_ma20 == "Giá nằm trên MA20 (Đang Uptrend)":
                        if last_row_scan['Close'] < last_row_scan['MA20']: passed = False
                            
                    # 2. Lọc Cơ bản (FA - ROE)
                    roe_str = fa_data['roe']
                    roe_val = 0.0
                    if roe_str != "N/A":
                        try: roe_val = float(roe_str)
                        except: pass
                    else:
                        if filter_roe != "Không lọc": passed = False # Bỏ qua mã thiếu dữ liệu ROE
                        
                    if filter_roe == "ROE > 10% (Hoạt động Tốt)" and roe_val <= 10: passed = False
                    elif filter_roe == "ROE > 15% (Sinh lời Rất Tốt)" and roe_val <= 15: passed = False
                    elif filter_roe == "ROE > 20% (Doanh nghiệp Xuất sắc)" and roe_val <= 20: passed = False
                    
                    if passed:
                        temp_results.append({
                            "Mã CP": t,
                            "Giá Đóng Cửa": f"{last_row_scan['Close']:,.2f}",
                            "RSI": round(last_row_scan['RSI'], 2),
                            "MA20": f"{last_row_scan['MA20']:,.2f}",
                            "Khối Lượng": f"{last_row_scan['Volume']:,.0f}",
                            "ROE (%)": roe_str,
                            "P/E": fa_data['pe']
                        })
                        
            my_bar.empty() 
            st.session_state.radar_results = temp_results

        # 3. HIỂN THỊ KẾT QUẢ VÀ NÚT CHỌN LỌC AI
        if st.session_state.has_run_radar:
            if st.session_state.radar_results:
                st.success(f"🎉 Rà soát hoàn tất! Có {len(st.session_state.radar_results)} mã lọt qua bộ lọc khắt khe của anh.")
                st.dataframe(pd.DataFrame(st.session_state.radar_results), use_container_width=True, hide_index=True)
                
                # --- NÚT HIỆN HỮU KHI CÓ NHIỀU HƠN 1 MÃ ---
                if len(st.session_state.radar_results) > 1:
                    st.markdown("---")
                    st.subheader("🤖 Giám Đốc Đầu Tư AI: Chọn Lọc Tinh Hoa")
                    if st.button("🏆 Nhờ AI chấm điểm & Chọn ra 1 mã an toàn nhất", use_container_width=True):
                        pick_result = get_ai_best_pick(api_key, st.session_state.radar_results)
                        st.session_state.radar_ai_pick = pick_result
                        
                if st.session_state.radar_ai_pick:
                    st.info(st.session_state.radar_ai_pick)
            else:
                st.warning("Khung thị trường hiện tại không có mã nào thỏa mãn toàn bộ các điều kiện này.")
if __name__ == "__main__":
    main()
