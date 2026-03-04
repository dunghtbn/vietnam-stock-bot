import streamlit as st
import pandas as pd
import ta
import plotly.graph_objects as go
from vnstock import Vnstock  # <--- Cú pháp import mới của v3
from datetime import date, timedelta
import google.generativeai as genai

# --- 1. CẤU HÌNH TRANG ---
st.set_page_config(
    page_title="Pro Stock Analyst AI",
    layout="wide",
    page_icon="📈",
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

# --- 2. HÀM XỬ LÝ DỮ LIỆU (Cập nhật theo vnstock v3.4.2) ---
@st.cache_data(ttl=300)
def load_data(symbol, timeframe):
    end_date = date.today().strftime("%Y-%m-%d")
    start_date = (date.today() - timedelta(days=730)).strftime("%Y-%m-%d") 
    
    try:
        # Trong v3, interval dùng '1D' (Ngày) và '1W' (Tuần)
        resolution = '1W' if timeframe == 'Tuần' else '1D'
        
        # Cú pháp khởi tạo đối tượng lấy dữ liệu của thế hệ 3 (sử dụng nguồn TCBS)
        stock = Vnstock().stock(symbol=symbol, source='KBS')
        df = stock.quote.history(start=start_date, end=end_date, interval=resolution)
        
        if df is None or df.empty:
            return None
            
        df['time'] = pd.to_datetime(df['time'])
        df.set_index('time', inplace=True)
        
        # vnstock v3 trả về tên cột chữ thường, ta đổi lại cho khớp code
        mapping = {
            'open': 'Open', 'high': 'High', 'low': 'Low', 
            'close': 'Close', 'volume': 'Volume'
        }
        df.rename(columns=mapping, inplace=True)
        
        for col in mapping.values():
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
        df = df.dropna()
        return df
    except Exception as e:
        st.error(f"Lỗi API dữ liệu: {e}")
        return None

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
        name='Giá',
        increasing_line_color='#26a69a',
        decreasing_line_color='#ef5350'
    ))

    fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['MA20'], 
                             line=dict(color='yellow', width=1.5), name='MA20'))
    fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['MA50'], 
                             line=dict(color='purple', width=1.5), name='MA50'))

    fig.update_layout(
        title=f"Biểu đồ kỹ thuật {symbol}",
        yaxis_title='Giá (VND)',
        xaxis_rangeslider_visible=False,
        height=600,
        margin=dict(l=20, r=20, t=40, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig

# --- 4. HÀM GỌI AI ---
def get_ai_analysis(symbol, current_price, rsi, ma20, status_ma20, bb_status, avg_vol, vol_today):
    # Tự động lấy API Key từ hệ thống Secrets của Streamlit
    try:
        api_key = st.secrets["GEMINI_API_KEY"]
    except KeyError:
        return "⚠️ Lỗi: Hệ thống chưa được cấu hình API Key. Vui lòng kiểm tra lại cài đặt Secrets trên Streamlit."
    
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-2.5-flash')
    
    prompt = f"""Role: Bạn là Chuyên gia Phân tích Kỹ thuật Top 1 tại thị trường chứng khoán Việt Nam (VNI). Phong cách của bạn là: Ngắn gọn, súc tích, dựa trên số liệu, không đoán mò.

Task: Phân tích mã {symbol} dựa trên dữ liệu kỹ thuật sau:
- Giá hiện tại: {current_price:,.0f}
- RSI(14): {rsi:.2f}
- MA20: {ma20:,.0f} (Giá đang {status_ma20} đường MA20)
- Bollinger Bands: {bb_status}
- Volume trung bình 10 phiên: {avg_vol:,.0f} vs Volume hôm nay: {vol_today:,.0f}

Yêu cầu output format:
📊 **TỔNG QUAN:** [Xu hướng chính: Tăng/Giảm/Sideway]
🎯 **KHUYẾN NGHỊ:** [MUA / BÁN / QUAN SÁT]
1. Vùng mua an toàn: [Giá A - Giá B]
2. Mục tiêu chốt lời (Target): [Giá C] (Ngắn hạn)
3. Điểm cắt lỗ (Stoploss): [Giá D] (Bắt buộc phải có)
💡 **LÝ DO:** [Giải thích ngắn gọn 2 dòng dựa trên RSI và Volume]
"""
    try:
        with st.spinner('🤖 AI đang phân tích biểu đồ theo format chuyên gia...'):
            response = model.generate_content(prompt)
            return response.text
    except Exception as e:
        return f"Lỗi AI: {e}"

# --- 5. GIAO DIỆN CHÍNH (MAIN) ---
def main():
    # Tiêu đề chính của App
    st.markdown("<h1 style='text-align: center; color: #1E88E5;'>BOT Phân tích chứng khoán Việt Nam Bằng AI - Phiên bản 1.0</h1>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center; color: #1E88E5;'>Lập trình và thiết kế bởi: Hoàng Trung Dũng @2026</h3>", unsafe_allow_html=True)
    st.markdown("---")
    
    with st.sidebar:
        st.title("🎛️ Control Panel")
        # Đã loại bỏ hoàn toàn ô nhập API Key ở đây
        symbol = st.text_input("Mã Cổ Phiếu", value="DBC").upper()
        timeframe = st.selectbox("Khung thời gian", ["Ngày", "Tuần"])
        st.info("💡 Mẹo: Chọn 'Tuần' để xem xu hướng dài hạn.")
    
    if symbol:
        df = load_data(symbol, timeframe)
        
        if df is not None:
            df = calculate_indicators(df)
            last_row = df.iloc[-1]
            prev_row = df.iloc[-2]
            
            change = last_row['Close'] - prev_row['Close
