import streamlit as st
import pandas as pd
import pandas_ta as ta
import plotly.graph_objects as go
from vnstock import stock_historical_data
from datetime import date, timedelta
import google.generativeai as genai

# --- 1. CẤU HÌNH TRANG ---
st.set_page_config(
    page_title="Pro Stock Analyst AI",
    layout="wide",
    page_icon="📈",
    initial_sidebar_state="expanded"
)

# CSS tùy chỉnh để giao diện gọn gàng hơn
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
    """Lấy dữ liệu và xử lý khung thời gian"""
    end_date = date.today().strftime("%Y-%m-%d")
    start_date = (date.today() - timedelta(days=730)).strftime("%Y-%m-%d") 
    
    try:
        resolution = '1W' if timeframe == 'Tuần' else '1D'
        df = stock_historical_data(symbol, start_date, end_date, resolution, "stock")
        
        if df is None or df.empty:
            return None
            
        df['time'] = pd.to_datetime(df['time'])
        df.set_index('time', inplace=True)
        
        mapping = {
            'open': 'Open', 'high': 'High', 'low': 'Low', 
            'close': 'Close', 'volume': 'Volume'
        }
        df.rename(columns=mapping, inplace=True)
        
        for col in mapping.values():
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
        return df
    except Exception as e:
        st.error(f"Lỗi tải dữ liệu: {e}")
        return None

def calculate_indicators(df):
    """Tính toán các chỉ báo kỹ thuật"""
    df['RSI'] = df.ta.rsi(length=14)
    df['MA20'] = df.ta.sma(length=20)
    df['MA50'] = df.ta.sma(length=50)
    
    bb = df.ta.bbands(length=20, std=2)
    df['BB_Upper'] = bb['BBU_20_2.0']
    df['BB_Lower'] = bb['BBL_20_2.0']
    
    return df

# --- 3. HÀM VẼ BIỂU ĐỒ (PLOTLY) ---
def plot_chart(df, symbol):
    """Vẽ biểu đồ nến và các đường MA"""
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
def get_ai_analysis(api_key, symbol, current_price, rsi, ma20, status_ma20, bb_status, avg_vol, vol_today):
    """Gửi prompt cho AI và nhận về phân tích"""
    if not api_key:
        return "⚠️ Vui lòng nhập API Key để xem phân tích."
    
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-pro')
    
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
    # --- Sidebar ---
    with st.sidebar:
        st.title("🎛️ Control Panel")
        api_key = st.text_input("Gemini API Key", type="password")
        symbol = st.text_input("Mã Cổ Phiếu", value="HPG").upper()
        timeframe = st.selectbox("Khung thời gian", ["Ngày", "Tuần"])
        st.info("💡 Mẹo: Chọn 'Tuần' để xem xu hướng dài hạn.")
    
    # --- Main Content ---
    if symbol:
        df = load_data(symbol, timeframe)
        
        if df is not None:
            df = calculate_indicators(df)
            last_row = df.iloc[-1]
            prev_row = df.iloc[-2]
            
            # --- Top Metrics Row ---
            change = last_row['Close'] - prev_row['Close']
            pct_change = (change / prev_row['Close']) * 100
            
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Giá đóng cửa", f"{last_row['Close']:,.0f}", f"{pct_change:.2f}%")
            m2.metric("Khối lượng", f"{last_row['Volume']:,.0f}")
            m3.metric("RSI (14)", f"{last_row['RSI']:.1f}")
            m4.metric("MA20 Trend", "Tăng" if last_row['Close'] > last_row['MA20'] else "Giảm")
            
            st.divider()

            # --- Layout 2 Cột ---
            col_left, col_right = st.columns([2, 1])
            
            with col_left:
                st.subheader(f"📊 Biểu đồ {symbol} ({timeframe})")
                fig = plot_chart(df, symbol)
                st.plotly_chart(fig, use_container_width=True)

            with col_right:
                st.subheader("📋 Chỉ số Kỹ thuật")
                
                tech_data = {
                    "Chỉ số": ["MA20", "MA50", "BB Upper", "BB Lower"],
                    "Giá trị": [
                        f"{last_row['MA20']:,.0f}",
                        f"{last_row['MA50']:,.0f}",
                        f"{last_row['BB_Upper']:,.0f}",
                        f"{last_row['BB_Lower']:,.0f}"
                    ]
                }
                st.table(pd.DataFrame(tech_data))
                
                # Tính toán các thông số cho AI Prompt
                status_ma20 = "nằm trên" if last_row['Close'] > last_row['MA20'] else "nằm dưới"
                
                if last_row['Close'] >= last_row['BB_Upper']:
                    bb_status = "Chạm/Vượt Band trên (Cảnh báo Quá mua)"
                elif last_row['Close'] <= last_row['BB_Lower']:
                    bb_status = "Chạm/Thủng Band dưới (Cảnh báo Quá bán)"
                else:
                    bb_status = "Dao động bình thường trong dải Band"
                
                avg_vol = df['Volume'].tail(10).mean()
                vol_today = last_row['Volume']
                
                # Nút gọi AI
                st.subheader("🤖 AI Khuyến Nghị")
                if api_key:
                    if st.button("Phân tích ngay", use_container_width=True):
                        analysis = get_ai_analysis(
                            api_key=api_key, 
                            symbol=symbol, 
                            current_price=last_row['Close'], 
                            rsi=last_row['RSI'], 
                            ma20=last_row['MA20'], 
                            status_ma20=status_ma20, 
                            bb_status=bb_status, 
                            avg_vol=avg_vol, 
                            vol_today=vol_today
                        )
                        st.markdown(analysis)
                else:
                    st.warning("Nhập API Key để xem nhận định.")
                    
        else:
            st.error(f"Không tìm thấy dữ liệu cho mã {symbol}")

if __name__ == "__main__":
    main()