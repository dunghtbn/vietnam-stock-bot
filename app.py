import streamlit as st
import pandas as pd
import ta
import plotly.graph_objects as go
from vnstock import Vnstock
from datetime import date, timedelta, datetime, timezone
from google import genai  # <-- ĐÃ CẬP NHẬT CHUẨN MỚI CỦA GOOGLE
import requests
import yfinance as yf
from plotly.subplots import make_subplots
import math

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

# --- THAY THẾ HÀM load_fundamental_data CŨ BẰNG ĐOẠN NÀY ---
@st.cache_data(ttl=86400) 
def load_fundamental_data(symbol):
    """Cỗ máy FA Hybrid: Dùng vnstock + Yahoo (Có in Log hệ thống)"""
    result = {'pe': 'N/A', 'pb': 'N/A', 'roe': 'N/A', 'market_cap': 'N/A', 'div_yield': 'N/A', 'debt_to_equity': 'N/A'}
    
    # In một dòng phân cách ra Log cho dễ nhìn
    print(f"\n[{current_time}] ⏳ BẮT ĐẦU TẢI DỮ LIỆU FA MÃ: {symbol}")
    
    # 1. ĐỘNG CƠ CHÍNH: Lấy P/E, P/B, ROE từ Vnstock
    try:
        stock = Vnstock().stock(symbol=symbol, source='VCI')
        df_overview = stock.company.overview()
        
        if df_overview is not None and not df_overview.empty:
            df_overview.columns = [str(c).lower().strip() for c in df_overview.columns]
            data = df_overview.iloc[0]
            
            if pd.notna(data.get('pe')): result['pe'] = f"{float(data.get('pe')):.2f}"
            if pd.notna(data.get('pb')): result['pb'] = f"{float(data.get('pb')):.2f}"
            
            roe = data.get('roe')
            if pd.notna(roe): 
                roe_val = float(roe)
                if -2.0 < roe_val < 2.0: roe_val *= 100 
                result['roe'] = f"{roe_val:.2f}" 
                
            mc = data.get('marketcap')
            if pd.notna(mc): result['market_cap'] = f"{float(mc):,.0f} Tỷ"
            
            # BÁO CÁO VÀO LOG: Đã lấy thành công từ Vnstock
            print(f"  ✅ [Nguồn 1] Lấy P/E, P/B, ROE, Vốn hóa thành công từ: VNSTOCK (VCI)")
        else:
            print(f"  ⚠️ [Nguồn 1] Vnstock không có dữ liệu (Empty).")
            
    except Exception as e:
        print(f"  ❌ [Nguồn 1] Lỗi khi gọi Vnstock: {e}")

    # 2. ĐỘNG CƠ PHỤ: Lấy Tỷ suất Cổ tức & Nợ/Vốn từ Yahoo Finance
    try:
        ticker = yf.Ticker(f"{symbol}.VN")
        info = ticker.info
        
        div_yield = info.get('dividendYield')
        debt_to_equity = info.get('debtToEquity')
        
        if div_yield is not None:
            dy_val = float(div_yield)
            result['div_yield'] = f"{dy_val:.2f}%" if dy_val > 1 else f"{dy_val * 100:.2f}%"
            print(f"  ✅ [Nguồn 2] Lấy Tỷ suất cổ tức thành công từ: YAHOO")
        else:
            print(f"  ⚠️ [Nguồn 2] Yahoo không có dữ liệu Cổ tức.")
            
        if debt_to_equity is not None:
            result['debt_to_equity'] = f"{float(debt_to_equity):.2f}%"
            print(f"  ✅ [Nguồn 2] Lấy Nợ/Vốn thành công từ: YAHOO")
        else:
            print(f"  ⚠️ [Nguồn 2] Yahoo không có dữ liệu Nợ/Vốn.")
            
        # 3. HỆ THỐNG TỰ ĐỘNG BÙ ĐẮP (FALLBACK)
        if result['pe'] == 'N/A' and info.get('trailingPE'): 
            result['pe'] = f"{float(info.get('trailingPE')):.2f}"
            print(f"  🔄 [Fallback] P/E bị thiếu ở Vnstock -> Đã lấy bù thành công từ YAHOO")
            
        if result['pb'] == 'N/A' and info.get('priceToBook'): 
            result['pb'] = f"{float(info.get('priceToBook')):.2f}"
            print(f"  🔄 [Fallback] P/B bị thiếu ở Vnstock -> Đã lấy bù thành công từ YAHOO")
            
    except Exception as e:
        print(f"  ❌ [Nguồn 2] Lỗi khi gọi Yahoo Finance: {e}")

    print(f"[{current_time}] 🏁 KẾT THÚC TẢI FA MÃ: {symbol}\n")
    return result

# --- 2. THAY THẾ HÀM get_valuation_metrics CŨ BẰNG ĐOẠN NÀY ---
@st.cache_data(ttl=86400)
def get_valuation_metrics(symbol, current_price):
    """Tính Giá trị Hợp lý: Ưu tiên Yahoo, nếu thiếu thì tự tính ngược từ P/E, P/B"""
    eps, bvps = 0, 0
    
    # Thử lấy EPS, BVPS trực tiếp từ Yahoo Finance
    try:
        ticker = yf.Ticker(f"{symbol}.VN")
        info = ticker.info
        eps = info.get('trailingEps') or 0
        bvps = info.get('bookValue') or 0
    except Exception:
        pass
        
    # Nếu Yahoo bị N/A, dùng Toán học để tính ngược từ PE, PB của Vnstock
    if eps == 0 or bvps == 0:
        try:
            stock = Vnstock().stock(symbol=symbol, source='VCI')
            df_overview = stock.company.overview()
            if df_overview is not None and not df_overview.empty:
                df_overview.columns = [str(c).lower().strip() for c in df_overview.columns]
                pe = df_overview.get('pe', pd.Series([None])).iloc[0]
                pb = df_overview.get('pb', pd.Series([None])).iloc[0]
                
                if pd.notna(pe) and pe > 0 and current_price > 0: eps = current_price / float(pe)
                if pd.notna(pb) and pb > 0 and current_price > 0: bvps = current_price / float(pb)
        except Exception:
            pass
            
    # Tính Công thức Graham nếu có đủ dữ liệu
    if eps > 0 and bvps > 0:
        graham_value = math.sqrt(22.5 * eps * bvps)
        upside_pct = ((graham_value - current_price) / current_price) * 100 if current_price > 0 else 0
        return {'price': current_price, 'fair_value': graham_value, 'upside': upside_pct, 'eps': eps, 'bvps': bvps}
        
    return {'price': current_price, 'fair_value': 0, 'upside': 0, 'eps': 0, 'bvps': 0}

def calculate_indicators(df):
    df['RSI'] = ta.momentum.RSIIndicator(close=df['Close'], window=14).rsi()
    df['MA20'] = ta.trend.SMAIndicator(close=df['Close'], window=20).sma_indicator()
    df['MA50'] = ta.trend.SMAIndicator(close=df['Close'], window=50).sma_indicator()
    
    indicator_bb = ta.volatility.BollingerBands(close=df['Close'], window=20, window_dev=2)
    df['BB_Upper'] = indicator_bb.bollinger_hband()
    df['BB_Lower'] = indicator_bb.bollinger_lband()
    
    macd = ta.trend.MACD(close=df['Close'])
    df['MACD'] = macd.macd()
    df['MACD_Signal'] = macd.macd_signal()
    df['MACD_Hist'] = macd.macd_diff()
    return df

# --- 3. HÀM VẼ BIỂU ĐỒ NÂNG CẤP ---
def plot_chart(df, symbol, indicator_choice="RSI"): 
    df['Vol_MA20'] = df['Volume'].rolling(window=20).mean()
    plot_df = df.tail(150)
    current_price = df['Close'].iloc[-1]
    
    colors = []
    for index, row in plot_df.iterrows():
        is_up = row['Close'] >= row['Open']
        if pd.notna(row['Vol_MA20']) and row['Volume'] >= 1.5 * row['Vol_MA20']:
            colors.append('#00FF00' if is_up else '#FF0000') 
        else:
            colors.append('#26a69a' if is_up else '#ef5350')

    fig = make_subplots(
        rows=3, cols=1, 
        shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.6, 0.2, 0.2] 
    )

    fig.add_trace(go.Candlestick(x=plot_df.index, open=plot_df['Open'], high=plot_df['High'], low=plot_df['Low'], close=plot_df['Close'], name='Giá', increasing_line_color='#26a69a', decreasing_line_color='#ef5350'), row=1, col=1)
    fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['MA20'], line=dict(color='yellow', width=1.5), name='MA20'), row=1, col=1)
    fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['MA50'], line=dict(color='purple', width=1.5), name='MA50'), row=1, col=1)
    fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['BB_Lower'], line=dict(color='rgba(33, 150, 243, 0.3)', width=1), name='BB Lower', showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['BB_Upper'], fill='tonexty', fillcolor='rgba(33, 150, 243, 0.08)', line=dict(color='rgba(33, 150, 243, 0.3)', width=1), name='Bollinger Bands'), row=1, col=1)
    fig.add_hline(y=current_price, line_dash="dash", line_color="#ff9800", line_width=1.5, annotation_text=f"Giá hiện tại: {current_price:,.2f}", annotation_position="bottom left", annotation_font=dict(color="#ff9800", size=12), row=1, col=1)

    fig.add_trace(go.Bar(x=plot_df.index, y=plot_df['Volume'], name='Khối lượng', marker_color=colors, showlegend=False), row=2, col=1)
    fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['Vol_MA20'], mode='lines', line=dict(color='#ff9800', width=1.5), name='MA20 Khối lượng', showlegend=False), row=2, col=1)

    if indicator_choice == "RSI":
        fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['RSI'], mode='lines', line=dict(color='#E1BEE7', width=1.5), name='RSI', showlegend=False), row=3, col=1)
        fig.add_hrect(y0=30, y1=70, fillcolor="purple", opacity=0.1, line_width=0, row=3, col=1)
        fig.add_hline(y=70, line_dash="dot", line_color="red", line_width=1, row=3, col=1)
        fig.add_hline(y=30, line_dash="dot", line_color="#26a69a", line_width=1, row=3, col=1)
        fig.update_yaxes(range=[0, 100], row=3, col=1) 
        yaxis3_title = 'RSI (14)'
    else:
        macd_colors = ['#26a69a' if val >= 0 else '#ef5350' for val in plot_df['MACD_Hist']]
        fig.add_trace(go.Bar(x=plot_df.index, y=plot_df['MACD_Hist'], marker_color=macd_colors, name='MACD Hist', showlegend=False), row=3, col=1)
        fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['MACD'], mode='lines', line=dict(color='#2962FF', width=1.5), name='MACD', showlegend=False), row=3, col=1)
        fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['MACD_Signal'], mode='lines', line=dict(color='#FF6D00', width=1.5), name='Signal', showlegend=False), row=3, col=1)
        fig.update_yaxes(autorange=True, row=3, col=1) 
        yaxis3_title = 'MACD'

    fig.update_layout(
        title=f"Biểu đồ Kỹ thuật {symbol}", yaxis_title='Giá (VND)', yaxis2_title='Khối lượng', yaxis3_title=yaxis3_title, 
        height=850, margin=dict(l=20, r=20, t=40, b=20), legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    fig.update_xaxes(rangeslider_visible=False)
    return fig

# --- 4. HÀM GỌI AI PHÂN TÍCH (ĐÃ CẬP NHẬT CHUẨN MỚI CỦA GOOGLE) ---
def get_ai_analysis(api_key, symbol, current_price, rsi, ma20, status_ma20, bb_status, avg_vol, vol_today, stock_perf, vnindex_perf, rs_status, pe, pb, roe, market_cap, div_yield, debt_to_equity, indicator_choice, macd, macd_signal, macd_hist):
    if not api_key: return "⚠️ Vui lòng nhập API Key để xem phân tích."
    
    client = genai.Client(api_key=api_key)
    
    if indicator_choice == "RSI": momentum_prompt = f"- RSI(14): {rsi:.2f} (Hãy đánh giá xem đang ở vùng quá mua, quá bán hay tích lũy)"
    else:
        giao_cat = "CẮT LÊN (Tín hiệu mua/tích cực)" if macd > macd_signal else "CẮT XUỐNG (Tín hiệu bán/tiêu cực)"
        momentum_prompt = f"- MACD: Đường MACD ({macd:.3f}) đang {giao_cat} đường Signal ({macd_signal:.3f}). Histogram đang ở mức {macd_hist:.3f}."

    prompt = f"""Role: Bạn là Chuyên gia Phân tích Chứng khoán. Phân tích mã {symbol}:
1. DỮ LIỆU KỸ THUẬT: Giá: {current_price:,.2f} | {momentum_prompt} | MA20: {ma20:,.2f} ({status_ma20}) | BB: {bb_status} | Vol nay: {vol_today:,.0f} | RS 20 phiên: {rs_status}
2. DỮ LIỆU CƠ BẢN: Vốn hóa: {market_cap} | P/E: {pe} | P/B: {pb} | ROE: {roe}% | Cổ tức: {div_yield} | Nợ/Vốn: {debt_to_equity}
Output: 📊 TỔNG QUAN | 🎯 KHUYẾN NGHỊ (Vùng mua, Target, Stoploss) | 💡 LÝ DO."""
    try:
        with st.spinner(f'🤖 AI đang đọc biểu đồ {indicator_choice}...'):
            return client.models.generate_content(model='gemini-2.5-flash', contents=prompt).text
    except Exception as e: return f"Lỗi AI: {e}"

def get_ai_best_pick(api_key, results_list):
    if not api_key: return "⚠️ Vui lòng nhập API Key."
    client = genai.Client(api_key=api_key)
    stocks_info = "\n".join([f"- {r['Mã CP']}: Giá {r['Giá Đóng Cửa']}, RSI {r['RSI']}, MA20 {r['MA20']}, ROE {r['ROE (%)']}%, P/E {r['P/E']}" for r in results_list])
    prompt = f"Role: CIO quỹ phòng hộ. Chọn 1 mã tốt nhất từ danh sách:\n{stocks_info}\nOutput: 🏆 MÃ CHỌN | 💡 LÝ DO | 🎯 KẾ HOẠCH HÀNH ĐỘNG."
    try:
        with st.spinner('🤖 AI đang chấm điểm...'): 
            return client.models.generate_content(model='gemini-2.5-flash', contents=prompt).text
    except Exception as e: return f"Lỗi AI: {e}"

# --- HÀM MỚI: AI PHÂN TÍCH VALUE INVESTING TỪ TAB 3 ---
def get_ai_value_pick(api_key, top_3_info):
    if not api_key: return "⚠️ Vui lòng nhập API Key."
    client = genai.Client(api_key=api_key)
    prompt = f"""Role: Chuyên gia Đầu tư Giá trị (Value Investor) kiểu Warren Buffett.
    Task: Đây là Top 3 cổ phiếu đang bị định giá thấp nhất theo công thức Graham:
    {top_3_info}
    Hãy chọn ra ĐÚNG 1 MÃ xứng đáng mua tích sản nhất.
    Output: 🏆 **CỔ PHIẾU TÍCH SẢN TỐT NHẤT:** [Tên mã] | 💡 **LÝ DO:** [Phân tích FA] | 🎯 **LƯU Ý RỦI RO:** [Chỉ ra Value Trap nếu có]."""
    try:
        with st.spinner('🤖 AI đang soi Báo cáo tài chính & lọc Bẫy giá trị...'): 
            return client.models.generate_content(model='gemini-2.5-flash', contents=prompt).text
    except Exception as e: return f"Lỗi AI: {e}"

# --- 5. GIAO DIỆN CHÍNH (MAIN) ---
def main():
    vn_tz = timezone(timedelta(hours=7))
    current_time = datetime.now(vn_tz).strftime("%H:%M:%S - %d/%m/%Y")
    
    st.markdown(f"<h2 style='text-align: center; color: #1E88E5;'>🚀 Bot Phân Tích Chứng Khoán Hybrid AI 4.0</h2>", unsafe_allow_html=True)
    st.markdown("---")
    
    with st.sidebar:
        st.title("🎛️ Control Panel")
        try: api_key = st.secrets["GEMINI_API_KEY"]
        except KeyError: api_key = ""
            
        symbol = st.text_input("Mã Cổ Phiếu", value="DBC").upper()
        timeframe = st.selectbox("Khung thời gian", ["Ngày", "Tuần"])
        indicator_choice = st.radio("Tầng 3: Chọn chỉ báo dao động", ["RSI", "MACD"], horizontal=True)
        
        # --- ĐÃ SỬA width="stretch" ---
        if st.button("🔄 Làm mới dữ liệu (Real-time)", width="stretch"):
            st.cache_data.clear() 
            st.rerun()            
        st.info("💡 Mẹo: Chọn 'Tuần' để xem xu hướng dài hạn.")
        st.success("✨ V2.1 - Cập nhật 27.3.2026 - Thiết kế và lập trình bởi Hoàng Trung Dũng - Email: dung@hdbn.vip")

    tab1, tab2, tab3 = st.tabs(["📊 Phân Tích Chuyên Sâu", "🎯 Radar Quét Cổ Phiếu", "💎 Săn Cổ Phiếu Rẻ (Value)"])
    
    with tab1:
        if symbol:
            df = load_data(symbol, timeframe)
            df_vnindex = load_vnindex_data(timeframe)
            fa_data = load_fundamental_data(symbol)
            
            if df is not None and not df.empty:
                df = calculate_indicators(df)
                last_row, prev_row = df.iloc[-1], df.iloc[-2]
                pct_change = ((last_row['Close'] - prev_row['Close']) / prev_row['Close']) * 100
                
                lookback = 20 if len(df) >= 20 else len(df) - 1
                stock_perf = ((last_row['Close'] - df['Close'].iloc[-1 - lookback]) / df['Close'].iloc[-1 - lookback]) * 100 if lookback > 0 else 0.0
                vnindex_perf = ((df_vnindex['Close'].iloc[-1] - df_vnindex['Close'].iloc[-1 - lookback]) / df_vnindex['Close'].iloc[-1 - lookback]) * 100 if df_vnindex is not None and len(df_vnindex) > lookback else 0.0
                rs_status = "KHỎE HƠN 💪" if stock_perf > vnindex_perf else "YẾU HƠN ⚠️" if stock_perf < vnindex_perf else "TƯƠNG ĐƯƠNG ⚖️"
                
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Giá hiện tại", f"{last_row['Close']:,.2f}", f"{pct_change:.2f}%")
                m2.metric("Khối lượng", f"{last_row['Volume']:,.0f}")
                m3.metric("RSI (14)", f"{last_row['RSI']:.1f}")
                m4.metric("MA20 Trend", "Tăng" if last_row['Close'] > last_row['MA20'] else "Giảm")
                
                st.divider()
                col_left, col_right = st.columns([2, 1])
                
                with col_left:
                    st.info(f"📈 **Đo lường RS (20 phiên):** Mã **{symbol}** thay đổi **{stock_perf:.2f}%** | VN-Index: **{vnindex_perf:.2f}%** ➔ Cổ phiếu đang **{rs_status}**")
                    st.subheader(f"📊 Biểu đồ {symbol} ({timeframe})")
                    fig = plot_chart(df, symbol, indicator_choice) 
                    st.plotly_chart(fig, use_container_width=True)

                with col_right:
                    st.subheader("📋 Chỉ số Kỹ thuật (TA)")
                    st.table(pd.DataFrame({"Chỉ số": ["MA20", "MA50", "BB Upper", "BB Lower"], "Giá trị": [f"{last_row['MA20']:,.2f}", f"{last_row['MA50']:,.2f}", f"{last_row['BB_Upper']:,.2f}", f"{last_row['BB_Lower']:,.2f}"]}))
                    
                    st.subheader("🏢 Chỉ số Cơ bản & Sức khỏe")
                    debt_display, pe_display, roe_display = fa_data['debt_to_equity'], fa_data['pe'], fa_data['roe']
                    
                    try:
                        d_val = float(fa_data['debt_to_equity'].replace('%', '').strip())
                        debt_display += " 🟢" if d_val < 100 else " 🟡" if d_val <= 200 else " 🔴"
                    except: pass
                    try:
                        p_val = float(fa_data['pe'])
                        pe_display += " 🟢" if p_val < 10 else " 🟡" if p_val <= 20 else " 🔴"
                    except: pass
                    try:
                        r_val = float(fa_data['roe'].replace('%', '').strip())
                        roe_display += " 🟢" if r_val > 15 else " 🟡" if r_val >= 10 else " 🔴"
                    except: pass

                    st.table(pd.DataFrame({
                        "Chỉ số": ["Vốn hóa thị trường", "Tỷ lệ Nợ / Vốn", "Tỷ suất Cổ tức", "P/E", "P/B", "ROE (%)"],
                        "Giá trị": [fa_data['market_cap'], debt_display, fa_data['div_yield'], pe_display, fa_data['pb'], roe_display]
                    }))                   
                    
                    if api_key:
                        # --- ĐÃ SỬA width="stretch" ---
                        if st.button("Phân tích chuyên sâu", width="stretch"):
                            status_ma20 = "nằm trên" if last_row['Close'] > last_row['MA20'] else "nằm dưới"
                            bb_status = "Quá mua" if last_row['Close'] >= last_row['BB_Upper'] else "Quá bán" if last_row['Close'] <= last_row['BB_Lower'] else "Bình thường"
                            analysis = get_ai_analysis(api_key, symbol, last_row['Close'], last_row['RSI'], last_row['MA20'], status_ma20, bb_status, df['Volume'].tail(10).mean(), last_row['Volume'], stock_perf, vnindex_perf, rs_status.split()[0], fa_data['pe'], fa_data['pb'], fa_data['roe'], fa_data['market_cap'], fa_data['div_yield'], fa_data['debt_to_equity'], indicator_choice, last_row['MACD'], last_row['MACD_Signal'], last_row['MACD_Hist'])
                            st.session_state.ai_analysis, st.session_state.analyzed_symbol = analysis, symbol

                        if st.session_state.get('ai_analysis') and st.session_state.get('analyzed_symbol') == symbol:
                            st.markdown(st.session_state.ai_analysis)
                    else:
                        st.warning("Vui lòng thiết lập API Key trong thư mục Secrets.")
            else:
                st.error("Không tìm thấy dữ liệu.")

    with tab2:
        st.subheader("📡 Radar Quét Đa Chiều (TA + FA)")
        tickers_input = st.text_input("Nhập danh sách mã:", "SSI, VND, HPG, HSG, NVL, SHS, MSB, VIX, CII, EVF, DBC, VNM, DXG, DIG, PDR, PVD")
        
        col1, col2, col3 = st.columns(3)
        with col1: f_rsi = st.selectbox("🎯 Điều kiện RSI:", ["Không lọc", "RSI < 30 (Bắt đáy)", "RSI > 50 (Tích cực)", "RSI > 70 (Quá mua)"])
        with col2: f_ma = st.selectbox("📈 Điều kiện MA20:", ["Không lọc", "Giá cắt lên MA20", "Nằm trên MA20"])
        with col3: f_roe = st.selectbox("🏢 Điều kiện ROE:", ["Không lọc", "ROE > 10%", "ROE > 15%", "ROE > 20%"])

        # --- ĐÃ SỬA width="stretch" ---
        if st.button("🚀 Kích Hoạt Radar", width="stretch"):
            st.session_state.radar_results, st.session_state.radar_ai_pick = [], ""
            tickers = [x.strip().upper() for x in tickers_input.split(",") if x.strip()]
            bar = st.progress(0)
            
            for i, t in enumerate(tickers):
                bar.progress((i + 1) / len(tickers), text=f"Đang phân tích {t}...")
                df_s = load_data(t, "Ngày")
                if df_s is not None and len(df_s) > 2:
                    df_s = calculate_indicators(df_s)
                    fa = load_fundamental_data(t)
                    lr, pr = df_s.iloc[-1], df_s.iloc[-2]
                    
                    passed = True
                    if "RSI < 30" in f_rsi and lr['RSI'] >= 30: passed = False
                    elif "RSI > 50" in f_rsi and lr['RSI'] <= 50: passed = False
                    elif "RSI > 70" in f_rsi and lr['RSI'] <= 70: passed = False
                    
                    if "cắt lên" in f_ma and not (pr['Close'] < pr['MA20'] and lr['Close'] > lr['MA20']): passed = False
                    elif "Nằm trên" in f_ma and lr['Close'] < lr['MA20']: passed = False
                    
                    roe_val = float(fa['roe'].replace('%','')) if fa['roe'] != "N/A" else 0
                    if "Không lọc" not in f_roe and fa['roe'] == "N/A": passed = False
                    if "ROE > 10%" in f_roe and roe_val <= 10: passed = False
                    elif "ROE > 15%" in f_roe and roe_val <= 15: passed = False
                    elif "ROE > 20%" in f_roe and roe_val <= 20: passed = False
                    
                    if passed: st.session_state.radar_results.append({"Mã CP": t, "Giá Đóng Cửa": f"{lr['Close']:,.2f}", "RSI": round(lr['RSI'], 2), "MA20": f"{lr['MA20']:,.2f}", "Khối Lượng": f"{lr['Volume']:,.0f}", "ROE (%)": fa['roe'], "P/E": fa['pe']})
            bar.empty()
            st.session_state.has_run_radar = True

        if st.session_state.get('has_run_radar'):
            if st.session_state.radar_results:
                st.dataframe(pd.DataFrame(st.session_state.radar_results), use_container_width=True)
                # --- ĐÃ SỬA width="stretch" ---
                if len(st.session_state.radar_results) > 1 and st.button("🏆 Nhờ AI chấm điểm", width="stretch"):
                    st.session_state.radar_ai_pick = get_ai_best_pick(api_key, st.session_state.radar_results)
                if st.session_state.radar_ai_pick: st.info(st.session_state.radar_ai_pick)
            else: st.warning("Không có mã nào thỏa mãn điều kiện.")

    with tab3:
        st.subheader("⚖️ Bảng Xếp Hạng Định Giá Cổ Phiếu (Fair Value)")
        val_tickers_input = st.text_input("Nhập danh sách mã để chấm điểm định giá:", "HPG, SSI, VND, DBC, VNM, TCB, MBB, FPT, MWG, REE, VCB", key="val_input")
        
        # --- ĐÃ SỬA width="stretch" ---
        if st.button("🔍 Quét Định Giá & Tiềm Năng", width="stretch"):
            val_tickers = [x.strip().upper() for x in val_tickers_input.split(",") if x.strip()]
            val_results = []
            pb = st.progress(0)
            
            # --- 3. TÌM VÀ SỬA ĐOẠN CODE NÀY TRONG VÒNG LẶP CỦA TAB 3 ---
            for i, t in enumerate(val_tickers):
                pb.progress((i + 1) / len(val_tickers), text=f"Đang định giá mã {t}...")
                
                # Cần tải giá hiện tại trước để tính định giá bù
                df_temp = load_data(t, "Ngày")
                curr_price = df_temp['Close'].iloc[-1] if (df_temp is not None and not df_temp.empty) else 0
                
                val_data = get_valuation_metrics(t, curr_price)
                fa_data = load_fundamental_data(t)
                             
                if val_data['fair_value'] > 0:
                    roe_val = float(fa_data['roe'].replace('%', '')) if fa_data['roe'] != "N/A" else 0
                    debt_val = float(fa_data['debt_to_equity'].replace('%', '')) if fa_data['debt_to_equity'] != "N/A" else 999
                    h_score = "Xuất sắc 🌟" if roe_val > 15 and debt_val < 100 else "Tốt ✅" if roe_val > 10 and debt_val < 200 else "Yếu ⚠️"
                    
                    pe_val = float(fa_data['pe']) if fa_data['pe'] != "N/A" else 15
                    pe_rating = "Rất Rẻ 🟢" if pe_val < 10 else "Đắt 🔴" if pe_val > 20 else "Hợp lý"

                    val_results.append({
                        "Mã CP": t, "Giá HT": val_data['price'], "Giá trị Hợp lý": val_data['fair_value'],
                        "Tăng lên (%)": val_data['upside'], "Sức Khỏe TC": h_score, "Định giá P/E": pe_rating,
                        "Tỉ số P/E": fa_data['pe'], "ROE (%)": fa_data['roe']
                    })
            pb.empty()
            
            if val_results:
                df_val = pd.DataFrame(val_results).sort_values(by="Tăng lên (%)", ascending=False).reset_index(drop=True)
                st.session_state.top_3_value = df_val.head(3) # Lưu top 3 cho AI
                
                df_display = df_val.copy()
                df_display['Giá HT'] = df_display['Giá HT'].apply(lambda x: f"{x:,.0f}")
                df_display['Giá trị Hợp lý'] = df_display['Giá trị Hợp lý'].apply(lambda x: f"{x:,.0f}")
                
                def color_upside(val): return 'color: #00FF00; font-weight: bold' if val > 20 else 'color: #26a69a' if val > 0 else 'color: #ef5350'
                
                styled_df = df_display.style.map(color_upside, subset=['Tăng lên (%)']).format({"Tăng lên (%)": "{:+.2f}%"})
                st.dataframe(styled_df, use_container_width=True, hide_index=True)
            else:
                st.warning("Không có dữ liệu hợp lệ (Lưu ý: Các mã thua lỗ EPS âm sẽ bị loại bỏ khỏi bảng).")

        # NÚT GỌI AI PHÂN TÍCH VALUE ĐƯỢC CHUYỂN RA NGOÀI ĐỂ KHÔNG BỊ MẤT KHI RERUN
        if st.session_state.get('top_3_value') is not None:
            st.markdown("---")
            st.subheader("🤖 Giám Đốc AI: Tích Sản Đầu Tư Giá Trị")
            # --- ĐÃ SỬA width="stretch" ---
            if st.button("🏆 Nhờ AI phân tích Top 3 mã định giá thấp nhất", width="stretch"):
                top_3_str = ""
                for _, row in st.session_state.top_3_value.iterrows():
                    top_3_str += f"- {row['Mã CP']}: Upside {row['Tăng lên (%)']:.1f}%, P/E {row['Tỉ số P/E']}, ROE {row['ROE (%)']}, {row['Sức Khỏe TC']}\n"
                
                ai_value_result = get_ai_value_pick(api_key, top_3_str)
                st.info(ai_value_result)

if __name__ == "__main__":
    main()
