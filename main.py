# ==========================================
# حل مشكلة event loop
# ==========================================
import asyncio
import nest_asyncio
nest_asyncio.apply()

try:
    loop = asyncio.get_event_loop()
except RuntimeError:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

# ==========================================
# استيراد المكتبات
# ==========================================
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import ta
import yfinance as yf
from openai import OpenAI
import warnings
warnings.filterwarnings('ignore')

# ==========================================
# استيراد من المجلدات
# ==========================================
from models.base_model import LocalAITradingEngine
from brokers.ibkr_broker import IBKRBroker
from strategies.ml_strategy import MLStrategy
from utils.indicators import TechnicalIndicators

# ==========================================
# إعدادات الصفحة
# ==========================================
st.set_page_config(
    page_title="AI Trading Bot",
    page_icon="🤖",
    layout="wide"
)

# ==========================================
# قائمة الأسهم المفضلة
# ==========================================
DEFAULT_WATCHLIST = ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA", "NVDA"]

def load_watchlist():
    if 'watchlist' not in st.session_state:
        st.session_state['watchlist'] = DEFAULT_WATCHLIST.copy()
    return st.session_state['watchlist']

# ==========================================
# دوال جلب البيانات
# ==========================================
def get_market_data(symbol, period="5d", interval="5m"):
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period, interval=interval)
        if df.empty:
            return None, f"❌ لا توجد بيانات للرمز: {symbol}"
        
        df.rename(columns={
            'Open': 'open', 'High': 'high', 'Low': 'low',
            'Close': 'close', 'Volume': 'volume'
        }, inplace=True)
        
        df.index = df.index.tz_localize(None)
        df['date'] = df.index
        
        df = TechnicalIndicators.add_all_indicators(df)
        df.dropna(inplace=True)
        return df, None
    except Exception as e:
        return None, f"❌ خطأ: {str(e)}"

# ==========================================
# دوال الرسم البياني
# ==========================================
def plot_chart(df, symbol_name):
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        vertical_spacing=0.05, row_heights=[0.7, 0.3],
        subplot_titles=(f'📈 {symbol_name}', '📊 RSI')
    )
    
    fig.add_trace(go.Candlestick(
        x=df['date'], open=df['open'], high=df['high'],
        low=df['low'], close=df['close'], name='السعر'
    ), row=1, col=1)
    
    fig.add_trace(go.Scatter(x=df['date'], y=df['SMA_20'], 
                  mode='lines', name='SMA 20', line=dict(color='orange')), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['date'], y=df['SMA_50'], 
                  mode='lines', name='SMA 50', line=dict(color='cyan')), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['date'], y=df['RSI'], 
                  mode='lines', name='RSI', line=dict(color='purple')), row=2, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
    
    fig.update_layout(height=500, template='plotly_dark', showlegend=True)
    return fig

# ==========================================
# التطبيق الرئيسي
# ==========================================
def main():
    st.title("🤖 بوت التداول الذكي")
    
    if 'ai_engine' not in st.session_state:
        st.session_state['ai_engine'] = LocalAITradingEngine()
    
    # الشريط الجانبي
    with st.sidebar:
        st.header("⚙️ الإعدادات")
        
        watchlist = load_watchlist()
        symbol = st.selectbox("اختر السهم:", watchlist)
        
        new_symbol = st.text_input("إضافة رمز:", placeholder="AAPL")
        if st.button("➕ إضافة"):
            if new_symbol and new_symbol.upper() not in watchlist:
                watchlist.append(new_symbol.upper())
                st.rerun()
        
        period = st.selectbox("الفترة:", ["1d", "5d", "1mo", "3mo"], index=1)
        interval = st.selectbox("الفاصل:", ["1m", "5m", "15m", "60m"], index=1)
        quantity = st.number_input("الكمية", value=10, step=1)
        
        st.divider()
        engine = st.session_state['ai_engine']
        if engine.is_trained:
            st.success("✅ النموذج جاهز")
        else:
            st.warning("⚠️ غير مدرب")
    
    # الأعمدة الرئيسية
    col1, col2 = st.columns([1.5, 1])
    
    with col1:
        st.subheader("📊 البيانات")
        
        if st.button("🔄 جلب البيانات", use_container_width=True):
            with st.spinner("جاري الجلب..."):
                df, error = get_market_data(symbol, period, interval)
                if error:
                    st.error(error)
                else:
                    st.session_state['df'] = df
                    engine = st.session_state['ai_engine']
                    if engine.train_quick_model(df):
                        st.success("✅ تم تدريب النموذج!")
                    st.success(f"✅ {len(df)} شمعة")
        
        if 'df' in st.session_state:
            df = st.session_state['df']
            last_price = df['close'].iloc[-1]
            change = ((df['close'].iloc[-1] - df['close'].iloc[-2]) / df['close'].iloc[-2]) * 100 if len(df) > 1 else 0
            st.metric("السعر", f"${last_price:.2f}", f"{change:+.2f}%")
            
            fig = plot_chart(df, symbol)
            st.plotly_chart(fig, use_container_width=True)
            
            with st.expander("📋 البيانات"):
                st.dataframe(df.tail(10))
    
    with col2:
        st.subheader("🤖 التحليل")
        
        if st.button("🧠 تحليل", use_container_width=True, type="primary"):
            if 'df' not in st.session_state:
                st.warning("⚠️ جلب البيانات أولاً")
            else:
                df = st.session_state['df']
                action, conf, reason = engine.predict_opportunity(df)
                st.session_state['result'] = reason
                st.session_state['action'] = action
                st.session_state['confidence'] = conf
                st.success("✅ تم التحليل")
        
        if 'result' in st.session_state:
            st.divider()
            action = st.session_state['action']
            conf = st.session_state.get('confidence', 0)
            
            if action == "BUY":
                st.success(f"🟢 **شراء** ({conf}%)")
            elif action == "SELL":
                st.error(f"🔴 **بيع** ({conf}%)")
            else:
                st.warning(f"⏸️ **انتظار** ({conf}%)")
            
            st.text_area("التفاصيل:", st.session_state['result'], height=100)
            
            st.divider()
            if action == "BUY":
                st.button(f"🚀 شراء {quantity} سهم", use_container_width=True)
            elif action == "SELL":
                st.button(f"🔻 بيع {quantity} سهم", use_container_width=True)

if __name__ == "__main__":
    main()
