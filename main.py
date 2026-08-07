# ==========================================
# 1. حل مشكلة event loop قبل استيراد أي مكتبة!
# ==========================================
import sys
import asyncio
import warnings
import yfinance as yf

# إنشاء Event Loop وضبطه للـ Policy مباشرة قبل أي import
try:
    loop = asyncio.get_running_loop()
except RuntimeError:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

warnings.filterwarnings('ignore')

# ==========================================
# 2. استيراد المكتبات العامة
# ==========================================
import random
import time
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import ta
from openai import OpenAI

# ==========================================
# 3. استيراد IBKR (يدعم ib_async أو ib_insync)
# ==========================================
try:
    from ib_async import IB, Stock, MarketOrder, util
except ImportError:
    from ib_insync import IB, Stock, MarketOrder, util

# تفعيل الـ Loop الخاص بـ IBKR
util.startLoop()

# ==========================================
# 4. استيراد من المجلدات المحلية
# ==========================================
try:
    from models.base_model import LocalAITradingEngine
except ImportError:
    # محاكاة محرك الذكاء الاصطناعي لو لم تكن الوحدة موجودة
    class LocalAITradingEngine:
        def __init__(self):
            self.is_trained = False
            self.feature_importance = None
        def train_quick_model(self, df):
            self.is_trained = True
            return True
        def predict_opportunity(self, df):
            return "BUY", 85, "RSI تقاطع إيجابي"

# ==========================================
# إعدادات Streamlit
# ==========================================
st.set_page_config(
    page_title="AI Trading Bot (IBKR & Yahoo)",
    page_icon="🤖",
    layout="wide"
)

# ==========================================
# الثوابت والقوائم الافتراضية
# ==========================================
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 7497
DEFAULT_SYMBOL = "AAPL"
DEFAULT_QUANTITY = 10

DEFAULT_WATCHLIST = [
    "AAPL", "GOOGL", "MSFT", "AMZN", "TSLA",
    "NVDA", "META", "NFLX", "JPM", "VTI", "SPY", "QQQ"
]

def load_watchlist():
    if 'watchlist' not in st.session_state:
        st.session_state['watchlist'] = DEFAULT_WATCHLIST.copy()
    return st.session_state['watchlist']

def add_to_watchlist(symbol):
    if symbol and symbol.upper() not in st.session_state['watchlist']:
        st.session_state['watchlist'].append(symbol.upper())
        return True
    return False

def remove_from_watchlist(symbol):
    if symbol in st.session_state['watchlist']:
        st.session_state['watchlist'].remove(symbol)
        return True
    return False

# ==========================================
# دالة جلب البيانات من Yahoo Finance
# ==========================================

def get_market_data(symbol, period="5d", interval="5m"):
    """جلب البيانات الفنية مجاناً من Yahoo Finance"""
    try:
        print(f"🔄 جلب بيانات {symbol} من Yahoo Finance...")
        
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period, interval=interval)
        
        if df.empty:
            return None, f"❌ لا توجد بيانات للرمز: {symbol}"
        
        df.rename(columns={
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Volume': 'volume'
        }, inplace=True)
        
        df['date'] = df.index
        
        # حساب المؤشرات الفنية
        df['RSI'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
        df['SMA_20'] = ta.trend.sma_indicator(df['close'], window=20)
        df['SMA_50'] = ta.trend.sma_indicator(df['close'], window=50)
        df['volume_ma'] = ta.trend.sma_indicator(df['volume'], window=10)
        
        df.dropna(subset=['RSI', 'SMA_20'], inplace=True)
        
        return df, None
        
    except Exception as e:
        return None, f"❌ خطأ أثناء جلب البيانات من ياهو: {str(e)}"

# ==========================================
# دالة تنفيذ الأوامر عبر IBKR
# ==========================================

def execute_ib_order(action, symbol, quantity, host, port):
    """تنفيذ أمر التداول عبر IBKR بأمان"""
    ib = IB()
    try:
        client_id = random.randint(1000, 9999)
        ib.connect(host, port, clientId=client_id, timeout=5)
        
        contract = Stock(symbol, 'SMART', 'USD')
        ib.qualifyContracts(contract)
        
        order = MarketOrder(action, quantity)
        trade = ib.placeOrder(contract, order)
        
        ib.sleep(1)
        ib.disconnect()
        return f"✅ تم إرسال أمر {action} لعدد {quantity} سهم من {symbol} بنجاح!"
    except Exception as e:
        if ib.isConnected():
            ib.disconnect()
        return f"❌ فشل تنفيذ الأمر عبر IBKR: {str(e)}"

# ==========================================
# دوال التحليل والـ Plotly
# ==========================================

def analyze_with_local_ai(df):
    engine = st.session_state['ai_engine']
    if not engine.is_trained:
        with st.spinner("تدريب النموذج..."):
            engine.train_quick_model(df)
    
    action, confidence, reason = engine.predict_opportunity(df)
    result = f"[RECOMMENDATION: {action}]\nالثقة: {confidence}%\nالسبب: {reason}\n"
    return result, action, confidence

def analyze_with_openai(df_summary, api_key, symbol_name):
    if not api_key:
        return "⚠️ مطلوب مفتاح OpenAI", "HOLD", 0
    
    client = OpenAI(api_key=api_key)
    prompt = f"حلل البيانات الفنية لسهم {symbol_name}:\n{df_summary}\nأجب بتنسيق: [RECOMMENDATION: BUY] أو [RECOMMENDATION: SELL] أو [RECOMMENDATION: HOLD] ثم اشرح السبب."
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2
        )
        result = response.choices[0].message.content
        
        if "[RECOMMENDATION: BUY]" in result:
            action = "BUY"
        elif "[RECOMMENDATION: SELL]" in result:
            action = "SELL"
        else:
            action = "HOLD"
            
        return result, action, 70
    except Exception as e:
        return f"❌ خطأ: {e}", "HOLD", 0

def analyze_hybrid(df, api_key, symbol_name):
    local_result, local_action, local_conf = analyze_with_local_ai(df)
    if api_key:
        openai_result, openai_action, openai_conf = analyze_with_openai(df.tail(10).to_string(), api_key, symbol_name)
        if local_action == openai_action and local_action != "HOLD":
            final_action = local_action
            hybrid_result = f"✅ توافق: {local_action}\n{local_result}\n\n{openai_result}"
        else:
            final_action = "HOLD"
            hybrid_result = f"⚠️ تباين - انتظار\n{local_result}\n\n{openai_result}"
    else:
        final_action = local_action
        hybrid_result = local_result
    return hybrid_result, final_action, local_conf

def plot_chart(df, symbol_name):
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.7, 0.3],
        subplot_titles=(f'📈 {symbol_name}', 'RSI')
    )
    fig.add_trace(go.Candlestick(x=df['date'], open=df['open'], high=df['high'], low=df['low'], close=df['close'], name='Price'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['date'], y=df['SMA_20'], mode='lines', name='SMA 20', line=dict(color='orange')), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['date'], y=df['SMA_50'], mode='lines', name='SMA 50', line=dict(color='cyan')), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['date'], y=df['RSI'], mode='lines', name='RSI', line=dict(color='purple')), row=2, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
    
    fig.update_layout(height=500, template='plotly_dark', showlegend=True, hovermode='x unified', xaxis_rangeslider_visible=False)
    return fig

# ==========================================
# الدالة الرئيسية
# ==========================================

def main():
    st.title("🤖 بوت التداول بالذكاء الاصطناعي (Yahoo & IBKR)")
    
    if 'ai_engine' not in st.session_state:
        st.session_state['ai_engine'] = LocalAITradingEngine()
        
    with st.sidebar:
        st.header("⚙️ الإعدادات")
        st.subheader("⭐ الأسهم المفضلة")
        
        watchlist = load_watchlist()
        if watchlist:
            selected_symbol = st.selectbox("اختر من المفضلة:", options=watchlist, index=0, key="symbol_selector")
            if st.button("📌 اختر هذا السهم", use_container_width=True):
                st.session_state['selected_symbol'] = selected_symbol
                st.success(f"✅ تم اختيار {selected_symbol}")
            
            current_symbol = st.session_state.get('selected_symbol', selected_symbol)
            st.info(f"📌 السهم الحالي: **{current_symbol}**")
        else:
            st.warning("⚠️ لا توجد أسهم في المفضلة")
            current_symbol = DEFAULT_SYMBOL
            
        st.divider()
        
        with st.expander("➕ إضافة رمز جديد"):
            new_symbol = st.text_input("رمز السهم:", placeholder="مثل: AAPL, GOOGL", key="new_symbol_input")
            if st.button("إضافة", key="add_symbol_btn", use_container_width=True):
                if new_symbol and add_to_watchlist(new_symbol.upper()):
                    st.success(f"✅ تم إضافة {new_symbol.upper()}")
                    st.rerun()
                elif new_symbol:
                    st.warning(f"⚠️ {new_symbol.upper()} موجود بالفعل")
                    
        with st.expander("🗑️ حذف من المفضلة"):
            if watchlist:
                symbol_to_remove = st.selectbox("اختر رمز للحذف:", options=watchlist, index=None, placeholder="اختر رمز...", key="remove_selector")
                if symbol_to_remove and st.button("حذف", key="remove_btn", use_container_width=True):
                    if remove_from_watchlist(symbol_to_remove):
                        st.success(f"✅ تم حذف {symbol_to_remove}")
                        st.rerun()
                        
        st.caption(f"📊 عدد الأسهم في المفضلة: {len(watchlist)}")
        st.divider()
        
        st.subheader("⏱️ إعدادات البيانات")
        selected_period = st.selectbox("الفترة الزمنية:", options=["1d", "5d", "1mo", "3mo"], index=1)
        selected_interval = st.selectbox("حجم الشمعة:", options=["1m", "2m", "5m", "15m", "60m", "1d"], index=2)
        
        st.divider()
        st.subheader("🔌 إعدادات IBKR للتنفيذ")
        ib_host = st.text_input("IB Host", DEFAULT_HOST)
        ib_port = st.number_input("IB Port", value=DEFAULT_PORT)
        api_key = st.text_input("OpenAI API Key (اختياري)", type="password")
        
        symbol = st.session_state.get('selected_symbol', current_symbol)
        quantity = st.number_input("الكمية", value=DEFAULT_QUANTITY, step=1)
        analysis_mode = st.radio("وضع التحليل", ["المحرك المحلي", "OpenAI", "هجين"])
        
        st.divider()
        st.subheader("📊 حالة المحرك")
        engine = st.session_state['ai_engine']
        if engine.is_trained:
            st.success("✅ النموذج جاهز")
        else:
            st.warning("⚠️ غير مدرب")

    # الواجهة الرئيسية
    col1, col2 = st.columns([1.5, 1])
    
    with col1:
        st.subheader("📊 البيانات الفنية (Yahoo Finance)")
        st.info(f"📌 السهم الحالي: **{symbol}**")
        
        if st.button("🔄 جلب البيانات", use_container_width=True):
            with st.spinner("جاري جلب البيانات من Yahoo Finance..."):
                df, error = get_market_data(symbol, period=selected_period, interval=selected_interval)
                if error:
                    st.error(error)
                else:
                    st.session_state['df'] = df
                    with st.spinner("تدريب النموذج..."):
                        engine = st.session_state['ai_engine']
                        if engine.train_quick_model(df):
                            st.success("✅ تم تدريب النموذج بنجاح!")
                    
                    st.success(f"✅ {len(df)} شمعة جاهزة")
                    last_price = df['close'].iloc[-1]
                    change = ((df['close'].iloc[-1] - df['close'].iloc[-2]) / df['close'].iloc[-2]) * 100 if len(df) > 1 else 0
                    st.metric("السعر الحالي", f"${last_price:.2f}", f"{change:.2f}%")

        if 'df' in st.session_state:
            df = st.session_state['df']
            fig = plot_chart(df, symbol)
            st.plotly_chart(fig, use_container_width=True)
            
            engine = st.session_state['ai_engine']
            if engine.is_trained:
                action, confidence, _ = engine.predict_opportunity(df)
                if action == "BUY":
                    st.success(f"🟢 إشارة: شراء (ثقة: {confidence}%)")
                elif action == "SELL":
                    st.error(f"🔴 إشارة: بيع (ثقة: {confidence}%)")
                else:
                    st.warning(f"⏸️ إشارة: انتظار (ثقة: {confidence}%)")
                    
            with st.expander("📋 البيانات"):
                cols = ['date', 'open', 'high', 'low', 'close', 'volume', 'RSI', 'SMA_20', 'SMA_50']
                st.dataframe(df[cols].tail(10), use_container_width=True)

    with col2:
        st.subheader("🤖 التحليل والتنفيذ")
        
        if st.button("🧠 تحليل", use_container_width=True, type="primary"):
            if 'df' not in st.session_state:
                st.warning("⚠️ جلب البيانات أولاً")
            else:
                df = st.session_state['df']
                with st.spinner("جاري التحليل..."):
                    if analysis_mode == "المحرك المحلي":
                        result, action, conf = analyze_with_local_ai(df)
                    elif analysis_mode == "OpenAI":
                        result, action, conf = analyze_with_openai(df.tail(10).to_string(), api_key, symbol)
                    else:
                        result, action, conf = analyze_hybrid(df, api_key, symbol)
                    
                    st.session_state['result'] = result
                    st.session_state['action'] = action
                    st.session_state['confidence'] = conf
                    st.success("✅ تم التحليل")

        if 'result' in st.session_state:
            st.divider()
            action = st.session_state['action']
            conf = st.session_state.get('confidence', 0)
            
            if action == "BUY":
                st.success(f"🟢 **شراء** (ثقة: {conf}%)")
            elif action == "SELL":
                st.error(f"🔴 **بيع** (ثقة: {conf}%)")
            else:
                st.warning(f"⏸️ **انتظار** (ثقة: {conf}%)")
                
            st.text_area("التفاصيل:", st.session_state['result'], height=150)
            
            st.divider()
            st.subheader("💼 التنفيذ (عبر IBKR)")
            
            if action == "BUY":
                if st.button(f"🚀 شراء {quantity} سهم", use_container_width=True, type="primary"):
                    msg = execute_ib_order("BUY", symbol, quantity, ib_host, ib_port)
                    st.success(msg) if "✅" in msg else st.error(msg)
            elif action == "SELL":
                if st.button(f"🔻 بيع {quantity} سهم", use_container_width=True, type="primary"):
                    msg = execute_ib_order("SELL", symbol, quantity, ib_host, ib_port)
                    st.success(msg) if "✅" in msg else st.error(msg)
            else:
                st.info("⏸️ لا توجد صفقة للتنفيذ")
                
            if st.button("🗑️ مسح", use_container_width=True):
                for key in ['result', 'action', 'confidence']:
                    st.session_state.pop(key, None)
                st.rerun()

if __name__ == "__main__":
    main()
