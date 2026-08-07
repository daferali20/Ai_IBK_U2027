# ==========================================
# حل مشكلة event loop - يجب أن يكون في البداية
# ==========================================
import sys
import asyncio
import warnings
import nest_asyncio

import random

def get_market_data(symbol, host, port):
    """جلب البيانات من IBKR - نسخة آمنة مع Streamlit"""
    # استخدام loop الفرعي الخاص بـ ib_insync بدلاً من nest_asyncio اليدوي
    util.startLoop() 
    ib = IB()
    client_id = random.randint(1000, 9999) # clientId عشوائي لتجنب التضارب
    
    try:
        print(f"🔄 محاولة الاتصال بـ {host}:{port} (ID: {client_id})...")
        ib.connect(host, int(port), clientId=client_id, timeout=10)
        
        contract = Stock(symbol, 'SMART', 'USD')
        
        bars = ib.reqHistoricalData(
            contract,
            endDateTime='',
            durationStr='2 D',
            barSizeSetting='5 mins',
            whatToShow='TRADES',
            useRTH=True
        )
        
        df = util.df(bars)
        
        if df is None or df.empty:
            ib.disconnect()
            return None, "❌ لا توجد بيانات للرمز المحدد"
            
        # حساب المؤشرات
        df['RSI'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
        df['SMA_20'] = ta.trend.sma_indicator(df['close'], window=20)
        df['SMA_50'] = ta.trend.sma_indicator(df['close'], window=50)
        df['volume_ma'] = ta.trend.sma_indicator(df['volume'], window=10)
        df['date'] = df.index
        
        ib.disconnect()
        return df, None
        
    except Exception as e:
        if ib.isConnected():
            ib.disconnect()
        return None, f"❌ خطأ في الاتصال: {str(e)[:150]}"

warnings.filterwarnings('ignore')

# ==========================================
# استيراد المكتبات
# ==========================================
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import ta
from openai import OpenAI
import time

# ==========================================
# استيراد IBKR
# ==========================================
from ib_insync import IB, Stock, MarketOrder, util

# ==========================================
# استيراد من المجلدات
# ==========================================
from models.base_model import LocalAITradingEngine

# ==========================================
# إعدادات Streamlit
# ==========================================
st.set_page_config(
    page_title="AI Trading Bot (IBKR)",
    page_icon="🤖",
    layout="wide"
)

# ==========================================
# الثوابت
# ==========================================
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 7497
DEFAULT_SYMBOL = "AAPL"
DEFAULT_QUANTITY = 10

# ==========================================
# قائمة الأسهم المفضلة
# ==========================================

# القائمة الافتراضية
DEFAULT_WATCHLIST = [
    "AAPL",   # Apple
    "GOOGL",  # Google
    "MSFT",   # Microsoft
    "AMZN",   # Amazon
    "TSLA",   # Tesla
    "NVDA",   # NVIDIA
    "META",   # Meta (Facebook)
    "NFLX",   # Netflix
    "JPM",    # JPMorgan
    "VTI",    # Vanguard Total Stock Market
    "SPY",    # S&P 500 ETF
    "QQQ",    # NASDAQ ETF
]

def load_watchlist():
    """تحميل قائمة المفضلة من session_state"""
    if 'watchlist' not in st.session_state:
        st.session_state['watchlist'] = DEFAULT_WATCHLIST.copy()
    return st.session_state['watchlist']

def save_watchlist(watchlist):
    """حفظ قائمة المفضلة في session_state"""
    st.session_state['watchlist'] = watchlist

def add_to_watchlist(symbol):
    """إضافة رمز إلى المفضلة"""
    if symbol and symbol.upper() not in st.session_state['watchlist']:
        st.session_state['watchlist'].append(symbol.upper())
        return True
    return False

def remove_from_watchlist(symbol):
    """حذف رمز من المفضلة"""
    if symbol in st.session_state['watchlist']:
        st.session_state['watchlist'].remove(symbol)
        return True
    return False

# ==========================================
# دوال IBKR - نسخة متزامنة (لاستخدامها في Streamlit)
# ==========================================

def get_market_data(symbol, host, port):
    """جلب البيانات من IBKR - نسخة متزامنة"""
    ib = IB()
    try:
        print(f"🔄 محاولة الاتصال بـ {host}:{port}...")
        ib.connect(host, int(port), clientId=99, timeout=10)
        
        contract = Stock(symbol, 'SMART', 'USD')
        
        bars = ib.reqHistoricalData(
            contract,
            endDateTime='',
            durationStr='2 D',
            barSizeSetting='5 mins',
            whatToShow='TRADES',
            useRTH=True
        )
        
        df = util.df(bars)
        
        if df.empty:
            return None, "لا توجد بيانات"
        
        # المؤشرات الفنية
        df['RSI'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
        df['SMA_20'] = ta.trend.sma_indicator(df['close'], window=20)
        df['SMA_50'] = ta.trend.sma_indicator(df['close'], window=50)
        df['volume_ma'] = ta.trend.sma_indicator(df['volume'], window=10)
        df['date'] = df.index
        
        ib.disconnect()
        print("✅ تم جلب البيانات بنجاح")
        return df, None
        
    except Exception as e:
        try:
            ib.disconnect()
        except:
            pass
        
        error_msg = str(e)
        if "Connect call failed" in error_msg:
            return None, "❌ لا يمكن الاتصال بـ IBKR.\n\nتأكد من:\n1. تشغيل TWS أو IB Gateway\n2. تمكين API في الإعدادات\n3. استخدام 127.0.0.1 و 7497"
        else:
            return None, f"❌ خطأ: {error_msg[:200]}"

def execute_ib_order(action, symbol, qty, host, port):
    """تنفيذ أمر تداول"""
    ib = IB()
    try:
        ib.connect(host, int(port), clientId=100, timeout=10)
        contract = Stock(symbol, 'SMART', 'USD')
        order = MarketOrder(action, qty)
        trade = ib.placeOrder(contract, order)
        ib.sleep(2)
        
        status = trade.orderStatus.status
        ib.disconnect()
        
        if status in ['Filled', 'Submitted']:
            return f"✅ تم تنفيذ أمر {action} بنجاح!"
        else:
            return f"⚠️ الحالة: {status}"
            
    except Exception as e:
        try:
            ib.disconnect()
        except:
            pass
        return f"❌ خطأ: {e}"

# ==========================================
# دوال التحليل
# ==========================================

def analyze_with_local_ai(df):
    """تحليل باستخدام المحرك المحلي"""
    engine = st.session_state['ai_engine']
    
    if not engine.is_trained:
        with st.spinner("تدريب النموذج..."):
            engine.train_quick_model(df)
    
    action, confidence, reason = engine.predict_opportunity(df)
    
    result = f"[RECOMMENDATION: {action}]\n"
    result += f"الثقة: {confidence}%\n"
    result += f"السبب: {reason}\n"
    
    return result, action, confidence

def analyze_with_openai(df_summary, api_key, symbol_name):
    """تحليل باستخدام OpenAI"""
    if not api_key:
        return "⚠️ مطلوب مفتاح OpenAI", "HOLD", 0
    
    client = OpenAI(api_key=api_key)
    
    prompt = f"""
    حلل البيانات الفنية لسهم {symbol_name}:
    {df_summary}
    
    أجب بتنسيق:
    [RECOMMENDATION: BUY] أو [RECOMMENDATION: SELL] أو [RECOMMENDATION: HOLD]
    ثم اشرح السبب.
    """
    
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
    """تحليل هجين"""
    local_result, local_action, local_conf = analyze_with_local_ai(df)
    
    if api_key:
        openai_result, openai_action, openai_conf = analyze_with_openai(
            df.tail(10).to_string(), api_key, symbol_name
        )
        
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

# ==========================================
# دوال الرسم البياني
# ==========================================

def plot_chart(df, symbol_name):
    """رسم بياني تفاعلي"""
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.7, 0.3],
        subplot_titles=(f'📈 {symbol_name}', 'RSI')
    )
    
    # الشموع
    fig.add_trace(
        go.Candlestick(
            x=df['date'],
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name='Price'
        ),
        row=1, col=1
    )
    
    # المتوسطات
    fig.add_trace(
        go.Scatter(x=df['date'], y=df['SMA_20'], mode='lines', name='SMA 20', line=dict(color='orange')),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=df['date'], y=df['SMA_50'], mode='lines', name='SMA 50', line=dict(color='cyan')),
        row=1, col=1
    )
    
    # RSI
    fig.add_trace(
        go.Scatter(x=df['date'], y=df['RSI'], mode='lines', name='RSI', line=dict(color='purple')),
        row=2, col=1
    )
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
    
    fig.update_layout(
        height=500,
        template='plotly_dark',
        showlegend=True,
        hovermode='x unified',
        xaxis_rangeslider_visible=False
    )
    
    return fig

# ==========================================
# الدالة الرئيسية
# ==========================================

def main():
    """التطبيق الرئيسي"""
    
    st.title("🤖 بوت التداول بالذكاء الاصطناعي (IBKR)")
    
    # تنبيه للمستخدم
    st.info("💡 تأكد من تشغيل TWS أو IB Gateway مع تمكين API")
    
    # تهيئة المحرك
    if 'ai_engine' not in st.session_state:
        st.session_state['ai_engine'] = LocalAITradingEngine()
    
    # ===== الشريط الجانبي =====
    with st.sidebar:
        st.header("⚙️ الإعدادات")
        
        # ===== قسم الأسهم المفضلة =====
        st.subheader("⭐ الأسهم المفضلة")
        
        # تحميل القائمة
        watchlist = load_watchlist()
        
        # اختيار من القائمة
        if watchlist:
            selected_symbol = st.selectbox(
                "اختر من المفضلة:",
                options=watchlist,
                index=0,
                key="symbol_selector"
            )
            
            # زر اختيار السهم
            col_sel1, col_sel2 = st.columns([2, 1])
            with col_sel1:
                if st.button("📌 اختر هذا السهم", use_container_width=True):
                    st.session_state['selected_symbol'] = selected_symbol
                    st.success(f"✅ تم اختيار {selected_symbol}")
            
            # عرض السهم المختار حالياً
            if 'selected_symbol' in st.session_state:
                current_symbol = st.session_state['selected_symbol']
                st.info(f"📌 السهم الحالي: **{current_symbol}**")
            else:
                current_symbol = selected_symbol
        else:
            st.warning("⚠️ لا توجد أسهم في المفضلة")
            current_symbol = DEFAULT_SYMBOL
        
        st.divider()
        
        # ===== إضافة رمز جديد =====
        with st.expander("➕ إضافة رمز جديد"):
            new_symbol = st.text_input("رمز السهم:", placeholder="مثل: AAPL, GOOGL", key="new_symbol_input")
            if st.button("إضافة", key="add_symbol_btn", use_container_width=True):
                if new_symbol:
                    if add_to_watchlist(new_symbol.upper()):
                        st.success(f"✅ تم إضافة {new_symbol.upper()}")
                        st.rerun()
                    else:
                        st.warning(f"⚠️ {new_symbol.upper()} موجود بالفعل")
                else:
                    st.warning("⚠️ أدخل رمز السهم")
        
        # ===== حذف من المفضلة =====
        with st.expander("🗑️ حذف من المفضلة"):
            if watchlist:
                symbol_to_remove = st.selectbox(
                    "اختر رمز للحذف:",
                    options=watchlist,
                    index=None,
                    placeholder="اختر رمز...",
                    key="remove_selector"
                )
                if symbol_to_remove and st.button("حذف", key="remove_btn", use_container_width=True):
                    if remove_from_watchlist(symbol_to_remove):
                        st.success(f"✅ تم حذف {symbol_to_remove}")
                        st.rerun()
            else:
                st.info("📭 القائمة فارغة")
        
        # ===== عرض عدد الأسهم =====
        st.caption(f"📊 عدد الأسهم في المفضلة: {len(watchlist)}")
        
        st.divider()
        
        # ===== باقي الإعدادات =====
        api_key = st.text_input("OpenAI API Key (اختياري)", type="password")
        ib_host = st.text_input("IB Host", DEFAULT_HOST)
        ib_port = st.number_input("IB Port", value=DEFAULT_PORT)
        
        # استخدام السهم المختار
        if 'selected_symbol' in st.session_state:
            symbol = st.session_state['selected_symbol']
        else:
            symbol = current_symbol if watchlist else DEFAULT_SYMBOL
        
        quantity = st.number_input("الكمية", value=DEFAULT_QUANTITY, step=1)
        
        analysis_mode = st.radio(
            "وضع التحليل",
            ["المحرك المحلي", "OpenAI", "هجين"]
        )
        
        st.divider()
        st.subheader("📊 حالة المحرك")
        engine = st.session_state['ai_engine']
        if engine.is_trained:
            st.success("✅ النموذج جاهز")
            if hasattr(engine, 'feature_importance') and engine.feature_importance is not None:
                with st.expander("أهم الميزات"):
                    st.dataframe(engine.feature_importance.head(5))
        else:
            st.warning("⚠️ غير مدرب")
    
    # ===== الأعمدة الرئيسية =====
    col1, col2 = st.columns([1.5, 1])
    
    # العمود الأول - البيانات
    with col1:
        st.subheader("📊 البيانات الفنية")
        
        # عرض السهم المختار
        st.info(f"📌 السهم الحالي: **{symbol}**")
        
        if st.button("🔄 جلب البيانات", use_container_width=True):
            with st.spinner("جاري الاتصال بـ IBKR..."):
                df, error = get_market_data(symbol, ib_host, ib_port)
                
                if error:
                    st.error(f"{error}")
                else:
                    st.session_state['df'] = df
                    
                    with st.spinner("تدريب النموذج..."):
                        engine = st.session_state['ai_engine']
                        if engine.train_quick_model(df):
                            st.success("✅ تم تدريب النموذج بنجاح!")
                        else:
                            st.warning("⚠️ بيانات غير كافية للتدريب")
                    
                    st.success(f"✅ {len(df)} شمعة جاهزة")
                    
                    last_price = df['close'].iloc[-1]
                    change = ((df['close'].iloc[-1] - df['close'].iloc[-2]) / df['close'].iloc[-2]) * 100 if len(df) > 1 else 0
                    st.metric("السعر الحالي", f"${last_price:.2f}", f"{change:.2f}%")
        
        if 'df' in st.session_state:
            df = st.session_state['df']
            
            # الرسم البياني
            fig = plot_chart(df, symbol)
            st.plotly_chart(fig, use_container_width=True)
            
            # إشارة التداول
            engine = st.session_state['ai_engine']
            if engine.is_trained:
                action, confidence, _ = engine.predict_opportunity(df)
                if action == "BUY":
                    st.success(f"🟢 إشارة: شراء (ثقة: {confidence}%)")
                elif action == "SELL":
                    st.error(f"🔴 إشارة: بيع (ثقة: {confidence}%)")
                else:
                    st.warning(f"⏸️ إشارة: انتظار (ثقة: {confidence}%)")
            
            # الجدول
            with st.expander("📋 البيانات"):
                cols = ['date', 'open', 'high', 'low', 'close', 'volume', 'RSI', 'SMA_20', 'SMA_50']
                st.dataframe(
                    df[cols].tail(10).style.format({
                        'close': '${:.2f}',
                        'open': '${:.2f}',
                        'high': '${:.2f}',
                        'low': '${:.2f}',
                        'volume': '{:,.0f}',
                        'RSI': '{:.1f}',
                        'SMA_20': '${:.2f}',
                        'SMA_50': '${:.2f}'
                    }),
                    use_container_width=True
                )
    
    # العمود الثاني - التحليل والتنفيذ
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
                        result, action, conf = analyze_with_openai(
                            df.tail(10).to_string(), api_key, symbol
                        )
                    else:  # هجين
                        result, action, conf = analyze_hybrid(df, api_key, symbol)
                    
                    st.session_state['result'] = result
                    st.session_state['action'] = action
                    st.session_state['confidence'] = conf
                    
                    st.success("✅ تم التحليل")
        
        # عرض النتيجة
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
            
            # أزرار التنفيذ
            st.divider()
            st.subheader("💼 التنفيذ")
            
            if action == "BUY":
                if st.button(f"🚀 شراء {quantity} سهم", use_container_width=True, type="primary"):
                    msg = execute_ib_order("BUY", symbol, quantity, ib_host, ib_port)
                    if "✅" in msg:
                        st.success(msg)
                    else:
                        st.error(msg)
                    
            elif action == "SELL":
                if st.button(f"🔻 بيع {quantity} سهم", use_container_width=True, type="primary"):
                    msg = execute_ib_order("SELL", symbol, quantity, ib_host, ib_port)
                    if "✅" in msg:
                        st.success(msg)
                    else:
                        st.error(msg)
            else:
                st.info("⏸️ لا توجد صفقة للتنفيذ")
            
            if st.button("🗑️ مسح", use_container_width=True):
                for key in ['result', 'action', 'confidence']:
                    st.session_state.pop(key, None)
                st.rerun()
    
    # ===== معلومات =====
    with st.expander("ℹ️ معلومات"):
        st.markdown("""
        ### 🤖 بوت التداول بالذكاء الاصطناعي
        
        **المميزات:**
        - الاتصال بـ Interactive Brokers
        - تحليل فني باستخدام RSI, SMA
        - محرك ذكاء اصطناعي محلي (Random Forest)
        - دعم OpenAI GPT-4o
        - تنفيذ الأوامر تلقائياً
        - قائمة أسهم مفضلة قابلة للتخصيص
        
        **⚠️ تنبيه:** للاستخدام التعليمي فقط
        """)

# ==========================================
# التشغيل
# ==========================================

if __name__ == "__main__":
    main()
