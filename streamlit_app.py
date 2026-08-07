# app.py
# ==========================================
# حل مشكلة event loop
# ==========================================
import asyncio
import nest_asyncio
import warnings

try:
    nest_asyncio.apply()
except:
    pass

try:
    loop = asyncio.get_event_loop()
except RuntimeError:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

warnings.filterwarnings('ignore')

# ==========================================
# استيراد المكتبات
# ==========================================
import streamlit as st
import pandas as pd
from core.config import config
from models.base_model import LocalAITradingEngine
from data.fetcher import DataFetcher
from ui.charts import ChartBuilder
from ui.sidebar import Sidebar
from brokers.ibkr_broker import IBKRBroker

# ==========================================
# إعدادات الصفحة
# ==========================================
st.set_page_config(
    page_title=config.APP_TITLE,
    page_icon=config.APP_ICON,
    layout=config.APP_LAYOUT,
    initial_sidebar_state="expanded"
)

# ==========================================
# تهيئة المحرك
# ==========================================
if 'engine' not in st.session_state:
    st.session_state['engine'] = LocalAITradingEngine()

engine = st.session_state['engine']

# ==========================================
# التطبيق الرئيسي
# ==========================================
def main():
    st.title(config.APP_TITLE)
    st.caption("📊 بوت تداول بالذكاء الاصطناعي مع Yahoo Finance و IBKR")
    
    # ===== الشريط الجانبي =====
    settings = Sidebar.render(engine)
    
    symbol = settings['symbol']
    period = settings['period']
    interval = settings['interval']
    quantity = settings['quantity']
    
    # ===== الأعمدة الرئيسية =====
    col1, col2 = st.columns([1.6, 1])
    
    # ===== العمود الأول: البيانات =====
    with col1:
        st.subheader("📊 البيانات والتحليل الفني")
        st.info(f"📌 السهم النشط: **{symbol}**")
        
        if st.button("🔄 جلب بيانات السوق", use_container_width=True):
            with st.spinner(f"جاري جلب بيانات {symbol}..."):
                df, error = DataFetcher.get_stock_data(symbol, period, interval)
                
                if error:
                    st.error(f"❌ {error}")
                else:
                    st.session_state['df'] = df
                    result, message = engine.train(df)
                    if result:
                        st.success(f"✅ {message}")
                    else:
                        st.warning(f"⚠️ {message}")
                    
                    st.success(f"✅ تم تحديث {len(df)} شمعة")
        
        if 'df' in st.session_state:
            df = st.session_state['df']
            
            # عرض السعر الحالي
            last_price = df['close'].iloc[-1]
            prev_price = df['close'].iloc[-2] if len(df) > 1 else last_price
            change = ((last_price - prev_price) / prev_price) * 100 if prev_price != 0 else 0
            
            col_m1, col_m2, col_m3, col_m4 = st.columns(4)
            with col_m1:
                st.metric("💰 السعر", f"${last_price:.2f}", f"{change:+.2f}%")
            with col_m2:
                st.metric("📊 RSI", f"{df['RSI'].iloc[-1]:.1f}")
            with col_m3:
                st.metric("📈 SMA 20", f"${df['SMA_20'].iloc[-1]:.2f}")
            with col_m4:
                st.metric("📉 SMA 50", f"${df['SMA_50'].iloc[-1]:.2f}")
            
            # الرسم البياني
            fig = ChartBuilder.create_candlestick_chart(df, symbol)
            st.plotly_chart(fig, use_container_width=True)
            
            # عرض الإشارة الفورية
            if engine.is_trained:
                action, confidence, reason = engine.predict(df)
                st.info(f"🚦 **إشارة فورية:** {ChartBuilder.display_signal(action, confidence)}")
                st.caption(f"📝 {reason}")
            
            # عرض البيانات
            with st.expander("📋 معاينة البيانات"):
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
    
    # ===== العمود الثاني: التحليل والتنفيذ =====
    with col2:
        st.subheader("🤖 التحليل والتنفيذ")
        
        if st.button("🧠 تحليل الفرصة", use_container_width=True, type="primary"):
            if 'df' not in st.session_state:
                st.warning("⚠️ يرجى جلب البيانات أولاً")
            else:
                df = st.session_state['df']
                
                with st.spinner("جاري التحليل..."):
                    action, confidence, reason = engine.predict(df)
                    
                    st.session_state['action'] = action
                    st.session_state['confidence'] = confidence
                    st.session_state['reason'] = reason
                    
                    st.success("✅ تم التحليل")
        
        # عرض النتيجة
        if 'action' in st.session_state:
            st.divider()
            
            action = st.session_state['action']
            confidence = st.session_state.get('confidence', 0)
            reason = st.session_state.get('reason', '')
            
            if action == "BUY":
                st.success(f"🟢 **توصية: شراء** (ثقة: {confidence}%)")
            elif action == "SELL":
                st.error(f"🔴 **توصية: بيع** (ثقة: {confidence}%)")
            else:
                st.warning(f"⏸️ **توصية: انتظار** (ثقة: {confidence}%)")
            
            st.text_area("📝 التفاصيل:", reason, height=100)
            
            st.divider()
            st.subheader("💼 تنفيذ الصفقة")
            
            if action == "BUY":
                col_b1, col_b2 = st.columns(2)
                with col_b1:
                    if st.button(f"🚀 شراء {quantity} سهم", use_container_width=True, type="primary"):
                        st.info("✅ تم تنفيذ أمر الشراء (محاكاة)")
                with col_b2:
                    if st.button("📊 محاكاة", use_container_width=True):
                        df = st.session_state['df']
                        last_price = df['close'].iloc[-1]
                        st.info(f"📊 محاكاة: شراء {quantity} سهم بسعر ${last_price:.2f}")
            
            elif action == "SELL":
                col_s1, col_s2 = st.columns(2)
                with col_s1:
                    if st.button(f"🔻 بيع {quantity} سهم", use_container_width=True, type="primary"):
                        st.info("✅ تم تنفيذ أمر البيع (محاكاة)")
                with col_s2:
                    if st.button("📊 محاكاة", use_container_width=True):
                        df = st.session_state['df']
                        last_price = df['close'].iloc[-1]
                        st.info(f"📊 محاكاة: بيع {quantity} سهم بسعر ${last_price:.2f}")
            
            else:
                st.info("⏸️ لا توجد صفقة للتنفيذ")
                
                # عرض تفاصيل إضافية
                if 'df' in st.session_state:
                    df = st.session_state['df']
                    latest = df.iloc[-1]
                    with st.expander("📈 تفاصيل السوق"):
                        col_d1, col_d2 = st.columns(2)
                        with col_d1:
                            st.metric("RSI", f"{latest['RSI']:.1f}")
                            st.metric("SMA 20", f"${latest['SMA_20']:.2f}")
                        with col_d2:
                            diff_sma = ((latest['close'] - latest['SMA_20']) / latest['SMA_20']) * 100
                            st.metric("البعد عن SMA20", f"{diff_sma:+.1f}%")
                            st.metric("حجم التداول", f"{latest['volume']:,.0f}")
            
            # زر مسح النتائج
            if st.button("🗑️ مسح النتائج", use_container_width=True):
                for key in ['action', 'confidence', 'reason']:
                    st.session_state.pop(key, None)
                st.rerun()
    
    # ===== معلومات إضافية =====
    with st.expander("ℹ️ معلومات عن البوت"):
        st.markdown("""
        ### 🤖 بوت التداول بالذكاء الاصطناعي
        
        **المميزات:**
        - 📊 **جلب البيانات**: من Yahoo Finance (مجاني وسريع)
        - 🧠 **محرك محلي**: Random Forest Classifier مع ميزات فنية
        - 🤖 **OpenAI**: دعم GPT-4o-mini للتحليل المتقدم (اختياري)
        - 🔌 **IBKR**: تنفيذ الأوامر عبر Interactive Brokers (اختياري)
        - ⭐ **قائمة مفضلة**: إدارة الأسهم المفضلة
        - 📈 **قوائم السوق**: الأكثر ارتفاعاً/انخفاضاً
        
        **⚠️ تنبيهات:**
        - هذا البوت **للأغراض التعليمية فقط**
        - استخدم الحساب التجريبي (Paper Trading) أولاً
        - لا تخاطر بأكثر مما يمكنك تحمل خسارته
        - راجع أداء النموذج باستمرار
        
        **📚 المتطلبات:**
        - Python 3.11+
        - المكتبات: streamlit, pandas, numpy, plotly, yfinance, ta, scikit-learn
        - (اختياري) IBKR: ib_async أو ib_insync
        """)

# ==========================================
# تشغيل التطبيق
# ==========================================
if __name__ == "__main__":
    main()
