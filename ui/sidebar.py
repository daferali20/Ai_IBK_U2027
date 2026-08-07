# ui/sidebar.py
import streamlit as st
from core.config import config
from core.constants import PERIODS, INTERVALS, ANALYSIS_MODES

class Sidebar:
    """إدارة الشريط الجانبي"""
    
    @staticmethod
    def render(engine):
        """عرض الشريط الجانبي"""
        with st.sidebar:
            st.header("⚙️ الإعدادات")
            
            # الأسهم المفضلة
            st.subheader("⭐ الأسهم المفضلة")
            
            if 'watchlist' not in st.session_state:
                st.session_state['watchlist'] = config.DEFAULT_WATCHLIST.copy()
            
            watchlist = st.session_state['watchlist']
            
            if watchlist:
                selected_symbol = st.selectbox(
                    "اختر من المفضلة:",
                    options=watchlist,
                    index=0
                )
                
                if st.button("📌 اختيار السهم", use_container_width=True):
                    st.session_state['selected_symbol'] = selected_symbol
                    st.success(f"✅ تم اختيار {selected_symbol}")
                
                current_symbol = st.session_state.get('selected_symbol', selected_symbol)
                st.info(f"📌 السهم الحالي: **{current_symbol}**")
            else:
                current_symbol = config.DEFAULT_SYMBOL
            
            st.divider()
            
            # إضافة رمز
            with st.expander("➕ إضافة رمز جديد"):
                new_symbol = st.text_input("رمز السهم:", placeholder="مثل: NVDA")
                col_add1, col_add2 = st.columns(2)
                with col_add1:
                    if st.button("إضافة", use_container_width=True):
                        if new_symbol and new_symbol.upper() not in watchlist:
                            watchlist.append(new_symbol.upper())
                            st.success(f"✅ تم إضافة {new_symbol.upper()}")
                            st.rerun()
                        elif new_symbol:
                            st.warning("⚠️ الرمز موجود مسبقاً")
            
            # حذف رمز
            with st.expander("🗑️ حذف من المفضلة"):
                if watchlist:
                    symbol_to_remove = st.selectbox(
                        "اختر رمزاً للحذف:",
                        options=watchlist,
                        index=None
                    )
                    if symbol_to_remove and st.button("حذف", use_container_width=True):
                        watchlist.remove(symbol_to_remove)
                        st.success(f"✅ تم حذف {symbol_to_remove}")
                        st.rerun()
            
            st.caption(f"📊 إجمالي المفضلة: {len(watchlist)}")
            st.divider()
            
            # إعدادات البيانات
            st.subheader("⏱️ إعدادات البيانات")
            period = st.selectbox("الفترة:", PERIODS, index=1)
            interval = st.selectbox("الفاصل الزمني:", INTERVALS, index=2)
            
            st.divider()
            
            # إعدادات أخرى
            st.subheader("🔌 الإعدادات")
            ib_host = st.text_input("IBKR Host", config.IB_HOST)
            ib_port = st.number_input("IBKR Port", value=config.IB_PORT)
            api_key = st.text_input("🔑 OpenAI Key (اختياري)", type="password")
            quantity = st.number_input("📊 الكمية", value=config.DEFAULT_QUANTITY, step=1)
            
            analysis_mode = st.radio(
                "🧠 وضع التحليل",
                ANALYSIS_MODES,
                index=0
            )
            
            st.divider()
            
            # حالة المحرك
            st.subheader("📊 حالة المحرك")
            if engine.is_trained:
                st.success(f"✅ النموذج جاهز (دقة: {engine.accuracy*100:.1f}%)")
                if engine.feature_importance is not None:
                    with st.expander("📊 أهمية الميزات"):
                        st.dataframe(engine.feature_importance.head(5))
            else:
                st.warning("⚠️ غير مدرب")
            
            return {
                'symbol': current_symbol,
                'period': period,
                'interval': interval,
                'ib_host': ib_host,
                'ib_port': ib_port,
                'api_key': api_key,
                'quantity': quantity,
                'analysis_mode': analysis_mode
            }
