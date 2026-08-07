# ==========================================
# ui/sidebar.py
# إدارة الشريط الجانبي - نسخة محسّنة
# ==========================================

import streamlit as st
from typing import Dict, Any, Optional
from core.config import config
from core.constants import PERIODS, INTERVALS, ANALYSIS_MODES


class Sidebar:
    """
    إدارة الشريط الجانبي للتطبيق
    """
    
    @staticmethod
    def render(engine) -> Dict[str, Any]:
        """
        عرض الشريط الجانبي
        
        Args:
            engine: محرك الذكاء الاصطناعي
        
        Returns:
            Dict: إعدادات المستخدم
        """
        with st.sidebar:
            st.header("⚙️ الإعدادات")
            
            # ==========================================
            # الأسهم المفضلة
            # ==========================================
            st.subheader("⭐ الأسهم المفضلة")
            
            if 'watchlist' not in st.session_state:
                st.session_state['watchlist'] = config.DEFAULT_WATCHLIST.copy()
            
            watchlist = st.session_state['watchlist']
            
            if watchlist:
                selected_symbol = st.selectbox(
                    "اختر من المفضلة:",
                    options=watchlist,
                    index=0,
                    key="symbol_selector"
                )
                
                col_select, col_remove = st.columns([1, 1])
                with col_select:
                    if st.button("📌 اختيار", use_container_width=True):
                        st.session_state['selected_symbol'] = selected_symbol
                        st.success(f"✅ تم اختيار {selected_symbol}")
                        st.rerun()
                
                current_symbol = st.session_state.get('selected_symbol', selected_symbol)
                st.info(f"📌 السهم الحالي: **{current_symbol}**")
            else:
                current_symbol = config.DEFAULT_SYMBOL
                st.warning("⚠️ القائمة فارغة")
            
            st.divider()
            
            # ==========================================
            # إدارة القائمة
            # ==========================================
            with st.expander("➕ إضافة رمز جديد", expanded=False):
                new_symbol = st.text_input(
                    "رمز السهم:",
                    placeholder="مثل: NVDA",
                    key="new_symbol_input"
                )
                
                if st.button("إضافة", use_container_width=True):
                    if new_symbol:
                        symbol_upper = new_symbol.upper()
                        if symbol_upper not in watchlist:
                            watchlist.append(symbol_upper)
                            st.success(f"✅ تم إضافة {symbol_upper}")
                            st.rerun()
                        else:
                            st.warning("⚠️ الرمز موجود مسبقاً")
            
            with st.expander("🗑️ حذف من المفضلة", expanded=False):
                if watchlist:
                    symbol_to_remove = st.selectbox(
                        "اختر رمزاً للحذف:",
                        options=watchlist,
                        index=None,
                        key="remove_selector"
                    )
                    if symbol_to_remove:
                        if st.button("🗑️ حذف", use_container_width=True):
                            watchlist.remove(symbol_to_remove)
                            st.success(f"✅ تم حذف {symbol_to_remove}")
                            st.rerun()
            
            st.caption(f"📊 إجمالي المفضلة: {len(watchlist)}")
            st.divider()
            
            # ==========================================
            # إعدادات البيانات
            # ==========================================
            st.subheader("⏱️ إعدادات البيانات")
            
            period = st.selectbox(
                "الفترة:",
                PERIODS,
                index=1,
                key="period_selector"
            )
            
            interval = st.selectbox(
                "الفاصل الزمني:",
                INTERVALS,
                index=2,
                key="interval_selector"
            )
            
            st.divider()
            
            # ==========================================
            # إعدادات التداول
            # ==========================================
            st.subheader("🔌 إعدادات التداول")
            
            ib_host = st.text_input(
                "IBKR Host:",
                value=config.IB_HOST,
                key="ib_host_input"
            )
            
            ib_port = st.number_input(
                "IBKR Port:",
                value=config.IB_PORT,
                step=1,
                key="ib_port_input"
            )
            
            api_key = st.text_input(
                "🔑 OpenAI/NewsAPI Key:",
                type="password",
                placeholder="اختياري للتحليل المتقدم",
                key="api_key_input"
            )
            
            quantity = st.number_input(
                "📊 الكمية الافتراضية:",
                value=config.DEFAULT_QUANTITY,
                step=1,
                min_value=1,
                key="quantity_input"
            )
            
            analysis_mode = st.radio(
                "🧠 وضع التحليل:",
                ANALYSIS_MODES,
                index=0,
                key="analysis_mode"
            )
            
            st.divider()
            
            # ==========================================
            # حالة المحرك
            # ==========================================
            st.subheader("📊 حالة المحرك")
            
            if engine.is_trained:
                st.success("✅ النموذج جاهز")
                
                # عرض معلومات النموذج
                if hasattr(engine, 'get_training_status'):
                    status = engine.get_training_status()
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("الميزات", status.get('feature_count', 0))
                    with col2:
                        st.metric("التنبؤات", status.get('prediction_count', 0))
                
                # عرض أهمية الميزات
                if hasattr(engine, 'get_feature_importance'):
                    importance = engine.get_feature_importance()
                    if importance:
                        with st.expander("📊 أهمية الميزات", expanded=False):
                            sorted_features = sorted(
                                importance.items(),
                                key=lambda x: x[1],
                                reverse=True
                            )[:5]
                            
                            for feature, value in sorted_features:
                                st.progress(
                                    value,
                                    text=f"{feature}: {value:.3f}"
                                )
            else:
                st.warning("⚠️ غير مدرب")
            
            st.divider()
            
            # ==========================================
            # زر تحديث
            # ==========================================
            if st.button("🔄 تحديث البيانات", use_container_width=True):
                st.rerun()
            
            return {
                'symbol': current_symbol,
                'period': period,
                'interval': interval,
                'ib_host': ib_host,
                'ib_port': ib_port,
                'api_key': api_key,
                'quantity': quantity,
                'analysis_mode': analysis_mode,
                'watchlist': watchlist
            }
