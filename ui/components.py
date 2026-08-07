# ui/components.py
"""
مكونات واجهة المستخدم القابلة لإعادة الاستخدام
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from typing import Optional, Tuple, List, Dict, Any

# ==========================================
# مكونات العرض الأساسية
# ==========================================

class UIComponents:
    """مكونات واجهة المستخدم"""
    
    @staticmethod
    def display_metric_with_delta(
        label: str,
        value: Any,
        delta: Optional[Any] = None,
        delta_color: str = "normal",
        help_text: Optional[str] = None
    ):
        """
        عرض مقياس مع تغير
        """
        return st.metric(
            label=label,
            value=value,
            delta=delta,
            delta_color=delta_color,
            help=help_text
        )
    
    @staticmethod
    def display_price_card(
        symbol: str,
        price: float,
        change: float,
        volume: Optional[int] = None
    ):
        """
        عرض بطاقة السعر
        """
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                label=f"💰 {symbol}",
                value=f"${price:,.2f}",
                delta=f"{change:+.2f}%"
            )
        
        with col2:
            if volume:
                st.metric(
                    label="📊 حجم التداول",
                    value=f"{volume:,.0f}"
                )
        
        with col3:
            if change > 0:
                st.success(f"🟢 تغير إيجابي: {change:+.2f}%")
            elif change < 0:
                st.error(f"🔴 تغير سلبي: {change:+.2f}%")
            else:
                st.info(f"⏸️ لا تغير: {change:+.2f}%")
    
    @staticmethod
    def display_signal_badge(
        action: str,
        confidence: float,
        size: str = "large"
    ):
        """
        عرض شارة الإشارة
        """
        if action == "BUY":
            color = "green"
            icon = "🟢"
            label = "شراء"
        elif action == "SELL":
            color = "red"
            icon = "🔴"
            label = "بيع"
        else:
            color = "orange"
            icon = "⏸️"
            label = "انتظار"
        
        if size == "large":
            st.markdown(f"""
            <div style='
                padding: 20px;
                background-color: {color}22;
                border-radius: 10px;
                border: 2px solid {color};
                text-align: center;
            '>
                <h1 style='margin: 0; font-size: 2.5em;'>{icon}</h1>
                <h2 style='margin: 5px 0; color: {color};'>{label}</h2>
                <p style='margin: 0; font-size: 1.2em;'>الثقة: {confidence}%</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style='
                padding: 10px;
                background-color: {color}22;
                border-radius: 5px;
                border: 1px solid {color};
                text-align: center;
                display: inline-block;
            '>
                <span style='font-size: 1.5em;'>{icon}</span>
                <span style='font-weight: bold; color: {color};'>{label}</span>
                <span style='font-size: 0.8em;'>({confidence}%)</span>
            </div>
            """, unsafe_allow_html=True)

# ==========================================
# مكونات الجداول
# ==========================================

class TableComponents:
    """مكونات الجداول"""
    
    @staticmethod
    def display_data_table(
        df: pd.DataFrame,
        columns: List[str],
        title: str = "📋 البيانات",
        height: int = 300,
        use_container_width: bool = True
    ):
        """
        عرض جدول بيانات بتنسيق جميل
        """
        with st.expander(title, expanded=False):
            # تنسيق الأعمدة الرقمية
            formatted_df = df[columns].copy()
            
            # تنسيق الأعمدة
            format_dict = {}
            for col in formatted_df.columns:
                if col in ['close', 'open', 'high', 'low', 'SMA_20', 'SMA_50']:
                    format_dict[col] = '${:.2f}'
                elif col == 'volume':
                    format_dict[col] = '{:,.0f}'
                elif col in ['RSI', 'rsi', 'volatility', 'change']:
                    format_dict[col] = '{:.1f}'
                elif 'change' in col or 'return' in col:
                    format_dict[col] = '{:+.2f}%'
            
            if format_dict:
                st.dataframe(
                    formatted_df.tail(10).style.format(format_dict),
                    height=height,
                    use_container_width=use_container_width
                )
            else:
                st.dataframe(
                    formatted_df.tail(10),
                    height=height,
                    use_container_width=use_container_width
                )
    
    @staticmethod
    def display_watchlist_table(
        watchlist: List[str],
        prices: Dict[str, float] = None,
        changes: Dict[str, float] = None
    ):
        """
        عرض قائمة المفضلة كجدول
        """
        if not watchlist:
            st.info("📭 القائمة فارغة")
            return
        
        data = []
        for symbol in watchlist:
            row = {'Symbol': symbol}
            if prices and symbol in prices:
                row['Price'] = f"${prices[symbol]:.2f}"
            else:
                row['Price'] = '---'
            
            if changes and symbol in changes:
                change = changes[symbol]
                row['Change'] = f"{change:+.2f}%"
                row['Status'] = '🟢' if change > 0 else ('🔴' if change < 0 else '⏸️')
            else:
                row['Change'] = '---'
                row['Status'] = '⏸️'
            
            data.append(row)
        
        df = pd.DataFrame(data)
        st.dataframe(df, use_container_width=True, hide_index=True)

# ==========================================
# مكونات الرسوم البيانية الإضافية
# ==========================================

class ChartComponents:
    """مكونات الرسوم البيانية"""
    
    @staticmethod
    def display_metrics_row(
        metrics: List[Tuple[str, Any, Optional[str]]],
        columns: int = 4
    ):
        """
        عرض صف من المقاييس
        """
        cols = st.columns(columns)
        for i, (label, value, delta) in enumerate(metrics):
            with cols[i % columns]:
                if delta:
                    st.metric(label, value, delta)
                else:
                    st.metric(label, value)
    
    @staticmethod
    def display_signal_history(
        signals: List[Dict[str, Any]],
        max_items: int = 20
    ):
        """
        عرض تاريخ الإشارات
        """
        if not signals:
            st.info("📭 لا توجد إشارات سابقة")
            return
        
        # عرض آخر الإشارات
        recent = signals[-max_items:]
        
        data = []
        for signal in recent:
            action = signal.get('action', 'HOLD')
            confidence = signal.get('confidence', 0)
            timestamp = signal.get('timestamp', '')
            symbol = signal.get('symbol', '')
            
            if action == 'BUY':
                emoji = '🟢'
            elif action == 'SELL':
                emoji = '🔴'
            else:
                emoji = '⏸️'
            
            data.append({
                'الوقت': timestamp,
                'السهم': symbol,
                'الإشارة': f"{emoji} {action}",
                'الثقة': f"{confidence}%"
            })
        
        df = pd.DataFrame(data)
        st.dataframe(df, use_container_width=True, hide_index=True)

# ==========================================
# مكونات التحليل
# ==========================================

class AnalysisComponents:
    """مكونات التحليل"""
    
    @staticmethod
    def display_analysis_result(
        action: str,
        confidence: float,
        reason: str,
        details: Optional[Dict[str, Any]] = None
    ):
        """
        عرض نتيجة التحليل
        """
        # عرض الإشارة
        if action == "BUY":
            st.success(f"🟢 **توصية: شراء** (ثقة: {confidence}%)")
        elif action == "SELL":
            st.error(f"🔴 **توصية: بيع** (ثقة: {confidence}%)")
        else:
            st.warning(f"⏸️ **توصية: انتظار** (ثقة: {confidence}%)")
        
        # عرض التفاصيل
        st.text_area("📝 التفاصيل:", reason, height=100)
        
        # عرض تفاصيل إضافية
        if details:
            with st.expander("📊 تفاصيل إضافية"):
                cols = st.columns(2)
                for i, (key, value) in enumerate(details.items()):
                    with cols[i % 2]:
                        st.metric(key, value)
    
    @staticmethod
    def display_model_info(
        is_trained: bool,
        accuracy: Optional[float] = None,
        features: Optional[List[str]] = None,
        feature_importance: Optional[pd.DataFrame] = None
    ):
        """
        عرض معلومات النموذج
        """
        with st.expander("🧠 معلومات النموذج", expanded=False):
            if is_trained:
                st.success("✅ النموذج مدرب")
                if accuracy:
                    st.metric("دقة النموذج", f"{accuracy*100:.1f}%")
                
                if feature_importance is not None:
                    st.subheader("📊 أهمية الميزات")
                    st.dataframe(feature_importance.head(5), use_container_width=True)
            else:
                st.warning("⚠️ النموذج غير مدرب")
            
            if features:
                st.caption(f"📋 عدد الميزات: {len(features)}")
                with st.expander("عرض الميزات"):
                    st.write(features)

# ==========================================
# مكونات الإعدادات
# ==========================================

class SettingsComponents:
    """مكونات الإعدادات"""
    
    @staticmethod
    def display_connection_status(
        name: str,
        connected: bool,
        details: Optional[str] = None
    ):
        """
        عرض حالة الاتصال
        """
        if connected:
            st.success(f"✅ {name}: متصل")
        else:
            st.error(f"❌ {name}: غير متصل")
        
        if details:
            st.caption(details)
    
    @staticmethod
    def display_trading_controls(
        quantity: int,
        action: str,
        symbol: str,
        on_buy: callable = None,
        on_sell: callable = None,
        on_simulate: callable = None
    ):
        """
        عرض أزرار التحكم في التداول
        """
        st.subheader("💼 تنفيذ الصفقة")
        
        if action == "BUY":
            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button(f"🚀 شراء {quantity} سهم", use_container_width=True, type="primary"):
                    if on_buy:
                        on_buy()
            with col2:
                if st.button("📊 محاكاة شراء", use_container_width=True):
                    if on_simulate:
                        on_simulate()
            with col3:
                st.caption(f"📊 {symbol}")
        
        elif action == "SELL":
            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button(f"🔻 بيع {quantity} سهم", use_container_width=True, type="primary"):
                    if on_sell:
                        on_sell()
            with col2:
                if st.button("📊 محاكاة بيع", use_container_width=True):
                    if on_simulate:
                        on_simulate()
            with col3:
                st.caption(f"📊 {symbol}")
        
        else:
            st.info("⏸️ لا توجد صفقة للتنفيذ")
        
        # عرض كمية التداول
        st.caption(f"📊 الكمية المحددة: {quantity} سهم")

# ==========================================
# مكونات مساعدة
# ==========================================

class HelpComponents:
    """مكونات مساعدة"""
    
    @staticmethod
    def display_info_box(
        title: str,
        content: str,
        icon: str = "ℹ️"
    ):
        """
        عرض صندوق معلومات
        """
        with st.expander(f"{icon} {title}", expanded=False):
            st.markdown(content)
    
    @staticmethod
    def display_warning_box(
        message: str,
        icon: str = "⚠️"
    ):
        """
        عرض صندوق تحذير
        """
        st.warning(f"{icon} {message}")
    
    @staticmethod
    def display_success_box(
        message: str,
        icon: str = "✅"
    ):
        """
        عرض صندوق نجاح
        """
        st.success(f"{icon} {message}")
    
    @staticmethod
    def display_error_box(
        message: str,
        icon: str = "❌"
    ):
        """
        عرض صندوق خطأ
        """
        st.error(f"{icon} {message}")
    
    @staticmethod
    def display_progress_bar(
        value: float,
        max_value: float = 100,
        label: str = "التقدم"
    ):
        """
        عرض شريط التقدم
        """
        progress = min(value / max_value, 1.0)
        st.progress(progress)
        st.caption(f"{label}: {value:.1f}%")

# ==========================================
# مكونات متقدمة
# ==========================================

class AdvancedComponents:
    """مكونات متقدمة"""
    
    @staticmethod
    def display_trading_summary(
        stats: Dict[str, Any]
    ):
        """
        عرض ملخص التداول
        """
        with st.expander("📊 ملخص التداول", expanded=False):
            cols = st.columns(3)
            
            with cols[0]:
                st.metric("إجمالي الصفقات", stats.get('total_trades', 0))
                st.metric("الصفقات الناجحة", stats.get('winning_trades', 0))
            
            with cols[1]:
                st.metric("نسبة النجاح", f"{stats.get('win_rate', 0):.1f}%")
                st.metric("إجمالي الربح", f"${stats.get('total_profit', 0):,.2f}")
            
            with cols[2]:
                st.metric("أكبر ربح", f"${stats.get('max_profit', 0):,.2f}")
                st.metric("أكبر خسارة", f"${stats.get('max_loss', 0):,.2f}")
    
    @staticmethod
    def display_performance_chart(
        data: List[float],
        labels: List[str],
        title: str = "أداء النموذج"
    ):
        """
        عرض رسم بياني للأداء
        """
        import plotly.graph_objects as go
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=labels,
            y=data,
            mode='lines+markers',
            name='الأداء'
        ))
        
        fig.update_layout(
            title=title,
            template='plotly_dark',
            height=300
        )
        
        st.plotly_chart(fig, use_container_width=True)

# ==========================================
# تصدير المكونات للاستخدام
# ==========================================

__all__ = [
    'UIComponents',
    'TableComponents',
    'ChartComponents',
    'AnalysisComponents',
    'SettingsComponents',
    'HelpComponents',
    'AdvancedComponents'
]
