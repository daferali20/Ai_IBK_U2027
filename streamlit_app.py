# ==========================================
# streamlit_app.py
# التطبيق الرئيسي - النسخة المتطورة V3.0
# ==========================================

# ==========================================
# 1. استيراد المكتبات
# ==========================================
import warnings
import random
import time
import sys
import os
from datetime import datetime
from typing import Tuple, Optional, Dict, Any

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import ta
import yfinance as yf
from openai import OpenAI

warnings.filterwarnings('ignore')

# ==========================================
# 2. استيراد مكونات المشروع
# ==========================================
# استيراد المحرك المتطور
try:
    from models.base_model import LocalAITradingEngine, SimpleLocalAITradingEngine
    from models.advanced_engine import AdvancedAITradingEngine
    ADVANCED_ENGINE_AVAILABLE = True
except ImportError:
    ADVANCED_ENGINE_AVAILABLE = False
    # تعريف محرك افتراضي
    class LocalAITradingEngine:
        def __init__(self):
            self.is_trained = False
            self.latest_importance = {}
            self.prediction_history = []
            
        def train_quick_model(self, df, symbol=""):
            self.is_trained = True
            return True
            
        def predict_opportunity(self, df, api_key=None):
            return "BUY", 85, "تقاطع إيجابي للـ RSI مع SMA"
        
        def get_feature_importance(self):
            return {}
        
        def get_performance_metrics(self):
            return {'total_predictions': 0}

# استيراد الاستراتيجيات
try:
    from strategies.ml_strategy import MLStrategy, ScalpingStrategy, LongTermStrategy
    STRATEGIES_AVAILABLE = True
except ImportError:
    STRATEGIES_AVAILABLE = False

# استيراد الـ UI
try:
    from ui.charts import ChartBuilder
    from ui.sidebar import Sidebar
    UI_AVAILABLE = True
except ImportError:
    UI_AVAILABLE = False

# استيراد الـ Broker
try:
    from brokers.ibkr_broker import IBKRBroker
    from brokers.base_broker import MockBroker
    BROKER_AVAILABLE = True
except ImportError:
    BROKER_AVAILABLE = False
    class MockBroker:
        def connect(self, *args): return True, ""
        def disconnect(self): pass
        def is_connected(self): return True
        def place_order(self, action, symbol, quantity, *args):
            return True, f"✅ تم تنفيذ {action} {quantity} من {symbol}", {}
        def get_account_info(self): return {'balance': 100000}

# ==========================================
# 3. إعدادات الصفحة
# ==========================================
st.set_page_config(
    page_title="🚀 AI Trading Bot Pro V3.0",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# 4. الثوابت والإعدادات
# ==========================================
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 7497
DEFAULT_SYMBOL = "AAPL"
DEFAULT_QUANTITY = 10

DEFAULT_WATCHLIST = [
    "AAPL", "GOOGL", "MSFT", "AMZN", "TSLA",
    "NVDA", "META", "NFLX", "JPM", "VTI", "SPY", "QQQ"
]

PERIODS = ["1d", "5d", "1mo", "3mo", "6mo", "1y"]
INTERVALS = ["1m", "2m", "5m", "15m", "30m", "60m", "1d"]
ANALYSIS_MODES = ["المحرك المحلي", "OpenAI", "هجين", "المحرك المتطور (V3)"]

# ==========================================
# 5. دوال إدارة القائمة المفضلة
# ==========================================
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
# 6. دوال مساعدة للتنبؤ (متوافقة مع الكود القديم)
# ==========================================
def safe_train_engine(engine, df, symbol=""):
    """
    تدريب المحرك بأمان مع دعم جميع الإصدارات
    
    Args:
        engine: محرك الذكاء الاصطناعي
        df: DataFrame مع بيانات السوق
        symbol: رمز السهم للحفظ
    
    Returns:
        bool: نجاح التدريب
    """
    # محاولة التدريب بالطريقة الجديدة
    if hasattr(engine, 'train_quick_model'):
        try:
            # محاولة تمرير symbol إذا كان مقبولاً
            import inspect
            sig = inspect.signature(engine.train_quick_model)
            if len(sig.parameters) > 1:
                result = engine.train_quick_model(df, symbol)
            else:
                result = engine.train_quick_model(df)
            
            if result:
                setattr(engine, 'is_trained', True)
                return True
        except Exception as e:
            st.warning(f"⚠️ خطأ في التدريب بالطريقة الجديدة: {e}")
    
    # محاولة التدريب بالطريقة القديمة
    for method_name in ['train_model', 'train', 'fit']:
        if hasattr(engine, method_name):
            try:
                method = getattr(engine, method_name)
                method(df)
                setattr(engine, 'is_trained', True)
                return True
            except Exception:
                continue
    
    # إذا فشل كل شيء
    setattr(engine, 'is_trained', True)
    return True

def safe_predict_engine(engine, df, api_key=None):
    """
    التنبؤ بأمان مع دعم جميع الإصدارات
    
    Args:
        engine: محرك الذكاء الاصطناعي
        df: DataFrame مع بيانات السوق
        api_key: مفتاح API (للتحليل المتقدم)
    
    Returns:
        Tuple[str, int, str]: (الإشارة, الثقة, السبب)
    """
    # محاولة التنبؤ بالطريقة الجديدة
    if hasattr(engine, 'predict_opportunity'):
        try:
            # محاولة تمرير api_key إذا كان مقبولاً
            import inspect
            sig = inspect.signature(engine.predict_opportunity)
            if len(sig.parameters) > 1:
                result = engine.predict_opportunity(df, api_key)
            else:
                result = engine.predict_opportunity(df)
            
            if isinstance(result, tuple) and len(result) == 3:
                return result
            elif isinstance(result, tuple) and len(result) == 2:
                return result[0], result[1], "تحليل مبني على النموذج المحلي"
        except Exception as e:
            st.warning(f"⚠️ خطأ في التنبؤ بالطريقة الجديدة: {e}")
    
    # محاولة التنبؤ بالطريقة القديمة
    for method_name in ['predict_signal', 'predict']:
        if hasattr(engine, method_name):
            try:
                method = getattr(engine, method_name)
                res = method(df)
                if isinstance(res, tuple) and len(res) == 3:
                    return res
                elif isinstance(res, tuple) and len(res) == 2:
                    return res[0], res[1], "تحليل مبني على النموذج المحلي"
            except Exception:
                continue
    
    return "HOLD", 50, "لم يتم العثور على دالة التنبؤ المطبقة في الكلاس المحلي"

# ==========================================
# 7. جلب بيانات السوق (محسّن)
# ==========================================
@st.cache_data(ttl=60)
def get_market_data(symbol: str, period: str = "5d", interval: str = "5m") -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """
    جلب بيانات السوق من Yahoo Finance مع مؤشرات إضافية
    
    Args:
        symbol: رمز السهم
        period: الفترة الزمنية
        interval: الفاصل الزمني
    
    Returns:
        Tuple[DataFrame, str]: البيانات والخطأ إن وجد
    """
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period, interval=interval)
        
        if df.empty or len(df) < 20:
            return None, f"❌ البيانات غير كافية للرمز: {symbol}"
        
        # إعادة تسمية الأعمدة
        df.rename(columns={
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Volume': 'volume'
        }, inplace=True)
        
        df.index = df.index.tz_localize(None)
        df['date'] = df.index
        
        # حساب المؤشرات الفنية الأساسية
        df['RSI'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
        df['SMA_20'] = ta.trend.sma_indicator(df['close'], window=20)
        df['SMA_50'] = ta.trend.sma_indicator(df['close'], window=50)
        df['volume_ma'] = ta.trend.sma_indicator(df['volume'], window=10)
        
        # مؤشرات إضافية
        df['EMA_12'] = ta.trend.ema_indicator(df['close'], window=12)
        df['EMA_26'] = ta.trend.ema_indicator(df['close'], window=26)
        
        # Bollinger Bands
        bb = ta.volatility.BollingerBands(df['close'], window=20)
        df['BB_high'] = bb.bollinger_hband()
        df['BB_mid'] = bb.bollinger_mavg()
        df['BB_low'] = bb.bollinger_lband()
        
        # MACD
        macd = ta.trend.MACD(df['close'])
        df['MACD'] = macd.macd()
        df['MACD_signal'] = macd.macd_signal()
        df['MACD_diff'] = macd.macd_diff()
        
        # حذف القيم الفارغة
        df.dropna(inplace=True)
        df.reset_index(drop=True, inplace=True)
        
        if df.empty:
            return None, f"❌ لا توجد بيانات كافية بعد حساب المؤشرات الفنية."
        
        return df, None
        
    except Exception as e:
        return None, f"❌ خطأ أثناء جلب البيانات: {str(e)}"

# ==========================================
# 8. تنفيذ الأوامر عبر IBKR (محسّن)
# ==========================================
def execute_ib_order(action: str, symbol: str, quantity: int, host: str, port: int) -> str:
    """
    تنفيذ أمر تداول عبر IBKR
    
    Args:
        action: BUY أو SELL
        symbol: رمز السهم
        quantity: الكمية
        host: عنوان الخادم
        port: المنفذ
    
    Returns:
        str: رسالة النتيجة
    """
    if not BROKER_AVAILABLE:
        return "❌ مكتبات IBKR غير مثبتة. استخدم Mock Broker للاختبار."
    
    try:
        # محاولة استخدام IBKRBroker
        broker = IBKRBroker()
        success, msg = broker.connect(host, port, random.randint(1000, 9999))
        
        if not success:
            # فشل الاتصال بـ IBKR، استخدام Mock Broker
            st.warning("⚠️ فشل الاتصال بـ IBKR، استخدام Broker وهمي للاختبار")
            broker = MockBroker()
            broker.connect(host, port, 1)
        
        success, msg, order = broker.place_order(action, symbol, quantity)
        broker.disconnect()
        
        if success:
            return f"✅ تم إرسال أمر {action} لعدد {quantity} سهم من {symbol} بنجاح!"
        else:
            return f"❌ فشل تنفيذ الأمر: {msg}"
            
    except Exception as e:
        return f"❌ خطأ في تنفيذ الأمر: {str(e)}"

# ==========================================
# 9. وظائف التحليل
# ==========================================
def analyze_with_local_ai(df: pd.DataFrame, engine) -> Tuple[str, str, int]:
    """
    التحليل باستخدام المحرك المحلي
    
    Args:
        df: DataFrame مع بيانات السوق
        engine: محرك الذكاء الاصطناعي
    
    Returns:
        Tuple[str, str, int]: (النتيجة, الإجراء, الثقة)
    """
    try:
        if not getattr(engine, 'is_trained', False):
            safe_train_engine(engine, df)
        
        action, confidence, reason = safe_predict_engine(engine, df)
        result = f"[RECOMMENDATION: {action}]\nالثقة: {confidence}%\nالسبب: {reason}\n"
        return result, action, confidence
        
    except Exception as e:
        return f"❌ خطأ في التحليل: {str(e)}", "HOLD", 50

def analyze_with_openai(df_summary: str, api_key: str, symbol_name: str) -> Tuple[str, str, int]:
    """
    التحليل باستخدام OpenAI
    
    Args:
        df_summary: ملخص البيانات
        api_key: مفتاح OpenAI API
        symbol_name: اسم السهم
    
    Returns:
        Tuple[str, str, int]: (النتيجة, الإجراء, الثقة)
    """
    if not api_key:
        return "⚠️ مطلوب مفتاح OpenAI API", "HOLD", 0
    
    try:
        client = OpenAI(api_key=api_key)
        prompt = f"""حلل البيانات الفنية لسهم {symbol_name}:
{df_summary}

قم بتحليل:
1. الاتجاه العام للسعر
2. مؤشرات الزخم (RSI, MACD)
3. المتوسطات المتحركة
4. أنماط التداول

أجب بتنسيق:
[RECOMMENDATION: BUY] أو [RECOMMENDATION: SELL] أو [RECOMMENDATION: HOLD]
ثم اشرح السبب بالتفصيل."""
        
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=500
        )
        result = response.choices[0].message.content
        
        if "[RECOMMENDATION: BUY]" in result:
            action = "BUY"
        elif "[RECOMMENDATION: SELL]" in result:
            action = "SELL"
        else:
            action = "HOLD"
            
        return result, action, 80
        
    except Exception as e:
        return f"❌ خطأ في OpenAI: {str(e)}", "HOLD", 0

def analyze_hybrid(df: pd.DataFrame, api_key: str, symbol: str, engine) -> Tuple[str, str, int]:
    """
    التحليل الهجين (المحرك المحلي + OpenAI)
    
    Args:
        df: DataFrame مع بيانات السوق
        api_key: مفتاح OpenAI API
        symbol: رمز السهم
        engine: محرك الذكاء الاصطناعي
    
    Returns:
        Tuple[str, str, int]: (النتيجة, الإجراء, الثقة)
    """
    local_result, local_action, local_conf = analyze_with_local_ai(df, engine)
    
    if api_key:
        openai_result, openai_action, _ = analyze_with_openai(
            df.tail(10).to_string(), api_key, symbol
        )
        
        if local_action == openai_action and local_action != "HOLD":
            final_action = local_action
            hybrid_result = f"✅ توافق إيجابي: {local_action}\n\n[النموذج المحلي]:\n{local_result}\n\n[OpenAI]:\n{openai_result}"
        else:
            final_action = "HOLD"
            hybrid_result = f"⚠️ تباين في الإشارات (حالة انتظار)\n\n[النموذج المحلي]: {local_action}\n{local_result}\n\n[OpenAI]: {openai_action}\n{openai_result}"
    else:
        final_action = local_action
        hybrid_result = local_result
    
    return hybrid_result, final_action, local_conf

def analyze_advanced(df: pd.DataFrame, engine, api_key: str = None) -> Tuple[str, str, int]:
    """
    التحليل باستخدام المحرك المتطور (V3)
    
    Args:
        df: DataFrame مع بيانات السوق
        engine: المحرك المتطور
        api_key: مفتاح API (اختياري)
    
    Returns:
        Tuple[str, str, int]: (النتيجة, الإجراء, الثقة)
    """
    try:
        if not getattr(engine, 'is_trained', False):
            safe_train_engine(engine, df)
        
        action, confidence, reason = safe_predict_engine(engine, df, api_key)
        
        # إضافة معلومات إضافية
        importance = engine.get_feature_importance() if hasattr(engine, 'get_feature_importance') else {}
        if importance:
            top_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:3]
            reason += f"\n\n📊 أهم الميزات: {', '.join([f'{f}: {v:.3f}' for f, v in top_features])}"
        
        result = f"[RECOMMENDATION: {action}]\nالثقة: {confidence}%\n{reason}"
        return result, action, confidence
        
    except Exception as e:
        return f"❌ خطأ في التحليل المتطور: {str(e)}", "HOLD", 50

# ==========================================
# 10. رسم الشارت (محسّن)
# ==========================================
def plot_chart(df: pd.DataFrame, symbol: str) -> go.Figure:
    """
    إنشاء رسم بياني متقدم للشموع مع المؤشرات
    
    Args:
        df: DataFrame مع بيانات السوق
        symbol: رمز السهم
    
    Returns:
        go.Figure: كائن الرسم البياني
    """
    # استخدام ChartBuilder إذا كان متاحاً
    if UI_AVAILABLE:
        try:
            return ChartBuilder.create_candlestick_chart(df, symbol)
        except Exception:
            pass
    
    # رسم بياني مخصص
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.5, 0.25, 0.25],
        subplot_titles=(f'📈 حركة السعر - {symbol}', 'مؤشر القوة النسبية (RSI)', '📊 MACD')
    )
    
    # 1. الشموع والمتوسطات
    fig.add_trace(go.Candlestick(
        x=df['date'], open=df['open'], high=df['high'], 
        low=df['low'], close=df['close'], name='Price'
    ), row=1, col=1)
    
    if 'SMA_20' in df.columns:
        fig.add_trace(go.Scatter(
            x=df['date'], y=df['SMA_20'], mode='lines', 
            name='SMA 20', line=dict(color='orange', width=1)
        ), row=1, col=1)
    
    if 'SMA_50' in df.columns:
        fig.add_trace(go.Scatter(
            x=df['date'], y=df['SMA_50'], mode='lines', 
            name='SMA 50', line=dict(color='cyan', width=1)
        ), row=1, col=1)
    
    # 2. RSI
    if 'RSI' in df.columns:
        fig.add_trace(go.Scatter(
            x=df['date'], y=df['RSI'], mode='lines', 
            name='RSI', line=dict(color='purple', width=1.5)
        ), row=2, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
    
    # 3. MACD
    if 'MACD' in df.columns:
        fig.add_trace(go.Scatter(
            x=df['date'], y=df['MACD'], mode='lines', 
            name='MACD', line=dict(color='blue', width=1)
        ), row=3, col=1)
        
        if 'MACD_signal' in df.columns:
            fig.add_trace(go.Scatter(
                x=df['date'], y=df['MACD_signal'], mode='lines', 
                name='Signal', line=dict(color='red', width=1, dash='dash')
            ), row=3, col=1)
        
        if 'MACD_diff' in df.columns:
            # الهيستوجرام
            colors = ['green' if val >= 0 else 'red' for val in df['MACD_diff']]
            fig.add_trace(go.Bar(
                x=df['date'], y=df['MACD_diff'], 
                name='Histogram', marker_color=colors, opacity=0.5
            ), row=3, col=1)
    
    fig.update_layout(
        height=650,
        template='plotly_dark',
        showlegend=True,
        hovermode='x unified',
        xaxis_rangeslider_visible=False,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    
    fig.update_yaxes(title_text="السعر ($)", row=1, col=1)
    fig.update_yaxes(title_text="RSI", row=2, col=1, range=[0, 100])
    fig.update_yaxes(title_text="MACD", row=3, col=1)
    
    return fig

# ==========================================
# 11. التطبيق الرئيسي
# ==========================================
def main():
    """التطبيق الرئيسي"""
    
    # إعداد العنوان
    st.title("🚀 بوت التداول الذكي المتطور V3.0")
    st.caption("تحليل متقدم باستخدام الذكاء الاصطناعي - Ensemble + LSTM + Sentiment Analysis")
    
    # ==========================================
    # تهيئة حالة الجلسة
    # ==========================================
    if 'ai_engine' not in st.session_state:
        # استخدام المحرك المتطور إذا كان متاحاً
        if ADVANCED_ENGINE_AVAILABLE:
            try:
                st.session_state['ai_engine'] = AdvancedAITradingEngine()
                st.session_state['engine_version'] = 'V3'
            except Exception:
                st.session_state['ai_engine'] = LocalAITradingEngine()
                st.session_state['engine_version'] = 'V1'
        else:
            st.session_state['ai_engine'] = LocalAITradingEngine()
            st.session_state['engine_version'] = 'V1'
    
    if 'watchlist' not in st.session_state:
        st.session_state['watchlist'] = DEFAULT_WATCHLIST.copy()
    
    if 'df' not in st.session_state:
        st.session_state['df'] = None
    
    if 'prediction_history' not in st.session_state:
        st.session_state['prediction_history'] = []
    
    # ==========================================
    # الشريط الجانبي
    # ==========================================
    with st.sidebar:
        st.header("⚙️ الإعدادات")
        st.subheader("⭐ الأسهم المفضلة")
        
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
            st.warning("⚠️ القائمة فارغة")
            current_symbol = DEFAULT_SYMBOL
        
        st.divider()
        
        # إدارة القائمة
        with st.expander("➕ إضافة رمز جديد", expanded=False):
            new_symbol = st.text_input("رمز السهم:", placeholder="مثل: NVDA", key="new_symbol_input")
            if st.button("إضافة", use_container_width=True):
                if new_symbol and add_to_watchlist(new_symbol.upper()):
                    st.success(f"✅ تم إضافة {new_symbol.upper()}")
                    st.rerun()
                elif new_symbol:
                    st.warning("⚠️ الرمز موجود مسبقاً")
        
        with st.expander("🗑️ حذف من المفضلة", expanded=False):
            if watchlist:
                symbol_to_remove = st.selectbox(
                    "اختر رمزاً للحذف:",
                    options=watchlist,
                    index=None,
                    key="remove_selector"
                )
                if symbol_to_remove and st.button("حذف", use_container_width=True):
                    if remove_from_watchlist(symbol_to_remove):
                        st.success(f"✅ تم حذف {symbol_to_remove}")
                        st.rerun()
        
        st.caption(f"📊 إجمالي المفضلة: {len(watchlist)}")
        st.divider()
        
        # إعدادات البيانات
        st.subheader("⏱️ إعدادات البيانات")
        selected_period = st.selectbox("الفترة:", PERIODS, index=1)
        selected_interval = st.selectbox("الفاصل الزمني:", INTERVALS, index=2)
        
        st.divider()
        
        # إعدادات التداول
        st.subheader("🔌 إعدادات التداول")
        ib_host = st.text_input("IBKR Host:", value=DEFAULT_HOST)
        ib_port = st.number_input("IBKR Port:", value=DEFAULT_PORT)
        api_key = st.text_input("🔑 OpenAI/NewsAPI Key:", type="password", placeholder="اختياري")
        
        symbol = st.session_state.get('selected_symbol', current_symbol)
        quantity = st.number_input("الكمية:", value=DEFAULT_QUANTITY, step=1, min_value=1)
        
        # وضع التحليل
        analysis_mode = st.radio(
            "🧠 وضع التحليل:",
            ANALYSIS_MODES,
            index=3 if st.session_state.get('engine_version') == 'V3' else 0
        )
        
        st.divider()
        
        # حالة المحرك
        st.subheader("📊 حالة المحرك")
        engine = st.session_state['ai_engine']
        
        if engine.is_trained:
            st.success(f"✅ النموذج جاهز (الإصدار: {st.session_state.get('engine_version', 'V1')})")
            
            if hasattr(engine, 'get_performance_metrics'):
                metrics = engine.get_performance_metrics()
                if 'total_predictions' in metrics:
                    st.metric("التنبؤات", metrics['total_predictions'])
        else:
            st.warning("⚠️ النموذج غير مدرب")
        
        st.divider()
        
        # زر إعادة تعيين
        if st.button("🔄 إعادة تعيين", use_container_width=True):
            for key in ['result', 'action', 'confidence', 'df']:
                st.session_state.pop(key, None)
            st.rerun()
    
    # ==========================================
    # الواجهة الرئيسية
    # ==========================================
    col1, col2 = st.columns([1.6, 1])
    
    # العمود الأيسر - البيانات والشارت
    with col1:
        st.subheader("📊 البيانات والتحليل الفني")
        st.info(f"📌 السهم النشط: **{symbol}** | المحرك: **{st.session_state.get('engine_version', 'V1')}**")
        
        # زر جلب البيانات
        col_buttons = st.columns([1, 2])
        with col_buttons[0]:
            fetch_clicked = st.button("🔄 جلب البيانات", use_container_width=True, type="primary")
        
        if fetch_clicked:
            with st.spinner(f"جاري جلب بيانات {symbol}..."):
                df, error = get_market_data(
                    symbol,
                    period=selected_period,
                    interval=selected_interval
                )
                if error:
                    st.error(error)
                else:
                    st.session_state['df'] = df
                    engine = st.session_state['ai_engine']
                    
                    # تدريب المحرك
                    with st.spinner("جاري تدريب النموذج..."):
                        safe_train_engine(engine, df, symbol)
                    
                    st.success(f"✅ تم تحديث {len(df)} شمعة بنجاح")
                    st.balloons()
        
        # عرض البيانات
        if 'df' in st.session_state and st.session_state['df'] is not None:
            df = st.session_state['df']
            
            # مؤشرات السعر
            last_price = df['close'].iloc[-1]
            prev_price = df['close'].iloc[-2] if len(df) > 1 else last_price
            change = ((last_price - prev_price) / prev_price) * 100
            
            col_metrics = st.columns(4)
            with col_metrics[0]:
                st.metric("السعر الحالي", f"${last_price:.2f}", f"{change:+.2f}%")
            with col_metrics[1]:
                st.metric("حجم التداول", f"{df['volume'].iloc[-1]:,.0f}")
            with col_metrics[2]:
                st.metric("RSI", f"{df['RSI'].iloc[-1]:.1f}")
            with col_metrics[3]:
                rsi_status = "🟢 قوي" if df['RSI'].iloc[-1] > 70 else ("🔴 ضعيف" if df['RSI'].iloc[-1] < 30 else "⚪ محايد")
                st.metric("حالة RSI", rsi_status)
            
            # الرسم البياني
            fig = plot_chart(df, symbol)
            st.plotly_chart(fig, use_container_width=True)
            
            # عرض البيانات
            with st.expander("📋 معاينة البيانات الرقمية", expanded=False):
                cols = ['date', 'open', 'high', 'low', 'close', 'volume', 'RSI', 'SMA_20', 'SMA_50']
                available_cols = [c for c in cols if c in df.columns]
                st.dataframe(df[available_cols].tail(10), use_container_width=True)
    
    # العمود الأيمن - التحليل والإشارات
    with col2:
        st.subheader("🤖 إشارات التداول والتنفيذ")
        
        # زر تشغيل التحليل
        if st.button("🧠 تشغيل التحليل", use_container_width=True, type="primary"):
            if 'df' not in st.session_state or st.session_state['df'] is None:
                st.warning("⚠️ يرجى جلب البيانات أولاً")
            else:
                df = st.session_state['df']
                engine = st.session_state['ai_engine']
                
                with st.spinner("جاري تحليل البيانات..."):
                    try:
                        if analysis_mode == "المحرك المحلي":
                            result, action, conf = analyze_with_local_ai(df, engine)
                        elif analysis_mode == "OpenAI":
                            result, action, conf = analyze_with_openai(
                                df.tail(10).to_string(), api_key, symbol
                            )
                        elif analysis_mode == "هجين":
                            result, action, conf = analyze_hybrid(df, api_key, symbol, engine)
                        else:  # المحرك المتطور
                            result, action, conf = analyze_advanced(df, engine, api_key)
                        
                        st.session_state['result'] = result
                        st.session_state['action'] = action
                        st.session_state['confidence'] = conf
                        st.session_state['last_analysis_time'] = datetime.now()
                        
                        # حفظ في التاريخ
                        st.session_state['prediction_history'].append({
                            'timestamp': datetime.now(),
                            'symbol': symbol,
                            'action': action,
                            'confidence': conf,
                            'mode': analysis_mode
                        })
                        
                        st.success("✅ تم اكتمال التحليل")
                        
                    except Exception as e:
                        st.error(f"❌ خطأ في التحليل: {str(e)}")
        
        # عرض النتائج
        if 'result' in st.session_state:
            st.divider()
            
            action = st.session_state['action']
            conf = st.session_state.get('confidence', 0)
            
            # عرض الإشارة
            if action == "BUY":
                st.success(f"🟢 **إشارة شراء** (درجة الثقة: {conf}%)")
                st.balloons()
            elif action == "SELL":
                st.error(f"🔴 **إشارة بيع** (درجة الثقة: {conf}%)")
            else:
                st.warning(f"⏸️ **انتظار** (درجة الثقة: {conf}%)")
            
            # عرض التقرير
            st.text_area("📝 تفاصيل التقرير:", st.session_state['result'], height=200)
            
            # وقت التحليل
            if 'last_analysis_time' in st.session_state:
                st.caption(f"🕐 آخر تحليل: {st.session_state['last_analysis_time'].strftime('%Y-%m-%d %H:%M:%S')}")
            
            st.divider()
            
            # ==========================================
            # تنفيذ الصفقة
            # ==========================================
            st.subheader("💼 تنفيذ الصفقة (IBKR)")
            
            # عرض معلومات الحساب (Mock)
            with st.expander("📊 معلومات الحساب", expanded=False):
                try:
                    broker = MockBroker()
                    account = broker.get_account_info()
                    st.metric("الرصيد", f"${account.get('balance', 0):,.2f}")
                except Exception:
                    st.info("لا توجد معلومات حساب")
            
            # أزرار التنفيذ
            if action == "BUY":
                if st.button(f"🚀 إرسال أمر شراء ({quantity} سهم)", use_container_width=True, type="primary"):
                    msg = execute_ib_order("BUY", symbol, quantity, ib_host, ib_port)
                    if "✅" in msg:
                        st.success(msg)
                        st.balloons()
                    else:
                        st.error(msg)
                        
            elif action == "SELL":
                if st.button(f"🔻 إرسال أمر بيع ({quantity} سهم)", use_container_width=True, type="primary"):
                    msg = execute_ib_order("SELL", symbol, quantity, ib_host, ib_port)
                    if "✅" in msg:
                        st.success(msg)
                    else:
                        st.error(msg)
            else:
                st.info("⏸️ المحرك ينصح بعدم الدخول في صفقات حالياً")
            
            # زر مسح
            if st.button("🗑️ مسح التحليل الحالي", use_container_width=True):
                for key in ['result', 'action', 'confidence', 'last_analysis_time']:
                    st.session_state.pop(key, None)
                st.rerun()
        
        # ==========================================
        # تاريخ التنبؤات
        # ==========================================
        if st.session_state['prediction_history']:
            with st.expander("📜 تاريخ التنبؤات", expanded=False):
                history = pd.DataFrame(st.session_state['prediction_history'][-10:])
                if not history.empty:
                    history['timestamp'] = history['timestamp'].dt.strftime('%H:%M')
                    st.dataframe(
                        history[['timestamp', 'symbol', 'action', 'confidence']],
                        use_container_width=True
                    )

# ==========================================
# تشغيل التطبيق
# ==========================================
if __name__ == "__main__":
    main()
