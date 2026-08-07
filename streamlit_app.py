# ==========================================
# 🔧 إصلاح مشكلة anyio.NoEventLoopError
# ==========================================
import sys
import asyncio
import os
import warnings

# تعيين متغيرات البيئة
os.environ['PYTHONASYNCIODEBUG'] = '0'
os.environ['ANYIO_BACKEND'] = 'asyncio'

# إنشاء event loop
try:
    loop = asyncio.get_event_loop()
except RuntimeError:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

# تطبيق nest_asyncio
try:
    import nest_asyncio
    nest_asyncio.apply()
except ImportError:
    pass

# إصلاح AnyIO
try:
    import anyio
    if hasattr(anyio, '_core') and hasattr(anyio._core, '_eventloop'):
        try:
            anyio._core._eventloop._async_backend = None
        except:
            pass
except ImportError:
    pass

warnings.filterwarnings('ignore')

# ==========================================
# استيراد المكتبات
# ==========================================
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf
import ta
from openai import OpenAI
import time
import random

# ==========================================
# نموذج الذكاء الاصطناعي (مدمج)
# ==========================================
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

class LocalAITradingEngine:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
        self.is_trained = False
        self.feature_importance = None
        self.accuracy = 0
        
    def extract_features(self, df):
        data = df.copy()
        data['returns'] = data['close'].pct_change()
        data['volatility'] = data['returns'].rolling(10).std()
        data['ma_diff'] = data['SMA_20'] - data['SMA_50']
        data['dist_sma20'] = (data['close'] - data['SMA_20']) / data['SMA_20']
        data['rsi_momentum'] = data['RSI'].diff()
        data['volume_ratio'] = data['volume'] / data['volume'].rolling(20).mean()
        return data
    
    def train(self, df):
        data = self.extract_features(df).dropna()
        if len(data) < 30:
            return False, "بيانات غير كافية"
        
        data['target'] = np.where(data['close'].shift(-1) > data['close'], 1, 0)
        features = ['RSI', 'SMA_20', 'SMA_50', 'returns', 'volatility', 
                   'ma_diff', 'dist_sma20', 'rsi_momentum', 'volume_ratio']
        
        X = data[features][:-1]
        y = data['target'][:-1]
        
        if len(X) < 10:
            return False, "بيانات غير كافية للتدريب"
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        self.model.fit(X_train, y_train)
        self.accuracy = self.model.score(X_test, y_test)
        self.is_trained = True
        
        self.feature_importance = pd.DataFrame({
            'feature': features,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return True, f"✅ تدريب بنجاح! الدقة: {self.accuracy*100:.1f}%"
    
    def predict(self, df):
        data = self.extract_features(df).dropna()
        if data.empty:
            return 'HOLD', 0, "بيانات غير كافية"
        
        latest = data.iloc[-1]
        features = ['RSI', 'SMA_20', 'SMA_50', 'returns', 'volatility', 
                   'ma_diff', 'dist_sma20', 'rsi_momentum', 'volume_ratio']
        
        latest_features = np.array(latest[features]).reshape(1, -1)
        
        if not self.is_trained:
            return 'HOLD', 50, "النموذج غير مدرب"
        
        prob_up = self.model.predict_proba(latest_features)[0][1]
        confidence = round(prob_up * 100, 1)
        
        rsi = latest['RSI']
        sma20 = latest['SMA_20']
        price = latest['close']
        
        if prob_up > 0.6 and rsi < 60 and price > sma20:
            return 'BUY', confidence, f"🚀 شراء - RSI: {rsi:.1f}"
        elif prob_up < 0.4 or rsi > 70:
            return 'SELL', confidence, f"🔻 بيع - RSI: {rsi:.1f}"
        else:
            return 'HOLD', confidence, f"⏸️ انتظار - RSI: {rsi:.1f}"

# ==========================================
# دوال جلب البيانات
# ==========================================
def get_market_data(symbol, period="5d", interval="5m"):
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period, interval=interval)
        
        if df.empty:
            return None, f"لا توجد بيانات لـ {symbol}"
        
        df.rename(columns={
            'Open': 'open', 'High': 'high', 'Low': 'low',
            'Close': 'close', 'Volume': 'volume'
        }, inplace=True)
        
        df.index = df.index.tz_localize(None)
        df['date'] = df.index
        
        df['RSI'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
        df['SMA_20'] = ta.trend.sma_indicator(df['close'], window=20)
        df['SMA_50'] = ta.trend.sma_indicator(df['close'], window=50)
        df.dropna(inplace=True)
        
        return df, None
    except Exception as e:
        return None, str(e)

# ==========================================
# دوال الرسم البياني
# ==========================================
def plot_chart(df, symbol):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        vertical_spacing=0.05, row_heights=[0.7, 0.3])
    
    fig.add_trace(go.Candlestick(
        x=df['date'], open=df['open'], high=df['high'],
        low=df['low'], close=df['close'], name='Price'
    ), row=1, col=1)
    
    fig.add_trace(go.Scatter(x=df['date'], y=df['SMA_20'],
                  mode='lines', name='SMA 20', line=dict(color='orange')), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['date'], y=df['SMA_50'],
                  mode='lines', name='SMA 50', line=dict(color='cyan')), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['date'], y=df['RSI'],
                  mode='lines', name='RSI', line=dict(color='purple')), row=2, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
    
    fig.update_layout(height=500, template='plotly_dark', showlegend=True,
                      xaxis_rangeslider_visible=False)
    return fig

# ==========================================
# إعدادات الصفحة
# ==========================================
st.set_page_config(page_title="AI Trading Bot", page_icon="🤖", layout="wide")

# ==========================================
# التطبيق الرئيسي
# ==========================================
def main():
    st.title("🤖 بوت التداول الذكي (Yahoo Finance)")
    
    # تهيئة المحرك
    if 'engine' not in st.session_state:
        st.session_state['engine'] = LocalAITradingEngine()
    
    if 'watchlist' not in st.session_state:
        st.session_state['watchlist'] = ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA", "NVDA"]
    
    engine = st.session_state['engine']
    
    # ===== الشريط الجانبي =====
    with st.sidebar:
        st.header("⚙️ الإعدادات")
        
        symbol = st.selectbox("اختر السهم:", st.session_state['watchlist'])
        
        new_symbol = st.text_input("إضافة رمز:")
        if st.button("➕ إضافة"):
            if new_symbol and new_symbol.upper() not in st.session_state['watchlist']:
                st.session_state['watchlist'].append(new_symbol.upper())
                st.rerun()
        
        period = st.selectbox("الفترة:", ["5d", "1mo", "3mo"], index=0)
        interval = st.selectbox("الفاصل:", ["5m", "15m", "60m"], index=0)
        quantity = st.number_input("الكمية", value=10, step=1)
        
        st.divider()
        if engine.is_trained:
            st.success(f"✅ النموذج جاهز (دقة: {engine.accuracy*100:.1f}%)")
        else:
            st.warning("⚠️ غير مدرب")
    
    # ===== الأعمدة =====
    col1, col2 = st.columns([1.5, 1])
    
    with col1:
        st.subheader("📊 بيانات السوق")
        
        if st.button("🔄 جلب البيانات", use_container_width=True):
            with st.spinner(f"جاري جلب {symbol}..."):
                df, error = get_market_data(symbol, period, interval)
                if error:
                    st.error(f"❌ {error}")
                else:
                    st.session_state['df'] = df
                    result, msg = engine.train(df)
                    if result:
                        st.success(msg)
                    else:
                        st.warning(msg)
                    st.success(f"✅ {len(df)} شمعة")
        
        if 'df' in st.session_state:
            df = st.session_state['df']
            
            last_price = df['close'].iloc[-1]
            change = ((df['close'].iloc[-1] - df['close'].iloc[-2]) / df['close'].iloc[-2]) * 100 if len(df) > 1 else 0
            st.metric("السعر الحالي", f"${last_price:.2f}", f"{change:+.2f}%")
            
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
                action, confidence, reason = engine.predict(df)
                st.session_state['action'] = action
                st.session_state['confidence'] = confidence
                st.session_state['reason'] = reason
                st.success("✅ تم التحليل")
        
        if 'action' in st.session_state:
            st.divider()
            
            action = st.session_state['action']
            confidence = st.session_state.get('confidence', 0)
            
            if action == "BUY":
                st.success(f"🟢 **شراء** ({confidence}%)")
            elif action == "SELL":
                st.error(f"🔴 **بيع** ({confidence}%)")
            else:
                st.warning(f"⏸️ **انتظار** ({confidence}%)")
            
            st.info(st.session_state.get('reason', ''))
            
            st.divider()
            st.subheader("💼 التنفيذ")
            
            if action == "BUY":
                st.button(f"🚀 شراء {quantity} سهم", use_container_width=True, type="primary")
            elif action == "SELL":
                st.button(f"🔻 بيع {quantity} سهم", use_container_width=True, type="primary")
            else:
                st.info("⏸️ لا توجد صفقة")

if __name__ == "__main__":
    main()
