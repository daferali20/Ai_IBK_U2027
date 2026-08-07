# ==========================================
# حل مشكلة event loop
# ==========================================
import asyncio
import nest_asyncio
import warnings

# تطبيق nest_asyncio
try:
    nest_asyncio.apply()
except:
    pass

# إنشاء event loop
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
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf
import ta
import time

# ==========================================
# نموذج الذكاء الاصطناعي (مدمج)
# ==========================================
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

class LocalAITradingEngine:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.is_trained = False
        
    def train_quick_model(self, df):
        try:
            df = df.copy()
            df['returns'] = df['close'].pct_change()
            df['volatility'] = df['returns'].rolling(10).std()
            df['target'] = np.where(df['close'].shift(-1) > df['close'], 1, 0)
            
            features = ['returns', 'volatility']
            df = df.dropna()
            
            if len(df) < 20:
                return False
            
            X = df[features][:-1]
            y = df['target'][:-1]
            
            self.model.fit(X, y)
            self.is_trained = True
            return True
        except:
            return False
    
    def predict_opportunity(self, df):
        try:
            df = df.copy()
            df['returns'] = df['close'].pct_change()
            df['volatility'] = df['returns'].rolling(10).std()
            df = df.dropna()
            
            if df.empty or not self.is_trained:
                return 'HOLD', 50, "بيانات غير كافية"
            
            latest = df.iloc[-1]
            features = np.array([[latest['returns'], latest['volatility']]])
            
            prob = self.model.predict_proba(features)[0][1]
            confidence = round(prob * 100, 1)
            
            if prob > 0.6:
                return 'BUY', confidence, "إشارة شراء"
            elif prob < 0.4:
                return 'SELL', confidence, "إشارة بيع"
            else:
                return 'HOLD', confidence, "انتظار"
        except:
            return 'HOLD', 50, "خطأ في التنبؤ"

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
        
        # مؤشرات فنية
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
# القوائم
# ==========================================
DEFAULT_WATCHLIST = ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA", "NVDA", "META"]

# ==========================================
# التطبيق الرئيسي
# ==========================================
def main():
    st.title("🤖 AI Trading Bot (Yahoo Finance)")
    
    # تهيئة المحرك
    if 'engine' not in st.session_state:
        st.session_state['engine'] = LocalAITradingEngine()
    
    if 'watchlist' not in st.session_state:
        st.session_state['watchlist'] = DEFAULT_WATCHLIST.copy()
    
    engine = st.session_state['engine']
    
    # ===== الشريط الجانبي =====
    with st.sidebar:
        st.header("⚙️ Settings")
        
        symbol = st.selectbox("Select Symbol:", st.session_state['watchlist'])
        
        new_symbol = st.text_input("Add Symbol:", placeholder="AAPL")
        if st.button("➕ Add"):
            if new_symbol and new_symbol.upper() not in st.session_state['watchlist']:
                st.session_state['watchlist'].append(new_symbol.upper())
                st.rerun()
        
        period = st.selectbox("Period:", ["5d", "1mo", "3mo"], index=0)
        interval = st.selectbox("Interval:", ["5m", "15m", "60m"], index=0)
        quantity = st.number_input("Quantity", value=10, step=1)
        
        st.divider()
        if engine.is_trained:
            st.success("✅ Model Ready")
        else:
            st.warning("⚠️ Not Trained")
    
    # ===== الأعمدة =====
    col1, col2 = st.columns([1.5, 1])
    
    with col1:
        st.subheader("📊 Market Data")
        
        if st.button("🔄 Fetch Data", use_container_width=True):
            with st.spinner(f"Fetching {symbol}..."):
                df, error = get_market_data(symbol, period, interval)
                if error:
                    st.error(f"❌ {error}")
                else:
                    st.session_state['df'] = df
                    if engine.train_quick_model(df):
                        st.success("✅ Model trained successfully!")
                    st.success(f"✅ {len(df)} candles loaded")
        
        if 'df' in st.session_state:
            df = st.session_state['df']
            
            last_price = df['close'].iloc[-1]
            change = ((df['close'].iloc[-1] - df['close'].iloc[-2]) / df['close'].iloc[-2]) * 100 if len(df) > 1 else 0
            st.metric("Current Price", f"${last_price:.2f}", f"{change:+.2f}%")
            
            fig = plot_chart(df, symbol)
            st.plotly_chart(fig, use_container_width=True)
            
            with st.expander("📋 Data Preview"):
                st.dataframe(df.tail(10))
    
    with col2:
        st.subheader("🤖 AI Analysis")
        
        if st.button("🧠 Analyze", use_container_width=True, type="primary"):
            if 'df' not in st.session_state:
                st.warning("⚠️ Fetch data first")
            else:
                df = st.session_state['df']
                action, confidence, reason = engine.predict_opportunity(df)
                st.session_state['action'] = action
                st.session_state['confidence'] = confidence
                st.session_state['reason'] = reason
                st.success("✅ Analysis complete")
        
        if 'action' in st.session_state:
            st.divider()
            
            action = st.session_state['action']
            confidence = st.session_state.get('confidence', 0)
            
            if action == "BUY":
                st.success(f"🟢 **BUY** ({confidence}%)")
            elif action == "SELL":
                st.error(f"🔴 **SELL** ({confidence}%)")
            else:
                st.warning(f"⏸️ **HOLD** ({confidence}%)")
            
            st.info(st.session_state.get('reason', ''))
            
            st.divider()
            st.subheader("💼 Execution")
            
            if action == "BUY":
                st.button(f"🚀 Buy {quantity} shares", use_container_width=True, type="primary")
            elif action == "SELL":
                st.button(f"🔻 Sell {quantity} shares", use_container_width=True, type="primary")
            else:
                st.info("⏸️ No action")

if __name__ == "__main__":
    main()
