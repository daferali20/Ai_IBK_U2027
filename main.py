# ==========================================
# حل مشكلة anyio.NoEventLoopError
# ==========================================
import sys
import asyncio
import warnings
import nest_asyncio
import anyio

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

# إصلاح AnyIO
def fix_anyio():
    try:
        backend = anyio.get_async_backend()
        if backend is None:
            anyio._core._eventloop._async_backend = None
    except:
        pass

fix_anyio()

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
import yfinance as yf
from openai import OpenAI
import random
import time

# ==========================================
# استيراد IBKR (اختياري)
# ==========================================
IBKR_AVAILABLE = False
try:
    from ib_async import IB, Stock, MarketOrder
    IBKR_AVAILABLE = True
except ImportError:
    try:
        from ib_insync import IB, Stock, MarketOrder
        IBKR_AVAILABLE = True
    except ImportError:
        pass

# ==========================================
# نموذج الذكاء الاصطناعي المحلي
# ==========================================
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

class LocalAITradingEngine:
    """محرك تداول محلي بالذكاء الاصطناعي"""
    
    def __init__(self, min_samples=30):
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=6,
            min_samples_split=5,
            random_state=42
        )
        self.is_trained = False
        self.min_samples = min_samples
        self.feature_importance = None
        self.last_accuracy = 0
        
    def extract_features(self, df):
        data = df.copy()
        data['returns'] = data['close'].pct_change()
        data['volatility'] = data['returns'].rolling(10).std()
        data['ma_diff'] = data['SMA_20'] - data['SMA_50']
        data['dist_sma20'] = (data['close'] - data['SMA_20']) / data['SMA_20']
        data['dist_sma50'] = (data['close'] - data['SMA_50']) / data['SMA_50']
        data['rsi_momentum'] = data['RSI'].diff()
        data['volume_ratio'] = data['volume'] / data['volume'].rolling(20).mean()
        data['price_range'] = (data['high'] - data['low']) / data['close']
        for lag in [1, 2]:
            data[f'return_lag_{lag}'] = data['returns'].shift(lag)
            data[f'rsi_lag_{lag}'] = data['RSI'].shift(lag)
        return data
    
    def get_feature_names(self):
        return ['RSI', 'SMA_20', 'SMA_50', 'returns', 'volatility', 
                'ma_diff', 'dist_sma20', 'dist_sma50', 'rsi_momentum',
                'volume_ratio', 'price_range',
                'return_lag_1', 'return_lag_2', 'rsi_lag_1', 'rsi_lag_2']
    
    def train_quick_model(self, df):
        data = self.extract_features(df).dropna()
        if len(data) < self.min_samples:
            return False
        
        data['target'] = np.where(data['close'].shift(-1) > data['close'], 1, 0)
        features = self.get_feature_names()
        X = data[features][:-1]
        y = data['target'][:-1]
        
        if len(X) < 10:
            return False
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        self.model.fit(X_train, y_train)
        accuracy = self.model.score(X_test, y_test)
        self.last_accuracy = accuracy
        
        if accuracy > 0.55:
            self.is_trained = True
            self.feature_importance = pd.DataFrame({
                'feature': features,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            return True
        return False
    
    def predict_opportunity(self, df):
        data = self.extract_features(df).dropna()
        if data.empty:
            return 'HOLD', 0.0, "بيانات غير كافية"
        
        latest = data.iloc[-1]
        if not self.is_trained:
            self.train_quick_model(df)
        
        features = self.get_feature_names()
        latest_features = np.array(latest[features]).reshape(1, -1)
        
        if self.is_trained:
            prob_up = self.model.predict_proba(latest_features)[0][1]
            probas = [tree.predict_proba(latest_features)[0][1] 
                     for tree in self.model.estimators_]
            std_prob = np.std(probas)
            uncertainty = std_prob * 0.4
            calibrated_prob = prob_up * (1 - uncertainty)
            calibrated_prob = np.clip(calibrated_prob, 0.25, 0.75)
        else:
            calibrated_prob = 0.5
        
        rsi_val = round(latest['RSI'], 2)
        sma20 = latest['SMA_20']
        close_price = latest['close']
        volume_ratio = latest['volume_ratio']
        
        if (calibrated_prob > 0.58 and rsi_val < 60 and close_price > sma20 and volume_ratio > 0.8):
            confidence = round(calibrated_prob * 100, 1)
            reason = f"🚀 إشارة شراء قوية!\nثقة: {confidence}%\nRSI: {rsi_val}"
            return 'BUY', confidence, reason
        elif (calibrated_prob < 0.42 or rsi_val > 72):
            confidence = round((1 - calibrated_prob) * 100, 1)
            reason = f"🔻 إشارة بيع!\nثقة: {confidence}%\nRSI: {rsi_val}"
            return 'SELL', confidence, reason
        else:
            confidence = round(abs(calibrated_prob - 0.5) * 200, 1)
            reason = f"⏸️ منطقة انتظار\nRSI: {rsi_val}"
            return 'HOLD', confidence, reason

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
        
        df['RSI'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
        df['SMA_20'] = ta.trend.sma_indicator(df['close'], window=20)
        df['SMA_50'] = ta.trend.sma_indicator(df['close'], window=50)
        df['volume_ma'] = ta.trend.sma_indicator(df['volume'], window=10)
        df.dropna(inplace=True)
        return df, None
    except Exception as e:
        return None, f"❌ خطأ: {str(e)}"

# ==========================================
# إعدادات الصفحة
# ==========================================

st.set_page_config(
    page_title="AI Trading Bot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# قائمة الأسهم المفضلة
# ==========================================

DEFAULT_WATCHLIST = ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA", "NVDA", "META", "NFLX", "JPM", "VTI"]

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
# دوال التحليل
# ==========================================

def analyze_with_local_ai(df):
    engine = st.session_state['ai_engine']
    return engine.predict_opportunity(df)

def analyze_with_openai(df_summary, api_key, symbol_name):
    if not api_key:
        return 'HOLD', 0.0, "⚠️ مطلوب مفتاح OpenAI API"
    
    client = OpenAI(api_key=api_key)
    prompt = f"""
    أنت خبير تداول. حلل البيانات الفنية لسهم {symbol_name}:
    {df_summary}
    أجب بالتنسيق: [RECOMMENDATION: BUY] أو [RECOMMENDATION: SELL] أو [RECOMMENDATION: HOLD]
    ثم اشرح السبب باختصار.
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2
        )
        result = response.choices[0].message.content
        if "[RECOMMENDATION: BUY]" in result:
            return 'BUY', 75, result
        elif "[RECOMMENDATION: SELL]" in result:
            return 'SELL', 75, result
        return 'HOLD', 50, result
    except Exception as e:
        return 'HOLD', 0, f"❌ خطأ: {str(e)}"

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
                  mode='lines', name='SMA 20', line=dict(color='orange', width=1.5)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['date'], y=df['SMA_50'], 
                  mode='lines', name='SMA 50', line=dict(color='cyan', width=1.5)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['date'], y=df['RSI'], 
                  mode='lines', name='RSI', line=dict(color='purple', width=2)), row=2, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
    
    fig.update_layout(height=550, template='plotly_dark', showlegend=True, 
                      hovermode='x unified', xaxis_rangeslider_visible=False)
    return fig

# ==========================================
# تنفيذ أوامر IBKR
# ==========================================

def execute_ib_order(action, symbol, quantity, host, port):
    if not IBKR_AVAILABLE:
        return "⚠️ IBKR غير مثبت"
    
    ib = IB()
    try:
        client_id = random.randint(1000, 9999)
        ib.connect(host, int(port), clientId=client_id, timeout=5)
        contract = Stock(symbol, 'SMART', 'USD')
        ib.qualifyContracts(contract)
        order = MarketOrder(action, quantity)
        ib.placeOrder(contract, order)
        ib.sleep(1)
        ib.disconnect()
        return f"✅ تم إرسال أمر {action} لـ {quantity} سهم"
    except Exception as e:
        try:
            ib.disconnect()
        except:
            pass
        return f"❌ فشل: {str(e)}"

# ==========================================
# التطبيق الرئيسي
# ==========================================

def main():
    st.title("🤖 بوت التداول الذكي (Yahoo + IBKR)")
    st.caption("📊 تحليل فني بالذكاء الاصطناعي مع تنفيذ أوامر IBKR")
    
    if not IBKR_AVAILABLE:
        st.warning("⚠️ IBKR غير مثبت. سيتم استخدام Yahoo فقط للبيانات والتحليل")
    
    if 'ai_engine' not in st.session_state:
        st.session_state['ai_engine'] = LocalAITradingEngine()
    
    # ===== الشريط الجانبي =====
    with st.sidebar:
        st.header("⚙️ الإعدادات")
        
        # الأسهم المفضلة
        st.subheader("⭐ الأسهم المفضلة")
        watchlist = load_watchlist()
        
        if watchlist:
            selected_symbol = st.selectbox("اختر من المفضلة:", options=watchlist, index=0)
            if st.button("📌 اختيار السهم", use_container_width=True):
                st.session_state['selected_symbol'] = selected_symbol
                st.success(f"✅ تم اختيار {selected_symbol}")
            current_symbol = st.session_state.get('selected_symbol', selected_symbol)
            st.info(f"📌 السهم الحالي: **{current_symbol}**")
        else:
            current_symbol = "AAPL"
        
        st.divider()
        
        # إضافة رمز
        with st.expander("➕ إضافة رمز جديد"):
            new_symbol = st.text_input("رمز السهم:", placeholder="مثل: NVDA")
            if st.button("إضافة", use_container_width=True):
                if new_symbol and add_to_watchlist(new_symbol.upper()):
                    st.success(f"✅ تم إضافة {new_symbol.upper()}")
                    st.rerun()
                elif new_symbol:
                    st.warning("⚠️ الرمز موجود مسبقاً")
        
        # حذف رمز
        with st.expander("🗑️ حذف من المفضلة"):
            if watchlist:
                symbol_to_remove = st.selectbox("اختر رمزاً للحذف:", options=watchlist, index=None)
                if symbol_to_remove and st.button("حذف", use_container_width=True):
                    if remove_from_watchlist(symbol_to_remove):
                        st.success(f"✅ تم حذف {symbol_to_remove}")
                        st.rerun()
        
        st.caption(f"📊 إجمالي المفضلة: {len(watchlist)}")
        st.divider()
        
        # إعدادات البيانات
        st.subheader("⏱️ إعدادات البيانات")
        selected_period = st.selectbox("الفترة:", ["1d", "5d", "1mo", "3mo", "6mo", "1y"], index=1)
        selected_interval = st.selectbox("الفاصل الزمني:", ["1m", "5m", "15m", "30m", "60m", "1d"], index=2)
        
        st.divider()
        
        # إعدادات أخرى
        st.subheader("🔌 الإعدادات")
        ib_host = st.text_input("IBKR Host", "127.0.0.1")
        ib_port = st.number_input("IBKR Port", value=7497)
        api_key = st.text_input("🔑 OpenAI Key (اختياري)", type="password")
        quantity = st.number_input("📊 الكمية", value=10, step=1, min_value=1)
        analysis_mode = st.radio("🧠 وضع التحليل", ["المحرك المحلي", "OpenAI", "هجين"])
        
        st.divider()
        
        # حالة المحرك
        st.subheader("📊 حالة المحرك")
        engine = st.session_state['ai_engine']
        if engine.is_trained:
            st.success("✅ النموذج جاهز")
        else:
            st.warning("⚠️ غير مدرب")
    
    # ===== السهم النشط =====
    symbol = st.session_state.get('selected_symbol', current_symbol)
    
    # ===== الأعمدة الرئيسية =====
    col1, col2 = st.columns([1.6, 1])
    
    with col1:
        st.subheader("📊 البيانات والتحليل الفني")
        st.info(f"📌 السهم النشط: **{symbol}**")
        
        if st.button("🔄 جلب بيانات السوق", use_container_width=True):
            with st.spinner(f"جاري جلب بيانات {symbol}..."):
                df, error = get_market_data(symbol, period=selected_period, interval=selected_interval)
                if error:
                    st.error(error)
                else:
                    st.session_state['df'] = df
                    engine = st.session_state['ai_engine']
                    if engine.train_quick_model(df):
                        st.success("✅ تم تدريب النموذج بنجاح!")
                    st.success(f"✅ تم تحديث {len(df)} شمعة")
        
        if 'df' in st.session_state:
            df = st.session_state['df']
            
            last_price = df['close'].iloc[-1]
            prev_price = df['close'].iloc[-2] if len(df) > 1 else last_price
            change = ((last_price - prev_price) / prev_price) * 100 if prev_price != 0 else 0
            
            col_m1, col_m2, col_m3 = st.columns(3)
            with col_m1:
                st.metric("💰 السعر", f"${last_price:.2f}", f"{change:+.2f}%")
            with col_m2:
                st.metric("📊 RSI", f"{df['RSI'].iloc[-1]:.1f}")
            with col_m3:
                st.metric("📈 SMA 20", f"${df['SMA_20'].iloc[-1]:.2f}")
            
            fig = plot_chart(df, symbol)
            st.plotly_chart(fig, use_container_width=True)
            
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
                        action, conf, result = analyze_with_local_ai(df)
                    elif analysis_mode == "OpenAI":
                        action, conf, result = analyze_with_openai(df.tail(10).to_string(), api_key, symbol)
                    else:
                        action1, conf1, r1 = analyze_with_local_ai(df)
                        action2, conf2, r2 = analyze_with_openai(df.tail(10).to_string(), api_key, symbol)
                        if action1 == action2 and action1 != "HOLD":
                            action, conf, result = action1, (conf1+conf2)/2, f"✅ توافق\n{r1}\n{r2}"
                        else:
                            action, conf, result = "HOLD", 50, f"⚠️ تباين\nمحلي: {action1}\nOpenAI: {action2}"
                    
                    st.session_state['result'] = result
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
            
            st.text_area("التفاصيل:", st.session_state['result'], height=150)
            
            st.divider()
            st.subheader("💼 التنفيذ")
            
            if action == "BUY":
                if st.button(f"🚀 شراء {quantity} سهم", use_container_width=True, type="primary"):
                    msg = execute_ib_order("BUY", symbol, quantity, ib_host, ib_port)
                    st.info(msg)
            elif action == "SELL":
                if st.button(f"🔻 بيع {quantity} سهم", use_container_width=True, type="primary"):
                    msg = execute_ib_order("SELL", symbol, quantity, ib_host, ib_port)
                    st.info(msg)
            else:
                st.info("⏸️ لا توجد صفقة")

# ==========================================
# التشغيل
# ==========================================

if __name__ == "__main__":
    main()
