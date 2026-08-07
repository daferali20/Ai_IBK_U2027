# ==========================================
# حل مشكلة event loop - يجب أن يكون في البداية
# ==========================================
import sys
import asyncio
import warnings
import nest_asyncio

# تطبيق nest_asyncio
nest_asyncio.apply()

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
import ta
import yfinance as yf
from openai import OpenAI
import random
import time

# ==========================================
# استيراد IBKR (اختياري - للتنفيذ فقط)
# ==========================================
try:
    from ib_async import IB, Stock, MarketOrder
    IBKR_AVAILABLE = True
except ImportError:
    try:
        from ib_insync import IB, Stock, MarketOrder
        IBKR_AVAILABLE = True
    except ImportError:
        IBKR_AVAILABLE = False
        print("⚠️ IBKR غير مثبت - سيتم استخدام Yahoo فقط للبيانات والتحليل")

# ==========================================
# نموذج الذكاء الاصطناعي المحلي (مدمج)
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
        """استخراج الميزات الفنية"""
        data = df.copy()
        
        # الميزات الأساسية
        data['returns'] = data['close'].pct_change()
        data['volatility'] = data['returns'].rolling(10).std()
        data['ma_diff'] = data['SMA_20'] - data['SMA_50']
        data['dist_sma20'] = (data['close'] - data['SMA_20']) / data['SMA_20']
        data['dist_sma50'] = (data['close'] - data['SMA_50']) / data['SMA_50']
        data['rsi_momentum'] = data['RSI'].diff()
        data['volume_ratio'] = data['volume'] / data['volume'].rolling(20).mean()
        data['price_range'] = (data['high'] - data['low']) / data['close']
        
        # ميزات التأخر
        for lag in [1, 2]:
            data[f'return_lag_{lag}'] = data['returns'].shift(lag)
            data[f'rsi_lag_{lag}'] = data['RSI'].shift(lag)
        
        return data
    
    def get_feature_names(self):
        """أسماء الميزات المستخدمة"""
        return ['RSI', 'SMA_20', 'SMA_50', 'returns', 'volatility', 
                'ma_diff', 'dist_sma20', 'dist_sma50', 'rsi_momentum',
                'volume_ratio', 'price_range',
                'return_lag_1', 'return_lag_2', 'rsi_lag_1', 'rsi_lag_2']
    
    def train_quick_model(self, df):
        """تدريب سريع للنموذج"""
        data = self.extract_features(df).dropna()
        
        if len(data) < self.min_samples:
            return False
        
        # الهدف: هل سيرتفع السعر في الشمعة التالية؟
        data['target'] = np.where(data['close'].shift(-1) > data['close'], 1, 0)
        
        features = self.get_feature_names()
        X = data[features][:-1]
        y = data['target'][:-1]
        
        if len(X) < 10:
            return False
        
        # تقسيم البيانات
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # تدريب النموذج
        self.model.fit(X_train, y_train)
        
        # تقييم الأداء
        accuracy = self.model.score(X_test, y_test)
        self.last_accuracy = accuracy
        
        if accuracy > 0.55:
            self.is_trained = True
            self.feature_importance = pd.DataFrame({
                'feature': features,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            print(f"✅ النموذج مدرب بدقة: {accuracy*100:.1f}%")
            return True
        else:
            self.is_trained = False
            print(f"⚠️ النموذج ضعيف (دقة {accuracy*100:.1f}%)")
            return False
    
    def predict_opportunity(self, df):
        """التنبؤ بالفرصة"""
        data = self.extract_features(df).dropna()
        if data.empty:
            return 'HOLD', 0.0, "بيانات غير كافية"
        
        latest = data.iloc[-1]
        
        # تدريب إذا لم يكن مدرباً
        if not self.is_trained:
            self.train_quick_model(df)
        
        features = self.get_feature_names()
        latest_features = np.array(latest[features]).reshape(1, -1)
        
        # حساب الثقة
        if self.is_trained:
            prob_up = self.model.predict_proba(latest_features)[0][1]
            
            # حساب عدم اليقين
            probas = [tree.predict_proba(latest_features)[0][1] 
                     for tree in self.model.estimators_]
            std_prob = np.std(probas)
            
            # معايرة الثقة
            uncertainty = std_prob * 0.4
            calibrated_prob = prob_up * (1 - uncertainty)
            calibrated_prob = np.clip(calibrated_prob, 0.25, 0.75)
        else:
            calibrated_prob = 0.5
        
        # المؤشرات الفنية
        rsi_val = round(latest['RSI'], 2)
        sma20 = latest['SMA_20']
        sma50 = latest['SMA_50']
        close_price = latest['close']
        volume_ratio = latest['volume_ratio']
        
        # قواعد القرار
        if (calibrated_prob > 0.58 and 
            rsi_val < 60 and 
            close_price > sma20 and 
            volume_ratio > 0.8):
            
            confidence = round(calibrated_prob * 100, 1)
            reason = (f"🚀 **إشارة شراء قوية!**\n"
                     f"• ثقة النموذج: {confidence}%\n"
                     f"• RSI: {rsi_val} (منطقة آمنة)\n"
                     f"• السعر فوق SMA20: ${close_price:.2f} > ${sma20:.2f}\n"
                     f"• حجم التداول: {volume_ratio:.2f}x المتوسط")
            return 'BUY', confidence, reason
        
        elif (calibrated_prob < 0.42 or 
              rsi_val > 72 or 
              (rsi_val > 65 and calibrated_prob < 0.50)):
            
            confidence = round((1 - calibrated_prob) * 100, 1)
            reason = (f"🔻 **إشارة بيع!**\n"
                     f"• ثقة النموذج: {confidence}%\n"
                     f"• RSI: {rsi_val}" + 
                     (" (تشبع شرائي!)" if rsi_val > 72 else "") + "\n" +
                     f"• السعر نسبة لـ SMA20: {((close_price/sma20)-1)*100:.1f}%")
            return 'SELL', confidence, reason
        
        else:
            confidence = round(abs(calibrated_prob - 0.5) * 200, 1)
            reason = (f"⏸️ **منطقة انتظار**\n"
                     f"• RSI: {rsi_val}\n"
                     f"• احتمالية الصعود: {round(calibrated_prob*100,1)}%\n"
                     f"• السوق في حالة تذبذب")
            return 'HOLD', confidence, reason
    
    def get_model_info(self):
        """معلومات عن النموذج"""
        if self.is_trained:
            return {
                'trained': True,
                'accuracy': f"{self.last_accuracy*100:.1f}%",
                'features': len(self.get_feature_names())
            }
        return {'trained': False}

# ==========================================
# دوال جلب البيانات
# ==========================================

def get_market_data(symbol, period="5d", interval="5m"):
    """جلب البيانات من Yahoo Finance"""
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period, interval=interval)
        
        if df.empty:
            return None, f"❌ لا توجد بيانات للرمز: {symbol}"
        
        # إعادة تسمية الأعمدة
        df.rename(columns={
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Volume': 'volume'
        }, inplace=True)
        
        # معالجة المنطقة الزمنية
        df.index = df.index.tz_localize(None)
        df['date'] = df.index
        
        # حساب المؤشرات الفنية
        df['RSI'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
        df['SMA_20'] = ta.trend.sma_indicator(df['close'], window=20)
        df['SMA_50'] = ta.trend.sma_indicator(df['close'], window=50)
        df['volume_ma'] = ta.trend.sma_indicator(df['volume'], window=10)
        
        # إزالة القيم المفقودة
        df.dropna(inplace=True)
        
        return df, None
        
    except Exception as e:
        return None, f"❌ خطأ في جلب البيانات: {str(e)}"

def get_top_gainers_yahoo(limit=10):
    """جلب الأسهم الأكثر ارتفاعاً من Yahoo"""
    symbols = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 
               'NVDA', 'META', 'NFLX', 'JPM', 'VTI',
               'SPY', 'QQQ', 'AMD', 'INTC', 'PYPL']
    results = []
    
    for symbol in symbols:
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            results.append({
                'symbol': symbol,
                'name': info.get('longName', symbol)[:20],
                'price': info.get('regularMarketPrice', 0),
                'change': info.get('regularMarketChangePercent', 0),
                'volume': info.get('regularMarketVolume', 0)
            })
        except:
            continue
    
    # ترتيب حسب التغير
    results.sort(key=lambda x: abs(x['change']), reverse=True)
    return results[:limit]

def get_top_losers_yahoo(limit=10):
    """جلب الأسهم الأكثر انخفاضاً"""
    stocks = get_top_gainers_yahoo(limit*2)
    losers = [s for s in stocks if s['change'] < 0]
    losers.sort(key=lambda x: x['change'])
    return losers[:limit]

# ==========================================
# دوال التحليل
# ==========================================

def analyze_with_local_ai(df):
    """تحليل باستخدام المحرك المحلي"""
    engine = st.session_state['ai_engine']
    return engine.predict_opportunity(df)

def analyze_with_openai(df_summary, api_key, symbol_name):
    """تحليل باستخدام OpenAI"""
    if not api_key:
        return 'HOLD', 0.0, "⚠️ مطلوب مفتاح OpenAI API"
    
    client = OpenAI(api_key=api_key)
    
    prompt = f"""
    أنت خبير تداول. حلل البيانات الفنية لسهم {symbol_name}:
    {df_summary}
    
    أجب بالتنسيق التالي:
    1. السطر الأول: [RECOMMENDATION: BUY] أو [RECOMMENDATION: SELL] أو [RECOMMENDATION: HOLD]
    2. ثم اشرح السبب باختصار (2-3 أسطر)
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
        else:
            return 'HOLD', 50, result
            
    except Exception as e:
        return 'HOLD', 0, f"❌ خطأ في OpenAI: {str(e)}"

def analyze_hybrid(df, api_key, symbol_name):
    """تحليل هجين (محلي + OpenAI)"""
    # التحليل المحلي
    action1, conf1, reason1 = analyze_with_local_ai(df)
    
    # تحليل OpenAI
    if api_key:
        action2, conf2, reason2 = analyze_with_openai(df.tail(10).to_string(), api_key, symbol_name)
        
        # دمج النتائج
        if action1 == action2 and action1 != "HOLD":
            final_action = action1
            final_reason = f"✅ **توافق كامل**: {action1}\n\n"
            final_reason += f"📊 **المحرك المحلي** (ثقة: {conf1}%):\n{reason1}\n\n"
            final_reason += f"🤖 **OpenAI** (ثقة: {conf2}%):\n{reason2}"
            final_conf = (conf1 + conf2) / 2
        elif action1 == "HOLD" and action2 != "HOLD":
            final_action = "HOLD"
            final_reason = f"⚠️ **تباين - انتظار احترازي**\n\n"
            final_reason += f"📊 **المحرك المحلي**: {action1} (ثقة: {conf1}%)\n{reason1}\n\n"
            final_reason += f"🤖 **OpenAI**: {action2} (ثقة: {conf2}%)\n{reason2}"
            final_conf = max(conf1, conf2) * 0.5
        elif action2 == "HOLD" and action1 != "HOLD":
            final_action = "HOLD"
            final_reason = f"⚠️ **تباين - انتظار احترازي**\n\n"
            final_reason += f"📊 **المحرك المحلي**: {action1} (ثقة: {conf1}%)\n{reason1}\n\n"
            final_reason += f"🤖 **OpenAI**: {action2} (ثقة: {conf2}%)\n{reason2}"
            final_conf = max(conf1, conf2) * 0.5
        else:
            final_action = "HOLD"
            final_reason = f"⏸️ **حياد - انتظار**\n\n"
            final_reason += f"📊 **المحرك المحلي**: {action1} (ثقة: {conf1}%)\n{reason1}\n\n"
            final_reason += f"🤖 **OpenAI**: {action2} (ثقة: {conf2}%)\n{reason2}"
            final_conf = 50
    else:
        final_action = action1
        final_reason = reason1
        final_conf = conf1
    
    return final_action, final_conf, final_reason

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
        subplot_titles=(f'📈 {symbol_name}', '📊 RSI')
    )
    
    # الشموع
    fig.add_trace(
        go.Candlestick(
            x=df['date'],
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name='السعر',
            increasing_line_color='#00FF00',
            decreasing_line_color='#FF0000'
        ),
        row=1, col=1
    )
    
    # المتوسطات
    fig.add_trace(
        go.Scatter(x=df['date'], y=df['SMA_20'], 
                  mode='lines', name='SMA 20', 
                  line=dict(color='orange', width=1.5)),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=df['date'], y=df['SMA_50'], 
                  mode='lines', name='SMA 50', 
                  line=dict(color='cyan', width=1.5)),
        row=1, col=1
    )
    
    # RSI
    fig.add_trace(
        go.Scatter(x=df['date'], y=df['RSI'], 
                  mode='lines', name='RSI', 
                  line=dict(color='purple', width=2)),
        row=2, col=1
    )
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
    
    # منطقة RSI الطبيعية
    fig.add_hrect(y0=30, y1=70, line_width=0, 
                  fillcolor="gray", opacity=0.1, row=2, col=1)
    
    fig.update_layout(
        height=550,
        template='plotly_dark',
        showlegend=True,
        hovermode='x unified',
        xaxis_rangeslider_visible=False,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    
    return fig

# ==========================================
# تنفيذ أوامر IBKR
# ==========================================

def execute_ib_order(action, symbol, quantity, host, port):
    """تنفيذ أمر تداول عبر IBKR"""
    if not IBKR_AVAILABLE:
        return "⚠️ IBKR غير مثبت. يرجى تثبيت ib_async أو ib_insync"
    
    ib = IB()
    try:
        client_id = random.randint(1000, 9999)
        ib.connect(host, int(port), clientId=client_id, timeout=5)
        
        contract = Stock(symbol, 'SMART', 'USD')
        ib.qualifyContracts(contract)
        
        order = MarketOrder(action, quantity)
        trade = ib.placeOrder(contract, order)
        
        ib.sleep(1)
        status = trade.orderStatus.status
        ib.disconnect()
        
        if status in ['Filled', 'Submitted']:
            return f"✅ تم إرسال أمر {action} لعدد {quantity} سهم من {symbol} بنجاح!"
        else:
            return f"⚠️ الحالة: {status}"
            
    except Exception as e:
        try:
            ib.disconnect()
        except:
            pass
        return f"❌ فشل تنفيذ الأمر عبر IBKR: {str(e)}"

# ==========================================
# إدارة قائمة الأسهم المفضلة
# ==========================================

DEFAULT_WATCHLIST = [
    "AAPL", "GOOGL", "MSFT", "AMZN", "TSLA",
    "NVDA", "META", "NFLX", "JPM", "VTI"
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
# إعدادات الصفحة
# ==========================================

st.set_page_config(
    page_title="AI Trading Bot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# التطبيق الرئيسي
# ==========================================

def main():
    """التطبيق الرئيسي"""
    
    st.title("🤖 بوت التداول الذكي (Yahoo + IBKR)")
    st.caption("📊 تحليل فني بالذكاء الاصطناعي مع تنفيذ أوامر IBKR")
    
    # تنبيه للمستخدم
    if not IBKR_AVAILABLE:
        st.warning("⚠️ IBKR غير مثبت. سيتم استخدام Yahoo فقط للبيانات والتحليل (بدون تنفيذ)")
    
    # تهيئة المحرك
    if 'ai_engine' not in st.session_state:
        st.session_state['ai_engine'] = LocalAITradingEngine()
    
    # ===== الشريط الجانبي =====
    with st.sidebar:
        st.header("⚙️ الإعدادات")
        
        # === الأسهم المفضلة ===
        st.subheader("⭐ الأسهم المفضلة")
        
        watchlist = load_watchlist()
        
        if watchlist:
            selected_symbol = st.selectbox(
                "اختر من المفضلة:",
                options=watchlist,
                index=0,
                key="symbol_selector"
            )
            
            if st.button("📌 اختيار السهم", use_container_width=True):
                st.session_state['selected_symbol'] = selected_symbol
                st.success(f"✅ تم اختيار {selected_symbol}")
            
            current_symbol = st.session_state.get('selected_symbol', selected_symbol)
            st.info(f"📌 السهم الحالي: **{current_symbol}**")
        else:
            st.warning("⚠️ القائمة فارغة")
            current_symbol = "AAPL"
        
        st.divider()
        
        # === إضافة/حذف من المفضلة ===
        with st.expander("➕ إضافة رمز جديد"):
            new_symbol = st.text_input("رمز السهم:", placeholder="مثل: NVDA", key="new_symbol_input")
            if st.button("إضافة", key="add_symbol_btn", use_container_width=True):
                if new_symbol:
                    if add_to_watchlist(new_symbol.upper()):
                        st.success(f"✅ تم إضافة {new_symbol.upper()}")
                        st.rerun()
                    else:
                        st.warning(f"⚠️ {new_symbol.upper()} موجود بالفعل")
                else:
                    st.warning("⚠️ أدخل رمز السهم")
        
        with st.expander("🗑️ حذف من المفضلة"):
            if watchlist:
                symbol_to_remove = st.selectbox(
                    "اختر رمزاً للحذف:",
                    options=watchlist,
                    index=None,
                    key="remove_selector"
                )
                if symbol_to_remove and st.button("حذف", key="remove_btn", use_container_width=True):
                    if remove_from_watchlist(symbol_to_remove):
                        st.success(f"✅ تم حذف {symbol_to_remove}")
                        st.rerun()
            else:
                st.info("📭 القائمة فارغة")
        
        st.caption(f"📊 إجمالي المفضلة: {len(watchlist)}")
        st.divider()
        
        # === قوائم السوق من Yahoo ===
        st.subheader("📊 قوائم السوق")
        
        col_g1, col_g2 = st.columns(2)
        
        with col_g1:
            if st.button("🔥 الأكثر ارتفاعاً", use_container_width=True):
                with st.spinner("جاري جلب البيانات..."):
                    gainers = get_top_gainers_yahoo(10)
                    if gainers:
                        st.session_state['market_gainers'] = gainers
                        st.success(f"✅ تم جلب {len(gainers)} سهم")
                    else:
                        st.error("❌ فشل جلب البيانات")
        
        with col_g2:
            if st.button("📉 الأكثر انخفاضاً", use_container_width=True):
                with st.spinner("جاري جلب البيانات..."):
                    losers = get_top_losers_yahoo(10)
                    if losers:
                        st.session_state['market_losers'] = losers
                        st.success(f"✅ تم جلب {len(losers)} سهم")
                    else:
                        st.error("❌ فشل جلب البيانات")
        
        # عرض النتائج
        if 'market_gainers' in st.session_state:
            st.divider()
            st.subheader("🔥 الأكثر ارتفاعاً")
            for item in st.session_state['market_gainers'][:8]:
                col_sym, col_chg, col_add = st.columns([2, 1.5, 1])
                with col_sym:
                    st.write(f"• {item['symbol']}")
                with col_chg:
                    st.write(f"🟢 +{item['change']:.1f}%")
                with col_add:
                    if st.button("➕", key=f"add_g_{item['symbol']}"):
                        if add_to_watchlist(item['symbol']):
                            st.success(f"✅ تم إضافة {item['symbol']}")
                            st.rerun()
        
        if 'market_losers' in st.session_state:
            st.divider()
            st.subheader("📉 الأكثر انخفاضاً")
            for item in st.session_state['market_losers'][:8]:
                col_sym, col_chg, col_add = st.columns([2, 1.5, 1])
                with col_sym:
                    st.write(f"• {item['symbol']}")
                with col_chg:
                    st.write(f"🔴 {item['change']:.1f}%")
                with col_add:
                    if st.button("➕", key=f"add_l_{item['symbol']}"):
                        if add_to_watchlist(item['symbol']):
                            st.success(f"✅ تم إضافة {item['symbol']}")
                            st.rerun()
        
        st.divider()
        
        # === إعدادات البيانات ===
        st.subheader("⏱️ إعدادات البيانات")
        selected_period = st.selectbox(
            "الفترة:",
            options=["1d", "5d", "1mo", "3mo", "6mo", "1y"],
            index=1
        )
        selected_interval = st.selectbox(
            "الفاصل الزمني:",
            options=["1m", "2m", "5m", "15m", "30m", "60m", "1d"],
            index=2
        )
        
        st.divider()
        
        # === إعدادات IBKR ===
        st.subheader("🔌 إعدادات IBKR")
        ib_host = st.text_input("عنوان Host", "127.0.0.1")
        ib_port = st.number_input("المنفذ Port", value=7497)
        
        # === OpenAI ===
        api_key = st.text_input("🔑 OpenAI Key (اختياري)", type="password")
        
        # === الكمية ===
        quantity = st.number_input("📊 الكمية", value=10, step=1, min_value=1)
        
        # === وضع التحليل ===
        analysis_mode = st.radio(
            "🧠 وضع التحليل",
            ["المحرك المحلي", "OpenAI", "هجين"]
        )
        
        st.divider()
        
        # === حالة المحرك ===
        st.subheader("📊 حالة المحرك")
        engine = st.session_state['ai_engine']
        if engine.is_trained:
            st.success("✅ النموذج جاهز")
            if hasattr(engine, 'feature_importance') and engine.feature_importance is not None:
                with st.expander("📊 أهم الميزات"):
                    st.dataframe(engine.feature_importance.head(5), use_container_width=True)
        else:
            st.warning("⚠️ غير مدرب (جلب البيانات لتدريبه)")
    
    # ===== السهم النشط =====
    symbol = st.session_state.get('selected_symbol', current_symbol)
    
    # ===== الأعمدة الرئيسية =====
    col1, col2 = st.columns([1.6, 1])
    
    # ===== العمود الأول: البيانات والرسم البياني =====
    with col1:
        st.subheader("📊 البيانات والتحليل الفني")
        st.info(f"📌 السهم النشط: **{symbol}**")
        
        # زر جلب البيانات
        if st.button("🔄 جلب بيانات السوق", use_container_width=True):
            with st.spinner(f"جاري جلب بيانات {symbol}..."):
                df, error = get_market_data(symbol, period=selected_period, interval=selected_interval)
                
                if error:
                    st.error(error)
                else:
                    st.session_state['df'] = df
                    
                    # تدريب النموذج
                    engine = st.session_state['ai_engine']
                    with st.spinner("تدريب النموذج..."):
                        if engine.train_quick_model(df):
                            st.success("✅ تم تدريب النموذج بنجاح!")
                        else:
                            st.warning("⚠️ بيانات غير كافية لتدريب النموذج (يحتاج 30+ شمعة)")
                    
                    st.success(f"✅ تم تحديث {len(df)} شمعة بنجاح")
        
        # عرض البيانات إذا كانت موجودة
        if 'df' in st.session_state:
            df = st.session_state['df']
            
            # السعر الحالي والتغير
            last_price = df['close'].iloc[-1]
            prev_price = df['close'].iloc[-2] if len(df) > 1 else last_price
            change = ((last_price - prev_price) / prev_price) * 100 if prev_price != 0 else 0
            
            col_metric1, col_metric2, col_metric3 = st.columns(3)
            with col_metric1:
                st.metric("💰 السعر الحالي", f"${last_price:.2f}", f"{change:+.2f}%")
            with col_metric2:
                rsi_val = df['RSI'].iloc[-1]
                st.metric("📊 RSI", f"{rsi_val:.1f}", 
                         delta="مفرط شراء" if rsi_val > 70 else ("مفرط بيع" if rsi_val < 30 else "طبيعي"))
            with col_metric3:
                sma20_val = df['SMA_20'].iloc[-1]
                st.metric("📈 SMA 20", f"${sma20_val:.2f}")
            
            # الرسم البياني
            fig = plot_chart(df, symbol)
            st.plotly_chart(fig, use_container_width=True)
            
            # معاينة البيانات
            with st.expander("📋 معاينة البيانات الرقمية"):
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
            
            # إشارة التداول الفورية
            engine = st.session_state['ai_engine']
            if engine.is_trained:
                action, confidence, _ = engine.predict_opportunity(df)
                if action == "BUY":
                    st.success(f"🟢 **إشارة فورية: شراء** (ثقة: {confidence}%)")
                elif action == "SELL":
                    st.error(f"🔴 **إشارة فورية: بيع** (ثقة: {confidence}%)")
                else:
                    st.warning(f"⏸️ **إشارة فورية: انتظار** (ثقة: {confidence}%)")
    
    # ===== العمود الثاني: التحليل والتنفيذ =====
    with col2:
        st.subheader("🤖 التحليل والتنفيذ")
        
        # زر التحليل
        if st.button("🧠 تحليل الفرصة", use_container_width=True, type="primary"):
            if 'df' not in st.session_state:
                st.warning("⚠️ يرجى جلب البيانات أولاً")
            else:
                df = st.session_state['df']
                
                with st.spinner(f"جاري التحليل باستخدام {analysis_mode}..."):
                    if analysis_mode == "المحرك المحلي":
                        action, confidence, result = analyze_with_local_ai(df)
                    elif analysis_mode == "OpenAI":
                        action, confidence, result = analyze_with_openai(
                            df.tail(10).to_string(), api_key, symbol
                        )
                    else:  # هجين
                        action, confidence, result = analyze_hybrid(df, api_key, symbol)
                    
                    st.session_state['result'] = result
                    st.session_state['action'] = action
                    st.session_state['confidence'] = confidence
                    
                    st.success("✅ تم التحليل بنجاح")
        
        # عرض النتيجة
        if 'result' in st.session_state:
            st.divider()
            
            action = st.session_state['action']
            confidence = st.session_state.get('confidence', 0)
            
            # عرض التوصية
            if action == "BUY":
                st.success(f"🟢 **توصية: شراء** (ثقة: {confidence}%)")
            elif action == "SELL":
                st.error(f"🔴 **توصية: بيع** (ثقة: {confidence}%)")
            else:
                st.warning(f"⏸️ **توصية: انتظار** (ثقة: {confidence}%)")
            
            st.text_area("📝 تفاصيل التقرير:", st.session_state['result'], height=180)
            
            st.divider()
            st.subheader("💼 تنفيذ الصفقة")
            
            # أزرار التنفيذ
            if action == "BUY":
                col_exec1, col_exec2 = st.columns(2)
                with col_exec1:
                    if st.button(f"🚀 شراء {quantity} سهم", use_container_width=True, type="primary"):
                        msg = execute_ib_order("BUY", symbol, quantity, ib_host, ib_port)
                        if "✅" in msg:
                            st.success(msg)
                        else:
                            st.error(msg)
                with col_exec2:
                    if st.button("📊 محاكاة", use_container_width=True):
                        last_price = st.session_state['df']['close'].iloc[-1]
                        st.info(f"✅ محاكاة: شراء {quantity} سهم بسعر ${last_price:.2f} = ${last_price * quantity:,.2f}")
            
            elif action == "SELL":
                col_exec1, col_exec2 = st.columns(2)
                with col_exec1:
                    if st.button(f"🔻 بيع {quantity} سهم", use_container_width=True, type="primary"):
                        msg = execute_ib_order("SELL", symbol, quantity, ib_host, ib_port)
                        if "✅" in msg:
                            st.success(msg)
                        else:
                            st.error(msg)
                with col_exec2:
                    if st.button("📊 محاكاة", use_container_width=True):
                        last_price = st.session_state['df']['close'].iloc[-1]
                        st.info(f"✅ محاكاة: بيع {quantity} سهم بسعر ${last_price:.2f} = ${last_price * quantity:,.2f}")
            
            else:
                st.info("⏸️ لا توجد صفقة للتنفيذ")
                
                # عرض تفاصيل إضافية
                if 'df' in st.session_state:
                    df = st.session_state['df']
                    latest = df.iloc[-1]
                    
                    with st.expander("📈 تفاصيل السوق"):
                        col_m1, col_m2, col_m3 = st.columns(3)
                        with col_m1:
                            st.metric("RSI", f"{latest['RSI']:.1f}")
                            st.metric("SMA 20", f"${latest['SMA_20']:.2f}")
                        with col_m2:
                            diff_sma = ((latest['close'] - latest['SMA_20']) / latest['SMA_20']) * 100
                            st.metric("البعد عن SMA20", f"{diff_sma:+.1f}%")
                            st.metric("حجم التداول", f"{latest['volume']:,.0f}")
                        with col_m3:
                            st.metric("SMA 50", f"${latest['SMA_50']:.2f}")
                            st.metric("الكمية", f"{quantity}")
            
            # زر مسح النتائج
            if st.button("🗑️ مسح النتائج", use_container_width=True):
                for key in ['result', 'action', 'confidence']:
                    st.session_state.pop(key, None)
                st.rerun()
    
    # ===== معلومات إضافية =====
    with st.expander("ℹ️ معلومات عن البوت"):
        st.markdown("""
        ### 🤖 بوت التداول بالذكاء الاصطناعي
        
        **المميزات:**
        - 📊 **جلب البيانات**: من Yahoo Finance (مجاني وسريع)
        - 🧠 **محرك محلي**: Random Forest Classifier
        - 🤖 **OpenAI**: دعم GPT-4o-mini للتحليل المتقدم
        - 🔌 **IBKR**: تنفيذ الأوامر عبر Interactive Brokers
        - ⭐ **قائمة مفضلة**: إدارة الأسهم المفضلة
        - 📈 **قوائم السوق**: الأكثر ارتفاعاً/انخفاضاً
        
        **⚠️ تنبيهات:**
        - هذا البوت **للأغراض التعليمية فقط**
        - استخدم الحساب التجريبي (Paper Trading) أولاً
        - لا تخاطر بأكثر مما يمكنك تحمل خسارته
        - راجع أداء النموذج باستمرار
        
        **📚 المتطلبات:**
        - Python 3.9+
        - المكتبات: streamlit, pandas, numpy, plotly, yfinance, ta, openai, scikit-learn
        - (اختياري) IBKR: ib_async أو ib_insync
        """)

# ==========================================
# نقطة الدخول
# ==========================================

if __name__ == "__main__":
    main()
