# ==========================================
# محرك التحليل المتطور V3.0
# ==========================================

import numpy as np
import pandas as pd
import ta
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
import warnings
warnings.filterwarnings('ignore')

# محاولة استيراد مكتبات التعلم العميق
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("⚠️ TensorFlow غير مثبت - سيتم استخدام النماذج الكلاسيكية فقط")

try:
    from textblob import TextBlob
    import requests
    from newsapi import NewsApiClient
    NLP_AVAILABLE = True
except ImportError:
    NLP_AVAILABLE = False

class AdvancedAITradingEngine:
    """
    محرك تداول ذكي متطور يجمع بين:
    - Ensemble Learning (RF + GB + XGB)
    - LSTM للتنبؤ بالأسعار
    - تحليل المشاعر من الأخبار
    - إدارة مخاطر ديناميكية
    - تحديث النموذج التكيفي
    """
    
    def __init__(self):
        self.is_trained = False
        self.scaler = StandardScaler()
        
        # النماذج الأساسية
        self.rf_model = RandomForestClassifier(
            n_estimators=150, 
            max_depth=5, 
            min_samples_split=10,
            random_state=42,
            class_weight='balanced'
        )
        self.gb_model = GradientBoostingClassifier(
            n_estimators=80, 
            max_depth=4, 
            learning_rate=0.1,
            random_state=42
        )
        
        # النموذج الجماعي (Voting)
        self.ensemble_model = VotingClassifier(
            estimators=[
                ('rf', self.rf_model),
                ('gb', self.gb_model)
            ],
            voting='soft',
            weights=[1, 1]
        )
        
        # نموذج LSTM للتنبؤ بالأسعار
        self.lstm_model = None
        self.lstm_trained = False
        
        # تخزين البيانات التاريخية للتحديث التكيفي
        self.training_history = []
        self.performance_history = []
        
        # إدارة المخاطر
        self.risk_parameters = {
            'max_position_size': 0.1,  # 10% من المحفظة
            'stop_loss': 0.02,          # 2%
            'take_profit': 0.04,        # 4%
            'volatility_adjustment': True,
            'risk_per_trade': 0.01      # 1% من رأس المال
        }
        
        # قائمة الميزات المحسنة
        self.feature_cols = [
            # العوائد والتذبذب
            'returns', 'log_returns', 'volatility_5', 'volatility_20', 'volatility_50',
            # المؤشرات الاتجاهية
            'sma_ratio_10_30', 'sma_ratio_20_50', 'price_to_sma10', 'price_to_sma50',
            # مؤشرات الزخم
            'rsi', 'macd', 'macd_signal', 'macd_diff', 'stoch_k', 'stoch_d',
            'williams_r', 'cci', 'adx', 'dx',
            # البولنجر باندز
            'bb_pband', 'bb_width', 'bb_position',
            # مؤشرات الحجم
            'volume_change', 'volume_sma_ratio', 'volume_breakout',
            # المؤشرات المتقدمة
            'atr', 'obv', 'vwap_diff', 'pivot_distance',
            'fibonacci_retracement', 'support_resistance'
        ]
        
        self.latest_importance = {}
        self.prediction_history = []
        
        # تحليل الفريمات المتعددة
        self.timeframes = {
            '1m': {'weight': 0.2},
            '5m': {'weight': 0.3},
            '15m': {'weight': 0.3},
            '1h': {'weight': 0.2}
        }
        
        # أنماط الشارت
        self.patterns_detected = []
        self.pattern_weights = {
            'double_bottom': 0.8,
            'double_top': -0.8,
            'head_shoulders': -0.9,
            'inverse_head_shoulders': 0.9,
            'bull_flag': 0.6,
            'bear_flag': -0.6,
            'wedge_up': -0.5,
            'wedge_down': 0.5,
            'triangle_ascending': 0.7,
            'triangle_descending': -0.7
        }
        
    # ==========================================
    # 1. استخراج الميزات المتقدمة
    # ==========================================
    def extract_advanced_features(self, df):
        """استخراج أكثر من 40 ميزة فنية متقدمة"""
        df_feat = df.copy()
        
        # ========== العوائد والتذبذب ==========
        df_feat['returns'] = df_feat['close'].pct_change()
        df_feat['log_returns'] = np.log(df_feat['close'] / df_feat['close'].shift(1))
        df_feat['volatility_5'] = df_feat['returns'].rolling(5).std()
        df_feat['volatility_20'] = df_feat['returns'].rolling(20).std()
        df_feat['volatility_50'] = df_feat['returns'].rolling(50).std()
        
        # ========== المتوسطات المتحركة ==========
        df_feat['sma_10'] = ta.trend.sma_indicator(df_feat['close'], window=10)
        df_feat['sma_20'] = ta.trend.sma_indicator(df_feat['close'], window=20)
        df_feat['sma_30'] = ta.trend.sma_indicator(df_feat['close'], window=30)
        df_feat['sma_50'] = ta.trend.sma_indicator(df_feat['close'], window=50)
        df_feat['sma_200'] = ta.trend.sma_indicator(df_feat['close'], window=200)
        
        df_feat['sma_ratio_10_30'] = df_feat['sma_10'] / (df_feat['sma_30'] + 1e-8)
        df_feat['sma_ratio_20_50'] = df_feat['sma_20'] / (df_feat['sma_50'] + 1e-8)
        df_feat['price_to_sma10'] = df_feat['close'] / (df_feat['sma_10'] + 1e-8)
        df_feat['price_to_sma50'] = df_feat['close'] / (df_feat['sma_50'] + 1e-8)
        
        # EMA للاستجابة السريعة
        df_feat['ema_9'] = ta.trend.ema_indicator(df_feat['close'], window=9)
        df_feat['ema_21'] = ta.trend.ema_indicator(df_feat['close'], window=21)
        df_feat['ema_200'] = ta.trend.ema_indicator(df_feat['close'], window=200)
        
        # ========== مؤشرات الزخم والاتجاه ==========
        df_feat['rsi'] = ta.momentum.rsi(df_feat['close'], window=14)
        
        macd = ta.trend.MACD(df_feat['close'])
        df_feat['macd'] = macd.macd()
        df_feat['macd_signal'] = macd.macd_signal()
        df_feat['macd_diff'] = macd.macd_diff()
        
        # Stochastics
        stoch = ta.momentum.StochasticOscillator(
            df_feat['high'], df_feat['low'], df_feat['close'], 
            window=14, smooth_window=3
        )
        df_feat['stoch_k'] = stoch.stoch()
        df_feat['stoch_d'] = stoch.stoch_signal()
        
        # Williams %R
        df_feat['williams_r'] = ta.momentum.WilliamsRIndicator(
            df_feat['high'], df_feat['low'], df_feat['close'], lbp=14
        ).williams_r()
        
        # CCI
        df_feat['cci'] = ta.trend.CCIIndicator(
            df_feat['high'], df_feat['low'], df_feat['close'], window=20
        ).cci()
        
        # ADX
        adx = ta.trend.ADXIndicator(
            df_feat['high'], df_feat['low'], df_feat['close'], window=14
        )
        df_feat['adx'] = adx.adx()
        df_feat['dx'] = adx.adx_neg() / (adx.adx_pos() + 1e-8)
        
        # ========== بولنجر باندز ==========
        bollinger = ta.volatility.BollingerBands(df_feat['close'], window=20, window_dev=2)
        df_feat['bb_high'] = bollinger.bollinger_hband()
        df_feat['bb_mid'] = bollinger.bollinger_mavg()
        df_feat['bb_low'] = bollinger.bollinger_lband()
        df_feat['bb_pband'] = bollinger.bollinger_pband()  # %B
        df_feat['bb_width'] = bollinger.bollinger_wband()  # Band Width
        df_feat['bb_position'] = (df_feat['close'] - df_feat['bb_low']) / (df_feat['bb_high'] - df_feat['bb_low'] + 1e-8)
        
        # ========== مؤشرات الحجم ==========
        df_feat['volume_change'] = df_feat['volume'].pct_change()
        df_feat['volume_sma_ratio'] = df_feat['volume'] / (df_feat['volume'].rolling(20).mean() + 1e-8)
        df_feat['volume_breakout'] = (df_feat['volume'] > 1.5 * df_feat['volume'].rolling(20).mean()).astype(int)
        
        # On-Balance Volume
        df_feat['obv'] = (np.sign(df_feat['close'].diff()) * df_feat['volume']).cumsum()
        df_feat['obv_ma'] = df_feat['obv'].rolling(20).mean()
        
        # ========== المؤشرات المتقدمة ==========
        # Average True Range
        df_feat['atr'] = ta.volatility.average_true_range(
            df_feat['high'], df_feat['low'], df_feat['close'], window=14
        )
        
        # VWAP (حساب تقريبي)
        df_feat['vwap'] = (df_feat['volume'] * (df_feat['high'] + df_feat['low'] + df_feat['close']) / 3).cumsum() / df_feat['volume'].cumsum()
        df_feat['vwap_diff'] = (df_feat['close'] / (df_feat['vwap'] + 1e-8) - 1) * 100
        
        # النقاط المحورية
        df_feat['pivot_high'] = df_feat['high'].rolling(5).max()
        df_feat['pivot_low'] = df_feat['low'].rolling(5).min()
        df_feat['pivot_distance'] = (df_feat['close'] - df_feat['pivot_low']) / (df_feat['pivot_high'] - df_feat['pivot_low'] + 1e-8)
        
        # ========== فيبوناتشي ==========
        high_20 = df_feat['high'].rolling(20).max()
        low_20 = df_feat['low'].rolling(20).min()
        df_feat['fibonacci_retracement'] = (df_feat['close'] - low_20) / (high_20 - low_20 + 1e-8)
        
        # ========== مستويات الدعم والمقاومة ==========
        df_feat['resistance_level'] = df_feat['high'].rolling(10).max()
        df_feat['support_level'] = df_feat['low'].rolling(10).min()
        df_feat['support_resistance'] = (df_feat['close'] - df_feat['support_level']) / (df_feat['resistance_level'] - df_feat['support_level'] + 1e-8)
        
        # تنظيف القيم اللانهائية
        df_feat.replace([np.inf, -np.inf], np.nan, inplace=True)
        
        return df_feat
    
    # ==========================================
    # 2. كشف الأنماط الشارتية
    # ==========================================
    def detect_chart_patterns(self, df):
        """كشف الأنماط الشارتية الكلاسيكية"""
        patterns = []
        close = df['close'].values
        high = df['high'].values
        low = df['low'].values
        n = len(close)
        
        if n < 50:
            return patterns
        
        # 1. الكتف والرأس (Head and Shoulders)
        def detect_head_shoulders(data):
            peaks = []
            for i in range(5, len(data)-5):
                if data[i] > data[i-1] and data[i] > data[i+1]:
                    peaks.append((i, data[i]))
            
            if len(peaks) >= 3:
                for i in range(len(peaks)-2):
                    left = peaks[i]
                    head = peaks[i+1]
                    right = peaks[i+2]
                    if (left[1] < head[1] and right[1] < head[1] and
                        abs(left[1] - right[1]) / head[1] < 0.1):
                        return 'head_shoulders'
                    elif (left[1] > head[1] and right[1] > head[1] and
                          abs(left[1] - right[1]) / head[1] < 0.1):
                        return 'inverse_head_shoulders'
            return None
        
        # 2. القاع المزدوج والقمة المزدوجة
        def detect_double_pattern(data):
            recent = data[-20:]
            if len(recent) < 10:
                return None
            lows = np.array([i for i in range(len(recent)) if recent[i] == min(recent[:i+1])])
            highs = np.array([i for i in range(len(recent)) if recent[i] == max(recent[:i+1])])
            
            if len(lows) >= 2 and len(highs) >= 2:
                # قاع مزدوج
                if abs(recent[lows[-1]] - recent[lows[-2]]) / recent[lows[-1]] < 0.02:
                    return 'double_bottom'
                # قمة مزدوجة
                if abs(recent[highs[-1]] - recent[highs[-2]]) / recent[highs[-1]] < 0.02:
                    return 'double_top'
            return None
        
        # 3. الأعلام والمثلثات
        def detect_flags_and_triangles(data):
            if len(data) < 20:
                return None
            recent = data[-30:]
            slope = np.polyfit(range(len(recent)), recent, 1)[0]
            std = np.std(recent)
            
            if abs(slope) < 0.01 * std:  # أفقي
                return 'flag'
            elif slope > 0.01 * std:  # صاعد
                return 'ascending_triangle'
            else:  # هابط
                return 'descending_triangle'
            return None
        
        # تطبيق الكشف على آخر 100 شمعة
        pattern = detect_head_shoulders(close[-100:])
        if pattern:
            patterns.append(pattern)
            
        pattern = detect_double_pattern(close)
        if pattern:
            patterns.append(pattern)
            
        pattern = detect_flags_and_triangles(close)
        if pattern:
            patterns.append(pattern)
        
        self.patterns_detected = patterns
        return patterns
    
    # ==========================================
    # 3. تحليل المشاعر (Sentiment Analysis)
    # ==========================================
    def analyze_sentiment(self, symbol, api_key=None):
        """تحليل المشاعر من الأخبار والتويتر"""
        sentiment_score = 0
        sentiment_magnitude = 0
        
        if not NLP_AVAILABLE:
            return 0, 0, "⚠️ مكتبات تحليل النصوص غير مثبتة"
        
        try:
            # جلب الأخبار من NewsAPI
            if api_key:
                newsapi = NewsApiClient(api_key=api_key)
                headlines = newsapi.get_everything(
                    q=symbol,
                    language='en',
                    sort_by='relevancy',
                    page_size=50
                )
                
                total_polarity = 0
                count = 0
                
                for article in headlines['articles'][:20]:
                    if article['title']:
                        blob = TextBlob(article['title'])
                        total_polarity += blob.sentiment.polarity
                        count += 1
                
                if count > 0:
                    sentiment_score = total_polarity / count
                    sentiment_magnitude = min(abs(sentiment_score) * 10, 10)
                    
        except Exception as e:
            print(f"⚠️ خطأ في تحليل المشاعر: {e}")
            
        return sentiment_score, sentiment_magnitude, f"درجة المشاعر: {sentiment_score:.2f}"
    
    # ==========================================
    # 4. نموذج LSTM للتنبؤ بالأسعار
    # ==========================================
    def build_lstm_model(self, input_shape):
        """بناء نموذج LSTM للتنبؤ بالأسعار"""
        if not TF_AVAILABLE:
            return None
            
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            LSTM(50, return_sequences=False),
            Dropout(0.2),
            Dense(25, activation='relu'),
            Dense(1, activation='linear')
        ])
        
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        return model
    
    def prepare_lstm_data(self, df, lookback=20):
        """تجهيز البيانات لنموذج LSTM"""
        if not TF_AVAILABLE:
            return None, None, None
            
        prices = df['close'].values
        features = []
        targets = []
        
        for i in range(lookback, len(prices)-1):
            features.append(prices[i-lookback:i])
            targets.append(prices[i+1] - prices[i])  # التنبؤ بالتغير
        
        features = np.array(features)
        targets = np.array(targets)
        
        # تطبيع البيانات
        features_mean = features.mean(axis=1, keepdims=True)
        features_std = features.std(axis=1, keepdims=True) + 1e-8
        features = (features - features_mean) / features_std
        
        # تقسيم البيانات
        split = int(len(features) * 0.8)
        X_train = features[:split]
        X_test = features[split:]
        y_train = targets[:split]
        y_test = targets[split:]
        
        # إعادة تشكيل لـ LSTM [samples, timesteps, features]
        X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
        X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
        
        return X_train, X_test, y_train, y_test
    
    def train_lstm(self, df):
        """تدريب نموذج LSTM"""
        if not TF_AVAILABLE:
            return False
            
        X_train, X_test, y_train, y_test = self.prepare_lstm_data(df)
        if X_train is None or len(X_train) < 20:
            return False
        
        self.lstm_model = self.build_lstm_model((X_train.shape[1], 1))
        
        early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        
        history = self.lstm_model.fit(
            X_train, y_train,
            epochs=50,
            batch_size=32,
            validation_data=(X_test, y_test),
            callbacks=[early_stop],
            verbose=0
        )
        
        self.lstm_trained = True
        return True
    
    def predict_lstm(self, df):
        """التنبؤ باستخدام LSTM"""
        if not self.lstm_trained or not TF_AVAILABLE:
            return None
            
        lookback = 20
        prices = df['close'].values[-lookback:].reshape(1, lookback, 1)
        prediction = self.lstm_model.predict(prices, verbose=0)[0][0]
        return prediction
    
    # ==========================================
    # 5. تحليل الفريمات المتعددة
    # ==========================================
    def analyze_multiple_timeframes(self, symbol, intervals):
        """تحليل السهم عبر أطر زمنية مختلفة"""
        signals = {}
        weights = self.timeframes
        
        for interval, weight_info in weights.items():
            try:
                if interval in intervals:
                    data, _ = get_market_data(symbol, period="5d", interval=interval)
                    if data is not None and len(data) > 30:
                        signal = self.predict_opportunity(data)
                        signals[interval] = {
                            'signal': signal[0],
                            'confidence': signal[1],
                            'weight': weight_info['weight']
                        }
            except Exception as e:
                print(f"⚠️ خطأ في تحليل الفريم {interval}: {e}")
                
        # حساب الإشارة المجمعة
        weighted_signal = 0
        total_weight = 0
        
        for interval, data in signals.items():
            signal_val = 1 if data['signal'] == 'BUY' else (-1 if data['signal'] == 'SELL' else 0)
            weighted_signal += signal_val * data['weight'] * (data['confidence'] / 100)
            total_weight += data['weight'] * (data['confidence'] / 100)
        
        if total_weight > 0:
            final_score = weighted_signal / total_weight
            if final_score > 0.3:
                return 'BUY', int(abs(final_score) * 100)
            elif final_score < -0.3:
                return 'SELL', int(abs(final_score) * 100)
        
        return 'HOLD', 50
    
    # ==========================================
    # 6. إدارة المخاطر المتقدمة
    # ==========================================
    def calculate_risk_metrics(self, df):
        """حساب مقاييس المخاطر"""
        returns = df['close'].pct_change().dropna()
        
        # العوائد اليومية
        daily_returns = returns.mean() * 100
        daily_volatility = returns.std() * 100
        
        # VaR (95% confidence)
        var_95 = np.percentile(returns, 5) * 100
        
        # Sharpe Ratio (افتراضياً 0% خالية من المخاطر)
        sharpe = (daily_returns / (daily_volatility + 1e-8)) * np.sqrt(252)
        
        # Maximum Drawdown
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative / running_max - 1) * 100
        max_drawdown = drawdown.min()
        
        return {
            'daily_return': daily_returns,
            'daily_volatility': daily_volatility,
            'var_95': var_95,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown,
            'risk_level': 'High' if daily_volatility > 3 else ('Medium' if daily_volatility > 1.5 else 'Low')
        }
    
    def adjust_risk_parameters(self, df):
        """ضبط بارامترات المخاطر بناءً على التقلبات الحالية"""
        risk_metrics = self.calculate_risk_metrics(df)
        
        if risk_metrics['risk_level'] == 'High':
            self.risk_parameters['max_position_size'] = 0.05  # 5%
            self.risk_parameters['stop_loss'] = 0.015         # 1.5%
            self.risk_parameters['risk_per_trade'] = 0.005    # 0.5%
        elif risk_metrics['risk_level'] == 'Medium':
            self.risk_parameters['max_position_size'] = 0.08  # 8%
            self.risk_parameters['stop_loss'] = 0.025         # 2.5%
            self.risk_parameters['risk_per_trade'] = 0.01     # 1%
        else:  # Low
            self.risk_parameters['max_position_size'] = 0.12  # 12%
            self.risk_parameters['stop_loss'] = 0.035         # 3.5%
            self.risk_parameters['risk_per_trade'] = 0.015    # 1.5%
            
        return self.risk_parameters
    
    # ==========================================
    # 7. التحديث التكيفي للنموذج (Online Learning)
    # ==========================================
    def adaptive_update(self, df, new_data):
        """تحديث النموذج تدريجياً مع البيانات الجديدة"""
        if not self.is_trained:
            return self.train_quick_model(df)
        
        # إضافة البيانات الجديدة للتاريخ
        self.training_history.append(new_data)
        if len(self.training_history) > 100:
            self.training_history.pop(0)
        
        # إعادة التدريب كل 50 شمعة جديدة
        if len(self.training_history) % 20 == 0:
            # جمع كل البيانات
            combined_df = pd.concat([df, pd.DataFrame(self.training_history)], ignore_index=True)
            self.train_quick_model(combined_df.tail(500))  # تدريب على آخر 500 شمعة فقط
        
        return True
    
    # ==========================================
    # 8. التنبؤ النهائي المتكامل
    # ==========================================
    def predict_opportunity(self, df):
        """التنبؤ النهائي باستخدام جميع النماذج"""
        if not self.is_trained:
            success = self.train_quick_model(df)
            if not success:
                return "HOLD", 50, "بيانات غير كافية لتدريب النموذج"
        
        # 1. استخراج الميزات
        df_feat = self.extract_advanced_features(df)
        
        # 2. كشف الأنماط الشارتية
        patterns = self.detect_chart_patterns(df)
        pattern_signal = 0
        pattern_confidence = 0
        for pattern in patterns:
            if pattern in self.pattern_weights:
                pattern_signal += self.pattern_weights[pattern]
                pattern_confidence += abs(self.pattern_weights[pattern])
        if pattern_confidence > 0:
            pattern_signal = pattern_signal / len(patterns)
            pattern_confidence = (pattern_confidence / len(patterns)) * 100
        
        # 3. تحليل المشاعر (إذا كان متاحاً)
        sentiment_score, sentiment_magnitude, _ = self.analyze_sentiment("")
        
        # 4. التنبؤ بالـ LSTM
        lstm_prediction = self.predict_lstm(df) if self.lstm_trained else None
        
        # 5. التنبؤ بالنماذج الكلاسيكية
        latest_row = df_feat[self.feature_cols].tail(1)
        latest_clean = np.nan_to_num(latest_row.values, nan=0.0, posinf=0.0, neginf=0.0)
        latest_scaled = self.scaler.transform(latest_clean)
        
        rf_proba = self.rf_model.predict_proba(latest_scaled)[0]
        gb_proba = self.gb_model.predict_proba(latest_scaled)[0]
        
        classes = list(self.rf_model.classes_)
        avg_proba = (rf_proba + gb_proba) / 2.0
        
        best_class_idx = np.argmax(avg_proba)
        pred_class = classes[best_class_idx]
        model_confidence = int(avg_proba[best_class_idx] * 100)
        
        # 6. دمج جميع الإشارات (الباقي)
        final_score = 0
        total_weight = 0
        
        # إشارة النماذج الكلاسيكية (وزن 50%)
        model_signal = 1 if pred_class == 1 else (-1 if pred_class == -1 else 0)
        final_score += model_signal * 0.5 * (model_confidence / 100)
        total_weight += 0.5 * (model_confidence / 100)
        
        # إشارة الأنماط الشارتية (وزن 20%)
        final_score += pattern_signal * 0.2
        total_weight += 0.2
        
        # إشارة المشاعر (وزن 10%)
        final_score += sentiment_score * 0.1
        total_weight += 0.1
        
        # إشارة LSTM (وزن 20%)
        if lstm_prediction is not None:
            lstm_signal = 1 if lstm_prediction > 0 else (-1 if lstm_prediction < 0 else 0)
            lstm_confidence = min(abs(lstm_prediction) * 100, 100)
            final_score += lstm_signal * 0.2 * (lstm_confidence / 100)
            total_weight += 0.2 * (lstm_confidence / 100)
        
        # حساب الإشارة النهائية
        if total_weight > 0:
            final_score = final_score / total_weight
        else:
            final_score = model_signal
        
        # تحديد الإجراء النهائي
        threshold = 0.15  # عتبة للحماية من الإشارات الضعيفة
        if final_score > threshold:
            action = "BUY"
            confidence = min(int(abs(final_score) * 100), 95)
        elif final_score < -threshold:
            action = "SELL"
            confidence = min(int(abs(final_score) * 100), 95)
        else:
            action = "HOLD"
            confidence = int((1 - abs(final_score)) * 50)
        
        # 7. ضبط بارامترات المخاطر
        risk_params = self.adjust_risk_parameters(df)
        
        # 8. توليد تقرير مفصل
        top_feature = max(self.latest_importance, key=self.latest_importance.get) if self.latest_importance else "N/A"
        
        reason = f"""
📊 **تحليل متقدم شامل:**

🔹 **المؤشرات التقنية:**
   - أفضل ميزة: [{top_feature}]
   - RSI: {df_feat['rsi'].iloc[-1]:.1f}
   - MACD: {df_feat['macd'].iloc[-1]:.3f}
   - Bollinger %B: {df_feat['bb_pband'].iloc[-1]:.2f}

🔹 **الأنماط الشارتية:**
   - الأنماط المكتشفة: {patterns if patterns else 'لا يوجد'}
   - قوة الإشارة: {pattern_confidence:.0f}%

🔹 **تحليل المشاعر:**
   - درجة المشاعر: {sentiment_score:.2f}
   - القوة: {sentiment_magnitude:.1f}/10

🔹 **نموذج LSTM:**
   - التنبؤ: {'صاعد' if lstm_prediction is not None and lstm_prediction > 0 else 'هابط' if lstm_prediction is not None else 'غير متاح'}
   - الثقة: {min(abs(lstm_prediction) * 100, 100) if lstm_prediction is not None else 0:.0f}%

🔹 **إدارة المخاطر:**
   - مستوى المخاطرة: {risk_params['risk_level']}
   - الحد الأقصى للصفقة: {risk_params['max_position_size']*100:.0f}%
   - وقف الخسارة: {risk_params['stop_loss']*100:.1f}%
   - حجم المخاطرة لكل صفقة: {risk_params['risk_per_trade']*100:.1f}%

🔹 **الإشارة النهائية:** {'شراء 🟢' if action == 'BUY' else 'بيع 🔴' if action == 'SELL' else 'انتظار ⏸️'}
   - درجة الثقة: {confidence}%
   """
        
        self.prediction_history.append({
            'timestamp': pd.Timestamp.now(),
            'action': action,
            'confidence': confidence,
            'score': final_score,
            'features': {col: df_feat[col].iloc[-1] for col in self.feature_cols[:5]}
        })
        
        return action, confidence, reason
    
    def train_quick_model(self, df):
        """تدريب سريع للنماذج"""
        if len(df) < 50:
            return False
        
        df_feat = self.extract_advanced_features(df)
        
        # إنشاء الهدف - استراتيجية محسنة
        future_return_3 = (df_feat['close'].shift(-3) - df_feat['close']) / df_feat['close']
        future_return_6 = (df_feat['close'].shift(-6) - df_feat['close']) / df_feat['close']
        future_return = future_return_3 * 0.6 + future_return_6 * 0.4  # دمج أطر زمنية متعددة
        
        conditions = [
            (future_return > 0.001),   # BUY
            (future_return < -0.001)   # SELL
        ]
        choices = [1, -1]
        df_feat['target'] = np.select(conditions, choices, default=0)
        
        # تنظيف البيانات
        clean_cols = self.feature_cols + ['target']
        df_clean = df_feat.dropna(subset=clean_cols).copy()
        
        # التحقق من وجود بيانات كافية
        if len(df_clean) < 20:
            return False
            
        X = df_clean[self.feature_cols]
        y = df_clean['target']
        
        # تنظيف البيانات
        X_clean = np.nan_to_num(X.values, nan=0.0, posinf=0.0, neginf=0.0)
        
        # تحجيم الميزات
        X_scaled = self.scaler.fit_transform(X_clean)
        
        # تدريب النماذج
        self.rf_model.fit(X_scaled, y)
        self.gb_model.fit(X_scaled, y)
        
        # حفظ أهمية الميزات
        importances = self.rf_model.feature_importances_
        self.latest_importance = dict(zip(self.feature_cols, np.round(importances, 3)))
        
        # تدريب LSTM إذا كان متاحاً
        if TF_AVAILABLE and len(df) > 100:
            self.train_lstm(df)
        
        self.is_trained = True
        return True

# ==========================================
# دالة مساعدة لجلب البيانات من Yahoo
# ==========================================
def get_market_data(symbol, period="5d", interval="5m"):
    """جلب بيانات السوق من Yahoo Finance"""
    try:
        import yfinance as yf
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period, interval=interval)
        
        if df.empty or len(df) < 20:
            return None, f"❌ البيانات غير كافية للرمز: {symbol}"
        
        df.rename(columns={
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Volume': 'volume'
        }, inplace=True)
        
        df.index = df.index.tz_localize(None)
        df['date'] = df.index
        
        return df, None
        
    except Exception as e:
        return None, f"❌ خطأ أثناء جلب البيانات: {str(e)}"

# ==========================================
# مثال على استخدام المحرك
# ==========================================
if __name__ == "__main__":
    # إنشاء المحرك
    engine = AdvancedAITradingEngine()
    
    # جلب البيانات
    symbol = "AAPL"
    df, error = get_market_data(symbol, period="1mo", interval="5m")
    
    if df is not None:
        # تدريب النموذج
        engine.train_quick_model(df)
        
        # التنبؤ
        action, confidence, report = engine.predict_opportunity(df)
        
        print(f"📊 الإشارة: {action}")
        print(f"📈 الثقة: {confidence}%")
        print(f"📝 التقرير:\n{report}")
