# ==========================================
# models/advanced_engine.py
# محرك التداول المتطور V3.0
# ==========================================

import numpy as np
import pandas as pd
import pickle
import os
import json
from datetime import datetime
from typing import Tuple, Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')

# استيراد المكونات من المشروع
from core.config import get_config
from core.constants import DEFAULT_SYMBOL, DEFAULT_PERIOD, DEFAULT_INTERVAL
from data.fetcher import get_market_data
from data.indicators import AdvancedIndicators
from utils.logger import setup_logger
from utils.helpers import safe_divide

# محاولة استيراد مكتبات التعلم الآلي
try:
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import TimeSeriesSplit
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

try:
    from textblob import TextBlob
    NLP_AVAILABLE = True
except ImportError:
    NLP_AVAILABLE = False

# تهيئة السجل
logger = setup_logger(__name__)


class AdvancedAITradingEngine:
    """
    محرك تداول ذكي متطور يجمع بين:
    - Ensemble Learning (RF + GB)
    - LSTM للتنبؤ بالأسعار
    - تحليل المشاعر من الأخبار
    - إدارة مخاطر ديناميكية
    - تحديث النموذج التكيفي
    - تحليل الفريمات المتعددة
    - كشف الأنماط الشارتية
    """
    
    def __init__(self, model_dir: str = "models/saved/"):
        """
        تهيئة المحرك المتطور
        
        Args:
            model_dir: مسار حفظ النماذج
        """
        self.is_trained = False
        self.scaler = StandardScaler()
        self.model_dir = model_dir
        
        # إنشاء مجلد النماذج
        os.makedirs(model_dir, exist_ok=True)
        
        # النماذج الأساسية
        self.rf_model = RandomForestClassifier(
            n_estimators=150,
            max_depth=5,
            min_samples_split=10,
            random_state=42,
            class_weight='balanced',
            n_jobs=-1
        )
        self.gb_model = GradientBoostingClassifier(
            n_estimators=80,
            max_depth=4,
            learning_rate=0.1,
            random_state=42
        )
        
        # النموذج الجماعي
        self.ensemble_model = VotingClassifier(
            estimators=[
                ('rf', self.rf_model),
                ('gb', self.gb_model)
            ],
            voting='soft',
            weights=[1, 1]
        )
        
        # نموذج LSTM
        self.lstm_model = None
        self.lstm_trained = False
        
        # أدوات مساعدة
        self.indicators = AdvancedIndicators()
        
        # تخزين البيانات
        self.training_history = []
        self.performance_history = []
        self.prediction_history = []
        self.latest_importance = {}
        self.patterns_detected = []
        self.feature_importance_history = []
        
        # إعدادات التحليل
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
        
        # قائمة الميزات المحسنة
        self.feature_cols = [
            # العوائد والتذبذب
            'returns', 'log_returns', 'volatility_5', 'volatility_20', 'volatility_50',
            # المتوسطات المتحركة
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
        
        # تحميل النموذج المحفوظ
        self._load_model()
        
        logger.info("✅ تم تهيئة المحرك المتطور بنجاح")
    
    # ==========================================
    # حفظ وتحميل النموذج
    # ==========================================
    def _save_model(self, symbol: str = "") -> bool:
        """
        حفظ النموذج المدرب
        
        Args:
            symbol: رمز السهم للتمييز
        """
        try:
            suffix = f"_{symbol}" if symbol else ""
            model_path = os.path.join(self.model_dir, f"model{suffix}.pkl")
            scaler_path = os.path.join(self.model_dir, f"scaler{suffix}.pkl")
            
            # حفظ النماذج
            with open(model_path, 'wb') as f:
                pickle.dump({
                    'rf_model': self.rf_model,
                    'gb_model': self.gb_model,
                    'feature_cols': self.feature_cols,
                    'latest_importance': self.latest_importance,
                    'is_trained': self.is_trained,
                    'feature_importance_history': self.feature_importance_history,
                    'training_date': datetime.now().isoformat()
                }, f)
            
            # حفظ الـ Scaler
            with open(scaler_path, 'wb') as f:
                pickle.dump(self.scaler, f)
            
            logger.info(f"✅ تم حفظ النموذج لـ {symbol if symbol else 'default'}")
            return True
            
        except Exception as e:
            logger.error(f"❌ خطأ في حفظ النموذج: {e}")
            return False
    
    def _load_model(self, symbol: str = "") -> bool:
        """
        تحميل نموذج محفوظ
        
        Args:
            symbol: رمز السهم للتمييز
        """
        try:
            suffix = f"_{symbol}" if symbol else ""
            model_path = os.path.join(self.model_dir, f"model{suffix}.pkl")
            scaler_path = os.path.join(self.model_dir, f"scaler{suffix}.pkl")
            
            if os.path.exists(model_path) and os.path.exists(scaler_path):
                with open(model_path, 'rb') as f:
                    data = pickle.load(f)
                    self.rf_model = data['rf_model']
                    self.gb_model = data['gb_model']
                    self.feature_cols = data.get('feature_cols', self.feature_cols)
                    self.latest_importance = data.get('latest_importance', {})
                    self.is_trained = data.get('is_trained', False)
                    self.feature_importance_history = data.get('feature_importance_history', [])
                
                with open(scaler_path, 'rb') as f:
                    self.scaler = pickle.load(f)
                
                logger.info(f"✅ تم تحميل النموذج لـ {symbol if symbol else 'default'}")
                return True
                
        except Exception as e:
            logger.warning(f"⚠️ خطأ في تحميل النموذج: {e}")
        
        return False
    
    # ==========================================
    # استخراج الميزات
    # ==========================================
    def extract_advanced_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        استخراج الميزات المتقدمة
        
        Args:
            df: DataFrame مع بيانات السوق
        
        Returns:
            DataFrame مع الميزات المضافة
        """
        return self.indicators.extract_all_features(df)
    
    # ==========================================
    # كشف الأنماط الشارتية
    # ==========================================
    def detect_chart_patterns(self, df: pd.DataFrame) -> List[str]:
        """
        كشف الأنماط الشارتية
        
        Args:
            df: DataFrame مع بيانات السوق
        
        Returns:
            قائمة بالأنماط المكتشفة
        """
        patterns = []
        close = df['close'].values
        
        if len(close) < 50:
            return patterns
        
        try:
            # 1. الكتف والرأس
            peaks = []
            for i in range(5, len(close)-5):
                if close[i] > close[i-1] and close[i] > close[i+1]:
                    peaks.append((i, close[i]))
            
            if len(peaks) >= 3:
                for i in range(len(peaks)-2):
                    left, head, right = peaks[i], peaks[i+1], peaks[i+2]
                    if (left[1] < head[1] and right[1] < head[1] and
                        abs(left[1] - right[1]) / (head[1] + 1e-8) < 0.1):
                        patterns.append('head_shoulders')
                    elif (left[1] > head[1] and right[1] > head[1] and
                          abs(left[1] - right[1]) / (head[1] + 1e-8) < 0.1):
                        patterns.append('inverse_head_shoulders')
            
            # 2. القاع/القمة المزدوجة
            recent = close[-20:]
            if len(recent) >= 10:
                min_idx = np.argmin(recent)
                max_idx = np.argmax(recent)
                
                if len(np.where(recent == recent[min_idx])[0]) >= 2:
                    patterns.append('double_bottom')
                if len(np.where(recent == recent[max_idx])[0]) >= 2:
                    patterns.append('double_top')
            
            # 3. الأعلام والمثلثات
            recent = close[-30:]
            if len(recent) >= 20:
                slope = np.polyfit(range(len(recent)), recent, 1)[0]
                std = np.std(recent)
                
                if abs(slope) < 0.01 * std:
                    patterns.append('flag')
                elif slope > 0.01 * std:
                    patterns.append('ascending_triangle')
                else:
                    patterns.append('descending_triangle')
                    
        except Exception as e:
            logger.warning(f"⚠️ خطأ في كشف الأنماط: {e}")
        
        self.patterns_detected = patterns
        return patterns
    
    # ==========================================
    # تحليل المشاعر
    # ==========================================
    def analyze_sentiment(self, symbol: str = "", api_key: str = None) -> Tuple[float, float, str]:
        """
        تحليل المشاعر من الأخبار
        
        Args:
            symbol: رمز السهم
            api_key: مفتاح NewsAPI
        
        Returns:
            (درجة المشاعر, قوة المشاعر, نص التقرير)
        """
        if not NLP_AVAILABLE:
            return 0, 0, "⚠️ مكتبات تحليل النصوص غير مثبتة"
        
        try:
            sentiment_score = 0
            sentiment_magnitude = 0
            
            if api_key and symbol:
                try:
                    from newsapi import NewsApiClient
                    newsapi = NewsApiClient(api_key=api_key)
                    
                    headlines = newsapi.get_everything(
                        q=symbol,
                        language='en',
                        sort_by='relevancy',
                        page_size=20
                    )
                    
                    total_polarity = 0
                    count = 0
                    
                    for article in headlines.get('articles', [])[:20]:
                        if article.get('title'):
                            blob = TextBlob(article['title'])
                            total_polarity += blob.sentiment.polarity
                            count += 1
                    
                    if count > 0:
                        sentiment_score = total_polarity / count
                        sentiment_magnitude = min(abs(sentiment_score) * 10, 10)
                        
                except Exception as e:
                    logger.warning(f"⚠️ خطأ في جلب الأخبار: {e}")
            
            return sentiment_score, sentiment_magnitude, f"درجة المشاعر: {sentiment_score:.2f}"
            
        except Exception as e:
            logger.error(f"❌ خطأ في تحليل المشاعر: {e}")
            return 0, 0, f"⚠️ خطأ: {str(e)}"
    
    # ==========================================
    # نموذج LSTM
    # ==========================================
    def build_lstm_model(self, input_shape: tuple):
        """بناء نموذج LSTM"""
        if not TF_AVAILABLE:
            return None
        
        try:
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
            
        except Exception as e:
            logger.error(f"❌ خطأ في بناء LSTM: {e}")
            return None
    
    def train_lstm(self, df: pd.DataFrame, lookback: int = 20) -> bool:
        """
        تدريب نموذج LSTM
        
        Args:
            df: DataFrame مع بيانات السوق
            lookback: عدد الخطوات الزمنية للرجوع
        
        Returns:
            True إذا تم التدريب بنجاح
        """
        if not TF_AVAILABLE:
            return False
        
        try:
            prices = df['close'].values
            features = []
            targets = []
            
            for i in range(lookback, len(prices)-1):
                features.append(prices[i-lookback:i])
                targets.append(prices[i+1] - prices[i])
            
            if len(features) < 20:
                return False
            
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
            
            # إعادة تشكيل لـ LSTM
            X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
            X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
            
            # بناء النموذج
            self.lstm_model = self.build_lstm_model((lookback, 1))
            if self.lstm_model is None:
                return False
            
            early_stop = EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            )
            
            self.lstm_model.fit(
                X_train, y_train,
                epochs=50,
                batch_size=32,
                validation_data=(X_test, y_test),
                callbacks=[early_stop],
                verbose=0
            )
            
            self.lstm_trained = True
            logger.info("✅ تم تدريب نموذج LSTM بنجاح")
            return True
            
        except Exception as e:
            logger.error(f"❌ خطأ في تدريب LSTM: {e}")
            return False
    
    def predict_lstm(self, df: pd.DataFrame) -> Optional[float]:
        """
        التنبؤ باستخدام LSTM
        
        Args:
            df: DataFrame مع بيانات السوق
        
        Returns:
            قيمة التنبؤ أو None
        """
        if not self.lstm_trained or not TF_AVAILABLE:
            return None
        
        try:
            lookback = 20
            if len(df) < lookback:
                return None
            
            prices = df['close'].values[-lookback:].reshape(1, lookback, 1)
            prediction = self.lstm_model.predict(prices, verbose=0)[0][0]
            return prediction
            
        except Exception as e:
            logger.error(f"❌ خطأ في تنبؤ LSTM: {e}")
            return None
    
    # ==========================================
    # التنبؤ الرئيسي
    # ==========================================
    def predict_opportunity(self, df: pd.DataFrame) -> Tuple[str, int, str]:
        """
        التنبؤ النهائي باستخدام جميع النماذج
        
        Args:
            df: DataFrame مع بيانات السوق
        
        Returns:
            (الإجراء, درجة الثقة, التقرير)
        """
        try:
            # 1. التحقق من التدريب
            if not self.is_trained:
                success = self.train_quick_model(df)
                if not success:
                    return "HOLD", 50, "⚠️ بيانات غير كافية لتدريب النموذج"
            
            # 2. استخراج الميزات
            df_feat = self.extract_advanced_features(df)
            
            # 3. كشف الأنماط الشارتية
            patterns = self.detect_chart_patterns(df)
            pattern_signal, pattern_confidence = self._calculate_pattern_signal(patterns)
            
            # 4. تحليل المشاعر
            sentiment_score, sentiment_magnitude, sentiment_report = self.analyze_sentiment()
            
            # 5. تنبؤ LSTM
            lstm_prediction = self.predict_lstm(df)
            
            # 6. تنبؤ النماذج الكلاسيكية
            model_prediction, model_confidence = self._predict_ensemble(df_feat)
            
            # 7. دمج الإشارات
            final_action, final_confidence = self._combine_signals(
                model_prediction, model_confidence,
                pattern_signal, pattern_confidence,
                sentiment_score, lstm_prediction
            )
            
            # 8. إدارة المخاطر
            risk_params = self._calculate_risk_parameters(df)
            
            # 9. إنشاء التقرير
            report = self._generate_report(
                df_feat, patterns, pattern_confidence,
                sentiment_score, sentiment_magnitude,
                lstm_prediction, risk_params,
                final_action, final_confidence
            )
            
            # 10. حفظ التاريخ
            self.prediction_history.append({
                'timestamp': datetime.now().isoformat(),
                'action': final_action,
                'confidence': final_confidence,
                'score': final_confidence / 100
            })
            
            return final_action, final_confidence, report
            
        except Exception as e:
            logger.error(f"❌ خطأ في التنبؤ: {e}")
            return "HOLD", 50, f"⚠️ خطأ في التحليل: {str(e)}"
    
    # ==========================================
    # دوال مساعدة للتنبؤ
    # ==========================================
    def _calculate_pattern_signal(self, patterns: List[str]) -> Tuple[float, float]:
        """حساب إشارة الأنماط"""
        signal = 0
        confidence = 0
        
        for pattern in patterns:
            if pattern in self.pattern_weights:
                signal += self.pattern_weights[pattern]
                confidence += abs(self.pattern_weights[pattern])
        
        if confidence > 0 and patterns:
            signal = signal / len(patterns)
            confidence = (confidence / len(patterns)) * 100
        
        return signal, confidence
    
    def _predict_ensemble(self, df_feat: pd.DataFrame) -> Tuple[int, float]:
        """التنبؤ باستخدام النماذج الجماعية"""
        try:
            latest_row = df_feat[self.feature_cols].tail(1)
            latest_clean = np.nan_to_num(latest_row.values, nan=0.0, posinf=0.0, neginf=0.0)
            latest_scaled = self.scaler.transform(latest_clean)
            
            rf_proba = self.rf_model.predict_proba(latest_scaled)[0]
            gb_proba = self.gb_model.predict_proba(latest_scaled)[0]
            
            classes = list(self.rf_model.classes_)
            avg_proba = (rf_proba + gb_proba) / 2.0
            
            best_idx = np.argmax(avg_proba)
            pred_class = classes[best_idx]
            confidence = int(avg_proba[best_idx] * 100)
            
            return pred_class, confidence
            
        except Exception as e:
            logger.error(f"❌ خطأ في تنبؤ الـ Ensemble: {e}")
            return 0, 50
    
    def _combine_signals(self, model_pred: int, model_conf: float,
                         pattern_signal: float, pattern_conf: float,
                         sentiment_score: float, lstm_pred: Optional[float]) -> Tuple[str, int]:
        """دمج جميع الإشارات"""
        final_score = 0
        total_weight = 0
        
        # 1. إشارة النماذج الكلاسيكية (وزن 50%)
        model_signal = 1 if model_pred == 1 else (-1 if model_pred == -1 else 0)
        final_score += model_signal * 0.5 * (model_conf / 100)
        total_weight += 0.5 * (model_conf / 100)
        
        # 2. إشارة الأنماط (وزن 20%)
        final_score += pattern_signal * 0.2
        total_weight += 0.2
        
        # 3. إشارة المشاعر (وزن 10%)
        final_score += sentiment_score * 0.1
        total_weight += 0.1
        
        # 4. إشارة LSTM (وزن 20%)
        if lstm_pred is not None:
            lstm_signal = 1 if lstm_pred > 0 else (-1 if lstm_pred < 0 else 0)
            lstm_confidence = min(abs(lstm_pred) * 100, 100)
            final_score += lstm_signal * 0.2 * (lstm_confidence / 100)
            total_weight += 0.2 * (lstm_confidence / 100)
        
        # حساب النتيجة النهائية
        if total_weight > 0:
            final_score = final_score / total_weight
        else:
            final_score = model_signal
        
        # تحديد الإجراء
        threshold = 0.15
        if final_score > threshold:
            return "BUY", min(int(abs(final_score) * 100), 95)
        elif final_score < -threshold:
            return "SELL", min(int(abs(final_score) * 100), 95)
        else:
            return "HOLD", int((1 - abs(final_score)) * 50)
    
    def _calculate_risk_parameters(self, df: pd.DataFrame) -> Dict:
        """حساب بارامترات المخاطر"""
        returns = df['close'].pct_change().dropna()
        
        if len(returns) < 2:
            return {
                'risk_level': 'Unknown',
                'max_position_size': 0.1,
                'stop_loss': 0.02,
                'risk_per_trade': 0.01
            }
        
        daily_volatility = returns.std() * 100
        
        if daily_volatility > 3:
            return {
                'risk_level': 'High',
                'max_position_size': 0.05,
                'stop_loss': 0.015,
                'risk_per_trade': 0.005
            }
        elif daily_volatility > 1.5:
            return {
                'risk_level': 'Medium',
                'max_position_size': 0.08,
                'stop_loss': 0.025,
                'risk_per_trade': 0.01
            }
        else:
            return {
                'risk_level': 'Low',
                'max_position_size': 0.12,
                'stop_loss': 0.035,
                'risk_per_trade': 0.015
            }
    
    def _generate_report(self, df_feat: pd.DataFrame, patterns: List[str],
                         pattern_conf: float, sentiment_score: float,
                         sentiment_magnitude: float, lstm_pred: Optional[float],
                         risk_params: Dict, action: str, confidence: int) -> str:
        """توليد تقرير مفصل"""
        top_feature = max(self.latest_importance, key=self.latest_importance.get) if self.latest_importance else "N/A"
        
        return f"""
📊 **تحليل متقدم شامل:**

🔹 **المؤشرات التقنية:**
   - أفضل ميزة: [{top_feature}]
   - RSI: {df_feat['rsi'].iloc[-1]:.1f}
   - MACD: {df_feat['macd'].iloc[-1]:.3f}
   - Bollinger %B: {df_feat['bb_pband'].iloc[-1]:.2f}
   - ADX: {df_feat['adx'].iloc[-1]:.1f}

🔹 **الأنماط الشارتية:**
   - الأنماط المكتشفة: {patterns if patterns else 'لا يوجد'}
   - قوة الإشارة: {pattern_conf:.0f}%

🔹 **تحليل المشاعر:**
   - درجة المشاعر: {sentiment_score:.2f}
   - القوة: {sentiment_magnitude:.1f}/10

🔹 **نموذج LSTM:**
   - التنبؤ: {'صاعد 📈' if lstm_pred is not None and lstm_pred > 0 else 'هابط 📉' if lstm_pred is not None else 'غير متاح'}
   - الثقة: {min(abs(lstm_pred) * 100, 100) if lstm_pred is not None else 0:.0f}%

🔹 **إدارة المخاطر:**
   - مستوى المخاطرة: {risk_params['risk_level']}
   - الحد الأقصى للصفقة: {risk_params['max_position_size']*100:.0f}%
   - وقف الخسارة: {risk_params['stop_loss']*100:.1f}%
   - حجم المخاطرة: {risk_params['risk_per_trade']*100:.1f}%

🔹 **الإشارة النهائية:** {'شراء 🟢' if action == 'BUY' else 'بيع 🔴' if action == 'SELL' else 'انتظار ⏸️'}
   - درجة الثقة: {confidence}%
"""
    
    # ==========================================
    # التدريب
    # ==========================================
    def train_quick_model(self, df: pd.DataFrame, symbol: str = "") -> bool:
        """
        تدريب سريع للنماذج
        
        Args:
            df: DataFrame مع بيانات السوق
            symbol: رمز السهم للحفظ
        
        Returns:
            True إذا تم التدريب بنجاح
        """
        if not SKLEARN_AVAILABLE:
            logger.error("❌ مكتبة scikit-learn غير مثبتة")
            return False
        
        if len(df) < 50:
            logger.warning("⚠️ البيانات غير كافية للتدريب")
            return False
        
        try:
            # 1. استخراج الميزات
            df_feat = self.extract_advanced_features(df)
            
            # 2. إنشاء الهدف
            future_return_3 = (df_feat['close'].shift(-3) - df_feat['close']) / df_feat['close']
            future_return_6 = (df_feat['close'].shift(-6) - df_feat['close']) / df_feat['close']
            future_return = future_return_3 * 0.6 + future_return_6 * 0.4
            
            conditions = [
                (future_return > 0.001),
                (future_return < -0.001)
            ]
            choices = [1, -1]
            df_feat['target'] = np.select(conditions, choices, default=0)
            
            # 3. تنظيف البيانات
            clean_cols = self.feature_cols + ['target']
            df_clean = df_feat.dropna(subset=clean_cols).copy()
            
            if len(df_clean) < 20:
                logger.warning("⚠️ البيانات بعد التنظيف غير كافية")
                return False
            
            # 4. تجهيز البيانات للتدريب
            X = df_clean[self.feature_cols]
            y = df_clean['target']
            
            X_clean = np.nan_to_num(X.values, nan=0.0, posinf=0.0, neginf=0.0)
            X_scaled = self.scaler.fit_transform(X_clean)
            
            # 5. تدريب النماذج
            self.rf_model.fit(X_scaled, y)
            self.gb_model.fit(X_scaled, y)
            
            # 6. حفظ أهمية الميزات
            importances = self.rf_model.feature_importances_
            self.latest_importance = dict(zip(self.feature_cols, np.round(importances, 3)))
            self.feature_importance_history.append({
                'date': datetime.now().isoformat(),
                'importance': self.latest_importance.copy()
            })
            
            # 7. تدريب LSTM إذا كان متاحاً
            if TF_AVAILABLE and len(df) > 100:
                self.train_lstm(df)
            
            self.is_trained = True
            
            # 8. حفظ النموذج
            self._save_model(symbol)
            
            logger.info(f"✅ تم تدريب النموذج بنجاح لـ {symbol if symbol else 'default'}")
            return True
            
        except Exception as e:
            logger.error(f"❌ خطأ في تدريب النموذج: {e}")
            return False
    
    # ==========================================
    # مقاييس الأداء
    # ==========================================
    def get_performance_metrics(self) -> Dict:
        """الحصول على مقاييس أداء النموذج"""
        if not self.prediction_history:
            return {"error": "لا توجد تنبؤات سابقة"}
        
        total = len(self.prediction_history)
        buys = sum(1 for p in self.prediction_history if p['action'] == 'BUY')
        sells = sum(1 for p in self.prediction_history if p['action'] == 'SELL')
        holds = sum(1 for p in self.prediction_history if p['action'] == 'HOLD')
        
        return {
            'total_predictions': total,
            'buys': buys,
            'sells': sells,
            'holds': holds,
            'buy_percentage': (buys / total * 100) if total > 0 else 0,
            'sell_percentage': (sells / total * 100) if total > 0 else 0,
            'hold_percentage': (holds / total * 100) if total > 0 else 0,
            'average_confidence': sum(p['confidence'] for p in self.prediction_history) / total if total > 0 else 0
        }
    
    def get_feature_importance(self) -> Dict:
        """الحصول على أهمية الميزات الحالية"""
        return self.latest_importance
    
    def get_training_status(self) -> Dict:
        """الحصول على حالة التدريب"""
        return {
            'is_trained': self.is_trained,
            'lstm_trained': self.lstm_trained,
            'feature_count': len(self.feature_cols),
            'prediction_count': len(self.prediction_history),
            'model_dir': self.model_dir
        }
