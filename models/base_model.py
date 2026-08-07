# models/base_model.py
"""
محرك التداول بالذكاء الاصطناعي - النموذج الأساسي
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

class LocalAITradingEngine:
    """
    محرك تداول محلي بالذكاء الاصطناعي
    يستخدم Random Forest للتنبؤ باتجاه السعر
    """
    
    def __init__(self, n_estimators=100, max_depth=6, min_samples=30):
        """
        تهيئة المحرك
        
        Args:
            n_estimators: عدد الأشجار في Random Forest
            max_depth: أقصى عمق للشجرة
            min_samples: الحد الأدنى للعينات للتدريب
        """
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        self.scaler = StandardScaler()
        self.is_trained = False
        self.min_samples = min_samples
        self.feature_importance = None
        self.accuracy = 0
        self.features = []
        self.training_history = []
        
    def extract_features(self, df):
        """
        استخراج الميزات الفنية من البيانات
        
        Args:
            df: DataFrame مع بيانات الشموع
        
        Returns:
            DataFrame مع الميزات المضافة
        """
        data = df.copy()
        
        # ===== الميزات الأساسية =====
        # العوائد (Returns)
        data['returns'] = data['close'].pct_change()
        
        # التقلب (Volatility)
        data['volatility'] = data['returns'].rolling(10).std()
        
        # الفرق بين المتوسطات
        data['ma_diff'] = data['SMA_20'] - data['SMA_50']
        
        # المسافة من SMA20
        data['dist_sma20'] = (data['close'] - data['SMA_20']) / data['SMA_20']
        
        # المسافة من SMA50
        data['dist_sma50'] = (data['close'] - data['SMA_50']) / data['SMA_50']
        
        # زخم RSI
        data['rsi_momentum'] = data['RSI'].diff()
        
        # نسبة الحجم
        data['volume_ratio'] = data['volume'] / data['volume'].rolling(20).mean()
        
        # نطاق السعر
        data['price_range'] = (data['high'] - data['low']) / data['close']
        
        # نسبة الجسم
        data['body_ratio'] = abs(data['close'] - data['open']) / (data['high'] - data['low'] + 0.001)
        
        # ===== ميزات التأخر (Lags) =====
        for lag in [1, 2, 3]:
            data[f'return_lag_{lag}'] = data['returns'].shift(lag)
            data[f'rsi_lag_{lag}'] = data['RSI'].shift(lag)
            data[f'volume_lag_{lag}'] = data['volume'].shift(lag)
        
        # ===== ميزات المتوسطات المتحركة =====
        data['sma_ratio'] = data['SMA_20'] / data['SMA_50']
        data['price_sma_diff'] = (data['close'] - data['SMA_20']) / data['SMA_20']
        
        # قائمة الميزات
        self.features = [
            'RSI', 'SMA_20', 'SMA_50', 'returns', 'volatility',
            'ma_diff', 'dist_sma20', 'dist_sma50', 'rsi_momentum',
            'volume_ratio', 'price_range', 'body_ratio',
            'return_lag_1', 'return_lag_2', 'return_lag_3',
            'rsi_lag_1', 'rsi_lag_2', 'rsi_lag_3',
            'volume_lag_1', 'volume_lag_2', 'volume_lag_3',
            'sma_ratio', 'price_sma_diff'
        ]
        
        return data
    
    def train(self, df, test_size=0.2):
        """
        تدريب النموذج على البيانات
        
        Args:
            df: DataFrame مع بيانات التدريب
            test_size: نسبة بيانات الاختبار
        
        Returns:
            (bool, str): نجاح التدريب والرسالة
        """
        try:
            # استخراج الميزات
            data = self.extract_features(df).dropna()
            
            if len(data) < self.min_samples:
                return False, f"بيانات غير كافية (تحتاج {self.min_samples} شمعة، لديك {len(data)})"
            
            # إعداد الهدف
            data['target'] = np.where(data['close'].shift(-1) > data['close'], 1, 0)
            
            # تجهيز البيانات
            X = data[self.features][:-1]
            y = data['target'][:-1]
            
            if len(X) < 10:
                return False, "بيانات غير كافية للتدريب"
            
            # تقسيم البيانات
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42, shuffle=False
            )
            
            # تطبيع البيانات
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # تدريب النموذج
            self.model.fit(X_train_scaled, y_train)
            
            # تقييم النموذج
            train_score = self.model.score(X_train_scaled, y_train)
            test_score = self.model.score(X_test_scaled, y_test)
            self.accuracy = test_score
            
            # أهمية الميزات
            self.feature_importance = pd.DataFrame({
                'feature': self.features,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            self.is_trained = True
            
            # تسجيل تاريخ التدريب
            self.training_history.append({
                'timestamp': pd.Timestamp.now(),
                'train_accuracy': train_score,
                'test_accuracy': test_score,
                'samples': len(X),
                'features': len(self.features)
            })
            
            return True, f"✅ تم التدريب بنجاح! الدقة: {test_score*100:.1f}%"
            
        except Exception as e:
            return False, f"❌ خطأ في التدريب: {str(e)}"
    
    def predict(self, df):
        """
        التنبؤ بالفرصة باستخدام النموذج المدرب
        
        Args:
            df: DataFrame مع البيانات الحالية
        
        Returns:
            (str, float, str): (الإشارة, الثقة, السبب)
        """
        try:
            data = self.extract_features(df).dropna()
            if data.empty:
                return 'HOLD', 0.0, "بيانات غير كافية"
            
            latest = data.iloc[-1]
            latest_features = np.array(latest[self.features]).reshape(1, -1)
            
            if not self.is_trained:
                return 'HOLD', 50.0, "النموذج غير مدرب"
            
            # تطبيع الميزات
            latest_scaled = self.scaler.transform(latest_features)
            
            # التنبؤ
            prob_up = self.model.predict_proba(latest_scaled)[0][1]
            
            # حساب الثقة باستخدام انحراف معياري للأشجار
            probas = [tree.predict_proba(latest_scaled)[0][1] 
                     for tree in self.model.estimators_]
            std_prob = np.std(probas)
            
            # معايرة الثقة
            uncertainty = std_prob * 0.4
            calibrated_prob = prob_up * (1 - uncertainty)
            calibrated_prob = np.clip(calibrated_prob, 0.25, 0.75)
            
            # المؤشرات الفنية
            rsi = latest['RSI']
            sma20 = latest['SMA_20']
            sma50 = latest['SMA_50']
            price = latest['close']
            volume_ratio = latest['volume_ratio']
            
            confidence = round(calibrated_prob * 100, 1)
            
            # ===== قواعد اتخاذ القرار =====
            # شراء قوي
            if (calibrated_prob > 0.58 and 
                rsi < 60 and 
                price > sma20 and 
                volume_ratio > 0.8):
                
                reason = (
                    f"🚀 **إشارة شراء قوية!**\n"
                    f"• ثقة النموذج: {confidence}%\n"
                    f"• RSI: {rsi:.1f} (منطقة آمنة)\n"
                    f"• السعر فوق SMA20: ${price:.2f} > ${sma20:.2f}\n"
                    f"• حجم التداول: {volume_ratio:.2f}x المتوسط"
                )
                return 'BUY', confidence, reason
            
            # بيع قوي
            elif (calibrated_prob < 0.42 or 
                  rsi > 72 or 
                  (rsi > 65 and calibrated_prob < 0.50)):
                
                reason = (
                    f"🔻 **إشارة بيع!**\n"
                    f"• ثقة النموذج: {confidence}%\n"
                    f"• RSI: {rsi:.1f}" + 
                    (" (تشبع شرائي!)" if rsi > 72 else "") + "\n" +
                    f"• السعر نسبة لـ SMA20: {((price/sma20)-1)*100:+.1f}%"
                )
                return 'SELL', confidence, reason
            
            # شراء معتدل
            elif (calibrated_prob > 0.52 and 
                  rsi < 65 and 
                  price > sma20):
                
                reason = (
                    f"📈 **إشارة شراء معتدلة**\n"
                    f"• ثقة النموذج: {confidence}%\n"
                    f"• RSI: {rsi:.1f}\n"
                    f"• السعر فوق SMA20: ${price:.2f} > ${sma20:.2f}"
                )
                return 'BUY', confidence, reason
            
            # بيع معتدل
            elif (calibrated_prob < 0.48 or 
                  rsi > 65):
                
                reason = (
                    f"📉 **إشارة بيع معتدلة**\n"
                    f"• ثقة النموذج: {confidence}%\n"
                    f"• RSI: {rsi:.1f}"
                )
                return 'SELL', confidence, reason
            
            # انتظار
            else:
                reason = (
                    f"⏸️ **منطقة انتظار**\n"
                    f"• RSI: {rsi:.1f}\n"
                    f"• احتمالية الصعود: {round(calibrated_prob*100,1)}%\n"
                    f"• السوق في حالة تذبذب"
                )
                return 'HOLD', confidence, reason
                
        except Exception as e:
            return 'HOLD', 0.0, f"❌ خطأ في التنبؤ: {str(e)}"
    
    def save(self, path='models/model.pkl'):
        """حفظ النموذج"""
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            joblib.dump(self, path)
            return f"✅ تم حفظ النموذج في {path}"
        except Exception as e:
            return f"❌ فشل الحفظ: {str(e)}"
    
    def load(self, path='models/model.pkl'):
        """تحميل النموذج"""
        try:
            if os.path.exists(path):
                model = joblib.load(path)
                self.model = model.model
                self.scaler = model.scaler
                self.is_trained = model.is_trained
                self.features = model.features
                self.accuracy = model.accuracy
                self.feature_importance = model.feature_importance
                return True
            return False
        except Exception as e:
            print(f"❌ فشل التحميل: {e}")
            return False
    
    def get_info(self):
        """الحصول على معلومات النموذج"""
        return {
            'is_trained': self.is_trained,
            'accuracy': f"{self.accuracy*100:.1f}%" if self.is_trained else "N/A",
            'features_count': len(self.features),
            'samples': self.min_samples,
            'estimators': self.model.n_estimators if self.is_trained else 0
        }
