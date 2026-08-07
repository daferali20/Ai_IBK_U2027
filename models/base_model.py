# models/base_model.py
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib
import os

class LocalAITradingEngine:
    """محرك تداول محلي بالذكاء الاصطناعي"""
    
    def __init__(self, n_estimators=100, max_depth=6, min_samples=30):
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=5,
            random_state=42
        )
        self.is_trained = False
        self.min_samples = min_samples
        self.feature_importance = None
        self.accuracy = 0
        self.features = []
        
    def extract_features(self, df):
        """استخراج الميزات الفنية"""
        data = df.copy()
        
        # الميزات الأساسية
        data['returns'] = data['close'].pct_change()
        data['volatility'] = data['returns'].rolling(10).std()
        data['ma_diff'] = data['SMA_20'] - data['SMA_50']
        data['dist_sma20'] = (data['close'] - data['SMA_20']) / data['SMA_20']
        data['rsi_momentum'] = data['RSI'].diff()
        data['volume_ratio'] = data['volume'] / data['volume'].rolling(20).mean()
        
        # ميزات التأخر
        for lag in [1, 2]:
            data[f'return_lag_{lag}'] = data['returns'].shift(lag)
            data[f'rsi_lag_{lag}'] = data['RSI'].shift(lag)
        
        # قائمة الميزات
        self.features = [
            'RSI', 'SMA_20', 'SMA_50', 'returns', 'volatility',
            'ma_diff', 'dist_sma20', 'rsi_momentum', 'volume_ratio',
            'return_lag_1', 'return_lag_2', 'rsi_lag_1', 'rsi_lag_2'
        ]
        
        return data
    
    def train(self, df):
        """تدريب النموذج"""
        data = self.extract_features(df).dropna()
        
        if len(data) < self.min_samples:
            return False, f"بيانات غير كافية (تحتاج {self.min_samples} شمعة)"
        
        # الهدف: هل سيرتفع السعر؟
        data['target'] = np.where(data['close'].shift(-1) > data['close'], 1, 0)
        
        X = data[self.features][:-1]
        y = data['target'][:-1]
        
        if len(X) < 10:
            return False, "بيانات غير كافية للتدريب"
        
        # تقسيم البيانات
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # تدريب
        self.model.fit(X_train, y_train)
        
        # تقييم
        self.accuracy = self.model.score(X_test, y_test)
        self.is_trained = True
        
        # أهمية الميزات
        self.feature_importance = pd.DataFrame({
            'feature': self.features,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return True, f"تم التدريب بدقة {self.accuracy*100:.1f}%"
    
    def predict(self, df):
        """التنبؤ بالفرصة"""
        data = self.extract_features(df).dropna()
        if data.empty:
            return 'HOLD', 0.0, "بيانات غير كافية"
        
        latest = data.iloc[-1]
        latest_features = np.array(latest[self.features]).reshape(1, -1)
        
        if not self.is_trained:
            return 'HOLD', 50.0, "النموذج غير مدرب"
        
        # التنبؤ
        prob_up = self.model.predict_proba(latest_features)[0][1]
        
        # حساب الثقة
        probas = [tree.predict_proba(latest_features)[0][1] 
                 for tree in self.model.estimators_]
        std_prob = np.std(probas)
        uncertainty = std_prob * 0.4
        calibrated_prob = prob_up * (1 - uncertainty)
        calibrated_prob = np.clip(calibrated_prob, 0.25, 0.75)
        
        # المؤشرات الفنية
        rsi = latest['RSI']
        sma20 = latest['SMA_20']
        price = latest['close']
        volume_ratio = latest['volume_ratio']
        
        confidence = round(calibrated_prob * 100, 1)
        
        # قرار الشراء
        if calibrated_prob > 0.58 and rsi < 60 and price > sma20 and volume_ratio > 0.8:
            return 'BUY', confidence, f"🚀 شراء - RSI: {rsi:.1f}, السعر فوق SMA20"
        
        # قرار البيع
        elif calibrated_prob < 0.42 or rsi > 72:
            return 'SELL', confidence, f"🔻 بيع - RSI: {rsi:.1f}"
        
        # انتظار
        else:
            return 'HOLD', confidence, f"⏸️ انتظار - RSI: {rsi:.1f}"
    
    def save(self, path='models/model.pkl'):
        """حفظ النموذج"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self, path)
        return f"✅ تم حفظ النموذج في {path}"
    
    def load(self, path='models/model.pkl'):
        """تحميل النموذج"""
        if os.path.exists(path):
            model = joblib.load(path)
            self.model = model.model
            self.is_trained = model.is_trained
            self.features = model.features
            self.accuracy = model.accuracy
            return True
        return False
