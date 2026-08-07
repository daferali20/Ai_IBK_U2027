# models/base_model.py
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

class LocalAITradingEngine:
    """
    محرك تداول بالذكاء الاصطناعي - النسخة الأساسية
    """
    def __init__(self, min_samples=50, max_position=0.30):
        self.model = RandomForestClassifier(
            n_estimators=150,
            max_depth=8,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42
        )
        self.is_trained = False
        self.min_samples = min_samples
        self.max_position = max_position
        self.feature_importance = None
        
    def extract_features(self, df):
        """استخراج ميزات محسّنة"""
        data = df.copy()
        
        # الميزات الأساسية
        data['returns'] = data['close'].pct_change()
        data['volatility'] = data['returns'].rolling(10).std()
        data['ma_diff'] = data['SMA_20'] - data['SMA_50']
        data['dist_sma20'] = (data['close'] - data['SMA_20']) / data['SMA_20']
        data['dist_sma50'] = (data['close'] - data['SMA_50']) / data['SMA_50']
        data['rsi_momentum'] = data['RSI'].diff()
        
        # ميزات متقدمة
        data['volume_ratio'] = data['volume'] / data['volume'].rolling(20).mean()
        data['price_range'] = (data['high'] - data['low']) / data['close']
        data['body_ratio'] = abs(data['close'] - data['open']) / (data['high'] - data['low'] + 0.001)
        
        # ميزات التأخر
        for lag in [1, 2]:
            data[f'return_lag_{lag}'] = data['returns'].shift(lag)
            data[f'rsi_lag_{lag}'] = data['RSI'].shift(lag)
        
        return data
    
    def get_feature_names(self):
        """أسماء الميزات المستخدمة"""
        return ['RSI', 'SMA_20', 'SMA_50', 'returns', 'volatility', 
                'ma_diff', 'dist_sma20', 'dist_sma50', 'rsi_momentum',
                'volume_ratio', 'price_range', 'body_ratio',
                'return_lag_1', 'return_lag_2', 'rsi_lag_1', 'rsi_lag_2']
    
    def train_quick_model(self, df):
        """تدريب النموذج مع التحقق من الجودة"""
        data = self.extract_features(df).dropna()
        
        if len(data) < self.min_samples:
            return False
        
        data['target'] = np.where(data['close'].shift(-1) > data['close'], 1, 0)
        
        features = self.get_feature_names()
        X = data[features][:-1]
        y = data['target'][:-1]
        
        if len(X) < 20:
            return False
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        self.model.fit(X_train, y_train)
        accuracy = self.model.score(X_test, y_test)
        
        if accuracy > 0.55:
            self.is_trained = True
            self.feature_importance = pd.DataFrame({
                'feature': features,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            print(f"✅ تم تدريب النموذج بدقة: {accuracy*100:.1f}%")
            return True
        else:
            self.is_trained = False
            print(f"⚠️ النموذج ضعيف (دقة {accuracy*100:.1f}%)")
            return False
    
    def predict_opportunity(self, df):
        """التنبؤ بالفرصة مع حساب الثقة"""
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
    
    def save_model(self, filepath='models/saved_model.pkl'):
        """حفظ النموذج"""
        import joblib
        joblib.dump(self.model, filepath)
        print(f"✅ تم حفظ النموذج في {filepath}")
    
    def load_model(self, filepath='models/saved_model.pkl'):
        """تحميل النموذج"""
        import joblib
        import os
        if os.path.exists(filepath):
            self.model = joblib.load(filepath)
            self.is_trained = True
            print(f"✅ تم تحميل النموذج من {filepath}")
            return True
        return False