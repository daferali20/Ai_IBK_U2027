import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

class LocalAITradingEngine:
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
        
    def extract_features(self, df):
        data = df.copy()
        data['returns'] = data['close'].pct_change()
        data['volatility'] = data['returns'].rolling(10).std()
        data['ma_diff'] = data['SMA_20'] - data['SMA_50']
        data['dist_sma20'] = (data['close'] - data['SMA_20']) / data['SMA_20']
        data['rsi_momentum'] = data['RSI'].diff()
        data['volume_ratio'] = data['volume'] / data['volume'].rolling(20).mean()
        return data
    
    def get_feature_names(self):
        return ['RSI', 'SMA_20', 'SMA_50', 'returns', 'volatility', 
                'ma_diff', 'dist_sma20', 'rsi_momentum', 'volume_ratio']
    
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
        features = self.get_feature_names()
        latest_features = np.array(latest[features]).reshape(1, -1)
        
        if self.is_trained:
            prob_up = self.model.predict_proba(latest_features)[0][1]
        else:
            prob_up = 0.5
        
        rsi_val = round(latest['RSI'], 2)
        sma20 = latest['SMA_20']
        close_price = latest['close']
        
        if prob_up > 0.6 and rsi_val < 65 and close_price > sma20:
            confidence = round(prob_up * 100, 1)
            return 'BUY', confidence, f"🚀 شراء - RSI: {rsi_val}"
        elif prob_up < 0.4 or rsi_val > 70:
            confidence = round((1 - prob_up) * 100, 1)
            return 'SELL', confidence, f"🔻 بيع - RSI: {rsi_val}"
        else:
            confidence = round(abs(prob_up - 0.5) * 200, 1)
            return 'HOLD', confidence, f"⏸️ انتظار - RSI: {rsi_val}"
