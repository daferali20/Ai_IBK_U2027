# models/advanced_models.py
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
import xgboost as xgb
import lightgbm as lgb
from .base_model import LocalAITradingEngine
from .model_utils import ModelUtils

class AdvancedTradingModels(LocalAITradingEngine):
    """
    نماذج ذكاء اصطناعي متقدمة للتداول
    """
    
    def __init__(self):
        super().__init__()
        self.models = {}
        self.best_model = None
        self.performance = {}
        
    def create_models(self):
        """إنشاء جميع النماذج المتقدمة"""
        self.models = {
            'random_forest': RandomForestClassifier(
                n_estimators=300,
                max_depth=12,
                min_samples_split=5,
                random_state=42
            ),
            'xgboost': xgb.XGBClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.05,
                random_state=42
            ),
            'lightgbm': lgb.LGBMClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.05,
                random_state=42
            ),
            'gradient_boosting': GradientBoostingClassifier(
                n_estimators=200,
                max_depth=5,
                learning_rate=0.05,
                random_state=42
            ),
            'svm': SVC(
                kernel='rbf',
                C=1.0,
                probability=True,
                random_state=42
            ),
            'neural_network': MLPClassifier(
                hidden_layer_sizes=(100, 50),
                activation='relu',
                max_iter=500,
                random_state=42
            )
        }
        return self.models
    
    def train_all_models(self, X_train, y_train, X_test, y_test):
        """تدريب جميع النماذج ومقارنتها"""
        results = {}
        
        for name, model in self.models.items():
            try:
                model.fit(X_train, y_train)
                accuracy = model.score(X_test, y_test)
                results[name] = accuracy
                print(f"✅ {name}: {accuracy*100:.2f}%")
            except Exception as e:
                print(f"❌ {name}: {e}")
        
        self.performance = results
        self.best_model = max(results, key=results.get)
        self.model = self.models[self.best_model]
        self.is_trained = True
        
        print(f"\n🏆 أفضل نموذج: {self.best_model} ({results[self.best_model]*100:.2f}%)")
        return self.best_model, results
    
    def predict_opportunity(self, df):
        """التنبؤ باستخدام أفضل نموذج"""
        if self.model is None:
            self.create_models()
            return super().predict_opportunity(df)
        
        data = self.extract_features(df).dropna()
        if data.empty:
            return 'HOLD', 0.0, "بيانات غير كافية"
        
        latest = data.iloc[-1]
        features = self.get_feature_names()
        latest_features = np.array(latest[features]).reshape(1, -1)
        
        prob_up = self.model.predict_proba(latest_features)[0][1]
        confidence = round(prob_up * 100, 1)
        
        if prob_up > 0.6:
            return 'BUY', confidence, f"أفضل نموذج ({self.best_model}) يتوقع صعوداً"
        elif prob_up < 0.4:
            return 'SELL', confidence, f"أفضل نموذج ({self.best_model}) يتوقع هبوطاً"
        else:
            return 'HOLD', confidence, f"أفضل نموذج ({self.best_model}) غير حاسم"