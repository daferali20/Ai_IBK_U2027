# models/ensemble_models.py
import numpy as np
from sklearn.ensemble import VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
import lightgbm as lgb
from .base_model import LocalAITradingEngine

class EnsembleTradingModel(LocalAITradingEngine):
    """
    نماذج دمج (Ensemble) للتداول
    """
    
    def __init__(self):
        super().__init__()
        self.voting_model = None
        self.stacking_model = None
        
    def create_voting_model(self):
        """إنشاء نموذج تصويت"""
        self.voting_model = VotingClassifier(
            estimators=[
                ('rf', RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)),
                ('xgb', xgb.XGBClassifier(n_estimators=150, max_depth=5, random_state=42)),
                ('lgb', lgb.LGBMClassifier(n_estimators=150, max_depth=5, random_state=42))
            ],
            voting='soft'  # 'hard' للتصويت الصارم، 'soft' للتصويت المرجح
        )
        return self.voting_model
    
    def create_stacking_model(self):
        """إنشاء نموذج Stacking"""
        base_models = [
            ('rf', RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)),
            ('xgb', xgb.XGBClassifier(n_estimators=150, max_depth=5, random_state=42)),
            ('lgb', lgb.LGBMClassifier(n_estimators=150, max_depth=5, random_state=42))
        ]
        
        self.stacking_model = StackingClassifier(
            estimators=base_models,
            final_estimator=LogisticRegression(),
            cv=5
        )
        return self.stacking_model
    
    def train_ensemble(self, X_train, y_train, X_test, y_test):
        """تدريب نماذج الدمج"""
        results = {}
        
        # نموذج التصويت
        self.create_voting_model()
        self.voting_model.fit(X_train, y_train)
        voting_acc = self.voting_model.score(X_test, y_test)
        results['voting'] = voting_acc
        print(f"✅ Voting: {voting_acc*100:.2f}%")
        
        # نموذج Stacking
        self.create_stacking_model()
        self.stacking_model.fit(X_train, y_train)
        stacking_acc = self.stacking_model.score(X_test, y_test)
        results['stacking'] = stacking_acc
        print(f"✅ Stacking: {stacking_acc*100:.2f}%")
        
        # اختيار الأفضل
        best = max(results, key=results.get)
        if best == 'voting':
            self.model = self.voting_model
        else:
            self.model = self.stacking_model
        
        self.is_trained = True
        print(f"\n🏆 أفضل نموذج دمج: {best} ({results[best]*100:.2f}%)")
        
        return results
    
    def predict_opportunity(self, df):
        """التنبؤ باستخدام نموذج الدمج"""
        if self.model is None:
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
            return 'BUY', confidence, "نموذج الدمج يتوقع صعوداً"
        elif prob_up < 0.4:
            return 'SELL', confidence, "نموذج الدمج يتوقع هبوطاً"
        else:
            return 'HOLD', confidence, "نموذج الدمج غير حاسم"