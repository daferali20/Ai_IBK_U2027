# models/model_utils.py
import numpy as np
import pandas as pd
import joblib
import os
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report

class ModelUtils:
    """أدوات مساعدة للنماذج"""
    
    @staticmethod
    def save_model(model, filepath):
        """حفظ النموذج"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(model, filepath)
        print(f"✅ تم حفظ النموذج في {filepath}")
    
    @staticmethod
    def load_model(filepath):
        """تحميل النموذج"""
        if os.path.exists(filepath):
            model = joblib.load(filepath)
            print(f"✅ تم تحميل النموذج من {filepath}")
            return model
        return None
    
    @staticmethod
    def evaluate_model(model, X_test, y_test):
        """تقييم شامل للنموذج"""
        y_pred = model.predict(X_test)
        
        results = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted'),
            'recall': recall_score(y_test, y_pred, average='weighted'),
            'f1_score': f1_score(y_test, y_pred, average='weighted'),
            'confusion_matrix': confusion_matrix(y_test, y_pred)
        }
        
        print("="*50)
        print("📊 تقييم النموذج:")
        print("="*50)
        print(f"✅ الدقة: {results['accuracy']*100:.2f}%")
        print(f"✅ Precision: {results['precision']*100:.2f}%")
        print(f"✅ Recall: {results['recall']*100:.2f}%")
        print(f"✅ F1 Score: {results['f1_score']*100:.2f}%")
        print("="*50)
        
        return results
    
    @staticmethod
    def optimize_model(model, param_grid, X_train, y_train):
        """تحسين معلمات النموذج"""
        grid_search = GridSearchCV(
            model,
            param_grid,
            cv=5,
            scoring='accuracy',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        print(f"✅ أفضل معلمات: {grid_search.best_params_}")
        print(f"✅ أفضل دقة: {grid_search.best_score_*100:.2f}%")
        
        return grid_search.best_estimator_
    
    @staticmethod
    def get_feature_importance(model, feature_names):
        """استخراج أهمية الميزات"""
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importance = np.abs(model.coef_[0])
        else:
            return None
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        return importance_df