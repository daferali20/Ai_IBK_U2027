# models/model_utils.py
"""
أدوات مساعدة للنماذج
"""

import numpy as np
import pandas as pd
import joblib
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings('ignore')

def save_model(model, path='models/model.pkl'):
    """
    حفظ النموذج
    
    Args:
        model: النموذج المراد حفظه
        path: مسار الحفظ
    
    Returns:
        bool: نجاح الحفظ
    """
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(model, path)
        print(f"✅ تم حفظ النموذج في {path}")
        return True
    except Exception as e:
        print(f"❌ فشل الحفظ: {e}")
        return False

def load_model(path='models/model.pkl'):
    """
    تحميل النموذج
    
    Args:
        path: مسار النموذج
    
    Returns:
        النموذج المحمل أو None
    """
    try:
        if os.path.exists(path):
            model = joblib.load(path)
            print(f"✅ تم تحميل النموذج من {path}")
            return model
        print(f"❌ الملف غير موجود: {path}")
        return None
    except Exception as e:
        print(f"❌ فشل التحميل: {e}")
        return None

def evaluate_model(model, X_test, y_test):
    """
    تقييم شامل للنموذج
    
    Args:
        model: النموذج
        X_test: بيانات الاختبار
        y_test: الأهداف الحقيقية
    
    Returns:
        dict: نتائج التقييم
    """
    try:
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
        
    except Exception as e:
        print(f"❌ فشل التقييم: {e}")
        return None

def cross_validate_model(model, X, y, cv=5):
    """
    إجراء التحقق المتقاطع
    
    Args:
        model: النموذج
        X: الميزات
        y: الأهداف
        cv: عدد الطيات
    
    Returns:
        dict: نتائج التحقق المتقاطع
    """
    try:
        scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
        
        results = {
            'mean_accuracy': scores.mean(),
            'std_accuracy': scores.std(),
            'scores': scores.tolist()
        }
        
        print("="*50)
        print(f"📊 التحقق المتقاطع ({cv} طيات):")
        print("="*50)
        print(f"✅ متوسط الدقة: {results['mean_accuracy']*100:.2f}%")
        print(f"✅ الانحراف المعياري: {results['std_accuracy']*100:.2f}%")
        print(f"✅ النتائج: {[f'{s*100:.1f}%' for s in scores]}")
        print("="*50)
        
        return results
        
    except Exception as e:
        print(f"❌ فشل التحقق المتقاطع: {e}")
        return None

def get_feature_importance(model, feature_names):
    """
    استخراج أهمية الميزات
    
    Args:
        model: النموذج المدرب
        feature_names: أسماء الميزات
    
    Returns:
        DataFrame: أهمية الميزات
    """
    try:
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
        
    except Exception as e:
        print(f"❌ فشل استخراج أهمية الميزات: {e}")
        return None

class ModelUtils:
    """فئة أدوات النماذج"""
    
    @staticmethod
    def save_model(model, path='models/model.pkl'):
        return save_model(model, path)
    
    @staticmethod
    def load_model(path='models/model.pkl'):
        return load_model(path)
    
    @staticmethod
    def evaluate_model(model, X_test, y_test):
        return evaluate_model(model, X_test, y_test)
    
    @staticmethod
    def cross_validate_model(model, X, y, cv=5):
        return cross_validate_model(model, X, y, cv)
    
    @staticmethod
    def get_feature_importance(model, feature_names):
        return get_feature_importance(model, feature_names)
