# utils/data_processor.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler

class DataProcessor:
    """معالجة البيانات وتحضيرها للنماذج"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        
    def clean_data(self, df):
        """تنظيف البيانات"""
        df = df.copy()
        
        # إزالة القيم المفقودة
        df = df.dropna()
        
        # إزالة التكرارات
        df = df.drop_duplicates()
        
        return df
    
    def prepare_features(self, df, feature_list):
        """تحضير الميزات للتدريب"""
        features = df[feature_list].copy()
        
        # تطبيع البيانات
        scaled_features = self.scaler.fit_transform(features)
        
        return pd.DataFrame(scaled_features, columns=feature_list)
    
    def create_sequences(self, data, sequence_length=20):
        """إنشاء تسلسلات للتنبؤ الزمني"""
        X, y = [], []
        for i in range(sequence_length, len(data)):
            X.append(data[i-sequence_length:i])
            y.append(data[i])
        return np.array(X), np.array(y)
    
    def split_data(self, X, y, train_ratio=0.8):
        """تقسيم البيانات تدريب/اختبار"""
        split_idx = int(len(X) * train_ratio)
        
        X_train = X[:split_idx]
        X_test = X[split_idx:]
        y_train = y[:split_idx]
        y_test = y[split_idx:]
        
        return X_train, X_test, y_train, y_test