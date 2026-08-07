import numpy as np
import pandas as pd
import ta
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler

class LocalAITradingEngine:
    """
    محرك ذكاء اصطناعي متقدم للتداول يعتمد على هندسة الميزات ونماذج Ensemble Learning
    """
    def __init__(self):
        self.is_trained = False
        self.scaler = StandardScaler()
        # استخدام Random Forest مع Gradient Boosting كمزيج قوي للبيانات المجدولة
        self.rf_model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
        self.gb_model = GradientBoostingClassifier(n_estimators=50, max_depth=3, random_state=42)
        self.feature_cols = []
        self.latest_importance = {}

    def extract_features(self, df_input):
        """استخراج وهندسة الميزات الفنية المعقدة من السلسلة الزمنية"""
        df = df_input.copy()
        
        # 1. العوائد والتذبذب
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        df['volatility_5'] = df['returns'].rolling(5).std()
        df['volatility_20'] = df['returns'].rolling(20).std()
        
        # 2. متوسطات الحركة ونسبها
        df['sma_10'] = ta.trend.sma_indicator(df['close'], window=10)
        df['sma_30'] = ta.trend.sma_indicator(df['close'], window=30)
        df['sma_ratio'] = df['sma_10'] / (df['sma_30'] + 1e-8)
        df['price_to_sma10'] = df['close'] / (df['sma_10'] + 1e-8)
        
        # 3. المؤشرات الزخم والاتجاه
        df['rsi'] = ta.momentum.rsi(df['close'], window=14)
        
        macd = ta.trend.MACD(df['close'])
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        df['macd_diff'] = macd.macd_diff()
        
        # 4. النطاقات والتقلب
        bollinger = ta.volatility.BollingerBands(df['close'])
        df['bb_high'] = bollinger.bollinger_hband()
        df['bb_low'] = bollinger.bollinger_lband()
        df['bb_pband'] = bollinger.bollinger_pband() # نسبة الموقع داخل النطاق
        
        # 5. مؤشرات الأحجام
        df['volume_change'] = df['volume'].pct_change()
        df['volume_sma_ratio'] = df['volume'] / (df['volume'].rolling(10).mean() + 1e-8)
        
        return df

    def prepare_training_data(self, df):
        """إعداد الميزات وتحديد الهدف (Target Labeling)"""
        df_feat = self.extract_features(df)
        
        # التنبؤ باتجاه الحركة المستقبلية للشمعة القادمة (أكبر من 0.1% شراء، أقل من -0.1% بيع)
        future_return = (df_feat['close'].shift(-3) - df_feat['close']) / df_feat['close']
        
        conditions = [
            (future_return > 0.0015),   # BUY = 1
            (future_return < -0.0015)   # SELL = -1
        ]
        choices = [1, -1]
        df_feat['target'] = np.select(conditions, choices, default=0) # HOLD = 0
        
        # تنظيف البيانات الفارغة
        df_feat.dropna(inplace=True)
        
        self.feature_cols = [
            'returns', 'log_returns', 'volatility_5', 'volatility_20',
            'sma_ratio', 'price_to_sma10', 'rsi', 'macd', 'macd_signal',
            'macd_diff', 'bb_pband', 'volume_change', 'volume_sma_ratio'
        ]
        
        X = df_feat[self.feature_cols]
        y = df_feat['target']
        return X, y

    def train_quick_model(self, df):
        """تدريب نماذج الذكاء الاصطناعي على البيانات المدخلة"""
        if len(df) < 50:
            return False
            
        X, y = self.prepare_training_data(df)
        if len(X) < 20:
            return False
            
        # تحجيم الميزات (Normalization)
        X_scaled = self.scaler.fit_transform(X)
        
        # تدريب النماذج
        self.rf_model.fit(X_scaled, y)
        self.gb_model.fit(X_scaled, y)
        
        # حفظ أهمية الميزات للـ Random Forest
        importances = self.rf_model.feature_importances_
        self.latest_importance = dict(zip(self.feature_cols, np.round(importances, 3)))
        
        self.is_trained = True
        return True

    def predict_opportunity(self, df):
        """التنبؤ باتجاه التداول مع إعطاء تقرير تفصيلي ونسبة ثقة"""
        if not self.is_trained:
            success = self.train_quick_model(df)
            if not success:
                return "HOLD", 50, "بيانات غير كافية لتدريب النموذج"

        df_feat = self.extract_features(df)
        latest_row = df_feat[self.feature_cols].tail(1)
        
        if latest_row.isnull().values.any():
            latest_row = latest_row.fillna(0)
            
        latest_scaled = self.scaler.transform(latest_row)
        
        # التنبؤ بالاحتمالات عبر النموذج الهجين (Ensemble Average)
        rf_proba = self.rf_model.predict_proba(latest_scaled)[0]
        gb_proba = self.gb_model.predict_proba(latest_scaled)[0]
        
        classes = list(self.rf_model.classes_)
        avg_proba = (rf_proba + gb_proba) / 2.0
        
        best_class_idx = np.argmax(avg_proba)
        pred_class = classes[best_class_idx]
        confidence = int(avg_proba[best_class_idx] * 100)
        
        # ترجمة التوصية
        if pred_class == 1:
            action = "BUY"
        elif pred_class == -1:
            action = "SELL"
        else:
            action = "HOLD"
            
        # تحديد الأسباب بناءً على أكثر الميزات تأثيراً ومؤشر RSI
        top_feature = max(self.latest_importance, key=self.latest_importance.get)
        rsi_val = df_feat['rsi'].iloc[-1]
        reason = f"تحليل Ensemble (RF+GB): الميزة الأكثر تأثيراً هي [{top_feature}] بمعدل {self.latest_importance[top_feature]}. قيمة RSI الحالية: {rsi_val:.1f}"
        
        return action, confidence, reason
