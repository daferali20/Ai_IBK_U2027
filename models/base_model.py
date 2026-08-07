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
        self.rf_model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
        self.gb_model = GradientBoostingClassifier(n_estimators=50, max_depth=3, random_state=42)
        self.feature_cols = [
            'returns', 'log_returns', 'volatility_5', 'volatility_20',
            'sma_ratio', 'price_to_sma10', 'rsi', 'macd', 'macd_signal',
            'macd_diff', 'bb_pband', 'volume_change', 'volume_sma_ratio'
        ]
        self.latest_importance = {}

    def extract_features(self, df_input):
        """استخراج وهندسة الميزات الفنية مع تنظيف القيم اللانهائية"""
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
        
        # 3. مؤشرات الزخم والاتجاه
        df['rsi'] = ta.momentum.rsi(df['close'], window=14)
        
        macd = ta.trend.MACD(df['close'])
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        df['macd_diff'] = macd.macd_diff()
        
        # 4. النطاقات والتقلب
        bollinger = ta.volatility.BollingerBands(df['close'])
        df['bb_high'] = bollinger.bollinger_hband()
        df['bb_low'] = bollinger.bollinger_lband()
        df['bb_pband'] = bollinger.bollinger_pband()
        
        # 5. مؤشرات الأحجام
        df['volume_change'] = df['volume'].pct_change()
        df['volume_sma_ratio'] = df['volume'] / (df['volume'].rolling(10).mean() + 1e-8)
        
        # استبدال القيم اللانهائية (inf, -inf) بـ NaN
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        return df

    def prepare_training_data(self, df):
        """إعداد الميزات وتنظيف القيم الفارغة كلياً قبل التدريب"""
        df_feat = self.extract_features(df)
        
        # التنبؤ باتجاه الحركة المستقبلية للشمعات القادمة
        future_return = (df_feat['close'].shift(-3) - df_feat['close']) / df_feat['close']
        
        conditions = [
            (future_return > 0.0015),   # BUY = 1
            (future_return < -0.0015)   # SELL = -1
        ]
        choices = [1, -1]
        df_feat['target'] = np.select(conditions, choices, default=0) # HOLD = 0
        
        # حزمة التنظيف المطلقة: مسح الصفوف التي تحتوي على NaN في الميزات أو الهدف
        clean_cols = self.feature_cols + ['target']
        df_clean = df_feat.dropna(subset=clean_cols).copy()
        
        X = df_clean[self.feature_cols]
        y = df_clean['target']
        return X, y

    def train_quick_model(self, df):
        """تدريب نماذج الذكاء الاصطناعي بحماية ضد أخطاء NaN/Inf"""
        if len(df) < 40:
            return False
            
        X, y = self.prepare_training_data(df)
        
        # التأكد من وجود بيانات كافية بعد الحذف
        if len(X) < 15:
            return False
            
        # ضمان إضافي لاستبدال أي NaN متبقي بـ 0
        X_clean = np.nan_to_num(X.values, nan=0.0, posinf=0.0, neginf=0.0)
        
        # تحجيم الميزات (Normalization)
        X_scaled = self.scaler.fit_transform(X_clean)
        
        # تدريب النماذج
        self.rf_model.fit(X_scaled, y)
        self.gb_model.fit(X_scaled, y)
        
        # حفظ أهمية الميزات
        importances = self.rf_model.feature_importances_
        self.latest_importance = dict(zip(self.feature_cols, np.round(importances, 3)))
        
        self.is_trained = True
        return True

    def predict_opportunity(self, df):
        """التنبؤ باتجاه التداول مع حماية كاملة من القيم الفارغة"""
        if not self.is_trained:
            success = self.train_quick_model(df)
            if not success:
                return "HOLD", 50, "بيانات غير كافية لتدريب النموذج"

        df_feat = self.extract_features(df)
        latest_row = df_feat[self.feature_cols].tail(1)
        
        # استبدال أي NaN أو Inf في آخر صف بـ 0
        latest_clean = np.nan_to_num(latest_row.values, nan=0.0, posinf=0.0, neginf=0.0)
        latest_scaled = self.scaler.transform(latest_clean)
        
        # التنبؤ بالاحتمالات عبر Ensemble Average
        rf_proba = self.rf_model.predict_proba(latest_scaled)[0]
        gb_proba = self.gb_model.predict_proba(latest_scaled)[0]
        
        classes = list(self.rf_model.classes_)
        avg_proba = (rf_proba + gb_proba) / 2.0
        
        best_class_idx = np.argmax(avg_proba)
        pred_class = classes[best_class_idx]
        confidence = int(avg_proba[best_class_idx] * 100)
        
        if pred_class == 1:
            action = "BUY"
        elif pred_class == -1:
            action = "SELL"
        else:
            action = "HOLD"
            
        top_feature = max(self.latest_importance, key=self.latest_importance.get) if self.latest_importance else "N/A"
        rsi_val = df_feat['rsi'].iloc[-1] if 'rsi' in df_feat and not np.isnan(df_feat['rsi'].iloc[-1]) else 50.0
        
        reason = f"تحليل Ensemble (RF+GB): الميزة الأكثر تأثيراً هي [{top_feature}]. قيمة RSI الحالية: {rsi_val:.1f}"
        return action, confidence, reason
