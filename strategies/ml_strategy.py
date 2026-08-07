# strategies/ml_strategy.py
from models.base_model import LocalAITradingEngine

class MLStrategy:
    """استراتيجية تعلم آلي للتداول"""
    
    def __init__(self):
        self.model = LocalAITradingEngine()
        self.is_ready = False
    
    def prepare(self, df):
        """تحضير الاستراتيجية"""
        self.model.train_quick_model(df)
        self.is_ready = self.model.is_trained
        return self.is_ready
    
    def get_signal(self, df):
        """الحصول على إشارة التداول"""
        if not self.is_ready:
            return 'HOLD', 0, "النموذج غير جاهز"
        
        action, confidence, reason = self.model.predict_opportunity(df)
        return action, confidence, reason
    
    def backtest(self, df, initial_capital=10000):
        """اختبار الاستراتيجية على البيانات التاريخية"""
        # تنفيذ اختبار تاريخي
        pass