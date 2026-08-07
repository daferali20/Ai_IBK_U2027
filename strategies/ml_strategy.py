from models.base_model import LocalAITradingEngine

class MLStrategy:
    def __init__(self):
        self.model = LocalAITradingEngine()
        self.is_ready = False
    
    def prepare(self, df):
        self.model.train_quick_model(df)
        self.is_ready = self.model.is_trained
        return self.is_ready
    
    def get_signal(self, df):
        if not self.is_ready:
            return 'HOLD', 0, "النموذج غير جاهز"
        return self.model.predict_opportunity(df)
