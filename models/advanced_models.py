# models/advanced_models.py
import warnings
warnings.filterwarnings('ignore')

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("⚠️ xgboost غير مثبت - سيتم استخدام RandomForest فقط")

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("⚠️ lightgbm غير مثبت - سيتم استخدام RandomForest فقط")

from .base_model import LocalAITradingEngine

class AdvancedTradingModels(LocalAITradingEngine):
    def __init__(self):
        super().__init__()
        self.models = {}
        
    def create_models(self):
        """إنشاء النماذج المتاحة فقط"""
        from sklearn.ensemble import RandomForestClassifier
        
        self.models = {
            'random_forest': RandomForestClassifier(
                n_estimators=100, max_depth=6, random_state=42
            )
        }
        
        if XGBOOST_AVAILABLE:
            self.models['xgboost'] = xgb.XGBClassifier(
                n_estimators=100, max_depth=6, random_state=42
            )
        
        if LIGHTGBM_AVAILABLE:
            self.models['lightgbm'] = lgb.LGBMClassifier(
                n_estimators=100, max_depth=6, random_state=42
            )
        
        return self.models
