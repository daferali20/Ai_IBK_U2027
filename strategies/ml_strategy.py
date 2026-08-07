# ==========================================
# strategies/ml_strategy.py
# استراتيجية التداول بالذكاء الاصطناعي - نسخة محسّنة
# ==========================================

from models.base_model import LocalAITradingEngine, SimpleLocalAITradingEngine
from typing import Tuple, Dict, Any, Optional
import pandas as pd
import logging

logger = logging.getLogger(__name__)


class MLStrategy:
    """
    استراتيجية التداول بالذكاء الاصطناعي
    نسخة محسّنة مع دعم المحرك المتطور
    """
    
    def __init__(self, use_advanced: bool = True, model_dir: str = "models/saved/"):
        """
        تهيئة الاستراتيجية
        
        Args:
            use_advanced: استخدام المحرك المتطور أو المبسط
            model_dir: مسار حفظ النماذج
        """
        self.use_advanced = use_advanced
        
        if use_advanced:
            self.model = LocalAITradingEngine(model_dir=model_dir)
        else:
            self.model = SimpleLocalAITradingEngine()
            
        self.is_ready = False
        self.training_history = []
        self.prediction_history = []
        self.last_signal = None
        self.last_confidence = 0
        self.last_reason = ""
        
        logger.info(f"✅ تم تهيئة استراتيجية ML (المحرك المتقدم: {use_advanced})")
    
    def prepare(self, df: pd.DataFrame, symbol: str = "") -> bool:
        """
        تدريب النموذج على البيانات
        
        Args:
            df: DataFrame مع بيانات السوق
            symbol: رمز السهم (للحفظ)
        
        Returns:
            bool: نجاح التدريب
        """
        try:
            # تدريب النموذج
            success = self.model.train_quick_model(df, symbol)
            self.is_ready = self.model.is_trained
            
            if success:
                # حفظ تاريخ التدريب
                self.training_history.append({
                    'timestamp': pd.Timestamp.now(),
                    'symbol': symbol,
                    'samples': len(df),
                    'features': len(self.model.feature_cols)
                })
                
                logger.info(f"✅ تم تدريب النموذج بنجاح على {len(df)} شمعة")
                
                # عرض أهم الميزات
                importance = self.model.get_feature_importance()
                if importance:
                    top_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:5]
                    logger.info(f"📊 أهم 5 ميزات: {top_features}")
            else:
                logger.warning("⚠️ فشل تدريب النموذج")
            
            return success
            
        except Exception as e:
            logger.error(f"❌ خطأ في تدريب النموذج: {e}")
            return False
    
    def get_signal(self, df: pd.DataFrame, api_key: str = None) -> Tuple[str, int, str]:
        """
        الحصول على إشارة التداول
        
        Args:
            df: DataFrame مع بيانات السوق
            api_key: مفتاح NewsAPI (للتحليل المتقدم)
        
        Returns:
            Tuple[str, int, str]: (الإشارة, الثقة, السبب)
        """
        if not self.is_ready:
            # محاولة التدريب التلقائي
            if len(df) >= 50:
                self.prepare(df)
            else:
                return 'HOLD', 0, "⚠️ النموذج غير جاهز والبيانات غير كافية"
        
        try:
            # الحصول على التنبؤ
            action, confidence, reason = self.model.predict_opportunity(df, api_key)
            
            # حفظ التاريخ
            self.prediction_history.append({
                'timestamp': pd.Timestamp.now(),
                'action': action,
                'confidence': confidence,
                'reason': reason[:100] + "..." if len(reason) > 100 else reason
            })
            
            self.last_signal = action
            self.last_confidence = confidence
            self.last_reason = reason
            
            logger.info(f"📊 الإشارة: {action} (الثقة: {confidence}%)")
            
            return action, confidence, reason
            
        except Exception as e:
            logger.error(f"❌ خطأ في الحصول على الإشارة: {e}")
            return 'HOLD', 0, f"⚠️ خطأ: {str(e)}"
    
    def get_signal_with_risk(self, df: pd.DataFrame, portfolio_value: float = 10000) -> Dict[str, Any]:
        """
        الحصول على إشارة مع معلومات المخاطرة
        
        Args:
            df: DataFrame مع بيانات السوق
            portfolio_value: قيمة المحفظة
        
        Returns:
            Dict: معلومات الإشارة والمخاطرة
        """
        action, confidence, reason = self.get_signal(df)
        
        # حساب حجم الصفقة المقترح
        risk_params = self.model._calculate_risk_parameters(df)
        position_size = portfolio_value * risk_params['max_position_size']
        stop_loss = risk_params['stop_loss'] * position_size
        
        return {
            'action': action,
            'confidence': confidence,
            'reason': reason,
            'position_size': position_size,
            'stop_loss': stop_loss,
            'risk_level': risk_params['risk_level'],
            'risk_per_trade': risk_params['risk_per_trade'] * portfolio_value
        }
    
    def get_performance_metrics(self) -> Dict:
        """الحصول على مقاييس أداء الاستراتيجية"""
        if not self.is_ready:
            return {'error': 'النموذج غير جاهز'}
        
        metrics = self.model.get_performance_metrics()
        
        # إضافة معلومات إضافية
        metrics['is_ready'] = self.is_ready
        metrics['use_advanced'] = self.use_advanced
        metrics['training_count'] = len(self.training_history)
        metrics['prediction_count'] = len(self.prediction_history)
        
        if self.training_history:
            metrics['last_training'] = self.training_history[-1]['timestamp']
        
        return metrics
    
    def get_feature_importance(self) -> Dict:
        """الحصول على أهمية الميزات"""
        return self.model.get_feature_importance()
    
    def save_model(self, symbol: str = "") -> bool:
        """حفظ النموذج"""
        return self.model._save_model(symbol)
    
    def load_model(self, symbol: str = "") -> bool:
        """تحميل النموذج"""
        return self.model._load_model(symbol)
    
    def reset(self):
        """إعادة تعيين الاستراتيجية"""
        self.is_ready = False
        self.training_history = []
        self.prediction_history = []
        self.last_signal = None
        self.last_confidence = 0
        self.last_reason = ""
        logger.info("🔄 تم إعادة تعيين الاستراتيجية")
    
    def get_status(self) -> Dict:
        """الحصول على حالة الاستراتيجية"""
        return {
            'is_ready': self.is_ready,
            'use_advanced': self.use_advanced,
            'last_signal': self.last_signal,
            'last_confidence': self.last_confidence,
            'training_count': len(self.training_history),
            'prediction_count': len(self.prediction_history),
            'is_trained': self.model.is_trained,
            'feature_count': len(self.model.feature_cols)
        }


# ==========================================
# استراتيجية متخصصة للمضاربة (Scalping)
# ==========================================

class ScalpingStrategy(MLStrategy):
    """
    استراتيجية مضاربة سريعة باستخدام الذكاء الاصطناعي
    """
    
    def __init__(self):
        super().__init__(use_advanced=True)
        self.min_confidence = 70  # ثقة أعلى للمضاربة
        self.max_hold_time = 5    # عدد الشمعات القصوى للاحتفاظ
        
    def get_signal(self, df: pd.DataFrame) -> Tuple[str, int, str]:
        """الحصول على إشارة للمضاربة"""
        action, confidence, reason = super().get_signal(df)
        
        # تعديل الثقة للمضاربة
        if action != "HOLD" and confidence < self.min_confidence:
            return "HOLD", confidence, f"الثقة منخفضة للمضاربة ({confidence}% < {self.min_confidence}%)"
        
        return action, confidence, reason


# ==========================================
# استراتيجية استثمار طويل المدى
# ==========================================

class LongTermStrategy(MLStrategy):
    """
    استراتيجية استثمار طويل المدى
    """
    
    def __init__(self):
        super().__init__(use_advanced=True)
        self.min_confidence = 60
        self.required_features = ['SMA_50', 'SMA_200', 'ADX']
        
    def get_signal(self, df: pd.DataFrame) -> Tuple[str, int, str]:
        """الحصول على إشارة للاستثمار طويل المدى"""
        action, confidence, reason = super().get_signal(df)
        
        # التحقق من وجود المؤشرات المطلوبة
        df_feat = self.model.extract_features(df)
        missing = [f for f in self.required_features if f not in df_feat.columns]
        
        if missing:
            return "HOLD", 30, f"⚠️ المؤشرات المطلوبة غير موجودة: {missing}"
        
        # تعديل الثقة للاستثمار طويل المدى
        if action != "HOLD" and confidence < self.min_confidence:
            return "HOLD", confidence, f"الثقة منخفضة للاستثمار ({confidence}% < {self.min_confidence}%)"
        
        return action, confidence, reason
