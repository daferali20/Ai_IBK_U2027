# utils/logger.py
"""
نظام تسجيل الأخطاء والتقارير
"""

import logging
import sys
import os
from datetime import datetime
from typing import Optional, Dict, Any

# ==========================================
# إعدادات التسجيل
# ==========================================

LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
DATE_FORMAT = '%Y-%m-%d %H:%M:%S'

# مستويات التسجيل
LEVELS = {
    'DEBUG': logging.DEBUG,
    'INFO': logging.INFO,
    'WARNING': logging.WARNING,
    'ERROR': logging.ERROR,
    'CRITICAL': logging.CRITICAL
}

# ==========================================
# مدير التسجيل
# ==========================================

class LoggerManager:
    """مدير تسجيل الأخطاء"""
    
    _instance = None
    _logger = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(LoggerManager, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if self._logger is None:
            self._setup_logger()
    
    def _setup_logger(self):
        """إعداد نظام التسجيل"""
        self._logger = logging.getLogger('TradingBot')
        self._logger.setLevel(logging.DEBUG)
        
        # منع التكرار
        if self._logger.handlers:
            return
        
        # معالج للملف
        try:
            os.makedirs('logs', exist_ok=True)
            file_handler = logging.FileHandler(
                f'logs/trading_bot_{datetime.now().strftime("%Y%m%d")}.log'
            )
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(logging.Formatter(LOG_FORMAT, DATE_FORMAT))
            self._logger.addHandler(file_handler)
        except:
            pass
        
        # معالج للطرفية
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(logging.Formatter(LOG_FORMAT, DATE_FORMAT))
        self._logger.addHandler(console_handler)
    
    def get_logger(self) -> logging.Logger:
        """الحصول على كائن التسجيل"""
        return self._logger

# ==========================================
# دوال التسجيل السهلة
# ==========================================

def setup_logger(level: str = "INFO") -> None:
    """
    إعداد نظام التسجيل
    
    Args:
        level: مستوى التسجيل
    """
    logger = LoggerManager().get_logger()
    logger.setLevel(LEVELS.get(level.upper(), logging.INFO))

def get_logger() -> logging.Logger:
    """
    الحصول على كائن التسجيل
    
    Returns:
        كائن Logger
    """
    return LoggerManager().get_logger()

def log_info(message: str, *args, **kwargs) -> None:
    """تسجيل معلومات"""
    get_logger().info(message, *args, **kwargs)

def log_error(message: str, *args, **kwargs) -> None:
    """تسجيل خطأ"""
    get_logger().error(message, *args, **kwargs)

def log_warning(message: str, *args, **kwargs) -> None:
    """تسجيل تحذير"""
    get_logger().warning(message, *args, **kwargs)

def log_debug(message: str, *args, **kwargs) -> None:
    """تسجيل تصحيح"""
    get_logger().debug(message, *args, **kwargs)

def log_exception(message: str, *args, **kwargs) -> None:
    """تسجيل استثناء"""
    get_logger().exception(message, *args, **kwargs)

# ==========================================
# تسجيل الأحداث
# ==========================================

class EventLogger:
    """تسجيل الأحداث المهمة"""
    
    @staticmethod
    def log_connection(
        broker: str,
        status: bool,
        details: Optional[str] = None
    ):
        """تسجيل حدث الاتصال"""
        status_str = "✅" if status else "❌"
        message = f"{status_str} {broker} - {'متصل' if status else 'غير متصل'}"
        if details:
            message += f" - {details}"
        
        if status:
            log_info(message)
        else:
            log_error(message)
    
    @staticmethod
    def log_trade(
        symbol: str,
        action: str,
        quantity: int,
        price: float,
        status: str
    ):
        """تسجيل حدث الصفقة"""
        message = (
            f"📊 {action} - {symbol} - "
            f"الكمية: {quantity} - "
            f"السعر: ${price:.2f} - "
            f"الحالة: {status}"
        )
        log_info(message)
    
    @staticmethod
    def log_prediction(
        symbol: str,
        action: str,
        confidence: float,
        features: Dict[str, Any]
    ):
        """تسجيل حدث التنبؤ"""
        message = (
            f"🤖 {symbol} - {action} - "
            f"الثقة: {confidence}%"
        )
        log_info(message)
        log_debug(f"الميزات: {features}")
    
    @staticmethod
    def log_error_event(
        source: str,
        error: str,
        details: Optional[Dict[str, Any]] = None
    ):
        """تسجيل حدث خطأ"""
        message = f"❌ {source} - {error}"
        if details:
            message += f" - {details}"
        log_error(message)
    
    @staticmethod
    def log_performance(
        metric: str,
        value: float,
        unit: str = "%"
    ):
        """تسجيل أداء النموذج"""
        message = f"📊 {metric}: {value:.2f}{unit}"
        log_info(message)

# ==========================================
# تصدير الدوال
# ==========================================

__all__ = [
    'setup_logger',
    'get_logger',
    'log_info',
    'log_error',
    'log_warning',
    'log_debug',
    'log_exception',
    'EventLogger'
]
