# brokers/base_broker.py
"""
الواجهة الأساسية للوسطاء الماليين
"""

from abc import ABC, abstractmethod
from typing import Optional, Tuple, Dict, Any

class BaseBroker(ABC):
    """
    الواجهة الأساسية للوسطاء الماليين
    جميع الوسطاء يجب أن يطبقوا هذه الدوال
    """
    
    @abstractmethod
    def connect(self, host: str, port: int, client_id: int) -> Tuple[bool, str]:
        """
        الاتصال بالوسيط
        
        Args:
            host: عنوان الخادم
            port: المنفذ
            client_id: معرف العميل
        
        Returns:
            (bool, str): نجاح الاتصال والرسالة
        """
        pass
    
    @abstractmethod
    def disconnect(self) -> None:
        """قطع الاتصال بالوسيط"""
        pass
    
    @abstractmethod
    def is_connected(self) -> bool:
        """التحقق من حالة الاتصال"""
        pass
    
    @abstractmethod
    def get_historical_data(
        self,
        symbol: str,
        duration: str = "2 D",
        bar_size: str = "5 mins"
    ):
        """
        جلب البيانات التاريخية
        
        Args:
            symbol: رمز السهم
            duration: المدة
            bar_size: حجم الشمعة
        
        Returns:
            DataFrame: البيانات التاريخية
        """
        pass
    
    @abstractmethod
    def place_order(
        self,
        action: str,
        symbol: str,
        quantity: int,
        order_type: str = "MARKET"
    ) -> Tuple[bool, str]:
        """
        تنفيذ أمر تداول
        
        Args:
            action: BUY أو SELL
            symbol: رمز السهم
            quantity: الكمية
            order_type: نوع الأمر
        
        Returns:
            (bool, str): نجاح التنفيذ والرسالة
        """
        pass
    
    @abstractmethod
    def get_account_info(self) -> Dict[str, Any]:
        """الحصول على معلومات الحساب"""
        pass
    
    @abstractmethod
    def get_positions(self) -> list:
        """الحصول على المراكز المفتوحة"""
        pass
