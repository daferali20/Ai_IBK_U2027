# brokers/base_broker.py
from abc import ABC, abstractmethod

class BaseBroker(ABC):
    """واجهة موحدة للوسطاء"""
    
    @abstractmethod
    def connect(self):
        """الاتصال بالوسيط"""
        pass
    
    @abstractmethod
    def disconnect(self):
        """قطع الاتصال"""
        pass
    
    @abstractmethod
    def get_historical_data(self, symbol, duration, bar_size):
        """جلب البيانات التاريخية"""
        pass
    
    @abstractmethod
    def place_order(self, action, symbol, quantity):
        """تنفيذ أمر تداول"""
        pass
    
    @abstractmethod
    def get_account_info(self):
        """جلب معلومات الحساب"""
        pass