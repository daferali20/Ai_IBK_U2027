# ==========================================
# brokers/base_broker.py
# الواجهة الأساسية للوسطاء الماليين - محسّن
# ==========================================

from abc import ABC, abstractmethod
from typing import Optional, Tuple, Dict, Any, List
import pandas as pd
import logging

logger = logging.getLogger(__name__)


class BaseBroker(ABC):
    """
    الواجهة الأساسية للوسطاء الماليين
    جميع الوسطاء يجب أن يطبقوا هذه الدوال
    """
    
    def __init__(self):
        self._is_connected = False
        self._account_info = {}
        self._positions = []
        self._orders = []
        self._connection_attempts = 0
        self._max_retries = 3
        
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
    ) -> Optional[pd.DataFrame]:
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
        order_type: str = "MARKET",
        limit_price: Optional[float] = None,
        stop_price: Optional[float] = None
    ) -> Tuple[bool, str, Optional[Dict]]:
        """
        تنفيذ أمر تداول
        
        Args:
            action: BUY أو SELL
            symbol: رمز السهم
            quantity: الكمية
            order_type: MARKET, LIMIT, STOP, STOP_LIMIT
            limit_price: سعر الحد
            stop_price: سعر الوقف
        
        Returns:
            (bool, str, Dict): نجاح التنفيذ والرسالة وتفاصيل الأمر
        """
        pass
    
    @abstractmethod
    def get_account_info(self) -> Dict[str, Any]:
        """الحصول على معلومات الحساب"""
        pass
    
    @abstractmethod
    def get_positions(self) -> List[Dict[str, Any]]:
        """الحصول على المراكز المفتوحة"""
        pass
    
    # ==========================================
    # دوال إضافية مفيدة
    # ==========================================
    
    def get_balance(self) -> float:
        """الحصول على الرصيد المتاح"""
        return self._account_info.get('balance', 0.0)
    
    def get_pnl(self) -> float:
        """الحصول على الربح/الخسارة الإجمالي"""
        return self._account_info.get('pnl', 0.0)
    
    def get_open_positions_count(self) -> int:
        """عدد المراكز المفتوحة"""
        return len(self._positions)
    
    def cancel_all_orders(self) -> bool:
        """إلغاء جميع الأوامر المعلقة"""
        return True
    
    def get_order_status(self, order_id: str) -> Dict[str, Any]:
        """الحصول على حالة أمر معين"""
        for order in self._orders:
            if order.get('id') == order_id:
                return order
        return {'status': 'NOT_FOUND'}
    
    def get_trading_hours(self, symbol: str) -> Dict[str, str]:
        """الحصول على ساعات التداول"""
        return {
            'open': '09:30',
            'close': '16:00',
            'timezone': 'EST'
        }
    
    def get_market_status(self) -> str:
        """الحصول على حالة السوق"""
        return 'OPEN'
    
    def get_commission(self, action: str, quantity: int) -> float:
        """حساب العمولة"""
        # عمولة افتراضية
        return max(0.01, quantity * 0.001)
    
    def calculate_slippage(self, price: float, volume: int) -> float:
        """حساب الانزلاق السعري"""
        # انزلاق افتراضي 0.1%
        return price * 0.001


# ==========================================
# واجهة بروكر افتراضية للتجربة
# ==========================================

class MockBroker(BaseBroker):
    """
    وسيط وهمي للتجربة والاختبار
    """
    
    def connect(self, host: str, port: int, client_id: int) -> Tuple[bool, str]:
        self._is_connected = True
        self._account_info = {
            'balance': 100000.0,
            'equity': 100000.0,
            'pnl': 0.0,
            'margin_used': 0.0
        }
        logger.info(f"✅ تم الاتصال بالوسيط الوهمي (client_id: {client_id})")
        return True, "✅ تم الاتصال بالوسيط الوهمي"
    
    def disconnect(self) -> None:
        self._is_connected = False
        logger.info("🔌 تم قطع الاتصال بالوسيط الوهمي")
    
    def is_connected(self) -> bool:
        return self._is_connected
    
    def get_historical_data(self, symbol: str, duration: str = "2 D", bar_size: str = "5 mins") -> Optional[pd.DataFrame]:
        # استخدام بيانات وهمية
        import yfinance as yf
        ticker = yf.Ticker(symbol)
        df = ticker.history(period="5d", interval="5m")
        if df.empty:
            return None
        df.rename(columns={
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Volume': 'volume'
        }, inplace=True)
        df.index = df.index.tz_localize(None)
        return df
    
    def place_order(self, action: str, symbol: str, quantity: int, 
                   order_type: str = "MARKET", limit_price: Optional[float] = None,
                   stop_price: Optional[float] = None) -> Tuple[bool, str, Optional[Dict]]:
        order = {
            'id': f"mock_{len(self._orders)}",
            'action': action,
            'symbol': symbol,
            'quantity': quantity,
            'order_type': order_type,
            'status': 'FILLED',
            'limit_price': limit_price,
            'stop_price': stop_price,
            'timestamp': pd.Timestamp.now().isoformat()
        }
        self._orders.append(order)
        
        # تحديث الرصيد
        price = 100.0  # سعر افتراضي
        if action == 'BUY':
            self._account_info['balance'] -= price * quantity
        else:
            self._account_info['balance'] += price * quantity
            
        logger.info(f"✅ تم تنفيذ الأمر: {action} {quantity} من {symbol}")
        return True, f"✅ تم تنفيذ {action} {quantity} من {symbol}", order
    
    def get_account_info(self) -> Dict[str, Any]:
        return self._account_info
    
    def get_positions(self) -> List[Dict[str, Any]]:
        # محاكاة مراكز مفتوحة
        return []
