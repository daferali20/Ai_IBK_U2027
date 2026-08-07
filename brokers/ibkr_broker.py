# ==========================================
# brokers/ibkr_broker.py
# تنفيذ IBKR Broker
# ==========================================

import random
import pandas as pd
from typing import Tuple, Optional, Dict, Any, List
import logging

from brokers.base_broker import BaseBroker

# محاولة استيراد IBKR
try:
    from ib_async import IB, Stock, MarketOrder, LimitOrder, StopOrder
    IB_AVAILABLE = True
except ImportError:
    try:
        from ib_insync import IB, Stock, MarketOrder, LimitOrder, StopOrder
        IB_AVAILABLE = True
    except ImportError:
        IB_AVAILABLE = False
        IB = None
        Stock = None
        MarketOrder = None

logger = logging.getLogger(__name__)


class IBKRBroker(BaseBroker):
    """
    تنفيذ Broker لـ Interactive Brokers (IBKR)
    """
    
    def __init__(self):
        super().__init__()
        self.ib = None
        self.is_connected = False
        self.client_id = None
        self.account_id = None
        
    def connect(self, host: str = "127.0.0.1", port: int = 7497, client_id: int = None) -> Tuple[bool, str]:
        """
        الاتصال بـ IBKR TWS/Gateway
        
        Args:
            host: عنوان الخادم
            port: المنفذ (7497 لتطبيق TWS, 4001 لـ Gateway)
            client_id: معرف العميل (اختياري)
        
        Returns:
            Tuple[bool, str]: نجاح الاتصال والرسالة
        """
        if not IB_AVAILABLE:
            return False, "❌ مكتبة IBKR غير مثبتة. قم بتثبيت: pip install ib_async"
        
        if self.is_connected:
            return True, "✅ متصل بالفعل"
        
        try:
            if client_id is None:
                client_id = random.randint(1000, 9999)
            
            self.client_id = client_id
            self.ib = IB()
            self.ib.connect(host, port, clientId=client_id, timeout=5)
            
            self.is_connected = True
            logger.info(f"✅ تم الاتصال بـ IBKR (client_id: {client_id})")
            return True, f"✅ تم الاتصال بـ IBKR بنجاح (client_id: {client_id})"
            
        except Exception as e:
            logger.error(f"❌ فشل الاتصال بـ IBKR: {e}")
            return False, f"❌ فشل الاتصال بـ IBKR: {str(e)}"
    
    def disconnect(self) -> None:
        """قطع الاتصال بـ IBKR"""
        if self.ib is not None and self.is_connected:
            try:
                self.ib.disconnect()
                self.is_connected = False
                logger.info("🔌 تم قطع الاتصال بـ IBKR")
            except Exception as e:
                logger.error(f"❌ خطأ في قطع الاتصال: {e}")
    
    def is_connected(self) -> bool:
        """التحقق من حالة الاتصال"""
        return self.is_connected and self.ib is not None
    
    def get_historical_data(
        self,
        symbol: str,
        duration: str = "2 D",
        bar_size: str = "5 mins"
    ) -> Optional[pd.DataFrame]:
        """
        جلب البيانات التاريخية من IBKR
        
        Args:
            symbol: رمز السهم
            duration: المدة (مثل: "2 D", "1 M")
            bar_size: حجم الشمعة (مثل: "5 mins", "1 hour")
        
        Returns:
            DataFrame: البيانات التاريخية
        """
        if not self.is_connected():
            logger.warning("⚠️ غير متصل بـ IBKR")
            return None
        
        try:
            contract = Stock(symbol, 'SMART', 'USD')
            self.ib.qualifyContracts(contract)
            
            bars = self.ib.reqHistoricalData(
                contract,
                endDateTime='',
                durationStr=duration,
                barSizeSetting=bar_size,
                whatToShow='TRADES',
                useRTH=True,
                formatDate=1
            )
            
            if not bars:
                return None
            
            # تحويل البيانات إلى DataFrame
            data = []
            for bar in bars:
                data.append({
                    'date': bar.date,
                    'open': bar.open,
                    'high': bar.high,
                    'low': bar.low,
                    'close': bar.close,
                    'volume': bar.volume
                })
            
            df = pd.DataFrame(data)
            df['date'] = pd.to_datetime(df['date'])
            
            return df
            
        except Exception as e:
            logger.error(f"❌ خطأ في جلب البيانات: {e}")
            return None
    
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
            Tuple[bool, str, Dict]: نجاح التنفيذ والرسالة وتفاصيل الأمر
        """
        if not self.is_connected():
            return False, "❌ غير متصل بـ IBKR", None
        
        try:
            # إنشاء العقد
            contract = Stock(symbol, 'SMART', 'USD')
            self.ib.qualifyContracts(contract)
            
            # إنشاء الأمر حسب النوع
            if order_type == "MARKET":
                order = MarketOrder(action, quantity)
            elif order_type == "LIMIT" and limit_price is not None:
                order = LimitOrder(action, quantity, limit_price)
            elif order_type == "STOP" and stop_price is not None:
                order = StopOrder(action, quantity, stop_price)
            else:
                return False, f"❌ نوع الأمر غير مدعوم: {order_type}", None
            
            # تنفيذ الأمر
            trade = self.ib.placeOrder(contract, order)
            
            # انتظار التنفيذ
            self.ib.sleep(1)
            
            order_details = {
                'id': trade.order.orderId,
                'action': action,
                'symbol': symbol,
                'quantity': quantity,
                'order_type': order_type,
                'status': trade.orderStatus.status,
                'filled': trade.orderStatus.filled,
                'remaining': trade.orderStatus.remaining,
                'avg_price': trade.orderStatus.avgFillPrice,
                'limit_price': limit_price,
                'stop_price': stop_price
            }
            
            self._orders.append(order_details)
            
            logger.info(f"✅ تم تنفيذ {action} {quantity} من {symbol}")
            return True, f"✅ تم تنفيذ {action} {quantity} من {symbol}", order_details
            
        except Exception as e:
            logger.error(f"❌ خطأ في تنفيذ الأمر: {e}")
            return False, f"❌ خطأ في تنفيذ الأمر: {str(e)}", None
    
    def get_account_info(self) -> Dict[str, Any]:
        """
        الحصول على معلومات الحساب
        
        Returns:
            Dict: معلومات الحساب
        """
        if not self.is_connected():
            return {'error': 'غير متصل بـ IBKR'}
        
        try:
            account_summary = self.ib.accountSummary()
            
            info = {
                'balance': 0.0,
                'equity': 0.0,
                'pnl': 0.0,
                'margin_used': 0.0,
                'buying_power': 0.0
            }
            
            for item in account_summary:
                if item.tag == 'TotalCashBalance':
                    info['balance'] = float(item.value)
                elif item.tag == 'NetLiquidation':
                    info['equity'] = float(item.value)
                elif item.tag == 'UnrealizedPnL':
                    info['pnl'] = float(item.value)
                elif item.tag == 'AvailableFunds':
                    info['buying_power'] = float(item.value)
            
            self._account_info = info
            return info
            
        except Exception as e:
            logger.error(f"❌ خطأ في جلب معلومات الحساب: {e}")
            return {'error': str(e)}
    
    def get_positions(self) -> List[Dict[str, Any]]:
        """
        الحصول على المراكز المفتوحة
        
        Returns:
            List[Dict]: قائمة المراكز
        """
        if not self.is_connected():
            return []
        
        try:
            positions = self.ib.positions()
            result = []
            
            for pos in positions:
                result.append({
                    'symbol': pos.contract.symbol,
                    'quantity': pos.position,
                    'avg_cost': pos.avgCost,
                    'market_price': pos.marketPrice,
                    'unrealized_pnl': pos.unrealizedPNL
                })
            
            self._positions = result
            return result
            
        except Exception as e:
            logger.error(f"❌ خطأ في جلب المراكز: {e}")
            return []
    
    def cancel_order(self, order_id: int) -> bool:
        """
        إلغاء أمر معلق
        
        Args:
            order_id: معرف الأمر
        
        Returns:
            bool: نجاح الإلغاء
        """
        if not self.is_connected():
            return False
        
        try:
            self.ib.cancelOrder(order_id)
            logger.info(f"✅ تم إلغاء الأمر {order_id}")
            return True
        except Exception as e:
            logger.error(f"❌ خطأ في إلغاء الأمر: {e}")
            return False
    
    def get_order_status(self, order_id: int) -> Dict[str, Any]:
        """
        الحصول على حالة أمر
        
        Args:
            order_id: معرف الأمر
        
        Returns:
            Dict: حالة الأمر
        """
        for order in self._orders:
            if order.get('id') == order_id:
                return order
        return {'status': 'NOT_FOUND'}
    
    def get_market_data(self, symbol: str) -> Dict[str, Any]:
        """
        الحصول على بيانات السوق لحظية
        
        Args:
            symbol: رمز السهم
        
        Returns:
            Dict: بيانات السوق
        """
        if not self.is_connected():
            return {}
        
        try:
            contract = Stock(symbol, 'SMART', 'USD')
            self.ib.qualifyContracts(contract)
            
            ticker = self.ib.reqMktData(contract, '', False, False)
            self.ib.sleep(0.5)
            
            return {
                'symbol': symbol,
                'bid': ticker.bid,
                'ask': ticker.ask,
                'last': ticker.last,
                'volume': ticker.volume,
                'high': ticker.high,
                'low': ticker.low,
                'open': ticker.open,
                'close': ticker.close,
                'timestamp': ticker.time
            }
            
        except Exception as e:
            logger.error(f"❌ خطأ في جلب بيانات السوق: {e}")
            return {}
