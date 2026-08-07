# brokers/ibkr_broker.py
"""
وسيط Interactive Brokers (IBKR)
"""

import random
import time
from typing import Optional, Tuple, Dict, Any
from .base_broker import BaseBroker

class IBKRBroker(BaseBroker):
    """
    وسيط IBKR لتداول الأسهم
    يستخدم ib_insync أو ib_async
    """
    
    def __init__(self, host='127.0.0.1', port=7497, client_id=1):
        """
        تهيئة وسيط IBKR
        
        Args:
            host: عنوان الخادم
            port: المنفذ (7497 للتجريبي، 7496 للحقيقي)
            client_id: معرف العميل
        """
        self.host = host
        self.port = port
        self.client_id = client_id
        self.ib = None
        self._connected = False
        self._available = False
        self._init_ib()
    
    def _init_ib(self):
        """تهيئة مكتبة IBKR"""
        try:
            # محاولة استخدام ib_insync
            from ib_insync import IB, Stock, MarketOrder, util
            self.IB = IB
            self.Stock = Stock
            self.MarketOrder = MarketOrder
            self.util = util
            self._available = True
            self._lib = 'ib_insync'
        except ImportError:
            try:
                # محاولة استخدام ib_async
                from ib_async import IB, Stock, MarketOrder, util                self.IB = IB
                self.Stock = Stock
                self.MarketOrder = MarketOrder
                self.util = util
                self._available = True
                self._lib = 'ib_async'
            except ImportError:
                self._available = False
                self._lib = None
    
    def connect(self, host=None, port=None, client_id=None) -> Tuple[bool, str]:
        """الاتصال بـ IBKR"""
        if not self._available:
            return False, "IBKR غير مثبت. يرجى تثبيت ib_insync أو ib_async"
        
        try:
            # استخدام الإعدادات الممررة أو الافتراضية
            _host = host or self.host
            _port = port or self.port
            _client_id = client_id or self.client_id
            
            # إنشاء اتصال جديد
            self.ib = self.IB()
            self.ib.connect(_host, int(_port), clientId=_client_id, timeout=5)
            self._connected = True
            
            # اختبار الاتصال
            self.ib.managedAccounts()
            
            return True, f"✅ تم الاتصال بـ IBKR على {_host}:{_port}"
            
        except Exception as e:
            self._connected = False
            return False, f"❌ فشل الاتصال: {str(e)}"
    
    def disconnect(self) -> None:
        """قطع الاتصال بـ IBKR"""
        if self._connected and self.ib:
            try:
                self.ib.disconnect()
            except:
                pass
        self._connected = False
        self.ib = None
    
    def is_connected(self) -> bool:
        """التحقق من حالة الاتصال"""
        return self._connected and self.ib is not None
    
    def get_historical_data(self, symbol, duration="2 D", bar_size="5 mins"):
        """جلب البيانات التاريخية"""
        if not self.is_connected():
            return None
        
        try:
            contract = self.Stock(symbol, 'SMART', 'USD')
            bars = self.ib.reqHistoricalData(
                contract,
                endDateTime='',
                durationStr=duration,
                barSizeSetting=bar_size,
                whatToShow='TRADES',
                useRTH=True
            )
            return self.util.df(bars)
            
        except Exception as e:
            print(f"❌ فشل جلب البيانات: {e}")
            return None
    
    def place_order(self, action, symbol, quantity, order_type="MARKET") -> Tuple[bool, str]:
        """تنفيذ أمر تداول"""
        if not self.is_connected():
            return False, "غير متصل بـ IBKR"
        
        try:
            contract = self.Stock(symbol, 'SMART', 'USD')
            
            if order_type.upper() == "MARKET":
                order = self.MarketOrder(action.upper(), quantity)
            else:
                return False, f"نوع الأمر غير مدعوم: {order_type}"
            
            trade = self.ib.placeOrder(contract, order)
            self.ib.sleep(1)
            
            status = trade.orderStatus.status
            if status in ['Filled', 'Submitted']:
                return True, f"✅ تم إرسال أمر {action} لـ {quantity} سهم من {symbol}"
            else:
                return False, f"⚠️ الحالة: {status}"
                
        except Exception as e:
            return False, f"❌ فشل التنفيذ: {str(e)}"
    
    def get_account_info(self) -> Dict[str, Any]:
        """جلب معلومات الحساب"""
        if not self.is_connected():
            return {}
        
        try:
            accounts = self.ib.managedAccounts()
            return {
                'accounts': accounts,
                'connected': True
            }
        except:
            return {'connected': False}
    
    def get_positions(self) -> list:
        """جلب المراكز المفتوحة"""
        if not self.is_connected():
            return []
        
        try:
            accounts = self.ib.managedAccounts()
            positions = []
            for account in accounts:
                pos = self.ib.positions(account)
                for p in pos:
                    positions.append({
                        'symbol': p.contract.symbol,
                        'quantity': p.position,
                        'avg_cost': p.avgCost,
                        'market_price': p.marketPrice
                    })
            return positions
        except:
            return []
    
    def get_scanner_data(self, scan_code='TOP_PERC_GAIN', location='STK.NASDAQ'):
        """جلب بيانات المسح الضوئي"""
        if not self.is_connected():
            return []
        
        try:
            from ib_insync import ScannerSubscription
            
            scanner = ScannerSubscription()
            scanner.instrument = 'STK'
            scanner.locationCode = location
            scanner.scanCode = scan_code
            
            scan_data = self.ib.reqScannerData(scanner)
            
            results = []
            for data in scan_data[:20]:
                results.append({
                    'symbol': data.contract.symbol,
                    'rank': data.rank,
                    'value': round(data.distance, 2) if data.distance else 0
                })
            
            return results
            
        except Exception as e:
            print(f"❌ فشل المسح الضوئي: {e}")
            return []
