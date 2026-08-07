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
                from ib_async import IB, Stock, MarketOrder, util
