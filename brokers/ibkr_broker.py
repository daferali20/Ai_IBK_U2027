# brokers/ibkr_broker.py
from ib_insync import IB, Stock, util, MarketOrder
from .base_broker import BaseBroker

class IBKRBroker(BaseBroker):
    """وسيط IBKR"""
    
    def __init__(self, host='127.0.0.1', port=7497, client_id=1):
        self.host = host
        self.port = port
        self.client_id = client_id
        self.ib = None
        self.connected = False
    
    def connect(self):
        """الاتصال بـ IBKR"""
        self.ib = IB()
        try:
            self.ib.connect(self.host, int(self.port), clientId=self.client_id)
            self.connected = True
            print(f"✅ تم الاتصال بـ IBKR على {self.host}:{self.port}")
            return True
        except Exception as e:
            print(f"❌ فشل الاتصال: {e}")
            return False
    
    def disconnect(self):
        """قطع الاتصال"""
        if self.connected and self.ib:
            self.ib.disconnect()
            self.connected = False
            print("✅ تم قطع الاتصال بـ IBKR")
    
    def get_historical_data(self, symbol, duration='2 D', bar_size='5 mins'):
        """جلب البيانات التاريخية"""
        if not self.connected:
            self.connect()
        
        contract = Stock(symbol, 'SMART', 'USD')
        bars = self.ib.reqHistoricalData(
            contract,
            endDateTime='',
            durationStr=duration,
            barSizeSetting=bar_size,
            whatToShow='TRADES',
            useRTH=True
        )
        return util.df(bars)
    
    def place_order(self, action, symbol, quantity):
        """تنفيذ أمر تداول"""
        if not self.connected:
            self.connect()
        
        contract = Stock(symbol, 'SMART', 'USD')
        order = MarketOrder(action, quantity)
        trade = self.ib.placeOrder(contract, order)
        self.ib.sleep(2)
        return trade.orderStatus.status
    
    def get_account_info(self):
        """جلب معلومات الحساب"""
        if not self.connected:
            self.connect()
        return self.ib.managedAccounts()