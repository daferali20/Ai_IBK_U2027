import random

class IBKRBroker:
    def __init__(self, host='127.0.0.1', port=7497):
        self.host = host
        self.port = port
        self.connected = False
    
    def connect(self):
        try:
            # محاولة استيراد IBKR
            from ib_insync import IB, Stock, MarketOrder
            self.ib = IB()
            self.ib.connect(self.host, int(self.port), clientId=random.randint(1000, 9999))
            self.connected = True
            return True, "✅ تم الاتصال بـ IBKR"
        except ImportError:
            return False, "⚠️ IBKR غير مثبت"
        except Exception as e:
            return False, f"❌ فشل الاتصال: {e}"
    
    def place_order(self, action, symbol, quantity):
        if not self.connected:
            return "⚠️ غير متصل بـ IBKR"
        
        try:
            from ib_insync import Stock, MarketOrder
            contract = Stock(symbol, 'SMART', 'USD')
            order = MarketOrder(action, quantity)
            trade = self.ib.placeOrder(contract, order)
            self.ib.sleep(1)
            return f"✅ تم إرسال أمر {action} لـ {quantity} سهم"
        except Exception as e:
            return f"❌ فشل: {e}"
    
    def disconnect(self):
        if self.connected:
            self.ib.disconnect()
            self.connected = False
