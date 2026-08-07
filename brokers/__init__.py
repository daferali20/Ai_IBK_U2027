# ==========================================
# brokers/__init__.py
# حزمة الوسطاء الماليين
# ==========================================

from .base_broker import BaseBroker

# محاولة استيراد IBKRBroker
try:
    from .ibkr_broker import IBKRBroker
    IBKR_AVAILABLE = True
except ImportError as e:
    IBKR_AVAILABLE = False
    IBKRBroker = None
    print(f"⚠️ IBKRBroker غير متوفر: {e}")

# استيراد MockBroker (دائماً متوفر)
try:
    from .mock_broker import MockBroker
except ImportError:
    # تعريف MockBroker هنا إذا لم يكن موجوداً
    class MockBroker(BaseBroker):
        def connect(self, *args, **kwargs):
            self._is_connected = True
            return True, "✅ متصل بالوسيط الوهمي"
        
        def disconnect(self):
            self._is_connected = False
        
        def is_connected(self):
            return self._is_connected
        
        def get_historical_data(self, *args, **kwargs):
            import pandas as pd
            import yfinance as yf
            ticker = yf.Ticker("AAPL")
            df = ticker.history(period="5d")
            if df.empty:
                return None
            df.rename(columns={
                'Open': 'open',
                'High': 'high',
                'Low': 'low',
                'Close': 'close',
                'Volume': 'volume'
            }, inplace=True)
            df['date'] = df.index
            df.index = df.index.tz_localize(None)
            return df
        
        def place_order(self, action, symbol, quantity, *args, **kwargs):
            order = {
                'id': len(self._orders) + 1,
                'action': action,
                'symbol': symbol,
                'quantity': quantity,
                'status': 'FILLED'
            }
            self._orders.append(order)
            return True, f"✅ تم تنفيذ {action} {quantity} من {symbol}", order
        
        def get_account_info(self):
            return {'balance': 100000, 'equity': 100000, 'pnl': 0}
        
        def get_positions(self):
            return []

# تصدير الكلاسات المتاحة
__all__ = ['BaseBroker', 'IBKRBroker', 'MockBroker', 'IBKR_AVAILABLE']
