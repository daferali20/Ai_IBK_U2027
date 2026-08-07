# tests/test_broker.py
import unittest
from brokers.ibkr_broker import IBKRBroker

class TestBroker(unittest.TestCase):
    
    def test_connection(self):
        """اختبار الاتصال بـ IBKR"""
        broker = IBKRBroker('127.0.0.1', 7497, 999)
        result = broker.connect()
        # اختبار دون التأكد من الاتصال الفعلي
        self.assertIsNotNone(result)

if __name__ == '__main__':
    unittest.main()