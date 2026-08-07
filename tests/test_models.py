# tests/test_models.py
import unittest
import pandas as pd
from models.base_model import LocalAITradingEngine

class TestModels(unittest.TestCase):
    
    def setUp(self):
        """تحضير بيانات الاختبار"""
        self.df = pd.DataFrame({
            'open': [100, 101, 102, 103, 104],
            'high': [101, 102, 103, 104, 105],
            'low': [99, 100, 101, 102, 103],
            'close': [100.5, 101.5, 102.5, 103.5, 104.5],
            'volume': [1000, 1100, 1200, 1300, 1400]
        })
        self.model = LocalAITradingEngine()
    
    def test_extract_features(self):
        """اختبار استخراج الميزات"""
        result = self.model.extract_features(self.df)
        self.assertIsNotNone(result)
    
    def test_train_model(self):
        """اختبار تدريب النموذج"""
        result = self.model.train_quick_model(self.df)
        self.assertIsNotNone(result)

if __name__ == '__main__':
    unittest.main()