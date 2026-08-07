# data/indicators.py
import ta
import pandas as pd

class TechnicalIndicators:
    """حساب المؤشرات الفنية"""
    
    @staticmethod
    def add_all(df):
        """إضافة جميع المؤشرات"""
        df = df.copy()
        
        # مؤشرات الزخم
        df['RSI'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
        df['RSI'] = df['RSI'].fillna(50)
        
        # مؤشرات المتوسطات
        df['SMA_20'] = ta.trend.sma_indicator(df['close'], window=20)
        df['SMA_50'] = ta.trend.sma_indicator(df['close'], window=50)
        
        # مؤشرات الحجم
        df['volume_ma'] = ta.trend.sma_indicator(df['volume'], window=10)
        
        # MACD
        macd = ta.trend.MACD(df['close'])
        df['MACD'] = macd.macd()
        df['MACD_signal'] = macd.macd_signal()
        
        # بولينجر باند
        bb = ta.volatility.BollingerBands(df['close'])
        df['BB_high'] = bb.bollinger_hband()
        df['BB_low'] = bb.bollinger_lband()
        
        return df
    
    @staticmethod
    def get_indicators_list():
        """قائمة المؤشرات المتاحة"""
        return ['RSI', 'SMA_20', 'SMA_50', 'MACD', 'MACD_signal', 'BB_high', 'BB_low']
