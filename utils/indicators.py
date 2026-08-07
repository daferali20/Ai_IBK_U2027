# utils/indicators.py
import pandas as pd
import ta

class TechnicalIndicators:
    """مؤشرات فنية متقدمة"""
    
    @staticmethod
    def add_all_indicators(df):
        """إضافة جميع المؤشرات"""
        df = df.copy()
        
        # مؤشرات الزخم
        df['RSI'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
        df['Stoch_K'] = ta.momentum.StochRSI(df['close']).stochrsi_k()
        df['Stoch_D'] = ta.momentum.StochRSI(df['close']).stochrsi_d()
        
        # مؤشرات الاتجاه
        df['SMA_20'] = ta.trend.sma_indicator(df['close'], window=20)
        df['SMA_50'] = ta.trend.sma_indicator(df['close'], window=50)
        df['EMA_12'] = ta.trend.ema_indicator(df['close'], window=12)
        df['EMA_26'] = ta.trend.ema_indicator(df['close'], window=26)
        
        # MACD
        macd = ta.trend.MACD(df['close'])
        df['MACD'] = macd.macd()
        df['MACD_signal'] = macd.macd_signal()
        df['MACD_diff'] = macd.macd_diff()
        
        # بولينجر باند
        bb = ta.volatility.BollingerBands(df['close'])
        df['BB_high'] = bb.bollinger_hband()
        df['BB_mid'] = bb.bollinger_mavg()
        df['BB_low'] = bb.bollinger_lband()
        
        # مؤشرات الحجم
        df['OBV'] = ta.volume.OnBalanceVolumeIndicator(
            df['close'], df['volume']
        ).on_balance_volume()
        
        return df
    
    @staticmethod
    def get_indicator_list():
        """قائمة المؤشرات المتاحة"""
        return [
            'RSI', 'Stoch_K', 'Stoch_D',
            'SMA_20', 'SMA_50', 'EMA_12', 'EMA_26',
            'MACD', 'MACD_signal', 'MACD_diff',
            'BB_high', 'BB_mid', 'BB_low',
            'OBV'
        ]