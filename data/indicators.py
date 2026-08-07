# data/indicators.py
"""
حساب المؤشرات الفنية
"""

import pandas as pd
import numpy as np
import ta

class TechnicalIndicators:
    """
    حساب المؤشرات الفنية المختلفة
    """
    
    @staticmethod
    def add_all(df: pd.DataFrame) -> pd.DataFrame:
        """
        إضافة جميع المؤشرات الفنية
        
        Args:
            df: DataFrame مع بيانات الشموع
        
        Returns:
            DataFrame مع المؤشرات المضافة
        """
        df = df.copy()
        
        # ===== مؤشرات الزخم =====
        df['RSI'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
        df['RSI'] = df['RSI'].fillna(50)
        
        # StochRSI
        stoch = ta.momentum.StochRSIIndicator(df['close'], window=14)
        df['Stoch_K'] = stoch.stochrsi_k()
        df['Stoch_D'] = stoch.stochrsi_d()
        
        # Williams %R
        df['Williams_R'] = ta.momentum.WilliamsRIndicator(
            df['high'], df['low'], df['close'], lbp=14
        ).williams_r()
        
        # ===== مؤشرات الاتجاه =====
        df['SMA_20'] = ta.trend.sma_indicator(df['close'], window=20)
        df['SMA_50'] = ta.trend.sma_indicator(df['close'], window=50)
        df['EMA_12'] = ta.trend.ema_indicator(df['close'], window=12)
        df['EMA_26'] = ta.trend.ema_indicator(df['close'], window=26)
        
        # MACD
        macd = ta.trend.MACD(df['close'])
        df['MACD'] = macd.macd()
        df['MACD_signal'] = macd.macd_signal()
        df['MACD_diff'] = macd.macd_diff()
        
        # ADX
        df['ADX'] = ta.trend.ADXIndicator(
            df['high'], df['low'], df['close'], window=14
        ).adx()
        
        # ===== مؤشرات التقلب =====
        bb = ta.volatility.BollingerBands(df['close'], window=20)
        df['BB_high'] = bb.bollinger_hband()
        df['BB_mid'] = bb.bollinger_mavg()
        df['BB_low'] = bb.bollinger_lband()
        df['BB_width'] = bb.bollinger_wband()
        
        df['ATR'] = ta.volatility.AverageTrueRange(
            df['high'], df['low'], df['close'], window=14
        ).average_true_range()
        
        # ===== مؤشرات الحجم =====
        df['volume_ma'] = ta.trend.sma_indicator(df['volume'], window=10)
        
        df['OBV'] = ta.volume.OnBalanceVolumeIndicator(
            df['close'], df['volume']
        ).on_balance_volume()
        
        df['MFI'] = ta.volume.MoneyFlowIndex(
            df['high'], df['low'], df['close'], df['volume'], window=14
        ).money_flow_index()
        
        # ===== أنماط الشموع =====
        df['Doji'] = ta.candlestick.DojiPattern(
            df['open'], df['high'], df['low'], df['close']
        ).doji()
        
        df['Hammer'] = ta.candlestick.HammerPattern(
            df['open'], df['high'], df['low'], df['close']
        ).hammer()
        
        # ===== مؤشرات إضافية =====
        # نسبة السعر إلى المتوسط
        df['price_sma_ratio'] = df['close'] / df['SMA_20']
        
        # موقع السعر في Bollinger Bands
        df['bb_position'] = (df['close'] - df['BB_low']) / (df['BB_high'] - df['BB_low'])
        
        return df
    
    @staticmethod
    def get_indicators_list() -> dict:
        """
        الحصول على قائمة المؤشرات مع وصفها
        
        Returns:
            dict: المؤشرات وأوصافها
        """
        return {
            'RSI': 'مؤشر القوة النسبية (14)',
            'Stoch_K': 'Stochastic K',
            'Stoch_D': 'Stochastic D',
            'Williams_R': 'Williams %R',
            'SMA_20': 'المتوسط المتحرك البسيط (20)',
            'SMA_50': 'المتوسط المتحرك البسيط (50)',
            'EMA_12': 'المتوسط المتحرك الأسي (12)',
            'EMA_26': 'المتوسط المتحرك الأسي (26)',
            'MACD': 'MACD',
            'MACD_signal': 'إشارة MACD',
            'MACD_diff': 'الفرق في MACD',
            'ADX': 'مؤشر الاتجاه',
            'BB_high': 'Bollinger Bands - أعلى',
            'BB_mid': 'Bollinger Bands - وسط',
            'BB_low': 'Bollinger Bands - أدنى',
            'BB_width': 'عرض Bollinger Bands',
            'ATR': 'Average True Range',
            'volume_ma': 'متوسط الحجم المتحرك',
            'OBV': 'On Balance Volume',
            'MFI': 'مؤشر تدفق المال',
            'Doji': 'نمط Doji',
            'Hammer': 'نمط Hammer'
        }
