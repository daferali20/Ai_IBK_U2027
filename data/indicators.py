# ==========================================
# data/indicators.py
# حساب المؤشرات الفنية - نسخة محسّنة
# ==========================================

import pandas as pd
import numpy as np
import ta
from typing import Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')

class TechnicalIndicators:
    """
    حساب المؤشرات الفنية المختلفة - نسخة محسّنة
    مع إضافة مؤشرات متقدمة ومعالجة أفضل للبيانات
    """
    
    # قائمة المؤشرات الأساسية
    BASIC_INDICATORS = [
        'RSI', 'SMA_20', 'SMA_50', 'EMA_12', 'EMA_26',
        'MACD', 'MACD_signal', 'MACD_diff', 'BB_high', 'BB_mid',
        'BB_low', 'BB_width', 'ATR', 'volume_ma', 'OBV', 'MFI'
    ]
    
    # قائمة المؤشرات المتقدمة
    ADVANCED_INDICATORS = [
        'Stoch_K', 'Stoch_D', 'Williams_R', 'ADX', 'CCI', 'Aroon_up', 'Aroon_down',
        'Vortex_pos', 'Vortex_neg', 'Keltner_high', 'Keltner_low', 'Keltner_mid',
        'Donchian_high', 'Donchian_low', 'Donchian_mid', 'PSAR',
        'Trix', 'DMI_pos', 'DMI_neg', 'Mass_index'
    ]
    
    @staticmethod
    def add_all(df: pd.DataFrame, advanced: bool = True) -> pd.DataFrame:
        """
        إضافة جميع المؤشرات الفنية
        
        Args:
            df: DataFrame مع بيانات الشموع
            advanced: إضافة المؤشرات المتقدمة أو لا
        
        Returns:
            DataFrame مع المؤشرات المضافة
        """
        df = df.copy()
        
        # التأكد من وجود الأعمدة المطلوبة
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"العمود {col} غير موجود في البيانات")
        
        # ==========================================
        # 1. مؤشرات الزخم (Momentum)
        # ==========================================
        try:
            # RSI - مؤشر القوة النسبية
            df['RSI'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
            df['RSI'] = df['RSI'].fillna(50).clip(0, 100)
            
            # StochRSI
            stoch = ta.momentum.StochRSIIndicator(df['close'], window=14)
            df['Stoch_K'] = stoch.stochrsi_k().fillna(50)
            df['Stoch_D'] = stoch.stochrsi_d().fillna(50)
            
            # Williams %R
            df['Williams_R'] = ta.momentum.WilliamsRIndicator(
                df['high'], df['low'], df['close'], lbp=14
            ).williams_r().fillna(-50)
            
            # CCI - مؤشر القناة السلعية
            df['CCI'] = ta.trend.CCIIndicator(
                df['high'], df['low'], df['close'], window=20
            ).cci().fillna(0)
            
            # Ultimate Oscillator
            df['Ultimate_Osc'] = ta.momentum.UltimateOscillator(
                df['high'], df['low'], df['close']
            ).ultimate_oscillator().fillna(50)
            
        except Exception as e:
            print(f"⚠️ خطأ في مؤشرات الزخم: {e}")
        
        # ==========================================
        # 2. مؤشرات الاتجاه (Trend)
        # ==========================================
        try:
            # المتوسطات المتحركة
            df['SMA_20'] = ta.trend.sma_indicator(df['close'], window=20)
            df['SMA_50'] = ta.trend.sma_indicator(df['close'], window=50)
            df['SMA_200'] = ta.trend.sma_indicator(df['close'], window=200)
            
            df['EMA_12'] = ta.trend.ema_indicator(df['close'], window=12)
            df['EMA_26'] = ta.trend.ema_indicator(df['close'], window=26)
            df['EMA_50'] = ta.trend.ema_indicator(df['close'], window=50)
            
            # MACD
            macd = ta.trend.MACD(df['close'])
            df['MACD'] = macd.macd()
            df['MACD_signal'] = macd.macd_signal()
            df['MACD_diff'] = macd.macd_diff()
            
            # ADX - مؤشر الاتجاه المتوسط
            adx = ta.trend.ADXIndicator(
                df['high'], df['low'], df['close'], window=14
            )
            df['ADX'] = adx.adx().fillna(0)
            df['DMI_pos'] = adx.adx_pos().fillna(0)
            df['DMI_neg'] = adx.adx_neg().fillna(0)
            
            # Aroon
            aroon = ta.trend.AroonIndicator(df['high'], df['low'], window=25)
            df['Aroon_up'] = aroon.aroon_up().fillna(50)
            df['Aroon_down'] = aroon.aroon_down().fillna(50)
            
            # Trix
            df['Trix'] = ta.trend.TrixIndicator(df['close'], window=15).trix().fillna(0)
            
        except Exception as e:
            print(f"⚠️ خطأ في مؤشرات الاتجاه: {e}")
        
        # ==========================================
        # 3. مؤشرات التقلب (Volatility)
        # ==========================================
        try:
            # Bollinger Bands
            bb = ta.volatility.BollingerBands(df['close'], window=20, window_dev=2)
            df['BB_high'] = bb.bollinger_hband()
            df['BB_mid'] = bb.bollinger_mavg()
            df['BB_low'] = bb.bollinger_lband()
            df['BB_width'] = bb.bollinger_wband()
            df['BB_pband'] = bb.bollinger_pband()  # %B - موقع السعر داخل النطاق
            
            # ATR - متوسط المدى الحقيقي
            df['ATR'] = ta.volatility.AverageTrueRange(
                df['high'], df['low'], df['close'], window=14
            ).average_true_range().fillna(0)
            
            # Keltner Channels
            kc = ta.volatility.KeltnerChannel(
                df['high'], df['low'], df['close'], window=20
            )
            df['Keltner_high'] = kc.keltner_channel_hband()
            df['Keltner_low'] = kc.keltner_channel_lband()
            df['Keltner_mid'] = kc.keltner_channel_mband()
            
            # Donchian Channels
            dc = ta.volatility.DonchianChannel(
                df['high'], df['low'], df['close'], window=20
            )
            df['Donchian_high'] = dc.donchian_channel_hband()
            df['Donchian_low'] = dc.donchian_channel_lband()
            df['Donchian_mid'] = dc.donchian_channel_mband()
            
            # Mass Index
            df['Mass_index'] = ta.volatility.MassIndex(
                df['high'], df['low'], window=9, n2=25
            ).mass_index().fillna(0)
            
            # PSAR - بارابوليك SAR
            df['PSAR'] = ta.trend.PSARIndicator(
                df['high'], df['low'], df['close'], step=0.02, max_step=0.2
            ).psar().fillna(0)
            
        except Exception as e:
            print(f"⚠️ خطأ في مؤشرات التقلب: {e}")
        
        # ==========================================
        # 4. مؤشرات الحجم (Volume)
        # ==========================================
        try:
            # متوسط الحجم المتحرك
            df['volume_ma'] = ta.trend.sma_indicator(df['volume'], window=10)
            df['volume_ma_20'] = ta.trend.sma_indicator(df['volume'], window=20)
            
            # نسبة الحجم إلى المتوسط
            df['volume_ratio'] = df['volume'] / (df['volume_ma_20'] + 1e-8)
            
            # OBV - الرصيد عند الحجم
            df['OBV'] = ta.volume.OnBalanceVolumeIndicator(
                df['close'], df['volume']
            ).on_balance_volume()
            
            # MFI - مؤشر تدفق المال
            df['MFI'] = ta.volume.MoneyFlowIndex(
                df['high'], df['low'], df['close'], df['volume'], window=14
            ).money_flow_index().fillna(50).clip(0, 100)
            
            # VWAP - متوسط السعر المرجح بالحجم
            df['VWAP'] = (df['volume'] * (df['high'] + df['low'] + df['close']) / 3).cumsum() / df['volume'].cumsum()
            
        except Exception as e:
            print(f"⚠️ خطأ في مؤشرات الحجم: {e}")
        
        # ==========================================
        # 5. أنماط الشموع اليابانية
        # ==========================================
        try:
            df['Doji'] = ta.candlestick.DojiPattern(
                df['open'], df['high'], df['low'], df['close']
            ).doji().fillna(0)
            
            df['Hammer'] = ta.candlestick.HammerPattern(
                df['open'], df['high'], df['low'], df['close']
            ).hammer().fillna(0)
            
            df['Bullish_Engulfing'] = ta.candlestick.BullishEngulfingPattern(
                df['open'], df['high'], df['low'], df['close']
            ).bullish_engulfing_pattern().fillna(0)
            
            df['Bearish_Engulfing'] = ta.candlestick.BearishEngulfingPattern(
                df['open'], df['high'], df['low'], df['close']
            ).bearish_engulfing_pattern().fillna(0)
            
            df['Morning_Star'] = ta.candlestick.MorningStarPattern(
                df['open'], df['high'], df['low'], df['close']
            ).morning_star_pattern().fillna(0)
            
            df['Evening_Star'] = ta.candlestick.EveningStarPattern(
                df['open'], df['high'], df['low'], df['close']
            ).evening_star_pattern().fillna(0)
            
        except Exception as e:
            print(f"⚠️ خطأ في أنماط الشموع: {e}")
        
        # ==========================================
        # 6. مؤشرات Vortex (للكشف عن الاتجاه)
        # ==========================================
        try:
            vortex = ta.trend.VortexIndicator(
                df['high'], df['low'], df['close'], window=14
            )
            df['Vortex_pos'] = vortex.vortex_indicator_pos().fillna(0)
            df['Vortex_neg'] = vortex.vortex_indicator_neg().fillna(0)
            
        except Exception as e:
            print(f"⚠️ خطأ في مؤشرات Vortex: {e}")
        
        # ==========================================
        # 7. ميزات محسوبة إضافية (للتعلم الآلي)
        # ==========================================
        try:
            # العوائد والتذبذب
            df['returns'] = df['close'].pct_change()
            df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
            df['volatility_5'] = df['returns'].rolling(5).std()
            df['volatility_20'] = df['returns'].rolling(20).std()
            
            # نسب السعر
            df['price_sma_ratio'] = df['close'] / (df['SMA_20'] + 1e-8)
            df['price_sma50_ratio'] = df['close'] / (df['SMA_50'] + 1e-8)
            df['price_ema_ratio'] = df['close'] / (df['EMA_12'] + 1e-8)
            
            # موقع السعر في النطاقات
            df['bb_position'] = (df['close'] - df['BB_low']) / (df['BB_high'] - df['BB_low'] + 1e-8)
            df['bb_position'] = df['bb_position'].clip(0, 1)
            
            # مؤشرات مجمعة
            df['rsi_ma_diff'] = df['RSI'] - df['RSI'].rolling(5).mean()
            df['macd_hist'] = df['MACD'] - df['MACD_signal']
            
        except Exception as e:
            print(f"⚠️ خطأ في الميزات المحسوبة: {e}")
        
        # ==========================================
        # 8. تنظيف البيانات
        # ==========================================
        # استبدال القيم اللانهائية بـ NaN
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        
        # إزالة المؤشرات المتقدمة إذا لم تكن مطلوبة
        if not advanced:
            advanced_cols = [
                'Stoch_K', 'Stoch_D', 'Williams_R', 'CCI', 'Ultimate_Osc',
                'Aroon_up', 'Aroon_down', 'Vortex_pos', 'Vortex_neg',
                'Keltner_high', 'Keltner_low', 'Keltner_mid',
                'Donchian_high', 'Donchian_low', 'Donchian_mid',
                'PSAR', 'Trix', 'DMI_pos', 'DMI_neg', 'Mass_index',
                'Bullish_Engulfing', 'Bearish_Engulfing',
                'Morning_Star', 'Evening_Star'
            ]
            for col in advanced_cols:
                if col in df.columns:
                    df.drop(columns=[col], inplace=True)
        
        return df
    
    @staticmethod
    def get_indicators_list() -> Dict[str, str]:
        """
        الحصول على قائمة المؤشرات مع وصفها
        
        Returns:
            dict: المؤشرات وأوصافها
        """
        return {
            # مؤشرات الزخم
            'RSI': 'مؤشر القوة النسبية (14) - يقيس قوة الاتجاه',
            'Stoch_K': 'Stochastic K - مؤشر الزخم',
            'Stoch_D': 'Stochastic D - متوسط Stoch K',
            'Williams_R': 'Williams %R - مؤشر ذروة التشبع',
            'CCI': 'مؤشر القناة السلعية (20)',
            'Ultimate_Osc': 'المذبذب النهائي - يجمع 3 أطر زمنية',
            
            # مؤشرات الاتجاه
            'SMA_20': 'المتوسط المتحرك البسيط (20)',
            'SMA_50': 'المتوسط المتحرك البسيط (50)',
            'SMA_200': 'المتوسط المتحرك البسيط (200) - الاتجاه طويل المدى',
            'EMA_12': 'المتوسط المتحرك الأسي (12) - سريع',
            'EMA_26': 'المتوسط المتحرك الأسي (26) - بطيء',
            'MACD': 'MACD - تقاطع المتوسطات',
            'MACD_signal': 'إشارة MACD - متوسط MACD',
            'MACD_diff': 'الفرق في MACD - الهيستوجرام',
            'ADX': 'مؤشر الاتجاه (14) - يقوة قوة الاتجاه',
            'DMI_pos': 'DMI+ - مؤشر الاتجاه الإيجابي',
            'DMI_neg': 'DMI- - مؤشر الاتجاه السلبي',
            'Aroon_up': 'Aroon Up - قوة الاتجاه الصاعد',
            'Aroon_down': 'Aroon Down - قوة الاتجاه الهابط',
            
            # مؤشرات التقلب
            'BB_high': 'Bollinger Bands - أعلى',
            'BB_mid': 'Bollinger Bands - وسط (SMA 20)',
            'BB_low': 'Bollinger Bands - أدنى',
            'BB_width': 'عرض Bollinger Bands - التقلب',
            'BB_pband': '%B - موقع السعر في النطاق',
            'ATR': 'Average True Range - متوسط التقلب',
            'Keltner_high': 'Keltner Channel - أعلى',
            'Keltner_low': 'Keltner Channel - أدنى',
            'Keltner_mid': 'Keltner Channel - وسط',
            'Donchian_high': 'Donchian Channel - أعلى (20)',
            'Donchian_low': 'Donchian Channel - أدنى (20)',
            
            # مؤشرات الحجم
            'volume_ma': 'متوسط الحجم المتحرك (10)',
            'volume_ratio': 'نسبة الحجم إلى متوسط 20',
            'OBV': 'On Balance Volume - تدفق الحجم',
            'MFI': 'مؤشر تدفق المال (14) - RSI مع الحجم',
            'VWAP': 'متوسط السعر المرجح بالحجم',
            
            # أنماط الشموع
            'Doji': 'نمط Doji - تردد',
            'Hammer': 'نمط Hammer - انعكاس صاعد',
            'Bullish_Engulfing': 'نمط الابتلاع الصاعد',
            'Bearish_Engulfing': 'نمط الابتلاع الهابط',
            
            # ميزات محسوبة
            'returns': 'العوائد اليومية',
            'volatility_5': 'التقلب (5 أيام)',
            'volatility_20': 'التقلب (20 يوم)',
            'bb_position': 'موقع السعر في Bollinger Bands',
            'price_sma_ratio': 'نسبة السعر إلى SMA 20'
        }
    
    @staticmethod
    def get_feature_importance() -> Dict[str, float]:
        """
        الحصول على أهمية المؤشرات في النماذج (تقدير مبدئي)
        
        Returns:
            dict: المؤشرات وأهميتها التقريبية
        """
        return {
            # الأكثر أهمية في التداول
            'RSI': 0.15,
            'MACD_diff': 0.12,
            'BB_position': 0.10,
            'ADX': 0.09,
            'volume_ratio': 0.08,
            'price_sma_ratio': 0.07,
            'ATR': 0.06,
            'Stoch_K': 0.05,
            'MFI': 0.05,
            'OBV': 0.04,
            # متوسط الأهمية
            'SMA_20': 0.03,
            'SMA_50': 0.03,
            'CCI': 0.03,
            'Vortex_pos': 0.02,
            'Vortex_neg': 0.02,
            # أقل أهمية
            'Doji': 0.01,
            'Hammer': 0.01,
            'Ultimate_Osc': 0.01
        }

class AdvancedIndicators(TechnicalIndicators):
    """
    كلاس محسّن للتوافق مع المحرك المتطور
    يجمع كل المؤشرات في مكان واحد
    """
    
    @staticmethod
    def extract_all_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        استخراج جميع الميزات للتعلم الآلي
        (متوافق مع AdvancedAITradingEngine)
        
        Args:
            df: DataFrame مع بيانات السوق
        
        Returns:
            DataFrame مع جميع الميزات
        """
        # استخدام الدالة الأساسية مع المؤشرات المتقدمة
        return TechnicalIndicators.add_all(df, advanced=True)

# ==========================================
# وظائف مساعدة سريعة
# ==========================================

def get_momentum_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """الحصول على مؤشرات الزخم فقط"""
    return TechnicalIndicators.add_all(df, advanced=False)

def get_trend_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """الحصول على مؤشرات الاتجاه فقط"""
    df = df.copy()
    trend_cols = ['SMA_20', 'SMA_50', 'MACD', 'MACD_signal', 'MACD_diff', 'ADX']
    return df

def get_volatility_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """الحصول على مؤشرات التقلب فقط"""
    df = df.copy()
    volatility_cols = ['BB_high', 'BB_mid', 'BB_low', 'BB_width', 'ATR']
    return df

def get_volume_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """الحصول على مؤشرات الحجم فقط"""
    df = df.copy()
    volume_cols = ['volume_ma', 'OBV', 'MFI']
    return df
