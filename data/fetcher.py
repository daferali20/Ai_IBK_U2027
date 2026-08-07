# data/fetcher.py
"""
جلب البيانات من مصادر متعددة
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
from typing import Optional, Tuple, List, Dict
from utils.helpers import retry_on_error, log_info, log_error

class DataFetcher:
    """
    جلب بيانات السوق من Yahoo Finance
    """
    
    @staticmethod
    @retry_on_error(max_retries=3, delay=1.0)
    def get_stock_data(
        symbol: str,
        period: str = "5d",
        interval: str = "5m"
    ) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
        """
        جلب بيانات السهم من Yahoo Finance
        
        Args:
            symbol: رمز السهم
            period: الفترة الزمنية (1d, 5d, 1mo, 3mo, 6mo, 1y)
            interval: الفاصل الزمني (1m, 2m, 5m, 15m, 30m, 60m, 1d)
        
        Returns:
            (DataFrame, error): البيانات والخطأ إن وجد
        """
        try:
            log_info(f"جلب بيانات {symbol} - الفترة: {period}, الفاصل: {interval}")
            
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=period, interval=interval)
            
            if df.empty:
                return None, f"لا توجد بيانات للرمز: {symbol}"
            
            # إعادة تسمية الأعمدة
            df.rename(columns={
                'Open': 'open',
                'High': 'high',
                'Low': 'low',
                'Close': 'close',
                'Volume': 'volume'
            }, inplace=True)
            
            # معالجة المنطقة الزمنية
            if df.index.tz is not None:
                df.index = df.index.tz_localize(None)
            
            df['date'] = df.index
            
            # حساب المؤشرات الأساسية
            from .indicators import TechnicalIndicators
            df = TechnicalIndicators.add_all(df)
            
            # إزالة القيم المفقودة
            df.dropna(inplace=True)
            
            log_info(f"✅ تم جلب {len(df)} شمعة لـ {symbol}")
            return df, None
            
        except Exception as e:
            error_msg = f"خطأ في جلب {symbol}: {str(e)}"
            log_error(error_msg)
            return None, error_msg
    
    @staticmethod
    def get_multiple_stocks(
        symbols: List[str],
        period: str = "5d",
        interval: str = "5m"
    ) -> Dict[str, pd.DataFrame]:
        """
        جلب بيانات عدة أسهم
        
        Args:
            symbols: قائمة رموز الأسهم
            period: الفترة الزمنية
            interval: الفاصل الزمني
        
        Returns:
            dict: قاموس بالبيانات
        """
        results = {}
        for symbol in symbols:
            df, error = DataFetcher.get_stock_data(symbol, period, interval)
            if df is not None:
                results[symbol] = df
            time.sleep(0.5)  # تجنب الحظر
        return results
    
    @staticmethod
    def get_live_price(symbol: str) -> Tuple[Optional[float], Optional[str]]:
        """
        الحصول على السعر الحالي
        
        Args:
            symbol: رمز السهم
        
        Returns:
            (price, error): السعر والخطأ إن وجد
        """
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period="1d", interval="1m")
            
            if data.empty:
                return None, f"لا توجد بيانات لـ {symbol}"
            
            return data['Close'].iloc[-1], None
            
        except Exception as e:
            return None, str(e)
    
    @staticmethod
    def get_stock_info(symbol: str) -> Optional[Dict]:
        """
        الحصول على معلومات السهم
        
        Args:
            symbol: رمز السهم
        
        Returns:
            dict: معلومات السهم
        """
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            return {
                'symbol': symbol,
                'name': info.get('longName', symbol),
                'sector': info.get('sector', 'N/A'),
                'industry': info.get('industry', 'N/A'),
                'market_cap': info.get('marketCap', 0),
                'pe_ratio': info.get('trailingPE', 0),
                'dividend_yield': info.get('dividendYield', 0) * 100 if info.get('dividendYield') else 0
            }
            
        except Exception as e:
            log_error(f"خطأ في جلب معلومات {symbol}: {e}")
            return None
    
    @staticmethod
    def get_top_gainers(limit: int = 10) -> List[Dict]:
        """
        جلب الأسهم الأكثر ارتفاعاً
        
        Args:
            limit: عدد النتائج
        
        Returns:
            list: قائمة الأسهم الرائجة
        """
        symbols = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 
                   'NVDA', 'META', 'NFLX', 'JPM', 'VTI',
                   'SPY', 'QQQ', 'AMD', 'INTC', 'PYPL',
                   'KO', 'PEP', 'WMT', 'JNJ', 'PG']
        
        results = []
        for symbol in symbols:
            try:
                info = DataFetcher.get_stock_info(symbol)
                price, _ = DataFetcher.get_live_price(symbol)
                
                if price:
                    results.append({
                        'symbol': symbol,
                        'price': price,
                        'change': 0,  # يمكن حسابها من البيانات
                        'name': info.get('name', symbol) if info else symbol
                    })
            except:
                continue
        
        return results[:limit]
