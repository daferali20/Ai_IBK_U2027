# data/fetcher.py
import yfinance as yf
import pandas as pd
from data.indicators import TechnicalIndicators

class DataFetcher:
    """جلب البيانات من Yahoo Finance"""
    
    @staticmethod
    def get_stock_data(symbol, period="5d", interval="5m"):
        """جلب بيانات السهم"""
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=period, interval=interval)
            
            if df.empty:
                return None, f"لا توجد بيانات للرمز: {symbol}"
            
            # إعادة تسمية الأعمدة
            df.rename(columns={
                'Open': 'open', 'High': 'high', 'Low': 'low',
                'Close': 'close', 'Volume': 'volume'
            }, inplace=True)
            
            # معالجة المنطقة الزمنية
            df.index = df.index.tz_localize(None)
            df['date'] = df.index
            
            # إضافة المؤشرات
            df = TechnicalIndicators.add_all(df)
            df.dropna(inplace=True)
            
            return df, None
            
        except Exception as e:
            return None, str(e)
    
    @staticmethod
    def get_top_gainers(limit=10):
        """جلب الأسهم الأكثر ارتفاعاً"""
        symbols = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 
                   'NVDA', 'META', 'NFLX', 'JPM', 'VTI']
        results = []
        
        for symbol in symbols:
            try:
                ticker = yf.Ticker(symbol)
                info = ticker.info
                results.append({
                    'symbol': symbol,
                    'price': info.get('regularMarketPrice', 0),
                    'change': info.get('regularMarketChangePercent', 0)
                })
            except:
                continue
        
        results.sort(key=lambda x: x['change'], reverse=True)
        return results[:limit]
