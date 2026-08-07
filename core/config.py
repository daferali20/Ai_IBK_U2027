# core/config.py
import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    """الإعدادات العامة للتطبيق"""
    
    # IBKR
    IB_HOST = os.getenv('IB_HOST', '127.0.0.1')
    IB_PORT = int(os.getenv('IB_PORT', 7497))
    
    # OpenAI
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', '')
    
    # التطبيق
    APP_TITLE = "🤖 AI Trading Bot"
    APP_ICON = "🤖"
    APP_LAYOUT = "wide"
    
    # البيانات
    DEFAULT_SYMBOL = "AAPL"
    DEFAULT_PERIOD = "5d"
    DEFAULT_INTERVAL = "5m"
    DEFAULT_QUANTITY = 10
    
    # النموذج
    MODEL_MIN_SAMPLES = 30
    MODEL_ESTIMATORS = 100
    MODEL_MAX_DEPTH = 6
    
    # القائمة المفضلة
    DEFAULT_WATCHLIST = [
        "AAPL", "GOOGL", "MSFT", "AMZN", "TSLA",
        "NVDA", "META", "NFLX", "JPM", "VTI"
    ]

# إنشاء كائن الإعدادات
config = Config()
