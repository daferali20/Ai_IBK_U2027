# utils/helpers.py
"""
دوال مساعدة للاستخدام العام
"""

import pandas as pd
import numpy as np
from datetime import datetime
import time
from functools import wraps
from typing import Any, Optional, List, Dict, Callable, Union

# ==========================================
# دوال التنسيق
# ==========================================

def format_price(price: float, currency: str = "$") -> str:
    """
    تنسيق السعر
    
    Args:
        price: السعر
        currency: رمز العملة
    
    Returns:
        السعر بتنسيق جميل
    """
    if price is None or pd.isna(price):
        return "---"
    return f"{currency}{price:,.2f}"

def format_volume(volume: int) -> str:
    """
    تنسيق حجم التداول
    
    Args:
        volume: حجم التداول
    
    Returns:
        الحجم بتنسيق مختصر (K, M, B)
    """
    if volume is None or pd.isna(volume):
        return "---"
    
    volume = int(volume)
    if volume >= 1_000_000_000:
        return f"{volume/1_000_000_000:.1f}B"
    elif volume >= 1_000_000:
        return f"{volume/1_000_000:.1f}M"
    elif volume >= 1_000:
        return f"{volume/1_000:.1f}K"
    else:
        return f"{volume:,}"

def format_percentage(value: float, decimals: int = 2) -> str:
    """
    تنسيق النسبة المئوية
    
    Args:
        value: القيمة
        decimals: عدد الأرقام العشرية
    
    Returns:
        النسبة بتنسيق جميل
    """
    if value is None or pd.isna(value):
        return "---"
    sign = "+" if value > 0 else ""
    return f"{sign}{value:.{decimals}f}%"

def format_date(date: Union[datetime, pd.Timestamp, str], fmt: str = "%Y-%m-%d %H:%M") -> str:
    """
    تنسيق التاريخ
    
    Args:
        date: التاريخ
        fmt: صيغة التنسيق
    
    Returns:
        التاريخ بتنسيق نصي
    """
    if date is None:
        return "---"
    
    if isinstance(date, str):
        try:
            date = pd.to_datetime(date)
        except:
            return date
    
    if isinstance(date, pd.Timestamp):
        date = date.to_pydatetime()
    
    return date.strftime(fmt)

def truncate_string(text: str, max_length: int = 50, suffix: str = "...") -> str:
    """
    اختصار النص
    
    Args:
        text: النص الأصلي
        max_length: الطول الأقصى
        suffix: النهاية المضافة
    
    Returns:
        النص المختصر
    """
    if not text or len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)] + suffix

# ==========================================
# دوال التحقق والتحويل
# ==========================================

def validate_symbol(symbol: str) -> bool:
    """
    التحقق من صحة رمز السهم
    
    Args:
        symbol: رمز السهم
    
    Returns:
        True إذا كان صحيحاً
    """
    if not symbol or not isinstance(symbol, str):
        return False
    
    # إزالة المسافات
    symbol = symbol.strip().upper()
    
    # التحقق من الطول
    if len(symbol) < 1 or len(symbol) > 10:
        return False
    
    # التحقق من الأحرف
    import re
    pattern = r'^[A-Z0-9.-]+$'
    return bool(re.match(pattern, symbol))

def safe_divide(a: float, b: float, default: float = 0.0) -> float:
    """
    قسمة آمنة مع معالجة القسمة على صفر
    
    Args:
        a: البسط
        b: المقام
        default: القيمة الافتراضية في حالة الخطأ
    
    Returns:
        نتيجة القسمة
    """
    try:
        if b == 0 or np.isnan(b):
            return default
        return a / b
    except:
        return default

def calculate_change(current: float, previous: float) -> float:
    """
    حساب التغير المئوي
    
    Args:
        current: القيمة الحالية
        previous: القيمة السابقة
    
    Returns:
        التغير المئوي
    """
    if previous is None or previous == 0:
        return 0.0
    return ((current - previous) / previous) * 100

def get_color_for_change(value: float) -> str:
    """
    الحصول على لون التغير
    
    Args:
        value: قيمة التغير
    
    Returns:
        اسم اللون (green, red, gray)
    """
    if value > 0:
        return "green"
    elif value < 0:
        return "red"
    else:
        return "gray"

# ==========================================
# دوال البيانات
# ==========================================

def is_valid_dataframe(df: pd.DataFrame) -> bool:
    """
    التحقق من صحة DataFrame
    
    Args:
        df: البيانات
    
    Returns:
        True إذا كانت صالحة
    """
    if df is None:
        return False
    if not isinstance(df, pd.DataFrame):
        return False
    if df.empty:
        return False
    return True

def merge_dicts(dict1: Dict, dict2: Dict, deep: bool = False) -> Dict:
    """
    دمج قاموسين
    
    Args:
        dict1: القاموس الأول
        dict2: القاموس الثاني
        deep: دمج عميق (للقواميس المتداخلة)
    
    Returns:
        القاموس المدمج
    """
    if not dict1:
        return dict2.copy()
    if not dict2:
        return dict1.copy()
    
    result = dict1.copy()
    
    if deep:
        for key, value in dict2.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = merge_dicts(result[key], value, deep=True)
            else:
                result[key] = value
    else:
        result.update(dict2)
    
    return result

def safe_get(data: Union[Dict, List, pd.Series], *keys, default: Any = None) -> Any:
    """
    الحصول على قيمة بأمان من بنية متداخلة
    
    Args:
        data: البيانات
        *keys: المفاتيح بالتسلسل
        default: القيمة الافتراضية
    
    Returns:
        القيمة المطلوبة
    """
    result = data
    for key in keys:
        try:
            if isinstance(result, dict):
                result = result.get(key)
            elif isinstance(result, list):
                result = result[int(key)] if isinstance(key, int) else result
            elif isinstance(result, pd.Series):
                result = result.get(key)
            else:
                return default
        except:
            return default
    return result if result is not None else default

def chunks(lst: List, n: int) -> List[List]:
    """
    تقسيم القائمة إلى أجزاء
    
    Args:
        lst: القائمة
        n: حجم الجزء
    
    Returns:
        قائمة من الأجزاء
    """
    return [lst[i:i + n] for i in range(0, len(lst), n)]

# ==========================================
# دوال المعالجة
# ==========================================

def retry_on_error(
    func: Callable,
    max_retries: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: tuple = (Exception,)
):
    """
    إعادة محاولة الدالة عند حدوث خطأ
    
    Args:
        func: الدالة
        max_retries: عدد المحاولات القصوى
        delay: التأخير بين المحاولات
        backoff: مضاعفة التأخير
        exceptions: الاستثناءات المراد التقاطها
    """
    def wrapper(*args, **kwargs):
        _delay = delay
        for attempt in range(max_retries):
            try:
                return func(*args, **kwargs)
            except exceptions as e:
                if attempt == max_retries - 1:
                    raise e
                time.sleep(_delay)
                _delay *= backoff
        return None
    return wrapper

def timer(func: Callable):
    """
    مؤقت لقياس وقت تنفيذ الدالة
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"⏱️ {func.__name__} استغرق {end - start:.2f} ثانية")
        return result
    return wrapper

# ==========================================
# دوال إضافية
# ==========================================

def get_trading_hours() -> Dict:
    """
    الحصول على ساعات التداول
    """
    return {
        'market_open': '09:30',
        'market_close': '16:00',
        'timezone': 'EST'
    }

def is_market_open() -> bool:
    """
    التحقق من أن السوق مفتوح
    """
    now = datetime.now()
    # بسيط - يمكن تحسينه
    return 9 <= now.hour <= 16

def generate_summary(df: pd.DataFrame) -> Dict:
    """
    إنشاء ملخص للبيانات
    """
    if df.empty:
        return {}
    
    return {
        'rows': len(df),
        'columns': len(df.columns),
        'start_date': df.index[0] if hasattr(df, 'index') else None,
        'end_date': df.index[-1] if hasattr(df, 'index') else None,
        'null_count': df.isnull().sum().sum()
    }
