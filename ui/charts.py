# ==========================================
# ui/charts.py
# بناء الرسوم البيانية - نسخة محسّنة
# ==========================================

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Optional, Dict, Any


class ChartBuilder:
    """
    بناء الرسوم البيانية المتقدمة
    """
    
    @staticmethod
    def create_candlestick_chart(df: pd.DataFrame, symbol: str, 
                                 show_indicators: bool = True) -> go.Figure:
        """
        رسم بياني متقدم للشموع مع مؤشرات
        
        Args:
            df: DataFrame مع بيانات السوق
            symbol: رمز السهم
            show_indicators: عرض المؤشرات الفنية
        
        Returns:
            go.Figure: كائن الرسم البياني
        """
        # إنشاء رسم بياني متعدد الصفوف
        fig = make_subplots(
            rows=3 if show_indicators else 2,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            row_heights=[0.5, 0.25, 0.25] if show_indicators else [0.7, 0.3],
            subplot_titles=(
                f'📈 حركة السعر - {symbol}',
                '📊 مؤشر القوة النسبية (RSI)',
                '📉 حجم التداول'
            ) if show_indicators else (
                f'📈 حركة السعر - {symbol}',
                '📊 مؤشر القوة النسبية (RSI)'
            )
        )
        
        # ==========================================
        # 1. مخطط الشموع
        # ==========================================
        colors = ['#00FF00' if close >= open else '#FF0000' 
                 for close, open in zip(df['close'], df['open'])]
        
        fig.add_trace(
            go.Candlestick(
                x=df['date'],
                open=df['open'],
                high=df['high'],
                low=df['low'],
                close=df['close'],
                name='السعر',
                increasing_line_color='#00FF00',
                decreasing_line_color='#FF0000',
                showlegend=True
            ),
            row=1, col=1
        )
        
        # المتوسطات المتحركة
        if 'SMA_20' in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df['date'],
                    y=df['SMA_20'],
                    mode='lines',
                    name='SMA 20',
                    line=dict(color='orange', width=1.5)
                ),
                row=1, col=1
            )
        
        if 'SMA_50' in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df['date'],
                    y=df['SMA_50'],
                    mode='lines',
                    name='SMA 50',
                    line=dict(color='cyan', width=1.5)
                ),
                row=1, col=1
            )
        
        # Bollinger Bands
        if 'BB_high' in df.columns and 'BB_low' in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df['date'],
                    y=df['BB_high'],
                    mode='lines',
                    name='BB بالا',
                    line=dict(color='gray', width=1, dash='dash')
                ),
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(
                    x=df['date'],
                    y=df['BB_low'],
                    mode='lines',
                    name='BB پایین',
                    line=dict(color='gray', width=1, dash='dash')
                ),
                row=1, col=1
            )
        
        # ==========================================
        # 2. مؤشر RSI
        # ==========================================
        if 'RSI' in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df['date'],
                    y=df['RSI'],
                    mode='lines',
                    name='RSI',
                    line=dict(color='purple', width=2)
                ),
                row=2, col=1
            )
            
            # خطوط التشبع
            fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
            
            # منطقة RSI الطبيعية
            fig.add_hrect(
                y0=30, y1=70,
                line_width=0,
                fillcolor="gray",
                opacity=0.1,
                row=2, col=1
            )
        
        # ==========================================
        # 3. حجم التداول (إذا كان موجوداً)
        # ==========================================
        if show_indicators and 'volume' in df.columns:
            volume_colors = ['#00FF00' if close >= open else '#FF0000' 
                           for close, open in zip(df['close'], df['open'])]
            
            fig.add_trace(
                go.Bar(
                    x=df['date'],
                    y=df['volume'],
                    name='الحجم',
                    marker_color=volume_colors,
                    opacity=0.6
                ),
                row=3, col=1
            )
            
            # متوسط الحجم
            if 'volume_ma' in df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=df['date'],
                        y=df['volume_ma'],
                        mode='lines',
                        name='حجم متوسط',
                        line=dict(color='yellow', width=1, dash='dash')
                    ),
                    row=3, col=1
                )
        
        # ==========================================
        # تنسيق الرسم
        # ==========================================
        fig.update_layout(
            height=700 if show_indicators else 550,
            template='plotly_dark',
            showlegend=True,
            hovermode='x unified',
            xaxis_rangeslider_visible=False,
            margin=dict(l=20, r=20, t=40, b=20),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        # تحديث محاور الرسم
        fig.update_yaxes(title_text="السعر ($)", row=1, col=1)
        fig.update_yaxes(title_text="RSI", row=2, col=1, range=[0, 100])
        
        if show_indicators:
            fig.update_yaxes(title_text="الحجم", row=3, col=1)
            fig.update_xaxes(title_text="الوقت", row=3, col=1)
        else:
            fig.update_xaxes(title_text="الوقت", row=2, col=1)
        
        return fig
    
    @staticmethod
    def create_signal_indicator(action: str, confidence: int) -> Dict[str, Any]:
        """
        إنشاء مؤشر الإشارة
        
        Args:
            action: BUY, SELL, HOLD
            confidence: درجة الثقة (0-100)
        
        Returns:
            Dict: معلومات الإشارة
        """
        if action == "BUY":
            return {
                'text': f"🟢 **BUY** ({confidence}%)",
                'color': 'green',
                'icon': '🟢',
                'class': 'buy-signal'
            }
        elif action == "SELL":
            return {
                'text': f"🔴 **SELL** ({confidence}%)",
                'color': 'red',
                'icon': '🔴',
                'class': 'sell-signal'
            }
        else:
            return {
                'text': f"⏸️ **HOLD** ({confidence}%)",
                'color': 'gray',
                'icon': '⏸️',
                'class': 'hold-signal'
            }
    
    @staticmethod
    def create_technical_summary(df: pd.DataFrame) -> Dict[str, Any]:
        """
        إنشاء ملخص فني للبيانات
        
        Args:
            df: DataFrame مع بيانات السوق
        
        Returns:
            Dict: الملخص الفني
        """
        last_close = df['close'].iloc[-1]
        prev_close = df['close'].iloc[-2] if len(df) > 1 else last_close
        
        summary = {
            'current_price': last_close,
            'change': ((last_close - prev_close) / prev_close) * 100,
            'high_24h': df['high'].max(),
            'low_24h': df['low'].min(),
            'volume_24h': df['volume'].sum(),
            'avg_volume': df['volume'].mean()
        }
        
        # إضافة المؤشرات إذا كانت موجودة
        if 'RSI' in df.columns:
            summary['rsi'] = df['RSI'].iloc[-1]
            summary['rsi_status'] = 'Overbought' if df['RSI'].iloc[-1] > 70 else (
                'Oversold' if df['RSI'].iloc[-1] < 30 else 'Neutral'
            )
        
        if 'SMA_20' in df.columns:
            summary['sma_20'] = df['SMA_20'].iloc[-1]
            summary['price_vs_sma20'] = ((last_close - df['SMA_20'].iloc[-1]) / df['SMA_20'].iloc[-1]) * 100
        
        if 'MACD' in df.columns and 'MACD_signal' in df.columns:
            summary['macd'] = df['MACD'].iloc[-1]
            summary['macd_signal'] = df['MACD_signal'].iloc[-1]
            summary['macd_status'] = 'Bullish' if df['MACD'].iloc[-1] > df['MACD_signal'].iloc[-1] else 'Bearish'
        
        return summary
