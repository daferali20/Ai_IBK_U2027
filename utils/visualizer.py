# utils/visualizer.py
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

class TradingVisualizer:
    """رسوم بيانية متقدمة للتداول"""
    
    @staticmethod
    def plot_candlestick_with_indicators(df, symbol):
        """رسم شموع مع المؤشرات"""
        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            row_heights=[0.5, 0.25, 0.25],
            subplot_titles=(f'{symbol}', 'RSI', 'Volume')
        )
        
        # الشموع
        fig.add_trace(
            go.Candlestick(
                x=df['date'],
                open=df['open'],
                high=df['high'],
                low=df['low'],
                close=df['close'],
                name='Price'
            ),
            row=1, col=1
        )
        
        # المتوسطات
        fig.add_trace(
            go.Scatter(x=df['date'], y=df['SMA_20'], 
                      mode='lines', name='SMA 20'),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=df['date'], y=df['SMA_50'], 
                      mode='lines', name='SMA 50'),
            row=1, col=1
        )
        
        # RSI
        fig.add_trace(
            go.Scatter(x=df['date'], y=df['RSI'], 
                      mode='lines', name='RSI'),
            row=2, col=1
        )
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
        
        # Volume
        fig.add_trace(
            go.Bar(x=df['date'], y=df['volume'], name='Volume'),
            row=3, col=1
        )
        
        fig.update_layout(
            height=700,
            template='plotly_dark',
            showlegend=True
        )
        
        return fig
    
    @staticmethod
    def show_prediction_confidence(action, confidence):
        """عرض الثقة في التنبؤ"""
        if action == 'BUY':
            st.success(f"🟢 شراء (ثقة: {confidence}%)")
        elif action == 'SELL':
            st.error(f"🔴 بيع (ثقة: {confidence}%)")
        else:
            st.warning(f"⏸️ انتظار (ثقة: {confidence}%)")