# ui/charts.py
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class ChartBuilder:
    """بناء الرسوم البيانية"""
    
    @staticmethod
    def create_candlestick_chart(df, symbol):
        """رسم بياني للشموع"""
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            row_heights=[0.7, 0.3],
            subplot_titles=(f'📈 {symbol}', '📊 RSI')
        )
        
        # الشموع
        fig.add_trace(
            go.Candlestick(
                x=df['date'],
                open=df['open'],
                high=df['high'],
                low=df['low'],
                close=df['close'],
                name='السعر',
                increasing_line_color='#00FF00',
                decreasing_line_color='#FF0000'
            ),
            row=1, col=1
        )
        
        # المتوسطات
        fig.add_trace(
            go.Scatter(
                x=df['date'], y=df['SMA_20'],
                mode='lines', name='SMA 20',
                line=dict(color='orange', width=1.5)
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=df['date'], y=df['SMA_50'],
                mode='lines', name='SMA 50',
                line=dict(color='cyan', width=1.5)
            ),
            row=1, col=1
        )
        
        # RSI
        fig.add_trace(
            go.Scatter(
                x=df['date'], y=df['RSI'],
                mode='lines', name='RSI',
                line=dict(color='purple', width=2)
            ),
            row=2, col=1
        )
        
        # خطوط RSI
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
        
        # منطقة RSI الطبيعية
        fig.add_hrect(
            y0=30, y1=70, line_width=0,
            fillcolor="gray", opacity=0.1, row=2, col=1
        )
        
        # تنسيق الرسم
        fig.update_layout(
            height=550,
            template='plotly_dark',
            showlegend=True,
            hovermode='x unified',
            xaxis_rangeslider_visible=False,
            margin=dict(l=20, r=20, t=40, b=20)
        )
        
        # تحديث محاور الرسم
        fig.update_yaxes(title_text="السعر ($)", row=1, col=1)
        fig.update_yaxes(title_text="RSI", row=2, col=1)
        fig.update_xaxes(title_text="الوقت", row=2, col=1)
        
        return fig
    
    @staticmethod
    def display_signal(action, confidence):
        """عرض الإشارة"""
        if action == "BUY":
            return f"🟢 **BUY** ({confidence}%)"
        elif action == "SELL":
            return f"🔴 **SELL** ({confidence}%)"
        else:
            return f"⏸️ **HOLD** ({confidence}%)"
