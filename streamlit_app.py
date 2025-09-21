import streamlit as st
import requests
import pandas as pd
import plotly.graph_objs as go
from datetime import datetime
import time
import json
import os
import alpaca_trade_api as tradeapi
import yfinance as yf
from ta import add_all_ta_features
import numpy as np

# Configuración de página
st.set_page_config(
    page_title="Trading Bot - Real Trading",
    page_icon="🤖",
    layout="wide"
)

# Obtener configuración desde Streamlit Secrets
ALPACA_API_KEY = st.secrets["ALPACA_API_KEY"]
ALPACA_SECRET_KEY = st.secrets["ALPACA_SECRET_KEY"]
ALPACA_BASE_URL = st.secrets.get("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
GITHUB_TOKEN = st.secrets["GITHUB_TOKEN"]
GITHUB_REPO = st.secrets["GITHUB_REPO"]

# Inicializar Alpaca API
@st.cache_resource
def init_alpaca():
    return tradeapi.REST(
        ALPACA_API_KEY,
        ALPACA_SECRET_KEY,
        ALPACA_BASE_URL,
        api_version='v2'
    )

alpaca = init_alpaca()

# Funciones de datos
def get_market_status():
    try:
        clock = alpaca.get_clock()
        return {
            'is_open': clock.is_open,
            'next_open': clock.next_open.isoformat() if clock.next_open else None,
            'next_close': clock.next_close.isoformat() if clock.next_close else None
        }
    except Exception as e:
        st.error(f"Error: {e}")
        return None

def get_account_info():
    try:
        account = alpaca.get_account()
        return {
            'buying_power': float(account.buying_power),
            'cash': float(account.cash),
            'portfolio_value': float(account.portfolio_value),
            'pattern_day_trader': account.pattern_day_trader,
            'trading_blocked': account.trading_blocked
        }
    except Exception as e:
        st.error(f"Error: {e}")
        return None

def get_positions():
    try:
        positions = alpaca.list_positions()
        return [
            {
                'symbol': pos.symbol,
                'qty': float(pos.qty),
                'avg_entry_price': float(pos.avg_entry_price),
                'current_price': float(pos.current_price),
                'market_value': float(pos.market_value),
                'unrealized_pl': float(pos.unrealized_pl),
                'unrealized_plpc': float(pos.unrealized_plpc),
                'side': pos.side
            }
            for pos in positions
        ]
    except Exception as e:
        st.error(f"Error: {e}")
        return []

def get_realtime_data(symbol, timeframe='1Min'):
    try:
        end = datetime.now()
        if timeframe == '1Min':
            start = end - pd.Timedelta(hours=2)
        elif timeframe == '5Min':
            start = end - pd.Timedelta(hours=8)
        else:
            start = end - pd.Timedelta(days=1)
        
        bars = alpaca.get_bars(
            symbol,
            timeframe,
            start=start.isoformat(),
            end=end.isoformat(),
            limit=100
        ).df
        
        if not bars.empty:
            bars = add_all_ta_features(
                bars, open="open", high="high", low="low", close="close", volume="volume"
            )
            
            return {
                'symbol': symbol,
                'current_price': float(bars['close'].iloc[-1]),
                'volume': float(bars['volume'].iloc[-1]),
                'rsi': float(bars['momentum_rsi'].iloc[-1]) if 'momentum_rsi' in bars else 50,
                'macd': float(bars['trend_macd'].iloc[-1]) if 'trend_macd' in bars else 0,
                'data': bars
            }
    except Exception as e:
        st.error(f"Error fetching {symbol}: {e}")
        return None

def execute_trade(symbol, side, qty, order_type='market'):
    try:
        order = alpaca.submit_order(
            symbol=symbol,
            qty=qty,
            side=side,
            type=order_type,
            time_in_force='day'
        )
        return {
            'order_id': order.id,
            'symbol': order.symbol,
            'qty': order.qty,
            'side': order.side,
            'status': order.status
        }
    except Exception as e:
        st.error(f"Error executing trade: {e}")
        return None

# UI Principal
st.title("🤖 Trading Bot - Alpaca Real Trading")

# Sidebar
with st.sidebar:
    st.header("⚙️ Control Panel")
    
    # Estado del mercado
    market = get_market_status()
    if market:
        if market['is_open']:
            st.success("🟢 Mercado ABIERTO")
        else:
            st.error("🔴 Mercado CERRADO")
            if market['next_open']:
                st.info(f"Abre: {market['next_open']}")
    
    # Cuenta
    account = get_account_info()
    if account:
        st.metric("Portfolio", f"${account['portfolio_value']:,.2f}")
        st.metric("Cash", f"${account['cash']:,.2f}")
        st.metric("Buying Power", f"${account['buying_power']:,.2f}")
    
    st.markdown("---")
    
    # Selección de símbolos
    symbols = st.multiselect(
        "Activos",
        ['AAPL', 'TSLA', 'NVDA', 'SPY', 'QQQ', 'AMZN', 'MSFT', 'META', 
         'GOOGL', 'AMD', 'INTC', 'V', 'MA', 'JPM', 'BAC', 'WMT', 'DIS'],
        default=['AAPL', 'TSLA', 'NVDA']
    )
    
    timeframe = st.selectbox(
        "Timeframe",
        ['1Min', '5Min', '15Min', '30Min', '1Hour'],
        index=1
    )
    
    auto_refresh = st.checkbox("🔄 Auto-actualizar (10s)")

# Tabs principales
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Dashboard", "🎯 Análisis", "💼 Posiciones", "📈 Trading", "📉 Gráficos"])

with tab1:
    st.header("Dashboard en Tiempo Real")
    
    # Métricas en columnas
    cols = st.columns(3)
    
    for i, symbol in enumerate(symbols[:9]):  # Máximo 9 símbolos
        with cols[i % 3]:
            data = get_realtime_data(symbol, timeframe)
            if data:
                st.metric(
                    label=f"**{symbol}**",
                    value=f"${data['current_price']:.2f}",
                    delta=f"RSI: {data['rsi']:.1f}"
                )
                st.caption(f"Vol: {data['volume']:,.0f}")
                st.caption(f"MACD: {data['macd']:.2f}")
                st.divider()

with tab2:
    st.header("🎯 Análisis Técnico")
    
    for symbol in symbols:
        data = get_realtime_data(symbol, timeframe)
        if data:
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                st.subheader(symbol)
                st.write(f"${data['current_price']:.2f}")
            
            with col2:
                rsi = data['rsi']
                if rsi < 30:
                    st.success("⬆️ SOBREVENTA")
                    signal = "BUY"
                elif rsi > 70:
                    st.error("⬇️ SOBRECOMPRA")
                    signal = "SELL"
                else:
                    st.info("➡️ NEUTRAL")
                    signal = "HOLD"
            
            with col3:
                st.metric("RSI", f"{rsi:.1f}")
            
            with col4:
                st.metric("MACD", f"{data['macd']:.2f}")
            
            with col5:
                if signal == "BUY":
                    if st.button(f"Comprar {symbol}", key=f"buy_{symbol}"):
                        order = execute_trade(symbol, "buy", 1)
                        if order:
                            st.success(f"✅ Orden ejecutada: {order['order_id']}")
                elif signal == "SELL":
                    if st.button(f"Vender {symbol}", key=f"sell_{symbol}"):
                        order = execute_trade(symbol, "sell", 1)
                        if order:
                            st.success(f"✅ Orden ejecutada: {order['order_id']}")

with tab3:
    st.header("💼 Posiciones Actuales")
    
    positions = get_positions()
    
    if positions:
        for pos in positions:
            col1, col2, col3, col4, col5, col6 = st.columns(6)
            
            with col1:
                st.write(f"**{pos['symbol']}**")
            
            with col2:
                st.write(f"Qty: {pos['qty']}")
            
            with col3:
                st.write(f"Entry: ${pos['avg_entry_price']:.2f}")
            
            with col4:
                st.write(f"Current: ${pos['current_price']:.2f}")
            
            with col5:
                pl = pos['unrealized_pl']
                pl_pct = pos['unrealized_plpc'] * 100
                if pl >= 0:
                    st.success(f"P&L: ${pl:.2f} ({pl_pct:.1f}%)")
                else:
                    st.error(f"P&L: ${pl:.2f} ({pl_pct:.1f}%)")
            
            with col6:
                if st.button(f"Cerrar", key=f"close_{pos['symbol']}"):
                    order = execute_trade(pos['symbol'], "sell", pos['qty'])
                    if order:
                        st.success("Posición cerrada")
                        st.rerun()
    else:
        st.info("No hay posiciones abiertas")

with tab4:
    st.header("📈 Trading Manual")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Nueva Orden")
        trade_symbol = st.selectbox("Símbolo", symbols, key="trade_symbol")
        trade_side = st.radio("Operación", ["buy", "sell"])
        trade_qty = st.number_input("Cantidad", min_value=0.01, value=1.0, step=0.01)
        trade_type = st.selectbox("Tipo", ["market", "limit", "stop"])
        
        if trade_type == "limit":
            limit_price = st.number_input("Precio límite", min_value=0.01)
        else:
            limit_price = None
        
        if st.button("🚀 Ejecutar Orden", type="primary"):
            if trade_type == "market":
                order = execute_trade(trade_symbol, trade_side, trade_qty)
            else:
                st.warning("Límites y stops en desarrollo")
                order = None
            
            if order:
                st.success(f"✅ Orden {order['order_id']} ejecutada")
                st.json(order)
    
    with col2:
        st.subheader("Órdenes Recientes")
        try:
            orders = alpaca.list_orders(status='all', limit=10)
            for order in orders[:5]:
                st.write(f"• {order.symbol} - {order.side} {order.qty} - {order.status}")
        except:
            st.info("No hay órdenes recientes")

with tab5:
    st.header("📉 Gráficos en Tiempo Real")
    
    selected_symbol = st.selectbox("Seleccionar símbolo para gráfico", symbols)
    
    if selected_symbol:
        data = get_realtime_data(selected_symbol, timeframe)
        if data and 'data' in data:
            df = data['data']
            
            # Crear gráfico de velas
            fig = go.Figure(data=[go.Candlestick(
                x=df.index,
                open=df['open'],
                high=df['high'],
                low=df['low'],
                close=df['close'],
                name='Precio'
            )])
            
            # Agregar indicadores
            if 'momentum_rsi' in df.columns:
                fig.add_trace(go.Scatter(
                    x=df.index,
                    y=df['close'].rolling(20).mean(),
                    name='MA20',
                    line=dict(color='yellow', width=1)
                ))
            
            fig.update_layout(
                title=f'{selected_symbol} - {timeframe}',
                yaxis_title='Precio ($)',
                xaxis_title='Tiempo',
                template='plotly_dark',
                height=600
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Volumen
            fig_vol = go.Figure(data=[go.Bar(
                x=df.index,
                y=df['volume'],
                name='Volumen'
            )])
            
            fig_vol.update_layout(
                title='Volumen',
                yaxis_title='Volumen',
                template='plotly_dark',
                height=200
            )
            
            st.plotly_chart(fig_vol, use_container_width=True)

# Auto-refresh
if auto_refresh:
    time.sleep(10)
    st.rerun()

# Footer
st.markdown("---")
st.caption("🤖 Trading Bot con datos reales de Alpaca Markets")
