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
try:
    ALPACA_API_KEY = st.secrets["ALPACA_API_KEY"]
    ALPACA_SECRET_KEY = st.secrets["ALPACA_SECRET_KEY"]
    ALPACA_BASE_URL = st.secrets.get("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
    GITHUB_TOKEN = st.secrets.get("GITHUB_TOKEN", "")
    GITHUB_REPO = st.secrets.get("GITHUB_REPO", "")
except:
    st.error("Por favor configura los secrets en Streamlit Cloud")
    st.stop()

# Inicializar Alpaca API
@st.cache_resource
def init_alpaca():
    try:
        return tradeapi.REST(
            ALPACA_API_KEY,
            ALPACA_SECRET_KEY,
            ALPACA_BASE_URL,
            api_version='v2'
        )
    except Exception as e:
        st.error(f"Error conectando con Alpaca: {e}")
        return None

alpaca = init_alpaca()

if not alpaca:
    st.error("No se pudo conectar con Alpaca. Verifica tus credenciales.")
    st.stop()

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
        return {'is_open': False, 'next_open': None, 'next_close': None}

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
        st.error(f"Error obteniendo cuenta: {e}")
        return None

def get_positions():
    try:
        positions = alpaca.list_positions()
        return [
            {
                'symbol': pos.symbol,
                'qty': float(pos.qty),
                'avg_entry_price': float(pos.avg_entry_price),
                'current_price': float(pos.current_price) if hasattr(pos, 'current_price') else 0,
                'market_value': float(pos.market_value),
                'unrealized_pl': float(pos.unrealized_pl),
                'unrealized_plpc': float(pos.unrealized_plpc),
                'side': pos.side
            }
            for pos in positions
        ]
    except Exception as e:
        st.error(f"Error obteniendo posiciones: {e}")
        return []

def get_realtime_data(symbol, timeframe='5Min'):
    try:
        end = pd.Timestamp.now(tz='America/New_York')
        
        # Mapeo de timeframes
        delta_map = {
            '1Min': pd.Timedelta(hours=2),
            '5Min': pd.Timedelta(hours=8),
            '15Min': pd.Timedelta(days=1),
            '30Min': pd.Timedelta(days=2),
            '1Hour': pd.Timedelta(days=5)
        }
        
        start = end - delta_map.get(timeframe, pd.Timedelta(days=1))
        
        # Obtener datos de Alpaca
        bars = alpaca.get_bars(
            symbol,
            timeframe,
            start=start.isoformat(),
            end=end.isoformat(),
            limit=100
        ).df
        
        if bars.empty:
            # Fallback a Yahoo Finance
            ticker = yf.Ticker(symbol)
            bars = ticker.history(period="1d", interval="5m")
            
        if not bars.empty:
            # Renombrar columnas para compatibilidad
            bars.columns = bars.columns.str.lower()
            
            # Agregar indicadores técnicos
            try:
                bars = add_all_ta_features(
                    bars, open="open", high="high", low="low", close="close", volume="volume",
                    fillna=True
                )
            except:
                pass
            
            return {
                'symbol': symbol,
                'current_price': float(bars['close'].iloc[-1]),
                'volume': float(bars['volume'].iloc[-1]),
                'rsi': float(bars['momentum_rsi'].iloc[-1]) if 'momentum_rsi' in bars else 50,
                'macd': float(bars['trend_macd'].iloc[-1]) if 'trend_macd' in bars else 0,
                'data': bars
            }
    except Exception as e:
        st.error(f"Error obteniendo datos de {symbol}: {e}")
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
        st.error(f"Error ejecutando trade: {e}")
        return None

# UI Principal
st.title("🤖 Trading Bot - Alpaca Real Trading")
st.caption("Conectado a: " + ("Paper Trading" if "paper" in ALPACA_BASE_URL else "Live Trading"))

# Sidebar
with st.sidebar:
    st.header("⚙️ Control Panel")
    
    # Estado del mercado
    market = get_market_status()
    if market['is_open']:
        st.success("🟢 Mercado ABIERTO")
    else:
        st.error("🔴 Mercado CERRADO")
        if market['next_open']:
            next_open = pd.Timestamp(market['next_open'])
            st.info(f"Abre: {next_open.strftime('%H:%M')}")
    
    # Cuenta
    account = get_account_info()
    if account:
        st.metric("Portfolio", f"${account['portfolio_value']:,.2f}")
        st.metric("Cash", f"${account['cash']:,.2f}")
        st.metric("Buying Power", f"${account['buying_power']:,.2f}")
    
    st.markdown("---")
    
    # Selección de símbolos
    default_symbols = ['AAPL', 'TSLA', 'NVDA', 'SPY', 'QQQ']
    all_symbols = ['AAPL', 'TSLA', 'NVDA', 'SPY', 'QQQ', 'AMZN', 'MSFT', 'META', 
                   'GOOGL', 'AMD', 'INTC', 'V', 'MA', 'JPM', 'BAC', 'WMT', 'DIS']
    
    symbols = st.multiselect(
        "Activos a monitorear",
        options=all_symbols,
        default=default_symbols[:3]
    )
    
    timeframe = st.selectbox(
        "Timeframe",
        ['1Min', '5Min', '15Min', '30Min', '1Hour'],
        index=1
    )
    
    auto_refresh = st.checkbox("🔄 Auto-actualizar (30s)")

# Tabs principales
tab1, tab2, tab3, tab4 = st.tabs(["📊 Dashboard", "💼 Posiciones", "📈 Trading", "📉 Gráficos"])

with tab1:
    st.header("Dashboard en Tiempo Real")
    
    if not symbols:
        st.warning("Selecciona al menos un símbolo en el sidebar")
    else:
        # Crear columnas para mostrar datos
        cols = st.columns(min(len(symbols), 3))
        
        for i, symbol in enumerate(symbols[:9]):
            with cols[i % 3]:
                with st.container():
                    data = get_realtime_data(symbol, timeframe)
                    if data:
                        st.metric(
                            label=f"**{symbol}**",
                            value=f"${data['current_price']:.2f}",
                            delta=f"RSI: {data['rsi']:.1f}"
                        )
                        
                        # Señal basada en RSI
                        rsi = data['rsi']
                        if rsi < 30:
                            st.success("🟢 Señal de COMPRA (Sobreventa)")
                        elif rsi > 70:
                            st.error("🔴 Señal de VENTA (Sobrecompra)")
                        else:
                            st.info("⚪ Neutral")
                        
                        st.caption(f"Volumen: {data['volume']:,.0f}")
                        st.divider()

with tab2:
    st.header("💼 Posiciones Actuales")
    
    positions = get_positions()
    
    if positions:
        # Crear DataFrame para mejor visualización
        df_positions = pd.DataFrame(positions)
        
        # Mostrar métricas generales
        total_pl = sum([p['unrealized_pl'] for p in positions])
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Posiciones", len(positions))
        with col2:
            st.metric("P&L Total", f"${total_pl:,.2f}")
        with col3:
            total_value = sum([p['market_value'] for p in positions])
            st.metric("Valor Total", f"${total_value:,.2f}")
        
        st.divider()
        
        # Mostrar cada posición
        for pos in positions:
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                st.write(f"**{pos['symbol']}**")
                st.caption(f"Qty: {pos['qty']}")
            
            with col2:
                st.write("Entry")
                st.caption(f"${pos['avg_entry_price']:.2f}")
            
            with col3:
                st.write("Current")
                st.caption(f"${pos['current_price']:.2f}")
            
            with col4:
                pl = pos['unrealized_pl']
                pl_pct = pos['unrealized_plpc'] * 100
                st.write("P&L")
                if pl >= 0:
                    st.success(f"${pl:.2f} ({pl_pct:.1f}%)")
                else:
                    st.error(f"${pl:.2f} ({pl_pct:.1f}%)")
            
            with col5:
                if st.button(f"Cerrar", key=f"close_{pos['symbol']}"):
                    with st.spinner("Cerrando posición..."):
                        order = execute_trade(pos['symbol'], "sell", pos['qty'])
                        if order:
                            st.success(f"✅ Posición cerrada")
                            time.sleep(2)
                            st.rerun()
    else:
        st.info("No hay posiciones abiertas")

with tab3:
    st.header("📈 Trading Manual")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Nueva Orden")
        
        trade_symbol = st.selectbox(
            "Símbolo",
            options=all_symbols,
            key="trade_symbol"
        )
        
        trade_side = st.radio("Operación", ["buy", "sell"])
        
        trade_qty = st.number_input(
            "Cantidad",
            min_value=0.01,
            value=1.0,
            step=0.01,
            format="%.2f"
        )
        
        # Obtener precio actual
        current_data = get_realtime_data(trade_symbol, "1Min")
        if current_data:
            st.info(f"Precio actual: ${current_data['current_price']:.2f}")
            
            # Calcular valor de la operación
            order_value = current_data['current_price'] * trade_qty
            st.info(f"Valor de la orden: ${order_value:.2f}")
        
        if st.button("🚀 Ejecutar Orden", type="primary", use_container_width=True):
            with st.spinner("Ejecutando orden..."):
                order = execute_trade(trade_symbol, trade_side, trade_qty)
                if order:
                    st.success(f"✅ Orden ejecutada")
                    st.json(order)
                    time.sleep(2)
                    st.rerun()
    
    with col2:
        st.subheader("Órdenes Recientes")
        
        try:
            orders = alpaca.list_orders(status='all', limit=10)
            
            if orders:
                for order in orders[:5]:
                    order_time = order.submitted_at.strftime('%H:%M:%S')
                    
                    if order.status == 'filled':
                        st.success(f"✅ {order_time} - {order.symbol} - {order.side} {order.qty} @ ${order.filled_avg_price or 'Market'}")
                    elif order.status == 'canceled':
                        st.error(f"❌ {order_time} - {order.symbol} - Cancelada")
                    else:
                        st.warning(f"⏳ {order_time} - {order.symbol} - {order.status}")
            else:
                st.info("No hay órdenes recientes")
                
        except Exception as e:
            st.error(f"Error obteniendo órdenes: {e}")

with tab4:
    st.header("📉 Gráficos en Tiempo Real")
    
    if symbols:
        selected_symbol = st.selectbox(
            "Seleccionar símbolo para gráfico",
            symbols
        )
        
        if selected_symbol:
            data = get_realtime_data(selected_symbol, timeframe)
            
            if data and 'data' in data:
                df = data['data']
                
                # Gráfico de velas
                fig = go.Figure(data=[go.Candlestick(
                    x=df.index,
                    open=df['open'],
                    high=df['high'],
                    low=df['low'],
                    close=df['close'],
                    name='Precio'
                )])
                
                # Agregar media móvil si está disponible
                if len(df) > 20:
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
                    height=500
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Mostrar indicadores
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("RSI", f"{data['rsi']:.1f}")
                with col2:
                    st.metric("MACD", f"{data['macd']:.2f}")
                with col3:
                    st.metric("Volumen", f"{data['volume']:,.0f}")
    else:
        st.warning("Selecciona símbolos en el sidebar")

# Auto-refresh
if auto_refresh:
    time.sleep(30)
    st.rerun()

# Footer
st.markdown("---")
st.caption("💡 Trading Bot con datos reales de Alpaca Markets")
st.caption("⚠️ Este es un bot de trading educacional. Opera bajo tu propio riesgo.")
