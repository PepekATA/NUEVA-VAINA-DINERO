import streamlit as st
import pandas as pd
import plotly.graph_objs as go
from datetime import datetime
import alpaca_trade_api as tradeapi
import time

# Configuración de página
st.set_page_config(
    page_title="Trading Bot - Real Trading",
    page_icon="🤖",
    layout="wide"
)

# Inicializar session state
if 'initialized' not in st.session_state:
    st.session_state.initialized = False
    st.session_state.last_update = datetime.now()

# Obtener configuración desde Streamlit Secrets
try:
    ALPACA_API_KEY = st.secrets["ALPACA_API_KEY"]
    ALPACA_SECRET_KEY = st.secrets["ALPACA_SECRET_KEY"]
    ALPACA_BASE_URL = st.secrets.get("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
except:
    st.error("⚠️ Por favor configura los secrets en Streamlit Cloud Settings")
    st.info("Necesitas añadir: ALPACA_API_KEY, ALPACA_SECRET_KEY")
    st.stop()

# Inicializar Alpaca API
@st.cache_resource
def init_alpaca():
    try:
        api = tradeapi.REST(
            ALPACA_API_KEY,
            ALPACA_SECRET_KEY,
            ALPACA_BASE_URL,
            api_version='v2'
        )
        # Verificar conexión
        api.get_account()
        return api
    except Exception as e:
        return None

# Conexión con Alpaca
alpaca = init_alpaca()

if not alpaca:
    st.error("❌ No se pudo conectar con Alpaca. Verifica tus credenciales.")
    st.stop()

# FUNCIONES PRINCIPALES
def get_account():
    try:
        account = alpaca.get_account()
        return {
            'cash': float(account.cash),
            'buying_power': float(account.buying_power),
            'portfolio_value': float(account.portfolio_value)
        }
    except:
        return None

def get_positions():
    try:
        positions = alpaca.list_positions()
        result = []
        for pos in positions:
            result.append({
                'Symbol': pos.symbol,
                'Cantidad': float(pos.qty),
                'Precio Entrada': float(pos.avg_entry_price),
                'Valor': float(pos.market_value),
                'P&L': float(pos.unrealized_pl),
                'P&L %': float(pos.unrealized_plpc) * 100
            })
        return pd.DataFrame(result) if result else pd.DataFrame()
    except:
        return pd.DataFrame()

def get_price(symbol):
    try:
        bars = alpaca.get_latest_bar(symbol)
        return float(bars.c)
    except:
        return 0

def place_order(symbol, qty, side):
    try:
        order = alpaca.submit_order(
            symbol=symbol,
            qty=qty,
            side=side,
            type='market',
            time_in_force='day'
        )
        return True, order.id
    except Exception as e:
        return False, str(e)

# UI PRINCIPAL
st.title("🤖 Trading Bot - Alpaca Markets")

# Verificar estado del mercado
try:
    clock = alpaca.get_clock()
    if clock.is_open:
        st.success("🟢 Mercado ABIERTO")
    else:
        st.error("🔴 Mercado CERRADO")
except:
    pass

# TABS
tab1, tab2, tab3 = st.tabs(["📊 Dashboard", "💼 Posiciones", "📈 Trading"])

# TAB 1: DASHBOARD
with tab1:
    # Información de cuenta
    account_info = get_account()
    
    if account_info:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("💵 Cash", f"${account_info['cash']:,.2f}")
        
        with col2:
            st.metric("💰 Portfolio", f"${account_info['portfolio_value']:,.2f}")
        
        with col3:
            st.metric("🔥 Buying Power", f"${account_info['buying_power']:,.2f}")
    
    st.divider()
    
    # Watchlist
    st.subheader("Watchlist")
    
    symbols = ['AAPL', 'TSLA', 'SPY', 'QQQ', 'NVDA']
    prices = {}
    
    cols = st.columns(5)
    for i, symbol in enumerate(symbols):
        with cols[i]:
            price = get_price(symbol)
            prices[symbol] = price
            st.metric(symbol, f"${price:.2f}")

# TAB 2: POSICIONES
with tab2:
    st.subheader("Posiciones Actuales")
    
    positions_df = get_positions()
    
    if not positions_df.empty:
        st.dataframe(
            positions_df,
            use_container_width=True,
            hide_index=True
        )
        
        # Total P&L
        total_pl = positions_df['P&L'].sum() if 'P&L' in positions_df else 0
        if total_pl >= 0:
            st.success(f"Total P&L: ${total_pl:.2f}")
        else:
            st.error(f"Total P&L: ${total_pl:.2f}")
    else:
        st.info("No hay posiciones abiertas")

# TAB 3: TRADING
with tab3:
    st.subheader("Ejecutar Orden")
    
    col1, col2 = st.columns(2)
    
    with col1:
        symbol_trade = st.text_input("Símbolo", value="AAPL")
        cantidad = st.number_input("Cantidad", min_value=1, value=1, step=1)
        operacion = st.radio("Operación", ["Comprar", "Vender"])
        
        # Mostrar precio actual
        if symbol_trade:
            current_price = get_price(symbol_trade)
            if current_price > 0:
                st.info(f"Precio actual: ${current_price:.2f}")
                st.info(f"Valor total: ${current_price * cantidad:.2f}")
        
        if st.button("🚀 Ejecutar", type="primary"):
            side = "buy" if operacion == "Comprar" else "sell"
            success, result = place_order(symbol_trade, cantidad, side)
            
            if success:
                st.success(f"✅ Orden ejecutada: {result}")
            else:
                st.error(f"❌ Error: {result}")
    
    with col2:
        st.subheader("Órdenes Recientes")
        
        try:
            orders = alpaca.list_orders(status='all', limit=5)
            for order in orders:
                if order.status == 'filled':
                    icon = "✅"
                elif order.status == 'canceled':
                    icon = "❌"
                else:
                    icon = "⏳"
                
                st.write(f"{icon} {order.symbol} - {order.side} {order.qty} - {order.status}")
        except:
            st.info("No hay órdenes recientes")

# Botón de actualización manual
if st.button("🔄 Actualizar Datos"):
    st.rerun()

# Footer
st.markdown("---")
st.caption("Trading Bot conectado a Alpaca Markets (Paper Trading)")
st.caption("Última actualización: " + datetime.now().strftime("%H:%M:%S"))
