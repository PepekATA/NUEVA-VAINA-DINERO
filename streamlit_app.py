import streamlit as st
import requests
import pandas as pd
import plotly.graph_objs as go
from datetime import datetime
import time
import json
import os

# Configuración de página
st.set_page_config(
    page_title="Trading Bot - Real Trading",
    page_icon="🤖",
    layout="wide"
)

# Obtener API URL desde secrets o variable de entorno
API_URL = st.secrets.get("API_URL", os.getenv("API_URL", "https://your-api-url.herokuapp.com"))

# Funciones para API
def api_get(endpoint):
    try:
        response = requests.get(f"{API_URL}{endpoint}", timeout=5)
        if response.status_code == 200:
            return response.json()
    except:
        pass
    return None

def api_post(endpoint, data):
    try:
        response = requests.post(f"{API_URL}{endpoint}", json=data, timeout=5)
        if response.status_code == 200:
            return response.json()
    except:
        pass
    return None

# UI Principal
st.title("🤖 Trading Bot - Alpaca Real Trading")

# Verificar conexión
api_status = api_get("/")
if api_status:
    st.success(f"✅ API Conectada - Modo: {api_status.get('mode', 'UNKNOWN')}")
else:
    st.warning("⚠️ API no disponible - Modo Demo")

# Sidebar
with st.sidebar:
    st.header("⚙️ Control Panel")
    
    # Estado del mercado
    market = api_get("/market/status")
    if market:
        if market['is_open']:
            st.success("🟢 Mercado ABIERTO")
        else:
            st.error("🔴 Mercado CERRADO")
            if market['next_open']:
                st.info(f"Abre: {market['next_open']}")
    
    # Cuenta
    account = api_get("/account")
    if account:
        st.metric("Portfolio", f"${account['portfolio_value']:,.2f}")
        st.metric("Cash", f"${account['cash']:,.2f}")
        st.metric("Buying Power", f"${account['buying_power']:,.2f}")
    
    st.markdown("---")
    
    # Selección de símbolos
    symbols = st.multiselect(
        "Activos",
        ['AAPL', 'TSLA', 'NVDA', 'SPY', 'QQQ', 'AMZN', 'MSFT', 'META'],
        default=['AAPL', 'TSLA', 'NVDA']
    )
    
    timeframe = st.selectbox(
        "Timeframe",
        ['1Min', '5Min', '15Min', '30Min', '1Hour'],
        index=1
    )

# Tabs principales
tab1, tab2, tab3, tab4 = st.tabs(["📊 Dashboard", "🎯 Predicciones", "💼 Posiciones", "📈 Trading"])

with tab1:
    st.header("Dashboard en Tiempo Real")
    
    # Obtener datos reales
    col1, col2, col3 = st.columns(3)
    
    for i, symbol in enumerate(symbols):
        with [col1, col2, col3][i % 3]:
            # Datos reales del símbolo
            data = api_get(f"/data/{symbol}?timeframe={timeframe}")
            if data:
                st.subheader(symbol)
                st.metric("Precio", f"${data['current_price']:.2f}")
                st.metric("Volumen", f"{data['volume']:,.0f}")
                
                # Indicadores
                indicators = data.get('indicators', {})
                st.write(f"RSI: {indicators.get('rsi', 0):.1f}")
                st.write(f"MACD: {indicators.get('macd', 0):.2f}")

with tab2:
    st.header("🎯 Predicciones con IA")
    
    predictions_container = st.container()
    
    with predictions_container:
        for symbol in symbols:
            # Obtener predicción REAL
            pred = api_post(f"/predict/{symbol}", {"timeframe": timeframe})
            
            if pred:
                col1, col2, col3, col4, col5 = st.columns(5)
                
                with col1:
                    st.subheader(symbol)
                
                with col2:
                    if pred['direction'] == 'UP':
                        st.success(f"⬆️ COMPRA")
                    else:
                        st.error(f"⬇️ VENTA")
                
                with col3:
                    st.metric("Probabilidad", f"{pred['probability']:.1%}")
                
                with col4:
                    st.metric("Precio", f"${pred['current_price']:.2f}")
                
                with col5:
                    # Botón para ejecutar trade REAL
                    if pred['direction'] == 'UP':
                        if st.button(f"Comprar {symbol}", key=f"buy_{symbol}"):
                            # Ejecutar orden REAL
                            order = api_post("/trade/execute", {
                                "symbol": symbol,
                                "side": "buy",
                                "qty": 1,  # Ajustar cantidad
                                "order_type": "market"
                            })
                            if order:
                                st.success(f"✅ Orden ejecutada: {order['order_id']}")
                            else:
                                st.error("Error ejecutando orden")
                    else:
                        if st.button(f"Vender {symbol}", key=f"sell_{symbol}"):
                            order = api_post("/trade/execute", {
                                "symbol": symbol,
                                "side": "sell",
                                "qty": 1,
                                "order_type": "market"
                            })
                            if order:
                                st.success(f"✅ Orden ejecutada: {order['order_id']}")

with tab3:
    st.header("💼 Posiciones Reales")
    
    positions = api_get("/positions")
    
    if positions:
        df_pos = pd.DataFrame(positions)
        
        # Mostrar posiciones
        for pos in positions:
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                st.write(f"**{pos['symbol']}**")
            
            with col2:
                st.write(f"Qty: {pos['qty']}")
            
            with col3:
                st.write(f"Entry: ${pos['avg_entry_price']:.2f}")
            
            with col4:
                pl = pos['unrealized_pl']
                if pl >= 0:
                    st.success(f"P&L: ${pl:.2f}")
                else:
                    st.error(f"P&L: ${pl:.2f}")
            
            with col5:
                if st.button(f"Cerrar {pos['symbol']}", key=f"close_{pos['symbol']}"):
                    # Cerrar posición
                    order = api_post("/trade/execute", {
                        "symbol": pos['symbol'],
                        "side": "sell" if pos['side'] == 'long' else "buy",
                        "qty": pos['qty'],
                        "order_type": "market"
                    })
                    if order:
                        st.success("Posición cerrada")
                        st.rerun()
    else:
        st.info("No hay posiciones abiertas")

with tab4:
    st.header("📈 Trading Manual")
    
    col1, col2 = st.columns(2)
    
    with col1:
        trade_symbol = st.selectbox("Símbolo", symbols)
        trade_side = st.radio("Operación", ["buy", "sell"])
        trade_qty = st.number_input("Cantidad", min_value=1, value=1)
        trade_type = st.selectbox("Tipo", ["market", "limit"])
        
        if trade_type == "limit":
            limit_price = st.number_input("Precio límite", min_value=0.01)
        else:
            limit_price = None
        
        if st.button("Ejecutar Orden", type="primary"):
            order = api_post("/trade/execute", {
                "symbol": trade_symbol,
                "side": trade_side,
                "qty": trade_qty,
                "order_type": trade_type,
                "limit_price": limit_price
            })
            
            if order:
                st.success(f"✅ Orden {order['order_id']} ejecutada")
                st.json(order)
    
    with col2:
        st.subheader("Órdenes Abiertas")
        orders = api_get("/orders?status=open")
        if orders:
            for order in orders:
                st.write(f"{order['symbol']} - {order['side']} {order['qty']} - {order['status']}")
        else:
            st.info("No hay órdenes abiertas")

# Auto-refresh
if st.checkbox("🔄 Auto-actualizar (10s)"):
    time.sleep(10)
    st.rerun()
