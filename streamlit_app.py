import streamlit as st
import requests
import pandas as pd
import plotly.graph_objs as go
from datetime import datetime
import time

# Configuración
API_URL = st.secrets.get("API_URL", "https://your-api.railway.app")

st.set_page_config(
    page_title="Trading Bot Professional",
    page_icon="🤖",
    layout="wide"
)

# Funciones para llamar al API
@st.cache_data(ttl=60)
def get_predictions(symbols):
    predictions = []
    for symbol in symbols:
        try:
            response = requests.post(f"{API_URL}/predict/{symbol}")
            if response.status_code == 200:
                predictions.append(response.json())
        except:
            continue
    return predictions

@st.cache_data(ttl=30)
def get_account_info():
    try:
        response = requests.get(f"{API_URL}/account/info")
        if response.status_code == 200:
            return response.json()
    except:
        return None

@st.cache_data(ttl=30)
def get_positions():
    try:
        response = requests.get(f"{API_URL}/positions")
        if response.status_code == 200:
            return response.json()
    except:
        return []

# UI Principal
st.title("🤖 Trading Bot Professional Dashboard")

# Sidebar
with st.sidebar:
    st.header("Control Panel")
    
    symbols = st.multiselect(
        "Select Assets",
        ['AAPL', 'TSLA', 'NVDA', 'SPY', 'QQQ', 'AMZN', 'MSFT', 'META'],
        default=['AAPL', 'TSLA', 'NVDA']
    )
    
    if st.button("🔄 Refresh", type="primary"):
        st.cache_data.clear()
        st.rerun()
    
    if st.button("🔄 Sync Models"):
        requests.post(f"{API_URL}/models/sync")
        st.success("Sync initiated")

# Main Layout
tab1, tab2, tab3, tab4 = st.tabs(["📊 Dashboard", "📈 Predictions", "💼 Positions", "📜 History"])

with tab1:
    # Métricas principales
    col1, col2, col3, col4 = st.columns(4)
    
    account = get_account_info()
    if account:
        col1.metric("Portfolio", f"${account['portfolio_value']:,.2f}")
        col2.metric("Cash", f"${account['cash']:,.2f}")
        col3.metric("Buying Power", f"${account['buying_power']:,.2f}")
        col4.metric("Day Trading Power", f"${account.get('day_trading_buying_power', 0):,.2f}")
    
    # Gráfico de predicciones
    predictions = get_predictions(symbols)
    if predictions:
        df = pd.DataFrame(predictions)
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=df['symbol'],
            y=df['probability'],
            marker_color=['green' if d == 'UP' else 'red' for d in df['direction']],
            text=[f"{p:.1%}" for p in df['probability']],
            textposition='auto'
        ))
        fig.update_layout(
            title="Prediction Probabilities",
            yaxis_title="Probability",
            template="plotly_dark",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.header("AI Predictions")
    
    predictions = get_predictions(symbols)
    if predictions:
        for pred in predictions:
            col1, col2, col3, col4, col5 = st.columns(5)
            
            col1.write(f"**{pred['symbol']}**")
            
            if pred['direction'] == 'UP':
                col2.success(f"⬆️ {pred['direction']}")
            else:
                col2.error(f"⬇️ {pred['direction']}")
            
            col3.metric("Probability", f"{pred['probability']:.1%}")
            col4.metric("Confidence", f"{pred['confidence']:.1%}")
            col5.metric("Price", f"${pred['current_price']:.2f}")
            
            # Botones de acción
            c1, c2 = st.columns(2)
            with c1:
                if st.button(f"Buy {pred['symbol']}", key=f"buy_{pred['symbol']}"):
                    # Ejecutar compra
                    st.info(f"Buying {pred['symbol']}...")
            with c2:
                if st.button(f"Sell {pred['symbol']}", key=f"sell_{pred['symbol']}"):
                    # Ejecutar venta
                    st.info(f"Selling {pred['symbol']}...")

with tab3:
    st.header("Current Positions")
    
    positions = get_positions()
    if positions:
        df_pos = pd.DataFrame(positions)
        
        # Colorear P&L
        def color_pl(val):
            if val > 0:
                return 'color: green'
            elif val < 0:
                return 'color: red'
            return ''
        
        styled_df = df_pos.style.applymap(
            color_pl,
            subset=['unrealized_pl', 'unrealized_plpc']
        )
        
        st.dataframe(styled_df, use_container_width=True)
    else:
        st.info("No open positions")

with tab4:
    st.header("Trade History")
    st.info("Trade history will be displayed here")

# Auto-refresh
if st.checkbox("Auto-refresh (30s)"):
    time.sleep(30)
    st.rerun()
