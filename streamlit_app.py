import streamlit as st
import pandas as pd
import plotly.graph_objs as go
from datetime import datetime
import time
import os
from model_sync import sync_manager
from predictor import Predictor
from config import Config

# Configurar página
st.set_page_config(
    page_title="Trading Bot Dashboard",
    page_icon="🤖",
    layout="wide"
)

# CSS personalizado
st.markdown("""
    <style>
    .stButton > button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

# Inicializar
@st.cache_resource
def init_system():
    """Inicializar sistema y descargar modelos"""
    # Descargar modelos desde GitHub
    sync_manager.download_models_from_github()
    
    # Inicializar predictor
    predictor = Predictor()
    
    return predictor

predictor = init_system()

# Título
st.title("🤖 Trading Bot Dashboard")
st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuración")
    
    # Selección de símbolos
    selected_symbols = st.multiselect(
        "Seleccionar Activos",
        Config.SYMBOLS,
        default=['AAPL', 'TSLA', 'NVDA', 'SPY', 'QQQ']
    )
    
    # Timeframe
    timeframe = st.selectbox(
        "Timeframe",
        ['1Min', '5Min', '15Min', '30Min', '1Hour']
    )
    
    # Modo
    mode = st.radio(
        "Modo de Operación",
        ['Solo Predicción', 'Trading Automático']
    )
    
    # Botón actualizar modelos
    if st.button("🔄 Actualizar Modelos"):
        with st.spinner("Descargando modelos..."):
            if sync_manager.download_models_from_github():
                st.success("✅ Modelos actualizados")
            else:
                st.error("❌ Error actualizando modelos")

# Layout principal
col1, col2, col3 = st.columns([1, 2, 1])

with col1:
    st.subheader("📊 Estado del Mercado")
    
    # Placeholder para estado
    market_status = st.empty()
    
    # Actualizar estado
    is_open = datetime.now().hour >= 9 and datetime.now().hour < 16
    if is_open:
        market_status.success("🟢 Mercado Abierto")
    else:
        market_status.error("🔴 Mercado Cerrado")

with col2:
    st.subheader("🎯 Predicciones Actuales")
    
    # Container para predicciones
    predictions_container = st.container()

with col3:
    st.subheader("💰 Cuenta")
    
    # Placeholder para cuenta
    account_info = st.empty()
    account_info.metric("Balance", "$100,000", "+2.5%")

# Tabla de predicciones
st.subheader("📈 Señales de Trading")

# Obtener predicciones
predictions_data = []

for symbol in selected_symbols:
    pred = predictor.predict(symbol, timeframe)
    if pred:
        predictions_data.append({
            'Símbolo': symbol,
            'Dirección': pred['direction'],
            'Probabilidad': f"{pred['probability']:.1%}",
            'Precio': f"${pred['current_price']:.2f}",
            'Cambio Esperado': f"{pred['predicted_change_pct']:.2f}%",
            'Duración': f"{pred['duration_minutes']} min"
        })

if predictions_data:
    df_predictions = pd.DataFrame(predictions_data)
    
    # Colorear filas
    def highlight_direction(row):
        if 'UP' in row['Dirección']:
            return ['background-color: #90EE90'] * len(row)
        else:
            return ['background-color: #FFB6C1'] * len(row)
    
    styled_df = df_predictions.style.apply(highlight_direction, axis=1)
    st.dataframe(styled_df, use_container_width=True)

# Gráficos
st.subheader("📊 Visualizaciones")

col1, col2 = st.columns(2)

with col1:
    # Gráfico de probabilidades
    if predictions_data:
        fig = go.Figure(data=[
            go.Bar(
                x=[p['Símbolo'] for p in predictions_data],
                y=[float(p['Probabilidad'].strip('%'))/100 for p in predictions_data],
                marker_color=['green' if 'UP' in p['Dirección'] else 'red' 
                             for p in predictions_data]
            )
        ])
        fig.update_layout(
            title="Probabilidades de Predicción",
            yaxis_title="Probabilidad",
            template="plotly_dark"
        )
        st.plotly_chart(fig, use_container_width=True)

with col2:
    # Placeholder para gráfico de precios
    st.info("Gráfico de precios en tiempo real")

# Auto-actualización
if st.button("▶️ Iniciar Auto-Trading"):
    st.warning("⚠️ El trading automático requiere credenciales de Alpaca reales")

# Footer
st.markdown("---")
st.caption("Trading Bot v1.0 - Actualización automática cada 30 segundos")

# Auto refresh
if st.checkbox("Auto-actualizar"):
    time.sleep(30)
    st.rerun()
