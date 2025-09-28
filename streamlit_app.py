"""
Trading Bot 24/7 para Criptomonedas - Versión Estable
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import json

# Configuración de página
st.set_page_config(
    page_title="Trading Bot Crypto 24/7",
    page_icon="🤖",
    layout="wide"
)

# Título
st.title("🤖 Trading Bot Crypto 24/7")
st.caption("Sistema de Trading Automático para Criptomonedas")

# Inicialización básica
if 'bot_running' not in st.session_state:
    st.session_state.bot_running = False
    st.session_state.last_update = datetime.now()

# Verificar credenciales
def check_credentials():
    """Verificar si las credenciales están configuradas"""
    try:
        import alpaca_trade_api as tradeapi
        
        # Intentar obtener credenciales
        api_key = st.secrets.get("ALPACA_API_KEY", "")
        secret_key = st.secrets.get("ALPACA_SECRET_KEY", "")
        
        if not api_key or not secret_key:
            return None, "Credenciales no configuradas"
        
        # Intentar conexión
        api = tradeapi.REST(
            api_key,
            secret_key,
            'https://paper-api.alpaca.markets',
            api_version='v2'
        )
        
        # Verificar cuenta
        account = api.get_account()
        return api, "success"
        
    except Exception as e:
        return None, str(e)

# Verificar conexión
api, status = check_credentials()

if api is None:
    st.error("❌ No se puede conectar con Alpaca")
    st.info("""
    Por favor configura tus credenciales en Streamlit Cloud:
    1. Ve a Settings → Secrets
    2. Agrega:
