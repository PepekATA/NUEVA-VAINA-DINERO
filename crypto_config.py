"""
Configuración para Trading de Criptomonedas 24/7 en Streamlit
"""

import os
import streamlit as st

class CryptoTradingConfig:
    """Configuración para operar crypto 24/7"""
    
    # Obtener credenciales de Streamlit Secrets
    try:
        ALPACA_API_KEY = st.secrets["ALPACA_API_KEY"]
        ALPACA_SECRET_KEY = st.secrets["ALPACA_SECRET_KEY"]
        GITHUB_TOKEN = st.secrets["GITHUB_TOKEN"]
        GITHUB_REPO = st.secrets["GITHUB_REPO"]
    except:
        ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
        ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")
        GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
        GITHUB_REPO = os.getenv("GITHUB_REPO")
    
    # URL para crypto trading (siempre usa el endpoint de paper para crypto)
    ALPACA_BASE_URL = 'https://paper-api.alpaca.markets'
    
    # Símbolos de Crypto (24/7)
    CRYPTO_SYMBOLS = [
        'BTCUSD', 'ETHUSD', 'LTCUSD', 'BCHUSD', 'LINKUSD',
        'DOGEUSD', 'SHIBUSD', 'AVAXUSD', 'UNIUSD', 'DOTUSD',
        'MATICUSD', 'SOLUSD', 'ADAUSD', 'XLMUSD', 'ALGOUSD'
    ]
    
    # Trading Parameters para Crypto
    INITIAL_CAPITAL = 10000.0
    MAX_POSITION_SIZE = 0.05  # 5% máx por posición
    MAX_POSITIONS = 10
    MIN_CONFIDENCE = 0.70
    
    # Risk Management para Crypto (más conservador por la volatilidad)
    STOP_LOSS_PCT = 0.03      # 3%
    TAKE_PROFIT_PCT = 0.06    # 6%
    TRAILING_STOP_PCT = 0.02  # 2%
    
    # 10 Agentes por crypto
    AGENTS_PER_SYMBOL = 10
