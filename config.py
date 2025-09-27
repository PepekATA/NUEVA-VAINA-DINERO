import os
from dotenv import load_dotenv
import pytz
import json
from pathlib import Path

load_dotenv()

class Config:
    """Configuración del Trading Bot con 10+ agentes por activo"""
    
    # === CREDENCIALES ===
    # Detectar entorno (Streamlit Cloud, Colab o Local)
    IN_STREAMLIT = 'STREAMLIT' in os.environ or 'streamlit' in str(Path.cwd())
    IN_COLAB = 'COLAB_GPU' in os.environ
    
    # Cargar credenciales según entorno
    if IN_STREAMLIT:
        try:
            import streamlit as st
            ALPACA_API_KEY = st.secrets.get("ALPACA_API_KEY", "")
            ALPACA_SECRET_KEY = st.secrets.get("ALPACA_SECRET_KEY", "")
            GITHUB_TOKEN = st.secrets.get("GITHUB_TOKEN", "")
            GITHUB_REPO = st.secrets.get("GITHUB_REPO", "")
            USE_PAPER = st.secrets.get("USE_PAPER", "true").lower() == "true"
        except:
            ALPACA_API_KEY = os.getenv("ALPACA_API_KEY", "")
            ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY", "")
            GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "")
            GITHUB_REPO = os.getenv("GITHUB_REPO", "")
            USE_PAPER = os.getenv("USE_PAPER", "true").lower() == "true"
    else:
        ALPACA_API_KEY = os.getenv("ALPACA_API_KEY", "")
        ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY", "")
        GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "")
        GITHUB_REPO = os.getenv("GITHUB_REPO", "")
        USE_PAPER = os.getenv("USE_PAPER", "true").lower() == "true"
    
    # URL de Alpaca (Paper o Real)
    ALPACA_BASE_URL = 'https://paper-api.alpaca.markets' if USE_PAPER else 'https://api.alpaca.markets'
    
    # === CONFIGURACIÓN DE TRADING ===
    INITIAL_CAPITAL = 10000.0  # Capital inicial
    MAX_POSITION_SIZE = 0.1    # Máximo 10% por posición
    MAX_POSITIONS = 20          # Máximo de posiciones abiertas
    MIN_CONFIDENCE = 0.70       # Confianza mínima para operar (70%)
    
    # Stop Loss y Take Profit dinámicos
    STOP_LOSS_PCT = 0.02        # 2% stop loss
    TAKE_PROFIT_PCT = 0.05      # 5% take profit
    TRAILING_STOP_PCT = 0.015   # 1.5% trailing stop
    
    # === 10+ AGENTES POR ACTIVO ===
    AGENTS_PER_SYMBOL = 10
    
    # Tipos de agentes especializados
    AGENT_CONFIGS = {
        # Deep Learning Agents (1-5)
        'lstm_predictor': {
            'type': 'lstm',
            'layers': [256, 128, 64],
            'dropout': 0.3,
            'lookback': 60
        },
        'gru_analyzer': {
            'type': 'gru', 
            'layers': [256, 128, 64],
            'dropout': 0.3,
            'lookback': 60
        },
        'cnn_pattern': {
            'type': 'cnn',
            'filters': [128, 64, 32],
            'kernel_size': 3,
            'lookback': 60
        },
        'transformer_attention': {
            'type': 'transformer',
            'heads': 8,
            'dim': 256,
            'lookback': 60
        },
        'bidirectional_lstm': {
            'type': 'bilstm',
            'layers': [128, 64],
            'dropout': 0.25,
            'lookback': 60
        },
        
        # Machine Learning Agents (6-10)
        'xgboost_trend': {
            'type': 'xgboost',
            'estimators': 300,
            'max_depth': 10,
            'learning_rate': 0.1
        },
        'lightgbm_speed': {
            'type': 'lightgbm',
            'estimators': 300,
            'num_leaves': 31,
            'learning_rate': 0.1
        },
        'catboost_categorical': {
            'type': 'catboost',
            'iterations': 300,
            'depth': 8,
            'learning_rate': 0.1
        },
        'random_forest_ensemble': {
            'type': 'random_forest',
            'estimators': 500,
            'max_depth': 20,
            'min_samples_split': 5
        },
        'gradient_boost_optimizer': {
            'type': 'gradient_boost',
            'estimators': 300,
            'max_depth': 10,
            'learning_rate': 0.1
        }
    }
    
    # === SÍMBOLOS DE TRADING ===
    # Top símbolos por liquidez y volatilidad
    SYMBOLS = [
        # Tech Giants
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'NVDA', 
        
        # ETFs principales
        'SPY', 'QQQ', 'IWM', 'DIA', 'VOO', 'ARKK',
        
        # Financieras
        'JPM', 'BAC', 'V', 'MA', 'PYPL', 'SQ',
        
        # Cripto-related
        'COIN', 'MARA', 'RIOT', 'MSTR',
        
        # High volatility
        'AMD', 'ROKU', 'PLTR', 'NIO', 'LCID', 'RIVN'
    ]
    
    # === TIMEFRAMES ===
    TIMEFRAMES = {
        'scalping': '1Min',
        'day_trading': '5Min',
        'swing': '15Min',
        'position': '1Hour'
    }
    
    DEFAULT_TIMEFRAME = '5Min'
    
    # === HORARIO DE MERCADO ===
    MARKET_TIMEZONE = pytz.timezone('America/New_York')
    MARKET_OPEN = '09:30'
    MARKET_CLOSE = '16:00'
    EXTENDED_HOURS = True  # Operar en horario extendido
    
    # === PATHS ===
    if IN_STREAMLIT:
        BASE_DIR = '/tmp'
    elif IN_COLAB:
        BASE_DIR = '/content/drive/MyDrive/TradingBot'
    else:
        BASE_DIR = os.getcwd()
    
    MODELS_DIR = os.path.join(BASE_DIR, 'models')
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    LOGS_DIR = os.path.join(BASE_DIR, 'logs')
    CHECKPOINTS_DIR = os.path.join(BASE_DIR, 'checkpoints')
    
    # Crear directorios
    for dir_path in [MODELS_DIR, DATA_DIR, LOGS_DIR, CHECKPOINTS_DIR]:
        os.makedirs(dir_path, exist_ok=True)
    
    # === PARÁMETROS DE ENTRENAMIENTO ===
    EPOCHS = 50
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    VALIDATION_SPLIT = 0.2
    EARLY_STOPPING_PATIENCE = 10
    
    # === SINCRONIZACIÓN ===
    SYNC_INTERVAL_MINUTES = 30  # Sincronizar cada 30 minutos
    AUTO_SYNC_ENABLED = True
    
    @classmethod
    def get_agent_config(cls, agent_name):
        """Obtener configuración de un agente específico"""
        return cls.AGENT_CONFIGS.get(agent_name, {})
    
    @classmethod
    def save_config(cls):
        """Guardar configuración actual"""
        config_data = {
            'timestamp': str(datetime.now()),
            'agents': cls.AGENT_CONFIGS,
            'symbols': cls.SYMBOLS,
            'capital': cls.INITIAL_CAPITAL,
            'environment': 'paper' if cls.USE_PAPER else 'real'
        }
        
        config_file = os.path.join(cls.DATA_DIR, 'config_backup.json')
        with open(config_file, 'w') as f:
            json.dump(config_data, f, indent=2)
        
        return config_file
