"""
Configuración para Trading de Criptomonedas 24/7
Sistema Multi-Agente con 10 agentes por par
"""

import os
from datetime import datetime
import pytz

class CryptoConfig:
    """Configuración optimizada para crypto trading 24/7"""
    
    # ===== CREDENCIALES (Auto-cargadas) =====
    ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
    ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")
    GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
    GITHUB_REPO = os.getenv("GITHUB_REPO")
    
    # ===== CONFIGURACIÓN ALPACA CRYPTO =====
    # IMPORTANTE: Alpaca usa el mismo endpoint para crypto
    ALPACA_BASE_URL = 'https://paper-api.alpaca.markets'
    CRYPTO_DATA_URL = 'https://data.alpaca.markets'
    
    # ===== PARES DE CRYPTO 24/7 EN ALPACA =====
    # Lista completa de criptomonedas disponibles en Alpaca
    CRYPTO_PAIRS = [
        # Principales
        'BTC/USD',    # Bitcoin
        'ETH/USD',    # Ethereum
        
        # Large Cap
        'LTC/USD',    # Litecoin
        'BCH/USD',    # Bitcoin Cash
        'LINK/USD',   # Chainlink
        'UNI/USD',    # Uniswap
        'AAVE/USD',   # Aave
        
        # DeFi & Smart Contracts
        'AVAX/USD',   # Avalanche
        'DOT/USD',    # Polkadot
        'MATIC/USD',  # Polygon
        'SOL/USD',    # Solana
        'ADA/USD',    # Cardano
        
        # Meme & Community
        'DOGE/USD',   # Dogecoin
        'SHIB/USD',   # Shiba Inu
        
        # Stablecoins (para arbitraje)
        'USDT/USD',   # Tether
        'USDC/USD',   # USD Coin
        
        # Exchange Tokens
        'CRV/USD',    # Curve
        'SUSHI/USD',  # SushiSwap
        
        # Layer 2 & Scaling
        'MKR/USD',    # Maker
        'COMP/USD',   # Compound
        'YFI/USD',    # Yearn Finance
        
        # Metaverse & Gaming
        'BAT/USD',    # Basic Attention Token
        'GRT/USD',    # The Graph
    ]
    
    # Símbolos para formato de API (sin slash)
    CRYPTO_SYMBOLS = [pair.replace('/', '') for pair in CRYPTO_PAIRS]
    
    # ===== 10 AGENTES ESPECIALIZADOS POR PAR =====
    AGENTS_PER_CRYPTO = 10
    
    AGENT_TYPES = {
        # === Deep Learning Agents (1-5) ===
        'lstm_trend': {
            'type': 'LSTM',
            'specialization': 'Tendencias a largo plazo',
            'layers': [256, 128, 64],
            'dropout': 0.3,
            'lookback': 100,
            'weight': 0.15
        },
        'gru_momentum': {
            'type': 'GRU',
            'specialization': 'Momentum y velocidad',
            'layers': [256, 128],
            'dropout': 0.25,
            'lookback': 60,
            'weight': 0.12
        },
        'cnn_patterns': {
            'type': 'CNN',
            'specialization': 'Patrones de velas',
            'filters': [128, 64, 32],
            'kernel_size': 3,
            'lookback': 50,
            'weight': 0.10
        },
        'transformer_attention': {
            'type': 'Transformer',
            'specialization': 'Atención multi-temporal',
            'heads': 8,
            'dim': 256,
            'lookback': 80,
            'weight': 0.13
        },
        'bilstm_reversal': {
            'type': 'BiLSTM',
            'specialization': 'Detección de reversiones',
            'layers': [128, 64],
            'dropout': 0.2,
            'lookback': 70,
            'weight': 0.10
        },
        
        # === Machine Learning Agents (6-10) ===
        'xgboost_scalping': {
            'type': 'XGBoost',
            'specialization': 'Scalping rápido',
            'estimators': 300,
            'max_depth': 10,
            'learning_rate': 0.1,
            'weight': 0.08
        },
        'lightgbm_volatility': {
            'type': 'LightGBM',
            'specialization': 'Trading en volatilidad',
            'estimators': 300,
            'num_leaves': 31,
            'learning_rate': 0.1,
            'weight': 0.08
        },
        'random_forest_support': {
            'type': 'RandomForest',
            'specialization': 'Soportes y resistencias',
            'estimators': 500,
            'max_depth': 20,
            'min_samples_split': 5,
            'weight': 0.08
        },
        'catboost_arbitrage': {
            'type': 'CatBoost',
            'specialization': 'Arbitraje entre exchanges',
            'iterations': 300,
            'depth': 8,
            'learning_rate': 0.1,
            'weight': 0.08
        },
        'gradient_boost_breakout': {
            'type': 'GradientBoosting',
            'specialization': 'Breakouts y rupturas',
            'estimators': 300,
            'max_depth': 10,
            'learning_rate': 0.1,
            'weight': 0.08
        }
    }
    
    # ===== TIMEFRAMES PARA CRYPTO =====
    CRYPTO_TIMEFRAMES = {
        'scalping': '1Min',      # Para operaciones rápidas
        'intraday': '5Min',      # Trading intradía
        'swing': '15Min',        # Swing trading
        'position': '1Hour',     # Posiciones más largas
        'trend': '4Hour'         # Análisis de tendencia
    }
    
    # ===== PARÁMETROS DE TRADING CRYPTO =====
    # Ajustados para la alta volatilidad de crypto
    CRYPTO_TRADING_PARAMS = {
        'max_position_size': 0.03,      # 3% máximo por operación
        'max_positions': 15,             # Máximo 15 posiciones simultáneas
        'min_confidence': 0.75,          # 75% confianza mínima
        
        # Risk Management Crypto
        'stop_loss': 0.02,               # 2% stop loss
        'take_profit': 0.04,             # 4% take profit
        'trailing_stop': 0.015,          # 1.5% trailing stop
        
        # Gestión de volatilidad
        'max_volatility': 10,            # Volatilidad máxima 10%
        'min_volume_usd': 100000,        # Volumen mínimo $100k
        'max_spread_pct': 0.002,         # Spread máximo 0.2%
        
        # Tiempos
        'cooldown_minutes': 5,           # 5 minutos entre trades del mismo par
        'max_hold_hours': 24,            # Máximo 24 horas por posición
        
        # Capital
        'initial_capital': 10000,        # Capital inicial
        'compound_profits': True,        # Reinvertir ganancias
        'max_daily_loss': 0.05,          # Pérdida máxima diaria 5%
    }
    
    # ===== ESTRATEGIAS POR TIPO DE MERCADO =====
    MARKET_STRATEGIES = {
        'bull': {
            'long_bias': 0.7,
            'short_bias': 0.3,
            'take_profit_multiplier': 1.5
        },
        'bear': {
            'long_bias': 0.3,
            'short_bias': 0.7,
            'stop_loss_multiplier': 0.8
        },
        'sideways': {
            'long_bias': 0.5,
            'short_bias': 0.5,
            'range_trading': True
        }
    }
    
    # ===== HORARIOS ESPECIALES CRYPTO =====
    # Aunque crypto opera 24/7, hay horas de mayor actividad
    HIGH_ACTIVITY_HOURS = {
        'asia': range(20, 4),      # 8 PM - 4 AM UTC (Asia activa)
        'europe': range(7, 16),    # 7 AM - 4 PM UTC (Europa activa)
        'americas': range(13, 23)  # 1 PM - 11 PM UTC (Americas activa)
    }
    
    # ===== PATHS =====
    BASE_DIR = os.getcwd()
    MODELS_DIR = os.path.join(BASE_DIR, 'models', 'crypto')
    DATA_DIR = os.path.join(BASE_DIR, 'data', 'crypto')
    LOGS_DIR = os.path.join(BASE_DIR, 'logs', 'crypto')
    
    # Crear directorios
    for dir_path in [MODELS_DIR, DATA_DIR, LOGS_DIR]:
        os.makedirs(dir_path, exist_ok=True)
