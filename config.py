import os
from dotenv import load_dotenv
import pytz
import json
from pathlib import Path

load_dotenv()

class Config:
    # Detectar si estamos en Streamlit Cloud
    try:
        import streamlit as st
        IN_STREAMLIT = True
        # Si estamos en Streamlit, usar secrets
        ALPACA_API_KEY = st.secrets.get("ALPACA_API_KEY", os.getenv('ALPACA_API_KEY'))
        ALPACA_SECRET_KEY = st.secrets.get("ALPACA_SECRET_KEY", os.getenv('ALPACA_SECRET_KEY'))
        GITHUB_TOKEN = st.secrets.get("GITHUB_TOKEN", os.getenv('GITHUB_TOKEN'))
        GITHUB_REPO = st.secrets.get("GITHUB_REPO", os.getenv('GITHUB_REPO'))
        GDRIVE_FOLDER_ID = st.secrets.get("GDRIVE_FOLDER_ID", os.getenv('GDRIVE_FOLDER_ID'))
    except:
        IN_STREAMLIT = False
        # Si no estamos en Streamlit, usar variables de entorno o credenciales locales
        
        # Paths para credenciales
        COLAB_CREDS_PATH = '/content/drive/MyDrive/TradingBot/credentials.json'
        LOCAL_CREDS_PATH = 'credentials.json'
        
        # Intentar cargar credenciales desde archivo
        creds = {}
        if os.path.exists(COLAB_CREDS_PATH):
            with open(COLAB_CREDS_PATH, 'r') as f:
                creds = json.load(f)
        elif os.path.exists(LOCAL_CREDS_PATH):
            with open(LOCAL_CREDS_PATH, 'r') as f:
                creds = json.load(f)
        
        # Usar variables de entorno o credenciales del archivo
        ALPACA_API_KEY = os.getenv('ALPACA_API_KEY') or creds.get('ALPACA_API_KEY')
        ALPACA_SECRET_KEY = os.getenv('ALPACA_SECRET_KEY') or creds.get('ALPACA_SECRET_KEY')
        GITHUB_TOKEN = os.getenv('GITHUB_TOKEN') or creds.get('GITHUB_TOKEN')
        GITHUB_REPO = os.getenv('GITHUB_REPO') or creds.get('GITHUB_REPO')
        GDRIVE_FOLDER_ID = os.getenv('GDRIVE_FOLDER_ID') or creds.get('GDRIVE_FOLDER_ID')
    
    # Alpaca API Configuration
    ALPACA_BASE_URL = 'https://paper-api.alpaca.markets'  # Paper trading para empezar
    
    # Trading Parameters
    INITIAL_CAPITAL = 100.0
    
    # Complete list of symbols to trade
    SYMBOLS = [
        # FAANG+ Tech Giants
        'AAPL', 'AMZN', 'TSLA', 'MSFT', 'NVDA', 'GOOGL', 'META', 'NFLX',
        
        # Semiconductors
        'AMD', 'INTC', 'QCOM', 'MU', 'AVGO', 'TXN', 'ADI', 'MRVL',
        
        # ETFs
        'SPY', 'QQQ', 'DIA', 'IWM', 'VOO', 'VTI', 'ARKK', 'XLK',
        
        # Financials
        'V', 'MA', 'PYPL', 'SQ', 'AXP', 'DFS', 'COF', 'SYF',
        'JPM', 'BAC', 'WFC', 'GS', 'MS', 'C', 'USB', 'PNC',
        
        # Retail & Consumer
        'DIS', 'NKE', 'LULU', 'MCD', 'SBUX', 'CMG', 'YUM', 'DPZ',
        'WMT', 'HD', 'LOW', 'TGT', 'COST', 'KR', 'CVS', 'WBA',
        
        # Healthcare & Pharma
        'JNJ', 'PFE', 'MRK', 'BMY', 'GSK', 'ABBV', 'LLY', 'TMO',
        'UNH', 'CI', 'ANTM', 'HUM', 'MCK', 'ABC', 'CAH',
        
        # Energy
        'CVX', 'XOM', 'COP', 'EOG', 'PXD', 'DVN', 'FANG', 'MRO',
        'OXY', 'HES', 'MPC', 'VLO', 'PSX', 'SLB', 'HAL',
        
        # Real Estate & Infrastructure
        'AMT', 'CCI', 'PLD', 'EQIX', 'DLR', 'SBAC', 'PSA', 'O',
        
        # Telecom
        'VZ', 'T', 'TMUS', 'CHTR', 'CMCSA',
        
        # Enterprise Software
        'CSCO', 'ORCL', 'IBM', 'SAP', 'CRM', 'ADBE', 'INTU', 'NOW',
        'WDAY', 'SNOW', 'TEAM', 'DOCU', 'OKTA', 'ZM', 'TWLO',
        
        # Chinese Tech
        'BABA', 'JD', 'PDD', 'BIDU', 'NTES', 'IQ', 'YY', 'BILI',
        
        # Latin America
        'MELI', 'NU', 'PAGS', 'STNE', 'DESP', 'GLOB',
        
        # Gaming & Entertainment
        'RBLX', 'EA', 'TTWO', 'ATVI', 'U', 'DKNG', 'PENN',
        
        # Mobility & Delivery
        'UBER', 'LYFT', 'GRAB', 'DASH', 'ABNB',
        
        # EV & Clean Energy
        'RIVN', 'LCID', 'NIO', 'XPEV', 'LI', 'FSR', 'PLUG', 'FCEL',
        'ENPH', 'SEDG', 'RUN', 'NOVA', 'SPWR',
        
        # Industrials
        'CAT', 'DE', 'BA', 'LMT', 'RTX', 'NOC', 'GD', 'HII',
        'GE', 'HON', 'MMM', 'EMR', 'ROK', 'ITW', 'DOV',
        
        # Consumer Staples
        'PG', 'KO', 'PEP', 'MDLZ', 'MO', 'PM', 'KHC', 'GIS',
        'K', 'CPB', 'HSY', 'MNST', 'KDP'
    ]
    
    # Minimum 6 agents per asset
    AGENTS_PER_SYMBOL = 6
    
    # Agent types for ensemble learning
    AGENT_TYPES = [
        'lstm',           # Long Short-Term Memory
        'gru',            # Gated Recurrent Unit
        'cnn',            # Convolutional Neural Network
        'transformer',    # Transformer architecture
        'random_forest',  # Random Forest
        'xgboost',        # XGBoost
        'lightgbm',       # LightGBM
        'catboost'        # CatBoost
    ]
    
    TIMEFRAMES = ['1Min', '5Min', '15Min', '30Min', '1Hour', '4Hour', '1Day']
    
    # Market Hours (ET)
    MARKET_TIMEZONE = pytz.timezone('America/New_York')
    MARKET_OPEN_TIME = '09:30'
    MARKET_CLOSE_TIME = '16:00'
    
    # Model Parameters
    LOOKBACK_PERIOD = 60
    PREDICTION_THRESHOLD = 0.65
    MAX_POSITIONS = 10
    STOP_LOSS_PCT = 0.02
    TAKE_PROFIT_PCT = 0.05
    
    # File Paths
    if IN_STREAMLIT:
        # En Streamlit Cloud usar paths temporales
        MODELS_DIR = '/tmp/models'
        DATA_DIR = '/tmp/data'
        LOGS_DIR = '/tmp/logs'
    else:
        # En local o Colab usar paths normales
        MODELS_DIR = 'models'
        DATA_DIR = 'data'
        LOGS_DIR = 'logs'
    
    # Crear directorios si no existen
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(LOGS_DIR, exist_ok=True)
    
    # Google Colab specific paths
    COLAB_DRIVE_PATH = '/content/drive/MyDrive/TradingBot'
    
    @classmethod
    def save_credentials(cls, alpaca_key, alpaca_secret, github_token, github_repo, gdrive_folder_id):
        """Save credentials to file"""
        creds = {
            'ALPACA_API_KEY': alpaca_key,
            'ALPACA_SECRET_KEY': alpaca_secret,
            'GITHUB_TOKEN': github_token,
            'GITHUB_REPO': github_repo,
            'GDRIVE_FOLDER_ID': gdrive_folder_id
        }
        
        try:
            # Check if running in Google Colab
            if os.path.exists('/content'):
                creds_path = cls.COLAB_CREDS_PATH
                os.makedirs(os.path.dirname(creds_path), exist_ok=True)
            else:
                creds_path = cls.LOCAL_CREDS_PATH
                
            with open(creds_path, 'w') as f:
                json.dump(creds, f)
            return True
        except:
            return False
