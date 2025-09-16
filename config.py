import os
from dotenv import load_dotenv
import pytz
import json
from pathlib import Path

load_dotenv()

class Config:
    # Paths for credentials
    COLAB_CREDS_PATH = '/content/drive/MyDrive/TradingBot/credentials.json'
    LOCAL_CREDS_PATH = 'credentials.json'
    
    @classmethod
    def load_credentials(cls):
        """Load credentials from Google Drive in Colab or local file"""
        try:
            # Check if running in Google Colab
            if 'google.colab' in str(get_ipython()):
                creds_path = cls.COLAB_CREDS_PATH
            else:
                creds_path = cls.LOCAL_CREDS_PATH
                
            if os.path.exists(creds_path):
                with open(creds_path, 'r') as f:
                    creds = json.load(f)
                    return creds
            return None
        except:
            return None
    
    @classmethod
    def save_credentials(cls, alpaca_key, alpaca_secret, github_token, gdrive_folder_id):
        """Save credentials to file"""
        creds = {
            'ALPACA_API_KEY': alpaca_key,
            'ALPACA_SECRET_KEY': alpaca_secret,
            'GITHUB_TOKEN': github_token,
            'GDRIVE_FOLDER_ID': gdrive_folder_id
        }
        
        try:
            # Check if running in Google Colab
            if 'google.colab' in str(get_ipython()):
                creds_path = cls.COLAB_CREDS_PATH
                os.makedirs(os.path.dirname(creds_path), exist_ok=True)
            else:
                creds_path = cls.LOCAL_CREDS_PATH
                
            with open(creds_path, 'w') as f:
                json.dump(creds, f)
            return True
        except:
            return False
    
    # Load credentials on startup
    _creds = load_credentials.__func__(None) or {}
    
    # Alpaca API
    ALPACA_API_KEY = os.getenv('ALPACA_API_KEY') or _creds.get('ALPACA_API_KEY')
    ALPACA_SECRET_KEY = os.getenv('ALPACA_SECRET_KEY') or _creds.get('ALPACA_SECRET_KEY')
    ALPACA_BASE_URL = 'https://paper-api.alpaca.markets'  # Paper trading
    
    # GitHub
    GITHUB_TOKEN = os.getenv('GITHUB_TOKEN') or _creds.get('GITHUB_TOKEN')
    GITHUB_REPO = os.getenv('GITHUB_REPO') or _creds.get('GITHUB_REPO')
    
    # Google Drive
    GDRIVE_FOLDER_ID = os.getenv('GDRIVE_FOLDER_ID') or _creds.get('GDRIVE_FOLDER_ID')
    
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
    MODELS_DIR = 'models'
    DATA_DIR = 'data'
    LOGS_DIR = 'logs'
    
    # Google Colab specific paths
    COLAB_DRIVE_PATH = '/content/drive/MyDrive/TradingBot'
