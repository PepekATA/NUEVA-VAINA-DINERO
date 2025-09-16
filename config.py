import os
from dotenv import load_dotenv
import pytz

load_dotenv()

class Config:
    # Alpaca API
    ALPACA_API_KEY = os.getenv('ALPACA_API_KEY')
    ALPACA_SECRET_KEY = os.getenv('ALPACA_SECRET_KEY')
    ALPACA_BASE_URL = 'https://paper-api.alpaca.markets'  # Paper trading
    
    # GitHub
    GITHUB_TOKEN = os.getenv('GITHUB_TOKEN')
    GITHUB_REPO = os.getenv('GITHUB_REPO')
    
    # Google Drive
    GDRIVE_FOLDER_ID = os.getenv('GDRIVE_FOLDER_ID')
    
    # Trading Parameters
    INITIAL_CAPITAL = 100.0
    SYMBOLS = ['AAPL', 'AMZN', 'TSLA', 'MSFT', 'NVDA']
    TIMEFRAMES = ['1Min', '5Min', '15Min', '30Min', '1Hour', '4Hour', '1Day']
    
    # Market Hours (ET)
    MARKET_TIMEZONE = pytz.timezone('America/New_York')
    MARKET_OPEN_TIME = '09:30'
    MARKET_CLOSE_TIME = '16:00'
    
    # Model Parameters
    LOOKBACK_PERIOD = 60
    PREDICTION_THRESHOLD = 0.65
    MAX_POSITIONS = 5
    STOP_LOSS_PCT = 0.02
    TAKE_PROFIT_PCT = 0.05
    
    # File Paths
    MODELS_DIR = 'models'
    DATA_DIR = 'data'
    LOGS_DIR = 'logs'
