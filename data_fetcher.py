import alpaca_trade_api as tradeapi
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz
import yfinance as yf
from ta import add_all_ta_features
from ta.utils import dropna
import time
import logging
from config import Config

class DataFetcher:
    def __init__(self):
        self.api = tradeapi.REST(
            Config.ALPACA_API_KEY,
            Config.ALPACA_SECRET_KEY,
            Config.ALPACA_BASE_URL,
            api_version='v2'
        )
        self.setup_logging()
        
    def setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(f'{Config.LOGS_DIR}/data_fetcher.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def is_market_open(self):
        """Check if market is currently open"""
        clock = self.api.get_clock()
        return clock.is_open
        
    def get_market_hours(self):
        """Get market hours for today"""
        clock = self.api.get_clock()
        return {
            'is_open': clock.is_open,
            'next_open': clock.next_open,
            'next_close': clock.next_close
        }
        
    def fetch_realtime_data(self, symbol, timeframe='1Min', limit=1000):
        """Fetch real-time data from Alpaca"""
        try:
            end = datetime.now(Config.MARKET_TIMEZONE)
            
            if timeframe == '1Min':
                start = end - timedelta(minutes=limit)
                barset = self.api.get_bars(
                    symbol,
                    '1Min',
                    start=start.isoformat(),
                    end=end.isoformat(),
                    limit=limit
                ).df
            elif timeframe == '5Min':
                start = end - timedelta(minutes=limit*5)
                barset = self.api.get_bars(
                    symbol,
                    '5Min',
                    start=start.isoformat(),
                    end=end.isoformat(),
                    limit=limit
                ).df
            elif timeframe == '15Min':
                start = end - timedelta(minutes=limit*15)
                barset = self.api.get_bars(
                    symbol,
                    '15Min',
                    start=start.isoformat(),
                    end=end.isoformat(),
                    limit=limit
                ).df
            elif timeframe == '1Hour':
                start = end - timedelta(hours=limit)
                barset = self.api.get_bars(
                    symbol,
                    '1Hour',
                    start=start.isoformat(),
                    end=end.isoformat(),
                    limit=limit
                ).df
            elif timeframe == '1Day':
                start = end - timedelta(days=limit)
                barset = self.api.get_bars(
                    symbol,
                    '1Day',
                    start=start.isoformat(),
                    end=end.isoformat(),
                    limit=limit
                ).df
            else:
                barset = self.api.get_bars(
                    symbol,
                    '1Min',
                    start=start.isoformat(),
                    end=end.isoformat(),
                    limit=limit
                ).df
                
            if not barset.empty:
                barset = self.add_technical_indicators(barset)
                self.logger.info(f"Fetched {len(barset)} bars for {symbol} - {timeframe}")
                return barset
            else:
                self.logger.warning(f"No data available for {symbol} - {timeframe}")
                return pd.DataFrame()
                
        except Exception as e:
            self.logger.error(f"Error fetching data for {symbol}: {str(e)}")
            return pd.DataFrame()
            
    def add_technical_indicators(self, df):
        """Add technical indicators to dataframe"""
        try:
            df = df.copy()
            df.columns = ['open', 'high', 'low', 'close', 'volume', 'trade_count', 'vwap']
            
            # Add all technical indicators
            df = add_all_ta_features(
                df, open="open", high="high", low="low", close="close", volume="volume"
            )
            
            # Calculate custom indicators
            df['returns'] = df['close'].pct_change()
            df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
            
            # Price patterns
            df['high_low_pct'] = (df['high'] - df['low']) / df['close'] * 100
            df['close_open_pct'] = (df['close'] - df['open']) / df['open'] * 100
            
            # Volume indicators
            df['volume_ratio'] = df['volume'] / df['volume'].rolling(window=20).mean()
            
            # Clean data
            df = df.replace([np.inf, -np.inf], np.nan)
            df = df.fillna(method='ffill').fillna(0)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error adding indicators: {str(e)}")
            return df
            
    def get_streaming_quotes(self, symbols):
        """Stream real-time quotes"""
        try:
            stream = tradeapi.Stream(
                Config.ALPACA_API_KEY,
                Config.ALPACA_SECRET_KEY,
                base_url=Config.ALPACA_BASE_URL,
                data_feed='iex'
            )
            
            async def quote_callback(q):
                self.logger.info(f'Quote: {q}')
                
            for symbol in symbols:
                stream.subscribe_quotes(quote_callback, symbol)
                
            stream.run()
            
        except Exception as e:
            self.logger.error(f"Streaming error: {str(e)}")
            
    def save_data(self, data, symbol, timeframe):
        """Save data to CSV"""
        try:
            filename = f"{Config.DATA_DIR}/{symbol}_{timeframe}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            data.to_csv(filename)
            self.logger.info(f"Data saved to {filename}")
            return filename
        except Exception as e:
            self.logger.error(f"Error saving data: {str(e)}")
            return None
