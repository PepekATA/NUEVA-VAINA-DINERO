#!/usr/bin/env python3
import sys
import argparse
import logging
from datetime import datetime
from config import Config
from data_fetcher import DataFetcher
from train_model import ModelTrainer
from predictor import Predictor
from trader import Trader
from utils import Utils
import os

def setup_logging():
    os.makedirs(Config.LOGS_DIR, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f'{Config.LOGS_DIR}/main.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def main():
    logger = setup_logging()
    parser = argparse.ArgumentParser(description='Trading Bot')
    
    parser.add_argument('--mode', choices=['train', 'trade', 'predict', 'gui', 'backtest'],
                      default='gui', help='Operation mode')
    parser.add_argument('--symbols', nargs='+', default=Config.SYMBOLS,
                      help='Symbols to trade')
    parser.add_argument('--timeframe', default='1Min',
                      help='Timeframe for analysis')
    parser.add_argument('--continuous', action='store_true',
                      help='Run in continuous mode')
    
    args = parser.parse_args()
    
    # Initialize components
    utils = Utils()
    
    if args.mode == 'train':
        logger.info("Starting training mode...")
        trainer = ModelTrainer()
        
        if args.continuous:
            trainer.continuous_training(interval_minutes=60)
        else:
            for symbol in args.symbols:
                trainer.train_model(symbol, args.timeframe)
                
            # Sync with cloud
            utils.sync_with_github()
            utils.sync_with_gdrive()
            
    elif args.mode == 'trade':
        logger.info("Starting trading mode...")
        trader = Trader()
        trader.run_trading_bot(mode='auto', interval_seconds=60)
        
    elif args.mode == 'predict':
        logger.info("Starting prediction mode...")
        predictor = Predictor()
        
        if args.continuous:
            predictor.continuous_prediction(interval_seconds=60)
        else:
            predictions = predictor.predict_all_symbols(args.timeframe)
            for pred in predictions:
                print(f"{pred['symbol']}: {pred['direction']} "
                     f"({pred['probability']:.2%}) - "
                     f"Duration: {pred['duration_minutes']} min")
                     
    elif args.mode == 'gui':
        logger.info("Starting GUI mode...")
        from gui_app import app
        app.run(debug=False, host='0.0.0.0', port=5000)
        
    elif args.mode == 'backtest':
        logger.info("Starting backtest mode...")
        # Implement backtesting logic here
        pass
        
    logger.info("Bot execution completed")

if __name__ == "__main__":
    main()
