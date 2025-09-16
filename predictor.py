import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging
from config import Config
from data_fetcher import DataFetcher
from train_model import ModelTrainer

class Predictor:
    def __init__(self):
        self.data_fetcher = DataFetcher()
        self.model_trainer = ModelTrainer()
        self.predictions = {}
        self.setup_logging()
        self.load_all_models()
        
    def setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(f'{Config.LOGS_DIR}/predictor.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def load_all_models(self):
        """Load all available models"""
        for symbol in Config.SYMBOLS:
            for timeframe in Config.TIMEFRAMES:
                self.model_trainer.load_model(symbol, timeframe)
                
    def prepare_prediction_data(self, df, lookback=60):
        """Prepare data for prediction"""
        try:
            feature_columns = [col for col in df.columns if col not in 
                             ['open', 'high', 'low', 'close', 'volume', 'trade_count']]
            
            X = df[feature_columns].values
            
            if len(X) < lookback:
                return None
                
            X_seq = X[-lookback:].reshape(1, lookback, -1)
            
            return X_seq
            
        except Exception as e:
            self.logger.error(f"Error preparing prediction data: {str(e)}")
            return None
            
    def predict(self, symbol, timeframe='1Min'):
        """Make prediction for symbol"""
        try:
            # Fetch latest data
            df = self.data_fetcher.fetch_realtime_data(symbol, timeframe)
            
            if df.empty:
                self.logger.error(f"No data for {symbol}")
                return None
                
            # Prepare data
            X = self.prepare_prediction_data(df, Config.LOOKBACK_PERIOD)
            
            if X is None:
                return None
                
            # Get model
            model_key = f"{symbol}_{timeframe}"
            if model_key not in self.model_trainer.models:
                self.logger.warning(f"No model for {symbol} - {timeframe}")
                return None
                
            model_data = self.model_trainer.models[model_key]
            scaler = model_data.get('scaler')
            
            # Scale data
            if scaler:
                X_scaled = X.reshape(X.shape[0], -1)
                X_scaled = scaler.transform(X_scaled)
                X_scaled = X_scaled.reshape(X.shape)
            else:
                X_scaled = X
                
            predictions = {}
            
            # LSTM prediction
            if model_data.get('lstm'):
                lstm_pred = model_data['lstm'].predict(X_scaled)[0][0]
                predictions['lstm'] = lstm_pred
                
            # CNN prediction
            if model_data.get('cnn'):
                cnn_pred = model_data['cnn'].predict(X_scaled)[0][0]
                predictions['cnn'] = cnn_pred
                
            # Ensemble predictions
            if model_data.get('ensemble'):
                X_flat = X_scaled.reshape(X_scaled.shape[0], -1)
                for name, model in model_data['ensemble'].items():
                    pred = model.predict_proba(X_flat)[0][1]
                    predictions[name] = pred
                    
            # Calculate ensemble prediction
            if predictions:
                avg_prediction = np.mean(list(predictions.values()))
                
                # Determine trend duration based on timeframe
                duration_map = {
                    '1Min': 1,
                    '5Min': 5,
                    '15Min': 15,
                    '30Min': 30,
                    '1Hour': 60,
                    '4Hour': 240,
                    '1Day': 1440
                }
                
                duration_minutes = duration_map.get(timeframe, 5)
                
                result = {
                    'symbol': symbol,
                    'timeframe': timeframe,
                    'timestamp': datetime.now(),
                    'direction': 'UP' if avg_prediction > 0.5 else 'DOWN',
                    'probability': avg_prediction if avg_prediction > 0.5 else 1 - avg_prediction,
                    'confidence': abs(avg_prediction - 0.5) * 2,
                    'duration_minutes': duration_minutes,
                    'predictions': predictions,
                    'current_price': float(df['close'].iloc[-1]),
                    'predicted_change_pct': (avg_prediction - 0.5) * 10
                }
                
                # Store prediction
                self.predictions[f"{symbol}_{timeframe}"] = result
                
                self.logger.info(f"Prediction for {symbol}: {result['direction']} "
                               f"(prob: {result['probability']:.2%})")
                
                return result
            else:
                self.logger.warning(f"No predictions available for {symbol}")
                return None
                
        except Exception as e:
            self.logger.error(f"Error making prediction: {str(e)}")
            return None
            
    def predict_all_symbols(self, timeframe='1Min'):
        """Predict for all symbols"""
        results = []
        
        for symbol in Config.SYMBOLS:
            prediction = self.predict(symbol, timeframe)
            if prediction:
                results.append(prediction)
                
        # Sort by confidence
        results.sort(key=lambda x: x['confidence'], reverse=True)
        
        return results
        
    def get_best_opportunities(self, min_confidence=0.65):
        """Get best trading opportunities"""
        opportunities = []
        
        for timeframe in ['1Min', '5Min', '15Min']:
            predictions = self.predict_all_symbols(timeframe)
            
            for pred in predictions:
                if pred['probability'] >= min_confidence:
                    opportunities.append(pred)
                    
        # Sort by probability and confidence
        opportunities.sort(key=lambda x: (x['probability'], x['confidence']), reverse=True)
        
        return opportunities[:Config.MAX_POSITIONS]
        
    def continuous_prediction(self, interval_seconds=60):
        """Continuous prediction loop"""
        import time
        
        while True:
            try:
                if self.data_fetcher.is_market_open():
                    self.logger.info("Running predictions...")
                    
                    opportunities = self.get_best_opportunities()
                    
                    for opp in opportunities:
                        self.logger.info(f"Opportunity: {opp['symbol']} - "
                                       f"{opp['direction']} - "
                                       f"Prob: {opp['probability']:.2%}")
                        
                    time.sleep(interval_seconds)
                else:
                    self.logger.info("Market closed. Waiting...")
                    time.sleep(300)  # Check every 5 minutes
                    
            except KeyboardInterrupt:
                self.logger.info("Stopping continuous prediction")
                break
            except Exception as e:
                self.logger.error(f"Error in continuous prediction: {str(e)}")
                time.sleep(60)
                
    def save_predictions(self):
        """Save predictions to CSV"""
        try:
            if self.predictions:
                df = pd.DataFrame(list(self.predictions.values()))
                filename = f"{Config.DATA_DIR}/predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                df.to_csv(filename, index=False)
                self.logger.info(f"Predictions saved to {filename}")
                return filename
            return None
        except Exception as e:
            self.logger.error(f"Error saving predictions: {str(e)}")
            return None
