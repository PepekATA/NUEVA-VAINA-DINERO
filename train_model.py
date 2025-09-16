import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, Conv1D, MaxPooling1D, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import joblib
import os
from datetime import datetime
import logging
from config import Config
from data_fetcher import DataFetcher

class ModelTrainer:
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.data_fetcher = DataFetcher()
        self.setup_logging()
        
    def setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(f'{Config.LOGS_DIR}/model_trainer.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def prepare_features(self, df, lookback=60):
        """Prepare features for training"""
        try:
            # Select features
            feature_columns = [col for col in df.columns if col not in 
                             ['open', 'high', 'low', 'close', 'volume', 'trade_count']]
            
            X = df[feature_columns].values
            
            # Create labels (1 for price up, 0 for price down)
            y = (df['close'].shift(-1) > df['close']).astype(int).values
            
            # Remove last row (no label)
            X = X[:-1]
            y = y[:-1]
            
            # Create sequences for LSTM
            X_seq = []
            y_seq = []
            
            for i in range(lookback, len(X)):
                X_seq.append(X[i-lookback:i])
                y_seq.append(y[i])
                
            return np.array(X_seq), np.array(y_seq)
            
        except Exception as e:
            self.logger.error(f"Error preparing features: {str(e)}")
            return None, None
            
    def build_lstm_model(self, input_shape):
        """Build LSTM model"""
        model = Sequential([
            LSTM(100, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            LSTM(100, return_sequences=True),
            Dropout(0.2),
            LSTM(50),
            Dropout(0.2),
            Dense(25, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model
        
    def build_cnn_model(self, input_shape):
        """Build CNN model for time series"""
        model = Sequential([
            Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=input_shape),
            MaxPooling1D(pool_size=2),
            Conv1D(filters=32, kernel_size=3, activation='relu'),
            Flatten(),
            Dense(50, activation='relu'),
            Dropout(0.2),
            Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model
        
    def train_ensemble_models(self, X_train, y_train, X_test, y_test):
        """Train ensemble models"""
        models = {}
        
        # Random Forest
        rf_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        rf_model.fit(X_train.reshape(X_train.shape[0], -1), y_train)
        models['random_forest'] = rf_model
        
        # Gradient Boosting
        gb_model = GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        )
        gb_model.fit(X_train.reshape(X_train.shape[0], -1), y_train)
        models['gradient_boosting'] = gb_model
        
        # Evaluate models
        for name, model in models.items():
            predictions = model.predict(X_test.reshape(X_test.shape[0], -1))
            accuracy = accuracy_score(y_test, predictions)
            self.logger.info(f"{name} Accuracy: {accuracy:.4f}")
            
        return models
        
    def train_model(self, symbol, timeframe='1Min', epochs=50, batch_size=32):
        """Train model for specific symbol and timeframe"""
        try:
            self.logger.info(f"Training model for {symbol} - {timeframe}")
            
            # Fetch data
            df = self.data_fetcher.fetch_realtime_data(symbol, timeframe, limit=1000)
            
            if df.empty:
                self.logger.error(f"No data available for {symbol}")
                return None
                
            # Prepare features
            X, y = self.prepare_features(df, lookback=Config.LOOKBACK_PERIOD)
            
            if X is None or len(X) == 0:
                self.logger.error(f"Failed to prepare features for {symbol}")
                return None
                
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, shuffle=False
            )
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = X_train.reshape(X_train.shape[0], -1)
            X_test_scaled = X_test.reshape(X_test.shape[0], -1)
            
            X_train_scaled = scaler.fit_transform(X_train_scaled)
            X_test_scaled = scaler.transform(X_test_scaled)
            
            X_train_scaled = X_train_scaled.reshape(X_train.shape)
            X_test_scaled = X_test_scaled.reshape(X_test.shape)
            
            # Train LSTM model
            lstm_model = self.build_lstm_model((X_train.shape[1], X_train.shape[2]))
            
            early_stop = EarlyStopping(monitor='val_loss', patience=10)
            checkpoint = ModelCheckpoint(
                f"{Config.MODELS_DIR}/{symbol}_{timeframe}_lstm.h5",
                save_best_only=True,
                monitor='val_accuracy',
                mode='max'
            )
            
            history = lstm_model.fit(
                X_train_scaled, y_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=(X_test_scaled, y_test),
                callbacks=[early_stop, checkpoint],
                verbose=1
            )
            
            # Train CNN model
            cnn_model = self.build_cnn_model((X_train.shape[1], X_train.shape[2]))
            
            checkpoint_cnn = ModelCheckpoint(
                f"{Config.MODELS_DIR}/{symbol}_{timeframe}_cnn.h5",
                save_best_only=True,
                monitor='val_accuracy',
                mode='max'
            )
            
            cnn_model.fit(
                X_train_scaled, y_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=(X_test_scaled, y_test),
                callbacks=[early_stop, checkpoint_cnn],
                verbose=1
            )
            
            # Train ensemble models
            ensemble_models = self.train_ensemble_models(
                X_train_scaled, y_train, X_test_scaled, y_test
            )
            
            # Save models and scaler
            model_key = f"{symbol}_{timeframe}"
            self.models[model_key] = {
                'lstm': lstm_model,
                'cnn': cnn_model,
                'ensemble': ensemble_models,
                'scaler': scaler,
                'history': history.history,
                'timestamp': datetime.now()
            }
            
            self.scalers[model_key] = scaler
            
            # Save ensemble models
            for name, model in ensemble_models.items():
                joblib.dump(model, f"{Config.MODELS_DIR}/{symbol}_{timeframe}_{name}.pkl")
                
            # Save scaler
            joblib.dump(scaler, f"{Config.MODELS_DIR}/{symbol}_{timeframe}_scaler.pkl")
            
            self.logger.info(f"Model training completed for {symbol} - {timeframe}")
            
            return self.models[model_key]
            
        except Exception as e:
            self.logger.error(f"Error training model for {symbol}: {str(e)}")
            return None
            
    def train_all_models(self):
        """Train models for all symbols and timeframes"""
        results = {}
        
        for symbol in Config.SYMBOLS:
            for timeframe in Config.TIMEFRAMES:
                result = self.train_model(symbol, timeframe)
                results[f"{symbol}_{timeframe}"] = result is not None
                
        return results
        
    def continuous_training(self, interval_minutes=60):
        """Continuous training loop"""
        import schedule
        import time
        
        def train_job():
            if self.data_fetcher.is_market_open():
                self.logger.info("Starting continuous training...")
                self.train_all_models()
                self.logger.info("Continuous training completed")
                
        schedule.every(interval_minutes).minutes.do(train_job)
        
        while True:
            schedule.run_pending()
            time.sleep(1)
            
    def load_model(self, symbol, timeframe):
        """Load saved model"""
        try:
            model_key = f"{symbol}_{timeframe}"
            
            # Load LSTM model
            lstm_path = f"{Config.MODELS_DIR}/{symbol}_{timeframe}_lstm.h5"
            if os.path.exists(lstm_path):
                lstm_model = tf.keras.models.load_model(lstm_path)
            else:
                lstm_model = None
                
            # Load CNN model  
            cnn_path = f"{Config.MODELS_DIR}/{symbol}_{timeframe}_cnn.h5"
            if os.path.exists(cnn_path):
                cnn_model = tf.keras.models.load_model(cnn_path)
            else:
                cnn_model = None
                
            # Load ensemble models
            ensemble_models = {}
            for model_name in ['random_forest', 'gradient_boosting']:
                model_path = f"{Config.MODELS_DIR}/{symbol}_{timeframe}_{model_name}.pkl"
                if os.path.exists(model_path):
                    ensemble_models[model_name] = joblib.load(model_path)
                    
            # Load scaler
            scaler_path = f"{Config.MODELS_DIR}/{symbol}_{timeframe}_scaler.pkl"
            if os.path.exists(scaler_path):
                scaler = joblib.load(scaler_path)
            else:
                scaler = None
                
            if lstm_model or cnn_model or ensemble_models:
                self.models[model_key] = {
                    'lstm': lstm_model,
                    'cnn': cnn_model,
                    'ensemble': ensemble_models,
                    'scaler': scaler,
                    'timestamp': datetime.now()
                }
                
                self.logger.info(f"Model loaded for {symbol} - {timeframe}")
                return True
            else:
                self.logger.warning(f"No model found for {symbol} - {timeframe}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error loading model: {str(e)}")
            return False
