import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (LSTM, GRU, Dense, Dropout, Input, 
                                   Conv1D, MaxPooling1D, Flatten, 
                                   MultiHeadAttention, LayerNormalization)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
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
        self.agents = {}  # Store all agents per symbol
        
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
            # Select technical indicator features
            feature_columns = [col for col in df.columns if col not in 
                             ['open', 'high', 'low', 'close', 'volume', 'trade_count']]
            
            X = df[feature_columns].values
            
            # Create multiple label types for different agents
            labels = {
                'binary': (df['close'].shift(-1) > df['close']).astype(int).values,  # Up/Down
                'multiclass': pd.qcut(df['close'].pct_change().shift(-1), 
                                     q=5, labels=[0,1,2,3,4], duplicates='drop').fillna(2).astype(int).values,  # 5 classes
                'regression': df['close'].shift(-1).values  # Price prediction
            }
            
            # Remove last row (no label)
            X = X[:-1]
            for key in labels:
                labels[key] = labels[key][:-1]
            
            # Create sequences for time series models
            X_seq = []
            y_seq = {key: [] for key in labels}
            
            for i in range(lookback, len(X)):
                X_seq.append(X[i-lookback:i])
                for key in labels:
                    y_seq[key].append(labels[key][i])
                    
            return np.array(X_seq), {key: np.array(val) for key, val in y_seq.items()}
            
        except Exception as e:
            self.logger.error(f"Error preparing features: {str(e)}")
            return None, None
            
    def build_lstm_agent(self, input_shape):
        """Build LSTM agent"""
        model = Sequential([
            LSTM(128, return_sequences=True, input_shape=input_shape),
            Dropout(0.3),
            LSTM(64, return_sequences=True),
            Dropout(0.3),
            LSTM(32),
            Dropout(0.2),
            Dense(16, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model
        
    def build_gru_agent(self, input_shape):
        """Build GRU agent"""
        model = Sequential([
            GRU(128, return_sequences=True, input_shape=input_shape),
            Dropout(0.3),
            GRU(64, return_sequences=True),
            Dropout(0.3),
            GRU(32),
            Dropout(0.2),
            Dense(16, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model
        
    def build_cnn_agent(self, input_shape):
        """Build CNN agent for time series"""
        model = Sequential([
            Conv1D(filters=128, kernel_size=3, activation='relu', input_shape=input_shape),
            MaxPooling1D(pool_size=2),
            Conv1D(filters=64, kernel_size=3, activation='relu'),
            MaxPooling1D(pool_size=2),
            Conv1D(filters=32, kernel_size=3, activation='relu'),
            Flatten(),
            Dense(64, activation='relu'),
            Dropout(0.3),
            Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model
        
    def build_transformer_agent(self, input_shape):
        """Build Transformer agent"""
        inputs = Input(shape=input_shape)
        
        # Multi-head attention
        attention = MultiHeadAttention(
            num_heads=4,
            key_dim=32,
            dropout=0.2
        )(inputs, inputs)
        
        # Add & Norm
        attention = LayerNormalization(epsilon=1e-6)(inputs + attention)
        
        # Feed Forward
        ff = Dense(128, activation='relu')(attention)
        ff = Dense(input_shape[-1])(ff)
        ff = Dropout(0.2)(ff)
        
        # Add & Norm
        ff_out = LayerNormalization(epsilon=1e-6)(attention + ff)
        
        # Global pooling and output
        pooled = tf.keras.layers.GlobalAveragePooling1D()(ff_out)
        outputs = Dense(32, activation='relu')(pooled)
        outputs = Dense(1, activation='sigmoid')(outputs)
        
        model = Model(inputs=inputs, outputs=outputs)
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model
        
    def train_ensemble_agents(self, X_train, y_train, X_test, y_test):
        """Train ensemble machine learning agents"""
        agents = {}
        
        # Flatten data for traditional ML models
        X_train_flat = X_train.reshape(X_train.shape[0], -1)
        X_test_flat = X_test.reshape(X_test.shape[0], -1)
        
        # Random Forest Agent
        rf_agent = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            random_state=42,
            n_jobs=-1
        )
        rf_agent.fit(X_train_flat, y_train)
        agents['random_forest'] = rf_agent
        
        # XGBoost Agent
        xgb_agent = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=10,
            learning_rate=0.1,
            objective='binary:logistic',
            use_label_encoder=False,
            eval_metric='logloss',
            random_state=42
        )
        xgb_agent.fit(X_train_flat, y_train)
        agents['xgboost'] = xgb_agent
        
        # LightGBM Agent
        lgb_agent = lgb.LGBMClassifier(
            n_estimators=200,
            max_depth=10,
            learning_rate=0.1,
            num_leaves=31,
            random_state=42
        )
        lgb_agent.fit(X_train_flat, y_train)
        agents['lightgbm'] = lgb_agent
        
        # CatBoost Agent
        catboost_agent = CatBoostClassifier(
            iterations=200,
            depth=8,
            learning_rate=0.1,
            loss_function='Logloss',
            verbose=False,
            random_state=42
        )
        catboost_agent.fit(X_train_flat, y_train)
        agents['catboost'] = catboost_agent
        
        # Evaluate all agents
        for name, agent in agents.items():
            predictions = agent.predict(X_test_flat)
            accuracy = accuracy_score(y_test, predictions)
            self.logger.info(f"{name} Agent Accuracy: {accuracy:.4f}")
            
        return agents
        
    def train_all_agents_for_symbol(self, symbol, timeframe='1Min', epochs=50, batch_size=32):
        """Train all agents for a specific symbol"""
        try:
            self.logger.info(f"Training {Config.AGENTS_PER_SYMBOL} agents for {symbol} - {timeframe}")
            
            # Fetch data
            df = self.data_fetcher.fetch_realtime_data(symbol, timeframe, limit=2000)
            
            if df.empty:
                self.logger.error(f"No data available for {symbol}")
                return None
                
            # Prepare features
            X, y_dict = self.prepare_features(df, lookback=Config.LOOKBACK_PERIOD)
            
            if X is None or len(X) == 0:
                self.logger.error(f"Failed to prepare features for {symbol}")
                return None
                
            # Use binary labels for main training
            y = y_dict['binary']
            
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
            
            agents = {}
            
            # 1. Train LSTM Agent
            self.logger.info(f"Training LSTM agent for {symbol}")
            lstm_agent = self.build_lstm_agent((X_train.shape[1], X_train.shape[2]))
            
            early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
            checkpoint = ModelCheckpoint(
                f"{Config.MODELS_DIR}/{symbol}_{timeframe}_lstm.h5",
                save_best_only=True,
                monitor='val_accuracy',
                mode='max'
            )
            
            lstm_agent.fit(
                X_train_scaled, y_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=(X_test_scaled, y_test),
                callbacks=[early_stop, checkpoint],
                verbose=0
            )
            agents['lstm'] = lstm_agent
            
            # 2. Train GRU Agent
            self.logger.info(f"Training GRU agent for {symbol}")
            gru_agent = self.build_gru_agent((X_train.shape[1], X_train.shape[2]))
            
            checkpoint_gru = ModelCheckpoint(
                f"{Config.MODELS_DIR}/{symbol}_{timeframe}_gru.h5",
                save_best_only=True,
                monitor='val_accuracy',
                mode='max'
            )
            
            gru_agent.fit(
                X_train_scaled, y_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=(X_test_scaled, y_test),
                callbacks=[early_stop, checkpoint_gru],
                verbose=0
            )
            agents['gru'] = gru_agent
            
            # 3. Train CNN Agent
            self.logger.info(f"Training CNN agent for {symbol}")
            cnn_agent = self.build_cnn_agent((X_train.shape[1], X_train.shape[2]))
            
            checkpoint_cnn = ModelCheckpoint(
                f"{Config.MODELS_DIR}/{symbol}_{timeframe}_cnn.h5",
                save_best_only=True,
                monitor='val_accuracy',
                mode='max'
            )
            
            cnn_agent.fit(
                X_train_scaled, y_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=(X_test_scaled, y_test),
                callbacks=[early_stop, checkpoint_cnn],
                verbose=0
            )
            agents['cnn'] = cnn_agent
            
            # 4. Train Transformer Agent
            self.logger.info(f"Training Transformer agent for {symbol}")
            transformer_agent = self.build_transformer_agent((X_train.shape[1], X_train.shape[2]))
            
            checkpoint_transformer = ModelCheckpoint(
                f"{Config.MODELS_DIR}/{symbol}_{timeframe}_transformer.h5",
                save_best_only=True,
                monitor='val_accuracy',
                mode='max'
            )
            
            transformer_agent.fit(
                X_train_scaled, y_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=(X_test_scaled, y_test),
                callbacks=[early_stop, checkpoint_transformer],
                verbose=0
            )
            agents['transformer'] = transformer_agent
            
            # 5-8. Train Ensemble ML Agents
            self.logger.info(f"Training ensemble agents for {symbol}")
            ensemble_agents = self.train_ensemble_agents(
                X_train_scaled, y_train, X_test_scaled, y_test
            )
            agents.update(ensemble_agents)
            
            # Save all agents and scaler
            agent_key = f"{symbol}_{timeframe}"
            self.agents[agent_key] = agents
            self.scalers[agent_key] = scaler
            
            # Save ensemble models
            for name, agent in ensemble_agents.items():
                joblib.dump(agent, f"{Config.MODELS_DIR}/{symbol}_{timeframe}_{name}.pkl")
                
            # Save scaler
            joblib.dump(scaler, f"{Config.MODELS_DIR}/{symbol}_{timeframe}_scaler.pkl")
            
            # Calculate ensemble accuracy
            predictions = []
            for name, agent in agents.items():
                if name in ['lstm', 'gru', 'cnn', 'transformer']:
                    pred = agent.predict(X_test_scaled)
                else:
                    pred = agent.predict_proba(X_test_scaled.reshape(X_test_scaled.shape[0], -1))[:, 1:2]
                predictions.append(pred)
                
            ensemble_pred = np.mean(predictions, axis=0) > 0.5
            ensemble_accuracy = accuracy_score(y_test, ensemble_pred)
            
            self.logger.info(f"Ensemble accuracy for {symbol}: {ensemble_accuracy:.4f}")
            
            return {
                'agents': agents,
                'scaler': scaler,
                'accuracy': ensemble_accuracy,
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            self.logger.error(f"Error training agents for {symbol}: {str(e)}")
            return None
            
    def train_all_symbols(self, symbols=None):
        """Train models for all symbols"""
        if symbols is None:
            symbols = Config.SYMBOLS
            
        results = {}
        total_symbols = len(symbols)
        
        for i, symbol in enumerate(symbols, 1):
            self.logger.info(f"Training {symbol} ({i}/{total_symbols})")
            
            for timeframe in ['1Min', '5Min', '15Min']:  # Focus on short timeframes for scalping
                result = self.train_all_agents_for_symbol(symbol, timeframe)
                results[f"{symbol}_{timeframe}"] = result is not None
                
            # Save progress periodically
            if i % 10 == 0:
                self.save_training_progress(results)
                
        return results
        
    def save_training_progress(self, results):
        """Save training progress to file"""
        try:
            import json
            progress_file = f"{Config.DATA_DIR}/training_progress_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(progress_file, 'w') as f:
                json.dump(results, f, indent=4, default=str)
            self.logger.info(f"Training progress saved to {progress_file}")
        except Exception as e:
            self.logger.error(f"Error saving progress: {str(e)}")
