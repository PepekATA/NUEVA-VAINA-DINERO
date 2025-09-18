# train_model.py - Versión completa con auto-sincronización

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (LSTM, GRU, Dense, Dropout, Input, 
                                   Conv1D, MaxPooling1D, Flatten, 
                                   MultiHeadAttention, LayerNormalization)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
import joblib
import os
import json
from datetime import datetime
import logging
import gc
import warnings
warnings.filterwarnings('ignore')

from config import Config
from data_fetcher import DataFetcher

class ModelTrainer:
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.data_fetcher = DataFetcher()
        self.setup_logging()
        self.agents = {}
        self.training_history = {}
        self.sync_enabled = True  # Flag para activar/desactivar sincronización
        
    def setup_logging(self):
        os.makedirs(Config.LOGS_DIR, exist_ok=True)
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
                'binary': (df['close'].shift(-1) > df['close']).astype(int).values,
                'multiclass': pd.qcut(df['close'].pct_change().shift(-1), 
                                     q=5, labels=[0,1,2,3,4], duplicates='drop').fillna(2).astype(int).values,
                'regression': df['close'].shift(-1).values
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
        """Build LSTM agent with improved architecture"""
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
            metrics=['accuracy', tf.keras.metrics.AUC()]
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
            metrics=['accuracy', tf.keras.metrics.AUC()]
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
            metrics=['accuracy', tf.keras.metrics.AUC()]
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
            metrics=['accuracy', tf.keras.metrics.AUC()]
        )
        
        return model
        
    def train_ensemble_agents(self, X_train, y_train, X_test, y_test):
        """Train ensemble machine learning agents"""
        agents = {}
        
        # Flatten data for traditional ML models
        X_train_flat = X_train.reshape(X_train.shape[0], -1)
        X_test_flat = X_test.reshape(X_test.shape[0], -1)
        
        # Random Forest Agent
        self.logger.info("Training Random Forest agent...")
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
        self.logger.info("Training XGBoost agent...")
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
        self.logger.info("Training LightGBM agent...")
        lgb_agent = lgb.LGBMClassifier(
            n_estimators=200,
            max_depth=10,
            learning_rate=0.1,
            num_leaves=31,
            random_state=42,
            verbosity=-1
        )
        lgb_agent.fit(X_train_flat, y_train)
        agents['lightgbm'] = lgb_agent
        
        # CatBoost Agent
        self.logger.info("Training CatBoost agent...")
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
        """Train all agents for a specific symbol with auto-sync"""
        try:
            self.logger.info(f"="*60)
            self.logger.info(f"Training {Config.AGENTS_PER_SYMBOL} agents for {symbol} - {timeframe}")
            self.logger.info(f"="*60)
            
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
            agent_metrics = {}
            
            # Callbacks para deep learning
            early_stop = EarlyStopping(
                monitor='val_loss', 
                patience=10, 
                restore_best_weights=True
            )
            
            reduce_lr = ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=0.00001
            )
            
            # 1. Train LSTM Agent
            self.logger.info(f"[1/{Config.AGENTS_PER_SYMBOL}] Training LSTM agent...")
            lstm_agent = self.build_lstm_agent((X_train.shape[1], X_train.shape[2]))
            
            checkpoint_lstm = ModelCheckpoint(
                f"{Config.MODELS_DIR}/{symbol}_{timeframe}_lstm.h5",
                save_best_only=True,
                monitor='val_accuracy',
                mode='max'
            )
            
            history_lstm = lstm_agent.fit(
                X_train_scaled, y_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=(X_test_scaled, y_test),
                callbacks=[early_stop, reduce_lr, checkpoint_lstm],
                verbose=0
            )
            
            lstm_pred = (lstm_agent.predict(X_test_scaled) > 0.5).astype(int).flatten()
            agent_metrics['lstm'] = {
                'accuracy': accuracy_score(y_test, lstm_pred),
                'history': history_lstm.history
            }
            agents['lstm'] = lstm_agent
            self.logger.info(f"LSTM Accuracy: {agent_metrics['lstm']['accuracy']:.4f}")
            
            # 2. Train GRU Agent
            self.logger.info(f"[2/{Config.AGENTS_PER_SYMBOL}] Training GRU agent...")
            gru_agent = self.build_gru_agent((X_train.shape[1], X_train.shape[2]))
            
            checkpoint_gru = ModelCheckpoint(
                f"{Config.MODELS_DIR}/{symbol}_{timeframe}_gru.h5",
                save_best_only=True,
                monitor='val_accuracy',
                mode='max'
            )
            
            history_gru = gru_agent.fit(
                X_train_scaled, y_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=(X_test_scaled, y_test),
                callbacks=[early_stop, reduce_lr, checkpoint_gru],
                verbose=0
            )
            
            gru_pred = (gru_agent.predict(X_test_scaled) > 0.5).astype(int).flatten()
            agent_metrics['gru'] = {
                'accuracy': accuracy_score(y_test, gru_pred),
                'history': history_gru.history
            }
            agents['gru'] = gru_agent
            self.logger.info(f"GRU Accuracy: {agent_metrics['gru']['accuracy']:.4f}")
            
            # 3. Train CNN Agent
            self.logger.info(f"[3/{Config.AGENTS_PER_SYMBOL}] Training CNN agent...")
            cnn_agent = self.build_cnn_agent((X_train.shape[1], X_train.shape[2]))
            
            checkpoint_cnn = ModelCheckpoint(
                f"{Config.MODELS_DIR}/{symbol}_{timeframe}_cnn.h5",
                save_best_only=True,
                monitor='val_accuracy',
                mode='max'
            )
            
            history_cnn = cnn_agent.fit(
                X_train_scaled, y_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=(X_test_scaled, y_test),
                callbacks=[early_stop, reduce_lr, checkpoint_cnn],
                verbose=0
            )
            
            cnn_pred = (cnn_agent.predict(X_test_scaled) > 0.5).astype(int).flatten()
            agent_metrics['cnn'] = {
                'accuracy': accuracy_score(y_test, cnn_pred),
                'history': history_cnn.history
            }
            agents['cnn'] = cnn_agent
            self.logger.info(f"CNN Accuracy: {agent_metrics['cnn']['accuracy']:.4f}")
            
            # 4. Train Transformer Agent
            if Config.AGENTS_PER_SYMBOL >= 4:
                self.logger.info(f"[4/{Config.AGENTS_PER_SYMBOL}] Training Transformer agent...")
                transformer_agent = self.build_transformer_agent((X_train.shape[1], X_train.shape[2]))
                
                checkpoint_transformer = ModelCheckpoint(
                    f"{Config.MODELS_DIR}/{symbol}_{timeframe}_transformer.h5",
                    save_best_only=True,
                    monitor='val_accuracy',
                    mode='max'
                )
                
                history_transformer = transformer_agent.fit(
                    X_train_scaled, y_train,
                    epochs=epochs,
                    batch_size=batch_size,
                    validation_data=(X_test_scaled, y_test),
                    callbacks=[early_stop, reduce_lr, checkpoint_transformer],
                    verbose=0
                )
                
                transformer_pred = (transformer_agent.predict(X_test_scaled) > 0.5).astype(int).flatten()
                agent_metrics['transformer'] = {
                    'accuracy': accuracy_score(y_test, transformer_pred),
                    'history': history_transformer.history
                }
                agents['transformer'] = transformer_agent
                self.logger.info(f"Transformer Accuracy: {agent_metrics['transformer']['accuracy']:.4f}")
            
            # 5-8. Train Ensemble ML Agents
            self.logger.info(f"[5-8/{Config.AGENTS_PER_SYMBOL}] Training ensemble agents...")
            ensemble_agents = self.train_ensemble_agents(
                X_train_scaled, y_train, X_test_scaled, y_test
            )
            agents.update(ensemble_agents)
            
            # Evaluar ensemble ML agents
            X_test_flat = X_test_scaled.reshape(X_test_scaled.shape[0], -1)
            for name, agent in ensemble_agents.items():
                pred = agent.predict(X_test_flat)
                agent_metrics[name] = {
                    'accuracy': accuracy_score(y_test, pred)
                }
            
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
                    pred = (agent.predict(X_test_scaled) > 0.5).astype(int).flatten()
                else:
                    pred = agent.predict(X_test_scaled.reshape(X_test_scaled.shape[0], -1))
                predictions.append(pred)
                
            ensemble_pred = (np.mean(predictions, axis=0) > 0.5).astype(int)
            ensemble_accuracy = accuracy_score(y_test, ensemble_pred)
            
            self.logger.info(f"="*60)
            self.logger.info(f"ENSEMBLE ACCURACY for {symbol}: {ensemble_accuracy:.4f}")
            self.logger.info(f"="*60)
            
            # Guardar métricas
            training_result = {
                'symbol': symbol,
                'timeframe': timeframe,
                'timestamp': datetime.now().isoformat(),
                'ensemble_accuracy': ensemble_accuracy,
                'agent_metrics': agent_metrics,
                'total_agents': len(agents),
                'data_points': len(X),
                'train_size': len(X_train),
                'test_size': len(X_test)
            }
            
            # Guardar en historial
            self.training_history[f"{symbol}_{timeframe}"] = training_result
            
            # Guardar métricas en archivo
            self.save_training_metrics(training_result)
            
            # AUTO-SINCRONIZACIÓN
            if self.sync_enabled and ensemble_accuracy > 0:
                self.auto_sync_models(symbol, timeframe, training_result)
            
            # Limpiar memoria
            self.cleanup_memory()
            
            return training_result
            
        except Exception as e:
            self.logger.error(f"Error training agents for {symbol}: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return None
    
    def auto_sync_models(self, symbol, timeframe, training_result):
        """Auto-sincronizar modelos con GitHub y Google Drive"""
        try:
            self.logger.info("🔄 Iniciando auto-sincronización...")
            
            # Importar el manager de sincronización
            try:
                from model_sync import sync_manager
                
                # Sincronizar con GitHub
                self.logger.info("📤 Sincronizando con GitHub...")
                github_success = sync_manager.sync_to_github()
                
                # Sincronizar con Google Drive
                self.logger.info("📤 Sincronizando con Google Drive...")
                drive_success = sync_manager.sync_to_gdrive()
                
                # Actualizar registro de modelos
                sync_manager.update_model_registry()
                
                if github_success and drive_success:
                    self.logger.info(f"✅ Modelos de {symbol}-{timeframe} sincronizados exitosamente")
                    
                    # Guardar log de sincronización
                    sync_log = {
                        'timestamp': datetime.now().isoformat(),
                        'symbol': symbol,
                        'timeframe': timeframe,
                        'accuracy': training_result['ensemble_accuracy'],
                        'github': github_success,
                        'gdrive': drive_success
                    }
                    
                    log_file = f"{Config.LOGS_DIR}/sync_log.json"
                    if os.path.exists(log_file):
                        with open(log_file, 'r') as f:
                            logs = json.load(f)
                    else:
                        logs = []
                    
                    logs.append(sync_log)
                    
                    with open(log_file, 'w') as f:
                        json.dump(logs, f, indent=2)
                        
                elif github_success:
                    self.logger.warning("⚠️ Solo GitHub sincronizado")
                elif drive_success:
                    self.logger.warning("⚠️ Solo Google Drive sincronizado")
                else:
                    self.logger.error("❌ Sincronización fallida")
                    
            except ImportError:
                self.logger.warning("⚠️ Módulo model_sync no disponible, sincronización manual requerida")
                
        except Exception as e:
            self.logger.error(f"Error en auto-sincronización: {str(e)}")
    
    def save_training_metrics(self, result):
        """Guardar métricas de entrenamiento"""
        try:
            metrics_file = f"{Config.DATA_DIR}/training_metrics.json"
            
            if os.path.exists(metrics_file):
                with open(metrics_file, 'r') as f:
                    metrics = json.load(f)
            else:
                metrics = []
            
            metrics.append(result)
            
            # Mantener solo últimas 100 entradas
            metrics = metrics[-100:]
            
            with open(metrics_file, 'w') as f:
                json.dump(metrics, f, indent=2, default=str)
                
            self.logger.info(f"Métricas guardadas en {metrics_file}")
            
        except Exception as e:
            self.logger.error(f"Error guardando métricas: {str(e)}")
    
    def cleanup_memory(self):
        """Limpiar memoria después del entrenamiento"""
        try:
            # Limpiar sesión de TensorFlow
            tf.keras.backend.clear_session()
            
            # Garbage collection
            gc.collect()
            
            self.logger.info("Memoria limpiada")
            
        except Exception as e:
            self.logger.error(f"Error limpiando memoria: {str(e)}")
    
    def train_all_symbols(self, symbols=None):
        """Train models for all symbols"""
        if symbols is None:
            symbols = Config.SYMBOLS
            
        results = {}
        total_symbols = len(symbols)
        successful_trainings = 0
        failed_trainings = 0
        
        self.logger.info(f"🚀 Iniciando entrenamiento de {total_symbols} símbolos")
        
        for i, symbol in enumerate(symbols, 1):
            self.logger.info(f"\n[{i}/{total_symbols}] Procesando {symbol}...")
            
            for timeframe in Config.TIMEFRAMES[:3]:  # Focus on short timeframes
                try:
                    result = self.train_all_agents_for_symbol(symbol, timeframe)
                    
                    if result:
                        results[f"{symbol}_{timeframe}"] = result
                        successful_trainings += 1
                        self.logger.info(f"✅ {symbol}-{timeframe} completado")
                    else:
                        failed_trainings += 1
                        self.logger.error(f"❌ {symbol}-{timeframe} falló")
                        
                except Exception as e:
                    self.logger.error(f"❌ Error con {symbol}-{timeframe}: {str(e)}")
                    failed_trainings += 1
                    
            # Sincronización periódica cada 5 símbolos
            if i % 5 == 0:
                self.logger.info(f"📊 Progreso: {i}/{total_symbols} símbolos procesados")
                if self.sync_enabled:
                    self.auto_sync_models(symbol, timeframe, results)
                    
        # Resumen final
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"📊 RESUMEN DE ENTRENAMIENTO")
        self.logger.info(f"{'='*60}")
        self.logger.info(f"✅ Exitosos: {successful_trainings}")
        self.logger.info(f"❌ Fallidos: {failed_trainings}")
        self.logger.info(f"📊 Total: {successful_trainings + failed_trainings}")
        
        # Sincronización final
        if self.sync_enabled:
            self.logger.info("🔄 Sincronización final...")
            self.auto_sync_models("final", "batch", results)
        
        return results
    
    def load_model(self, symbol, timeframe):
        """Load a trained model"""
        try:
            model_key = f"{symbol}_{timeframe}"
            
            if model_key in self.models:
                return self.models[model_key]
            
            models = {}
            
            # Load deep learning models
            for model_type in ['lstm', 'gru', 'cnn', 'transformer']:
                model_path = f"{Config.MODELS_DIR}/{symbol}_{timeframe}_{model_type}.h5"
                if os.path.exists(model_path):
                    models[model_type] = tf.keras.models.load_model(model_path)
                    self.logger.info(f"Loaded {model_type} model for {symbol}-{timeframe}")
            
            # Load ML models
            for model_type in ['random_forest', 'xgboost', 'lightgbm', 'catboost']:
                model_path = f"{Config.MODELS_DIR}/{symbol}_{timeframe}_{model_type}.pkl"
                if os.path.exists(model_path):
                    models[model_type] = joblib.load(model_path)
                    self.logger.info(f"Loaded {model_type} model for {symbol}-{timeframe}")
            
            # Load scaler
            scaler_path = f"{Config.MODELS_DIR}/{symbol}_{timeframe}_scaler.pkl"
            if os.path.exists(scaler_path):
                scaler = joblib.load(scaler_path)
                self.logger.info(f"Loaded scaler for {symbol}-{timeframe}")
            else:
                scaler = None
            
            self.models[model_key] = {
                'models': models,
                'scaler': scaler
            }
            
            return self.models[model_key]
            
        except Exception as e:
            self.logger.error(f"Error loading model: {str(e)}")
            return None
    
    def continuous_training(self, interval_minutes=60):
        """Continuous training loop with auto-sync"""
        import time
        import schedule
        
        def training_job():
            self.logger.info("🔄 Iniciando ciclo de entrenamiento continuo...")
            
            # Seleccionar símbolos para entrenar
            symbols_to_train = Config.SYMBOLS[:10]  # Top 10 símbolos
            
            for symbol in symbols_to_train:
                for timeframe in ['1Min', '5Min']:
                    try:
                        result = self.train_all_agents_for_symbol(
                            symbol, 
                            timeframe, 
                            epochs=30  # Menos epochs para entrenamiento rápido
                        )
                        
                        if result:
                            self.logger.info(f"✅ {symbol}-{timeframe} actualizado")
                            
                    except Exception as e:
                        self.logger.error(f"❌ Error actualizando {symbol}: {str(e)}")
            
            self.logger.info("✅ Ciclo de entrenamiento completado")
        
        # Programar tarea
        schedule.every(interval_minutes).minutes.do(training_job)
        
        # Ejecutar primera vez
        training_job()
        
        # Loop principal
        self.logger.info(f"⏰ Entrenamiento continuo activado cada {interval_minutes} minutos")
        
        while True:
            try:
                schedule.run_pending()
                time.sleep(60)
            except KeyboardInterrupt:
                self.logger.info("⏹️ Entrenamiento continuo detenido")
                break
            except Exception as e:
                self.logger.error(f"Error en loop continuo: {str(e)}")
                time.sleep(60)

# Función de utilidad para entrenamiento rápido
def quick_train(symbol, timeframe='5Min'):
    """Función helper para entrenamiento rápido de un símbolo"""
    trainer = ModelTrainer()
    result = trainer.train_all_agents_for_symbol(symbol, timeframe, epochs=20)
    return result

if __name__ == "__main__":
    # Ejemplo de uso
    trainer = ModelTrainer()
    
    # Entrenar un símbolo específico
    result = trainer.train_all_agents_for_symbol('AAPL', '5Min')
    
    if result:
        print(f"✅ Entrenamiento completado con precisión: {result['ensemble_accuracy']:.4f}")
    else:
        print("❌ Entrenamiento falló")
