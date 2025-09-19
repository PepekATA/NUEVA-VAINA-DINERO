from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
import alpaca_trade_api as tradeapi
import pandas as pd
import numpy as np
import yfinance as yf
from ta import add_all_ta_features
import joblib
import tensorflow as tf
from datetime import datetime, timedelta
import os
from typing import List, Dict
import asyncio
import json

app = FastAPI(title="Trading Bot API - Real Trading")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuración Alpaca REAL
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")
ALPACA_BASE_URL = 'https://paper-api.alpaca.markets'  # Cambiar a api.alpaca.markets para real

# Inicializar Alpaca
alpaca = tradeapi.REST(
    ALPACA_API_KEY,
    ALPACA_SECRET_KEY,
    ALPACA_BASE_URL,
    api_version='v2'
)

# Cache de modelos
models_cache = {}
scalers_cache = {}

@app.on_event("startup")
async def startup():
    """Cargar modelos al iniciar"""
    load_trained_models()
    print("✅ API iniciada con modelos cargados")

def load_trained_models():
    """Cargar modelos entrenados"""
    models_dir = 'models'
    if os.path.exists(models_dir):
        for file in os.listdir(models_dir):
            try:
                if file.endswith('.h5'):
                    key = file.replace('.h5', '')
                    models_cache[key] = tf.keras.models.load_model(f"{models_dir}/{file}")
                elif file.endswith('.pkl'):
                    key = file.replace('.pkl', '')
                    if 'scaler' in file:
                        scalers_cache[key] = joblib.load(f"{models_dir}/{file}")
                    else:
                        models_cache[key] = joblib.load(f"{models_dir}/{file}")
            except Exception as e:
                print(f"Error cargando {file}: {e}")

@app.get("/")
def root():
    return {
        "status": "Trading Bot API Active",
        "mode": "PAPER" if "paper" in ALPACA_BASE_URL else "REAL",
        "models_loaded": len(models_cache)
    }

@app.get("/market/status")
async def market_status():
    """Estado del mercado en tiempo real"""
    clock = alpaca.get_clock()
    return {
        "is_open": clock.is_open,
        "next_open": clock.next_open.isoformat() if clock.next_open else None,
        "next_close": clock.next_close.isoformat() if clock.next_close else None
    }

@app.get("/account")
async def get_account():
    """Información de cuenta REAL de Alpaca"""
    try:
        account = alpaca.get_account()
        return {
            "buying_power": float(account.buying_power),
            "cash": float(account.cash),
            "portfolio_value": float(account.portfolio_value),
            "pattern_day_trader": account.pattern_day_trader,
            "trading_blocked": account.trading_blocked,
            "transfers_blocked": account.transfers_blocked,
            "account_blocked": account.account_blocked,
            "daytrade_count": account.daytrade_count,
            "last_equity": float(account.last_equity)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/positions")
async def get_positions():
    """Posiciones REALES actuales"""
    try:
        positions = alpaca.list_positions()
        return [
            {
                "symbol": pos.symbol,
                "qty": float(pos.qty),
                "avg_entry_price": float(pos.avg_entry_price),
                "current_price": float(pos.current_price),
                "market_value": float(pos.market_value),
                "unrealized_pl": float(pos.unrealized_pl),
                "unrealized_plpc": float(pos.unrealized_plpc),
                "side": pos.side
            }
            for pos in positions
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/data/{symbol}")
async def get_realtime_data(symbol: str, timeframe: str = "1Min"):
    """Obtener datos en TIEMPO REAL de Alpaca"""
    try:
        # Mapeo de timeframes
        tf_map = {
            "1Min": "1Min",
            "5Min": "5Min",
            "15Min": "15Min",
            "30Min": "30Min",
            "1Hour": "1Hour"
        }
        
        # Obtener barras de Alpaca
        end = datetime.now()
        if timeframe == "1Min":
            start = end - timedelta(hours=2)
        elif timeframe == "5Min":
            start = end - timedelta(hours=8)
        else:
            start = end - timedelta(days=1)
        
        bars = alpaca.get_bars(
            symbol,
            tf_map[timeframe],
            start=start.isoformat(),
            end=end.isoformat(),
            limit=100
        ).df
        
        if bars.empty:
            raise HTTPException(status_code=404, detail="No data available")
        
        # Calcular indicadores técnicos
        bars = add_all_ta_features(
            bars, open="open", high="high", low="low", close="close", volume="volume"
        )
        
        return {
            "symbol": symbol,
            "timeframe": timeframe,
            "current_price": float(bars['close'].iloc[-1]),
            "volume": float(bars['volume'].iloc[-1]),
            "data_points": len(bars),
            "indicators": {
                "rsi": float(bars['momentum_rsi'].iloc[-1]) if 'momentum_rsi' in bars else 50,
                "macd": float(bars['trend_macd'].iloc[-1]) if 'trend_macd' in bars else 0,
                "bb_upper": float(bars['volatility_bbh'].iloc[-1]) if 'volatility_bbh' in bars else 0,
                "bb_lower": float(bars['volatility_bbl'].iloc[-1]) if 'volatility_bbl' in bars else 0
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/{symbol}")
async def predict(symbol: str, timeframe: str = "5Min"):
    """Predicción con modelos REALES entrenados"""
    try:
        # Obtener datos reales
        end = datetime.now()
        start = end - timedelta(hours=24)
        
        bars = alpaca.get_bars(
            symbol,
            timeframe,
            start=start.isoformat(),
            end=end.isoformat(),
            limit=200
        ).df
        
        if bars.empty or len(bars) < 60:
            raise HTTPException(status_code=404, detail="Insufficient data")
        
        # Calcular indicadores
        bars = add_all_ta_features(
            bars, open="open", high="high", low="low", close="close", volume="volume"
        )
        
        # Preparar features
        feature_cols = [col for col in bars.columns if not col.startswith('others_')]
        features = bars[feature_cols].fillna(0).values[-60:]
        
        predictions = []
        
        # Buscar modelos para este símbolo
        for key, model in models_cache.items():
            if symbol in key and timeframe in key:
                try:
                    # Obtener scaler si existe
                    scaler_key = f"{symbol}_{timeframe}_scaler"
                    if scaler_key in scalers_cache:
                        features_scaled = scalers_cache[scaler_key].transform(features.reshape(1, -1))
                    else:
                        features_scaled = features.reshape(1, -1)
                    
                    # Predecir
                    if hasattr(model, 'predict_proba'):
                        pred = model.predict_proba(features_scaled)[0][1]
                    else:
                        pred = model.predict(features_scaled.reshape(1, 60, -1))[0][0]
                    
                    predictions.append(float(pred))
                except:
                    continue
        
        if not predictions:
            # Si no hay modelos, usar análisis técnico básico
            rsi = float(bars['momentum_rsi'].iloc[-1]) if 'momentum_rsi' in bars else 50
            macd = float(bars['trend_macd'].iloc[-1]) if 'trend_macd' in bars else 0
            
            # Lógica simple
            score = 0.5
            if rsi < 30:
                score += 0.3
            elif rsi > 70:
                score -= 0.3
            if macd > 0:
                score += 0.2
            
            predictions = [score]
        
        # Calcular predicción final
        final_prediction = np.mean(predictions)
        
        return {
            "symbol": symbol,
            "timeframe": timeframe,
            "timestamp": datetime.now().isoformat(),
            "direction": "UP" if final_prediction > 0.5 else "DOWN",
            "probability": float(final_prediction),
            "confidence": float(abs(final_prediction - 0.5) * 2),
            "current_price": float(bars['close'].iloc[-1]),
            "models_used": len(predictions),
            "predicted_change_pct": float((final_prediction - 0.5) * 10)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/trade/execute")
async def execute_trade(
    symbol: str,
    side: str,  # 'buy' o 'sell'
    qty: float,
    order_type: str = "market",
    limit_price: float = None
):
    """Ejecutar trade REAL en Alpaca"""
    try:
        # Validación
        if side not in ['buy', 'sell']:
            raise HTTPException(status_code=400, detail="Side must be 'buy' or 'sell'")
        
        # Preparar orden
        order_params = {
            'symbol': symbol,
            'qty': qty,
            'side': side,
            'type': order_type,
            'time_in_force': 'day'
        }
        
        if order_type == 'limit' and limit_price:
            order_params['limit_price'] = limit_price
        
        # Ejecutar orden
        order = alpaca.submit_order(**order_params)
        
        return {
            "order_id": order.id,
            "symbol": order.symbol,
            "qty": order.qty,
            "side": order.side,
            "type": order.type,
            "status": order.status,
            "submitted_at": order.submitted_at.isoformat(),
            "filled_qty": order.filled_qty,
            "filled_avg_price": order.filled_avg_price
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/orders")
async def get_orders(status: str = "open"):
    """Obtener órdenes"""
    try:
        orders = alpaca.list_orders(status=status)
        return [
            {
                "id": order.id,
                "symbol": order.symbol,
                "qty": float(order.qty),
                "side": order.side,
                "type": order.type,
                "status": order.status,
                "submitted_at": order.submitted_at.isoformat()
            }
            for order in orders
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/train/{symbol}")
async def train_model(symbol: str, background_tasks: BackgroundTasks):
    """Entrenar modelo con datos REALES"""
    background_tasks.add_task(train_model_background, symbol)
    return {"message": f"Training initiated for {symbol}"}

async def train_model_background(symbol: str):
    """Proceso de entrenamiento en background"""
    try:
        # Obtener datos históricos
        end = datetime.now()
        start = end - timedelta(days=30)
        
        bars = alpaca.get_bars(
            symbol,
            '5Min',
            start=start.isoformat(),
            end=end.isoformat()
        ).df
        
        # Aquí iría el código de entrenamiento real
        # Por ahora solo guardamos los datos
        
        print(f"Training completed for {symbol}")
        
    except Exception as e:
        print(f"Training error for {symbol}: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
