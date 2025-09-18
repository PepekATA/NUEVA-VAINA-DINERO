from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
import joblib
import json
from github import Github
import base64
from datetime import datetime
import os
from typing import List, Dict, Any
import alpaca_trade_api as tradeapi

app = FastAPI(title="Trading Bot API")

# CORS para Streamlit
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuración
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
GITHUB_REPO = os.getenv("GITHUB_REPO")
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")

# Inicializar clientes
github_client = Github(GITHUB_TOKEN)
repo = github_client.get_repo(GITHUB_REPO)

alpaca_api = tradeapi.REST(
    ALPACA_API_KEY,
    ALPACA_SECRET_KEY,
    base_url='https://paper-api.alpaca.markets',
    api_version='v2'
)

# Cache de modelos
models_cache = {}

@app.on_event("startup")
async def startup_event():
    """Cargar modelos al iniciar"""
    load_models_from_github()

def load_models_from_github():
    """Descargar y cachear modelos"""
    try:
        # Obtener lista de modelos
        contents = repo.get_contents("models")
        
        for content in contents:
            if content.name.endswith('.pkl'):
                # Descargar modelo
                file_content = repo.get_contents(content.path)
                decoded = base64.b64decode(file_content.content)
                
                # Guardar temporalmente
                temp_path = f"/tmp/{content.name}"
                with open(temp_path, 'wb') as f:
                    f.write(decoded)
                
                # Cargar en memoria
                model = joblib.load(temp_path)
                models_cache[content.name] = model
                
                # Limpiar archivo temporal
                os.remove(temp_path)
        
        return True
    except Exception as e:
        print(f"Error loading models: {e}")
        return False

@app.get("/")
def read_root():
    return {"status": "Trading Bot API Active", "models_loaded": len(models_cache)}

@app.get("/models/list")
def list_models():
    """Listar modelos disponibles"""
    return {"models": list(models_cache.keys()), "count": len(models_cache)}

@app.post("/predict/{symbol}")
async def predict(symbol: str, timeframe: str = "5Min"):
    """Hacer predicción para un símbolo"""
    try:
        # Buscar modelo apropiado
        model_key = f"{symbol}_{timeframe}_xgboost.pkl"
        
        if model_key not in models_cache:
            # Intentar con cualquier modelo disponible para el símbolo
            for key in models_cache.keys():
                if symbol in key and timeframe in key:
                    model_key = key
                    break
            else:
                raise HTTPException(status_code=404, detail=f"No model found for {symbol}")
        
        model = models_cache[model_key]
        
        # Obtener datos recientes de Alpaca
        bars = alpaca_api.get_bars(
            symbol,
            '1Min' if timeframe == '1Min' else '5Min',
            limit=100
        ).df
        
        if bars.empty:
            raise HTTPException(status_code=404, detail="No data available")
        
        # Preparar features (simplificado)
        features = prepare_features(bars)
        
        # Hacer predicción
        prediction = model.predict_proba(features)[-1]
        
        return {
            "symbol": symbol,
            "timeframe": timeframe,
            "direction": "UP" if prediction[1] > 0.5 else "DOWN",
            "probability": float(prediction[1]),
            "confidence": float(abs(prediction[1] - 0.5) * 2),
            "current_price": float(bars['close'].iloc[-1]),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/account/info")
async def get_account_info():
    """Obtener información de la cuenta"""
    try:
        account = alpaca_api.get_account()
        return {
            "buying_power": float(account.buying_power),
            "cash": float(account.cash),
            "portfolio_value": float(account.portfolio_value),
            "day_trading_buying_power": float(account.daytrading_buying_power),
            "pattern_day_trader": account.pattern_day_trader
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/positions")
async def get_positions():
    """Obtener posiciones actuales"""
    try:
        positions = alpaca_api.list_positions()
        return [
            {
                "symbol": pos.symbol,
                "qty": float(pos.qty),
                "avg_entry_price": float(pos.avg_entry_price),
                "current_price": float(pos.current_price),
                "market_value": float(pos.market_value),
                "unrealized_pl": float(pos.unrealized_pl),
                "unrealized_plpc": float(pos.unrealized_plpc)
            }
            for pos in positions
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/trade/execute")
async def execute_trade(
    symbol: str,
    action: str,
    qty: float,
    order_type: str = "market"
):
    """Ejecutar una operación"""
    try:
        order = alpaca_api.submit_order(
            symbol=symbol,
            qty=qty,
            side=action,
            type=order_type,
            time_in_force='day'
        )
        
        return {
            "order_id": order.id,
            "symbol": order.symbol,
            "qty": order.qty,
            "side": order.side,
            "status": order.status,
            "created_at": order.created_at
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/models/sync")
async def sync_models(background_tasks: BackgroundTasks):
    """Sincronizar modelos desde GitHub"""
    background_tasks.add_task(load_models_from_github)
    return {"message": "Model sync initiated"}

def prepare_features(df):
    """Preparar features para predicción"""
    # Calcular indicadores técnicos básicos
    df['returns'] = df['close'].pct_change()
    df['volume_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
    df['high_low_ratio'] = (df['high'] - df['low']) / df['close']
    
    # Tomar últimas features
    features = df[['returns', 'volume_ratio', 'high_low_ratio']].fillna(0).values
    
    return features.reshape(1, -1)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
