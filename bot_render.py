"""
BOT DE TRADING 24/7 PARA RENDER.COM
Optimizado para scalping de alta velocidad con Alpaca
"""

import os
import sys
import time
import json
import logging
import asyncio
import aiohttp
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import alpaca_trade_api as tradeapi
from concurrent.futures import ThreadPoolExecutor
import warnings
warnings.filterwarnings('ignore')

# Configuración optimizada para Render
class ScalpingBot:
    def __init__(self):
        # Verificar credenciales
        self.api_key = os.environ.get('ALPACA_API_KEY')
        self.secret_key = os.environ.get('ALPACA_SECRET_KEY')
        self.use_paper = os.environ.get('USE_PAPER', 'true').lower() == 'true'
        
        if not self.api_key or not self.secret_key:
            raise ValueError("❌ Credenciales de Alpaca no encontradas")
        
        # Inicializar API
        self.base_url = 'https://paper-api.alpaca.markets' if self.use_paper else 'https://api.alpaca.markets'
        self.api = tradeapi.REST(
            self.api_key,
            self.secret_key,
            self.base_url,
            api_version='v2'
        )
        
        # WebSocket para datos en tiempo real
        self.ws_client = tradeapi.Stream(
            self.api_key,
            self.secret_key,
            base_url=self.base_url,
            data_feed='iex'  # IEX para menor latencia
        )
        
        # Estado del bot
        self.is_running = True
        self.positions = {}
        self.pending_orders = {}
        self.models = {}
        self.last_sync = datetime.now()
        
        # Configuración de scalping
        self.scalping_config = {
            'max_position_size': 0.05,  # 5% máximo por operación
            'profit_target': 0.003,      # 0.3% de ganancia objetivo
            'stop_loss': 0.002,          # 0.2% de pérdida máxima
            'max_hold_time': 300,        # 5 minutos máximo
            'min_volume': 1000000,       # Volumen mínimo para entrar
            'max_spread': 0.001,         # Spread máximo 0.1%
            'positions_limit': 5,        # Máximo 5 posiciones simultáneas
            'cooldown_period': 60        # 60 segundos entre trades del mismo símbolo
        }
        
        # Símbolos para scalping (alta liquidez)
        self.symbols = [
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA',
            'SPY', 'QQQ', 'META', 'AMD', 'NFLX', 'JPM'
        ]
        
        # Trading state
        self.last_trades = {}
        self.performance = {
            'total_trades': 0,
            'winning_trades': 0,
            'total_pnl': 0,
            'sharpe_ratio': 0
        }
        
        self.setup_logging()
        self.load_models_from_github()
        
    def setup_logging(self):
        """Configurar logging para Render"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s [%(levelname)s] %(message)s',
            handlers=[logging.StreamHandler(sys.stdout)]
        )
        self.logger = logging.getLogger('ScalpingBot')
        self.logger.info(f"🤖 Bot inicializado - Modo: {'PAPER' if self.use_paper else '🔴 REAL'}")
    
    def load_models_from_github(self):
        """Cargar modelos entrenados desde GitHub"""
        try:
            import requests
            
            github_token = os.environ.get('GITHUB_TOKEN')
            repo = os.environ.get('GITHUB_REPO', 'tu-usuario/trading-bot')
            
            # URL del registro de modelos
            url = f"https://api.github.com/repos/{repo}/contents/models/registry.json"
            headers = {'Authorization': f'token {github_token}'} if github_token else {}
            
            response = requests.get(url, headers=headers)
            
            if response.status_code == 200:
                import base64
                content = base64.b64decode(response.json()['content']).decode()
                registry = json.loads(content)
                
                self.logger.info(f"📥 Cargando {len(registry.get('models', {}))} modelos desde GitHub")
                
                # Descargar modelos más recientes
                for model_name, model_info in registry.get('models', {}).items():
                    self.download_model(model_name, model_info)
                
                self.last_sync = datetime.now()
                self.logger.info("✅ Modelos cargados exitosamente")
            else:
                self.logger.warning("⚠️ No se pudieron cargar modelos, usando estrategia básica")
                
        except Exception as e:
            self.logger.error(f"Error cargando modelos: {e}")
    
    def download_model(self, model_name, model_info):
        """Descargar un modelo específico"""
        try:
            # Implementar descarga de modelos
            # Por ahora usar estrategia simple
            self.models[model_name] = {
                'type': 'basic',
                'accuracy': 0.65
            }
        except Exception as e:
            self.logger.error(f"Error descargando modelo {model_name}: {e}")
    
    async def get_real_time_data(self, symbol):
        """Obtener datos en tiempo real para un símbolo"""
        try:
            # Obtener último precio
            quote = self.api.get_latest_quote(symbol)
            
            # Obtener barras de 1 minuto
            bars = self.api.get_bars(
                symbol,
                '1Min',
                limit=20,
                page_limit=1
            ).df
            
            if len(bars) < 5:
                return None
            
            # Calcular indicadores rápidos
            data = {
                'symbol': symbol,
                'price': float(quote.ap),  # Ask price
                'bid': float(quote.bp),
                'ask': float(quote.ap),
                'spread': float(quote.ap - quote.bp),
                'spread_pct': (quote.ap - quote.bp) / quote.ap,
                'volume': float(bars['volume'].iloc[-1]),
                'vwap': float(bars['vwap'].iloc[-1]) if 'vwap' in bars else quote.ap,
                'rsi': self.calculate_rsi(bars['close'], 14),
                'momentum': float((bars['close'].iloc[-1] / bars['close'].iloc[-5] - 1) * 100),
                'volatility': float(bars['close'].pct_change().std() * 100),
                'timestamp': datetime.now()
            }
            
            return data
            
        except Exception as e:
            self.logger.error(f"Error obteniendo datos para {symbol}: {e}")
            return None
    
    def calculate_rsi(self, prices, period=14):
        """Calcular RSI rápido"""
        if len(prices) < period:
            return 50
        
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return float(rsi.iloc[-1]) if not np.isnan(rsi.iloc[-1]) else 50
    
    async def analyze_opportunity(self, data):
        """Analizar si hay oportunidad de scalping"""
        try:
            symbol = data['symbol']
            
            # Verificar cooldown
            if symbol in self.last_trades:
                time_since_last = (datetime.now() - self.last_trades[symbol]).seconds
                if time_since_last < self.scalping_config['cooldown_period']:
                    return None
            
            # Verificar spread
            if data['spread_pct'] > self.scalping_config['max_spread']:
                return None
            
            # Verificar volumen
            if data['volume'] < self.scalping_config['min_volume']:
                return None
            
            # Señales de entrada
            score = 0
            confidence = 0
            
            # RSI oversold/overbought
            if data['rsi'] < 30:
                score += 1
                confidence += 0.2
            elif data['rsi'] > 70:
                score -= 1
                confidence += 0.2
            
            # Momentum
            if abs(data['momentum']) > 0.5:
                if data['momentum'] > 0:
                    score += 1
                else:
                    score -= 1
                confidence += 0.3
            
            # Volatilidad favorable
            if 0.5 < data['volatility'] < 2.0:
                confidence += 0.2
            
            # Distancia del VWAP
            vwap_distance = (data['price'] - data['vwap']) / data['vwap']
            if abs(vwap_distance) > 0.002:
                if vwap_distance < 0:  # Precio bajo VWAP
                    score += 1
                else:
                    score -= 1
                confidence += 0.3
            
            # Usar modelos si están disponibles
            if self.models:
                model_prediction = self.predict_with_models(data)
                if model_prediction:
                    score += model_prediction['signal']
                    confidence = max(confidence, model_prediction['confidence'])
            
            # Decisión final
            if abs(score) >= 2 and confidence >= 0.5:
                return {
                    'symbol': symbol,
                    'action': 'buy' if score > 0 else 'sell',
                    'confidence': confidence,
                    'score': score,
                    'price': data['price'],
                    'data': data
                }
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error analizando oportunidad: {e}")
            return None
    
    def predict_with_models(self, data):
        """Predicción usando modelos entrenados"""
        try:
            # Simplificado para el ejemplo
            # Aquí iría la lógica real de predicción
            predictions = []
            
            for model_name, model in self.models.items():
                if model['type'] == 'basic':
                    # Estrategia básica
                    signal = 1 if data['rsi'] < 40 else (-1 if data['rsi'] > 60 else 0)
                    predictions.append(signal)
            
            if predictions:
                avg_signal = np.mean(predictions)
                return {
                    'signal': int(np.sign(avg_signal)),
                    'confidence': min(abs(avg_signal), 1.0)
                }
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error en predicción: {e}")
            return None
    
    async def execute_scalping_trade(self, signal):
        """Ejecutar operación de scalping"""
        try:
            symbol = signal['symbol']
            action = signal['action']
            confidence = signal['confidence']
            
            # Verificar cuenta
            account = self.api.get_account()
            buying_power = float(account.buying_power)
            
            if buying_power < 1000:
                self.logger.warning("⚠️ Poder de compra insuficiente")
                return None
            
            # Calcular tamaño de posición
            position_value = buying_power * self.scalping_config['max_position_size']
            position_value *= confidence  # Ajustar por confianza
            
            qty = int(position_value / signal['price'])
            
            if qty < 1:
                return None
            
            # Verificar si ya tenemos posición
            try:
                position = self.api.get_position(symbol)
                self.logger.info(f"Ya tenemos posición en {symbol}")
                
                # Evaluar si cerrar
                current_pnl = float(position.unrealized_plpc)
                
                if current_pnl > self.scalping_config['profit_target']:
                    self.close_position(symbol, "TAKE_PROFIT")
                elif current_pnl < -self.scalping_config['stop_loss']:
                    self.close_position(symbol, "STOP_LOSS")
                
                return None
            except:
                pass  # No hay posición
            
            # Crear orden con bracket (stop loss y take profit)
            if action == 'buy':
                # Calcular niveles
                entry_price = signal['price']
                stop_price = entry_price * (1 - self.scalping_config['stop_loss'])
                profit_price = entry_price * (1 + self.scalping_config['profit_target'])
                
                # Orden con bracket
                order = self.api.submit_order(
                    symbol=symbol,
                    qty=qty,
                    side='buy',
                    type='market',
                    time_in_force='day',
                    order_class='bracket',
                    stop_loss={'stop_price': round(stop_price, 2)},
                    take_profit={'limit_price': round(profit_price, 2)}
                )
                
                self.logger.info(f"🟢 COMPRA: {qty} {symbol} @ ${entry_price:.2f}")
                self.logger.info(f"   ├─ Stop Loss: ${stop_price:.2f}")
                self.logger.info(f"   └─ Take Profit: ${profit_price:.2f}")
                
                # Registrar trade
                self.last_trades[symbol] = datetime.now()
                self.performance['total_trades'] += 1
                
                # Guardar info de la posición
                self.positions[symbol] = {
                    'entry_time': datetime.now(),
                    'entry_price': entry_price,
                    'qty': qty,
                    'stop_loss': stop_price,
                    'take_profit': profit_price
                }
                
                return order
                
        except Exception as e:
            self.logger.error(f"Error ejecutando trade: {e}")
            return None
    
    def close_position(self, symbol, reason=""):
        """Cerrar posición"""
        try:
            position = self.api.get_position(symbol)
            pnl = float(position.unrealized_pl)
            pnl_pct = float(position.unrealized_plpc)
            
            # Cerrar posición
            self.api.close_position(symbol)
            
            # Actualizar estadísticas
            self.performance['total_pnl'] += pnl
            if pnl > 0:
                self.performance['winning_trades'] += 1
            
            # Log
            emoji = "✅" if pnl > 0 else "❌"
            self.logger.info(f"{emoji} CERRADA: {symbol} | PnL: ${pnl:.2f} ({pnl_pct:.2%}) | Razón: {reason}")
            
            # Limpiar
            if symbol in self.positions:
                del self.positions[symbol]
                
        except Exception as e:
            self.logger.error(f"Error cerrando posición {symbol}: {e}")
    
    async def manage_positions(self):
        """Gestionar posiciones abiertas"""
        try:
            positions = self.api.list_positions()
            
            for position in positions:
                symbol = position.symbol
                entry_time = self.positions.get(symbol, {}).get('entry_time')
                
                if entry_time:
                    # Verificar tiempo máximo
                    time_held = (datetime.now() - entry_time).seconds
                    
                    if time_held > self.scalping_config['max_hold_time']:
                        self.logger.info(f"⏰ Cerrando {symbol} por tiempo máximo")
                        self.close_position(symbol, "MAX_TIME")
                        continue
                
                # Verificar PnL
                pnl_pct = float(position.unrealized_plpc)
                
                if pnl_pct > self.scalping_config['profit_target']:
                    self.close_position(symbol, "TAKE_PROFIT")
                elif pnl_pct < -self.scalping_config['stop_loss']:
                    self.close_position(symbol, "STOP_LOSS")
                    
        except Exception as e:
            self.logger.error(f"Error gestionando posiciones: {e}")
    
    async def scalping_loop(self):
        """Loop principal de scalping"""
        self.logger.info("🚀 Iniciando loop de scalping...")
        
        while self.is_running:
            try:
                # Verificar si el mercado está abierto
                clock = self.api.get_clock()
                
                if not clock.is_open:
                    self.logger.info("🔴 Mercado cerrado, esperando...")
                    await asyncio.sleep(60)
                    continue
                
                # Gestionar posiciones existentes
                await self.manage_positions()
                
                # Buscar nuevas oportunidades
                current_positions = len(self.api.list_positions())
                
                if current_positions < self.scalping_config['positions_limit']:
                    # Analizar todos los símbolos en paralelo
                    tasks = []
                    for symbol in self.symbols:
                        tasks.append(self.analyze_symbol(symbol))
                    
                    opportunities = await asyncio.gather(*tasks)
                    
                    # Filtrar y ordenar por confianza
                    valid_opportunities = [o for o in opportunities if o]
                    valid_opportunities.sort(key=lambda x: x['confidence'], reverse=True)
                    
                    # Ejecutar la mejor oportunidad
                    if valid_opportunities:
                        best = valid_opportunities[0]
                        self.logger.info(f"📊 Oportunidad: {best['symbol']} - {best['action']} (confianza: {best['confidence']:.1%})")
                        await self.execute_scalping_trade(best)
                
                # Sincronizar modelos cada hora
                if (datetime.now() - self.last_sync).seconds > 3600:
                    self.load_models_from_github()
                
                # Mostrar estadísticas cada 5 minutos
                if self.performance['total_trades'] % 10 == 0 and self.performance['total_trades'] > 0:
                    self.show_performance()
                
                # Esperar antes del siguiente ciclo (alta frecuencia)
                await asyncio.sleep(5)  # 5 segundos entre ciclos
                
            except Exception as e:
                self.logger.error(f"Error en loop principal: {e}")
                await asyncio.sleep(10)
    
    async def analyze_symbol(self, symbol):
        """Analizar un símbolo específico"""
        try:
            data = await self.get_real_time_data(symbol)
            if data:
                return await self.analyze_opportunity(data)
            return None
        except:
            return None
    
    def show_performance(self):
        """Mostrar estadísticas de rendimiento"""
        try:
            total = self.performance['total_trades']
            wins = self.performance['winning_trades']
            win_rate = (wins / total * 100) if total > 0 else 0
            
            self.logger.info("="*50)
            self.logger.info("📊 ESTADÍSTICAS DE RENDIMIENTO")
            self.logger.info("="*50)
            self.logger.info(f"Total Trades: {total}")
            self.logger.info(f"Ganadores: {wins}")
            self.logger.info(f"Win Rate: {win_rate:.1f}%")
            self.logger.info(f"PnL Total: ${self.performance['total_pnl']:.2f}")
            self.logger.info("="*50)
            
        except Exception as e:
            self.logger.error(f"Error mostrando rendimiento: {e}")
    
    def run(self):
        """Ejecutar el bot"""
        try:
            self.logger.info("="*60)
            self.logger.info("🤖 SCALPING BOT PROFESIONAL v2.0")
            self.logger.info(f"📈 Modo: {'PAPER' if self.use_paper else '🔴 REAL TRADING'}")
            self.logger.info(f"🎯 Símbolos: {', '.join(self.symbols)}")
            self.logger.info("="*60)
            
            # Verificar cuenta
            account = self.api.get_account()
            self.logger.info(f"💰 Capital: ${float(account.cash):,.2f}")
            self.logger.info(f"💪 Poder de compra: ${float(account.buying_power):,.2f}")
            
            # Ejecutar loop asíncrono
            asyncio.run(self.scalping_loop())
            
        except KeyboardInterrupt:
            self.logger.info("⏹️ Bot detenido por el usuario")
        except Exception as e:
            self.logger.error(f"Error crítico: {e}")
            raise

# Health check para Render
from flask import Flask
app = Flask(__name__)

@app.route('/health')
def health():
    return {'status': 'healthy', 'timestamp': datetime.now().isoformat()}

if __name__ == "__main__":
    # Iniciar bot en thread separado
    import threading
    
    bot = ScalpingBot()
    bot_thread = threading.Thread(target=bot.run)
    bot_thread.daemon = True
    bot_thread.start()
    
    # Iniciar servidor de health check
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port)
