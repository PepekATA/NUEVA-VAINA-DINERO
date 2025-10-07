"""
SISTEMA COMPLETO DE TRADING CRYPTO 24/7
Bot profesional con 10 agentes por par, gestión de riesgo avanzada
y protección total del capital
"""

import os
import sys
import time
import asyncio
import aiohttp
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import alpaca_trade_api as tradeapi
from alpaca_trade_api.rest import APIError
import logging
import json
import joblib
from concurrent.futures import ThreadPoolExecutor
import warnings
warnings.filterwarnings('ignore')

# Configuración para máxima protección del capital
class UltraSecureConfig:
    """Configuración ultra-segura para proteger el capital"""
    
    # Credenciales (desde variables de entorno)
    ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
    ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")
    ALPACA_BASE_URL = 'https://paper-api.alpaca.markets'
    
    # LISTA COMPLETA DE CRYPTOS EN ALPACA (24/7)
    CRYPTO_PAIRS = {
        # Tier 1 - Máxima liquidez y menor riesgo
        'TIER_1': [
            'BTCUSD',   # Bitcoin - El más estable
            'ETHUSD',   # Ethereum - Smart contracts líder
        ],
        # Tier 2 - Alta liquidez, riesgo moderado
        'TIER_2': [
            'LTCUSD',   # Litecoin
            'BCHUSD',   # Bitcoin Cash
            'LINKUSD',  # Chainlink
            'UNIUSD',   # Uniswap
            'MATICUSD', # Polygon
            'SOLUSD',   # Solana
            'ADAUSD',   # Cardano
            'AVAXUSD',  # Avalanche
        ],
        # Tier 3 - Volatilidad alta, oportunidades de scalping
        'TIER_3': [
            'DOGEUSD',  # Dogecoin
            'SHIBUSD',  # Shiba Inu
            'DOTUSD',   # Polkadot
            'AAVEUSD',  # Aave
            'CRVUSD',   # Curve
            'SUSHIUSD', # SushiSwap
            'MKRUSD',   # Maker
            'COMPUSD',  # Compound
            'YFIUSD',   # Yearn Finance
            'BATUSD',   # Basic Attention Token
            'GRTUSD',   # The Graph
        ]
    }
    
    # PROTECCIÓN MÁXIMA DEL CAPITAL
    CAPITAL_PROTECTION = {
        'max_risk_per_trade': 0.01,        # Máximo 1% de riesgo por trade
        'max_daily_loss': 0.02,             # Máximo 2% de pérdida diaria
        'max_drawdown': 0.05,               # Drawdown máximo 5%
        'emergency_stop': 0.10,             # Parar todo si -10%
        'profit_protection': 0.50,          # Proteger 50% de ganancias
        'compound_after': 0.03,             # Compound después de +3%
    }
    
    # GESTIÓN DE POSICIONES
    POSITION_MANAGEMENT = {
        'tier_1_allocation': 0.40,          # 40% en Tier 1
        'tier_2_allocation': 0.35,          # 35% en Tier 2
        'tier_3_allocation': 0.25,          # 25% en Tier 3
        'max_positions': 10,                # Máximo 10 posiciones
        'max_per_crypto': 0.10,             # Máximo 10% por crypto
        'min_profit_to_close': 0.003,       # Cerrar con 0.3% de ganancia mínima
    }
    
    # TIMEFRAMES Y ANÁLISIS
    TIMEFRAMES = {
        'ultra_fast': '1Min',
        'fast': '5Min',
        'medium': '15Min',
        'slow': '1Hour'
    }
    
    # 10 AGENTES ESPECIALIZADOS
    AGENTS = {
        'scalper_pro': {'weight': 0.15, 'min_confidence': 0.80},
        'trend_follower': {'weight': 0.12, 'min_confidence': 0.75},
        'mean_reverter': {'weight': 0.10, 'min_confidence': 0.70},
        'breakout_hunter': {'weight': 0.10, 'min_confidence': 0.75},
        'support_resistance': {'weight': 0.10, 'min_confidence': 0.70},
        'volume_analyzer': {'weight': 0.10, 'min_confidence': 0.65},
        'momentum_rider': {'weight': 0.08, 'min_confidence': 0.70},
        'pattern_detector': {'weight': 0.08, 'min_confidence': 0.75},
        'arbitrage_finder': {'weight': 0.08, 'min_confidence': 0.80},
        'risk_manager': {'weight': 0.09, 'min_confidence': 0.85}
    }


class ProfitMaximizerBot:
    """Bot principal con máxima protección del capital y enfoque ganar-ganar"""
    
    def __init__(self):
        self.config = UltraSecureConfig()
        self.validate_credentials()
        
        # Inicializar API
        self.api = tradeapi.REST(
            self.config.ALPACA_API_KEY,
            self.config.ALPACA_SECRET_KEY,
            self.config.ALPACA_BASE_URL,
            api_version='v2'
        )
        
        # Estado del sistema
        self.is_running = True
        self.emergency_stop_activated = False
        self.positions = {}
        self.daily_pnl = 0
        self.total_pnl = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.peak_balance = 0
        self.current_drawdown = 0
        
        # Agentes por crypto
        self.agents = {}
        self.initialize_all_agents()
        
        # Sistema de logs
        self.setup_advanced_logging()
        
        # Verificar cuenta inicial
        self.initial_check()
    
    def validate_credentials(self):
        """Validar que las credenciales existan"""
        if not self.config.ALPACA_API_KEY or not self.config.ALPACA_SECRET_KEY:
            print("❌ ERROR: Credenciales no configuradas")
            print("Configure las variables de entorno:")
            print("  ALPACA_API_KEY=your_key")
            print("  ALPACA_SECRET_KEY=your_secret")
            sys.exit(1)
    
    def setup_advanced_logging(self):
        """Sistema de logging avanzado"""
        log_format = '%(asctime)s | %(levelname)s | %(message)s'
        
        # Crear directorio de logs
        os.makedirs('logs', exist_ok=True)
        
        # Logger principal
        logging.basicConfig(
            level=logging.INFO,
            format=log_format,
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.FileHandler('logs/trading.log'),
                logging.FileHandler(f'logs/trading_{datetime.now().strftime("%Y%m%d")}.log')
            ]
        )
        self.logger = logging.getLogger('ProfitBot')
        
        # Logger de trades
        self.trade_logger = logging.getLogger('Trades')
        trade_handler = logging.FileHandler('logs/trades.log')
        trade_handler.setFormatter(logging.Formatter(log_format))
        self.trade_logger.addHandler(trade_handler)
    
    def initial_check(self):
        """Verificación inicial del sistema"""
        try:
            account = self.api.get_account()
            
            self.initial_balance = float(account.cash)
            self.peak_balance = self.initial_balance
            
            self.logger.info("="*70)
            self.logger.info("🚀 SISTEMA DE TRADING CRYPTO 24/7 INICIADO")
            self.logger.info("="*70)
            self.logger.info(f"💰 Balance Inicial: ${self.initial_balance:,.2f}")
            self.logger.info(f"💪 Poder de Compra: ${float(account.buying_power):,.2f}")
            self.logger.info(f"📊 Valor Portfolio: ${float(account.portfolio_value):,.2f}")
            self.logger.info(f"🛡️ Protección de Capital ACTIVADA")
            self.logger.info(f"🎯 Objetivo: GANAR-GANAR con riesgo mínimo")
            self.logger.info("="*70)
            
        except Exception as e:
            self.logger.error(f"Error en verificación inicial: {e}")
            sys.exit(1)
    
    def initialize_all_agents(self):
        """Inicializar 10 agentes para cada crypto"""
        self.logger.info("🤖 Inicializando sistema multi-agente...")
        
        all_cryptos = []
        for tier_cryptos in self.config.CRYPTO_PAIRS.values():
            all_cryptos.extend(tier_cryptos)
        
        for crypto in all_cryptos:
            self.agents[crypto] = TradingAgentTeam(crypto, self.config)
            
        self.logger.info(f"✅ {len(self.agents)} equipos de agentes creados")
        self.logger.info(f"📊 Total: {len(self.agents) * 10} agentes activos")
    
    async def get_market_data(self, symbol, timeframe='1Min'):
        """Obtener datos de mercado con indicadores"""
        try:
            # Obtener barras históricas
            bars = self.api.get_crypto_bars(
                symbol,
                timeframe,
                limit=100
            ).df
            
            if bars.empty or len(bars) < 20:
                return None
            
            # Calcular indicadores técnicos
            bars['sma_20'] = bars['close'].rolling(window=20).mean()
            bars['sma_50'] = bars['close'].rolling(window=50).mean() if len(bars) >= 50 else bars['sma_20']
            bars['ema_12'] = bars['close'].ewm(span=12, adjust=False).mean()
            bars['ema_26'] = bars['close'].ewm(span=26, adjust=False).mean()
            
            # RSI
            bars['rsi'] = self.calculate_rsi(bars['close'])
            
            # MACD
            bars['macd'] = bars['ema_12'] - bars['ema_26']
            bars['signal'] = bars['macd'].ewm(span=9, adjust=False).mean()
            bars['histogram'] = bars['macd'] - bars['signal']
            
            # Bollinger Bands
            bars['bb_middle'] = bars['close'].rolling(window=20).mean()
            bb_std = bars['close'].rolling(window=20).std()
            bars['bb_upper'] = bars['bb_middle'] + (bb_std * 2)
            bars['bb_lower'] = bars['bb_middle'] - (bb_std * 2)
            
            # Volume indicators
            bars['volume_sma'] = bars['volume'].rolling(window=20).mean()
            bars['volume_ratio'] = bars['volume'] / bars['volume_sma']
            
            # Volatility
            bars['volatility'] = bars['close'].pct_change().rolling(window=20).std() * np.sqrt(252)
            
            # Support and Resistance levels
            bars['resistance'] = bars['high'].rolling(window=20).max()
            bars['support'] = bars['low'].rolling(window=20).min()
            
            return bars
            
        except Exception as e:
            self.logger.error(f"Error obteniendo datos para {symbol}: {e}")
            return None
    
    def calculate_rsi(self, prices, period=14):
        """Calcular RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50)
    
    async def analyze_opportunity(self, symbol, data):
        """Analizar oportunidad con los 10 agentes"""
        if data is None or len(data) < 20:
            return None
        
        # Obtener análisis del equipo de agentes
        team_analysis = await self.agents[symbol].analyze(data)
        
        # Verificar condiciones de entrada
        current_price = float(data['close'].iloc[-1])
        rsi = float(data['rsi'].iloc[-1])
        macd_hist = float(data['histogram'].iloc[-1])
        volume_ratio = float(data['volume_ratio'].iloc[-1])
        
        # Análisis combinado
        signal_strength = team_analysis['signal_strength']
        confidence = team_analysis['confidence']
        
        # Decisión final con múltiples filtros de seguridad
        decision = {
            'symbol': symbol,
            'action': 'HOLD',
            'confidence': confidence,
            'price': current_price,
            'reasons': []
        }
        
        # Condiciones para COMPRA (todas deben cumplirse)
        buy_conditions = [
            signal_strength > 0.6,                    # Señal fuerte del equipo
            confidence > 0.75,                        # Alta confianza
            rsi < 70,                                  # No sobrecomprado
            volume_ratio > 1.2,                       # Volumen superior al promedio
            self.check_risk_limits(),                 # Límites de riesgo OK
            not self.emergency_stop_activated        # Sin stop de emergencia
        ]
        
        if all(buy_conditions):
            decision['action'] = 'BUY'
            decision['reasons'] = [
                f"Señal: {signal_strength:.2f}",
                f"RSI: {rsi:.0f}",
                f"Volumen: {volume_ratio:.1f}x",
                f"MACD: {'Bullish' if macd_hist > 0 else 'Turning'}"
            ]
        
        # Condiciones para VENTA (si tenemos posición)
        elif symbol in self.positions:
            position = self.positions[symbol]
            current_pnl = (current_price - position['entry_price']) / position['entry_price']
            
            sell_conditions = [
                current_pnl > self.config.POSITION_MANAGEMENT['min_profit_to_close'],  # Ganancia mínima
                signal_strength < -0.3,                                                  # Señal de venta
                rsi > 75,                                                               # Sobrecomprado
            ]
            
            # Venta por take profit
            if current_pnl > 0.01:  # 1% de ganancia
                decision['action'] = 'SELL'
                decision['reasons'].append(f"Take Profit: {current_pnl:.2%}")
            
            # Venta por stop loss
            elif current_pnl < -0.005:  # -0.5% de pérdida
                decision['action'] = 'SELL'
                decision['reasons'].append(f"Stop Loss: {current_pnl:.2%}")
            
            # Venta por señales técnicas
            elif any(sell_conditions):
                decision['action'] = 'SELL'
                decision['reasons'].append("Señales técnicas de venta")
        
        return decision
    
    def check_risk_limits(self):
        """Verificar límites de riesgo antes de operar"""
        try:
            account = self.api.get_account()
            current_balance = float(account.cash)
            
            # Calcular drawdown actual
            if current_balance < self.peak_balance:
                self.current_drawdown = (self.peak_balance - current_balance) / self.peak_balance
            else:
                self.peak_balance = current_balance
                self.current_drawdown = 0
            
            # Verificar límites
            if self.current_drawdown > self.config.CAPITAL_PROTECTION['max_drawdown']:
                self.logger.warning(f"⚠️ Drawdown alto: {self.current_drawdown:.2%}")
                return False
            
            if self.daily_pnl < -self.initial_balance * self.config.CAPITAL_PROTECTION['max_daily_loss']:
                self.logger.warning(f"⚠️ Pérdida diaria máxima alcanzada")
                return False
            
            # Todo OK
            return True
            
        except Exception as e:
            self.logger.error(f"Error verificando límites: {e}")
            return False
    
    async def execute_trade(self, decision):
        """Ejecutar trade con máxima protección"""
        try:
            symbol = decision['symbol']
            action = decision['action']
            
            if action == 'HOLD':
                return None
            
            account = self.api.get_account()
            buying_power = float(account.buying_power)
            
            if action == 'BUY':
                # Calcular tamaño de posición con Kelly Criterion modificado
                confidence = decision['confidence']
                kelly_fraction = (confidence - 0.5) * 2  # Simplificado
                position_pct = min(
                    kelly_fraction * 0.25,  # 25% del Kelly
                    self.config.CAPITAL_PROTECTION['max_risk_per_trade']
                )
                
                # Determinar tier del crypto
                tier = self.get_crypto_tier(symbol)
                if tier == 1:
                    position_pct *= 1.5  # Más capital a cryptos estables
                elif tier == 3:
                    position_pct *= 0.5  # Menos capital a cryptos volátiles
                
                # Calcular cantidad
                position_value = buying_power * position_pct
                qty = position_value / decision['price']
                
                # Ajustar precisión según crypto
                if 'BTC' in symbol:
                    qty = round(qty, 6)
                elif 'ETH' in symbol:
                    qty = round(qty, 4)
                else:
                    qty = round(qty, 2)
                
                if qty < 0.001:
                    return None
                
                # Stop loss y take profit dinámicos
                volatility = await self.get_volatility(symbol)
                stop_loss_pct = min(0.005, volatility * 0.5)  # Máximo 0.5%
                take_profit_pct = max(0.003, volatility * 1.5)  # Mínimo 0.3%
                
                stop_price = decision['price'] * (1 - stop_loss_pct)
                profit_price = decision['price'] * (1 + take_profit_pct)
                
                # Crear orden con protección
                order = self.api.submit_order(
                    symbol=symbol,
                    qty=qty,
                    side='buy',
                    type='market',
                    time_in_force='gtc',
                    order_class='bracket',
                    stop_loss={'stop_price': round(stop_price, 2)},
                    take_profit={'limit_price': round(profit_price, 2)}
                )
                
                # Registrar posición
                self.positions[symbol] = {
                    'qty': qty,
                    'entry_price': decision['price'],
                    'entry_time': datetime.now(),
                    'stop_loss': stop_price,
                    'take_profit': profit_price,
                    'confidence': confidence,
                    'tier': tier
                }
                
                # Log detallado
                self.trade_logger.info(f"""
                ✅ COMPRA EJECUTADA:
                ├─ Símbolo: {symbol}
                ├─ Cantidad: {qty}
                ├─ Precio: ${decision['price']:.4f}
                ├─ Stop Loss: ${stop_price:.4f} (-{stop_loss_pct:.2%})
                ├─ Take Profit: ${profit_price:.4f} (+{take_profit_pct:.2%})
                ├─ Valor: ${position_value:.2f}
                ├─ Confianza: {confidence:.2%}
                └─ Razones: {', '.join(decision['reasons'])}
                """)
                
                return order
            
            elif action == 'SELL' and symbol in self.positions:
                position = self.positions[symbol]
                
                # Calcular PnL
                current_price = decision['price']
                pnl = (current_price - position['entry_price']) * position['qty']
                pnl_pct = (current_price / position['entry_price'] - 1) * 100
                
                # Ejecutar venta
                order = self.api.submit_order(
                    symbol=symbol,
                    qty=position['qty'],
                    side='sell',
                    type='market',
                    time_in_force='gtc'
                )
                
                # Actualizar estadísticas
                self.daily_pnl += pnl
                self.total_pnl += pnl
                
                if pnl > 0:
                    self.winning_trades += 1
                else:
                    self.losing_trades += 1
                
                # Log de cierre
                emoji = "💰" if pnl > 0 else "📉"
                self.trade_logger.info(f"""
                {emoji} VENTA EJECUTADA:
                ├─ Símbolo: {symbol}
                ├─ PnL: ${pnl:.2f} ({pnl_pct:.2f}%)
                ├─ Duración: {(datetime.now() - position['entry_time']).seconds // 60} minutos
                └─ Razón: {', '.join(decision['reasons'])}
                """)
                
                # Limpiar posición
                del self.positions[symbol]
                
                return order
                
        except Exception as e:
            self.logger.error(f"Error ejecutando trade: {e}")
            return None
    
    def get_crypto_tier(self, symbol):
        """Determinar el tier del crypto"""
        for tier, cryptos in enumerate(self.config.CRYPTO_PAIRS.values(), 1):
            if symbol in cryptos:
                return tier
        return 3  # Default tier más conservador
    
    async def get_volatility(self, symbol):
        """Calcular volatilidad actual"""
        try:
            bars = self.api.get_crypto_bars(symbol, '5Min', limit=20).df
            if not bars.empty:
                return float(bars['close'].pct_change().std())
            return 0.01  # Default 1%
        except:
            return 0.01
    
    async def monitor_all_positions(self):
        """Monitorear y gestionar todas las posiciones"""
        try:
            positions = self.api.list_positions()
            
            for position in positions:
                symbol = position.symbol
                
                if symbol not in self.positions:
                    continue
                
                current_price = float(position.current_price)
                entry_price = self.positions[symbol]['entry_price']
                pnl_pct = (current_price / entry_price - 1)
                
                # Trailing stop si hay ganancias
                if pnl_pct > 0.005:  # 0.5% de ganancia
                    new_stop = current_price * 0.997  # Trailing de 0.3%
                    
                    if new_stop > self.positions[symbol]['stop_loss']:
                        self.positions[symbol]['stop_loss'] = new_stop
                        self.logger.info(f"📈 Trailing stop actualizado para {symbol}: ${new_stop:.4f}")
                
                # Verificar tiempo máximo de hold
                hold_time = (datetime.now() - self.positions[symbol]['entry_time']).seconds / 3600
                
                if hold_time > 4:  # Máximo 4 horas
                    self.logger.info(f"⏰ Cerrando {symbol} por tiempo máximo")
                    self.api.close_position(symbol)
                    del self.positions[symbol]
                
        except Exception as e:
            self.logger.error(f"Error monitoreando posiciones: {e}")
    
    async def emergency_check(self):
        """Verificación de emergencia del sistema"""
        try:
            account = self.api.get_account()
            current_balance = float(account.portfolio_value)
            
            # Check de emergencia
            loss_pct = (self.initial_balance - current_balance) / self.initial_balance
            
            if loss_pct > self.config.CAPITAL_PROTECTION['emergency_stop']:
                self.emergency_stop_activated = True
                self.logger.critical("🚨 EMERGENCY STOP ACTIVADO - Cerrando todas las posiciones")
                
                # Cerrar todo
                positions = self.api.list_positions()
                for pos in positions:
                    self.api.close_position(pos.symbol)
                
                self.logger.critical(f"🛑 Bot detenido. Pérdida: {loss_pct:.2%}")
                self.is_running = False
                
        except Exception as e:
            self.logger.error(f"Error en check de emergencia: {e}")
    
    def show_performance_dashboard(self):
        """Mostrar dashboard de rendimiento"""
        try:
            account = self.api.get_account()
            current_balance = float(account.portfolio_value)
            
            total_trades = self.winning_trades + self.losing_trades
            win_rate = (self.winning_trades / total_trades * 100) if total_trades > 0 else 0
            
            roi = ((current_balance - self.initial_balance) / self.initial_balance * 100)
            
            self.logger.info("\n" + "="*70)
            self.logger.info("📊 DASHBOARD DE RENDIMIENTO")
            self.logger.info("="*70)
            self.logger.info(f"💰 Balance Actual: ${current_balance:,.2f}")
            self.logger.info(f"📈 ROI: {roi:.2f}%")
            self.logger.info(f"💵 PnL Total: ${self.total_pnl:.2f}")
            self.logger.info(f"📊 PnL Diario: ${self.daily_pnl:.2f}")
            self.logger.info(f"✅ Trades Ganadores: {self.winning_trades}")
            self.logger.info(f"❌ Trades Perdedores: {self.losing_trades}")
            self.logger.info(f"🎯 Win Rate: {win_rate:.1f}%")
            self.logger.info(f"📉 Drawdown Actual: {self.current_drawdown:.2%}")
            self.logger.info(f"🔒 Posiciones Abiertas: {len(self.positions)}")
            self.logger.info("="*70)
            
        except Exception as e:
            self.logger.error(f"Error mostrando dashboard: {e}")
    
    async def main_trading_loop(self):
        """Loop principal de trading 24/7"""
        self.logger.info("🚀 Iniciando trading loop 24/7...")
        
        cycle = 0
        last_daily_reset = datetime.now().date()
        
        while self.is_running:
            try:
                cycle += 1
                
                # Reset diario
                current_date = datetime.now().date()
                if current_date > last_daily_reset:
                    self.daily_pnl = 0
                    last_daily_reset = current_date
                    self.logger.info("📅 Nuevo día de trading - Contadores reseteados")
                
                # Check de emergencia
                await self.emergency_check()
                
                if self.emergency_stop_activated:
                    break
                
                # Monitorear posiciones existentes
                await self.monitor_all_positions()
                
                # Buscar nuevas oportunidades
                all_cryptos = []
                for cryptos in self.config.CRYPTO_PAIRS.values():
                    all_cryptos.extend(cryptos)
                
                # Analizar cryptos por tiers (prioridad a Tier 1)
                opportunities = []
                
                # Tier 1 - Mayor prioridad
                for symbol in self.config.CRYPTO_PAIRS['TIER_1']:
                    if symbol not in self.positions:
                        data = await self.get_market_data(symbol)
                        if data is not None:
                            analysis = await self.analyze_opportunity(symbol, data)
                            if analysis and analysis['action'] != 'HOLD':
                                analysis['priority'] = 1
                                opportunities.append(analysis)
                
                # Tier 2
                for symbol in self.config.CRYPTO_PAIRS['TIER_2']:
                    if symbol not in self.positions:
                        data = await self.get_market_data(symbol)
                        if data is not None:
                            analysis = await self.analyze_opportunity(symbol, data)
                            if analysis and analysis['action'] != 'HOLD':
                                analysis['priority'] = 2
                                opportunities.append(analysis)
                
                # Tier 3 - Solo si hay espacio
                if len(self.positions) < 5:
                    for symbol in self.config.CRYPTO_PAIRS['TIER_3']:
                        if symbol not in self.positions:
                            data = await self.get_market_data(symbol, '1Min')
                            if data is not None:
                                analysis = await self.analyze_opportunity(symbol, data)
                                if analysis and analysis['action'] != 'HOLD':
                                    analysis['priority'] = 3
                                    opportunities.append(analysis)
                
                # Ordenar por prioridad y confianza
                opportunities.sort(key=lambda x: (x['priority'], -x['confidence']))
                
                # Ejecutar las mejores oportunidades
                max_new_positions = self.config.POSITION_MANAGEMENT['max_positions'] - len(self.positions)
                
                for opp in opportunities[:max_new_positions]:
                    if self.check_risk_limits():
                        await self.execute_trade(opp)
                        await asyncio.sleep(1)  # Pausa entre trades
                
                # Mostrar dashboard cada 10 ciclos
                if cycle % 10 == 0:
                    self.show_performance_dashboard()
                
                # Log de ciclo
                if cycle % 5 == 0:
                    self.logger.info(f"📍 Ciclo {cycle} | Posiciones: {len(self.positions)}/{self.config.POSITION_MANAGEMENT['max_positions']} | PnL Diario: ${self.daily_pnl:.2f}")
                
                # Guardar estado cada hora
                if cycle % 60 == 0:
                    self.save_state()
                
                # Esperar antes del próximo ciclo
                await asyncio.sleep(30)  # 30 segundos entre ciclos
                
            except Exception as e:
                self.logger.error(f"Error en trading loop: {e}")
                await asyncio.sleep(60)
        
        # Cleanup al salir
        self.cleanup()
    
    def save_state(self):
        """Guardar estado del bot"""
        try:
            state = {
                'timestamp': datetime.now().isoformat(),
                'positions': self.positions,
                'daily_pnl': self.daily_pnl,
                'total_pnl': self.total_pnl,
                'winning_trades': self.winning_trades,
                'losing_trades': self.losing_trades,
                'peak_balance': self.peak_balance,
                'current_drawdown': self.current_drawdown
            }
            
            os.makedirs('state', exist_ok=True)
            filename = f"state/bot_state_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
            with open(filename, 'w') as f:
                json.dump(state, f, default=str, indent=2)
            
            self.logger.info(f"💾 Estado guardado: {filename}")
            
        except Exception as e:
            self.logger.error(f"Error guardando estado: {e}")
    
    def cleanup(self):
        """Limpieza al cerrar el bot"""
        self.logger.info("🛑 Cerrando bot...")
        
        # Cerrar todas las posiciones
        try:
            positions = self.api.list_positions()
            for pos in positions:
                self.logger.info(f"Cerrando posición: {pos.symbol}")
                self.api.close_position(pos.symbol)
        except:
            pass
        
        # Guardar estado final
        self.save_state()
        
        # Mostrar resumen final
        self.show_performance_dashboard()
        
        self.logger.info("✅ Bot cerrado correctamente")
    
    def run(self):
        """Ejecutar el bot"""
        try:
            # Ejecutar loop asíncrono
            asyncio.run(self.main_trading_loop())
            
        except KeyboardInterrupt:
            self.logger.info("⏹️ Bot detenido por el usuario")
            self.cleanup()
        except Exception as e:
            self.logger.critical(f"Error crítico: {e}")
            self.cleanup()
            raise


class TradingAgentTeam:
    """Equipo de 10 agentes especializados para cada crypto"""
    
    def __init__(self, symbol, config):
        self.symbol = symbol
        self.config = config
        self.agents = self.create_agents()
    
    def create_agents(self):
        """Crear los 10 agentes especializados"""
        return {
            'scalper_pro': ScalperAgent(),
            'trend_follower': TrendAgent(),
            'mean_reverter': MeanReversionAgent(),
            'breakout_hunter': BreakoutAgent(),
            'support_resistance': SupportResistanceAgent(),
            'volume_analyzer': VolumeAgent(),
            'momentum_rider': MomentumAgent(),
            'pattern_detector': PatternAgent(),
            'arbitrage_finder': ArbitrageAgent(),
            'risk_manager': RiskAgent()
        }
    
    async def analyze(self, data):
        """Análisis conjunto de todos los agentes"""
        signals = []
        confidences = []
        
        for agent_name, agent in self.agents.items():
            signal, confidence = agent.analyze(data)
            weight = self.config.AGENTS[agent_name]['weight']
            
            signals.append(signal * weight)
            confidences.append(confidence * weight)
        
        # Consenso del equipo
        team_signal = sum(signals)
        team_confidence = sum(confidences) / len(confidences) if confidences else 0
        
        return {
            'signal_strength': team_signal,
            'confidence': team_confidence,
            'consensus': sum(1 for s in signals if s > 0) / len(signals)
        }


# AGENTES ESPECIALIZADOS

class ScalperAgent:
    """Agente especializado en scalping rápido"""
    
    def analyze(self, data):
        """Análisis para scalping"""
        try:
            # Últimas velas
            last_close = data['close'].iloc[-1]
            last_open = data['open'].iloc[-1]
            
            # Micro tendencia (últimas 5 velas)
            micro_trend = data['close'].iloc[-5:].pct_change().mean()
            
            # Spread bid-ask simulado
            spread = abs(data['high'].iloc[-1] - data['low'].iloc[-1]) / last_close
            
            # Momentum inmediato
            momentum = (last_close - data['close'].iloc[-3]) / data['close'].iloc[-3]
            
            # Señal de scalping
            signal = 0
            confidence = 0
            
            # Compra en dip rápido
            if micro_trend < -0.001 and momentum > 0:
                signal = 1
                confidence = 0.7
            
            # Venta en spike rápido
            elif micro_trend > 0.003 and momentum < 0:
                signal = -1
                confidence = 0.7
            
            # Ajustar por spread
            if spread > 0.002:  # Spread alto
                confidence *= 0.8
            
            return signal, confidence
            
        except:
            return 0, 0


class TrendAgent:
    """Agente seguidor de tendencias"""
    
    def analyze(self, data):
        """Análisis de tendencia"""
        try:
            # Moving averages
            sma_20 = data['sma_20'].iloc[-1]
            sma_50 = data['sma_50'].iloc[-1] if 'sma_50' in data else sma_20
            current_price = data['close'].iloc[-1]
            
            # Tendencia
            trend_strength = (current_price - sma_50) / sma_50
            
            # MACD
            macd_hist = data['histogram'].iloc[-1] if 'histogram' in data else 0
            
            signal = 0
            confidence = 0
            
            # Tendencia alcista fuerte
            if current_price > sma_20 > sma_50 and macd_hist > 0:
                signal = 1
                confidence = min(0.9, abs(trend_strength) * 10)
            
            # Tendencia bajista fuerte
            elif current_price < sma_20 < sma_50 and macd_hist < 0:
                signal = -1
                confidence = min(0.9, abs(trend_strength) * 10)
            
            return signal, confidence
            
        except:
            return 0, 0


class MeanReversionAgent:
    """Agente de reversión a la media"""
    
    def analyze(self, data):
        """Análisis de reversión a la media"""
        try:
            current_price = data['close'].iloc[-1]
            bb_upper = data['bb_upper'].iloc[-1]
            bb_lower = data['bb_lower'].iloc[-1]
            bb_middle = data['bb_middle'].iloc[-1]
            rsi = data['rsi'].iloc[-1]
            
            # Distancia de las bandas
            bb_position = (current_price - bb_lower) / (bb_upper - bb_lower)
            
            signal = 0
            confidence = 0
            
            # Sobreventa extrema
            if bb_position < 0.2 and rsi < 30:
                signal = 1
                confidence = 0.85
            
            # Sobrecompra extrema
            elif bb_position > 0.8 and rsi > 70:
                signal = -1
                confidence = 0.85
            
            # Reversión desde bandas
            elif current_price < bb_lower:
                signal = 1
                confidence = 0.7
            elif current_price > bb_upper:
                signal = -1
                confidence = 0.7
            
            return signal, confidence
            
        except:
            return 0, 0


class BreakoutAgent:
    """Agente cazador de rupturas"""
    
    def analyze(self, data):
        """Análisis de breakouts"""
        try:
            current_price = data['close'].iloc[-1]
            resistance = data['resistance'].iloc[-1]
            support = data['support'].iloc[-1]
            volume_ratio = data['volume_ratio'].iloc[-1]
            
            # Rango actual
            range_size = (resistance - support) / support
            
            signal = 0
            confidence = 0
            
            # Breakout alcista con volumen
            if current_price > resistance and volume_ratio > 1.5:
                signal = 1
                confidence = min(0.9, volume_ratio / 2)
            
            # Breakdown bajista con volumen
            elif current_price < support and volume_ratio > 1.5:
                signal = -1
                confidence = min(0.9, volume_ratio / 2)
            
            # Pre-breakout
            elif current_price > resistance * 0.99 and volume_ratio > 1.2:
                signal = 0.5
                confidence = 0.6
            
            return signal, confidence
            
        except:
            return 0, 0


class SupportResistanceAgent:
    """Agente de soportes y resistencias"""
    
    def analyze(self, data):
        """Análisis de S/R"""
        try:
            current_price = data['close'].iloc[-1]
            high_20 = data['high'].rolling(20).max().iloc[-1]
            low_20 = data['low'].rolling(20).min().iloc[-1]
            
            # Niveles clave
            pivot = (high_20 + low_20 + current_price) / 3
            r1 = 2 * pivot - low_20
            s1 = 2 * pivot - high_20
            
            signal = 0
            confidence = 0
            
            # Rebote en soporte
            if abs(current_price - s1) / s1 < 0.002:
                signal = 1
                confidence = 0.75
            
            # Rechazo en resistencia
            elif abs(current_price - r1) / r1 < 0.002:
                signal = -1
                confidence = 0.75
            
            # Cerca del pivot
            elif abs(current_price - pivot) / pivot < 0.001:
                # Dirección de la tendencia previa
                prev_trend = data['close'].iloc[-5:].pct_change().mean()
                signal = 1 if prev_trend > 0 else -1
                confidence = 0.6
            
            return signal, confidence
            
        except:
            return 0, 0


class VolumeAgent:
    """Agente analizador de volumen"""
    
    def analyze(self, data):
        """Análisis de volumen"""
        try:
            volume = data['volume'].iloc[-1]
            volume_ma = data['volume_sma'].iloc[-1]
            volume_ratio = volume / volume_ma if volume_ma > 0 else 1
            
            # Cambio de precio con volumen
            price_change = data['close'].pct_change().iloc[-1]
            
            signal = 0
            confidence = 0
            
            # Volumen alto con precio subiendo
            if volume_ratio > 2 and price_change > 0.001:
                signal = 1
                confidence = min(0.85, volume_ratio / 3)
            
            # Volumen alto con precio bajando (distribución)
            elif volume_ratio > 2 and price_change < -0.001:
                signal = -1
                confidence = min(0.85, volume_ratio / 3)
            
            # Volumen bajo (no trade)
            elif volume_ratio < 0.5:
                signal = 0
                confidence = 0.3
            
            return signal, confidence
            
        except:
            return 0, 0


class MomentumAgent:
    """Agente de momentum"""
    
    def analyze(self, data):
        """Análisis de momentum"""
        try:
            # Rate of change
            roc_5 = (data['close'].iloc[-1] / data['close'].iloc[-6] - 1) * 100
            roc_10 = (data['close'].iloc[-1] / data['close'].iloc[-11] - 1) * 100
            
            # RSI momentum
            rsi = data['rsi'].iloc[-1]
            rsi_prev = data['rsi'].iloc[-2]
            rsi_momentum = rsi - rsi_prev
            
            signal = 0
            confidence = 0
            
            # Momentum alcista fuerte
            if roc_5 > 0.5 and roc_10 > 0.3 and rsi_momentum > 0:
                signal = 1
                confidence = min(0.85, roc_5 / 2)
            
            # Momentum bajista fuerte
            elif roc_5 < -0.5 and roc_10 < -0.3 and rsi_momentum < 0:
                signal = -1
                confidence = min(0.85, abs(roc_5) / 2)
            
            # Divergencia
            elif roc_5 > 0 and rsi_momentum < -5:
                signal = -0.5
                confidence = 0.6
            
            return signal, confidence
            
        except:
            return 0, 0


class PatternAgent:
    """Agente detector de patrones"""
    
    def analyze(self, data):
        """Detectar patrones de velas"""
        try:
            # Últimas 3 velas
            opens = data['open'].iloc[-3:].values
            closes = data['close'].iloc[-3:].values
            highs = data['high'].iloc[-3:].values
            lows = data['low'].iloc[-3:].values
            
            signal = 0
            confidence = 0
            
            # Doji (indecisión)
            body = abs(closes[-1] - opens[-1])
            shadow = highs[-1] - lows[-1]
            
            if body < shadow * 0.1:
                signal = 0
                confidence = 0.5
                
            # Hammer (reversión alcista)
            elif (closes[-1] > opens[-1] and 
                  body < shadow * 0.3 and
                  lows[-1] < min(lows[-3:-1])):
                signal = 1
                confidence = 0.75
            
            # Shooting star (reversión bajista)
            elif (closes[-1] < opens[-1] and
                  body < shadow * 0.3 and
                  highs[-1] > max(highs[-3:-1])):
                signal = -1
                confidence = 0.75
            
            # Engulfing alcista
            elif (closes[-2] < opens[-2] and  # Vela bajista previa
                  closes[-1] > opens[-1] and  # Vela alcista actual
                  opens[-1] < closes[-2] and  # Abre por debajo
                  closes[-1] > opens[-2]):    # Cierra por encima
                signal = 1
                confidence = 0.8
            
            return signal, confidence
            
        except:
            return 0, 0


class ArbitrageAgent:
    """Agente de arbitraje (simulado)"""
    
    def analyze(self, data):
        """Análisis de arbitraje"""
        try:
            # Simular diferencias de precio entre exchanges
            current_price = data['close'].iloc[-1]
            high = data['high'].iloc[-1]
            low = data['low'].iloc[-1]
            
            # Spread simulado
            spread_pct = (high - low) / current_price
            
            signal = 0
            confidence = 0
            
            # Oportunidad de arbitraje
            if spread_pct > 0.003:  # 0.3% de spread
                # Comprar en el precio bajo
                if current_price < (high + low) / 2:
                    signal = 1
                    confidence = min(0.7, spread_pct * 100)
                else:
                    signal = -1
                    confidence = min(0.7, spread_pct * 100)
            
            return signal, confidence
            
        except:
            return 0, 0


class RiskAgent:
    """Agente gestor de riesgo"""
    
    def analyze(self, data):
        """Análisis de riesgo"""
        try:
            # Volatilidad
            volatility = data['volatility'].iloc[-1] if 'volatility' in data else 0.01
            
            # RSI para extremos
            rsi = data['rsi'].iloc[-1]
            
            # Volume para liquidez
            volume_ratio = data['volume_ratio'].iloc[-1]
            
            signal = 0
            confidence = 0
            
            # Condiciones de bajo riesgo
            if 0.005 < volatility < 0.02 and 40 < rsi < 60 and volume_ratio > 0.8:
                # Señal neutral con alta confianza
                signal = 0.1  # Ligeramente alcista
                confidence = 0.9
            
            # Alto riesgo - evitar
            elif volatility > 0.05 or rsi > 80 or rsi < 20:
                signal = 0
                confidence = 0.2  # Baja confianza
            
            # Riesgo moderado
            else:
                signal = 0
                confidence = 0.6
            
            return signal, confidence
            
        except:
            return 0, 0


# FUNCIÓN PRINCIPAL
def main():
    """Función principal para ejecutar el bot"""
    
    # Verificar variables de entorno
    required_env = ['ALPACA_API_KEY', 'ALPACA_SECRET_KEY']
    missing = [var for var in required_env if not os.getenv(var)]
    
    if missing:
        print("❌ ERROR: Variables de entorno faltantes:")
        for var in missing:
            print(f"  - {var}")
        print("\nConfigura las variables en Render.com o en tu archivo .env")
        sys.exit(1)
    
    # Crear y ejecutar el bot
    print("="*70)
    print("🚀 INICIANDO BOT DE TRADING CRYPTO 24/7")
    print("="*70)
    print("⚠️  IMPORTANTE:")
    print("  - Este bot opera con dinero real (o paper trading)")
    print("  - Monitorea constantemente el rendimiento")
    print("  - Comienza con cantidades pequeñas")
    print("="*70)
    
    bot = ProfitMaximizerBot()
    bot.run()


# Health check para Render
from flask import Flask, jsonify
app = Flask(__name__)

bot_instance = None

@app.route('/health')
def health():
    """Endpoint de health check"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'bot_running': bot_instance.is_running if bot_instance else False
    })

@app.route('/stats')
def stats():
    """Endpoint de estadísticas"""
    if bot_instance:
        return jsonify({
            'positions': len(bot_instance.positions),
            'daily_pnl': bot_instance.daily_pnl,
            'total_pnl': bot_instance.total_pnl,
            'winning_trades': bot_instance.winning_trades,
            'losing_trades': bot_instance.losing_trades
        })
    return jsonify({'error': 'Bot not running'})


if __name__ == "__main__":
    import threading
    
    # Iniciar bot en thread separado
    bot_instance = ProfitMaximizerBot()
    bot_thread = threading.Thread(target=bot_instance.run)
    bot_thread.daemon = True
    bot_thread.start()
    
    # Iniciar servidor Flask para health checks
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port)
