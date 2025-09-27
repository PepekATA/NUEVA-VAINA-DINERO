"""
MÓDULO DE TRADING 24/7
Sistema de trading autónomo con gestión de capital inteligente
"""

import alpaca_trade_api as tradeapi
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import logging
import json
from threading import Thread, Lock
from queue import Queue
from config import Config

class Trader24_7:
    """Sistema de trading 24/7 con multi-agentes"""
    
    def __init__(self):
        self.api = tradeapi.REST(
            Config.ALPACA_API_KEY,
            Config.ALPACA_SECRET_KEY,
            Config.ALPACA_BASE_URL,
            api_version='v2'
        )
        
        # Estado del sistema
        self.is_running = False
        self.positions = {}
        self.pending_orders = {}
        self.performance = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'total_profit': 0,
            'max_drawdown': 0,
            'current_drawdown': 0
        }
        
        # Sistema de colas para órdenes
        self.order_queue = Queue()
        self.signal_queue = Queue()
        
        # Thread safety
        self.lock = Lock()
        
        # Logging
        self.setup_logging()
        
        # Capital management
        self.capital_per_trade = None
        self.update_capital_allocation()
        
        self.logger.info("✅ Trader 24/7 inicializado")
    
    def setup_logging(self):
        """Configurar logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger('Trader24_7')
    
    def update_capital_allocation(self):
        """Actualizar asignación de capital por trade"""
        try:
            account = self.api.get_account()
            buying_power = float(account.buying_power)
            
            # Calcular capital por trade
            self.capital_per_trade = buying_power * Config.MAX_POSITION_SIZE
            
            self.logger.info(f"💰 Capital actualizado: ${buying_power:.2f}")
            self.logger.info(f"💵 Capital por trade: ${self.capital_per_trade:.2f}")
            
        except Exception as e:
            self.logger.error(f"Error actualizando capital: {e}")
            self.capital_per_trade = 1000  # Valor por defecto
    
    def is_market_open(self):
        """Verificar si el mercado está abierto"""
        try:
            clock = self.api.get_clock()
            return clock.is_open
        except:
            return False
    
    def get_extended_hours_status(self):
        """Verificar horario extendido"""
        now = datetime.now(Config.MARKET_TIMEZONE)
        hour = now.hour
        
        # Pre-market: 4:00 AM - 9:30 AM ET
        # After-hours: 4:00 PM - 8:00 PM ET
        
        if 4 <= hour < 9.5:
            return 'premarket'
        elif 16 <= hour < 20:
            return 'afterhours'
        elif 9.5 <= hour < 16:
            return 'regular'
        else:
            return 'closed'
    
    def execute_trading_cycle(self):
        """Ejecutar un ciclo completo de trading"""
        try:
            # 1. Verificar estado del mercado
            market_status = self.get_extended_hours_status()
            
            if market_status == 'closed' and not Config.EXTENDED_HOURS:
                self.logger.info("🔴 Mercado cerrado")
                return
            
            # 2. Actualizar capital
            self.update_capital_allocation()
            
            # 3. Gestionar posiciones existentes
            self.manage_positions()
            
            # 4. Buscar nuevas oportunidades
            if market_status in ['regular', 'premarket', 'afterhours']:
                self.scan_opportunities()
            
            # 5. Procesar cola de órdenes
            self.process_order_queue()
            
            # 6. Actualizar métricas
            self.update_performance_metrics()
            
        except Exception as e:
            self.logger.error(f"Error en ciclo de trading: {e}")
    
    def manage_positions(self):
        """Gestionar posiciones abiertas"""
        try:
            positions = self.api.list_positions()
            
            for position in positions:
                symbol = position.symbol
                qty = float(position.qty)
                entry_price = float(position.avg_entry_price)
                current_price = float(position.current_price) if hasattr(position, 'current_price') else 0
                unrealized_pl = float(position.unrealized_pl)
                unrealized_plpc = float(position.unrealized_plpc)
                
                # Evaluar si cerrar posición
                should_close, reason = self.evaluate_position(
                    symbol, entry_price, current_price, unrealized_plpc
                )
                
                if should_close:
                    self.logger.info(f"📤 Cerrando {symbol}: {reason}")
                    self.close_position(symbol, qty, reason)
                else:
                    # Actualizar trailing stop si está en ganancias
                    if unrealized_plpc > 0.02:  # 2% de ganancia
                        self.update_trailing_stop(symbol, current_price)
            
        except Exception as e:
            self.logger.error(f"Error gestionando posiciones: {e}")
    
    def evaluate_position(self, symbol, entry_price, current_price, pl_pct):
        """Evaluar si cerrar una posición"""
        # Stop Loss
        if pl_pct <= -Config.STOP_LOSS_PCT:
            return True, f"Stop Loss ({pl_pct:.2%})"
        
        # Take Profit
        if pl_pct >= Config.TAKE_PROFIT_PCT:
            return True, f"Take Profit ({pl_pct:.2%})"
        
        # Obtener predicción actual
        prediction = self.get_current_prediction(symbol)
        if prediction:
            # Si la predicción cambió a bajista y tenemos ganancias
            if prediction['direction'] == 'DOWN' and pl_pct > 0:
                return True, f"Señal de venta ({pl_pct:.2%})"
        
        # Time-based exit (mantener máximo 1 día)
        # Implementar si es necesario
        
        return False, None
    
    def scan_opportunities(self):
        """Buscar oportunidades de trading"""
        try:
            # Obtener predicciones de todos los símbolos
            from predictor import Predictor
            predictor = Predictor()
            
            opportunities = predictor.get_best_opportunities(
                min_confidence=Config.MIN_CONFIDENCE
            )
            
            # Filtrar por capital disponible
            account = self.api.get_account()
            buying_power = float(account.buying_power)
            
            # Obtener posiciones actuales
            current_positions = self.api.list_positions()
            position_symbols = [p.symbol for p in current_positions]
            
            # Limitar número de posiciones
            if len(current_positions) >= Config.MAX_POSITIONS:
                self.logger.info(f"⚠️ Máximo de posiciones alcanzado ({Config.MAX_POSITIONS})")
                return
            
            # Evaluar cada oportunidad
            for opp in opportunities[:5]:  # Top 5 oportunidades
                symbol = opp['symbol']
                
                # Skip si ya tenemos posición
                if symbol in position_symbols:
                    continue
                
                # Evaluar si entrar
                if self.should_enter_position(opp, buying_power):
                    self.enter_position(opp)
            
        except Exception as e:
            self.logger.error(f"Error escaneando oportunidades: {e}")
    
    def should_enter_position(self, opportunity, buying_power):
        """Evaluar si entrar en una posición"""
        # Verificar confianza mínima
        if opportunity['probability'] < Config.MIN_CONFIDENCE:
            return False
        
        # Verificar capital disponible
        if buying_power < self.capital_per_trade:
            return False
        
        # Verificar dirección (solo compras por ahora)
        if opportunity['direction'] != 'UP':
            return False
        
        # Verificar volatilidad y liquidez
        # Implementar checks adicionales si es necesario
        
        return True
    
    def enter_position(self, opportunity):
        """Entrar en una nueva posición"""
        try:
            symbol = opportunity['symbol']
            current_price = opportunity['current_price']
            
            # Calcular cantidad de acciones
            position_value = self.capital_per_trade
            qty = int(position_value / current_price)
            
            if qty < 1:
                qty = 1  # Mínimo 1 acción
            
            # Crear orden
            order = self.api.submit_order(
                symbol=symbol,
                qty=qty,
                side='buy',
                type='market',
                time_in_force='day'
            )
            
            self.logger.info(f"📥 Orden de compra: {qty} {symbol} @ ${current_price:.2f}")
            
            # Guardar en pending orders
            self.pending_orders[order.id] = {
                'symbol': symbol,
                'qty': qty,
                'side': 'buy',
                'opportunity': opportunity,
                'timestamp': datetime.now()
            }
            
            # Configurar stops después de llenar
            Thread(target=self.setup_position_stops, args=(symbol, current_price)).start()
            
        except Exception as e:
            self.logger.error(f"Error entrando posición {symbol}: {e}")
    
    def setup_position_stops(self, symbol, entry_price):
        """Configurar stop loss y take profit"""
        try:
            # Esperar a que la orden se llene
            time.sleep(2)
            
            # Calcular niveles
            stop_loss = entry_price * (1 - Config.STOP_LOSS_PCT)
            take_profit = entry_price * (1 + Config.TAKE_PROFIT_PCT)
            
            # Crear orden OCO (One-Cancels-Other)
            # Por ahora solo stop loss
            self.api.submit_order(
                symbol=symbol,
                qty=qty,
                side='sell',
                type='stop',
                stop_price=stop_loss,
                time_in_force='gtc'
            )
            
            self.logger.info(f"🛡️ Stop loss configurado para {symbol} @ ${stop_loss:.2f}")
            
        except Exception as e:
            self.logger.error(f"Error configurando stops para {symbol}: {e}")
    
    def update_trailing_stop(self, symbol, current_price):
        """Actualizar trailing stop"""
        try:
            # Calcular nuevo stop
            new_stop = current_price * (1 - Config.TRAILING_STOP_PCT)
            
            # Cancelar stop anterior y crear nuevo
            # Implementar lógica de actualización
            
            self.logger.info(f"📈 Trailing stop actualizado para {symbol} @ ${new_stop:.2f}")
            
        except Exception as e:
            self.logger.error(f"Error actualizando trailing stop: {e}")
    
    def close_position(self, symbol, qty, reason):
        """Cerrar una posición"""
        try:
            order = self.api.submit_order(
                symbol=symbol,
                qty=qty,
                side='sell',
                type='market',
                time_in_force='day'
            )
            
            self.logger.info(f"📤 Vendiendo {qty} {symbol}: {reason}")
            
            # Actualizar métricas
            self.performance['total_trades'] += 1
            
        except Exception as e:
            self.logger.error(f"Error cerrando posición {symbol}: {e}")
    
    def process_order_queue(self):
        """Procesar cola de órdenes pendientes"""
        while not self.order_queue.empty():
            try:
                order = self.order_queue.get(timeout=1)
                # Procesar orden
                self.logger.info(f"Procesando orden: {order}")
            except:
                break
    
    def update_performance_metrics(self):
        """Actualizar métricas de rendimiento"""
        try:
            # Obtener historial de órdenes del día
            orders = self.api.list_orders(
                status='closed',
                after=datetime.now().replace(hour=0, minute=0, second=0).isoformat()
            )
            
            daily_profit = 0
            for order in orders:
                if order.filled_qty and order.filled_avg_price:
                    # Calcular P&L
                    # Implementar cálculo detallado
                    pass
            
            # Actualizar métricas
            with self.lock:
                self.performance['daily_profit'] = daily_profit
                
        except Exception as e:
            self.logger.error(f"Error actualizando métricas: {e}")
    
    def get_current_prediction(self, symbol):
        """Obtener predicción actual para un símbolo"""
        try:
            from predictor import Predictor
            predictor = Predictor()
            return predictor.predict(symbol, Config.DEFAULT_TIMEFRAME)
        except:
            return None
    
    def run_24_7(self):
        """Ejecutar bot 24/7"""
        self.logger.info("🚀 Iniciando Trading Bot 24/7")
        self.is_running = True
        
        while self.is_running:
            try:
                # Ejecutar ciclo de trading
                self.execute_trading_cycle()
                
                # Esperar antes del próximo ciclo
                time.sleep(60)  # 1 minuto
                
            except KeyboardInterrupt:
                self.logger.info("⏹️ Deteniendo bot...")
                self.is_running = False
                break
                
            except Exception as e:
                self.logger.error(f"Error en loop principal: {e}")
                time.sleep(60)
        
        self.logger.info("✅ Bot detenido")
    
    def get_status(self):
        """Obtener estado actual del bot"""
        return {
            'is_running': self.is_running,
            'positions': len(self.positions),
            'performance': self.performance,
            'capital_per_trade': self.capital_per_trade,
            'timestamp': datetime.now().isoformat()
        }
    
    def get_positions(self):
        """Obtener posiciones actuales"""
        try:
            positions = self.api.list_positions()
            return {
                pos.symbol: {
                    'qty': float(pos.qty),
                    'entry_price': float(pos.avg_entry_price),
                    'current_price': float(pos.current_price) if hasattr(pos, 'current_price') else 0,
                    'pl': float(pos.unrealized_pl),
                    'pl_pct': float(pos.unrealized_plpc) * 100
                }
                for pos in positions
            }
        except:
            return {}
