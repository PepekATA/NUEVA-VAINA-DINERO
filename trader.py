import alpaca_trade_api as tradeapi
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import logging
from config import Config
from predictor import Predictor

class Trader:
    def __init__(self):
        self.api = tradeapi.REST(
            Config.ALPACA_API_KEY,
            Config.ALPACA_SECRET_KEY,
            Config.ALPACA_BASE_URL,
            api_version='v2'
        )
        self.predictor = Predictor()
        self.positions = {}
        self.orders = {}
        self.capital = Config.INITIAL_CAPITAL
        self.setup_logging()
        
    def setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(f'{Config.LOGS_DIR}/trader.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def get_account_info(self):
        """Get account information"""
        try:
            account = self.api.get_account()
            return {
                'buying_power': float(account.buying_power),
                'cash': float(account.cash),
                'portfolio_value': float(account.portfolio_value),
                'pattern_day_trader': account.pattern_day_trader,
                'trading_blocked': account.trading_blocked,
                'account_blocked': account.account_blocked
            }
        except Exception as e:
            self.logger.error(f"Error getting account info: {str(e)}")
            return None
            
    def get_positions(self):
        """Get current positions"""
        try:
            positions = self.api.list_positions()
            pos_dict = {}
            
            for pos in positions:
                pos_dict[pos.symbol] = {
                    'qty': float(pos.qty),
                    'avg_entry_price': float(pos.avg_entry_price),
                    'current_price': float(pos.current_price
                     'market_value': float(pos.market_value),
                    'unrealized_pl': float(pos.unrealized_pl),
                    'unrealized_plpc': float(pos.unrealized_plpc),
                    'side': pos.side
                }
                
            return pos_dict
            
        except Exception as e:
            self.logger.error(f"Error getting positions: {str(e)}")
            return {}
            
    def calculate_position_size(self, symbol, price, confidence):
        """Calculate position size based on confidence and capital"""
        try:
            account = self.get_account_info()
            if not account:
                return 0
                
            available_capital = min(float(account['buying_power']), self.capital)
            
            # Risk management: use Kelly Criterion
            kelly_fraction = (confidence - 0.5) / 0.5
            position_pct = min(kelly_fraction * 0.25, 0.2)  # Max 20% per position
            
            position_value = available_capital * position_pct
            shares = position_value / price
            
            # Round to 2 decimal places for fractional shares
            shares = round(shares, 2)
            
            self.logger.info(f"Position size for {symbol}: {shares} shares "
                           f"(${position_value:.2f})")
            
            return shares
            
        except Exception as e:
            self.logger.error(f"Error calculating position size: {str(e)}")
            return 0
            
    def place_order(self, symbol, qty, side, order_type='market', 
                   limit_price=None, stop_price=None):
        """Place an order"""
        try:
            if qty <= 0:
                self.logger.warning(f"Invalid quantity {qty} for {symbol}")
                return None
                
            order_params = {
                'symbol': symbol,
                'qty': qty,
                'side': side,
                'type': order_type,
                'time_in_force': 'day'
            }
            
            if order_type == 'limit' and limit_price:
                order_params['limit_price'] = limit_price
            elif order_type == 'stop' and stop_price:
                order_params['stop_price'] = stop_price
            elif order_type == 'stop_limit' and limit_price and stop_price:
                order_params['limit_price'] = limit_price
                order_params['stop_price'] = stop_price
                
            order = self.api.submit_order(**order_params)
            
            self.orders[order.id] = {
                'symbol': symbol,
                'qty': qty,
                'side': side,
                'type': order_type,
                'status': order.status,
                'created_at': order.created_at,
                'filled_at': order.filled_at,
                'filled_qty': order.filled_qty,
                'filled_avg_price': order.filled_avg_price
            }
            
            self.logger.info(f"Order placed: {side} {qty} {symbol} @ {order_type}")
            
            return order
            
        except Exception as e:
            self.logger.error(f"Error placing order: {str(e)}")
            return None
            
    def close_position(self, symbol, qty=None):
        """Close a position"""
        try:
            if qty:
                order = self.api.submit_order(
                    symbol=symbol,
                    qty=qty,
                    side='sell',
                    type='market',
                    time_in_force='day'
                )
            else:
                order = self.api.close_position(symbol)
                
            self.logger.info(f"Position closed: {symbol}")
            return order
            
        except Exception as e:
            self.logger.error(f"Error closing position: {str(e)}")
            return None
            
    def set_stop_loss_take_profit(self, symbol, entry_price):
        """Set stop loss and take profit orders"""
        try:
            positions = self.get_positions()
            if symbol not in positions:
                return None
                
            qty = positions[symbol]['qty']
            
            # Calculate stop loss and take profit prices
            stop_loss_price = round(entry_price * (1 - Config.STOP_LOSS_PCT), 2)
            take_profit_price = round(entry_price * (1 + Config.TAKE_PROFIT_PCT), 2)
            
            # Place stop loss order
            stop_loss_order = self.place_order(
                symbol=symbol,
                qty=qty,
                side='sell',
                order_type='stop',
                stop_price=stop_loss_price
            )
            
            # Place take profit order
            take_profit_order = self.place_order(
                symbol=symbol,
                qty=qty,
                side='sell',
                order_type='limit',
                limit_price=take_profit_price
            )
            
            self.logger.info(f"Stop loss: ${stop_loss_price}, "
                           f"Take profit: ${take_profit_price}")
            
            return {
                'stop_loss': stop_loss_order,
                'take_profit': take_profit_order
            }
            
        except Exception as e:
            self.logger.error(f"Error setting SL/TP: {str(e)}")
            return None
            
    def execute_trade(self, prediction):
        """Execute trade based on prediction"""
        try:
            symbol = prediction['symbol']
            direction = prediction['direction']
            probability = prediction['probability']
            current_price = prediction['current_price']
            
            # Check if we already have a position
            positions = self.get_positions()
            if symbol in positions:
                position = positions[symbol]
                unrealized_plpc = position['unrealized_plpc']
                
                # Only sell if profitable
                if direction == 'DOWN' and unrealized_plpc > 0:
                    self.logger.info(f"Selling {symbol} with {unrealized_plpc:.2%} profit")
                    return self.close_position(symbol)
                else:
                    self.logger.info(f"Holding {symbol} - PL: {unrealized_plpc:.2%}")
                    return None
                    
            # Only buy if prediction is UP with high confidence
            if direction == 'UP' and probability >= Config.PREDICTION_THRESHOLD:
                # Check if we have room for more positions
                if len(positions) >= Config.MAX_POSITIONS:
                    self.logger.warning("Max positions reached")
                    return None
                    
                # Calculate position size
                qty = self.calculate_position_size(symbol, current_price, probability)
                
                if qty > 0:
                    # Place buy order
                    order = self.place_order(
                        symbol=symbol,
                        qty=qty,
                        side='buy',
                        order_type='market'
                    )
                    
                    if order:
                        # Wait for fill
                        time.sleep(2)
                        
                        # Set stop loss and take profit
                        self.set_stop_loss_take_profit(symbol, current_price)
                        
                        return order
                        
            return None
            
        except Exception as e:
            self.logger.error(f"Error executing trade: {str(e)}")
            return None
            
    def manage_positions(self):
        """Manage existing positions"""
        try:
            positions = self.get_positions()
            
            for symbol, position in positions.items():
                unrealized_plpc = position['unrealized_plpc']
                
                # Get latest prediction
                prediction = self.predictor.predict(symbol, '1Min')
                
                if prediction:
                    # Exit if prediction changed to DOWN and we're profitable
                    if prediction['direction'] == 'DOWN' and unrealized_plpc > 0:
                        self.logger.info(f"Exit signal for {symbol}")
                        self.close_position(symbol)
                        
                    # Trail stop loss if profitable
                    elif unrealized_plpc > Config.TAKE_PROFIT_PCT:
                        current_price = position['current_price']
                        new_stop = current_price * (1 - Config.STOP_LOSS_PCT)
                        self.logger.info(f"Trailing stop for {symbol} to ${new_stop:.2f}")
                        
        except Exception as e:
            self.logger.error(f"Error managing positions: {str(e)}")
            
    def run_trading_bot(self, mode='auto', interval_seconds=60):
        """Main trading bot loop"""
        self.logger.info(f"Starting trading bot in {mode} mode")
        
        while True:
            try:
                # Check if market is open
                if not self.predictor.data_fetcher.is_market_open():
                    self.logger.info("Market closed. Waiting...")
                    
                    # Close all positions before market close
                    positions = self.get_positions()
                    if positions:
                        self.logger.info("Closing all positions before market close")
                        for symbol in positions.keys():
                            self.close_position(symbol)
                            
                    time.sleep(300)  # Wait 5 minutes
                    continue
                    
                # Get predictions
                opportunities = self.predictor.get_best_opportunities()
                
                if mode == 'auto':
                    # Execute trades
                    for opp in opportunities:
                        self.execute_trade(opp)
                        
                    # Manage existing positions
                    self.manage_positions()
                    
                elif mode == 'predict_only':
                    # Just show predictions
                    for opp in opportunities:
                        self.logger.info(
                            f"Prediction: {opp['symbol']} - {opp['direction']} - "
                            f"Prob: {opp['probability']:.2%} - "
                            f"Duration: {opp['duration_minutes']} min"
                        )
                        
                # Show account status
                account = self.get_account_info()
                if account:
                    self.logger.info(
                        f"Account - Cash: ${account['cash']:.2f}, "
                        f"Portfolio: ${account['portfolio_value']:.2f}"
                    )
                    
                # Wait before next iteration
                time.sleep(interval_seconds)
                
            except KeyboardInterrupt:
                self.logger.info("Stopping trading bot")
                break
            except Exception as e:
                self.logger.error(f"Error in trading loop: {str(e)}")
                time.sleep(60)
                
    def get_trade_history(self, days=30):
        """Get trade history"""
        try:
            end = datetime.now()
            start = end - timedelta(days=days)
            
            orders = self.api.list_orders(
                status='all',
                after=start.isoformat(),
                until=end.isoformat(),
                limit=500,
                direction='desc'
            )
            
            history = []
            for order in orders:
                history.append({
                    'symbol': order.symbol,
                    'qty': float(order.qty),
                    'side': order.side,
                    'type': order.type,
                    'status': order.status,
                    'filled_at': order.filled_at,
                    'filled_avg_price': float(order.filled_avg_price) if order.filled_avg_price else None
                })
                
            df = pd.DataFrame(history)
            filename = f"{Config.DATA_DIR}/trade_history_{datetime.now().strftime('%Y%m%d')}.csv"
            df.to_csv(filename, index=False)
            
            self.logger.info(f"Trade history saved to {filename}")
            return df
            
        except Exception as e:
            self.logger.error(f"Error getting trade history: {str(e)}")
            return pd.DataFrame()                      
