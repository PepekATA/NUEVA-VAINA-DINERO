from flask import Flask, render_template, jsonify, request
import plotly.graph_objs as go
import plotly.utils
import json
import threading
from datetime import datetime, timedelta
import pandas as pd
from config import Config
from predictor import Predictor
from trader import Trader
from data_fetcher import DataFetcher

app = Flask(__name__)
predictor = Predictor()
trader = Trader()
data_fetcher = DataFetcher()

# Global variables for bot state
bot_running = False
bot_mode = 'stopped'
bot_thread = None

@app.route('/')
def index():
    return render_template('index.html', symbols=Config.SYMBOLS)

@app.route('/start_bot', methods=['POST'])
def start_bot():
    global bot_running, bot_mode, bot_thread
    
    if not bot_running:
        bot_mode = 'auto'
        bot_running = True
        bot_thread = threading.Thread(
            target=trader.run_trading_bot,
            args=('auto', 60)
        )
        bot_thread.daemon = True
        bot_thread.start()
        return jsonify({'status': 'success', 'message': 'Bot started in auto mode'})
    else:
        return jsonify({'status': 'error', 'message': 'Bot already running'})

@app.route('/predict_only', methods=['POST'])
def predict_only():
    global bot_running, bot_mode, bot_thread
    
    if not bot_running:
        bot_mode = 'predict_only'
        bot_running = True
        bot_thread = threading.Thread(
            target=trader.run_trading_bot,
            args=('predict_only', 60)
        )
        bot_thread.daemon = True
        bot_thread.start()
        return jsonify({'status': 'success', 'message': 'Bot started in predict mode'})
    else:
        return jsonify({'status': 'error', 'message': 'Bot already running'})

@app.route('/stop_bot', methods=['POST'])
def stop_bot():
    global bot_running, bot_mode
    
    bot_running = False
    bot_mode = 'stopped'
    return jsonify({'status': 'success', 'message': 'Bot stopped'})

@app.route('/get_predictions', methods=['POST'])
def get_predictions():
    data = request.json
    symbols = data.get('symbols', Config.SYMBOLS)
    timeframe = data.get('timeframe', '1Min')
    
    predictions = []
    for symbol in symbols:
        pred = predictor.predict(symbol, timeframe)
        if pred:
            predictions.append(pred)
            
    return jsonify(predictions)

@app.route('/get_market_status')
def get_market_status():
    market_hours = data_fetcher.get_market_hours()
    return jsonify(market_hours)

@app.route('/get_account_info')
def get_account_info():
    account = trader.get_account_info()
    positions = trader.get_positions()
    
    return jsonify({
        'account': account,
        'positions': positions,
        'bot_status': bot_mode
    })

@app.route('/get_chart_data/<symbol>/<timeframe>')
def get_chart_data(symbol, timeframe):
    df = data_fetcher.fetch_realtime_data(symbol, timeframe, limit=100)
    
    if df.empty:
        return jsonify({'error': 'No data available'})
        
    # Create candlestick chart
    trace = go.Candlestick(
        x=df.index,
        open=df['open'],
        high=df['high'],
        low=df['low'],
        close=df['close'],
        name=symbol
    )
    
    data = [trace]
    layout = go.Layout(
        title=f'{symbol} - {timeframe}',
        xaxis={'title': 'Time'},
        yaxis={'title': 'Price'},
        template='plotly_dark'
    )
    
    fig = go.Figure(data=data, layout=layout)
    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    
    return graphJSON

@app.route('/get_prediction_chart')
def get_prediction_chart():
    opportunities = predictor.get_best_opportunities()
    
    up_symbols = []
    up_probs = []
    down_symbols = []
    down_probs = []
    
    for opp in opportunities:
        if opp['direction'] == 'UP':
            up_symbols.append(opp['symbol'])
            up_probs.append(opp['probability'] * 100)
        else:
            down_symbols.append(opp['symbol'])
            down_probs.append(opp['probability'] * 100)
            
    # Create bar charts
    trace1 = go.Bar(
        x=up_symbols,
        y=up_probs,
        name='Bullish',
        marker=dict(color='green')
    )
    
    trace2 = go.Bar(
        x=down_symbols,
        y=down_probs,
        name='Bearish',
        marker=dict(color='red')
    )
    
    data = [trace1, trace2]
    layout = go.Layout(
        title='Prediction Probabilities',
        xaxis={'title': 'Symbol'},
        yaxis={'title': 'Probability (%)'},
        template='plotly_dark'
    )
    
    fig = go.Figure(data=data, layout=layout)
    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    
    return graphJSON

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5000)
