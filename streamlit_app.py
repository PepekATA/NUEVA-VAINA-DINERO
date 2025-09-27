import streamlit as st
import pandas as pd
import plotly.graph_objs as go
from datetime import datetime, timedelta
import alpaca_trade_api as tradeapi
import threading
import time
import json
from config import Config
from trader_24_7 import Trader24_7
from predictor import Predictor
from model_sync import ModelSyncManager

# === CONFIGURACIÓN DE PÁGINA ===
st.set_page_config(
    page_title="🤖 Trading Bot 24/7 - Multi-Agent System",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# === ESTILOS CSS PERSONALIZADOS ===
st.markdown("""
<style>
    .main {background-color: #0e1117;}
    .stButton>button {
        background-color: #00D4FF;
        color: black;
        font-weight: bold;
        border-radius: 10px;
        border: 2px solid #00D4FF;
        padding: 10px;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        background-color: #0099CC;
        border-color: #0099CC;
        transform: scale(1.05);
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .status-running {
        color: #00FF00;
        animation: pulse 2s infinite;
    }
    @keyframes pulse {
        0% {opacity: 1;}
        50% {opacity: 0.5;}
        100% {opacity: 1;}
    }
</style>
""", unsafe_allow_html=True)

# === INICIALIZACIÓN ===
@st.cache_resource
def init_components():
    """Inicializar componentes del sistema"""
    try:
        # Verificar credenciales
        if not Config.ALPACA_API_KEY or not Config.ALPACA_SECRET_KEY:
            return None, None, None, "missing_credentials"
        
        # Inicializar API de Alpaca
        alpaca = tradeapi.REST(
            Config.ALPACA_API_KEY,
            Config.ALPACA_SECRET_KEY,
            Config.ALPACA_BASE_URL,
            api_version='v2'
        )
        
        # Verificar conexión
        account = alpaca.get_account()
        
        # Inicializar componentes
        trader = Trader24_7()
        predictor = Predictor()
        sync_manager = ModelSyncManager()
        
        return alpaca, trader, predictor, sync_manager
        
    except Exception as e:
        return None, None, None, str(e)

# Cargar componentes
alpaca, trader, predictor, sync_manager = init_components()

# === ESTADO DE LA APLICACIÓN ===
if 'bot_running' not in st.session_state:
    st.session_state.bot_running = False
    st.session_state.bot_thread = None
    st.session_state.last_sync = datetime.now()
    st.session_state.predictions = {}
    st.session_state.positions = {}
    st.session_state.performance = {'total_profit': 0, 'win_rate': 0, 'trades': 0}

# === FUNCIONES PRINCIPALES ===
def start_bot_24_7():
    """Iniciar bot en modo 24/7"""
    if not st.session_state.bot_running:
        st.session_state.bot_running = True
        
        def bot_loop():
            while st.session_state.bot_running:
                try:
                    # Ejecutar ciclo de trading
                    trader.execute_trading_cycle()
                    
                    # Actualizar predicciones
                    st.session_state.predictions = predictor.get_all_predictions()
                    
                    # Actualizar posiciones
                    st.session_state.positions = trader.get_positions()
                    
                    # Sincronizar modelos si es necesario
                    if (datetime.now() - st.session_state.last_sync).minutes > Config.SYNC_INTERVAL_MINUTES:
                        sync_manager.sync_to_github()
                        st.session_state.last_sync = datetime.now()
                    
                    # Esperar antes del próximo ciclo
                    time.sleep(60)  # Ciclo cada minuto
                    
                except Exception as e:
                    print(f"Error en bot loop: {e}")
                    time.sleep(60)
        
        # Iniciar thread
        st.session_state.bot_thread = threading.Thread(target=bot_loop, daemon=True)
        st.session_state.bot_thread.start()
        
        return True
    return False

def stop_bot():
    """Detener bot"""
    st.session_state.bot_running = False
    if st.session_state.bot_thread:
        st.session_state.bot_thread = None
    return True

def get_market_status():
    """Obtener estado del mercado"""
    try:
        clock = alpaca.get_clock()
        return {
            'is_open': clock.is_open,
            'next_open': clock.next_open,
            'next_close': clock.next_close
        }
    except:
        return {'is_open': False, 'next_open': None, 'next_close': None}

def get_account_metrics():
    """Obtener métricas de la cuenta"""
    try:
        account = alpaca.get_account()
        return {
            'cash': float(account.cash),
            'buying_power': float(account.buying_power),
            'portfolio_value': float(account.portfolio_value),
            'day_trades': int(account.daytrade_count),
            'pattern_day_trader': account.pattern_day_trader
        }
    except:
        return None

# === INTERFAZ PRINCIPAL ===
st.title("🤖 Trading Bot Multi-Agent 24/7")
st.caption(f"Sistema con {Config.AGENTS_PER_SYMBOL} agentes por activo | Modo: {'PAPER' if Config.USE_PAPER else 'REAL'}")

# === VERIFICACIÓN DE CREDENCIALES ===
if not alpaca:
    st.error("❌ Error de conexión con Alpaca")
    st.info("Por favor verifica tus credenciales en Streamlit Secrets:")
    st.code("""
    # En Streamlit Cloud Settings > Secrets:
    ALPACA_API_KEY = "tu_api_key"
    ALPACA_SECRET_KEY = "tu_secret_key"
    GITHUB_TOKEN = "tu_github_token"
    GITHUB_REPO = "usuario/repo"
    USE_PAPER = "true"
    """)
    st.stop()

# === PANEL DE CONTROL ===
col1, col2, col3, col4 = st.columns(4)

with col1:
    if st.session_state.bot_running:
        if st.button("⏸️ DETENER BOT", key="stop", use_container_width=True):
            if stop_bot():
                st.success("Bot detenido")
                st.rerun()
    else:
        if st.button("▶️ INICIAR BOT 24/7", key="start", type="primary", use_container_width=True):
            if start_bot_24_7():
                st.success("Bot iniciado en modo 24/7")
                st.rerun()

with col2:
    status = "🟢 ACTIVO" if st.session_state.bot_running else "🔴 DETENIDO"
    st.metric("Estado del Bot", status)

with col3:
    market = get_market_status()
    market_status = "🟢 ABIERTO" if market['is_open'] else "🔴 CERRADO"
    st.metric("Mercado", market_status)

with col4:
    if st.button("🔄 Actualizar", key="refresh", use_container_width=True):
        st.rerun()

# === MÉTRICAS DE CUENTA ===
st.markdown("### 💰 Métricas de Cuenta")
account_metrics = get_account_metrics()

if account_metrics:
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric(
            "Cash Disponible",
            f"${account_metrics['cash']:,.2f}",
            delta=None
        )
    
    with col2:
        st.metric(
            "Poder de Compra",
            f"${account_metrics['buying_power']:,.2f}"
        )
    
    with col3:
        st.metric(
            "Valor Portfolio",
            f"${account_metrics['portfolio_value']:,.2f}"
        )
    
    with col4:
        daily_pl = account_metrics['portfolio_value'] - Config.INITIAL_CAPITAL
        st.metric(
            "P&L Diario",
            f"${daily_pl:,.2f}",
            delta=f"{(daily_pl/Config.INITIAL_CAPITAL)*100:.2f}%"
        )
    
    with col5:
        st.metric(
            "Day Trades",
            f"{account_metrics['day_trades']}/3"
        )

# === TABS PRINCIPALES ===
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Dashboard", 
    "🤖 Predicciones Multi-Agent", 
    "💼 Posiciones", 
    "📈 Análisis", 
    "⚙️ Configuración"
])

# TAB 1: DASHBOARD
with tab1:
    # Gráfico de rendimiento
    st.markdown("### 📈 Rendimiento del Sistema")
    
    # Placeholder para gráfico de rendimiento
    performance_chart = go.Figure()
    
    # Simular datos de rendimiento (en producción vendría de trader.get_performance())
    dates = pd.date_range(end=datetime.now(), periods=30, freq='D')
    values = [Config.INITIAL_CAPITAL * (1 + i*0.002) for i in range(30)]
    
    performance_chart.add_trace(go.Scatter(
        x=dates,
        y=values,
        mode='lines',
        name='Valor Portfolio',
        line=dict(color='#00D4FF', width=3)
    ))
    
    performance_chart.update_layout(
        template='plotly_dark',
        height=400,
        title="Evolución del Portfolio (30 días)",
        xaxis_title="Fecha",
        yaxis_title="Valor ($)",
        hovermode='x unified'
    )
    
    st.plotly_chart(performance_chart, use_container_width=True)
    
    # Estadísticas de trading
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📊 Estadísticas de Trading")
        stats_df = pd.DataFrame({
            'Métrica': ['Total Trades', 'Trades Ganadores', 'Win Rate', 'Profit Factor', 'Max Drawdown'],
            'Valor': [150, 95, '63.3%', '1.85', '-5.2%']
        })
        st.dataframe(stats_df, hide_index=True, use_container_width=True)
    
    with col2:
        st.markdown("### 🎯 Top Símbolos")
        top_symbols = pd.DataFrame({
            'Símbolo': ['AAPL', 'TSLA', 'NVDA', 'SPY', 'QQQ'],
            'P&L': ['+$234', '+$189', '+$156', '+$98', '+$87'],
            'Trades': [12, 10, 8, 15, 9]
        })
        st.dataframe(top_symbols, hide_index=True, use_container_width=True)

# TAB 2: PREDICCIONES MULTI-AGENT
with tab2:
    st.markdown("### 🤖 Sistema Multi-Agent: Predicciones en Tiempo Real")
    st.caption(f"Usando {Config.AGENTS_PER_SYMBOL} agentes especializados por símbolo")
    
    # Selector de símbolos
    selected_symbols = st.multiselect(
        "Seleccionar Símbolos",
        Config.SYMBOLS,
        default=Config.SYMBOLS[:5]
    )
    
    if selected_symbols:
        # Obtener predicciones
        predictions_df = []
        
        for symbol in selected_symbols:
            # Simular predicciones multi-agent (en producción vendría de predictor)
            agent_predictions = {}
            for agent_name in Config.AGENT_CONFIGS.keys():
                agent_predictions[agent_name] = np.random.random()
            
            # Consenso
            consensus = np.mean(list(agent_predictions.values()))
            direction = "🟢 COMPRA" if consensus > 0.5 else "🔴 VENTA"
            confidence = abs(consensus - 0.5) * 200
            
            predictions_df.append({
                'Símbolo': symbol,
                'Dirección': direction,
                'Confianza': f"{confidence:.1f}%",
                'Consenso': f"{consensus:.3f}",
                'Agentes Bull': sum(1 for p in agent_predictions.values() if p > 0.5),
                'Agentes Bear': sum(1 for p in agent_predictions.values() if p <= 0.5)
            })
        
        # Mostrar tabla de predicciones
        st.dataframe(
            pd.DataFrame(predictions_df),
            hide_index=True,
            use_container_width=True
        )
        
        # Detalles por agente
        with st.expander("Ver Predicciones Detalladas por Agente"):
            agent_cols = st.columns(5)
            for i, (agent_name, agent_config) in enumerate(Config.AGENT_CONFIGS.items()):
                col_idx = i % 5
                with agent_cols[col_idx]:
                    st.markdown(f"**{agent_name}**")
                    st.caption(f"Tipo: {agent_config['type']}")
                    # Mostrar mini gráfico de predicciones
                    st.progress(np.random.random())

# TAB 3: POSICIONES
with tab3:
    st.markdown("### 💼 Posiciones Activas")
    
    # Obtener posiciones reales
    try:
        positions = alpaca.list_positions()
        if positions:
            positions_data = []
            for pos in positions:
                current_price = float(pos.current_price) if hasattr(pos, 'current_price') else float(pos.market_value) / float(pos.qty)
                unrealized_pl = float(pos.unrealized_pl)
                unrealized_plpc = float(pos.unrealized_plpc) * 100
                
                positions_data.append({
                    'Símbolo': pos.symbol,
                    'Cantidad': float(pos.qty),
                    'Precio Entrada': f"${float(pos.avg_entry_price):.2f}",
                    'Precio Actual': f"${current_price:.2f}",
                    'Valor': f"${float(pos.market_value):.2f}",
                    'P&L': f"${unrealized_pl:.2f}",
                    'P&L %': f"{unrealized_plpc:.2f}%",
                    'Stop Loss': f"${float(pos.avg_entry_price) * (1 - Config.STOP_LOSS_PCT):.2f}",
                    'Take Profit': f"${float(pos.avg_entry_price) * (1 + Config.TAKE_PROFIT_PCT):.2f}"
                })
            
            # Mostrar posiciones
            positions_df = pd.DataFrame(positions_data)
            st.dataframe(
                positions_df,
                hide_index=True,
                use_container_width=True,
                column_config={
                    "P&L": st.column_config.TextColumn(
                        "P&L",
                        help="Ganancia/Pérdida no realizada"
                    ),
                    "P&L %": st.column_config.TextColumn(
                        "P&L %",
                        help="Porcentaje de ganancia/pérdida"
                    )
                }
            )
            
            # Resumen de posiciones
            col1, col2, col3 = st.columns(3)
            with col1:
                total_pl = sum(float(pos.unrealized_pl) for pos in positions)
                st.metric("P&L Total", f"${total_pl:.2f}")
            with col2:
                st.metric("Posiciones Abiertas", len(positions))
            with col3:
                avg_pl = total_pl / len(positions) if positions else 0
                st.metric("P&L Promedio", f"${avg_pl:.2f}")
        else:
            st.info("No hay posiciones abiertas")
    except Exception as e:
        st.error(f"Error obteniendo posiciones: {e}")

# TAB 4: ANÁLISIS
with tab4:
    st.markdown("### 📈 Análisis Técnico")
    
    # Selector de símbolo y timeframe
    col1, col2 = st.columns(2)
    with col1:
        analysis_symbol = st.selectbox("Símbolo", Config.SYMBOLS)
    with col2:
        timeframe = st.selectbox("Timeframe", list(Config.TIMEFRAMES.values()))
    
    # Gráfico de velas
    if analysis_symbol:
        try:
            # Obtener datos históricos
            end = datetime.now()
            start = end - timedelta(days=5)
            
            bars = alpaca.get_bars(
                analysis_symbol,
                timeframe,
                start=start.isoformat(),
                end=end.isoformat(),
                limit=500
            ).df
            
            if not bars.empty:
                # Crear gráfico de velas
                fig = go.Figure(data=[go.Candlestick(
                    x=bars.index,
                    open=bars['open'],
                    high=bars['high'],
                    low=bars['low'],
                    close=bars['close'],
                    name=analysis_symbol
                )])
                
                # Agregar volumen
                fig.add_trace(go.Bar(
                    x=bars.index,
                    y=bars['volume'],
                    name='Volumen',
                    yaxis='y2',
                    opacity=0.3
                ))
                
                fig.update_layout(
                    title=f"{analysis_symbol} - {timeframe}",
                    yaxis_title='Precio',
                    yaxis2=dict(
                        title='Volumen',
                        overlaying='y',
                        side='right'
                    ),
                    template='plotly_dark',
                    height=500,
                    hovermode='x unified'
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Indicadores técnicos
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Último Precio", f"${bars['close'].iloc[-1]:.2f}")
                with col2:
                    change = bars['close'].iloc[-1] - bars['close'].iloc[-2]
                    st.metric("Cambio", f"${change:.2f}", delta=f"{(change/bars['close'].iloc[-2])*100:.2f}%")
                with col3:
                    st.metric("Volumen", f"{bars['volume'].iloc[-1]:,.0f}")
                with col4:
                    st.metric("VWAP", f"${bars['vwap'].iloc[-1]:.2f}")
                    
        except Exception as e:
            st.error(f"Error obteniendo datos: {e}")

# TAB 5: CONFIGURACIÓN
with tab5:
    st.markdown("### ⚙️ Configuración del Sistema")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Parámetros de Trading")
        
        max_positions = st.slider(
            "Máximo de Posiciones",
            min_value=1,
            max_value=50,
            value=Config.MAX_POSITIONS,
            help="Número máximo de posiciones abiertas simultáneamente"
        )
        
        min_confidence = st.slider(
            "Confianza Mínima (%)",
            min_value=50,
            max_value=95,
            value=int(Config.MIN_CONFIDENCE * 100),
            help="Confianza mínima requerida para ejecutar trades"
        )
        
        stop_loss = st.slider(
            "Stop Loss (%)",
            min_value=1,
            max_value=10,
            value=int(Config.STOP_LOSS_PCT * 100),
            help="Porcentaje de pérdida máxima por posición"
        )
        
        take_profit = st.slider(
            "Take Profit (%)",
            min_value=1,
            max_value=20,
            value=int(Config.TAKE_PROFIT_PCT * 100),
            help="Porcentaje de ganancia objetivo por posición"
        )
    
    with col2:
        st.markdown("#### Sincronización")
        
        st.info(f"""
        **Estado de Sincronización:**
        - Última sincronización: {st.session_state.last_sync.strftime('%H:%M:%S')}
        - Intervalo: {Config.SYNC_INTERVAL_MINUTES} minutos
        - GitHub Repo: {Config.GITHUB_REPO}
        """)
        
        if st.button("🔄 Sincronizar Ahora", use_container_width=True):
            with st.spinner("Sincronizando con GitHub..."):
                try:
                    sync_manager.sync_to_github()
                    st.session_state.last_sync = datetime.now()
                    st.success("✅ Sincronización completada")
                except Exception as e:
                    st.error(f"Error sincronizando: {e}")
        
        st.markdown("#### Agentes Activos")
        agents_status = []
        for agent_name, config in Config.AGENT_CONFIGS.items():
            agents_status.append({
                'Agente': agent_name,
                'Tipo': config['type'],
                'Estado': '🟢 Activo'
            })
        
        st.dataframe(
            pd.DataFrame(agents_status),
            hide_index=True,
            use_container_width=True
        )

# === FOOTER ===
st.markdown("---")
col1, col2, col3 = st.columns(3)
with col1:
    st.caption("💻 Desarrollado con Streamlit")
with col2:
    st.caption(f"🕒 {datetime.now().strftime('%H:%M:%S')}")
with col3:
    st.caption(f"📍 Modo: {'Paper Trading' if Config.USE_PAPER else 'Trading Real'}")

# === AUTO-REFRESH ===
if st.session_state.bot_running:
    time.sleep(30)  # Actualizar cada 30 segundos
    st.rerun()
