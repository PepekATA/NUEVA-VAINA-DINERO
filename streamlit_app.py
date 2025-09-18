import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from datetime import datetime
import requests
import json
import base64
from github import Github
import joblib
import os

st.set_page_config(
    page_title="Trading Bot Dashboard",
    page_icon="🤖",
    layout="wide"
)

# CSS
st.markdown("""
    <style>
    .main {background-color: #1a1a1a;}
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 5px;
    }
    </style>
""", unsafe_allow_html=True)

class LightPredictor:
    """Versión ligera del predictor para Streamlit"""
    def __init__(self):
        self.models = {}
        self.load_models()
    
    def load_models(self):
        """Cargar solo modelos ligeros (pkl)"""
        if os.path.exists('models'):
            for file in os.listdir('models'):
                if file.endswith('.pkl'):
                    try:
                        self.models[file] = joblib.load(f'models/{file}')
                    except:
                        pass
    
    def predict(self, symbol, timeframe='5Min'):
        """Predicción simplificada"""
        # Generar predicción aleatoria para demo
        # En producción, usar modelos reales
        prob = np.random.random()
        return {
            'symbol': symbol,
            'timeframe': timeframe,
            'direction': 'UP' if prob > 0.5 else 'DOWN',
            'probability': prob,
            'confidence': abs(prob - 0.5) * 2,
            'current_price': 100 + np.random.randn() * 10,
            'predicted_change_pct': (prob - 0.5) * 10
        }

@st.cache_resource
def init_github():
    """Inicializar conexión con GitHub"""
    try:
        token = st.secrets.get("GITHUB_TOKEN", os.getenv("GITHUB_TOKEN"))
        if token:
            g = Github(token)
            repo_name = st.secrets.get("GITHUB_REPO", "username/trading-bot")
            return g.get_repo(repo_name)
    except:
        return None
    return None

@st.cache_data(ttl=300)
def download_models():
    """Descargar modelos desde GitHub"""
    repo = init_github()
    if not repo:
        return False
    
    try:
        # Buscar archivo de registro
        contents = repo.get_contents("models/registry.json")
        registry = json.loads(base64.b64decode(contents.content).decode())
        
        st.sidebar.success(f"📊 {len(registry.get('models', {}))} modelos disponibles")
        return True
    except:
        st.sidebar.warning("⚠️ No se pudieron cargar modelos")
        return False

# Inicializar
predictor = LightPredictor()

# UI Principal
st.title("🤖 Trading Bot Dashboard")

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuración")
    
    # Estado de modelos
    if st.button("🔄 Actualizar Modelos"):
        download_models()
        st.rerun()
    
    # Selección de activos
    symbols = st.multiselect(
        "Activos",
        ['AAPL', 'TSLA', 'NVDA', 'SPY', 'QQQ', 'AMZN', 'MSFT', 'META'],
        default=['AAPL', 'TSLA', 'NVDA']
    )
    
    timeframe = st.selectbox(
        "Timeframe",
        ['1Min', '5Min', '15Min', '30Min'],
        index=1
    )
    
    mode = st.radio(
        "Modo",
        ['Predicción', 'Trading Demo']
    )

# Layout principal
col1, col2, col3 = st.columns([1, 2, 1])

with col1:
    st.subheader("📊 Estado")
    
    # Simular estado del mercado
    hour = datetime.now().hour
    if 9 <= hour < 16:
        st.success("🟢 Mercado Abierto")
    else:
        st.error("🔴 Mercado Cerrado")
    
    st.metric("Modelos Cargados", len(predictor.models))

with col2:
    st.subheader("🎯 Predicciones")
    
    # Generar predicciones
    predictions = []
    for symbol in symbols:
        pred = predictor.predict(symbol, timeframe)
        predictions.append(pred)
    
    # Mostrar tabla
    if predictions:
        df = pd.DataFrame(predictions)
        df['Prob %'] = df['probability'].apply(lambda x: f"{x:.1%}")
        df['Precio'] = df['current_price'].apply(lambda x: f"${x:.2f}")
        
        # Colorear
        def color_direction(val):
            color = 'green' if val == 'UP' else 'red'
            return f'color: {color}'
        
        styled_df = df[['symbol', 'direction', 'Prob %', 'Precio']].style.applymap(
            color_direction, 
            subset=['direction']
        )
        
        st.dataframe(styled_df, use_container_width=True)

with col3:
    st.subheader("💰 Portfolio")
    st.metric("Balance", "$100,000", "+2.5%")
    st.metric("P&L Diario", "+$2,500", "+2.5%")

# Gráficos
st.subheader("📈 Visualizaciones")

col1, col2 = st.columns(2)

with col1:
    # Gráfico de probabilidades
    if predictions:
        fig = go.Figure([
            go.Bar(
                x=[p['symbol'] for p in predictions],
                y=[p['probability'] for p in predictions],
                marker_color=['green' if p['direction']=='UP' else 'red' for p in predictions]
            )
        ])
        fig.update_layout(
            title="Probabilidades",
            template="plotly_dark",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)

with col2:
    # Gráfico de cambios esperados
    if predictions:
        fig = go.Figure([
            go.Scatter(
                x=[p['symbol'] for p in predictions],
                y=[p['predicted_change_pct'] for p in predictions],
                mode='markers+lines',
                marker=dict(
                    size=15,
                    color=[p['probability'] for p in predictions],
                    colorscale='RdYlGn',
                    showscale=True
                )
            )
        ])
        fig.update_layout(
            title="Cambio Esperado %",
            template="plotly_dark",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)

# Historial de trades (simulado)
st.subheader("📜 Historial de Trades")

trades = pd.DataFrame({
    'Hora': pd.date_range(start='today', periods=5, freq='H'),
    'Símbolo': np.random.choice(symbols[:3] if symbols else ['AAPL'], 5),
    'Tipo': np.random.choice(['BUY', 'SELL'], 5),
    'Precio': np.random.uniform(95, 105, 5),
    'Cantidad': np.random.randint(10, 100, 5),
    'P&L': np.random.uniform(-100, 200, 5)
})

trades['P&L'] = trades['P&L'].apply(lambda x: f"${x:.2f}")
trades['Precio'] = trades['Precio'].apply(lambda x: f"${x:.2f}")

st.dataframe(
    trades.style.applymap(
        lambda x: 'color: green' if 'BUY' in str(x) else 'color: red' if 'SELL' in str(x) else '',
        subset=['Tipo']
    ),
    use_container_width=True
)

# Footer
st.markdown("---")
st.caption(f"🔄 Última actualización: {datetime.now().strftime('%H:%M:%S')}")

# Auto-refresh
if st.checkbox("Auto-actualizar (30s)"):
    import time
    time.sleep(30)
    st.rerun()
