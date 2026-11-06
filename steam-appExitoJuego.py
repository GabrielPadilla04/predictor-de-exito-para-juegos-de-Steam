import streamlit as st
import pandas as pd
import joblib

# Configuración de la página
st.set_page_config(
    page_title="Predictor de Éxito en Steam",
    page_icon="🎮",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Estilos CSS personalizados
st.markdown("""
    <style>
        .stButton>button {
            width: 100%;
            background-color: #ff4b4b;
            color: white;
            font-weight: bold;
            border-radius: 8px;
            padding: 0.6rem;
            border: none;
        }
        .stButton>button:hover {
            background-color: #ff2b2b;
            color: white;
        }
    </style>
""", unsafe_allow_html=True)

# Título principal
st.title("🎮 Predictor de Éxito de Juegos en Steam")
st.markdown("Herramienta avanzada de análisis predictivo para evaluar la viabilidad comercial de un videojuego en la plataforma.")
st.divider()

# Cargar el modelo con caché
@st.cache_resource
def load_model():
    model = joblib.load('steam_success_model.pkl')
    model_features = joblib.load('model_features.pkl')
    return model, model_features

try:
    with st.spinner("Cargando componentes del modelo..."):
        model, model_features = load_model()
except Exception as e:
    st.error(f"Error crítico al cargar los archivos del modelo: {e}")
    st.stop()

# --- PANEL LATERAL (SIDEBAR) ---
with st.sidebar:
    st.header("⚙️ Parámetros del Juego")
    
    # Botones de perfiles rápidos usando session_state
    st.subheader("⚡ Perfiles Rápidos")
    col_b1, col_b2 = st.columns(2)
    with col_b1:
        if st.button("🎮 Indie"):
            st.session_state['price'] = 4.99
            st.session_state['age'] = 0
            st.session_state['dlc'] = 0
            st.session_state['ach'] = 15
    with col_b2:
        if st.button("🔥 AAA"):
            st.session_state['price'] = 59.99
            st.session_state['age'] = 16
            st.session_state['dlc'] = 5
            st.session_state['ach'] = 45

    st.divider()
    
    # Entradas conectadas a estados para admitir los perfiles rápidos
    price = st.number_input("Precio ($)", min_value=0.0, value=st.session_state.get('price', 9.99), step=1.0, format="%.2f")
    required_age = st.slider("Edad requerida (PEGI/ESRB)", min_value=0, max_value=18, value=st.session_state.get('age', 0))
    discount_dlc = st.number_input("DLCs con descuento", min_value=0, value=st.session_state.get('dlc', 0), step=1)
    achievements = st.number_input("Cantidad de logros", min_value=0, value=st.session_state.get('ach', 10), step=5)

    st.divider()
    st.subheader("🖥️ Plataformas compatibles")
    col_p1, col_p2 = st.columns(2)
    with col_p1:
        windows = st.checkbox("Windows", value=True)
        mac = st.checkbox("Mac", value=False)
    with col_p2:
        linux = st.checkbox("Linux", value=False)

# --- CONTENIDO PRINCIPAL ---
st.subheader("🎯 Categorización del Juego")

genre_options = ['Action', 'Adventure', 'Indie', 'Strategy', 'RPG', 'Simulation', 'Casual']
selected_genre = st.selectbox("Selecciona el género principal del juego", genre_options)

st.markdown("<br>", unsafe_allow_html=True)

# Botón de predicción
if st.button("🔮 Calcular Predicción de Éxito"):
    with st.spinner("Analizando características con el modelo..."):
        input_data = {col: 0 for col in model_features}
        
        if 'Price' in input_data: input_data['Price'] = price
        if 'Required age' in input_data: input_data['Required age'] = required_age
        if 'DiscountDLC count' in input_data: input_data['DiscountDLC count'] = discount_dlc
        if 'Achievements' in input_data: input_data['Achievements'] = achievements
        if 'Windows' in input_data: input_data['Windows'] = int(windows)
        if 'Mac' in input_data: input_data['Mac'] = int(mac)
        if 'Linux' in input_data: input_data['Linux'] = int(linux)
        
        if selected_genre in input_data:
            input_data[selected_genre] = 1
            
        input_df = pd.DataFrame([input_data])[model_features]
        
        # Predicción y probabilidad
        prediction = model.predict(input_df)[0]
        
        probability = None
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(input_df)[0]
            probability = probs[1] * 100  # Porcentaje de éxito
        
    st.divider()
    
    # Resultados avanzados con métricas
    if prediction == 1:
        st.success("### 🎉 ¡Resultado: Juego EXITOSO!")
        if probability is not None:
            st.metric(label="Confianza del Modelo (Probabilidad de Éxito)", value=f"{probability:.1f}%")
        st.markdown("Las características ingresadas se alinean positivamente con los patrones de juegos con alta recepción en Steam.")
        st.balloons()
    else:
        st.warning("### ⚠️ Resultado: Éxito No Garantizado")
        if probability is not None:
            st.metric(label="Probabilidad estimada de éxito", value=f"{probability:.1f}%", delta="Bajo el umbral")
        st.markdown("El modelo indica que bajo estas condiciones específicas el juego **podría tener dificultades** para alcanzar el umbral de éxito esperado.")