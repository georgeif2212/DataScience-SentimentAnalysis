import streamlit as st
import pandas as pd
import json
from pathlib import Path
from PIL import Image

st.set_page_config(
    page_title='Análisis de Sentimiento de Tweets',
    page_icon=':bar_chart:',
)

# ----------------------------------------------------------------------------
# Cargar datos desde JSON
DATA_PATH = Path(__file__).parent / 'data/sentiment_results.json'
with open(DATA_PATH, 'r', encoding='utf-8') as f:
    data = json.load(f)

tweets = pd.DataFrame(data['tweets'])
timeline = pd.DataFrame(data['timeline'])

# Convertir la columna de fecha en datetime
tweets['tweet_created'] = pd.to_datetime(tweets['tweet_created'])
timeline['date'] = pd.to_datetime(timeline['date'])

# ----------------------------------------------------------------------------
# Título y documentación
st.title('Análisis de Sentimiento de Tweets de Aerolíneas en Estados Unidos')

with st.expander('📝 Documentación del proyecto'):
    st.markdown("""
    **Origen del dataset:**
    El dataset fue recolectado desde Twitter, conteniendo más de 87,000 tweets relacionados con la marca Starbucks.

    **Preprocesamiento:**
    - Se eliminaron menciones, hashtags, URLs, signos de puntuación y caracteres especiales.
    - Los textos fueron convertidos a minúsculas.
    - Se eliminaron palabras vacías (stopwords).

    **Modelo utilizado:**
    - Se empleó VADER (Valence Aware Dictionary and sEntiment Reasoner) para el análisis de sentimientos.
    - Los tweets fueron clasificados como **positivos**, **negativos** o **neutrales** con base en su puntaje de compound.

    **Limitaciones:**
    - El análisis se basa en texto sin contexto, lo que puede causar errores de interpretación (por ejemplo, sarcasmo).
    - VADER está optimizado para inglés y puede fallar en expresiones coloquiales o multilingües.
    """)

# ----------------------------------------------------------------------------
# Estadísticas generales
st.subheader('📊 Estadísticas Generales')
st.markdown("Porcentaje de cada categoría de sentimiento")

col1, col2 = st.columns(2)

with col1:
    st.image('./data/grafico_porcentaje_sentimiento.png', caption='Distribución de Sentimientos')

with col2:
    st.json(data['summary'])

# ----------------------------------------------------------------------------
# Gráfico de línea: serie temporal de sentimientos
st.subheader('📈 Sentimientos a lo largo del tiempo')

st.line_chart(
    timeline.set_index('date')[['positive', 'neutral', 'negative']],
    use_container_width=True
)

# ----------------------------------------------------------------------------
# Nube de palabras por sentimiento
st.subheader('☁️ Nube de palabras por sentimiento')

sentiment_option = st.selectbox('Selecciona un sentimiento:', ['positive', 'neutral', 'negative'])

st.image(data['wordclouds'][sentiment_option], caption=f'Nube de palabras: {sentiment_option}')

# ----------------------------------------------------------------------------
# Filtros para tweets
st.subheader('🔍 Filtro de tweets')

col1, col2 = st.columns(2)

with col1:
    selected_sentiment = st.selectbox('Filtrar por sentimiento:', ['Todos'] + list(tweets['sentiment'].unique()))

with col2:
    date_range = st.date_input(
        'Seleccionar rango de fechas:',
        value=(tweets['tweet_created'].min().date(), tweets['tweet_created'].max().date())
    )

filtered_df = tweets.copy()
if selected_sentiment != 'Todos':
    filtered_df = filtered_df[filtered_df['sentiment'] == selected_sentiment]

filtered_df = filtered_df[
    (filtered_df['tweet_created'].dt.date >= date_range[0]) &
    (filtered_df['tweet_created'].dt.date <= date_range[1])
]

st.write(f"Se encontraron {len(filtered_df)} tweets con los filtros aplicados.")

st.dataframe(filtered_df[['tweet_created', 'sentiment', 'clean_text']].head(20), use_container_width=True)

# ----------------------------------------------------------------------------
# Footer
st.markdown("""
---
**Creado por .** Proyecto académico de análisis de sentimientos con Streamlit.
""")
