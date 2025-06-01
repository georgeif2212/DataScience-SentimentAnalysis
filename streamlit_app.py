import streamlit as st
import pandas as pd
import json
from pathlib import Path
from PIL import Image

st.set_page_config(
    page_title="Análisis de Sentimiento de Tweets",
    page_icon=":bar_chart:",
)

# ----------------------------------------------------------------------------
# Cargar datos desde JSON
DATA_PATH = Path(__file__).parent / "data/sentiment_results.json"
with open(DATA_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

tweets = pd.DataFrame(data["tweets"])
timeline = pd.DataFrame(data["timeline"])

# Convertir la columna de fecha en datetime
tweets["tweet_created"] = pd.to_datetime(tweets["tweet_created"])
timeline["date"] = pd.to_datetime(timeline["date"])

# ----------------------------------------------------------------------------
# Título y documentación
st.title("Análisis de Sentimiento de Tweets de Aerolíneas en Estados Unidos")

with st.expander("📝 Documentación del proyecto"):
    st.markdown(
        """
    ### 📄 Origen del dataset:
    El dataset fue recolectado desde Kaggle y contiene más de 87,000 tweets relacionados con un tema específico. Cada registro incluye texto, fecha de publicación y un identificador único del tweet.

    ### 🧹 Preprocesamiento:
    Se utilizó la librería **NLTK** (Natural Language Toolkit) por su robustez en tareas de procesamiento de lenguaje natural. Las etapas del preprocesamiento incluyeron:
    - Conversión del texto a minúsculas.
    - Eliminación de signos de puntuación, menciones, hashtags, URLs y caracteres no alfabéticos.
    - Tokenización usando `nltk.word_tokenize`.
    - Eliminación de stopwords en inglés mediante `nltk.corpus.stopwords`.
    - Lematización con `WordNetLemmatizer` para reducir las palabras a su forma base (por ejemplo, *running* → *run*), mejorando la consistencia del análisis.

    ### 🤖 Modelo de análisis de sentimientos:
    Se utilizó **VADER (Valence Aware Dictionary and sEntiment Reasoner)**, una herramienta especializada en análisis de sentimientos de texto corto, especialmente eficaz con lenguaje informal como el que se encuentra en redes sociales. 
    Se eligió VADER en lugar de TextBlob por su mayor precisión en el análisis de textos con emoticonos, siglas y puntuación emocional como mayúsculas o signos de exclamación.

    Cada tweet fue etiquetado como **positivo**, **neutral** o **negativo**, según la puntuación compuesta (`compound`) calculada por VADER:
    - `compound ≥ 0.05`: Positivo  
    - `compound ≤ -0.05`: Negativo  
    - En otro caso: Neutral

    ### 📊 Resultados presentados:
    - **Estadísticas agregadas:** Cantidad y porcentaje de tweets por categoría de sentimiento.
    - **Visualizaciones:** 
        - Gráfico de barras con la distribución de sentimientos.
        - Serie temporal de la evolución de sentimientos por fecha.
        - Nubes de palabras generadas para cada tipo de sentimiento.
    - **Exploración de tweets:** Posibilidad de filtrar los comentarios por sentimiento y fecha.

    ### 📌 Limitaciones:
    - Los resultados pueden verse afectados por la ambigüedad, sarcasmo o errores gramaticales.
    - El análisis solo considera el texto del tweet, ignorando contexto, imágenes o enlaces asociados.
    - VADER no es multilingüe, por lo tanto solo se analizaron tweets en inglés.

    ### 🎯 Objetivo:
    Este proyecto busca reforzar habilidades en la limpieza y análisis de texto, así como en la presentación dinámica de resultados mediante una interfaz web interactiva.
    """
    )


# ----------------------------------------------------------------------------
# Estadísticas generales
st.subheader("📊 Estadísticas Generales")
st.markdown("Porcentaje de cada categoría de sentimiento")

col1, col2 = st.columns(2)

with col1:
    st.image(
        "./data/grafico_porcentaje_sentimiento.png",
        caption="Distribución de Sentimientos",
    )

with col2:
    st.json(data["summary"])

# ----------------------------------------------------------------------------
# Gráfico de línea: serie temporal de sentimientos
st.subheader("📈 Sentimientos a lo largo del tiempo")

st.line_chart(
    timeline.set_index("date")[["positive", "neutral", "negative"]],
    use_container_width=True,
)

# ----------------------------------------------------------------------------
# Nube de palabras por sentimiento
st.subheader("☁️ Nube de palabras por sentimiento")

sentiment_option = st.selectbox(
    "Selecciona un sentimiento:", ["positive", "neutral", "negative"]
)

st.image(
    data["wordclouds"][sentiment_option],
    caption=f"Nube de palabras: {sentiment_option}",
)

# ----------------------------------------------------------------------------
# Filtros para tweets
st.subheader("🔍 Filtro de tweets")

col1, col2 = st.columns(2)

with col1:
    selected_sentiment = st.selectbox(
        "Filtrar por sentimiento:", ["Todos"] + list(tweets["sentiment"].unique())
    )

with col2:
    date_range = st.date_input(
        "Seleccionar rango de fechas:",
        value=(
            tweets["tweet_created"].min().date(),
            tweets["tweet_created"].max().date(),
        ),
    )

filtered_df = tweets.copy()
if selected_sentiment != "Todos":
    filtered_df = filtered_df[filtered_df["sentiment"] == selected_sentiment]

filtered_df = filtered_df[
    (filtered_df["tweet_created"].dt.date >= date_range[0])
    & (filtered_df["tweet_created"].dt.date <= date_range[1])
]

st.write(f"Se encontraron {len(filtered_df)} tweets con los filtros aplicados.")

st.dataframe(
    filtered_df[["tweet_created", "sentiment", "clean_text"]].head(20),
    use_container_width=True,
)

# ----------------------------------------------------------------------------
# Footer
st.markdown(
    """
---
**Creado por .** Proyecto académico de análisis de sentimientos con Streamlit.
"""
)
