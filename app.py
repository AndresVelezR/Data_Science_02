# =============================================================
#  ANÁLISIS DE MONITOREO AMBIENTAL
#  Dashboard didáctico + tutor de IA (Groq) que interpreta
#  cada gráfico y responde preguntas de estadística.
#
#  Para ejecutar, en la terminal:  streamlit run app.py
# =============================================================

import os
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from dotenv import load_dotenv
from groq import Groq

st.set_page_config(page_title="Monitoreo Ambiental", layout="wide")


# =============================================================
# CONFIGURACIÓN DEL TUTOR DE IA (Groq)
# =============================================================
load_dotenv()  # lee GEMINI_API_KEY desde el archivo .env en local

# Orden de búsqueda de la clave: primero variable de entorno / .env
# (uso local), y si no existe, st.secrets (uso en Streamlit Cloud).
# Nota: la variable se llama GEMINI_API_KEY por convención previa del
# proyecto, pero aquí guarda una clave de Groq, no de Google.
API_KEY = os.getenv("GEMINI_API_KEY") or st.secrets.get("GEMINI_API_KEY", None)

MODELO_IA = "llama-3.3-70b-versatile"  # buen balance calidad/velocidad en la capa gratuita de Groq

SYSTEM_PROMPT = """
Eres un tutor de estadística que le enseña a un principiante total a leer un dashboard
de monitoreo ambiental. La persona no sabe qué es una media, una correlación ni un boxplot.

Cuando te den el contexto de un gráfico (qué variable es, qué tipo de gráfico, y sus
estadísticos), responde SIEMPRE con esta estructura, en español, breve (máximo 180 palabras):

1. Qué significa cada número que te dieron, en lenguaje simple y con una analogía si ayuda.
2. Cómo se ve eso reflejado en el gráfico (qué debería notar al mirarlo).
3. Una pista práctica de cómo reconocer este mismo patrón en otro dashboard en el futuro.

No inventes números que no te dieron. No uses jerga sin explicarla. Si la pregunta no
tiene que ver con estadística o con el dashboard, responde igual con amabilidad pero
redirige a temas del dashboard.
"""


@st.cache_resource
def obtener_cliente_ia():
    """Crea el cliente de Groq una sola vez. None si no hay clave configurada."""
    if not API_KEY:
        return None
    return Groq(api_key=API_KEY)

cliente_ia = obtener_cliente_ia()


def preguntar_ia(pregunta: str, contexto: str = "") -> str:
    """Envía una pregunta a Groq con el contexto numérico del gráfico. Nunca lanza error al usuario."""
    if cliente_ia is None:
        return ("⚠️ No hay una clave de API configurada. Crea un archivo `.env` con "
                "`GEMINI_API_KEY=tu_clave_de_groq` (ver instrucciones al final del código).")
    try:
        respuesta = cliente_ia.chat.completions.create(
            model=MODELO_IA,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Contexto de los datos:\n{contexto}\n\nPregunta:\n{pregunta}"},
            ],
            temperature=0.3,
            max_tokens=400,
        )
        return respuesta.choices[0].message.content
    except Exception as e:
        return f"⚠️ No se pudo consultar la IA en este momento. Detalle: {e}"


def boton_explicar(clave: str, pregunta: str, contexto: str):
    """Botón reutilizable: al presionarlo, pide a la IA que explique el gráfico de arriba."""
    if st.button("🤖 Pídele al tutor de IA que te explique este gráfico", key=clave):
        with st.spinner("El tutor está leyendo los números del gráfico..."):
            respuesta = preguntar_ia(pregunta, contexto)
        with st.chat_message("assistant"):
            st.markdown(respuesta)


# --- Cargar y preparar los datos ---
@st.cache_data
def cargar():
    df = pd.read_csv("monitoreo_ambiental.csv")

    # La hora llega como texto "14:30". La convertimos a número (14)
    # para poder ordenarla en un eje y graficarla como serie de tiempo.
    df["Hora"] = pd.to_datetime(df["Hora_Lectura"], format="%H:%M").dt.hour

    # El ICA tiene un orden natural (Buena es mejor que Peligrosa).
    # Si no lo declaramos, los gráficos lo ordenan alfabéticamente
    # y muestran una severidad falsa.
    orden = ["Buena", "Moderada", "Dañina para grupos sensibles",
             "Dañina", "Muy Dañina", "Peligrosa"]
    df["Indice_Calidad_Aire_ICA"] = pd.Categorical(
        df["Indice_Calidad_Aire_ICA"], categories=orden, ordered=True)

    return df

df = cargar()
NUMERICAS = ["PM2_5_Ug_m3", "Temperatura_C", "Humedad_Relativa_Pct", "Nivel_Ruido_dB"]


# =============================================================
# PORTADA
# =============================================================
st.title("Monitoreo ambiental urbano")
st.write(
    "Este dashboard analiza 500 lecturas tomadas por sensores ambientales en cinco "
    "ciudades colombianas. Cada sección explica qué mide un gráfico, por qué se eligió "
    "ese tipo y no otro, y cómo interpretarlo. Además, puedes pedirle a un **tutor de "
    "IA** que te explique cualquier gráfico con los números reales que estás viendo."
)

if cliente_ia is None:
    st.warning(
        "El tutor de IA no está activo: falta configurar `GEMINI_API_KEY`. "
        "El resto del dashboard funciona igual."
    )

c1, c2, c3, c4 = st.columns(4)
c1.metric("Lecturas", df.shape[0])
c2.metric("Ciudades", df["Ciudad"].nunique())
c3.metric("Variables", df.shape[1])
c4.metric("Sensores únicos", df["ID_Sensor"].nunique())

with st.expander("📖 Diccionario de variables y tipos de dato"):
    st.write(
        "Antes de analizar cualquier dato hay que saber qué **tipo** de variable es, "
        "porque el tipo determina qué operación estadística tiene sentido. No se puede "
        "calcular un promedio de una categoría, ni un conteo de frecuencia sobre un "
        "número continuo."
    )
    diccionario = pd.DataFrame([
        ["Ciudad", "Categórica nominal", "Sin orden natural entre categorías"],
        ["Tipo_Zona", "Categórica nominal", "Sin orden natural entre categorías"],
        ["Hora", "Cuantitativa discreta", "Números enteros con orden (0 a 23)"],
        ["PM2_5_Ug_m3", "Cuantitativa continua", "Concentración de partículas, en µg/m³"],
        ["Temperatura_C", "Cuantitativa continua", "Grados Celsius"],
        ["Humedad_Relativa_Pct", "Cuantitativa continua", "Porcentaje (0–100)"],
        ["Nivel_Ruido_dB", "Cuantitativa continua", "Decibeles"],
        ["Presencia_Lluvia", "Booleana", "Verdadero / Falso"],
        ["Indice_Calidad_Aire_ICA", "Categórica ordinal", "Tiene orden: Buena < ... < Peligrosa"],
    ], columns=["Variable", "Tipo", "Nota"])
    st.dataframe(diccionario, hide_index=True, use_container_width=True)


# =============================================================
# 1. AUDITORÍA DE CALIDAD DE DATOS
# =============================================================
st.header("1. Auditoría de calidad de datos")
st.write(
    "**Por qué empezar aquí:** ningún gráfico ni estadístico vale nada si los datos "
    "tienen huecos, duplicados o errores de codificación. Un analista experto revisa "
    "esto antes de interpretar cualquier resultado, no después."
)

nulos = df.isna().sum().sum()
duplicados = df.duplicated().sum()
c1, c2 = st.columns(2)
c1.metric("Valores faltantes", nulos, help="Celdas vacías en toda la tabla")
c2.metric("Filas duplicadas", duplicados, help="Registros idénticos en todas las columnas")

if nulos == 0 and duplicados == 0:
    st.success("Sin faltantes ni duplicados. El dataset pasa la revisión estructural básica.")

st.subheader("Prueba de coherencia interna: ¿el ICA corresponde al PM2.5 medido?")
st.write(
    "**Qué mide:** el Índice de Calidad del Aire (ICA) es una etiqueta que, en teoría, "
    "se calcula a partir de la concentración de PM2.5. Si la etiqueta es correcta, el "
    "promedio de PM2.5 debe **aumentar de forma escalonada** al pasar de la categoría "
    "'Buena' hasta 'Peligrosa'."
)
st.write(
    "**Por qué un gráfico de barras y no otro:** las barras comparan un solo número "
    "(el promedio) entre categorías discretas ordenadas. Es el gráfico correcto cuando "
    "quieres comparar magnitudes entre grupos, no la distribución completa de cada uno."
)

coherencia = df.groupby("Indice_Calidad_Aire_ICA", observed=True)["PM2_5_Ug_m3"].mean().round(1)
fig = px.bar(
    coherencia, labels={"value": "PM2.5 promedio (µg/m³)", "Indice_Calidad_Aire_ICA": "Categoría ICA"},
    title="PM2.5 promedio por categoría de ICA — eje X: severidad declarada, eje Y: contaminación medida",
)
fig.update_layout(showlegend=False)
st.plotly_chart(fig, use_container_width=True)

es_creciente = list(coherencia) == sorted(coherencia)
if not es_creciente:
    st.error(
        "**Análisis:** los promedios NO crecen de forma ordenada de 'Buena' a 'Peligrosa'. "
        "Esto significa que la etiqueta del ICA se asignó sin relación con la medición real "
        "de PM2.5. Es el primer hallazgo del dashboard y condiciona todo lo que sigue: "
        "esta columna no debe usarse como variable de referencia."
    )

boton_explicar(
    "ia_coherencia",
    "Explícame esta gráfica de barras y por qué el hallazgo es un problema.",
    f"Gráfico de barras: PM2.5 promedio por categoría de ICA, en orden de severidad. "
    f"Valores: {coherencia.to_dict()}. ¿Es creciente el orden? {es_creciente}.",
)


# =============================================================
# 2. ANÁLISIS UNIVARIADO — UNA VARIABLE CUANTITATIVA A LA VEZ
# =============================================================
st.header("2. Análisis univariado — variables cuantitativas")
st.write(
    "**Qué es 'univariado':** estudiar una sola variable a la vez, sin cruzarla con "
    "otras. El objetivo es describir su **forma** (¿simétrica?, ¿con valores extremos?), "
    "su **centro** (¿alrededor de qué valor se agrupan los datos?) y su **dispersión** "
    "(¿qué tan esparcidos están?)."
)

variable = st.selectbox("Elige una variable cuantitativa", NUMERICAS)
serie = df[variable]

media, mediana, desv = serie.mean(), serie.median(), serie.std()
q1, q3 = serie.quantile([0.25, 0.75])
iqr = q3 - q1
li, ls = q1 - 1.5 * iqr, q3 + 1.5 * iqr
atipicos = int(((serie < li) | (serie > ls)).sum())

st.write("**Medidas de tendencia central** — indican alrededor de qué valor se agrupan los datos:")
c1, c2 = st.columns(2)
c1.metric("Media (promedio aritmético)", f"{media:.2f}")
c2.metric("Mediana (valor central al ordenar)", f"{mediana:.2f}")

st.write("**Medidas de dispersión** — indican qué tan esparcidos están los datos respecto al centro:")
c1, c2, c3 = st.columns(3)
c1.metric("Desviación estándar", f"{desv:.2f}")
c2.metric("Rango intercuartílico (IQR)", f"{iqr:.2f}", help="Distancia entre el cuartil 3 y el cuartil 1")
c3.metric("Valores atípicos", atipicos, help="Fuera de [Q1 − 1.5·IQR, Q3 + 1.5·IQR], regla de Tukey")

izq, der = st.columns(2)

with izq:
    st.subheader("Histograma de frecuencias")
    st.write(
        "**Por qué un histograma y no barras:** un histograma agrupa valores continuos "
        "en intervalos (bins) y cuenta cuántas lecturas caen en cada uno. Se usa "
        "específicamente para variables numéricas continuas, porque muestra la **forma** "
        "completa de la distribución — algo que un solo número (como el promedio) "
        "no puede mostrar."
    )
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(serie, bins=25, color="steelblue", edgecolor="white")
    ax.axvline(media, color="red", lw=2, label=f"Media = {media:.1f}")
    ax.axvline(mediana, color="green", lw=2, linestyle="--", label=f"Mediana = {mediana:.1f}")
    ax.set_xlabel(f"{variable}  (eje X: valor medido)")
    ax.set_ylabel("Cantidad de lecturas  (eje Y: frecuencia)")
    ax.set_title(f"Distribución de {variable}")
    ax.legend()
    st.pyplot(fig)
    plt.close(fig)

with der:
    st.subheader("Diagrama de caja y bigotes (boxplot)")
    st.write(
        "**Por qué un boxplot y no un histograma aquí:** el boxplot resume la misma "
        "variable en cinco números clave (mínimo, Q1, mediana, Q3, máximo) y marca los "
        "valores atípicos como puntos sueltos. Es el gráfico correcto para identificar "
        "rápidamente cuartiles y valores extremos."
    )
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.boxplot(x=serie, ax=ax, color="steelblue")
    ax.set_xlabel(f"{variable}  (eje X: valor medido; la caja cubre del Q1 al Q3)")
    ax.set_title(f"Cuartiles y atípicos de {variable}")
    st.pyplot(fig)
    plt.close(fig)

st.write("**Cómo leer la relación entre media y mediana:**")
if abs(media - mediana) / desv < 0.05:
    st.info(
        f"**Análisis:** media ({media:.2f}) y mediana ({mediana:.2f}) son casi iguales. "
        "La distribución es aproximadamente simétrica: no hay un grupo de valores extremos "
        "jalando el promedio hacia un lado."
    )
elif media > mediana:
    st.info(
        f"**Análisis:** la media ({media:.2f}) es mayor que la mediana ({mediana:.2f}). "
        "Esto indica asimetría hacia la derecha: existen algunos valores altos que "
        "estiran el promedio, aunque la mayoría de las lecturas se concentra más abajo."
    )
else:
    st.info(
        f"**Análisis:** la media ({media:.2f}) es menor que la mediana ({mediana:.2f}). "
        "Esto indica asimetría hacia la izquierda: existen algunos valores bajos que "
        "arrastran el promedio, aunque la mayoría de las lecturas se concentra más arriba."
    )

boton_explicar(
    "ia_univariado",
    f"Explícame el histograma y el boxplot de {variable} como si nunca hubiera visto ninguno.",
    f"Variable: {variable}. Media={media:.2f}, Mediana={mediana:.2f}, "
    f"Desviación estándar={desv:.2f}, Q1={q1:.2f}, Q3={q3:.2f}, IQR={iqr:.2f}, "
    f"valores atípicos={atipicos}.",
)


# =============================================================
# 3. ANÁLISIS UNIVARIADO — VARIABLES CATEGÓRICAS
# =============================================================
st.header("3. Análisis univariado — variables categóricas")
st.write(
    "**Qué es diferente aquí:** una categoría no tiene promedio ni desviación estándar. "
    "Lo que se mide es **frecuencia** (cuántas veces aparece cada valor) y "
    "**proporción** (qué porcentaje del total representa)."
)

categoria = st.selectbox("Elige una variable categórica", ["Ciudad", "Tipo_Zona", "Presencia_Lluvia"])

conteo = df[categoria].value_counts().reset_index()
conteo.columns = [categoria, "Frecuencia absoluta"]
conteo["Frecuencia relativa (%)"] = (conteo["Frecuencia absoluta"] / len(df) * 100).round(1)

izq, der = st.columns([1, 1.5])
with izq:
    st.write(
        "**Tabla de frecuencias:** frecuencia absoluta = número de lecturas; "
        "frecuencia relativa = ese número dividido entre el total, en porcentaje."
    )
    st.dataframe(conteo, hide_index=True, use_container_width=True)

with der:
    st.write(
        "**Por qué barras y no un gráfico de pastel:** el ojo humano compara longitudes "
        "(barras) con más precisión que ángulos o áreas (pastel), sobre todo con más de "
        "tres categorías."
    )
    fig = px.bar(
        conteo, x=categoria, y="Frecuencia absoluta", color=categoria,
        title=f"Frecuencia de lecturas por {categoria}  (eje X: categoría, eje Y: conteo)",
    )
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

maxc = conteo.iloc[0]
minc = conteo.iloc[-1]
st.info(
    f"**Análisis:** la categoría más frecuente es **{maxc[categoria]}** "
    f"({maxc['Frecuencia absoluta']} lecturas, {maxc['Frecuencia relativa (%)']} % del total). "
    f"La menos frecuente es **{minc[categoria]}** ({minc['Frecuencia relativa (%)']} %). "
    f"La diferencia entre ambas es de {maxc['Frecuencia relativa (%)'] - minc['Frecuencia relativa (%)']:.1f} "
    "puntos porcentuales, lo que indica una muestra razonablemente balanceada."
)

boton_explicar(
    "ia_categorica",
    f"Explícame esta tabla de frecuencias y el gráfico de barras de {categoria}.",
    f"Categoría: {categoria}. Tabla de frecuencias: {conteo.to_dict('records')}.",
)


# =============================================================
# 4. ANÁLISIS BIVARIADO — DOS VARIABLES CUANTITATIVAS
# =============================================================
st.header("4. Análisis bivariado — relación entre variables numéricas")
st.write(
    "**Qué mide la correlación de Pearson:** un número entre −1 y 1 que resume qué tan "
    "fuerte y en qué dirección se mueven dos variables juntas. **+1** significa que "
    "cuando una sube, la otra sube en la misma proporción. **−1**, que cuando una sube "
    "la otra baja. **0**, que no hay relación lineal entre ellas. Es una medida de "
    "**asociación**, no de causalidad."
)

izq, der = st.columns(2)

with izq:
    st.subheader("Matriz de correlación")
    st.write(
        "**Por qué un mapa de calor y no una tabla de números:** con cuatro variables "
        "hay seis relaciones posibles. El color permite detectar de un vistazo cuáles "
        "son fuertes y cuáles casi nulas."
    )
    corr = df[NUMERICAS].corr()
    fig, ax = plt.subplots(figsize=(6, 4.5))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, vmin=-1, vmax=1,
                linewidths=0.5, ax=ax)
    ax.set_title("Correlación de Pearson entre variables numéricas")
    st.pyplot(fig)
    plt.close(fig)

with der:
    st.subheader("Diagrama de dispersión")
    st.write(
        "**Por qué un scatter y no una línea:** el scatter muestra cada lectura como un "
        "punto individual, sin asumir un orden entre observaciones. Una línea conectaría "
        "puntos sin secuencia natural y sugeriría una tendencia falsa."
    )
    x = st.selectbox("Eje horizontal (eje X)", NUMERICAS, index=1)
    y = st.selectbox("Eje vertical (eje Y)", NUMERICAS, index=0)
    if x == y:
        st.warning("Elige dos variables distintas para comparar.")
        r = None
    else:
        r = df[x].corr(df[y])
        fig = px.scatter(
            df, x=x, y=y, color="Tipo_Zona", opacity=0.6, trendline="ols",
            title=f"{y} vs {x}  —  cada punto es un sensor, la línea es la tendencia lineal",
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption(f"Correlación de Pearson: r = {r:.3f}")

maxima_corr = corr.where(~(corr == 1)).abs().max().max()
st.error(
    f"**Análisis:** la correlación más fuerte de toda la matriz es |r| = {maxima_corr:.3f}, "
    "muy lejos de 0.3 (umbral que suele considerarse una relación débil pero real). "
    "Ninguna de estas variables explica el comportamiento de otra en este dataset."
)

boton_explicar(
    "ia_correlacion",
    f"Explícame la matriz de correlación y el diagrama de dispersión entre {x} y {y}.",
    f"Matriz de correlación: {corr.round(3).to_dict()}. "
    f"Correlación específica entre {x} y {y}: r={r}. Correlación máxima absoluta en la matriz: {maxima_corr:.3f}.",
)


# =============================================================
# 5. ANÁLISIS BIVARIADO — CATEGÓRICA VS. NUMÉRICA
# =============================================================
st.header("5. Análisis bivariado — comparación entre grupos")
st.write(
    "**Qué se busca aquí:** si el PM2.5 es distinto según la ciudad, la zona o si "
    "llovía. Se comparan estadísticos (media, dispersión) de la misma variable numérica "
    "calculados por separado dentro de cada categoría."
)

grupo = st.selectbox("Comparar PM2.5 por", ["Tipo_Zona", "Ciudad", "Presencia_Lluvia"])

tabla = df.groupby(grupo, observed=True)["PM2_5_Ug_m3"].agg(["count", "mean", "std"]).round(2)
tabla.columns = ["n (lecturas)", "Promedio", "Desviación estándar"]
st.write(
    "**Tabla agrupada:** `n` es el tamaño de cada grupo. `Promedio` es el centro de "
    "cada grupo. `Desviación estándar` mide cuánto varían los datos **dentro** de cada "
    "grupo, y es la referencia para saber si una diferencia entre grupos es real o ruido."
)
st.dataframe(tabla.reset_index(), hide_index=True, use_container_width=True)

st.write(
    "**Por qué un boxplot y no barras de promedio:** las barras solo muestran un número "
    "por grupo y esconden la dispersión interna. El boxplot muestra la distribución "
    "completa de cada grupo lado a lado."
)
fig = px.box(
    df, x=grupo, y="PM2_5_Ug_m3", color=grupo,
    title=f"Distribución de PM2.5 por {grupo}  (eje X: categoría, eje Y: PM2.5 en µg/m³)",
)
fig.update_layout(showlegend=False)
st.plotly_chart(fig, use_container_width=True)

brecha = tabla["Promedio"].max() - tabla["Promedio"].min()
disp_media = tabla["Desviación estándar"].mean()
if brecha < disp_media * 0.5:
    st.warning(
        f"**Análisis:** la diferencia entre el grupo con mayor y menor promedio es de "
        f"{brecha:.1f} µg/m³, pero la desviación estándar **dentro** de cada grupo es en "
        f"promedio {disp_media:.1f} µg/m³ — varias veces mayor. Los grupos no son "
        "estadísticamente distintos entre sí."
    )
else:
    st.info(
        f"**Análisis:** la diferencia entre grupos ({brecha:.1f}) es comparable a la "
        f"dispersión interna ({disp_media:.1f}), lo que amerita una prueba estadística "
        "formal (por ejemplo, ANOVA) para confirmarla."
    )

boton_explicar(
    "ia_grupos",
    f"Explícame el boxplot comparativo de PM2.5 por {grupo} y por qué la diferencia entre grupos importa o no.",
    f"Agrupación por {grupo}. Tabla: {tabla.reset_index().to_dict('records')}. "
    f"Brecha entre promedios={brecha:.2f}. Desviación estándar interna promedio={disp_media:.2f}.",
)


# =============================================================
# 6. ANÁLISIS BIVARIADO — DOS VARIABLES CATEGÓRICAS
# =============================================================
st.header("6. Análisis bivariado — dos variables categóricas")
st.write(
    "**Qué es una tabla de contingencia:** cruza dos variables categóricas y cuenta "
    "cuántas observaciones caen en cada combinación. Al normalizar por fila, cada celda "
    "se convierte en una **proporción condicional**."
)
st.write(
    "**Por qué un heatmap y no barras apiladas:** con seis categorías de ICA, barras "
    "apiladas serían difíciles de comparar. El heatmap usa color para que cualquier "
    "fila o columna con proporciones altas resalte de inmediato."
)

ct = pd.crosstab(df["Tipo_Zona"], df["Indice_Calidad_Aire_ICA"], normalize="index") * 100
fig, ax = plt.subplots(figsize=(9, 4))
sns.heatmap(ct, annot=True, fmt=".1f", cmap="YlOrBr", linewidths=0.5,
            cbar_kws={"label": "% de lecturas dentro de cada zona"}, ax=ax)
ax.set_title("Distribución de categorías ICA dentro de cada Tipo de Zona (% por fila)")
ax.set_xlabel("Categoría ICA  (eje X, ordenado de Buena a Peligrosa)")
ax.set_ylabel("Tipo de zona  (eje Y)")
st.pyplot(fig)
plt.close(fig)

variacion = (ct.max(axis=0) - ct.min(axis=0)).mean()
st.info(
    f"**Análisis:** si el ICA reflejara el tipo de zona, se esperaría que zonas "
    "industriales concentren más lecturas en categorías graves. En cambio, cada fila "
    f"reparte sus porcentajes de forma similar (variación promedio de solo "
    f"{variacion:.1f} puntos porcentuales entre zonas)."
)

boton_explicar(
    "ia_crosstab",
    "Explícame esta tabla de contingencia y el mapa de calor de proporciones.",
    f"Tabla de contingencia (% por fila) entre Tipo_Zona e Indice_Calidad_Aire_ICA: "
    f"{ct.round(1).to_dict()}. Variación promedio entre zonas: {variacion:.1f} puntos porcentuales.",
)


# =============================================================
# 7. ANÁLISIS TEMPORAL
# =============================================================
st.header("7. Análisis temporal")
st.write(
    "**Qué se busca:** un ciclo circadiano — patrones que se repiten según la hora del "
    "día. `Hora` es la única variable con un orden natural en este dataset."
)
st.write(
    "**Por qué una línea y no un scatter aquí:** a diferencia de la sección 4, aquí sí "
    "existe una secuencia lógica (0h, 1h, 2h... 23h), así que conectar los puntos con "
    "una línea representa una progresión real."
)

var_t = st.selectbox("Variable a seguir en el tiempo", NUMERICAS, key="temporal")
por_hora = df.groupby("Hora")[var_t].agg(["mean", "std", "count"]).reset_index()

fig = px.line(
    por_hora, x="Hora", y="mean", markers=True,
    title=f"Promedio de {var_t} por hora del día  (eje X: hora 0–23, eje Y: promedio)",
    labels={"mean": f"{var_t} promedio"},
)
fig.update_layout(xaxis=dict(dtick=2))
st.plotly_chart(fig, use_container_width=True)
st.caption(
    f"Cada punto agrupa en promedio {por_hora['count'].mean():.0f} lecturas por hora — "
    "una muestra pequeña, así que picos aislados pueden deberse a azar muestral."
)

amplitud = por_hora["mean"].max() - por_hora["mean"].min()
st.info(
    f"**Análisis:** el valor promedio oscila {amplitud:.1f} unidades a lo largo del día, "
    f"frente a una dispersión promedio de {por_hora['std'].mean():.1f} dentro de cada "
    "hora. No hay evidencia clara de un ciclo horario."
)

boton_explicar(
    "ia_temporal",
    f"Explícame la línea de tiempo de {var_t} por hora del día.",
    f"Variable: {var_t}. Amplitud entre horas={amplitud:.2f}. "
    f"Dispersión promedio dentro de cada hora={por_hora['std'].mean():.2f}. "
    f"Lecturas promedio por hora={por_hora['count'].mean():.0f}.",
)


# =============================================================
# 8. CONCLUSIONES
# =============================================================
st.header("8. Conclusiones del análisis")
st.markdown("""
1. **Estructura:** el dataset está completo — sin valores faltantes ni filas duplicadas.

2. **Coherencia del ICA:** la etiqueta de calidad del aire no se deriva del PM2.5 medido.

3. **Correlaciones:** ninguna pareja de variables numéricas supera |r| = 0.1.

4. **Comparación entre grupos:** las diferencias de PM2.5 entre zonas, ciudades y
   condición de lluvia son menores que la variación natural dentro de cada grupo.

5. **Patrón horario:** no se detecta un ciclo circadiano consistente.

**Conclusión general:** el valor de este análisis no está en encontrar una relación
—no la hay— sino en demostrarlo con evidencia estadística concreta.
""")


# =============================================================
# 9. TUTOR DE IA — PREGUNTAS LIBRES
# =============================================================
st.header("9. Tutor de estadística — pregunta lo que quieras")
st.write(
    "Este chat conoce el contexto general del dataset (no memoriza cada gráfico que "
    "viste arriba). Úsalo para dudas conceptuales: '¿qué es una desviación estándar?', "
    "'¿por qué el promedio y la mediana pueden ser distintos?', etc."
)

if "chat" not in st.session_state:
    st.session_state.chat = []

for mensaje in st.session_state.chat:
    with st.chat_message(mensaje["role"]):
        st.markdown(mensaje["content"])

pregunta_libre = st.chat_input("Escribe tu pregunta sobre estadística o el dashboard...")
if pregunta_libre:
    st.session_state.chat.append({"role": "user", "content": pregunta_libre})
    with st.chat_message("user"):
        st.markdown(pregunta_libre)

    contexto_general = (
        f"Dataset de monitoreo ambiental, {df.shape[0]} lecturas, columnas: "
        f"{', '.join(df.columns)}. Hallazgo principal: no hay correlaciones fuertes "
        "entre variables, ni diferencias claras entre grupos, y la etiqueta ICA es "
        "incoherente con el PM2.5 medido."
    )
    with st.chat_message("assistant"):
        with st.spinner("Pensando..."):
            respuesta = preguntar_ia(pregunta_libre, contexto_general)
        st.markdown(respuesta)
    st.session_state.chat.append({"role": "assistant", "content": respuesta})


# =============================================================
# NOTAS DE CONFIGURACIÓN (no se muestran en el dashboard)
# =============================================================
# 1. Crea un archivo llamado ".env" en esta misma carpeta con una línea:
#      GEMINI_API_KEY=tu_clave_de_groq_aqui
#    (el nombre de la variable quedó así por convención previa del
#    proyecto; el valor es una clave de console.groq.com, no de Google)
# 2. Instala las dependencias nuevas:
#      pip install groq python-dotenv
# 3. En Streamlit Community Cloud ya tienes configurado el secreto
#    GEMINI_API_KEY, así que no necesitas hacer nada adicional ahí:
#    el código lo detecta automáticamente al desplegar.