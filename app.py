# =============================================================
#  ANÁLISIS DE MONITOREO AMBIENTAL
#  Para ejecutar, en la terminal:  streamlit run app.py
# =============================================================

# --- BLOQUE 1: las herramientas que vamos a usar ---
import streamlit as st              # construye la página web
import pandas as pd                 # maneja la tabla de datos
import matplotlib.pyplot as plt     # gráficos básicos
import seaborn as sns               # gráficos estadísticos bonitos
import plotly.express as px         # gráficos interactivos

st.set_page_config(page_title="Monitoreo Ambiental", layout="wide")


# --- BLOQUE 2: cargar y preparar los datos ---
@st.cache_data                      # guarda el resultado en memoria: no relee el archivo cada vez
def cargar():
    df = pd.read_csv("monitoreo_ambiental.csv")

    # La hora viene como texto "14:30". La convertimos a número (14)
    # para poder ordenarla y graficarla.
    df["Hora"] = pd.to_datetime(df["Hora_Lectura"], format="%H:%M").dt.hour

    # El ICA tiene un orden natural (Buena es mejor que Peligrosa).
    # Si no lo declaramos, los gráficos lo ordenan alfabéticamente y engañan.
    orden = ["Buena", "Moderada", "Dañina para grupos sensibles",
             "Dañina", "Muy Dañina", "Peligrosa"]
    df["Indice_Calidad_Aire_ICA"] = pd.Categorical(
        df["Indice_Calidad_Aire_ICA"], categories=orden, ordered=True)

    return df

df = cargar()

# Las cuatro variables numéricas, en una lista para reutilizarla
NUMERICAS = ["PM2_5_Ug_m3", "Temperatura_C", "Humedad_Relativa_Pct", "Nivel_Ruido_dB"]


# --- BLOQUE 3: título y números generales ---
st.title("Monitoreo ambiental urbano")
st.write(f"{df.shape[0]} lecturas de sensores en {df['Ciudad'].nunique()} ciudades.")

c1, c2, c3 = st.columns(3)          # tres columnas lado a lado
c1.metric("PM2.5 promedio", f"{df['PM2_5_Ug_m3'].mean():.1f} µg/m³")
c2.metric("Temperatura promedio", f"{df['Temperatura_C'].mean():.1f} °C")
c3.metric("Ruido promedio", f"{df['Nivel_Ruido_dB'].mean():.1f} dB")


# --- BLOQUE 4: revisar la calidad de los datos ---
st.header("1. ¿Están completos los datos?")

st.write(f"Valores faltantes: **{df.isna().sum().sum()}** · "
         f"Filas duplicadas: **{df.duplicated().sum()}**")
st.dataframe(df.head(), use_container_width=True)

st.subheader("Prueba de coherencia: ¿el ICA corresponde al PM2.5?")
st.write("El índice de calidad del aire debería subir cuando sube el material particulado. "
         "Si es así, el promedio de PM2.5 tiene que crecer de *Buena* a *Peligrosa*.")

# groupby agrupa por categoría y calcula un promedio dentro de cada grupo
coherencia = df.groupby("Indice_Calidad_Aire_ICA", observed=True)["PM2_5_Ug_m3"].mean().round(1)
st.bar_chart(coherencia)

st.warning("Los promedios no crecen de forma ordenada: la etiqueta del ICA no coincide "
           "con lo que midió el sensor. Es un defecto del dataset, y es nuestro primer hallazgo.")


# --- BLOQUE 5: una variable a la vez (análisis univariado) ---
st.header("2. ¿Cómo se comporta cada variable?")

# selectbox crea un menú desplegable; devuelve la opción elegida
variable = st.selectbox("Elige una variable", NUMERICAS)
serie = df[variable]

c1, c2, c3 = st.columns(3)
c1.metric("Media", f"{serie.mean():.2f}")
c2.metric("Mediana", f"{serie.median():.2f}")
c3.metric("Desviación estándar", f"{serie.std():.2f}")

izq, der = st.columns(2)

with izq:
    # Histograma con matplotlib: cuenta cuántos datos caen en cada rango
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(serie, bins=25, color="steelblue", edgecolor="white")
    ax.axvline(serie.mean(), color="red", label="Media")
    ax.axvline(serie.median(), color="green", linestyle="--", label="Mediana")
    ax.set_xlabel(variable)
    ax.set_ylabel("Cantidad de lecturas")
    ax.legend()
    st.pyplot(fig)
    plt.close(fig)               # libera memoria; sin esto la app se pone lenta

with der:
    # Caja y bigotes con seaborn: muestra cuartiles y valores extremos
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.boxplot(x=serie, ax=ax, color="steelblue")
    ax.set_xlabel(variable)
    st.pyplot(fig)
    plt.close(fig)

# Si la media supera a la mediana, hay pocos valores muy altos jalando el promedio
if serie.mean() > serie.median():
    st.info("La media es mayor que la mediana: la distribución tiene cola hacia la derecha.")
else:
    st.info("La media es menor o igual que la mediana: la distribución no tiene cola derecha.")


# --- BLOQUE 6: variables de texto (categóricas) ---
st.header("3. ¿Cómo se reparten las categorías?")

categoria = st.selectbox("Elige una categoría", ["Ciudad", "Tipo_Zona", "Presencia_Lluvia"])

# value_counts cuenta cuántas veces aparece cada valor
conteo = df[categoria].value_counts().reset_index()
conteo.columns = [categoria, "Lecturas"]

izq, der = st.columns([1, 2])
izq.dataframe(conteo, hide_index=True, use_container_width=True)

# Gráfico de barras interactivo con plotly (puedes pasar el mouse encima)
fig = px.bar(conteo, x=categoria, y="Lecturas", color=categoria)
fig.update_layout(showlegend=False)
der.plotly_chart(fig, use_container_width=True)


# --- BLOQUE 7: relación entre dos variables numéricas ---
st.header("4. ¿Unas variables explican a otras?")

izq, der = st.columns(2)

with izq:
    # La correlación va de -1 a 1. Cerca de 0 significa "sin relación".
    corr = df[NUMERICAS].corr()
    fig, ax = plt.subplots(figsize=(6, 4.5))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, vmin=-1, vmax=1, ax=ax)
    ax.set_title("Matriz de correlación")
    st.pyplot(fig)
    plt.close(fig)

with der:
    x = st.selectbox("Eje horizontal", NUMERICAS, index=1)
    y = st.selectbox("Eje vertical", NUMERICAS, index=0)
    if x == y:
        st.warning("Elige dos variables distintas.")
    else:
        fig = px.scatter(df, x=x, y=y, color="Tipo_Zona", opacity=0.6)
        st.plotly_chart(fig, use_container_width=True)
        st.caption(f"Correlación entre las dos: r = {df[x].corr(df[y]):.3f}")

st.warning("Todas las correlaciones están cerca de cero. Ninguna variable explica a otra: "
           "los puntos forman una nube sin dirección, no una línea.")


# --- BLOQUE 8: comparar grupos ---
st.header("5. ¿Hay diferencias entre zonas o ciudades?")

grupo = st.selectbox("Comparar por", ["Tipo_Zona", "Ciudad", "Presencia_Lluvia"])

# Promedio y dispersión de PM2.5 dentro de cada grupo
tabla = df.groupby(grupo, observed=True)["PM2_5_Ug_m3"].agg(["mean", "std"]).round(2)
tabla.columns = ["Promedio", "Desviación estándar"]
st.dataframe(tabla.reset_index(), hide_index=True, use_container_width=True)

fig = px.box(df, x=grupo, y="PM2_5_Ug_m3", color=grupo)
fig.update_layout(showlegend=False)
st.plotly_chart(fig, use_container_width=True)

# La diferencia entre grupos solo importa si supera la variación dentro de cada grupo
brecha = tabla["Promedio"].max() - tabla["Promedio"].min()
st.info(f"La diferencia entre el grupo más alto y el más bajo es de {brecha:.1f} µg/m³, "
        f"pero dentro de cada grupo los datos varían {tabla['Desviación estándar'].mean():.1f} µg/m³. "
        "La variación interna es mucho mayor: los grupos no son distintos entre sí.")


# --- BLOQUE 9: conclusiones ---
st.header("6. Conclusiones")

st.markdown("""
1. **Los datos están completos.** Sin faltantes, sin duplicados, con rangos posibles.
   Una revisión superficial los aprobaría.

2. **La etiqueta del ICA no sirve.** Debería derivarse del PM2.5, pero cada categoría
   cubre el mismo rango de valores. Fue asignada al azar.

3. **Ninguna variable se relaciona con otra.** La correlación más fuerte no llega a 0.1.

4. **Los grupos no se diferencian.** Un parque natural registra en promedio *más*
   contaminación que una zona industrial, lo cual no tiene sentido físico.

5. **Conclusión.** El valor de este análisis no está en encontrar una relación —no la hay—
   sino en demostrar con números por qué no la hay. Antes de usar estos sensores para tomar
   decisiones, hay que auditar cómo se generó la etiqueta del ICA.
""")