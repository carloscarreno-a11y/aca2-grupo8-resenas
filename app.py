"""App Streamlit – ACA 2 Aprendizaje Automático (CUN) · Grupo 8
Análisis de sentimiento de reseñas de comercio electrónico con BETO (Transformer en español).

Ejecutar localmente:
    pip install -r requirements.txt
    streamlit run app.py
"""
import altair as alt
import pandas as pd
import streamlit as st

import utils as U

st.set_page_config(page_title="Sentimiento de reseñas – Grupo 8", page_icon="🛒", layout="wide")


@st.cache_resource(show_spinner="Cargando el modelo (la primera vez puede tardar ~1 minuto)…")
def _modelo():
    model_id = None
    try:
        model_id = st.secrets.get("MODEL_ID")
    except Exception:  # sin secrets.toml
        pass
    return U.cargar_modelo(model_id)


predecir, NOMBRE_MODELO, ES_RESPALDO, AVISOS = _modelo()
METRICAS = U.cargar_metricas()
U.iniciar_historial()


def barras_probabilidad(fila):
    datos = pd.DataFrame({"sentimiento": U.ETIQUETAS, "probabilidad": [fila[f"p_{e}"] for e in U.ETIQUETAS]})
    return (alt.Chart(datos).mark_bar(cornerRadiusEnd=4)
            .encode(x=alt.X("probabilidad:Q", scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format="%")),
                    y=alt.Y("sentimiento:N", sort=U.ETIQUETAS, title=None),
                    color=alt.Color("sentimiento:N", legend=None,
                                    scale=alt.Scale(domain=U.ETIQUETAS, range=[U.COLORES[e] for e in U.ETIQUETAS])),
                    tooltip=["sentimiento", alt.Tooltip("probabilidad:Q", format=".1%")])
            .properties(height=150))


def barras_conteo(df):
    datos = df["sentimiento"].value_counts().reindex(U.ETIQUETAS, fill_value=0).rename_axis("sentimiento").reset_index(name="n")
    return (alt.Chart(datos).mark_bar(cornerRadiusEnd=4)
            .encode(x=alt.X("sentimiento:N", sort=U.ETIQUETAS, title=None), y=alt.Y("n:Q", title="Reseñas"),
                    color=alt.Color("sentimiento:N", legend=None,
                                    scale=alt.Scale(domain=U.ETIQUETAS, range=[U.COLORES[e] for e in U.ETIQUETAS])))
            .properties(height=260))


# ------------------------------------------------------------------ barra lateral
with st.sidebar:
    st.header("🛒 Grupo 8 · CUN")
    st.caption("ACA 2 – Aprendizaje Automático")
    if ES_RESPALDO:
        st.warning(f"**Modelo activo:** {NOMBRE_MODELO}.\n\nBETO no está disponible; se usa el modelo clásico.",
                   icon="⚠️")
    else:
        st.success(f"**Modelo activo:** {NOMBRE_MODELO}", icon="✅")
    for a in AVISOS:
        st.caption("ℹ️ " + a)
    st.divider()
    st.caption("Clasifica reseñas en **negativo**, **neutro** o **positivo**. Es una herramienta académica "
               "de apoyo para equipos comerciales; no reemplaza la lectura humana de casos sensibles.")

st.title("🛒 ¿Qué opinan los clientes?")
st.markdown("Análisis automático del **sentimiento de reseñas** de comercio electrónico con **BETO**, "
            "un Transformer preentrenado en español y ajustado con reseñas reales de Amazon.")

t1, t2, t3, t4, t5 = st.tabs(["📝 Analizar reseña", "📂 Análisis por lotes", "🗄️ Datos e historial (SQL)",
                              "📊 Desempeño del modelo", "ℹ️ Acerca de"])

# ------------------------------------------------------------------ 1. individual
EJEMPLOS = {
    "(escribir mi propia reseña)": "",
    "Positiva": "Me encantó, llegó antes de lo esperado y funciona perfecto. Lo recomiendo.",
    "Negativa": "Pésimo. Se dañó a la semana y el vendedor nunca respondió.",
    "Mixta": "Está bien para el precio, aunque el material se siente un poco frágil.",
    "Con negación": "No es lo que esperaba, pero tampoco es malo.",
}
with t1:
    eleccion = st.selectbox("Cargar un ejemplo", list(EJEMPLOS))
    texto = st.text_area("Reseña del cliente", value=EJEMPLOS[eleccion], height=130,
                         placeholder="Escribe o pega aquí una reseña en español…")
    if st.button("Analizar sentimiento", type="primary", disabled=not texto.strip()) and texto.strip():
        res = U.clasificar(predecir, [texto]).iloc[0]
        U.guardar(res.to_frame().T, "individual", NOMBRE_MODELO)
        c1, c2 = st.columns([1, 2])
        with c1:
            st.markdown(f"### {U.EMOJI[res.sentimiento]} {res.sentimiento.capitalize()}")
            st.metric("Confianza del modelo", f"{res.confianza:.0%}")
            if res.confianza < 0.55:
                st.info("Confianza baja: la reseña probablemente mezcla aspectos positivos y negativos.")
        with c2:
            st.altair_chart(barras_probabilidad(res), use_container_width=True)

# ------------------------------------------------------------------ 2. lotes
with t2:
    st.markdown("Sube un **CSV** con una columna de texto (una reseña por fila) o usa las reseñas de ejemplo.")
    archivo = st.file_uploader("Archivo CSV", type="csv")
    usar_ejemplo = st.button("Usar reseñas de ejemplo")
    datos = None
    if archivo is not None:
        try:
            datos = pd.read_csv(archivo)
        except Exception as e:  # noqa: BLE001
            st.error(f"No se pudo leer el archivo: {e}")
    elif usar_ejemplo or st.session_state.get("ejemplo_lotes"):
        st.session_state["ejemplo_lotes"] = True
        datos = pd.read_csv(U.BASE / "resultados" / "ejemplos_app.csv")

    if datos is not None and len(datos):
        cols_texto = [c for c in datos.columns if pd.api.types.is_string_dtype(datos[c])] or list(datos.columns)
        col = st.selectbox("Columna con el texto", cols_texto,
                           index=cols_texto.index("texto") if "texto" in cols_texto else 0)
        serie = datos[col].fillna("").astype(str).str.strip()
        if (serie == "").any():
            st.caption(f"Se omitieron {(serie == '').sum()} filas sin texto.")
            datos = datos[serie != ""].reset_index(drop=True)
        limite = 500
        if len(datos) > limite:
            st.info(f"Se analizarán las primeras {limite} filas (límite de la versión gratuita).")
            datos = datos.head(limite)
        with st.spinner(f"Analizando {len(datos)} reseñas…"):
            res = U.clasificar(predecir, datos[col].fillna("").astype(str).tolist())
        U.guardar(res, "lote", NOMBRE_MODELO)
        salida = pd.concat([datos.drop(columns=[col]).reset_index(drop=True), res], axis=1)

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Reseñas", len(res))
        for k, e in zip([k2, k3, k4], U.ETIQUETAS):
            k.metric(f"{U.EMOJI[e]} {e.capitalize()}", f"{(res.sentimiento == e).mean():.0%}")
        g1, g2 = st.columns([1, 2])
        g1.altair_chart(barras_conteo(res), use_container_width=True)
        g2.dataframe(salida, use_container_width=True, height=300)
        st.download_button("⬇️ Descargar resultados (CSV)", salida.to_csv(index=False).encode("utf-8"),
                           file_name="sentimiento_resenas.csv", mime="text/csv")

# ------------------------------------------------------------------ 3. SQL
with t3:
    st.subheader("Historial de análisis de esta app")
    resumen = U.consultar("SELECT sentimiento, COUNT(*) AS n, ROUND(AVG(confianza), 3) AS confianza_media "
                          "FROM analisis GROUP BY sentimiento ORDER BY n DESC", U.DB_HISTORIAL)
    if len(resumen):
        st.dataframe(resumen, use_container_width=True, hide_index=True)
        st.dataframe(U.consultar("SELECT fecha, origen, sentimiento, confianza, texto FROM analisis "
                                 "ORDER BY id DESC LIMIT 20", U.DB_HISTORIAL), use_container_width=True, hide_index=True)
        st.caption("En Streamlit Community Cloud este historial se reinicia cuando la app se reinicia.")
    else:
        st.caption("Aún no hay análisis guardados.")

    st.subheader("Explorar el dataset con SQL")
    st.caption("Base SQLite `data/resenas.db` creada en el notebook (entrenamiento, validación y prueba).")
    if U.DB_DATASET.exists():
        nombre_q = st.selectbox("Consulta", list(U.CONSULTAS_DATASET))
        st.code(U.CONSULTAS_DATASET[nombre_q], language="sql")
        st.dataframe(U.consultar(U.CONSULTAS_DATASET[nombre_q], U.DB_DATASET), use_container_width=True, hide_index=True)
    else:
        st.info("No se encontró `data/resenas.db`. Cópiala desde los resultados del notebook.")

# ------------------------------------------------------------------ 4. métricas
with t4:
    if not METRICAS:
        st.info("No se encontró `resultados/metricas.json`.")
    else:
        b, m = METRICAS["baseline"], METRICAS["beto"]
        st.subheader(f"Resultados en el conjunto de prueba ({METRICAS['datos']['n_test']:,} reseñas)")
        c = st.columns(4)
        c[0].metric("Accuracy BETO", f"{m['accuracy']:.3f}", f"{m['accuracy'] - b['accuracy']:+.3f} vs. clásico")
        c[1].metric("F1 macro BETO", f"{m['f1_macro']:.3f}", f"{m['f1_macro'] - b['f1_macro']:+.3f} vs. clásico")
        c[2].metric("F1 neutro", f"{m['por_clase']['neutro']['f1']:.3f}")
        c[3].metric("Errores graves (neg↔pos)", m["errores_graves"], f"{m['errores_graves'] - b['errores_graves']:+d}",
                    delta_color="inverse")
        tabla = pd.DataFrame({
            "Modelo": ["TF-IDF + Regresión Logística", "BETO (fine-tuning)"],
            "Accuracy": [b["accuracy"], m["accuracy"]], "F1 macro": [b["f1_macro"], m["f1_macro"]],
            "F1 negativo": [b["por_clase"]["negativo"]["f1"], m["por_clase"]["negativo"]["f1"]],
            "F1 neutro": [b["por_clase"]["neutro"]["f1"], m["por_clase"]["neutro"]["f1"]],
            "F1 positivo": [b["por_clase"]["positivo"]["f1"], m["por_clase"]["positivo"]["f1"]]})
        st.dataframe(tabla.round(3), use_container_width=True, hide_index=True)
        figs = U.BASE / "figs"
        g1, g2 = st.columns(2)
        for g, f, cap in [(g1, "05_curvas.png", "Curvas de aprendizaje"),
                          (g2, "06_confusion_beto.png", "Matriz de confusión – BETO")]:
            if (figs / f).exists():
                g.image(str(figs / f), caption=cap)
        if (figs / "08_comparacion.png").exists():
            st.image(str(figs / "08_comparacion.png"), caption="Comparación con el modelo clásico")

# ------------------------------------------------------------------ 5. info
with t5:
    cfg = METRICAS.get("config", {})
    st.markdown(f"""
**Problema.** Una tienda en línea recibe cientos de reseñas al día y no alcanza a leerlas todas. Esta app
las clasifica automáticamente para detectar a tiempo a los clientes insatisfechos y los temas que generan quejas.

**Datos.** *Multilingual Amazon Reviews Corpus*, subconjunto en español (Keung et al., 2020). 1–2★ = negativo,
3★ = neutro, 4–5★ = positivo.

**Modelo.** BETO (`{cfg.get('modelo_base', 'dccuchile/bert-base-spanish-wwm-cased')}`; Cañete et al., 2020),
un Transformer de 12 capas preentrenado en español, ajustado durante {cfg.get('epocas', '–')} épocas con
pérdida ponderada por clase. Como respaldo, la app incluye un modelo clásico TF-IDF + Regresión Logística.

**Limitaciones.** Las reseñas de entrenamiento son de Amazon (España/México); expresiones muy locales de
Colombia pueden clasificarse peor. La clase neutro es la más difícil incluso para personas.

**Equipo – Grupo 8:** {", ".join(U.CONFIG.get("integrantes", []))}
""")
