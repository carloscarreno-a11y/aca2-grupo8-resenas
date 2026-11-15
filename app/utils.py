"""Lógica de la app separada de la interfaz: carga del modelo (con respaldo),
predicción y persistencia en SQLite. Se puede probar sin levantar Streamlit."""
from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime
from pathlib import Path

import pandas as pd

BASE = Path(__file__).resolve().parent
ETIQUETAS = ["negativo", "neutro", "positivo"]
COLORES = {"negativo": "#C8553D", "neutro": "#E0A458", "positivo": "#4C9F70"}
EMOJI = {"negativo": "😞", "neutro": "😐", "positivo": "😊"}

CONFIG = json.loads((BASE / "config.json").read_text(encoding="utf-8"))
DB_DATASET = BASE / "data" / "resenas.db"
DB_HISTORIAL = BASE / "data" / "historial.db"


# ------------------------------------------------------------------ modelo
def _normalizar(salida):
    """El pipeline de transformers devuelve [dict,...] o [[dict,...]] según versión/entrada."""
    if salida and isinstance(salida[0], list):
        return salida[0]
    return salida


def _desde_transformers(fuente: str, max_len: int):
    from transformers import pipeline  # import diferido: es pesado

    clf = pipeline("text-classification", model=fuente, tokenizer=fuente, top_k=None,
                   truncation=True, max_length=max_len, device=-1)

    def predecir(textos: list[str]) -> list[dict]:
        salidas = clf(textos, batch_size=16)
        return [{d["label"]: float(d["score"]) for d in _normalizar(s)} for s in salidas]

    return predecir


def _desde_baseline(ruta: Path):
    import joblib

    modelo = joblib.load(ruta)

    def predecir(textos: list[str]) -> list[dict]:
        probs = modelo.predict_proba(textos)
        return [{ETIQUETAS[int(c)]: float(p) for c, p in zip(modelo.classes_, fila)} for fila in probs]

    return predecir


def cargar_modelo(model_id: str | None = None):
    """Intenta, en orden: carpeta local con BETO → BETO desde Hugging Face Hub → modelo clásico.
    Devuelve (funcion_predecir, descripcion, es_respaldo, avisos)."""
    max_len = int(CONFIG.get("max_len", 128))
    model_id = model_id or os.environ.get("MODEL_ID") or CONFIG.get("model_id", "")
    avisos = []

    local = BASE / "modelo_beto_resenas"
    if (local / "config.json").exists():
        try:
            return _desde_transformers(str(local), max_len), "BETO ajustado (carpeta local)", False, avisos
        except Exception as e:  # noqa: BLE001
            avisos.append(f"No se pudo cargar BETO local: {e}")

    if model_id and "tu-usuario" not in model_id:
        try:
            return (_desde_transformers(model_id, max_len),
                    f"BETO ajustado (Hugging Face: {model_id})", False, avisos)
        except Exception as e:  # noqa: BLE001
            avisos.append(f"No se pudo descargar {model_id}: {e}")
    else:
        avisos.append("No hay un repositorio de Hugging Face configurado en config.json.")

    ruta = BASE / "modelo_base" / "tfidf_logreg.joblib"
    return _desde_baseline(ruta), "Modelo de respaldo (TF-IDF + Regresión Logística)", True, avisos


def clasificar(predecir, textos: list[str]) -> pd.DataFrame:
    textos = [str(t).strip() for t in textos]
    probs = predecir(textos)
    filas = []
    for t, p in zip(textos, probs):
        p = {e: p.get(e, 0.0) for e in ETIQUETAS}
        sent = max(p, key=p.get)
        filas.append({"texto": t, "sentimiento": sent, "confianza": round(p[sent], 4),
                      **{f"p_{e}": round(p[e], 4) for e in ETIQUETAS}})
    return pd.DataFrame(filas)


# ------------------------------------------------------------------ SQLite
def _conectar(ruta: Path) -> sqlite3.Connection:
    ruta.parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(ruta)


def iniciar_historial(ruta: Path = DB_HISTORIAL):
    with _conectar(ruta) as c:
        c.execute("""CREATE TABLE IF NOT EXISTS analisis (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        fecha TEXT NOT NULL,
                        origen TEXT NOT NULL,
                        texto TEXT NOT NULL,
                        sentimiento TEXT NOT NULL,
                        confianza REAL,
                        modelo TEXT)""")


def guardar(df: pd.DataFrame, origen: str, modelo: str, ruta: Path = DB_HISTORIAL):
    iniciar_historial(ruta)
    ahora = datetime.now().isoformat(timespec="seconds")
    filas = [(ahora, origen, r.texto, r.sentimiento, float(r.confianza), modelo) for r in df.itertuples()]
    with _conectar(ruta) as c:
        c.executemany("INSERT INTO analisis (fecha, origen, texto, sentimiento, confianza, modelo) "
                      "VALUES (?, ?, ?, ?, ?, ?)", filas)


def consultar(sql: str, ruta: Path) -> pd.DataFrame:
    if not ruta.exists():
        return pd.DataFrame()
    with _conectar(ruta) as c:
        return pd.read_sql_query(sql, c)


CONSULTAS_DATASET = {
    "Reseñas por partición y sentimiento":
        "SELECT particion, sentimiento, COUNT(*) AS n\nFROM resenas\nGROUP BY particion, sentimiento\n"
        "ORDER BY particion, sentimiento;",
    "Longitud promedio por estrellas":
        "SELECT estrellas, COUNT(*) AS n, ROUND(AVG(n_palabras), 1) AS palabras_promedio\nFROM resenas\n"
        "GROUP BY estrellas\nORDER BY estrellas;",
    "Reseñas neutras más largas":
        "SELECT estrellas, n_palabras, texto\nFROM resenas\nWHERE sentimiento = 'neutro'\n"
        "ORDER BY n_palabras DESC\nLIMIT 10;",
    "Reseñas que mencionan 'envío'":
        "SELECT sentimiento, COUNT(*) AS n\nFROM resenas\nWHERE texto LIKE '%envío%' OR texto LIKE '%envio%'\n"
        "GROUP BY sentimiento;",
}


def cargar_metricas() -> dict:
    ruta = BASE / "resultados" / "metricas.json"
    return json.loads(ruta.read_text(encoding="utf-8")) if ruta.exists() else {}
