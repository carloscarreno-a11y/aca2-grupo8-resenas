# App Streamlit – ACA 2 · Grupo 8 (CUN)

Análisis de sentimiento de reseñas de comercio electrónico con **BETO** (Transformer en español),
ajustado en el notebook `ACA2_Sentimiento_Resenas_BETO.ipynb`.

## Estructura
| Archivo / carpeta | Para qué sirve |
|---|---|
| `app.py` | Interfaz (5 pestañas: reseña individual, lotes, SQL, desempeño, acerca de) |
| `utils.py` | Carga del modelo con respaldo, predicción e historial en SQLite |
| `config.json` | Repositorio del modelo en Hugging Face (`model_id`) e integrantes |
| `requirements.txt` | Dependencias (misma versión de scikit-learn que en Colab) |
| `data/resenas.db` | Base SQLite del dataset (40 000 reseñas) creada en el notebook |
| `modelo_base/tfidf_logreg.joblib` | Modelo clásico de respaldo (TF-IDF + Regresión Logística) |
| `resultados/` | Métricas reales (`metricas.json`) y reseñas de ejemplo para la demo por lotes |
| `figs/` | Figuras que muestra la pestaña de desempeño |

## Cómo carga el modelo
1. Si existe la carpeta `modelo_beto_resenas/` junto a `app.py`, usa BETO desde ahí.
2. Si no, descarga BETO desde Hugging Face Hub usando `model_id` de `config.json`.
3. Si nada de lo anterior funciona, usa el **modelo clásico de respaldo** y lo avisa en la barra lateral.

La app funciona desde el primer momento con el respaldo. Para usar BETO hay que publicar el modelo (paso A).

## A. Publicar el modelo BETO en Hugging Face (una sola vez)
Pesa ~420 MB, demasiado para GitHub; por eso va al Hub.
1. Crear cuenta en https://huggingface.co y un token con permiso **Write** (Settings → Access Tokens).
2. En Colab, abrir el notebook, ir a **Secretos** (icono de llave), agregar `HF_TOKEN` con el token y activar el acceso del notebook.
3. En la celda 5.2 poner `SUBIR_A_HF = True` y `REPO_HF = "su-usuario/beto-sentimiento-resenas-es"`.
4. Entorno de ejecución con GPU → **Ejecutar todo** (~15 min). Al final se publica el modelo.
5. En `config.json`, reemplazar `tu-usuario/beto-sentimiento-resenas-es` por el repositorio real.

> Si la sesión de Colab donde entrenaron sigue abierta (aún existe `modelo_beto_resenas/`), basta con hacer los pasos 2–3 y ejecutar solo la celda 5.2.

## B. Probar en local
```bash
pip install -r requirements.txt
streamlit run app.py
```

## C. Publicar en Streamlit Community Cloud
1. Crear un repositorio **público** en GitHub y subir **todo el contenido de esta carpeta** (que `app.py` quede en la raíz).
2. Entrar a https://share.streamlit.io con la cuenta de GitHub → **Create app** → elegir el repositorio, rama `main`, archivo `app.py`.
3. En **Advanced settings** elegir **Python 3.12**.
4. **Deploy**. La primera vez tarda varios minutos (instala PyTorch y descarga BETO).

### Si algo falla
- *Se queda en "Your app is in the oven" y el log dice `Building scikit-learn==1.6.1`*: la app quedó con una versión de Python más nueva que 3.12. Borrar la app en share.streamlit.io y crearla de nuevo eligiendo **Python 3.12** en Advanced settings.
- *Aviso "No hay un repositorio de Hugging Face configurado"*: poner en `config.json` el nombre real del modelo (`usuario/beto-sentimiento-resenas-es`) y verificar que el modelo sea **público** en Hugging Face.
- *La instalación falla en `torch`*: quitar la primera línea de `requirements.txt` (`--extra-index-url …`) y volver a desplegar.
- *La app se reinicia o se queda sin memoria al cargar BETO*: el plan gratuito tiene poca RAM. Dejar `model_id` vacío para usar el respaldo, o probar BETO en local.
- *El historial se borra*: es normal en Streamlit Cloud; el archivo `data/historial.db` se reinicia cuando la app se reinicia.
- Ver el error exacto: **Manage app** (abajo a la derecha) → logs.
