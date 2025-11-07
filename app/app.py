import streamlit as st
import io
import numpy as np
from PIL import Image
import requests

# ------------------------------------------------------------
# Configuración de página
# ------------------------------------------------------------
st.set_page_config(
    page_title="Buoy Counter App",
    layout="wide"
)

st.title("Buoy Counter App (3x3)")
st.caption("Conteo automático de boyas usando Roboflow Hosted API (sin OpenCV, compatible con Streamlit Cloud).")

# ------------------------------------------------------------
# Configuración Roboflow (vía Secrets)
# ------------------------------------------------------------
# En Streamlit Cloud:
# Settings → Secrets → añadir:
# ROBOFLOW_API_KEY = "TU_API_KEY_NUEVA"
api_key = st.secrets.get("ROBOFLOW_API_KEY", None)

if not api_key:
    st.error(
        "No se encontró `ROBOFLOW_API_KEY` en los Secrets.\n"
        "Configura la API Key en Settings → Secrets antes de usar la aplicación."
    )
    st.stop()

# Ajusta con tu modelo real en Roboflow
MODEL_ID = "buoy-wn6n2/2"
BASE_URL = f"https://detect.roboflow.com/{MODEL_ID}"


def infer_crop(crop_bytes: bytes):
    """
    Envía un recorte de imagen al endpoint de Roboflow y retorna el JSON de predicciones.
    Sin dependencias pesadas, usando solo requests.
    """
    params = {
        "api_key": api_key,
        "format": "json",
        "size": 1280,
    }

    files = {
        "file": ("image.jpg", crop_bytes, "image/jpeg")
    }

    response = requests.post(BASE_URL, params=params, files=files, timeout=30)
    response.raise_for_status()
    return response.json()


# ------------------------------------------------------------
# Sidebar Parámetros
# ------------------------------------------------------------
with st.sidebar:
    st.header("Parámetros")
    show_debug = st.checkbox("Mostrar detalle de predicciones", value=False)
    st.markdown(
        "- La imagen se divide en una grilla **3x3**.\n"
        "- Cada bloque se envía a Roboflow.\n"
        "- Se ajustan coordenadas al tamaño completo."
    )

# ------------------------------------------------------------
# Carga de imagen
# ------------------------------------------------------------
uploaded_file = st.file_uploader(
    "Sube una imagen de las boyas",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is None:
    st.info("Sube una imagen para iniciar el conteo.")
    st.stop()

# Leemos bytes una sola vez
uploaded_bytes = uploaded_file.read()

# Mostrar imagen original
st.subheader("Imagen cargada")
st.image(uploaded_bytes, use_container_width=True)

# Convertimos a matriz (PIL → numpy)
image = Image.open(io.BytesIO(uploaded_bytes)).convert("RGB")
img = np.array(image)

if img.ndim != 3 or img.shape[2] != 3:
    st.error("Formato de imagen no válido para el procesamiento.")
    st.stop()

h, w, _ = img.shape
h_step = h // 3
w_step = w // 3

all_predictions = []

# ------------------------------------------------------------
# Proceso de detección 3x3
# ------------------------------------------------------------
st.subheader("Detección de boyas")

with st.spinner("Procesando imagen en grilla 3x3 y consultando Roboflow..."):
    try:
        for i in range(3):          # filas
            for j in range(3):      # columnas
                y1 = i * h_step
                y2 = (i + 1) * h_step if i < 2 else h
                x1 = j * w_step
                x2 = (j + 1) * w_step if j < 2 else w

                crop = img[y1:y2, x1:x2]

                # Convertir crop a JPEG en memoria
                crop_img = Image.fromarray(crop)
                buf = io.BytesIO()
                crop_img.save(buf, format="JPEG")
                crop_bytes = buf.getvalue()

                # Llamar Roboflow para este recorte
                result = infer_crop(crop_bytes)
                preds = result.get("predictions", [])

                # Ajustar coordenadas al sistema global de la imagen completa
                for p in preds:
                    if "x" in p and "y" in p:
                        p["x"] += x1
                        p["y"] += y1
                    p["grid_row"] = i
                    p["grid_col"] = j
                    all_predictions.append(p)

        total_boyas = len(all_predictions)
        st.success(f"Total de boyas detectadas: **{total_boyas}**")

        # Detalle opcional
        if show_debug and all_predictions:
            st.write("Detalle de predicciones (máximo 200 registros):")
            rows = []
            for p in all_predictions[:200]:
                rows.append({
                    "class": p.get("class", ""),
                    "conf": round(p.get("confidence", 0), 3),
                    "x": round(p.get("x", 0), 1),
                    "y": round(p.get("y", 0), 1),
                    "w": round(p.get("width", 0), 1),
                    "h": round(p.get("height", 0), 1),
                    "grid_row": p.get("grid_row"),
                    "grid_col": p.get("grid_col"),
                })
            st.dataframe(rows, use_container_width=True)

    except requests.exceptions.HTTPError as http_err:
        st.error(f"Error HTTP desde Roboflow: {http_err}")
    except Exception as e:
        st.error(f"Error durante la inferencia: {e}")
