import streamlit as st
import io
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import requests

# ------------------------------------------------------------
# Configuración de página
# ------------------------------------------------------------
st.set_page_config(
    page_title="Buoy Counter App",
    layout="wide"
)

st.title("Buoy Counter App (3x3)")
st.caption(
    "Conteo automático de boyas usando Roboflow Hosted API. "
    "Compatible con Streamlit Cloud (sin OpenCV ni librerías pesadas)."
)

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
    """
    params = {
        "api_key": api_key,
        "format": "json",
        "size": 1280,
    }

    files = {
        "file": ("image.jpg", crop_bytes, "image/jpeg")
    }

    response = requests.post(BASE_URL, params=params, files=files, timeout=60)
    response.raise_for_status()
    return response.json()


# ------------------------------------------------------------
# Sidebar Parámetros
# ------------------------------------------------------------
with st.sidebar:
    st.header("Parámetros")
    show_debug = st.checkbox("Mostrar detalle de predicciones", value=False)
    threshold_y = st.slider(
        "Sensibilidad agrupación por línea (px)",
        min_value=5,
        max_value=80,
        value=25,
        help="Valores más bajos: más líneas separadas. Valores altos: agrupa más boyas en una misma línea."
    )
    min_points_line = st.slider(
        "Mínimo de boyas para considerar una línea",
        min_value=2,
        max_value=50,
        value=5,
        help="Filtra líneas con muy pocas detecciones (ruido)."
    )
    st.markdown(
        "- La imagen se divide en una grilla **3x3**.\n"
        "- Cada bloque se envía a Roboflow.\n"
        "- Se ajustan coordenadas al tamaño completo.\n"
        "- Se agrupan detecciones por filas (coordenada Y)."
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

        # --------------------------------------------------------------------
        # Detalle tabular opcional
        # --------------------------------------------------------------------
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

        # --------------------------------------------------------------------
        # Dibujo de líneas con conteo por línea
        # --------------------------------------------------------------------
        if all_predictions:
            # Ordenamos detecciones por Y
            preds_sorted = sorted(all_predictions, key=lambda p: p["y"])

            grupos = []
            for p in preds_sorted:
                if not grupos:
                    grupos.append([p])
                    continue

                # promedio Y del último grupo
                last_group = grupos[-1]
                avg_y = np.mean([g["y"] for g in last_group])

                if abs(p["y"] - avg_y) <= threshold_y:
                    last_group.append(p)
                else:
                    grupos.append([p])

            # Imagen para anotar
            annotated = image.copy()
            draw = ImageDraw.Draw(annotated)

            # Fuente
            try:
                font = ImageFont.truetype("arial.ttf", 28)
            except Exception:
                font = ImageFont.load_default()

            line_index = 1
            for group in grupos:
                if len(group) < min_points_line:
                    continue  # filtra ruido

                xs = [p["x"] for p in group]
                ys = [p["y"] for p in group]
                x_min, x_max = min(xs), max(xs)
                y_avg = int(np.mean(ys))

                # Línea roja sobre la fila
                draw.line(
                    [(x_min, y_avg), (x_max, y_avg)],
                    fill="red",
                    width=3,
                )

                # Texto con el conteo de boyas de esa línea
                label = str(len(group))
                text_x = x_min
                text_y = max(y_avg - 40, 0)
                draw.text((text_x, text_y), label, fill="red", font=font)

                line_index += 1

            st.subheader("Líneas detectadas con conteo por línea")
            st.image(annotated, use_container_width=True)

        else:
            st.info("No se detectaron boyas en la imagen con la configuración actual del modelo.")

    except requests.exceptions.HTTPError as http_err:
        st.error(f"Error HTTP desde Roboflow: {http_err}")
    except Exception as e:
        st.error(f"Error durante la inferencia o el post-procesamiento: {e}")
