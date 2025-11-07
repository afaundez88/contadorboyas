import streamlit as st
import io
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import requests

from sklearn.linear_model import RANSACRegressor, LinearRegression

# ------------------------------------------------------------
# Configuración de página
# ------------------------------------------------------------
st.set_page_config(page_title="Buoy Counter App", layout="wide")
st.title("Buoy Counter App (3x3)")
st.caption("Conteo automático de boyas + líneas por RANSAC (sin OpenCV, compatible con Streamlit Cloud).")

# ------------------------------------------------------------
# Config Roboflow (vía Secrets)
# ------------------------------------------------------------
api_key = st.secrets.get("ROBOFLOW_API_KEY", None)
if not api_key:
    st.error("Falta `ROBOFLOW_API_KEY` en Settings → Secrets.")
    st.stop()

MODEL_ID = "buoy-wn6n2/2"
BASE_URL = f"https://detect.roboflow.com/{MODEL_ID}"

def infer_crop(crop_bytes: bytes):
    params = {"api_key": api_key, "format": "json", "size": 1280}
    files = {"file": ("image.jpg", crop_bytes, "image/jpeg")}
    r = requests.post(BASE_URL, params=params, files=files, timeout=60)
    r.raise_for_status()
    return r.json()

# ------------------------------------------------------------
# Sidebar
# ------------------------------------------------------------
with st.sidebar:
    st.header("Parámetros")
    show_debug = st.checkbox("Mostrar detalle de predicciones", value=False)
    tolerancia_px = st.slider(
        "Tolerancia a la recta (px)",  # distancia máx punto-recta para ser inlier
        min_value=3, max_value=60, value=20
    )
    min_points_line = st.slider(
        "Mínimo de boyas por línea", min_value=3, max_value=80, value=12
    )
    max_iter_ransac = st.slider(
        "Iteraciones RANSAC", min_value=50, max_value=2000, value=500, step=50
    )
    st.markdown(
        "- Grilla **3x3** → inferencia por bloque.\n"
        "- Agrupación robusta en líneas con **RANSAC**.\n"
        "- Se dibuja la recta y el **conteo** por línea."
    )

# ------------------------------------------------------------
# Carga imagen
# ------------------------------------------------------------
uploaded_file = st.file_uploader("Sube una imagen de las boyas", type=["jpg", "jpeg", "png"])
if uploaded_file is None:
    st.info("Sube una imagen para iniciar el conteo.")
    st.stop()

uploaded_bytes = uploaded_file.read()
st.subheader("Imagen cargada")
st.image(uploaded_bytes, use_container_width=True)

image = Image.open(io.BytesIO(uploaded_bytes)).convert("RGB")
img = np.array(image)
if img.ndim != 3 or img.shape[2] != 3:
    st.error("Formato de imagen no válido.")
    st.stop()

h, w, _ = img.shape
h_step, w_step = h // 3, w // 3

# ------------------------------------------------------------
# Inferencia 3x3
# ------------------------------------------------------------
st.subheader("Detección de boyas")
all_predictions = []
with st.spinner("Procesando grilla 3x3 + Roboflow..."):
    try:
        for i in range(3):
            for j in range(3):
                y1, y2 = i * h_step, (i + 1) * h_step if i < 2 else h
                x1, x2 = j * w_step, (j + 1) * w_step if j < 2 else w

                crop = img[y1:y2, x1:x2]
                crop_img = Image.fromarray(crop)
                buf = io.BytesIO()
                crop_img.save(buf, format="JPEG")
                crop_bytes = buf.getvalue()

                result = infer_crop(crop_bytes)
                preds = result.get("predictions", [])
                for p in preds:
                    if "x" in p and "y" in p:
                        p["x"] += x1
                        p["y"] += y1
                    all_predictions.append(p)

        st.success(f"Total de boyas detectadas: **{len(all_predictions)}**")

        if show_debug and all_predictions:
            st.write("Muestra (hasta 200):")
            rows = [{
                "conf": round(p.get("confidence", 0), 3),
                "x": round(p.get("x", 0), 1),
                "y": round(p.get("y", 0), 1),
                "w": round(p.get("width", 0), 1),
                "h": round(p.get("height", 0), 1),
                "class": p.get("class", "")
            } for p in all_predictions[:200]]
            st.dataframe(rows, use_container_width=True)

    except requests.exceptions.HTTPError as e:
        st.error(f"Error HTTP Roboflow: {e}")
        st.stop()
    except Exception as e:
        st.error(f"Error de inferencia: {e}")
        st.stop()

# ------------------------------------------------------------
# Agrupación de líneas por RANSAC
# ------------------------------------------------------------
if not all_predictions:
    st.info("Sin detecciones.")
    st.stop()

# Matriz de puntos (x, y)
pts = np.array([[p["x"], p["y"]] for p in all_predictions], dtype=float)

# Lista de grupos (cada grupo es dict con 'inliers_idx', 'a','b')
grupos = []
rest_idx = np.arange(len(pts))

# Funciones auxiliares
def dist_point_to_line(ax, bx, x, y):
    """
    Distancia ortogonal de puntos (x,y) a y = a*x + b
    """
    # Fórmula: |ax - y + b| / sqrt(a^2 + 1)
    return np.abs(ax * x - y + bx) / np.sqrt(ax**2 + 1)

# Itera RANSAC hasta agotar puntos
while len(rest_idx) >= min_points_line:
    # Datos restantes
    X = pts[rest_idx, 0].reshape(-1, 1)  # x
    y_vec = pts[rest_idx, 1]             # y

    # RANSAC robusto
    base = LinearRegression()
    ransac = RANSACRegressor(
        base_estimator=base,
        residual_threshold=tolerancia_px,
        max_trials=max_iter_ransac,
        random_state=42
    )

    try:
        ransac.fit(X, y_vec)
    except ValueError:
        break

    inlier_mask = ransac.inlier_mask_
    if inlier_mask is None or inlier_mask.sum() < min_points_line:
        # No hay una línea válida con los puntos restantes
        break

    # Parámetros de la recta y = a*x + b
    a = float(ransac.estimator_.coef_[0])
    b = float(ransac.estimator_.intercept_)

    # Índices de inliers en el espacio global
    inliers_global_idx = rest_idx[inlier_mask]

    grupos.append({
        "a": a, "b": b,
        "inliers_idx": inliers_global_idx
    })

    # Quitamos inliers y seguimos buscando más líneas
    rest_idx = rest_idx[~inlier_mask]

# ------------------------------------------------------------
# Dibujo de líneas + conteo por línea
# ------------------------------------------------------------
annotated = image.copy()
draw = ImageDraw.Draw(annotated)
try:
    font = ImageFont.truetype("arial.ttf", 28)
except Exception:
    font = ImageFont.load_default()

lineas_validas = 0
for g in grupos:
    idx = g["inliers_idx"]
    if len(idx) < min_points_line:
        continue

    a, b = g["a"], g["b"]
    # Extremos de la línea dentro del ancho de la imagen, usando el rango de los inliers
    xs = pts[idx, 0]
    x_min, x_max = float(xs.min()), float(xs.max())
    y_min = a * x_min + b
    y_max = a * x_max + b

    # Dibujo
    draw.line([(x_min, y_min), (x_max, y_max)], fill="red", width=3)

    # Etiqueta con el conteo
