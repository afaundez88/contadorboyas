import streamlit as st
import io
import numpy as np
from PIL import Image
import requests

# Roboflow client config
client = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="8J49lXK8I0S8aq9Zojqb"   # 👈 API Key خودت
)

st.title("🌊 Buoy Counter App (3x3 High-Resolution)")

# Upload image
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # نمایش تصویر آپلودی
    st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)

    # ذخیره فایل آپلودی روی دیسک
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name

    # مرحله inference با تقسیم 3x3
    with st.spinner("Detecting buoys..."):
        try:
            img = cv2.imread(tmp_path)
            h, w, _ = img.shape

            # ابعاد هر بخش
            h_step = h // 3
            w_step = w // 3

            all_predictions = []

            # تقسیم به 9 بخش
            for i in range(3):      # rows
                for j in range(3):  # cols
                    y1, y2 = i * h_step, (i + 1) * h_step if i < 2 else h
                    x1, x2 = j * w_step, (j + 1) * w_step if j < 2 else w
                    crop = img[y1:y2, x1:x2]

                    # ذخیره موقت هر بخش
                    crop_path = tmp_path.replace(".jpg", f"_crop_{i}{j}.jpg")
                    cv2.imwrite(crop_path, crop)

                    # inference روی هر بخش
                    result = client.infer(
                        crop_path,
                        model_id="buoy-wn6n2/2?size=1280"
                    )

                    preds = result.get("predictions", [])

                    # اصلاح مختصات به کل تصویر
                    for p in preds:
                        if "x" in p and "y" in p:
                            p["x"] += x1
                            p["y"] += y1
                        all_predictions.append(p)

            # نمایش فقط تعداد
            st.write(f"**Total buoys detected:** {len(all_predictions)}")

        except Exception as e:
            st.error(f"Error during inference: {e}")
