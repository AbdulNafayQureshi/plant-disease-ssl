import json
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

st.set_page_config(page_title="Plant Leaf Disease Classifier (SSL CNN)", page_icon="🌿", layout="centered")

IMG_SIZE = (160, 160)
MODEL_PATH = "ssl_plant_disease_cnn.keras"
CLASSES_PATH = "class_names.json"

@st.cache_resource
def load_model_and_classes():
    model = tf.keras.models.load_model(MODEL_PATH)
    with open(CLASSES_PATH, "r", encoding="utf-8") as f:
        class_names = json.load(f)
    return model, class_names

st.title("🌿 Plant Leaf Disease Classification")
st.caption("Semi-Supervised Learning (Teacher-Student CNN with Pseudo-Labeling)")

try:
    model, class_names = load_model_and_classes()
except Exception as e:
    st.error(f"Model ya class_names.json load karne mein masla aaya: {e}")
    st.stop()

uploaded_file = st.file_uploader("Leaf ki image upload karein", type=["jpg", "jpeg", "png", "bmp", "webp"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Leaf Image", use_container_width=True)

    resized = image.resize(IMG_SIZE)
    arr = np.array(resized, dtype=np.float32)
    arr = preprocess_input(arr)
    batch = np.expand_dims(arr, axis=0)

    probs = model.predict(batch, verbose=0)[0]
    top_idx = int(np.argmax(probs))
    top_conf = float(probs[top_idx])

    st.subheader("Prediction Result")
    st.success(f"**Predicted Class:** {class_names[top_idx]}")
    st.write(f"**Confidence:** `{top_conf * 100:.2f}%`")

    st.markdown("### Top 5 Predictions")
    top5_indices = np.argsort(probs)[::-1][:5]
    for i in top5_indices:
        st.write(f"- **{class_names[int(i)]}**: `{float(probs[int(i)]) * 100:.2f}%`")