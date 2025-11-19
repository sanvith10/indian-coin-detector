import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image

st.set_page_config(page_title="Indian Coin Detector", layout="wide")
st.title("🇮🇳 Indian Coin + Emblem Detector")
st.markdown("**Upload image or use camera → instant detection of ₹1, ₹2, ₹5, ₹10 + emblem**")

# Load model (caches automatically)
@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()
conf = st.slider("Confidence Threshold", 0.1, 1.0, 0.4, 0.05)

CLASSES = ['1', '10', '2', '5', 'emblem']
COLORS = [(0,255,255), (255,165,0), (0,255,0), (255,215,0), (255,0,255)]

def detect(img):
    results = model(img, conf=conf, verbose=False)
    img_copy = img.copy()
    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls = int(box.cls[0])
        score = box.conf[0]
        label = f"{CLASSES[cls]} ₹" if cls < 4 else CLASSES[cls]
        color = COLORS[cls]
        cv2.rectangle(img_copy, (x1, y1), (x2, y2), color, 5)
        cv2.putText(img_copy, f"{label} {score:.2f}", (x1, y1-10),
                    cv2.FONT_HERSHEY_DUPLEX, 1.3, (0,0,0), 6)
        cv2.putText(img_copy, f"{label} {score:.2f}", (x1, y1-10),
                    cv2.FONT_HERSHEY_DUPLEX, 1.3, (255,255,255), 4)
    return img_copy

# Layout
col1, col2 = st.columns(2)

with col1:
    st.header("📁 Upload Image")
    uploaded = st.file_uploader("Choose JPG/PNG", type=["jpg", "jpeg", "png"])
    if uploaded:
        img = cv2.imdecode(np.frombuffer(uploaded.read(), np.uint8), 1)
        result = detect(img)
        st.image(cv2.cvtColor(result, cv2.COLOR_BGR2RGB), caption="Detected Coins!", use_column_width=True)

with col2:
    st.header("📷 Live Camera")
    picture = st.camera_input("Take a photo")
    if picture:
        img = cv2.imdecode(np.frombuffer(picture.getvalue(), np.uint8), 1)
        result = detect(img)
        st.image(cv2.cvtColor(result, cv2.COLOR_BGR2RGB), caption="Live Detection!", use_column_width=True)

st.success("🚀 Works on phones & laptops worldwide!")
st.caption("Made by sanvith10 • Powered by YOLOv8 • 2025")
