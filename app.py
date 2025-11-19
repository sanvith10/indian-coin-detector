import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np

st.set_page_config(page_title="Indian Coin Detector", layout="centered")
st.title("Indian Coin + Emblem Detector")
st.markdown("**Upload or use camera → instant ₹ detection**")

@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()
conf = st.slider("Confidence", 0.1, 1.0, 0.4, 0.05)

CLASSES = ['1', '10', '2', '5', 'emblem']
COLORS = [(0,255,255),(255,165,0),(0,255,0),(255,215,0),(255,0,255)]

def detect(img):
    results = model(img, conf=conf, verbose=False)
    for box in results[0].boxes:
        x1,y1,x2,y2 = map(int, box.xyxy[0])
        cls = int(box.cls[0])
        c = box.conf[0]
        label = f"{CLASSES[cls]} ₹" if cls<4 else CLASSES[cls]
        cv2.rectangle(img,(x1,y1),(x2,y2),COLORS[cls],5)
        cv2.putText(img,f"{label} {c:.2f}",(x1,y1-10),
                    cv2.FONT_HERSHEY_DUPLEX,1.3,(0,0,0),6)
        cv2.putText(img,f"{label} {c:.2f}",(x1,y1-10),
                    cv2.FONT_HERSHEY_DUPLEX,1.3,(255,255,255),4)
    return img

col1, col2 = st.columns(2)
with col1:
    uploaded = st.file_uploader("Upload image", type=["jpg","jpeg","png"])
    if uploaded:
        img = cv2.imdecode(np.frombuffer(uploaded.read(), np.uint8), 1)
        st.image(cv2.cvtColor(detect(img.copy()), cv2.COLOR_BGR2RGB), use_column_width=True)

with col2:
    pic = st.camera_input("Live Camera")
    if pic:
        img = cv2.imdecode(np.frombuffer(pic.getvalue(), np.uint8), 1)
        st.image(cv2.cvtColor(detect(img.copy()), cv2.COLOR_BGR2RGB), use_column_width=True)

st.success("Live worldwide! Works on phones too")
