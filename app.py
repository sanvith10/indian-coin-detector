
import streamlit as st

from ultralytics import YOLO

import cv2

import numpy as np

st.set_page_config(page_title="Indian Coin Detector", layout="centered")

st.title("Indian Coin + Emblem Detector")

st.markdown("Upload a photo → instantly detect ₹1, ₹2, ₹5, ₹10 coins & emblem")

model = YOLO("best.pt")

model.conf = st.slider("Confidence Threshold", 0.1, 1.0, 0.4, 0.05)

CLASSES = ['1', '10', '2', '5', 'emblem']

COLORS = [(0,255,255), (255,165,0), (0,255,0), (255,215,0), (255,0,255)]

def detect(img):

    results = model(img, verbose=False)

    for box in results[0].boxes:

        x1,y1,x2,y2 = map(int, box.xyxy[0])

        cls = int(box.cls[0])

        conf = box.conf[0]

        label = f"{CLASSES[cls]} ₹" if cls < 4 else CLASSES[cls]

        cv2.rectangle(img, (x1,y1), (x2,y2), COLORS[cls], 5)

        cv2.putText(img, f"{label} {conf:.2f}", (x1, y1-15),

                    cv2.FONT_HERSHEY_DUPLEX, 1.3, (0,0,0), 6)

        cv2.putText(img, f"{label} {conf:.2f}", (x1, y1-15),

                    cv2.FONT_HERSHEY_DUPLEX, 1.3, (255,255,255), 4)

    return img

uploaded = st.file_uploader("Upload coin image", type=["jpg","jpeg","png"])

if uploaded:

    img = cv2.imdecode(np.frombuffer(uploaded.read(), np.uint8), 1)

    result = detect(img.copy())

    st.image(cv2.cvtColor(result, cv2.COLOR_BGR2RGB), use_column_width=True)

    st.success("Detection complete!")

st.caption("Made with YOLOv8 • Works on phone too!")

