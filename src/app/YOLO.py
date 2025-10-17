# type: ignore
################################################### Libraries #################################
import streamlit as st        # pip install streamlit

# For Images Data
import numpy as np
import cv2
import requests
from io import BytesIO

from PIL import Image # pip install pillow

import pandas as pd

# Model
from ultralytics import YOLO

# Warnings
import warnings
warnings.filterwarnings("ignore")

################################### Loading Trained & Saved Best.pt file #########################
# load model, set cache to prevent reloading
@st.cache_resource
def load_model():
    model = YOLO('best.pt')
    return model

with st.spinner("Loading Model...."):
    try:
        model = load_model()
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        st.stop()

####################################### Helper Function #######################################
def detect(uploaded):
    # read image
    image = Image.open(uploaded).convert("RGB")
    img = np.array(image)  # RGB

    # run prediction
    with st.spinner("Running YOLO inference..."):
        # ultralytics accepts numpy arrays as source
        results = model.predict(source=img, imgsz=640, conf=conf_thresh)

    # results is an iterable (usually length 1 for single image)
    if len(results[0].boxes) == 0:
        st.write("##### :blue[Detections:]")
        st.markdown("<hr style='margin-top: 1px;'>", unsafe_allow_html=True)
        st.warning("No Detections for the given Threshold.")
        st.stop()

    res = results[0]

    # copy image for drawing (OpenCV expects BGR)
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    detections = []
    # iterate boxes (robust to ultralytics versions)
    for box in getattr(res, "boxes", []):
        # try several ways to extract values depending on ultralytics version
        try:
            xyxy = box.xyxy[0].tolist()          # tensor-like
            conf = float(box.conf[0])
            cls = int(box.cls[0])
        except Exception:
            # fallback if res.boxes provides arrays
            try:
                xyxy = box.xyxy.tolist()[0]
                conf = float(box.conf.tolist()[0])
                cls = int(box.cls.tolist()[0])
            except Exception:
                # final fallback: skip this box
                continue

        x1, y1, x2, y2 = map(int, xyxy)
        label = model.names.get(cls, str(cls)) if hasattr(model, "names") else str(cls)
        text = f"{label} {conf:.2f}"
        # draw rectangle + label
        cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (0, 255, 255), 3)
        cv2.putText(img_bgr, text.title(), (x1, max(y1 - 6, 0)), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,255,255), 3)

        detections.append({"label": label, "confidence": round(conf, 4)})

    # convert back to RGB for display
    annotated = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    st.write("##### :blue[Detections:]")
    st.markdown("<hr style='margin-top: 1px;'>", unsafe_allow_html=True)
    st.image(annotated, use_container_width=True)

################################################ UI ###########################################
st.set_page_config(page_title="DL", layout="centered")

st.subheader(":green[🐟 🐙 UnderWater Species Detection 🦈]")
st.markdown("<hr style='margin-top: 1px;'>", unsafe_allow_html=True)
st.write("YOLOv11 Model Trained on Below Classes....")
# Classes with icons
classes = {
    "hammer": "🐟",
    "knife": "🦑",
    "sapnners": "🐧",
    "sewedrivers": "🐦‍⬛",
    "scissor": "🦈",
    "drill machine": "⭐",
    "pliers": "🐠"
}

# Display horizontally
st.markdown(
    "<div style='text-align: center; font-size: 18px;'>"
    + " &nbsp;&nbsp; ".join([f"{icon} {name}" for name, icon in classes.items()])
    + "</div>",
    unsafe_allow_html=True
)
st.markdown("<br>", unsafe_allow_html=True)

st.write('##### :blue[📤 Upload Image or Enter the Url of Image:]')
st.markdown("<hr style='margin-top: 1px;'>", unsafe_allow_html=True)
col1, col2 , col3 = st.columns([1,0.2,1])
with col1:
    uploaded = st.file_uploader("Select One Image/Multiple Images", type=['png','jpeg','jpg'], accept_multiple_files=True)
with col2:
    st.write(":blue[Or]")
with col3:
    url = st.text_input("Enter Image Url Here:")

conf_thresh = st.slider("Select Confidence Threshold For YOLO", 0.0, 1.0, 0.25)

colx, coly, colz = st.columns([1,0.8,0.5])
with coly:
    submit = st.button('Detect', type="primary")

if not (uploaded or url):
    st.info("Upload an image to run prediction.")
    st.stop()

if submit:
    col1, col2 = st.columns([0.5,0.5])

    # Image Display
    with col1:
        st.write("##### :green[Given Image:]")
        st.markdown("<hr style='margin-top: 1px;'>", unsafe_allow_html=True)
        if url!='':
            response = requests.get(url)
            st.image(Image.open(BytesIO(response.content)))
        else:
            for pic in uploaded:
                image = Image.open(pic)
                st.image(image)

    # Prediction & Display
    with col2:
        # with st.spinner('Analyzing...'):
        # For Url Readed Image
        if url!='':
            response = requests.get(url)
            uploaded  = BytesIO(response.content)
            detect(uploaded)

        # For Multiple Uploaded Images
        else:
            for image in uploaded:
                detect(image)


