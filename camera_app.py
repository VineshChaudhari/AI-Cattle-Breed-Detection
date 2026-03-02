import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
import pandas as pd
import datetime
import os
from collections import deque
import plotly.express as px

# -------------------------
# CONFIGURATION
# -------------------------
MODEL_PATH = "best.pt"
CONFIDENCE_THRESHOLD = 0.75
SMOOTHING_FRAMES = 5
LOG_FILE = "prediction_logs.csv"

# -------------------------
# LOAD MODEL
# -------------------------
model = YOLO(MODEL_PATH)

st.set_page_config(page_title="AI Livestock Registry System", layout="wide")

st.title("🐄 AI Livestock Breed Identification & Registry System")
st.markdown("### Real-Time Field-Grade Decision Support Tool")

# -------------------------
# SIDEBAR SETTINGS
# -------------------------
st.sidebar.header("System Controls")

farmer_id = st.sidebar.text_input("Enter Farmer ID", "FARMER001")

continuous_mode = st.sidebar.checkbox("Enable Continuous Scanning", True)

confidence_slider = st.sidebar.slider("Confidence Threshold",
                                       0.50, 0.95, 0.75)

# -------------------------
# CAMERA SECTION
# -------------------------
if st.button("Start Camera"):

    cap = cv2.VideoCapture(0)
    frame_placeholder = st.empty()

    prediction_buffer = deque(maxlen=SMOOTHING_FRAMES)

    while cap.isOpened():

        ret, frame = cap.read()
        if not ret:
            break

        # Simulated multi-animal box
        h, w, _ = frame.shape
        cv2.rectangle(frame, (50, 50), (w-50, h-50), (0, 255, 0), 2)

        results = model.predict(frame, verbose=False)
        probs = results[0].probs
        top1 = probs.top1
        confidence = float(probs.top1conf)

        prediction_buffer.append(top1)
        final_prediction = max(set(prediction_buffer), key=prediction_buffer.count)

        breed_name = model.names[final_prediction]

        # UI Overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (frame.shape[1], 120), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)

        # Display Info
        cv2.putText(frame,
                    "AI Livestock Registry",
                    (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 255),
                    2)

        cv2.putText(frame,
                    f"{breed_name} ({confidence*100:.2f}%)",
                    (20, 70),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 255, 0),
                    2)

        frame_placeholder.image(frame, channels="BGR")

        # -------------------------
        # AUTO CAPTURE
        # -------------------------
        if confidence >= confidence_slider:

            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            location = "Pune, Maharashtra, India"

            log_data = {
                "Timestamp": timestamp,
                "Farmer ID": farmer_id,
                "Location": location,
                "Breed": breed_name,
                "Confidence": round(confidence*100, 2)
            }

            df = pd.DataFrame([log_data])

            if os.path.exists(LOG_FILE):
                df.to_csv(LOG_FILE, mode='a', header=False, index=False)
            else:
                df.to_csv(LOG_FILE, index=False)

            image_name = f"capture_{timestamp.replace(':','-')}.jpg"
            cv2.imwrite(image_name, frame)

            st.success(f"Captured & Logged for {farmer_id}")

            if not continuous_mode:
                break

    cap.release()

# -------------------------
# DASHBOARD SECTION
# -------------------------
st.markdown("---")
st.header("Livestock Registration Dashboard")

if os.path.exists(LOG_FILE):

    data = pd.read_csv(LOG_FILE)

    col1, col2 = st.columns(2)

    with col1:
        breed_counts = data["Predicted Breed"].value_counts()
        fig = px.bar(breed_counts,
                     title="Breed Distribution",
                     labels={"value": "Count", "index": "Predicted Breed"})
        st.plotly_chart(fig)

    with col2:
        farmer_counts = data["Farmer ID"].value_counts()
        fig2 = px.pie(values=farmer_counts.values,
                      names=farmer_counts.index,
                      title="Registrations per Farmer")
        st.plotly_chart(fig2)

    st.dataframe(data)

else:
    st.info("No registrations yet.")

st.markdown("Model Accuracy: Top-1 = 70.27% | Top-5 = 98.35%")