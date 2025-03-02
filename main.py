import streamlit as st
import cv2
import numpy as np
import os
import pickle
import face_recognition
import tempfile
import pandas as pd
from datetime import datetime
from ultralytics import YOLO

# Create directory for face data
FACE_DATA_DIR = "face_data"
if not os.path.exists(FACE_DATA_DIR):
    os.makedirs(FACE_DATA_DIR)

# Load YOLO Model
model = YOLO("yolov8n.pt")

# Initialize Logs
motion_log = []
object_log = []
face_log = []

# Streamlit UI
st.title("AI-Based Smart CCTV with Facial Recognition")
st.sidebar.header("Settings")

video_source = st.sidebar.selectbox("Select Video Source", ("Webcam", "Upload Video"))

if video_source == "Upload Video":
    uploaded_file = st.sidebar.file_uploader("Upload Video", type=["mp4", "avi", "mov"])

selected_module = st.sidebar.selectbox(
    "Select Module", ("Motion Detection", "Object Detection", "Face Recognition")
)

# Load or Initialize Face Encodings
data_file = os.path.join(FACE_DATA_DIR, "face_encodings.pkl")
if os.path.exists(data_file):
    with open(data_file, "rb") as f:
        known_face_encodings, known_face_names = pickle.load(f)
else:
    known_face_encodings = []
    known_face_names = []

# Capture Face Data
def add_face(name, frame):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.uint8)
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    if face_encodings:
        known_face_encodings.append(face_encodings[0])
        known_face_names.append(name)

        with open(data_file, "wb") as f:
            pickle.dump((known_face_encodings, known_face_names), f)

        st.sidebar.success(f"✅ Face Registered for {name}")
    else:
        st.sidebar.error("❌ No face detected. Try again.")

# Video Processing
video_placeholder = st.empty()
cap = None

if video_source == "Webcam":
    cap = cv2.VideoCapture(0)
elif uploaded_file:
    temp_video = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    temp_video.write(uploaded_file.read())
    temp_video.close()
    cap = cv2.VideoCapture(temp_video.name)

if cap and cap.isOpened():
    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        # Ensure frame is in correct format
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.uint8)

        # Face Recognition
        if selected_module == "Face Recognition":
            face_locations = face_recognition.face_locations(rgb_frame)
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

            for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                name = "Unknown"

                if True in matches:
                    first_match_index = matches.index(True)
                    name = known_face_names[first_match_index]

                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                face_log.append({"Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "Name": name})

        video_placeholder.image(frame, channels="BGR")

    cap.release()
    cv2.destroyAllWindows()

# Display Logs
st.sidebar.subheader("Event Logs")
if face_log:
    log_data = pd.DataFrame(face_log)
    st.sidebar.dataframe(log_data)
else:
    st.sidebar.write("No face recognition events recorded yet.")

# Face Registration UI
st.sidebar.subheader("Register a New Face")
name_input = st.sidebar.text_input("Enter Name:")
if st.sidebar.button("Capture Face") and cap:
    ret, frame = cap.read()
    if ret:
        add_face(name_input, frame)
