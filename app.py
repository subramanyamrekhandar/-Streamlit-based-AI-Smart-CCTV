import streamlit as st
import cv2
import numpy as np
import os
import tempfile
import pandas as pd
import mediapipe as mp
from datetime import datetime
from ultralytics import YOLO

# Create a directory for face data
FACE_DATA_DIR = "face_data"
if not os.path.exists(FACE_DATA_DIR):
    os.makedirs(FACE_DATA_DIR)

# Load YOLO Model
model = YOLO("yolov8n.pt")

# Initialize Logs
motion_log = []
object_log = []
noise_log = []
pose_log = []
entry_exit_log = []  # Added missing entry_exit_log

# Streamlit UI
st.title("AI-Based Smart CCTV Surveillance System")
st.sidebar.header("Settings")

video_source = st.sidebar.selectbox("Select Video Source", ("Webcam", "Upload Video"))

if video_source == "Upload Video":
    uploaded_file = st.sidebar.file_uploader("Upload Video", type=["mp4", "avi", "mov"])

selected_module = st.sidebar.selectbox(
    "Select Module",
    ("Motion Detection", "Object Detection", "Noise Detection", "In & Out Detection", "Pose Detection"),
)

# Intrusion Alert UI Notification
alert_placeholder = st.empty()

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

# Function to detect motion
def motion_detection(prev_frame, frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)
    if prev_frame is None:
        return gray, None
    frame_delta = cv2.absdiff(prev_frame, gray)
    thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.dilate(thresh, None, iterations=2)
    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return gray, contours

# Function for Noise Detection
def detect_noise(threshold=500):
    def callback(indata, frames, time, status):
        volume_norm = np.linalg.norm(indata) * 10
        if volume_norm > threshold:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            noise_log.append({"Time": timestamp, "Noise Level": volume_norm})
            alert_placeholder.warning(f"🚨 Noise Detected: {volume_norm:.2f} dB")
    return callback

# Function for In & Out Detection
def detect_entry_exit(frame, previous_count):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter contours by size to reduce false positives
    min_area_threshold = 500  # Adjust based on the scene
    valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area_threshold]
    
    current_count = len(valid_contours)

    if current_count > previous_count:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        entry_exit_log.append({"Time": timestamp, "Event": "Person Entered"})
        alert_placeholder.success("✅ Person Entered")
        # alert_placeholder.error("❌ Person Exited")
    elif current_count < previous_count:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        entry_exit_log.append({"Time": timestamp, "Event": "Person Exited"})
        alert_placeholder.error("❌ Person Exited")
        # alert_placeholder.success("✅ Person Entered")

    return current_count

# Video Processing
prev_frame = None
video_placeholder = st.empty()

if video_source == "Webcam":
    cap = cv2.VideoCapture(0)
elif uploaded_file:
    temp_video = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    temp_video.write(uploaded_file.read())
    temp_video.close()
    cap = cv2.VideoCapture(temp_video.name)
else:
    cap = None

if cap:
    previous_people_count = 0  # Initialize people count before processing frames

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Object Detection
        detected_objects = []
        if selected_module == "Object Detection":
            results = model(frame)
            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = box.conf[0].item()
                    label = result.names[int(box.cls[0].item())]
                    if conf > 0.4:
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, f"{label}: {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        detected_objects.append(label)

        # Log detected objects
        if detected_objects:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            object_log.append({"Time": timestamp, "Objects": ", ".join(detected_objects)})

        # Motion Detection
        if selected_module == "Motion Detection":
            prev_frame, motion_contours = motion_detection(prev_frame, frame)
            if motion_contours:
                for contour in motion_contours:
                    if cv2.contourArea(contour) > 500:
                        x, y, w, h = cv2.boundingRect(contour)
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                        alert_placeholder.warning("🚨 Motion Detected!")
                        motion_log.append({"Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "Alert": "Motion Detected"})

        # Placeholder for Noise Detection
        if selected_module == "Noise Detection":
            cv2.putText(frame, "Noise Detected", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        # In & Out Detection
        if selected_module == "In & Out Detection":
            previous_people_count = detect_entry_exit(frame, previous_people_count)

        # Pose Detection
        if selected_module == "Pose Detection":
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = pose.process(img_rgb)
            if result.pose_landmarks:
                mp_drawing.draw_landmarks(frame, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                alert_placeholder.success("✅ Pose Detected")
                pose_log.append({"Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "Pose Detected": "Yes"})

        video_placeholder.image(frame, channels="BGR")

    cap.release()
    cv2.destroyAllWindows()

# Display Logs
st.sidebar.subheader("Event Logs")
if motion_log or object_log or noise_log or pose_log or entry_exit_log:
    log_data = pd.DataFrame(motion_log + object_log + noise_log + pose_log + entry_exit_log)
    st.sidebar.dataframe(log_data)
else:
    st.sidebar.write("No events recorded yet.")
