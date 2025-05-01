#heyyyyyy

import streamlit as st
import cv2
import numpy as np
import os
import time
import threading
import sys

# Set environment variable to disable weights_only restriction - CRITICAL FIX
os.environ['TORCH_WEIGHTS_ONLY'] = '0'

# Now import torch and YOLO
import torch
from ultralytics import YOLO

# Create audio directory if it doesn't exist
os.makedirs("audio", exist_ok=True)

# Try to import playsound, fallback to simple alert if not available
try:
    from playsound import playsound
    def play_sound(sound_file):
        try:
            threading.Thread(target=playsound, args=(sound_file,), daemon=True).start()
        except Exception as e:
            st.error(f"Error playing sound: {str(e)}")
except ImportError:
    def play_sound(sound_file):
        st.toast(f"Detected: {os.path.basename(sound_file).replace('_audio.wav', '')}")

# Dictionary to track when audio was last played to prevent constant replaying
last_played = {
    "pothole": 0,
    "speed_breaker": 0
}

# Dictionary to track current detections
current_detections = {
    "pothole": False,
    "speed_breaker": False
}

# Set page config
st.set_page_config(
    page_title="Road Hazard Detection",
    page_icon="🚧",
    layout="wide"
)

# Title and description
st.title("Road Hazard Detection System")
st.write("This application detects potholes and speed breakers using your camera feed.")

# Display PyTorch version
st.write(f"PyTorch version: {torch.__version__}")

# Load YOLOv8 model
try:
    model = YOLO("bes_pothole_model.pt")
    st.success("Model loaded successfully!")
except Exception as e:
    st.error(f"Failed to load model: {str(e)}")
    st.warning("Attempting to download a sample model instead...")
    try:
        model = YOLO("yolov8n.pt")
        st.success("Using sample YOLOv8 model instead.")
    except Exception as e2:
        st.error(f"Could not load any model: {str(e2)}")
        st.stop()

# Initialize session state for camera access
if 'camera_access' not in st.session_state:
    st.session_state.camera_access = False

# Camera access button
if not st.session_state.camera_access:
    if st.button("Allow Camera Access"):
        st.session_state.camera_access = True
        st.rerun()

# Main application
if st.session_state.camera_access:
    # Create two columns for video feed and detection results
    col1, col2 = st.columns(2)
    
    with col1:
        st.header("Live Camera Feed")
        # Get camera feed
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            st.error("Failed to open camera. Please check your camera connection.")
            st.stop()
            
        frame_placeholder = st.empty()
        
        # Detection results placeholder
        with col2:
            st.header("Detection Results")
            results_placeholder = st.empty()
        
        # Display information about audio files
        if not os.path.exists("audio/pothole_audio.wav") or not os.path.exists("audio/speedbreaker_audio.wav"):
            st.error("Audio files are missing. Please make sure you have the following files in the audio directory:")
            st.write("- audio/pothole_audio.wav")
            st.write("- audio/speedbreaker_audio.wav")
            st.stop()
        
        # Main detection loop
        while True:
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to read from camera")
                break
                
            # Run YOLO detection
            results = model(frame)
            
            # Process detections
            detection_results = []
            current_time = time.time()
            
            # Reset current detections
            current_detections = {
                "pothole": False,
                "speed_breaker": False
            }
            
            for result in results:
                for box in result.boxes:
                    cls = int(box.cls[0].item())
                    conf = float(box.conf[0].item())
                    
                    # Only process if confidence is high enough
                    if conf > 0.5:
                        # Draw bounding box
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        
                        # Add label
                        label = f"{model.names[cls]} {conf:.2f}"
                        cv2.putText(frame, label, (x1, y1 - 10),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        
                        # Add to detection results
                        detection_results.append((cls, conf, model.names[cls]))
                        
                        # Update current detections and play audio immediately
                        if cls == 0:  # Pothole class
                            current_detections["pothole"] = True
                            if os.path.exists("audio/pothole_audio.wav"):
                                play_sound("audio/pothole_audio.wav")
                        elif cls == 1:  # Speed breaker class
                            current_detections["speed_breaker"] = True
                            if os.path.exists("audio/speedbreaker_audio.wav"):
                                play_sound("audio/speedbreaker_audio.wav")
            
            # Display the frame
            frame_placeholder.image(frame, channels="BGR")
            
            # Display detection results
            with results_placeholder.container():
                if detection_results:
                    st.write("Detected Objects:")
                    for _, conf, name in detection_results:
                        st.write(f"- {name} (Confidence: {conf:.2f})")
                else:
                    st.write("No hazards detected")
            
            # Add a small delay to prevent high CPU usage
            time.sleep(0.1)
            
        # Release resources
        cap.release()
        st.write("Camera released. Refresh the page to restart.") 