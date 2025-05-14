import cv2
import numpy as np
import streamlit as st
import dlib
import imutils
from scipy.spatial import distance
from imutils import face_utils
from pygame import mixer
import time
import os

st.set_page_config(
    page_title="Drowsiness Detection",
    page_icon="😴",
    layout="wide"
)

mixer.init()

# Initialize dlib detectors
detect = dlib.get_frontal_face_detector()
try:
    predict = dlib.shape_predictor("models/model.dat")
except RuntimeError:
    st.error("Error: Could not load model.dat. Please ensure the file exists in the 'models' directory.")
    st.stop()

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]
(mStart, mEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["mouth"]

def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

def mouth_aspect_ratio(mouth):
    A = distance.euclidean(mouth[2], mouth[10])
    B = distance.euclidean(mouth[4], mouth[8])
    C = distance.euclidean(mouth[0], mouth[6])
    mar = (A + B) / (2.0 * C)
    return mar

def play_alarm():
    try:
        mixer.music.load("music.wav")
        mixer.music.play(-1)
    except Exception as e:
        st.error(f"Error playing alarm: {e}")

def stop_alarm():
    try:
        mixer.music.stop()
    except:
        pass

def prepare_image(frame):
    if frame is None or frame.size == 0:
        return None
    frame = frame.astype(np.uint8)
    if frame.max() > 255 or frame.min() < 0:
        frame = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    return frame

def detect_drowsiness(frame, ear_threshold, mar_threshold, frame_check, fps, frame_width):
    frame = prepare_image(frame)
    if frame is None:
        return None, "Error", 0, 0
    
    frame = imutils.resize(frame, width=frame_width)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    subjects = detect(gray, 0)
    
    status = "Awake"
    flag_ear = st.session_state.get('flag_ear', 0)
    flag_mar = st.session_state.get('flag_mar', 0)
    
    if not subjects:
        cv2.putText(frame, "No Face Detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        return frame, status, flag_ear, flag_mar
    
    for subject in subjects:
        shape = predict(gray, subject)
        shape = face_utils.shape_to_np(shape)
        
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        mouth = shape[mStart:mEnd]
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        ear = (leftEAR + rightEAR) / 2.0
        mar = mouth_aspect_ratio(mouth)
        
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        mouthHull = cv2.convexHull(mouth)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [mouthHull], -1, (0, 255, 255), 1)
        
        is_eyes_closed = ear < ear_threshold
        is_yawning = mar > mar_threshold
        
        if is_eyes_closed:
            flag_ear += 1
        else:
            flag_ear = 0
        
        if is_yawning:
            flag_mar += 1
        else:
            flag_mar = 0
        
        if flag_ear >= frame_check or flag_mar >= frame_check:
            status = "DROWSY!"
            cv2.putText(frame, "DROWSINESS ALERT!", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            if not mixer.music.get_busy():
                play_alarm()
        else:
            if flag_ear == 0 and flag_mar == 0:
                stop_alarm()
        
        cv2.putText(frame, f"EAR: {ear:.2f}", (frame_width - 150, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        cv2.putText(frame, f"MAR: {mar:.2f}", (frame_width - 150, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        cv2.putText(frame, f"Status: {status}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 
                   (0, 255, 0) if status == "Awake" else (0, 0, 255), 2)
    
    st.session_state.flag_ear = flag_ear
    st.session_state.flag_mar = flag_mar
    return frame, status, flag_ear, flag_mar

def main():
    if 'ear_threshold' not in st.session_state:
        st.session_state.ear_threshold = 0.25
    if 'mar_threshold' not in st.session_state:
        st.session_state.mar_threshold = 0.6
    if 'frame_check' not in st.session_state:
        st.session_state.frame_check = 20
    if 'running' not in st.session_state:
        st.session_state.running = False
    if 'flag_ear' not in st.session_state:
        st.session_state.flag_ear = 0
    if 'flag_mar' not in st.session_state:
        st.session_state.flag_mar = 0
    if 'frame_width' not in st.session_state:
        st.session_state.frame_width = 450

    st.title("Drowsiness Detection System")
    st.markdown("""
    This system monitors drowsiness using Eye Aspect Ratio (EAR) and Mouth Aspect Ratio (MAR) with dlib's facial landmark detection.
    Adjust parameters for accurate detection.
    """)

    st.sidebar.header("Detection Parameters")
    ear_threshold = st.sidebar.slider("EAR Threshold", 0.1, 0.4, 0.25, 0.01)
    mar_threshold = st.sidebar.slider("MAR Threshold (Yawn)", 0.4, 0.8, 0.6, 0.01)
    frame_check = st.sidebar.slider("Frame Check Count", 10, 50, 20, 1)
    frame_width = st.sidebar.slider("Frame Width (pixels)", 300, 800, 450, 10)

    debug_mode = st.sidebar.checkbox("Debug Mode", value=True)

    if not os.path.isfile("music.wav"):
        st.warning("Warning: 'music.wav' not found. Place an alarm sound file in the directory.")

    video_placeholder = st.empty()
    status_placeholder = st.empty()
    debug_placeholder = st.empty()

    col1, col2 = st.columns(2)
    start_button = col1.button("Start Detection")
    stop_button = col2.button("Stop Detection")

    if start_button:
        st.session_state.running = True
    if stop_button:
        st.session_state.running = False
        stop_alarm()
        st.session_state.flag_ear = 0
        st.session_state.flag_mar = 0
        st.rerun()

    if st.session_state.running:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("Error: Could not open webcam.")
            return

        prev_time = time.time()
        while cap.isOpened() and st.session_state.running:
            ret, frame = cap.read()
            if not ret or frame is None:
                st.error("Failed to grab frame.")
                break

            curr_time = time.time()
            fps = 1 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 10
            prev_time = curr_time

            frame, status, flag_ear, flag_mar = detect_drowsiness(frame, ear_threshold, mar_threshold, frame_check, fps, frame_width)

            if frame is None:
                continue

            if status == "DROWSY!":
                status_placeholder.error(f"Status: {status} - EAR Frames: {flag_ear}, MAR Frames: {flag_mar}")
            else:
                status_placeholder.success(f"Status: {status}")

            if debug_mode:
                debug_text = f"FPS: {fps:.1f}\nEAR Flag: {flag_ear}\nMAR Flag: {flag_mar}"
                debug_placeholder.text(debug_text)
            else:
                debug_placeholder.empty()

            video_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")

        cap.release()
        stop_alarm()

    
    st.markdown("---")
    st.markdown("""
**Name:** Subhadeep Pramanik  
**ID:** B221054  
**Branch:** ETC
""")

if __name__ == "__main__":
    main()

