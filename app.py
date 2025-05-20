# app.py
from flask import Flask, render_template, request, jsonify, session
import cv2
import numpy as np
import dlib
from scipy.spatial import distance
from imutils import face_utils

app = Flask(__name__)
app.secret_key = "replace_with_a_strong_random_secret_key"

# Face detector and landmark predictor
DETECTOR = dlib.get_frontal_face_detector()
PREDICTOR = dlib.shape_predictor("model/shape_predictor_68_face_landmarks.dat")
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]

# Calibration & detection parameters
DEFAULT_EAR_THRESH    = 0.25
CALIBRATION_FRAMES    = 30
CALIBRATION_RATIO     = 0.75
FRAME_CHECK           = 4

def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

@app.route('/')
def index():
    # Reset calibration session data on page refresh
    session.pop('ear_vals', None)
    session.pop('ear_thresh', None)
    session.pop('flag', None)
    return render_template('index.html')

@app.route('/calibrate', methods=['POST'])
def calibrate():
    ear_vals = session.get('ear_vals', [])

    if 'frame' not in request.files:
        return jsonify({'error': 'No frame provided'}), 400

    file = request.files['frame']
    img_data = file.read()
    npimg    = np.frombuffer(img_data, np.uint8)
    frame    = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    if frame is None:
        return jsonify({'error': 'Invalid image'}), 400

    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = DETECTOR(gray, 0)
    if not faces:
        return jsonify({
            'calibrated': False,
            'collected':  len(ear_vals),
            'needed':     CALIBRATION_FRAMES
        }), 200

    # compute EAR for first face
    shape    = PREDICTOR(gray, faces[0])
    coords   = face_utils.shape_to_np(shape)
    leftEye  = coords[lStart:lEnd]
    rightEye = coords[rStart:rEnd]
    ear      = (eye_aspect_ratio(leftEye) + eye_aspect_ratio(rightEye)) / 2.0

    ear_vals.append(ear)
    session['ear_vals'] = ear_vals

    if len(ear_vals) >= CALIBRATION_FRAMES:
        baseline  = float(np.mean(ear_vals))
        threshold = baseline * CALIBRATION_RATIO
        session['ear_thresh'] = threshold
        session.pop('ear_vals', None)
        session['flag'] = 0
        return jsonify({'calibrated': True, 'threshold': threshold}), 200

    return jsonify({
        'calibrated': False,
        'collected':  len(ear_vals),
        'needed':     CALIBRATION_FRAMES
    }), 200

@app.route('/process', methods=['POST'])
def process_frame():
    ear_thresh = session.get('ear_thresh', DEFAULT_EAR_THRESH)

    if 'frame' not in request.files:
        return jsonify({'error': 'No frame provided'}), 400

    file     = request.files['frame']
    img_data = file.read()
    npimg    = np.frombuffer(img_data, np.uint8)
    frame    = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    if frame is None:
        return jsonify({'error': 'Invalid image'}), 400

    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = DETECTOR(gray, 0)

    ear_value = None
    alert     = False
    flag      = session.get('flag', 0)

    if faces:
        shape    = PREDICTOR(gray, faces[0])
        coords   = face_utils.shape_to_np(shape)
        leftEye  = coords[lStart:lEnd]
        rightEye = coords[rStart:rEnd]
        leftEAR  = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        ear_value = (leftEAR + rightEAR) / 2.0

        if ear_value < ear_thresh:
            flag += 1
            if flag >= FRAME_CHECK:
                alert = True
        else:
            flag = 0

    session['flag'] = flag

    return jsonify({
        'ear':       ear_value,
        'threshold': ear_thresh,
        'alert':     alert
    }), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
