# inference_module.py
import os
import cv2
import math
import numpy as np
from ultralytics import YOLO
import mediapipe as mp
import time
from dotenv import load_dotenv

load_dotenv()

MODEL_PATH = os.getenv("YOLO_MODEL_PATH", "yolov8s.pt")
FALL_CONFIDENCE_THRESHOLD = int(os.getenv("FALL_CONFIDENCE_THRESHOLD", "80"))  # percent
SAMPLE_RATE = int(os.getenv("VIDEO_SAMPLE_RATE", "5"))  # sample every N frames for video

# Globals filled by load_model_if_needed()
_model = None
_classnames = []
_pose = None
_mp_pose = mp.solutions.pose
_mp_drawing = mp.solutions.drawing_utils

def load_model_if_needed():
    global _model, _classnames, _pose
    if _model is None:
        # Load classnames
        classes_file = os.getenv("CLASSES_FILE", "classes.txt")
        if os.path.exists(classes_file):
            with open(classes_file, "r") as f:
                _classnames = [c.strip() for c in f.read().splitlines()]
        else:
            _classnames = []
        # Load YOLO
        _model = YOLO(MODEL_PATH)
        # MediaPipe pose
        _pose = _mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    return

def calculate_angle(a, b, c):
    a = np.array(a); b = np.array(b); c = np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    if angle > 180.0:
        angle = 360-angle
    return angle

def detect_fall_advanced(landmarks):
    # safe extraction; landmarks are MediaPipe landmarks (normalized)
    try:
        left_shoulder = landmarks[_mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        right_shoulder = landmarks[_mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        left_hip = landmarks[_mp_pose.PoseLandmark.LEFT_HIP.value]
        right_hip = landmarks[_mp_pose.PoseLandmark.RIGHT_HIP.value]
        shoulder_center_y = (left_shoulder.y + right_shoulder.y) / 2
        hip_center_y = (left_hip.y + right_hip.y) / 2
        body_vertical = abs(shoulder_center_y - hip_center_y)
        hip_height = 1.0 - hip_center_y
        if body_vertical < 0.15 and hip_height < 0.3:
            return True, "horizontal", 0.95
        if hip_height < 0.2:
            return True, "collapsed", 0.90
        return False, None, 0.0
    except Exception:
        return False, None, 0.0

def infer_frame_np(frame_bgr):
    """
    Input: BGR numpy array (OpenCV image)
    Output: dict {fall: bool, type, confidence}
    """
    load_model_if_needed()
    try:
        # resize to speed up
        small = cv2.resize(frame_bgr, (640, 480))
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        yres = _model(small)
        person_detected = False
        # YOLO detection loop
        for info in yres:
            for box in info.boxes:
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                name = _classnames[cls] if cls < len(_classnames) else str(cls)
                if name == "person" and (conf * 100) > FALL_CONFIDENCE_THRESHOLD:
                    person_detected = True
                    break
            if person_detected:
                break

        if person_detected:
            pose_res = _pose.process(rgb)
            if pose_res.pose_landmarks:
                is_fallen, fall_type, conf = detect_fall_advanced(pose_res.pose_landmarks.landmark)
                return {"fall": bool(is_fallen), "type": fall_type, "confidence": float(conf)}
        return {"fall": False, "type": None, "confidence": 0.0}
    except Exception as e:
        return {"fall": False, "type": None, "confidence": 0.0, "error": str(e)}

def infer_video_path(path, sample_rate: int = None):
    load_model_if_needed()
    if sample_rate is None:
        sample_rate = SAMPLE_RATE
    cap = cv2.VideoCapture(path)
    frames = 0
    fall_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames += 1
        if frames % sample_rate != 0:
            continue
        r = infer_frame_np(frame)
        if r.get("fall"):
            fall_count += 1
    cap.release()
    return {"frames_sampled": frames // sample_rate if sample_rate>0 else frames, "falls": fall_count, "fall_detected": fall_count > 0}
