import mediapipe as mp
import numpy as np
import cv2
import time
from collections import deque

class EyeStrainDetector:
    """
    Refactored EyeStrainDetector.
    - Does NOT run its own MediaPipe model.
    - Receives landmarks from the main Holistic model.
    - Performs calculations (EAR, MAR, blinks) on those landmarks.
    """

    # FaceMesh landmark indices (MediaPipe)
    LEFT_EYE_IDX = [33, 160, 158, 133, 153, 144]
    RIGHT_EYE_IDX = [362, 385, 387, 263, 373, 380]
    MOUTH_TOP = 13
    MOUTH_BOTTOM = 14
    MOUTH_LEFT = 78
    MOUTH_RIGHT = 308

    def __init__(self,
                 ear_smoothing=5,
                 ear_threshold_default=0.21,
                 consec_frames_for_blink=2,
                 blink_window_seconds=60,
                 drowsy_time_seconds=0.8,
                 ear_calib_frames=60,
                 mar_threshold=0.65,
                 yawn_time_seconds=0.6):
        
        # Parameters
        self.ear_smoothing = ear_smoothing
        self.EAR_THRESHOLD_DEFAULT = ear_threshold_default
        self.CONSEC_FRAMES = consec_frames_for_blink
        self.blink_window_seconds = blink_window_seconds
        self.drowsy_time_seconds = drowsy_time_seconds
        self.ear_calib_frames = ear_calib_frames
        self.MAR_THRESHOLD = mar_threshold
        self.YAWN_TIME = yawn_time_seconds

        # State
        self.ear_history = deque(maxlen=ear_smoothing)
        self.blink_timestamps = deque()
        self.blink_count = 0

        # For blink detection (state machine)
        self._closed = False
        self._closure_start_time = None
        self._last_blink_time = None

        # Calibration
        self.calib_mode = False
        self.calib_values = []
        self.calibrated = False
        self.baseline_ear = None
        self.blink_threshold = self.EAR_THRESHOLD_DEFAULT
        self.drowsy_threshold = self.EAR_THRESHOLD_DEFAULT * 0.5

        # Yawn state
        self._yawn_start = None

    def start_calibration(self):
        """Begin calibration — collect ear samples for ear_calib_frames frames"""
        self.calib_mode = True
        self.calib_values = []
        print("Eye calibration started... please look at the camera with eyes open.")

    def calculate_EAR(self, landmarks, eye_indices, image_shape):
        coords = [(int(landmarks[i].x * image_shape[1]),
                   int(landmarks[i].y * image_shape[0])) for i in eye_indices]
        A = np.linalg.norm(np.array(coords[1]) - np.array(coords[5]))
        B = np.linalg.norm(np.array(coords[2]) - np.array(coords[4]))
        C = np.linalg.norm(np.array(coords[0]) - np.array(coords[3]))
        if C == 0:
            return 0.0, coords
        EAR = (A + B) / (2.0 * C)
        return EAR, coords

    def calculate_MAR(self, landmarks, image_shape):
        try:
            top = landmarks[self.MOUTH_TOP]
            bottom = landmarks[self.MOUTH_BOTTOM]
            left = landmarks[self.MOUTH_LEFT]
            right = landmarks[self.MOUTH_RIGHT]
            top_pt = np.array([int(top.x * image_shape[1]), int(top.y * image_shape[0])])
            bottom_pt = np.array([int(bottom.x * image_shape[1]), int(bottom.y * image_shape[0])])
            left_pt = np.array([int(left.x * image_shape[1]), int(left.y * image_shape[0])])
            right_pt = np.array([int(right.x * image_shape[1]), int(right.y * image_shape[0])])
            ver = np.linalg.norm(top_pt - bottom_pt)
            hor = np.linalg.norm(left_pt - right_pt)
            if hor == 0:
                return 0.0
            MAR = ver / hor
            return MAR
        except Exception:
            return 0.0

    def _register_blink(self):
        now = time.time()
        self.blink_timestamps.append(now)
        self.blink_count += 1
        cutoff = now - self.blink_window_seconds
        while self.blink_timestamps and self.blink_timestamps[0] < cutoff:
            self.blink_timestamps.popleft()

    def _blink_rate(self):
        window_count = len(self.blink_timestamps)
        return (window_count / max(1, self.blink_window_seconds)) * 60.0

    def _smooth_ear(self, ear):
        self.ear_history.append(ear)
        return float(np.mean(self.ear_history))

    def process_landmarks(self, landmarks, image_shape):
        """
        NEW METHOD: Takes landmarks from Holistic and performs calculations.
        Does not run its own model or draw on the frame.
        """
        try:
            left_ear, left_pts = self.calculate_EAR(landmarks, self.LEFT_EYE_IDX, image_shape)
            right_ear, right_pts = self.calculate_EAR(landmarks, self.RIGHT_EYE_IDX, image_shape)
            avg_ear_raw = (left_ear + right_ear) / 2.0
        except Exception as e:
            # print(f"Error calculating EAR: {e}")
            return None, [], []

        # --- Calibration ---
        if self.calib_mode:
            self.calib_values.append(avg_ear_raw)
            if len(self.calib_values) >= self.ear_calib_frames:
                self.baseline_ear = float(np.mean(self.calib_values))
                self.blink_threshold = max(0.12, self.baseline_ear * 0.75)
                self.drowsy_threshold = max(0.08, self.baseline_ear * 0.45)
                self.calibrated = True
                self.calib_mode = False
                self.ear_history.clear()
                print(f"Eye calibration complete. baseline EAR={self.baseline_ear:.3f}, blink_thr={self.blink_threshold:.3f}, drowsy_thr={self.drowsy_threshold:.3f}")

        avg_ear = self._smooth_ear(avg_ear_raw)

        # Thresholds
        thr = self.blink_threshold if self.calibrated else self.EAR_THRESHOLD_DEFAULT
        drowsy_thr = self.drowsy_threshold if self.calibrated else (self.EAR_THRESHOLD_DEFAULT * 0.5)

        # --- Blink State Machine ---
        now = time.time()
        if avg_ear < thr:
            if not self._closed:
                self._closed = True
                self._closure_start_time = now
        else:
            if self._closed:
                duration = now - (self._closure_start_time or now)
                if 0.03 <= duration <= 0.6:
                    self._register_blink()
                    self._last_blink_time = now
                self._closed = False
                self._closure_start_time = None

        # --- Drowsiness ---
        if avg_ear < drowsy_thr:
            if not hasattr(self, "_drowsy_start"):
                self._drowsy_start = now
            closure_duration = now - getattr(self, "_drowsy_start", now)
        else:
            if hasattr(self, "_drowsy_start"):
                delattr(self, "_drowsy_start")
            closure_duration = 0.0

        # --- Yawn Detection ---
        mar = self.calculate_MAR(landmarks, image_shape)
        yawned = False
        if mar > self.MAR_THRESHOLD:
            if self._yawn_start is None:
                self._yawn_start = now
            elif (now - self._yawn_start) >= self.YAWN_TIME:
                yawned = True
        else:
            self._yawn_start = None

        # --- Status ---
        blink_rate = self._blink_rate()
        status = "✅ Eyes Normal"
        color = (0, 255, 0)
        if closure_duration >= self.drowsy_time_seconds:
            status = "⚠️ You're getting drowsy"
            color = (0, 0, 255)
        elif blink_rate < 10 and self.calibrated: # Only show strain if calibrated
            status = "⚠️ Low Blink Rate"
            color = (0, 165, 255) # Orange

        # --- Return all info ---
        info = {
            "avg_ear": avg_ear,
            "blink_rate": blink_rate,
            "blink_count": self.blink_count,
            "status": status,
            "color": color,
            "yawn": yawned,
            "closure_duration": closure_duration # <-- ADDED THIS
        }
        
        return info, left_pts, right_pts