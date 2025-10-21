import cv2
import time
import mediapipe as mp
import numpy as np
# Import the refactored detector classes
from posture_detector_holistic import PostureDetector
from eye_strain_detector_holistic import EyeStrainDetector

# --------------------------- Initialize Holistic Model ---------------------------
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# Use fastest settings for real-time
holistic = mp_holistic.Holistic(
    static_image_mode=False,
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    refine_face_landmarks=True 
)

# --------------------------- Initialize Posture Detector ---------------------------
posture_detector = PostureDetector()
# --- REMOVED Automatic baseline variables ---
# baseline_metrics = []
# baseline_frames = 50
# frame_count = 0
# baseline_set = False
# --- END REMOVED ---

# --------------------------- Initialize Eye Strain Detector ---------------------------
eye_detector = EyeStrainDetector(
    ear_smoothing=5,
    ear_threshold_default=0.21,
    consec_frames_for_blink=2,
    blink_window_seconds=60,
    drowsy_time_seconds=0.8,
    ear_calib_frames=60,
    mar_threshold=0.65,
    yawn_time_seconds=0.6
)

# --------------------------- Smart alert parameters ---------------------------
FOCUS_LIMIT = 10.0
LOW_BLINK_THRESHOLD = 8.0
LOW_BLINK_SUSTAIN = 60.0
SESSION_LIMIT = 20 * 60.0
BREAK_DURATION = 20.0
ALERT_COOLDOWN = 30.0

# --------------------------- State tracking ---------------------------
last_blink_time = time.time()
low_blink_start = None
last_alert_time = 0
session_start = time.time()
in_break = False
break_start = None

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FPS, 30)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("Instructions:")
print(" - Running with optimized Holistic model.")
print(" - Press 'E' to calibrate BOTH posture and eyes.")
print(" - Press 'Q' or ESC to quit.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    ts = time.time()
    
    # Create copies for separate display windows
    frame_eye = frame.copy()
    frame_posture = frame.copy()

    # --- SINGLE HOLISTIC PROCESSING ---
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = holistic.process(rgb_frame)

    # --------------------------- Process Eye Strain (from Holistic) ---------------------------
    if results.face_landmarks:
        # Pass the landmarks to the detector
        eye_info, left_pts, right_pts = eye_detector.process_landmarks(
            results.face_landmarks.landmark, frame_eye.shape
        )
        
        # Draw eye contours
        try:
            cv2.polylines(frame_eye, [np.array(left_pts, dtype=np.int32)], isClosed=True, color=(0, 255, 0), thickness=1)
            cv2.polylines(frame_eye, [np.array(right_pts, dtype=np.int32)], isClosed=True, color=(0, 255, 0), thickness=1)
        except Exception:
            pass

        if eye_info is not None:
            blink_count = eye_info["blink_count"]
            blink_rate = eye_info["blink_rate"]
            eye_status = eye_info["status"]
            yawned = eye_info.get("yawn", False)

            # More Info Display
            cv2.putText(frame_eye, f"Blinks: {blink_count} | Rate: {blink_rate:.1f}/min",
                        (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(frame_eye, f"EAR: {eye_info['avg_ear']:.2f}", (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, eye_info['color'], 2)
            cv2.putText(frame_eye, f"Status: {eye_status}", (30, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, eye_info['color'], 2)

            y_pos = 140
            if eye_info['closure_duration'] > 0.1:
                cv2.putText(frame_eye, f"Closure: {eye_info['closure_duration']:.2f}s",
                            (30, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
                y_pos += 30
            
            if yawned:
                cv2.putText(frame_eye, "YAWN DETECTED",
                            (30, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                y_pos += 30
            
            # Show Calibration Feedback
            if eye_detector.calib_mode:
                cv2.putText(frame_eye, f"Calibrating Eyes... {len(eye_detector.calib_values)}/{eye_detector.ear_calib_frames}",
                            (30, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 200), 2)
            elif not eye_detector.calibrated:
                 cv2.putText(frame_eye, "Press 'E' to calibrate",
                            (30, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 200), 2)

            # Update last blink time
            if not hasattr(eye_detector, "_last_blink_cache"):
                eye_detector._last_blink_cache = blink_count
            if blink_count > eye_detector._last_blink_cache:
                last_blink_time = ts
                eye_detector._last_blink_cache = blink_count

            # --------------------------- SMART LOGIC ---------------------------
            alert_reason = None
            time_since_blink = ts - last_blink_time

            if eye_detector.calibrated and time_since_blink > FOCUS_LIMIT:
                alert_reason = "Focusing too long without blinking"

            if eye_detector.calibrated and blink_rate < LOW_BLINK_THRESHOLD:
                if low_blink_start is None:
                    low_blink_start = ts
                elif (ts - low_blink_start) > LOW_BLINK_SUSTAIN:
                    alert_reason = "Low blink rate - possible eye strain"
            else:
                low_blink_start = None

            if "drowsy" in eye_status.lower() or yawned:
                alert_reason = "You look tired — take a break"

            elapsed_session = ts - session_start
            if not in_break and elapsed_session >= SESSION_LIMIT:
                in_break = True
                break_start = ts
                session_start = ts
                alert_reason = "20–20–20 Reminder: Look 20 feet away for 20 seconds!"

            if alert_reason and (ts - last_alert_time) > ALERT_COOLDOWN:
                last_alert_time = ts
                cv2.rectangle(frame_eye, (0, 0), (frame_eye.shape[1], 40), (0, 0, 255), -1)
                cv2.putText(frame_eye, f"ALERT: {alert_reason}", (10, 28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                print("⚠️", alert_reason)

            if in_break:
                elapsed_break = ts - break_start
                remaining = max(0, BREAK_DURATION - elapsed_break)
                cv2.putText(frame_eye, f"👁️ BREAK TIME: Look away for {remaining:.0f}s",
                            (30, frame_eye.shape[0] - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 200, 255), 2)
                if elapsed_break >= BREAK_DURATION:
                    in_break = False
                    print("✅ Break complete. Back to work!")

    # --------------------------- Process Posture (from Holistic) ---------------------------
    if results.pose_landmarks:
        # Draw skeleton overlay
        mp_drawing.draw_landmarks(
            frame_posture,
            results.pose_landmarks,
            mp_holistic.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=(0, 150, 255), thickness=2, circle_radius=2)
        )

        # Pass landmarks to detector for calculation
        metrics = posture_detector.calculate_metrics(results.pose_landmarks.landmark)

        # --- UPDATED: Posture Calibration Logic ---
        if posture_detector.calib_mode:
            # We are calibrating, show feedback
            posture_detector.process_calibration(metrics) # Feed metrics to calibrator
            cv2.putText(frame_posture, f"Calibrating Posture... {len(posture_detector.calib_metrics)}/{posture_detector.calib_frames}",
                        (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        
        elif posture_detector.baseline is not None:
            # We are calibrated, detect posture
            posture = posture_detector.detect_posture(metrics)
            color = (0, 255, 0) if "✅" in posture else (0, 0, 255)
            cv2.putText(frame_posture, posture, (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
        else:
            # Not calibrating and no baseline exists
            cv2.putText(frame_posture, "Press 'E' to calibrate posture",
                        (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 200), 2)
        # --- END UPDATED ---


    # --------------------------- Display in Two Windows ---------------------------
    cv2.imshow("Eye Strain Detection (Holistic)", frame_eye)
    cv2.imshow("Posture Detection (Holistic)", frame_posture)

    key = cv2.waitKey(1) & 0xFF
    if key in [27, ord('q')]:
        break
    elif key == ord('e') or key == ord('E'):
        # --- UPDATED: Trigger BOTH calibrations ---
        eye_detector.start_calibration()
        posture_detector.start_calibration(frames=50) # Use 50 frames
        # --- END UPDATED ---

holistic.close()
cap.release()
cv2.destroyAllWindows()