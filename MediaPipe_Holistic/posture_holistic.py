import cv2
import time
import mediapipe as mp
import numpy as np
# Import only the posture detector
from posture_detector_holistic import PostureDetector


# --------------------------- Initialize Holistic Model ---------------------------
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# Use fastest settings for real-time
holistic = mp_holistic.Holistic(
    static_image_mode=False,
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
    # We don't need refine_face_landmarks here
)

# --------------------------- Initialize Posture Detector ---------------------------
posture_detector = PostureDetector()

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FPS, 30)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("Instructions:")
print(" - Running Posture Detection (Holistic).")
print(" - Press 'E' to calibrate posture.")
print(" - Press 'Q' or ESC to quit.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    ts = time.time()

    # --- SINGLE HOLISTIC PROCESSING ---
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = holistic.process(rgb_frame)

    # --------------------------- Process Posture (from Holistic) ---------------------------
    if results.pose_landmarks:
        # Draw skeleton overlay
        mp_drawing.draw_landmarks(
            frame,
            results.pose_landmarks,
            mp_holistic.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=(0, 150, 255), thickness=2, circle_radius=2)
        )

        # Pass landmarks to detector for calculation
        metrics = posture_detector.calculate_metrics(results.pose_landmarks.landmark)

        # --- Posture Calibration & Detection Logic ---
        if posture_detector.calib_mode:
            # We are calibrating, show feedback
            posture_detector.process_calibration(metrics) # Feed metrics to calibrator
            cv2.putText(frame, f"Calibrating Posture... {len(posture_detector.calib_metrics)}/{posture_detector.calib_frames}",
                        (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        
        elif posture_detector.baseline is not None:
            # We are calibrated, detect posture
            posture = posture_detector.detect_posture(metrics)
            color = (0, 255, 0) if "✅" in posture else (0, 0, 255)
            cv2.putText(frame, posture, (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
        else:
            # Not calibrating and no baseline exists
            cv2.putText(frame, "Press 'E' to calibrate posture",
                        (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 200), 2)

    # --------------------------- Display Window ---------------------------
    cv2.imshow("Posture Detection (Holistic)", frame)

    key = cv2.waitKey(1) & 0xFF
    if key in [27, ord('q')]:
        break
    elif key == ord('e') or key == ord('E'):
        # Trigger posture calibration
        posture_detector.start_calibration(frames=50)

holistic.close()
cap.release()
cv2.destroyAllWindows()