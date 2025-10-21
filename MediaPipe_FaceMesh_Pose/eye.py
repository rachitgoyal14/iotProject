import cv2
import time
from eye_strain_detector import EyeStrainDetector

# ---------------------------
# Initialize Eye Strain Detector
# ---------------------------
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

# ---------------------------
# Smart alert parameters
# ---------------------------
FOCUS_LIMIT = 10.0           # seconds without blinking
LOW_BLINK_THRESHOLD = 8.0    # blinks per minute (dry eyes)
LOW_BLINK_SUSTAIN = 60.0     # low-blink must persist this long before alert
SESSION_LIMIT = 20 * 60.0    # 20 minutes for 20-20-20 rule
BREAK_DURATION = 20.0        # look away for 20 seconds
ALERT_COOLDOWN = 30.0

# ---------------------------
# State tracking
# ---------------------------
last_blink_time = time.time()
low_blink_start = None
last_alert_time = 0
session_start = time.time()
in_break = False
break_start = None

cap = cv2.VideoCapture(0)
print("Instructions:")
print(" - Press 'E' to start eye calibration (look straight, eyes open).")
print(" - Press 'Q' or ESC to quit.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    ts = time.time()
    frame, eye_info = eye_detector.process_frame(frame)

    if eye_info is not None:
        blink_count = eye_info["blink_count"]
        blink_rate = eye_info["blink_rate"]
        eye_status = eye_info["status"]
        yawned = eye_info.get("yawn", False)

        # Minimal overlay
        cv2.putText(frame, f"Blinks: {blink_count} | Rate: {blink_rate:.1f}/min",
                    (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # Update last blink time when blink count increases
        if not hasattr(eye_detector, "_last_blink_cache"):
            eye_detector._last_blink_cache = blink_count
        if blink_count > eye_detector._last_blink_cache:
            last_blink_time = ts
            eye_detector._last_blink_cache = blink_count

        # ---------------------------
        # SMART LOGIC
        # ---------------------------
        alert_reason = None

        # 1️⃣ Focus too long (no blink)
        time_since_blink = ts - last_blink_time
        if time_since_blink > FOCUS_LIMIT:
            alert_reason = "Focusing too long without blinking"

        # 2️⃣ Low blink rate sustained
        if blink_rate < LOW_BLINK_THRESHOLD:
            if low_blink_start is None:
                low_blink_start = ts
            elif (ts - low_blink_start) > LOW_BLINK_SUSTAIN:
                alert_reason = "Low blink rate - possible eye strain"
        else:
            low_blink_start = None

        # 3️⃣ Drowsy or yawning
        if "drowsy" in eye_status.lower() or yawned:
            alert_reason = "You look tired — take a break"

        # 4️⃣ 20–20–20 Rule
        elapsed_session = ts - session_start
        if not in_break and elapsed_session >= SESSION_LIMIT:
            in_break = True
            break_start = ts
            session_start = ts  # reset next 20-min timer
            alert_reason = "20–20–20 Reminder: Look 20 feet away for 20 seconds!"

        # Show alert (only if cooldown passed)
        if alert_reason and (ts - last_alert_time) > ALERT_COOLDOWN:
            last_alert_time = ts
            cv2.rectangle(frame, (0, 0), (frame.shape[1], 40), (0, 0, 255), -1)
            cv2.putText(frame, f"ALERT: {alert_reason}", (10, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
            print("⚠️", alert_reason)

        # If in 20–20–20 break
        if in_break:
            elapsed_break = ts - break_start
            remaining = max(0, BREAK_DURATION - elapsed_break)
            cv2.putText(frame, f"👁️ BREAK TIME: Look away for {remaining:.0f}s",
                        (30, frame.shape[0] - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 200, 255), 2)
            if elapsed_break >= BREAK_DURATION:
                in_break = False
                print("✅ Break complete. Back to work!")

    cv2.imshow("Eye Strain Detection (20–20–20 Aware)", frame)

    key = cv2.waitKey(1) & 0xFF
    if key in [27, ord('q')]:
        break
    elif key == ord('e'):
        eye_detector.start_calibration()
        print("Starting eye calibration... look straight with eyes open.")

cap.release()
cv2.destroyAllWindows()
