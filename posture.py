import cv2
import mediapipe as mp
from posture_detector import PostureDetector

detector = PostureDetector()
baseline_metrics = []
baseline_frames = 50
frame_count = 0
baseline_set = False

cap = cv2.VideoCapture(0)

# For drawing landmarks
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    landmarks = detector.get_landmarks(rgb_frame)

    if landmarks:
        # Draw skeleton overlay
        results = detector.pose.process(rgb_frame)
        mp_drawing.draw_landmarks(
            frame,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=(0, 150, 255), thickness=2, circle_radius=2)
        )

        metrics = detector.calculate_metrics(landmarks)

        if not baseline_set:
            baseline_metrics.append(metrics)
            frame_count += 1
            cv2.putText(frame, f"Capturing baseline... {frame_count}/{baseline_frames}",
                        (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
            if frame_count >= baseline_frames:
                detector.set_baseline(baseline_metrics)
                baseline_set = True
        else:
            posture = detector.detect_posture(metrics)

            # Color logic for posture text
            if "⚠️" in posture:
                color = (0, 0, 255)  # Red for bad posture
            else:
                color = (0, 255, 0)  # Green for good posture

            cv2.putText(frame, posture, (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    cv2.imshow("Posture Detection", frame)
    if cv2.waitKey(10) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
