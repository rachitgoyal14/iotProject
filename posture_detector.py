import mediapipe as mp
import numpy as np

class PostureDetector:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose()
        self.baseline = None  # To store baseline posture metrics

    def get_landmarks(self, image):
        results = self.pose.process(image)
        if not results.pose_landmarks:
            return None
        return results.pose_landmarks.landmark

    def calculate_metrics(self, landmarks):
        # Extract important points
        left_eye = landmarks[self.mp_pose.PoseLandmark.LEFT_EYE]
        right_eye = landmarks[self.mp_pose.PoseLandmark.RIGHT_EYE]
        left_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
        nose = landmarks[self.mp_pose.PoseLandmark.NOSE]

        # Compute distances in normalized coordinates
        eye_center_y = (left_eye.y + right_eye.y) / 2
        shoulder_center_y = (left_shoulder.y + right_shoulder.y) / 2

        eye_to_shoulder = abs(eye_center_y - shoulder_center_y)
        shoulder_width = abs(left_shoulder.x - right_shoulder.x)

        # Ratio-based metric for camera-distance invariance
        normalized_eye_to_shoulder = eye_to_shoulder / shoulder_width

        # Shoulder angle (slouch or tilt)
        shoulder_slope = np.degrees(
            np.arctan2(left_shoulder.y - right_shoulder.y, left_shoulder.x - right_shoulder.x)
        )

        # Forward head posture check (nose alignment)
        head_forward = abs(nose.x - (left_shoulder.x + right_shoulder.x) / 2)

        return {
            "eye_shoulder_ratio": normalized_eye_to_shoulder,
            "shoulder_angle": shoulder_slope,
            "head_forward": head_forward
        }

    def set_baseline(self, metrics_list):
        avg_metrics = {key: np.mean([m[key] for m in metrics_list]) for key in metrics_list[0].keys()}
        self.baseline = avg_metrics
        print("✅ Baseline posture captured:", self.baseline)

    def detect_posture(self, metrics):
        if not self.baseline:
            return "Baseline not set"

        ratio_drop = (self.baseline["eye_shoulder_ratio"] - metrics["eye_shoulder_ratio"]) / self.baseline["eye_shoulder_ratio"]
        head_shift = abs(metrics["head_forward"] - self.baseline["head_forward"])
        shoulder_tilt = abs(metrics["shoulder_angle"] - self.baseline["shoulder_angle"])

        if ratio_drop > 0.15:
            return "⚠️ Possible hunchback detected (eye-shoulder drop)"
        elif shoulder_tilt > 10:
            return "⚠️ Uneven shoulders"
        elif head_shift > 0.05:
            return "⚠️ Forward head posture"
        else:
            return "✅ Good posture"
