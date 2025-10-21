import mediapipe as mp
import numpy as np

class PostureDetector:
    def __init__(self):
        # We still need the PoseLandmark enum
        self.mp_pose = mp.solutions.pose
        self.baseline = None  # To store baseline posture metrics

        # --- NEW: Calibration state variables ---
        self.calib_mode = False
        self.calib_metrics = []
        self.calib_frames = 50 # Default
        # --- END NEW ---

    # --- NEW: Method to start calibration ---
    def start_calibration(self, frames=50):
        self.calib_mode = True
        self.calib_metrics = []
        self.calib_frames = frames
        print("Posture calibration started... Sit in your ideal posture.")

    # --- NEW: Method to process a frame during calibration ---
    def process_calibration(self, metrics):
        if not self.calib_mode:
            return False # Not calibrating

        if metrics:
            self.calib_metrics.append(metrics)
        
        if len(self.calib_metrics) >= self.calib_frames:
            self.set_baseline(self.calib_metrics) # Call existing set_baseline
            self.calib_mode = False
            print("✅ Posture calibration complete.")
            return True # Calibration finished
        
        return False # Calibration ongoing

    def calculate_metrics(self, landmarks):
        if landmarks is None:
            return None
            
        try:
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

            # Ratio-based metric
            if shoulder_width < 0.01: # Avoid division by zero
                return None
            normalized_eye_to_shoulder = eye_to_shoulder / shoulder_width

            # Shoulder angle
            shoulder_slope = np.degrees(
                np.arctan2(left_shoulder.y - right_shoulder.y, left_shoulder.x - right_shoulder.x)
            )

            # Forward head posture
            head_forward = abs(nose.x - (left_shoulder.x + right_shoulder.x) / 2)

            return {
                "eye_shoulder_ratio": normalized_eye_to_shoulder,
                "shoulder_angle": shoulder_slope,
                "head_forward": head_forward
            }
        except Exception as e:
            return None

    def set_baseline(self, metrics_list):
        valid_metrics = [m for m in metrics_list if m is not None]
        if not valid_metrics:
            print("⚠️ Could not capture posture baseline. Please try again.")
            return

        avg_metrics = {key: np.mean([m[key] for m in valid_metrics]) for key in valid_metrics[0].keys()}
        self.baseline = avg_metrics
        print("✅ Baseline posture captured:", self.baseline)

    def detect_posture(self, metrics):
        if not self.baseline or metrics is None:
            return "Calculating..."

        ratio_drop = (self.baseline["eye_shoulder_ratio"] - metrics["eye_shoulder_ratio"]) / self.baseline["eye_shoulder_ratio"]
        head_shift = abs(metrics["head_forward"] - self.baseline["head_forward"])
        shoulder_tilt = abs(metrics["shoulder_angle"] - self.baseline["shoulder_angle"])

        if ratio_drop > 0.15:
            return "⚠️ Possible hunchback detected"
        elif shoulder_tilt > 10:
            return "⚠️ Uneven shoulders"
        elif head_shift > 0.05:
            return "⚠️ Forward head posture"
        else:
            return "✅ Good posture"