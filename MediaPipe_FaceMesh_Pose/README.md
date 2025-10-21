# Real-Time Posture & Eye Strain Monitor

A desktop webcam app that monitors your posture and eye behavior in real time to reduce hunching, "tech neck", and digital eye strain. It runs two MediaPipe models: FaceMesh for face/eye tracking and Pose for body posture.

## Features

- Real-time posture detection
    - Hunching (shrinking eye-to-shoulder distance)
    - Forward head posture (nose forward relative to shoulders)
    - Shoulder tilting (leaning to one side)
- Real-time eye strain detection
    - Blink counting and blink-rate monitoring (blinks/min)
    - Long-focus alerts (remind to blink)
    - Drowsiness detection (long eye closures)
    - Yawn detection (mouth aspect ratio)
- Smart alerts
    - 20-20-20 reminder (every 20 minutes take a 20-second break)
    - On-screen text + color-coded warnings (green = okay, red = alert)

## How it works

The application processes each webcam frame with two MediaPipe solutions:

1. Posture detection (`posture_detector.py`)
     - Uses `mediapipe.solutions.pose` (33 body landmarks).
     - Calibration: a 50-frame baseline capture (sit in your ideal posture).
     - Detection: compares current geometric ratios (eye-to-shoulder, nose-to-shoulder, shoulder angles) to baseline to detect deviations.

2. Eye strain detection (`eye_strain_detector.py`)
     - Uses `mediapipe.solutions.face_mesh` (468 facial landmarks).
     - EAR (Eye Aspect Ratio) measures eye openness.
     - Calibration: press `E` to capture your personal "eyes open" EAR and set thresholds.
     - Blink counting: detects quick EAR drops and counts blinks in a 60-second rolling window.
     - Drowsiness & yawning: prolonged low EAR signals fatigue; Mouth Aspect Ratio (MAR) detects yawns.

## Project files

- `main.py` — runs both posture and eye detectors together.
- `eye.py` — run only the eye strain detector (useful for testing).
- `posture.py` — run only the posture detector (useful for testing).
- `eye_strain_detector.py` — FaceMesh logic, EAR/MAR, blink/drowsiness logic.
- `posture_detector.py` — Pose logic, baseline calibration, posture math.

## Setup

Prerequisites
- Python 3.7+
- A webcam

Install
1. Place project files in a folder.
2. Create `requirements.txt` with:
     ```
     opencv-python
     mediapipe
     numpy
     ```
3. Install:
     ```
     pip install -r requirements.txt
     ```

## Usage

- Run full application:
    ```
    python main.py
    ```
    Wait for the 50-frame posture calibration, then press `E` to calibrate eyes.

- Run eye-only:
    ```
    python eye.py
    ```
    Press `E` to calibrate eyes.

- Run posture-only:
    ```
    python posture.py
    ```
    Wait for the 50-frame posture calibration.

## Performance note

`main.py` runs two heavy models on every frame (Pose + FaceMesh). This may cause low FPS on typical machines and can break time-sensitive blink detection. `eye.py` and `posture.py` are more reliable individually. Consider switching to MediaPipe Holistic or optimizing frame processing (process every Nth frame, reduce resolution, or run models asynchronously).

## Tips

- Sit in a stable, well-lit position during calibration.
- Use a neutral background and avoid rapid head movements during baseline capture.
- Recalibrate if lighting or camera position changes.

License: This repository contains example code — choose an appropriate license for your project.
