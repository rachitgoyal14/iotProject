# Real-Time Posture & Eye Strain Monitor

A webcam-based computer-vision tool that monitors posture and eye-blink patterns to provide real-time alerts for hunching, "tech neck," and digital eye strain.

This repository contains two approaches:

- `MediaPipe_FaceMesh_Pose/` — initial version using two separate models (works but slow).
- `MediaPipe_Holistic/` — optimized version using a single Holistic model (recommended).

---

## Requirements

Run `requirements.txt` with the following:
```
opencv-python
mediapipe
numpy
```

---

## Setup (Virtual Environment)

1. Create the virtual environment and activate it.

macOS / Linux
```bash
python3 -m venv venv
source venv/bin/activate
```

Windows (PowerShell / Command Prompt)
```powershell
python -m venv venv
.\venv\Scripts\activate
```

2. Install dependencies
```bash
pip install -r requirements.txt
```

---

## Part 1 — MediaPipe_FaceMesh_Pose (Basic)

Overview
- Uses two separate models:
    - `mediapipe.solutions.face_mesh` (eye_strain_detector.py) — 468 facial landmarks for Eye Aspect Ratio (EAR) and Mouth Aspect Ratio (MAR).
    - `mediapipe.solutions.pose` (posture_detector.py) — 33 body landmarks for posture detection.
- Drawback: running both full models per frame is computationally heavy and causes low framerates, which breaks time-sensitive blink detection.

How to run
- Full (laggy) app:
```bash
cd MediaPipe_FaceMesh_Pose
python main.py
```
(Wait for the 50-frame auto-calibration for posture, then press `E` to calibrate eyes.)

- Eye tracker only:
```bash
python eye.py
```
(Press `E` to calibrate.)

- Posture tracker only:
```bash
python posture.py
```
(Wait for the 50-frame auto-calibration.)

---

## Part 2 — MediaPipe_Holistic (Optimized, Recommended)

Overview
- Uses `mediapipe.solutions.holistic`, a single efficient pipeline.
- Workflow:
    - Fast pose detector finds the person.
    - Pose identifies Regions of Interest (ROIs) for face and hands.
    - Detailed face/hand models run on small ROIs only.
- Implementation: `posture_detector_holistic.py` and `eye_strain_detector_holistic.py` are refactored as calculators; `main_holistic.py` runs Holistic once and passes landmarks to calculators. This yields much better performance and reliable blink detection.

How to run
- Full optimized app:
```bash
cd MediaPipe_Holistic
python main_holistic.py
```
(Press `E` to calibrate both posture and eyes.)

- Eye tracker only:
```bash
python eye_holistic.py
```
(Press `E` to calibrate.)

- Posture tracker only:
```bash
python posture_holistic.py
```
(Press `E` to calibrate.)

---

## Notes
- Use the Holistic version for real-time use on standard laptops/PCs.
- If you encounter performance issues, ensure no other heavy apps are running and that your virtual environment is active when launching the scripts.
- Calibration prompts appear in the window; follow them for accurate detection.
- For development, run the individual modules (eye/posture) to test logic in isolation.

