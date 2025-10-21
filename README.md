# Real-Time Posture & Eye Strain Monitor

A webcam-based computer-vision tool that monitors posture and eye-blink patterns to provide real-time alerts for hunching, "tech neck," and digital eye strain.

This repository contains two approaches:

  - `MediaPipe_FaceMesh_Pose/` — initial version using two separate models (works but slow).
  - `MediaPipe_Holistic/` — optimized version using a single Holistic model (recommended).

-----

## Project Roadmap

This project is being developed in phases, moving from a software prototype to a final, standalone hardware device.

### Phase 1: Software Development & Optimization (Current)

This repository represents the current development phase. The focus is on building and perfecting the core computer vision logic using a standard laptop and webcam.

  - Prototyping two different backends (`FaceMesh_Pose` vs. `Holistic`).
  - Optimizing the `Holistic` pipeline for smooth, real-time performance.
  - Refining the algorithms for accurate blink detection (EAR), yawn detection (MAR), and posture analysis.

### Phase 2: IoT Integration & Distributed System (Future)

The next phase will focus on evolving this software from a PC-dependent application to a **distributed IoT system**.

  - **IoT Device:** A custom hardware device will be built using a lightweight single-board computer (e.g., Raspberry Pi or ESP32-CAM) and an external camera.
  - **Wireless Video Streaming:** This device's primary role will be to capture the video feed and **stream it wirelessly, live to a nearby laptop.**
  - **Centralized Processing:** The user's laptop will receive the video stream and run the computationally expensive Holistic model. This keeps the IoT device lightweight and low-power.
  - **Remote Display:** The laptop will process the alerts (e.g., "Hunching Detected\!", "Time for a 20s break\!") and send these simple commands **back to the IoT device**, which will display them on an attached LCD screen.

### Phase 3: Standalone IoT Device (Long-Term Goal)

The final goal is to create a fully **standalone, all-in-one hardware device** that does not require a laptop to function.

  - **On-Board Processing:** This device will use a more powerful single-board computer (e.g., Raspberry Pi 5, Jetson Nano) capable of running the MediaPipe Holistic model directly on the device.
  - **All-in-One:** It will integrate its own camera and LCD screen.
  - **True Standalone:** The user will simply plug it in, and it will monitor their posture and eye strain, displaying all alerts on its own screen. This removes any dependency on a host computer for processing.

-----

## Requirements

Run `requirements.txt` with the following:

```
opencv-python
mediapipe
numpy
```

-----

## Setup (Virtual Environment)

1.  Create the virtual environment and activate it.

**macOS / Linux**

```bash
python3 -m venv venv
source venv/bin/activate
```

**Windows (PowerShell / Command Prompt)**

```powershell
python -m venv venv
.\venv\Scripts\activate
```

2.  Install dependencies

<!-- end list -->

```bash
pip install -r requirements.txt
```

-----

## Part 1 — MediaPipe\_FaceMesh\_Pose (Basic)

### Overview

  - Uses two separate models:
      - `mediapipe.solutions.face_mesh` (`eye_strain_detector.py`) — 468 facial landmarks for Eye Aspect Ratio (EAR) and Mouth Aspect Ratio (MAR).
      - `mediapipe.solutions.pose` (`posture_detector.py`) — 33 body landmarks for posture detection.
  - **Drawback:** running both full models per frame is computationally heavy and causes low framerates, which breaks time-sensitive blink detection.

### How to run

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

-----

## Part 2 — MediaPipe\_Holistic (Optimized, Recommended)

### Overview

  - Uses `mediapipe.solutions.holistic`, a single efficient pipeline.
  - **Workflow:**
    1.  Fast pose detector finds the person.
    2.  Pose identifies Regions of Interest (ROIs) for face and hands.
    3.  Detailed face/hand models run on small ROIs only.
  - **Implementation:** `posture_detector_holistic.py` and `eye_strain_detector_holistic.py` are refactored as "calculators"; `main_holistic.py` runs Holistic once and passes landmarks to the calculators. This yields much better performance and reliable blink detection.

### How to run

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

-----

## Notes

  - Use the **Holistic** version for real-time use on standard laptops/PCs.
  - If you encounter performance issues, ensure no other heavy apps are running and that your virtual environment is active.
  - Calibration prompts appear in the terminal and on-screen; follow them for accurate detection.
  - For development, run the individual modules (`eye_holistic.py` / `posture_holistic.py`) to test logic in isolation.