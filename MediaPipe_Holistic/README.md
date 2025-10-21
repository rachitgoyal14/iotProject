# Real-Time Posture & Eye Strain Monitor (Holistic Version)

This project uses your webcam and computer vision to monitor your physical wellness while you work. It simultaneously tracks your posture and your eye-blink patterns to provide real-time alerts, helping you avoid hunching, "tech neck," and digital eye strain.

This is the **high-performance version** of the project. Instead of running two separate, heavy AI models, it uses the single, optimized **`MediaPipe Holistic`** model to get all the data it needs in one pass, resulting in much higher and smoother framerates.

---

## Features

- **Optimized Performance:** Uses the `Holistic` model to track pose and face landmarks simultaneously with minimal performance impact.
- **Real-Time Posture Detection:** Monitors your posture and provides alerts for:
    - **Hunching:** Detects when your shoulders slump and your head drops.
    - **Forward Head Posture:** Detects when your head juts forward ("tech neck").
    - **Shoulder Tilting:** Detects if you are leaning to one side.
- **Real-Time Eye Strain Detection:**
    - **Blink Rate:** Monitors your blinks-per-minute. A low rate is a key sign of eye strain.
    - **Long Focus Alert:** Reminds you to blink if you've been staring (no blinks) for too long.
    - **Drowsiness Detection:** Catches long eye closures that indicate fatigue.
    - **Yawn Detection:** Uses mouth tracking to detect yawns.
- **Smart Alerts:**
    - Implements the **20-20-20 rule**, reminding you every 20 minutes to take a 20-second break.
    - Provides on-screen text feedback and color-coded warnings (Green for good, Red for bad).
- **One-Press Calibration:** Pressing **'E'** calibrates *both* your ideal posture and your open-eye state at the same time.

---

## How It Works

This application's efficiency comes from using the **`mediapipe.solutions.holistic`** model.

### 1. The Holistic Pipeline

Instead of running a full `Pose` model and a full `FaceMesh` model, the `Holistic` model is a smart, multi-stage pipeline:

1. **Finds the Pose:** It first detects the person's body.
2. **Finds Regions of Interest (ROI):** Based on the pose, it determines where the head and hands are.
3. **Zooms In:** It then runs the detailed face and hand landmark models only on those small, specific regions.

This "pose-first, zoom-in" approach is dramatically faster than processing the entire image twice.

### 2. Refactored "Calculator" Classes

The `PostureDetector` and `EyeStrainDetector` classes in this project have been refactored. They **do not run their own AI models**. They are now simple "calculator" classes that:

1. Receive the landmark data from the main `Holistic` model.
2. Perform the specific math calculations (e.g., Eye Aspect Ratio, posture ratios).
3. Return the results.

This design keeps the main loop clean and concentrates all the heavy AI processing into a single call.

### 3. Unified Calibration

- When you press **'E'**, the `main_holistic.py` script tells *both* the `PostureDetector` and `EyeStrainDetector` to enter calibration mode.
- For the next 50–60 frames, the app gathers data to define your "good" posture and your "eyes open" EAR.
- Afterward, it uses these baselines to accurately detect deviations (like hunching or blinking).

---

## Project Files

- `main_holistic.py` — The main application that initializes the `Holistic` model, runs the main loop, and displays the video windows.
- `posture_detector_holistic.py` — The Python class that calculates posture metrics based on landmarks it receives.
- `eye_strain_detector_holistic.py` — The Python class that calculates EAR, MAR, and blinks based on landmarks it receives.

---

## Setup and Usage

### Prerequisites

- Python 3.7 or newer
- A webcam

### Installation

1. Clone or download the project files into a folder (e.g., `mediapipe_holistic`).
2. Create a `requirements.txt` file in that folder with the following content:
     ```
     opencv-python
     mediapipe
     numpy
     ```
3. Install the dependencies by opening a terminal in the project folder and running:
     ```bash
     pip install -r requirements.txt
     ```

### Running the Application

1. Navigate to the project directory in your terminal.
2. Run the main script:
     ```bash
     python main_holistic.py
     ```
3. Two windows will appear. Look at the camera.
4. Press **'E'** to start the calibration. Sit in your ideal posture with your eyes open and looking at the camera.
5. After calibration, the app will begin monitoring you in real-time.
6. Press **'Q'** or **Esc** to quit.
