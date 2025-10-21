import json
import os

BASELINE_FILE = "baseline.json"

def save_baseline(data):
    with open(BASELINE_FILE, "w") as f:
        json.dump(data, f)
    print("✅ Baseline saved.")

def load_baseline():
    if not os.path.exists(BASELINE_FILE):
        print("⚠️ No baseline found. Please calibrate.")
        return None
    with open(BASELINE_FILE, "r") as f:
        return json.load(f)
