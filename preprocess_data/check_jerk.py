import json
import numpy as np
import os
from pathlib import Path

#checking jerk signatures of specific verbs in the original videos to validate our synthetic anomaly design choices
ORIGINAL_JSON_PATH = r"C:\Users\Student3\Documents\xrtraceai\Data_set\Halo_assist_dataset\data-annotation-trainval-v1_1.json"
TRAIN_DIR = r"C:\Users\Student3\Documents\xrtraceai\preprocess_data\train"
FRAME_RATE = 30.0

SUSPECT_VERBS = ["hit", "break", "split", "smash", "strike"]
BASELINE_VERBS = ["hold", "turn", "insert", "pull", "push"]

print("Analyzing kinematic signatures of specific verbs (95th Percentile)...")

with open(ORIGINAL_JSON_PATH, 'r') as f:
    ms_data = json.load(f)

# Dictionaries to store peak velocities for comparison
verb_velocities = {verb: [] for verb in SUSPECT_VERBS + BASELINE_VERBS}

for video_data in ms_data:
    # 1. Get the exact Video ID
    video_name = video_data.get("video_name")
    if not video_name:
        continue
        
    brv_path = os.path.join(TRAIN_DIR, f"{video_name}_brv.npy")
    if not os.path.exists(brv_path) or "SYNTH_" in video_name:
        continue
        
    brv = np.load(brv_path)
    
    # Calculate velocity magnitude
    left_vel_mag = np.linalg.norm(brv[:, 16:19], axis=1)
    right_vel_mag = np.linalg.norm(brv[:, 19:22], axis=1)
    max_vel_mag = np.maximum(left_vel_mag, right_vel_mag)

    # 2. Dynamically find the array of annotations (since the array key was missing from the snippet)
    actions_list = []
    for key, value in video_data.items():
        if isinstance(value, list):
            actions_list = value
            break

    for action in actions_list:
        # Only look at the physical interactions
        if action.get("label") != "Fine grained action":
            continue
            
        # 3. Extract the exact Verb
        attributes = action.get("attributes", {})
        verb = attributes.get("Verb", "").lower()
        
        if verb in SUSPECT_VERBS or verb in BASELINE_VERBS:
            # 4. Convert seconds to exact frame indices!
            start_sec = action.get("start", 0)
            end_sec = action.get("end", 0)
            
            start_frame = int(start_sec * FRAME_RATE)
            end_frame = int(end_sec * FRAME_RATE)
            
            # Safety checks for boundaries
            if start_frame >= end_frame or start_frame >= len(max_vel_mag):
                continue
                
            end_frame = min(end_frame, len(max_vel_mag))
            
            # --- THE FIX: 95th Percentile to ignore tracking glitches ---
            # Using np.nanpercentile safely ignores any existing NaNs in the array
            segment_true_vel = np.nanpercentile(max_vel_mag[start_frame:end_frame], 95)
            verb_velocities[verb].append(segment_true_vel)

print("\n=== KINEMATIC SIGNATURE RESULTS (95th Percentile Velocity) ===")
print("Suspect 'Impulsive' Verbs:")
for verb in SUSPECT_VERBS:
    if verb_velocities[verb]:
        # Using nanmean ensures we don't crash if an entire segment was NaN
        avg_peak = np.nanmean(verb_velocities[verb])
        print(f"  {verb.capitalize():<8}: {avg_peak:.4f} m/s (across {len(verb_velocities[verb])} events)")
    else:
        print(f"  {verb.capitalize():<8}: 0 events found")

print("\nBaseline 'Smooth' Verbs:")
for verb in BASELINE_VERBS:
    if verb_velocities[verb]:
        avg_peak = np.nanmean(verb_velocities[verb])
        print(f"  {verb.capitalize():<8}: {avg_peak:.4f} m/s (across {len(verb_velocities[verb])} events)")
    else:
        print(f"  {verb.capitalize():<8}: 0 events found")