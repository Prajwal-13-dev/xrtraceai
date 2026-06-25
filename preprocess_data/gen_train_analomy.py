import os
import json
import numpy as np
import glob

# ============================================================================
# CONFIGURATION
# ============================================================================
TRAIN_DIR = r"C:\Users\Student3\Documents\xrtraceai\preprocess_data\train"
FRAME_RATE_HZ = 30.0

# We only augment the training set. We NEVER augment Val or Test!
# Valid BRV Indices:
# 5:11 -> Hand Positions (Left & Right)
# 16:22 -> Hand Velocities (Left & Right)

def find_interaction_segments(labels, min_length=30):
    """Finds continuous blocks of 'object_interaction' (class 2)"""
    is_interaction = (labels == 2).astype(int)
    diffs = np.diff(np.pad(is_interaction, (1, 1), constant_values=0))
    starts = np.where(diffs == 1)[0]
    ends = np.where(diffs == -1)[0]
    
    segments = []
    for s, e in zip(starts, ends):
        if (e - s) >= min_length:
            segments.append((s, e))
    return segments

def create_synthetic_session(brv, labels, meta, segments, anomaly_type):
    """
    Takes normal data, applies physical math to turn it into an anomaly,
    labels those frames as '3', and saves it as a new synthetic video.
    """
    synth_brv = np.copy(brv)
    synth_labels = np.copy(labels)
    
    anomaly_frames_created = 0
    
    for start, end in segments:
        size = end - start
        
        # Pick a random sub-window of 15 to 45 frames (0.5s to 1.5s) to corrupt.
        sub_len = np.random.randint(15, min(45, max(16, size)))
        sub_start = start + np.random.randint(0, max(1, size - sub_len))
        sub_end = sub_start + sub_len
        sub_size = sub_end - sub_start
        # -------------------------------------
        
        if anomaly_type == "HYPER_SPEED":
            speed_mult = np.random.uniform(1.8, 3.2)
            synth_brv[sub_start:sub_end, 16:22] *= speed_mult
            
        elif anomaly_type == "ERRATIC_TREMOR":
            # Add spatial jitter to position only
            noise_pos = np.random.normal(0, 0.05, (sub_size, 6))
            synth_brv[sub_start:sub_end, 5:11] += noise_pos
            
            # Recompute physically consistent velocity via differencing
            if sub_start > 0:
                spatial_segment = synth_brv[sub_start-1:sub_end, 5:11]
            else:
                spatial_segment = np.vstack([synth_brv[0:1, 5:11], synth_brv[sub_start:sub_end, 5:11]])
            
            recomputed_vel = np.diff(spatial_segment, axis=0) * FRAME_RATE_HZ
            synth_brv[sub_start:sub_end, 16:22] = recomputed_vel
            
        elif anomaly_type == "AGGRESSIVE_REACH":
            # Variable magnitude scaling
            reach_mult = np.random.uniform(1.2, 1.6)
            synth_brv[sub_start:sub_end, 5:11] *= reach_mult
            synth_brv[sub_start:sub_end, 16:22] *= reach_mult

        elif anomaly_type == "SUDDEN_HEAD_JERK":
            # We simulate a sudden vertical duck/dip instead of a lateral sway.
            t = np.linspace(0, np.pi, sub_size) # Half sine wave (dip down and back up)
            dip_magnitude = np.random.uniform(0.10, 0.25)  # Duck by 10cm to 25cm
            synth_brv[sub_start:sub_end, 0] -= np.sin(t) * dip_magnitude
            
            # Recompute ONLY head_pos_y_vel (Index 11) to avoid corrupting quaternions
            if sub_start > 0:
                spatial_segment = synth_brv[sub_start-1:sub_end, 0]
            else:
                spatial_segment = np.concatenate(([synth_brv[0, 0]], synth_brv[sub_start:sub_end, 0]))
                
            # Update strictly index 11
            synth_brv[sub_start:sub_end, 11] = np.diff(spatial_segment) * FRAME_RATE_HZ

        elif anomaly_type == "CONTROLLER_DROP":
            drop_scenario = np.random.choice(["LEFT", "RIGHT", "BOTH"])
            
            if drop_scenario == "LEFT":
                hands_to_process = ["LEFT"]
            elif drop_scenario == "RIGHT":
                hands_to_process = ["RIGHT"]
            else:
                hands_to_process = ["LEFT", "RIGHT"]

            # Calculate the raw gravity physics
            t_seconds = np.arange(sub_size) / FRAME_RATE_HZ
            raw_drop = 0.5 * 9.81 * (t_seconds ** 2)
            max_drop = np.random.uniform(0.10, 0.30)
            
        
            # Start "catching" the controller at 60% of freefall toward max_drop
            decel_start_frac = 0.6  
            if (raw_drop >= max_drop * decel_start_frac).any():
                catch_idx = np.argmax(raw_drop >= max_drop * decel_start_frac)
            else:
                catch_idx = sub_size

            gravity_drop = raw_drop.copy()
            if catch_idx < sub_size:
                tail = np.linspace(0, 1, sub_size - catch_idx)
                # Smooth ease-out using cosine
                ease = raw_drop[catch_idx] + (max_drop - raw_drop[catch_idx]) * (1 - np.cos(tail * np.pi/2))
                gravity_drop[catch_idx:] = ease
                
            # Final safety cap for floating point precision
            gravity_drop = np.minimum(gravity_drop, max_drop)
            

            for hand in hands_to_process:
                if hand == "LEFT":
                    pos_start, pos_end = 5, 8
                    y_idx = 6
                    vel_start, vel_end = 16, 19
                else:  # RIGHT hand
                    pos_start, pos_end = 8, 11
                    y_idx = 9
                    vel_start, vel_end = 19, 22

                # Apply the smoothed gravity to the Y-axis
                synth_brv[sub_start:sub_end, y_idx] -= gravity_drop
                
                # Recompute the velocity for this specific hand
                if sub_start > 0:
                    spatial_segment = synth_brv[sub_start-1:sub_end, pos_start:pos_end]
                else:
                    spatial_segment = np.vstack([synth_brv[0:1, pos_start:pos_end], synth_brv[sub_start:sub_end, pos_start:pos_end]])
                    
                synth_brv[sub_start:sub_end, vel_start:vel_end] = np.diff(spatial_segment, axis=0) * FRAME_RATE_HZ

        synth_labels[sub_start:sub_end] = 3
        anomaly_frames_created += sub_size

    if anomaly_frames_created > 0:
        orig_sid = meta["session_id"]
        synth_sid = f"SYNTH_{anomaly_type}_{orig_sid}"
        
        np.save(os.path.join(TRAIN_DIR, f"{synth_sid}_brv.npy"), synth_brv)
        np.save(os.path.join(TRAIN_DIR, f"{synth_sid}_labels.npy"), synth_labels)
        
        synth_meta = meta.copy()
        synth_meta["session_id"] = synth_sid
        synth_meta["task_type"] = "SYNTHETIC_ANOMALY"
        synth_meta["class_distribution"]["anomalous"] += int(anomaly_frames_created)
        synth_meta["class_distribution"]["object_interaction"] -= int(anomaly_frames_created)
        
        with open(os.path.join(TRAIN_DIR, f"{synth_sid}_meta.json"), "w") as f:
            json.dump(synth_meta, f, indent=2)
            
        return True
    return False

if __name__ == "__main__":
    # Ensure reproducible pseudo-random generation
    np.random.seed(42)
    
    print(f"Scanning {TRAIN_DIR} for normal sessions to augment...")
    
    meta_files = [f for f in glob.glob(os.path.join(TRAIN_DIR, "*_meta.json")) if "SYNTH_" not in f]
    
    synthetic_count = 0
    
    for meta_file in meta_files:
        with open(meta_file, "r") as f:
            meta = json.load(f)
            
        sid = meta["session_id"]
        brv_path = os.path.join(TRAIN_DIR, f"{sid}_brv.npy")
        lbl_path = os.path.join(TRAIN_DIR, f"{sid}_labels.npy")
        
        if not os.path.exists(brv_path) or not os.path.exists(lbl_path):
            continue
            
        labels = np.load(lbl_path)
        
        # Defensive assertion for distribution integrity
        assert sum(meta["class_distribution"].values()) == len(labels), f"Distribution mismatch in {sid}"
        
        segments = find_interaction_segments(labels)
        if not segments:
            continue
            
        brv = np.load(brv_path)

                
        # If the original file has HoloLens tracking dropouts (NaNs), 
        # replace them with 0.0 before we try to multiply or add noise to them.
        if np.isnan(brv).any() or np.isinf(brv).any():
            brv = np.nan_to_num(brv, nan=0.0, posinf=0.0, neginf=0.0)

        
        chance = np.random.random()
        
        if chance < 0.10:
            if create_synthetic_session(brv, labels, meta, segments, "HYPER_SPEED"):
                synthetic_count += 1
        elif chance < 0.20:
            if create_synthetic_session(brv, labels, meta, segments, "ERRATIC_TREMOR"):
                synthetic_count += 1
        elif chance < 0.30:
            if create_synthetic_session(brv, labels, meta, segments, "AGGRESSIVE_REACH"):
                synthetic_count += 1
        elif chance < 0.40:
            if create_synthetic_session(brv, labels, meta, segments, "SUDDEN_HEAD_JERK"):
                synthetic_count += 1
        elif chance < 0.50:
            if create_synthetic_session(brv, labels, meta, segments, "CONTROLLER_DROP"):
                synthetic_count += 1

    print(f"\nSUCCESS: Generated {synthetic_count} brand new synthetic anomalous sessions!")
    print("Run your original 'preprocess_data.py' script one more time to re-generate the statistics chart and see your new dataset balance!")