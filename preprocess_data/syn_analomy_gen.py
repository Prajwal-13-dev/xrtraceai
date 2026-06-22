import os
import json
import numpy as np
import glob

# ============================================================================
# CONFIGURATION
# ============================================================================
CURRENT_DIR = os.getcwd()
TRAIN_DIR = os.path.join(CURRENT_DIR, "preprocess_data", "train")
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
        
        if anomaly_type == "HYPER_SPEED":
            # Variable magnitude multiplier to prevent trivial separation
            speed_mult = np.random.uniform(1.8, 3.2)
            synth_brv[start:end, 16:22] *= speed_mult
            
        elif anomaly_type == "ERRATIC_TREMOR":
            # Add spatial jitter to position only
            noise_pos = np.random.normal(0, 0.05, (size, 6))
            synth_brv[start:end, 5:11] += noise_pos
            
            # Recompute physically consistent velocity via differencing
            if start > 0:
                spatial_segment = synth_brv[start-1:end, 5:11]
            else:
                spatial_segment = np.vstack([synth_brv[0:1, 5:11], synth_brv[start:end, 5:11]])
            
            recomputed_vel = np.diff(spatial_segment, axis=0) * FRAME_RATE_HZ
            synth_brv[start:end, 16:22] = recomputed_vel
            
        elif anomaly_type == "AGGRESSIVE_REACH":
            # Variable magnitude scaling
            reach_mult = np.random.uniform(1.2, 1.6)
            synth_brv[start:end, 5:11] *= reach_mult
            synth_brv[start:end, 16:22] *= reach_mult
            
        synth_labels[start:end] = 3
        anomaly_frames_created += (end - start)

    if anomaly_frames_created > 0:
        orig_sid = meta["session_id"]
        synth_sid = f"SYNTH_{anomaly_type}_{orig_sid}"
        
        np.save(os.path.join(TRAIN_DIR, f"{synth_sid}_brv.npy"), synth_brv)
        np.save(os.path.join(TRAIN_DIR, f"{synth_sid}_labels.npy"), synth_labels)
        
        synth_meta = meta.copy()
        synth_meta["session_id"] = synth_sid
        synth_meta["task_type"] = "SYNTHETIC_ANOMALY"
        synth_meta["class_distribution"]["anomalous"] += anomaly_frames_created
        synth_meta["class_distribution"]["object_interaction"] -= anomaly_frames_created
        
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

    print(f"\n✅ SUCCESS: Generated {synthetic_count} brand new synthetic anomalous sessions!")
    print("Run your original 'preprocess_data.py' script one more time to re-generate the statistics chart and see your new dataset balance!")