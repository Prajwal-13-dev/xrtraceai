import os
import json
import numpy as np
import glob

# ============================================================================
# CONFIGURATION - UPDATED FOR VALIDATION SPLIT
# ============================================================================
VAL_DIR = r"C:\Users\Student3\Documents\xrtraceai\preprocess_data\val"
FRAME_RATE_HZ = 30.0

def find_interaction_segments(labels, min_length=30):
    """Finds continuous blocks of any interaction class (2-6)"""
    is_interaction = np.isin(labels, [2, 3, 4, 5, 6]).astype(int)
    diffs = np.diff(np.pad(is_interaction, (1, 1), constant_values=0))
    starts = np.where(diffs == 1)[0]
    ends = np.where(diffs == -1)[0]
    
    segments = []
    for s, e in zip(starts, ends):
        if (e - s) >= min_length:
            segments.append((s, e))
    return segments

def create_synthetic_session(brv, labels, meta, segments, anomaly_type):
    synth_brv = np.copy(brv)
    synth_labels = np.copy(labels)
    
    anomaly_frames_created = 0
    
    for start, end in segments:
        size = end - start
        
        sub_len = np.random.randint(15, min(45, max(16, size)))
        sub_start = start + np.random.randint(0, max(1, size - sub_len))
        sub_end = sub_start + sub_len
        sub_size = sub_end - sub_start
        
        if anomaly_type == "HYPER_SPEED":
            speed_mult = np.random.uniform(1.8, 3.2)
            synth_brv[sub_start:sub_end, 20:26] *= speed_mult

        elif anomaly_type == "ERRATIC_TREMOR":
            noise_pos = np.random.normal(0, 0.05, (sub_size, 6))
            synth_brv[sub_start:sub_end, 7:13] += noise_pos

            if sub_start > 0:
                spatial_segment = synth_brv[sub_start-1:sub_end, 7:13]
            else:
                spatial_segment = np.vstack([synth_brv[0:1, 7:13], synth_brv[sub_start:sub_end, 7:13]])

            recomputed_vel = np.diff(spatial_segment, axis=0) * FRAME_RATE_HZ
            synth_brv[sub_start:sub_end, 20:26] = recomputed_vel

        elif anomaly_type == "AGGRESSIVE_REACH":
            reach_mult = np.random.uniform(1.2, 1.6)
            synth_brv[sub_start:sub_end, 7:13] *= reach_mult
            synth_brv[sub_start:sub_end, 20:26] *= reach_mult

        elif anomaly_type == "SUDDEN_HEAD_JERK":
            # head_pos_y is now index 1; vel_head_pos_y is index 14
            t = np.linspace(0, np.pi, sub_size)
            dip_magnitude = np.random.uniform(0.10, 0.25)
            synth_brv[sub_start:sub_end, 1] -= np.sin(t) * dip_magnitude

            if sub_start > 0:
                spatial_segment = synth_brv[sub_start-1:sub_end, 1]
            else:
                spatial_segment = np.concatenate(([synth_brv[0, 1]], synth_brv[sub_start:sub_end, 1]))

            synth_brv[sub_start:sub_end, 14] = np.diff(spatial_segment) * FRAME_RATE_HZ

        elif anomaly_type == "CONTROLLER_DROP":
            drop_scenario = np.random.choice(["LEFT", "RIGHT", "BOTH"])
            
            if drop_scenario == "LEFT":
                hands_to_process = ["LEFT"]
            elif drop_scenario == "RIGHT":
                hands_to_process = ["RIGHT"]
            else:
                hands_to_process = ["LEFT", "RIGHT"]

            t_seconds = np.arange(sub_size) / FRAME_RATE_HZ
            raw_drop = 0.5 * 9.81 * (t_seconds ** 2)
            max_drop = np.random.uniform(0.10, 0.30)
            
            decel_start_frac = 0.6  
            if (raw_drop >= max_drop * decel_start_frac).any():
                catch_idx = np.argmax(raw_drop >= max_drop * decel_start_frac)
            else:
                catch_idx = sub_size

            gravity_drop = raw_drop.copy()
            if catch_idx < sub_size:
                tail = np.linspace(0, 1, sub_size - catch_idx)
                ease = raw_drop[catch_idx] + (max_drop - raw_drop[catch_idx]) * (1 - np.cos(tail * np.pi/2))
                gravity_drop[catch_idx:] = ease
                
            gravity_drop = np.minimum(gravity_drop, max_drop)
            
            for hand in hands_to_process:
                if hand == "LEFT":
                    pos_start, pos_end = 7, 10
                    y_idx = 8
                    vel_start, vel_end = 20, 23
                else:
                    pos_start, pos_end = 10, 13
                    y_idx = 11
                    vel_start, vel_end = 23, 26

                synth_brv[sub_start:sub_end, y_idx] -= gravity_drop
                
                if sub_start > 0:
                    spatial_segment = synth_brv[sub_start-1:sub_end, pos_start:pos_end]
                else:
                    spatial_segment = np.vstack([synth_brv[0:1, pos_start:pos_end], synth_brv[sub_start:sub_end, pos_start:pos_end]])
                    
                synth_brv[sub_start:sub_end, vel_start:vel_end] = np.diff(spatial_segment, axis=0) * FRAME_RATE_HZ

        synth_labels[sub_start:sub_end] = 7
        anomaly_frames_created += sub_size

    if anomaly_frames_created > 0:
        orig_sid = meta["session_id"]
        synth_sid = f"SYNTH_{anomaly_type}_{orig_sid}"
        
        np.save(os.path.join(VAL_DIR, f"{synth_sid}_brv.npy"), synth_brv)
        np.save(os.path.join(VAL_DIR, f"{synth_sid}_labels.npy"), synth_labels)
        
        synth_meta = meta.copy()
        synth_meta["session_id"] = synth_sid
        synth_meta["task_type"] = "SYNTHETIC_ANOMALY"
        synth_counts = np.bincount(synth_labels.astype(np.int64), minlength=8)
        synth_meta["class_distribution"] = {
            "idle": int(synth_counts[0]), "locomotion": int(synth_counts[1]),
            "grasp_release": int(synth_counts[2]), "assembly": int(synth_counts[3]),
            "manipulation": int(synth_counts[4]), "control_action": int(synth_counts[5]),
            "transfer": int(synth_counts[6]), "anomalous": int(synth_counts[7]),
        }
        
        with open(os.path.join(VAL_DIR, f"{synth_sid}_meta.json"), "w") as f:
            json.dump(synth_meta, f, indent=2)
            
        return True
    return False

if __name__ == "__main__":
    # Changed seed so validation gets a different distribution of augmentations than train
    np.random.seed(43)
    
    print(f"Scanning {VAL_DIR} for normal sessions to augment...")
    
    meta_files = [f for f in glob.glob(os.path.join(VAL_DIR, "*_meta.json")) if "SYNTH_" not in f]
    synthetic_count = 0
    
    for meta_file in meta_files:
        with open(meta_file, "r") as f:
            meta = json.load(f)
            
        sid = meta["session_id"]
        brv_path = os.path.join(VAL_DIR, f"{sid}_brv.npy")
        lbl_path = os.path.join(VAL_DIR, f"{sid}_labels.npy")
        
        if not os.path.exists(brv_path) or not os.path.exists(lbl_path):
            continue
            
        labels = np.load(lbl_path)
        assert sum(meta["class_distribution"].values()) == len(labels), f"Distribution mismatch in {sid}"
        
        segments = find_interaction_segments(labels)
        if not segments:
            continue
            
        brv = np.load(brv_path)
                
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

    print(f"\nSUCCESS: Generated {synthetic_count} brand new synthetic anomalous sessions in VAL split!")