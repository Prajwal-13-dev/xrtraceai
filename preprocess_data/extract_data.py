import os
import json
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R
from collections import defaultdict
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ============================================================================
# CONFIGURATION
# ============================================================================

CURRENT_DIR = os.getcwd()

DATASET_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "..", "Data_set", "Halo_assist_dataset"))

OUTPUT_DIR = CURRENT_DIR
ANNOTATION_FILE = os.path.join(DATASET_DIR, "data-annotation-trainval-v1_1.json")
FRAME_RATE_HZ = 30.0
DT = 1.0 / FRAME_RATE_HZ

# Head_sync.txt verified layout (18 cols):
#   col 0       : timestamp, seconds
#   col 1       : secondary timestamp (.NET ticks-style integer)
#   cols 2:18   : flattened 4x4 row-major transform matrix (16 values)
HEAD_TOTAL_COLS = 18
HEAD_MATRIX_SLICE = slice(2, 18)

# Left_sync.txt / Right_sync.txt: assumed col0=timestamp, col1=ticks, cols2:5=xyz
# (not directly verified from a sample -- the assertion below will catch a
#  mismatch loudly instead of silently misaligning, unlike the original script)
HAND_POS_SLICE = slice(2, 5)
HAND_MIN_COLS = 5

os.makedirs(OUTPUT_DIR, exist_ok=True)

SPATIAL_DIM = 13
IDX = {
    "head_pos":  slice(0, 3),
    "head_quat": slice(3, 7),
    "left_rel":  slice(7, 10),
    "right_rel": slice(10, 13),
}
VEL_OFFSET = SPATIAL_DIM
IDX_VEL = {k: slice(v.start + VEL_OFFSET, v.stop + VEL_OFFSET)
           for k, v in IDX.items() if k in ("left_rel", "right_rel")}
BRV_TOTAL_DIM = SPATIAL_DIM * 2  # 26

# ============================================================================
# STATISTICS CONFIGURATION
# ============================================================================
SPLITS       = ["train", "val", "test"]
CLASS_NAMES  = ["idle", "locomotion", "grasp_release", "assembly", "manipulation", "control_action", "transfer", "anomalous"]
CLASS_COLORS = ["#6B7280", "#3B82F6", "#10B981", "#F59E0B", "#8B5CF6", "#EC4899", "#14B8A6", "#EF4444"]

TASK_KEYWORDS = {
    "GOPRO":        "gopro",
    "DSLR":         "dslr",
    "NESPRESSO":    "nespresso",
    "ESPRESSO":     "espresso",
    "SWITCH":       "switch",
    "NAVVIS":       "navvis",
    "SMALLPRINTER": "smallprinter",
    "PRINTER":      "printer",
    "RASHULT":      "rashult",
    "RAM":          "ram",
    "GRAPHICSCARD": "graphicscard",
    "ATV":            "atv",
    "BELT":           "belt",
    "CIRCUITBREAKER": "circuitbreaker",
    "COFFEE":         "coffee",
    "KNARREVIK":      "knarrevik",
    "GLADOM":         "gladom",
    "MARIUS":         "marius"
}

def body_relative_hand_position(hand_world: np.ndarray, head_pos: np.ndarray, head_quat: np.ndarray) -> np.ndarray:
    """
    True body-relative hand position: translate to head origin, then rotate
    into the head's local frame using the INVERSE head rotation.
    """
    translated = hand_world - head_pos
    head_rot = R.from_quat(head_quat)
    # Rotate world-frame vectors into the head's local frame
    rotated = head_rot.inv().apply(translated)
    return rotated

def relative_angular_velocity(quat: np.ndarray, dt: float) -> np.ndarray:
    """
    Mathematically correct angular velocity (rad/s) between consecutive orientations.
    """
    n = len(quat)
    angvel = np.zeros((n, 3))
    rots = R.from_quat(quat)
    for i in range(1, n):
        r_diff = rots[i] * rots[i - 1].inv()
        angvel[i] = r_diff.as_rotvec() / dt
    return angvel

def extract_behavioural_profile(brv: np.ndarray, head_pos_full: np.ndarray,
                                 head_quat_full: np.ndarray) -> dict:
    """
    brv columns: [0:11]=spatial, [11:22]=velocity-of-spatial.
    head_pos_full:  [N, 3] full scene-relative head position (for height proxy).
    head_quat_full: [N, 4] head orientation, needed for true angular velocity.
    """
    profile = {}
    profile["height_proxy"] = float(np.median(head_pos_full[:, 1]))

    wL_pos = brv[:, IDX["left_rel"]]
    wR_pos = brv[:, IDX["right_rel"]]
    wL_vel = brv[:, IDX_VEL["left_rel"]]
    wR_vel = brv[:, IDX_VEL["right_rel"]]

    vel_mag_L = np.linalg.norm(wL_vel, axis=1)
    vel_mag_R = np.linalg.norm(wR_vel, axis=1)

    profile["arm_reach_L"] = float(np.percentile(np.linalg.norm(wL_pos, axis=1), 95))
    profile["arm_reach_R"] = float(np.percentile(np.linalg.norm(wR_pos, axis=1), 95))
    profile["handedness_ratio"] = float(vel_mag_R.mean() / (vel_mag_L.mean() + 1e-8))
    profile["peak_vel_L"] = float(np.percentile(vel_mag_L, 99))
    profile["peak_vel_R"] = float(np.percentile(vel_mag_R, 99))

    # Acceleration = 2nd derivative of position = 1st derivative of velocity.
    accel_L = np.diff(wL_vel, axis=0, prepend=wL_vel[0:1]) / DT
    accel_R = np.diff(wR_vel, axis=0, prepend=wR_vel[0:1]) / DT

    # Jerk = 3rd derivative of position = 1st derivative of acceleration.
    jerk_L = np.diff(accel_L, axis=0, prepend=accel_L[0:1]) / DT
    jerk_R = np.diff(accel_R, axis=0, prepend=accel_R[0:1]) / DT
    jerk_mag_L = np.linalg.norm(jerk_L, axis=1)
    jerk_mag_R = np.linalg.norm(jerk_R, axis=1)
    profile["jerk_mean"] = float((jerk_mag_L.mean() + jerk_mag_R.mean()) / 2)

    # True angular velocity via relative rotation
    head_angvel = relative_angular_velocity(head_quat_full, DT)
    profile["head_sway"] = float(np.std(np.linalg.norm(head_angvel, axis=1)))

    n = min(len(vel_mag_L), len(vel_mag_R))
    profile["bilateral_symmetry"] = float(
        np.corrcoef(vel_mag_L[:n], vel_mag_R[:n])[0, 1]
    )
    return profile

def get_task_type(session_id: str) -> str:
    # Remove hyphens and underscores to make matching foolproof
    clean_sid = session_id.upper().replace("-", "").replace("_", "")
    
    for task in TASK_KEYWORDS:
        # Also clean the keyword just in case you defined them with underscores in your list
        clean_task = task.upper().replace("-", "").replace("_", "")
        
        if clean_task in clean_sid:
            return task 
            
    return "UNKNOWN"

VERB_TO_CLASS = {
    # grasp_release — reaching out, picking up / putting down
    "grab": "grasp_release", "place": "grasp_release",
    "lift": "grasp_release", "drop": "grasp_release",
    "hold": "grasp_release", "withdraw": "grasp_release",
    "touch": "grasp_release",

    # assembly — connecting, fastening, fitting parts together
    "assemble": "assembly", "attach": "assembly",
    "insert": "assembly", "remove": "assembly",
    "mount": "assembly", "unscrew": "assembly",
    "screw": "assembly", "lock": "assembly",
    "unlock": "assembly", "disassemble": "assembly",
    "exchange": "assembly",

    # manipulation — sustained movement / adjustment of held object
    "rotate": "manipulation", "slide": "manipulation",
    "adjust": "manipulation", "flip": "manipulation",
    "turn": "manipulation", "align": "manipulation",
    "push": "manipulation", "pull": "manipulation",
    "shift": "manipulation", "break": "manipulation",
    "hit": "manipulation",

    # control_action — brief activation / deactivation of a device
    "press": "control_action", "click": "control_action",
    "tap": "control_action", "turn_on": "control_action",
    "turn_off": "control_action", "open": "control_action",
    "close": "control_action",

    # transfer — moving contents, filling, emptying, cleaning
    "pour": "transfer", "load": "transfer",
    "empty": "transfer", "stack/pile": "transfer",
    "split": "transfer", "make": "transfer",
    "clean": "transfer", "mix/stir": "transfer",

    # locomotion — primarily NavVis sessions
    "walk": "locomotion", "move": "locomotion",
    "approach": "locomotion", "navigate": "locomotion",

    # idle — stationary observation / waiting
    "wait": "idle", "observe": "idle",
    "watch": "idle", "pause": "idle",
    "stand": "idle", "point": "idle",
    "inspect": "idle", "validate": "idle",
}

def load_session_labels(session_id: str, session_ann: list, n_frames: int,split: str) -> np.ndarray:
    """
    Returns frame-level label array of length n_frames.
    Takes the specific list of annotation events for this session.
    """
    CLASS_MAP = {
        "idle": 0, "locomotion": 1,
        "grasp_release": 2, "assembly": 3,
        "manipulation": 4, "control_action": 5,
        "transfer": 6, "anomalous": 7,
    }
    # If this is a test set video, mathematically guarantee it is blinded.
    if split == "test":
        return np.full(n_frames, -1, dtype=np.int8)
    # Check if we have data FIRST. 
    if not session_ann:
        print(f"  [warn] No annotation data found for {session_id} — setting labels to -1 (UNKNOWN)")
        # Return an array of -1s so PyTorch knows to ignore this during evaluation
        return np.full(n_frames, -1, dtype=np.int8)

    # If we DO have data, default to 0 (idle) and fill in the actions
    labels = np.zeros(n_frames, dtype=np.int8)
        
    task_type = get_task_type(session_id)
    fine_actions = [e for e in session_ann if e.get("label") == "Fine grained action"]

    # SAFEGUARD & DEBUG PRINT
    if fine_actions:
        sample_start = fine_actions[0].get("start", 0)
        # If the start time is massive (e.g., 15000), it's likely milliseconds or raw frames.
        # This print will help you verify the unit on your first run.
        if sample_start > 1000:
            print(f"  [URGENT WARN] {session_id} event start is {sample_start}. This is NOT seconds!")
        else:
            pass # It is safely in seconds
    
    for event in fine_actions:
        start_frame = int(event["start"] * FRAME_RATE_HZ)
        end_frame   = int(event["end"]   * FRAME_RATE_HZ)
        start_frame = max(0, min(start_frame, n_frames - 1))
        end_frame   = max(0, min(end_frame,   n_frames))
        
        verb = event["attributes"].get("Verb", "").lower().strip()
        correctness = event["attributes"].get("Action Correctness", "").lower()
        
        # Anomalous: wrong action not corrected
        if "wrong" in correctness and "not corrected" in correctness:
            labels[start_frame:end_frame] = CLASS_MAP["anomalous"]
            continue
            
        # NavVis specific
        if task_type == "NAVVIS" and verb == "approach":
            labels[start_frame:end_frame] = CLASS_MAP["locomotion"]
            continue
            
        mapped = VERB_TO_CLASS.get(verb)
        if mapped:
            labels[start_frame:end_frame] = CLASS_MAP[mapped]
        else:
            if verb:
                print(f"  [unmapped verb] '{verb}' in {session_id} — defaulting to idle")
                pass # Optional: print(f"  [unmapped] '{verb}'") to debug
                
    return labels

#Main processing function for a single session. 
def process_session_to_disk(session_id: str, split_output_dir: str,master_annotations: dict):
    meta_path = os.path.join(split_output_dir, f"{session_id}_meta.json")
    if os.path.exists(meta_path):
        # We don't need to print the skips anymore so it doesn't flood your screen
        return
    
    session_path = os.path.join(DATASET_DIR, session_id)
    export_dir = os.path.join(session_path, "Export_py")

    head_f = os.path.join(export_dir, "Head", "Head_sync.txt")
    hand_dir = (os.path.join(export_dir, "Hand")
                if os.path.exists(os.path.join(export_dir, "Hand"))
                else os.path.join(export_dir, "Hands"))
    left_f = os.path.join(hand_dir, "Left_sync.txt")
    right_f = os.path.join(hand_dir, "Right_sync.txt")

    head_raw_full = pd.read_csv(head_f, sep=r",|\s+", engine="python", header=None).values
    assert head_raw_full.shape[1] == HEAD_TOTAL_COLS, (
        f"{session_id}: expected exactly {HEAD_TOTAL_COLS} cols in Head_sync.txt, "
        f"got {head_raw_full.shape[1]}. Re-check the file format before proceeding."
    )
    head_raw = head_raw_full[:, HEAD_MATRIX_SLICE]   # the 16 matrix values


    left_full = pd.read_csv(left_f, sep=r",|\s+", engine="python", header=None).values
    right_full = pd.read_csv(right_f, sep=r",|\s+", engine="python", header=None).values
    assert left_full.shape[1] >= HAND_MIN_COLS and right_full.shape[1] >= HAND_MIN_COLS, (
        f"{session_id}: hand files have fewer than {HAND_MIN_COLS} columns "
        f"(left={left_full.shape[1]}, right={right_full.shape[1]}). "
        "Verify HAND_POS_SLICE against an actual sample line before proceeding."
    )
    left_raw = left_full[:, HAND_POS_SLICE]
    right_raw = right_full[:, HAND_POS_SLICE]

    n = min(len(head_raw), len(left_raw), len(right_raw))
    if not (len(head_raw) == len(left_raw) == len(right_raw)):
        print(f"  [warn] {session_id}: frame count mismatch "
              f"(head={len(head_raw)}, L={len(left_raw)}, R={len(right_raw)}) -> truncating to {n}")
    head_raw, left_raw, right_raw = head_raw[:n], left_raw[:n], right_raw[:n]

    matrices = head_raw.reshape(-1, 4, 4)
    head_pos = matrices[:, :3, 3]  # [N, 3] scene-relative
    
    # --- HARDWARE GLITCH FIX ---
    rot_matrices = matrices[:, :3, :3].copy()
    
    # 1. Catch NaN or Infinity values (Hardware dropouts)
    invalid_mask = np.isnan(rot_matrices).any(axis=(1, 2)) | np.isinf(rot_matrices).any(axis=(1, 2))
    rot_matrices[invalid_mask] = np.eye(3)
    
    # 2. Catch degenerate tracking matrices (A true rotation matrix determinant is 1.0)
    dets = np.linalg.det(rot_matrices)
    bad_dets = np.abs(dets - 1.0) > 0.1
    rot_matrices[bad_dets] = np.eye(3)

    # Now safely convert to quaternions
    head_quat = R.from_matrix(rot_matrices).as_quat()  # [N, 4]
    left_rel = body_relative_hand_position(left_raw, head_pos, head_quat)
    right_rel = body_relative_hand_position(right_raw, head_pos, head_quat)

    spatial = np.hstack([head_pos[:, :3], head_quat, left_rel, right_rel])
    assert spatial.shape[1] == SPATIAL_DIM, (
        f"spatial dim mismatch: {spatial.shape[1]} != {SPATIAL_DIM}"
    )

    vel = np.diff(spatial, axis=0, prepend=spatial[0:1]) / DT
    brv = np.hstack([spatial, vel])
    # Any 1-frame velocity spike > 10 m/s is physically impossible 
    brv[:, 20:26] = np.clip(brv[:, 20:26], -10.0, 10.0)
    assert brv.shape[1] == BRV_TOTAL_DIM, f"brv dim mismatch: {brv.shape[1]} != {BRV_TOTAL_DIM}"

    # 1. Save the 22-feature X Tensor
    np.save(os.path.join(split_output_dir, f"{session_id}_brv.npy"), brv)

    # 2. Extract and Save Behavioral Forensic Profile
    profile = extract_behavioural_profile(brv, head_pos, head_quat)
    with open(os.path.join(split_output_dir, f"{session_id}_profile.json"), "w") as f:
        json.dump(profile, f, indent=2)

    # 3. Extract and Save Ground Truth Labels (Y Tensor)
    task_type = get_task_type(session_id)
    #Exact string matching (ignoring file extensions)
    session_ann = []
    for key, val in master_annotations.items():
        clean_key = key.rsplit('.', 1)[0]
        if session_id == clean_key:
            session_ann = val
            break
            
    labels = load_session_labels(session_id, session_ann, n_frames=n, split=os.path.basename(split_output_dir))
    np.save(os.path.join(split_output_dir, f"{session_id}_labels.npy"), labels)


    # 4. Generate and Save Statistics Metadata
    metadata = {
        "session_id": session_id,
        "task_type": task_type,
        "n_frames": n,
        "frame_rate_hz": FRAME_RATE_HZ,
        "brv_shape": list(brv.shape),
        "class_distribution": {
            "idle": int((labels == 0).sum()),
            "locomotion": int((labels == 1).sum()),
            "grasp_release": int((labels == 2).sum()),
            "assembly": int((labels == 3).sum()),
            "manipulation": int((labels == 4).sum()),
            "control_action": int((labels == 5).sum()),
            "transfer": int((labels == 6).sum()),
            "anomalous": int((labels == 7).sum()),
        }
    }
    with open(os.path.join(split_output_dir, f"{session_id}_meta.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"  Processed: {session_id}  (brv shape={brv.shape}, task={task_type})")



if __name__ == "__main__":
    SPLITS_DIR = os.path.join(DATASET_DIR, "data-splits-v1_2")

    # ---> NEW: Load the giant JSON file into memory exactly once <---
    print(f"Loading master annotation file: {ANNOTATION_FILE} ...")
    with open(ANNOTATION_FILE, "r") as f:
        raw_annotations = json.load(f)
    
    master_annotations = {}
    
    if isinstance(raw_annotations, list):
        # DEBUG: Print the keys of the very first item so we know the exact structure
        if len(raw_annotations) > 0 and isinstance(raw_annotations[0], dict):
            print(f"  [Debug] JSON item keys: {list(raw_annotations[0].keys())}")

        for item in raw_annotations:
            if not isinstance(item, dict): 
                continue
            
            # Expanded search: checking every possible naming convention
            sid = (item.get("video_id") or item.get("session_id") or 
                   item.get("video") or item.get("name") or 
                   item.get("clip_uid") or item.get("video_uid") or 
                   item.get("id") or item.get("video_name") or
                   item.get("RecordingId"))
            
            # Ultimate Fallback: Check ALL string values to see if it looks like a session ID
            if not sid:
                for val in item.values():
                    if isinstance(val, str) and ("GoPro" in val or "DSLR" in val or "Nespresso" in val):
                        sid = val
                        break
            
            if sid:
                if "annotations" in item and isinstance(item["annotations"], list):
                    master_annotations[sid] = item["annotations"]
                elif "events" in item and isinstance(item["events"], list):
                    master_annotations[sid] = item["events"]
                else:
                    if sid not in master_annotations:
                        master_annotations[sid] = []
                    master_annotations[sid].append(item)

    elif isinstance(raw_annotations, dict):
        for key, val in raw_annotations.items():
            if isinstance(val, list):
                master_annotations[key] = val
            elif isinstance(val, dict):
                master_annotations[key] = val.get("annotations", val.get("events", []))

    print(f"Master annotations indexed successfully! ({len(master_annotations)} sessions found)\n")

    for split in SPLITS:
        split_file = os.path.join(SPLITS_DIR, f"{split}-v1_2.txt")
        
        if not os.path.exists(split_file):
            print(f"[skip] split file not found: {split_file}")
            continue
            
        split_output_dir = os.path.join(OUTPUT_DIR, split)
        os.makedirs(split_output_dir, exist_ok=True)
        
        print(f"\n=== {split.upper()} (Saving to {split_output_dir}) ===")
        
        with open(split_file, "r") as f:
            for line in f:
                session_id = line.strip()
                if not session_id:
                    continue
                try:
                    # Pass the master_annotations dictionary down into the function
                    process_session_to_disk(session_id, split_output_dir, master_annotations)
                except Exception as e:
                    print(f"  [error] {session_id}: {e}")
                    
    