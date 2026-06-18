import os
import json
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R

# ============================================================================
# CONFIGURATION
# ============================================================================
DATASET_DIR = r"C:\Users\Student3\Documents\xrtraceai\Data_set\Halo_assist_dataset"
OUTPUT_DIR = r"C:\Users\Student3\Documents\xrtraceai\preprocess_data"

# Verified directly from the data: col0 increments by 0.03333... -> 30 fps.
# (Original script hardcoded 1/29.5; docstring claimed 1/90. Both were wrong.)
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

# ----------------------------------------------------------------------------
# Explicit, named feature layout for the BRV tensor.
# spatial block (11 cols): [head_pos_y(1), head_quat(4), left_rel(3), right_rel(3)]
# Final brv = hstack([spatial, velocity_of_spatial]) -> 22 cols total, NOT 19.
# (Bug 1 fix: the original code claimed "19-Feature" in a comment, which caused
#  wL_vel/wR_vel to be sliced from the wrong columns -- mixing head-rotation
#  velocity into what was assumed to be wrist velocity.)
# ----------------------------------------------------------------------------
SPATIAL_DIM = 11
IDX = {
    "head_pos_y": slice(0, 1),
    "head_quat":  slice(1, 5),
    "left_rel":   slice(5, 8),
    "right_rel":  slice(8, 11),
}
VEL_OFFSET = SPATIAL_DIM
IDX_VEL = {k: slice(v.start + VEL_OFFSET, v.stop + VEL_OFFSET)
           for k, v in IDX.items() if k in ("left_rel", "right_rel")}
BRV_TOTAL_DIM = SPATIAL_DIM * 2  # 22


def quaternion_angular_velocity(quat: np.ndarray, dt: float) -> np.ndarray:
    """
    True angular velocity (rad/s) from a quaternion sequence.
    (Bug 3 fix: quaternions are not a linear space, so np.diff directly on
    quaternion components -- as the original code did for 'head_sway' -- is
    mathematically invalid. We convert to rotation vectors first.)
    quat: [N, 4] in (x, y, z, w) order (scipy convention).
    Returns: [N, 3], first row is zero (no t-1 reference available).
    """
    rotvecs = R.from_quat(quat).as_rotvec()
    angvel = np.diff(rotvecs, axis=0, prepend=rotvecs[0:1]) / dt
    return angvel


def extract_behavioural_profile(brv: np.ndarray, head_pos_full: np.ndarray) -> dict:
    """
    brv columns: [0:11]=spatial, [11:22]=velocity-of-spatial (see IDX / IDX_VEL).
    head_pos_full: full 3D scene-relative head position [N, 3] (for height proxy).
    """
    profile = {}
    profile["height_proxy"] = float(np.median(head_pos_full[:, 1]))

    wL_pos = brv[:, IDX["left_rel"]]
    wR_pos = brv[:, IDX["right_rel"]]
    wL_vel = brv[:, IDX_VEL["left_rel"]]   # now genuinely wrist velocity
    wR_vel = brv[:, IDX_VEL["right_rel"]]

    vel_mag_L = np.linalg.norm(wL_vel, axis=1)
    vel_mag_R = np.linalg.norm(wR_vel, axis=1)

    profile["arm_reach_L"] = float(np.percentile(np.linalg.norm(wL_pos, axis=1), 95))
    profile["arm_reach_R"] = float(np.percentile(np.linalg.norm(wR_pos, axis=1), 95))
    profile["handedness_ratio"] = float(vel_mag_R.mean() / (vel_mag_L.mean() + 1e-8))
    profile["peak_vel_L"] = float(np.percentile(vel_mag_L, 99))
    profile["peak_vel_R"] = float(np.percentile(vel_mag_R, 99))

    accel_L = np.diff(wL_vel, axis=0, prepend=wL_vel[0:1]) / DT
    accel_R = np.diff(wR_vel, axis=0, prepend=wR_vel[0:1]) / DT
    profile["jerk_mean"] = float((np.abs(accel_L).mean() + np.abs(accel_R).mean()) / 2)

    head_quat = brv[:, IDX["head_quat"]]
    profile["head_sway"] = float(np.std(quaternion_angular_velocity(head_quat, DT)))

    n = min(len(vel_mag_L), len(vel_mag_R))
    profile["bilateral_symmetry"] = float(
        np.corrcoef(vel_mag_L[:n], vel_mag_R[:n])[0, 1]
    )
    return profile


def process_session_to_disk(session_id: str, split_output_dir: str):
    session_path = os.path.join(DATASET_DIR, session_id)
    export_dir = os.path.join(session_path, "Export_py")

    head_f = os.path.join(export_dir, "Head", "Head_sync.txt")
    hand_dir = (os.path.join(export_dir, "Hand")
                if os.path.exists(os.path.join(export_dir, "Hand"))
                else os.path.join(export_dir, "Hands"))
    left_f = os.path.join(hand_dir, "Left_sync.txt")
    right_f = os.path.join(hand_dir, "Right_sync.txt")

    # ---- Head: verified 18-col layout (Bug 4/5 fix: was an unverified
    # assumption before; now checked against the real sample you provided) ----
    head_raw_full = pd.read_csv(head_f, sep=r",|\s+", engine="python", header=None).values
    assert head_raw_full.shape[1] == HEAD_TOTAL_COLS, (
        f"{session_id}: expected exactly {HEAD_TOTAL_COLS} cols in Head_sync.txt, "
        f"got {head_raw_full.shape[1]}. Re-check the file format before proceeding."
    )
    head_raw = head_raw_full[:, HEAD_MATRIX_SLICE]   # the 16 matrix values

    # ---- Hands: slice asserted, not assumed (fails loudly instead of
    # silently misaligning, unlike the original script) ----
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
    head_pos = matrices[:, :3, 3]                              # [N, 3] scene-relative
    head_quat = R.from_matrix(matrices[:, :3, :3]).as_quat()   # [N, 4]

    left_rel = left_raw - head_pos
    right_rel = right_raw - head_pos

    spatial = np.hstack([head_pos[:, 1:2], head_quat, left_rel, right_rel])
    assert spatial.shape[1] == SPATIAL_DIM, (
        f"spatial dim mismatch: {spatial.shape[1]} != {SPATIAL_DIM}"
    )

    vel = np.diff(spatial, axis=0, prepend=spatial[0:1]) / DT
    brv = np.hstack([spatial, vel])
    assert brv.shape[1] == BRV_TOTAL_DIM, f"brv dim mismatch: {brv.shape[1]} != {BRV_TOTAL_DIM}"

    np.save(os.path.join(split_output_dir, f"{session_id}_brv.npy"), brv)

    profile = extract_behavioural_profile(brv, head_pos)
    with open(os.path.join(split_output_dir, f"{session_id}_profile.json"), "w") as f:
        json.dump(profile, f, indent=2)

    print(f"  Processed: {session_id}  (brv shape={brv.shape}, fps={FRAME_RATE_HZ})")

if __name__ == "__main__":
    # Point directly to the folder containing the text files
    SPLITS_DIR = os.path.join(DATASET_DIR, "data-splits-v1_2")

    for split in ["train", "val", "test"]:
        split_file = os.path.join(SPLITS_DIR, f"{split}-v1_2.txt")
        
        if not os.path.exists(split_file):
            print(f"[skip] split file not found: {split_file}")
            continue
            
        # Create a specific folder for this split (e.g., preprocess_data/train)
        split_output_dir = os.path.join(OUTPUT_DIR, split)
        os.makedirs(split_output_dir, exist_ok=True)
        
        print(f"\n=== {split.upper()} (Saving to {split_output_dir}) ===")
        
        with open(split_file, "r") as f:
            for line in f:
                session_id = line.strip()
                if not session_id:
                    continue
                try:
                    # Pass the specific output folder so files land in the right place
                    process_session_to_disk(session_id, split_output_dir)
                except Exception as e:
                    print(f"  [error] {session_id}: {e}")