
"""
Corrected normalization verification script.
 
The original spot-check in normalizer.py samples the first 5 sessions via
glob(), which returns results in ALPHABETICAL order -- not a random sample.
If session naming clusters similar task types/dates together, this produces
a misleadingly large deviation from the expected mean~=0, std~=1 even when
the global scaler itself is perfectly correct.
 
This script does two things instead:
1. Recomputes the TRUE global mean/std across ALL normalized train frames
   (not a 5-session subset). This should be very close to 0/1 by
   construction -- if it's NOT, something in the transform itself is broken.
2. Separately reports per-session deviation statistics across a RANDOM
   sample, so you can see the natural session-to-session variance without
   confusing it for a scaler bug.
"""
import os
import numpy as np
from pathlib import Path
 
OUTPUT_DIR = os.getcwd()
TRAIN_DIR = os.path.join(OUTPUT_DIR, "train")
BRV_SUFFIX = "_brv.npy"
N_RANDOM_SESSIONS = 25
SEED = 42
 
train_files = sorted(Path(TRAIN_DIR).glob(f"*{BRV_SUFFIX}"))
if not train_files:
    raise FileNotFoundError(f"No {BRV_SUFFIX} files found in {TRAIN_DIR}")
 
print(f"Found {len(train_files)} normalized BRV files in train.")
 
# ---- Check 1: TRUE global mean/std across every frame, every session ----
print("\n=== Check 1: Global statistics across ALL train frames ===")
n_total = 0
sum_x = None
sum_x_sq = None
 
for fp in train_files:
    arr = np.load(fp).astype(np.float64)
    arr = np.nan_to_num(arr, nan=0.0)
    T, D = arr.shape
    if sum_x is None:
        sum_x = np.zeros(D)
        sum_x_sq = np.zeros(D)
    n_total += T
    sum_x += arr.sum(axis=0)
    sum_x_sq += (arr ** 2).sum(axis=0)
 
global_mean = sum_x / n_total
global_var = sum_x_sq / n_total - global_mean ** 2
global_std = np.sqrt(np.maximum(global_var, 0))
 
print(f"  Frames checked        : {n_total:,}")
print(f"  Mean of feature means : {global_mean.mean():.6f}  (expect ~0)")
print(f"  Max |mean| any feature: {np.abs(global_mean).max():.6f}  (expect < 0.01)")
print(f"  Mean of feature stds  : {global_std.mean():.4f}  (expect ~1)")
print(f"  Std range across feats: [{global_std.min():.4f}, {global_std.max():.4f}]")
 
if np.abs(global_mean).max() > 0.05:
    print("\n  >>> WARNING: global mean is NOT close to zero. This indicates a real")
    print("      problem with the normalization transform itself -- not sampling noise.")
    print("      Check: was the scaler fit on the same data being normalized? Did any")
    print("      file get skipped or double-processed?")
else:
    print("\n  Global statistics look correct. The normalization transform itself is sound.")
 
# ---- Check 2: per-session variance, on a RANDOM sample, for context only ----
print(f"\n=== Check 2: Per-session deviation, {N_RANDOM_SESSIONS} RANDOM sessions (context only) ===")
print("  (This is expected to show non-zero deviation -- individual sessions have")
print("   their own local behavioral distributions. This is NOT a scaler bug.)\n")
 
rng = np.random.default_rng(SEED)
sample_files = rng.choice(train_files, size=min(N_RANDOM_SESSIONS, len(train_files)), replace=False)
 
per_session_max_mean_dev = []
for fp in sample_files:
    arr = np.load(fp)
    m = np.abs(arr.mean(axis=0)).max()
    per_session_max_mean_dev.append(m)
 
per_session_max_mean_dev = np.array(per_session_max_mean_dev)
print(f"  Median per-session max|mean| deviation : {np.median(per_session_max_mean_dev):.4f}")
print(f"  Max  per-session max|mean| deviation   : {per_session_max_mean_dev.max():.4f}")
print(f"  (Large individual-session deviations here are normal and expected.")
print(f"   Only Check 1's GLOBAL deviation matters for verifying the scaler is correct.)")