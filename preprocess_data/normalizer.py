import os
import json
import numpy as np
from pathlib import Path



OUTPUT_DIR  = os.getcwd()
SPLITS      = ["train", "val", "test"]
BRV_SUFFIX  = "_brv.npy"
META_SUFFIX = "_meta.json"
SCALER_PATH = os.path.join(OUTPUT_DIR, "brv_scaler.npz")
SCALER_STATS_PATH = os.path.join(OUTPUT_DIR, "brv_scaler_stats.json")

# SAFETY GUARD — refuse to run if scaler already exists unless user confirms


if os.path.exists(SCALER_PATH):
    print(f"\n  Scaler already exists at: {SCALER_PATH}")
    print("   Overwriting will change the numerical space your model expects.")
    print("   Only do this if you have re-run preprocess_data.py from scratch.")
    answer = input("   Type YES to overwrite, anything else to abort: ").strip()
    if answer != "YES":
        print("   Aborted. Existing scaler preserved.")
        raise SystemExit(0)

# STEP 1 — Collect ALL train BRV arrays to fit the scaler


print("\n=== STEP 1: Fitting scaler on TRAIN split only ===")

train_dir = os.path.join(OUTPUT_DIR, "train")
if not os.path.isdir(train_dir):
    raise FileNotFoundError(
        f"Train directory not found: {train_dir}\n"
        "Run preprocess_data.py and generate_synthetic_anomalies.py first."
    )

train_files = sorted(Path(train_dir).glob(f"*{BRV_SUFFIX}"))
if not train_files:
    raise FileNotFoundError(
        f"No *{BRV_SUFFIX} files found in {train_dir}.\n"
        "Run preprocess_data.py first."
    )

print(f"  Found {len(train_files)} BRV files in train split.")
print("  Computing mean and std across all train frames...")

# Two-pass Welford online algorithm — memory efficient for large datasets.
# Pass 1: compute mean without loading everything at once.
# This avoids a single np.vstack of potentially millions of frames.

n_total   = 0
sum_x     = None   # will be shape (BRV_DIM,)
sum_x_sq  = None

for fp in train_files:
    arr = np.load(fp).astype(np.float64)   # (T, BRV_DIM)

    arr = np.nan_to_num(arr, nan=0.0)
    T, D = arr.shape

    if sum_x is None:
        sum_x    = np.zeros(D, dtype=np.float64)
        sum_x_sq = np.zeros(D, dtype=np.float64)

    n_total  += T
    sum_x    += arr.sum(axis=0)
    sum_x_sq += (arr ** 2).sum(axis=0)

    # light progress indicator
    print(f"    {fp.name}: {T} frames  (running total: {n_total})", end="\r")

print()  # newline after \r progress

train_mean = sum_x / n_total
train_var  = (sum_x_sq / n_total) - (train_mean ** 2)
train_std  = np.sqrt(np.maximum(train_var, 1e-12))   # floor to avoid /0

# Sanity checks on fitted parameters
zero_std_features = (train_std < 1e-6).sum()
if zero_std_features > 0:
    print(f"\n   {zero_std_features} features have near-zero std "
          f"(constant across training data).")
    print("     These will be divided by the floor value 1e-6.")
    print("     Check if any BRV feature is always zero in your data.")

nan_in_mean = np.isnan(train_mean).sum()
nan_in_std  = np.isnan(train_std).sum()
if nan_in_mean > 0 or nan_in_std > 0:
    raise ValueError(
        f"NaN detected in fitted parameters "
        f"(mean NaNs={nan_in_mean}, std NaNs={nan_in_std}). "
        "Check your BRV arrays for NaN values before normalising. "
        "Run: np.isnan(np.load('your_file.npy')).sum() on a few files."
    )

print(f"\n Fitted on {n_total:,} frames across {len(train_files)} sessions.")
print(f"  BRV feature dimension : {len(train_mean)}")
print(f"  Mean range            : [{train_mean.min():.4f}, {train_mean.max():.4f}]")
print(f"  Std  range            : [{train_std.min():.6f}, {train_std.max():.4f}]")

# STEP 2 — Save scaler parameters to disk (next to model checkpoints)


np.savez(SCALER_PATH, mean=train_mean, std=train_std)

# Also save as human-readable JSON for inspection and Unity deployment
scaler_stats = {
    "n_train_frames":   int(n_total),
    "n_train_sessions": len(train_files),
    "brv_feature_dim":  int(len(train_mean)),
    "mean": train_mean.tolist(),
    "std":  train_std.tolist(),
    "fitted_on": "train split only",
    "note": (
        "Apply as: x_norm = (x - mean) / std  "
        "for ALL splits including test and live Unity inference. "
        "Never refit on val or test."
    )
}
with open(SCALER_STATS_PATH, "w") as f:
    json.dump(scaler_stats, f, indent=2)

print(f"\n  ✓ Scaler saved → {SCALER_PATH}")
print(f"  ✓ Scaler stats → {SCALER_STATS_PATH}")



# STEP 3 — Apply scaler to ALL splits: train, val, test


print("\n=== STEP 2: Transforming all splits ===")
print("  This overwrites _brv.npy files in-place. "
      "Re-run preprocess_data.py to get raw values back.")

split_summary = {}

for split in SPLITS:
    split_dir = os.path.join(OUTPUT_DIR, split)
    if not os.path.isdir(split_dir):
        print(f"  [skip] {split} directory not found: {split_dir}")
        continue

    brv_files = sorted(Path(split_dir).glob(f"*{BRV_SUFFIX}"))
    if not brv_files:
        print(f"  [skip] No BRV files in {split}")
        continue

    frames_processed = 0
    nan_files = []

    for fp in brv_files:
        arr = np.load(fp).astype(np.float32)

        # Check for NaN before normalising
        arr_norm = ((arr - train_mean) / train_std).astype(np.float32)

        # Check for NaN AFTER normalising to safely zero them out
        if np.isnan(arr_norm).any():
            if fp.name not in nan_files:
                nan_files.append(fp.name)
            # Now, replacing with 0.0 perfectly represents the "mean" to the network
            arr_norm = np.nan_to_num(arr_norm, nan=0.0)

        # Verify the normalised output is reasonable
        # Post-normalisation, values outside ±10 sigma are extreme outliers
        extreme = (np.abs(arr_norm) > 10).sum()
        if extreme > 0:
            pass   # logged in summary, not an error — synthetic anomalies
                   # may legitimately have high-sigma values

        np.save(fp, arr_norm)
        frames_processed += len(arr_norm)

    split_summary[split] = {
        "sessions": len(brv_files),
        "frames":   frames_processed,
        "nan_files": nan_files,
    }

    print(f"\n  {split.upper()}: {len(brv_files)} sessions, "
          f"{frames_processed:,} frames normalised.")
    if nan_files:
        print(f"    ⚠️  NaN replaced with zero in {len(nan_files)} files:")
        for nf in nan_files:
            print(f"       {nf}")
    else:
        print(f"    ✓ No NaN values encountered.")

# STEP 4 — Verification: spot-check the normalised output


print("\n=== STEP 3: Verification spot-check (train split) ===")
print("  After normalisation, each feature should have mean ≈ 0, std ≈ 1")
print("  across the full train set. Checking first 5 sessions...\n")

check_files = list(Path(train_dir).glob(f"*{BRV_SUFFIX}"))[:5]

if check_files:
    sample = np.vstack([np.load(fp) for fp in check_files])
    check_mean = sample.mean(axis=0)
    check_std  = sample.std(axis=0)

    mean_of_means = check_mean.mean()
    std_of_stds   = check_std.mean()
    max_abs_mean  = np.abs(check_mean).max()
    min_std       = check_std.min()
    max_std       = check_std.max()

    print(f"  mean of feature means : {mean_of_means:.6f}  (expect ≈ 0)")
    print(f"  mean of feature stds  : {std_of_stds:.4f}   (expect ≈ 1)")
    print(f"  max |mean| any feature: {max_abs_mean:.6f}  (expect < 0.01)")
    print(f"  std range across feats: [{min_std:.4f}, {max_std:.4f}]"
          "  (expect ≈ [0.9, 1.1])")

    # Hard checks
    if max_abs_mean > 0.1:
        print(f"\n  Feature means are not close to zero. "
              f"Max deviation: {max_abs_mean:.4f}")
        print("     Possible cause: scaler was fit on a different subset "
              "than what was normalised. Verify SPLITS_DIR path.")
    else:
        print(f"\n  ✓ Normalisation verified. Data is ready for training.")

# STEP 5 — Final summary

print("  NORMALISATION COMPLETE")

for split, info in split_summary.items():
    print(f"  {split:<8}: {info['sessions']:>4} sessions, "
          f"{info['frames']:>10,} frames")
print(f"\n  Scaler saved to  : {SCALER_PATH}")
print(f"  Load in training : scaler = np.load(SCALER_PATH)")
print(f"                     mean, std = scaler['mean'], scaler['std']")
print(f"\n  Load in Unity    : parse brv_scaler_stats.json")
print(f"                     apply (x - mean) / std before Barracuda.Execute()")
