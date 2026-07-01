"""
Augment rare classes: control_action (5) and transfer (6).
Same approach as augument_loco.py — duplicate sessions with mild noise/scale/
time-warp so the model sees more examples without distorting the labels.

Run order (after augument_loco.py, before gen_train_analomy.py):
    python preprocess_data/augument_loco.py
    python preprocess_data/augment_rare.py
    python preprocess_data/gen_train_analomy.py
    ...
"""

import numpy as np
import json
from pathlib import Path

TRAIN_DIR             = r"C:\Users\Student3\Documents\xrtraceai\preprocess_data\train"
RARE_CLASSES          = {5: "control_action", 6: "transfer"}
MIN_PCT               = 0.05    # session must have >= 5 % of the rare class to qualify
AUGMENTATION_MULTIPLIER = 8     # create 8 copies per qualifying session


def augment_rare_sessions():
    brv_files = sorted(Path(TRAIN_DIR).glob("*_brv.npy"))
    created   = {cls: 0 for cls in RARE_CLASSES}

    for fp in brv_files:
        if "SYNTH_" in fp.stem or "LOCO_AUG_" in fp.stem or "RARE_AUG_" in fp.stem:
            continue

        labels_path = str(fp).replace("_brv.npy", "_labels.npy")
        meta_path   = str(fp).replace("_brv.npy", "_meta.json")

        if not Path(labels_path).exists() or not Path(meta_path).exists():
            continue

        labels     = np.load(labels_path)
        brv        = np.load(fp)
        session_id = fp.stem.replace("_brv", "")

        with open(meta_path, "r") as f:
            orig_meta = json.load(f)

        for cls_idx, cls_name in RARE_CLASSES.items():
            cls_pct = (labels == cls_idx).mean()
            if cls_pct < MIN_PCT:
                continue

            for aug_idx in range(AUGMENTATION_MULTIPLIER):
                aug_brv = brv.copy()

                # 1. Gaussian noise
                noise_scale = 0.02 * np.random.uniform(0.5, 1.5)
                aug_brv += np.random.normal(0, noise_scale, aug_brv.shape)

                # 2. Global magnitude scale
                scale = np.random.uniform(0.92, 1.08)
                aug_brv *= scale

                # 3. Time-warp ±8 %
                T    = len(aug_brv)
                warp = np.random.uniform(0.92, 1.08)
                new_T   = max(int(T * warp), 60)
                indices = np.linspace(0, T - 1, new_T)
                aug_brv = np.array([
                    np.interp(indices, np.arange(T), aug_brv[:, f])
                    for f in range(aug_brv.shape[1])
                ]).T

                # Trim/pad labels
                if len(indices) > T:
                    aug_labels = np.concatenate([
                        labels, np.full(len(indices) - T, labels[-1])
                    ])[:len(indices)]
                else:
                    aug_labels = labels[:len(indices)]

                out_name = f"RARE_AUG_{cls_name}_{aug_idx:02d}_{session_id}"

                np.save(Path(TRAIN_DIR) / f"{out_name}_brv.npy",    aug_brv.astype(np.float32))
                np.save(Path(TRAIN_DIR) / f"{out_name}_labels.npy", aug_labels.astype(np.int8))

                counts   = np.bincount(aug_labels.astype(np.int64), minlength=8)
                aug_meta = orig_meta.copy()
                aug_meta["session_id"] = out_name
                aug_meta["task_type"]  = "RARE_AUG"
                aug_meta["class_distribution"] = {
                    "idle":           int(counts[0]),
                    "locomotion":     int(counts[1]),
                    "grasp_release":  int(counts[2]),
                    "assembly":       int(counts[3]),
                    "manipulation":   int(counts[4]),
                    "control_action": int(counts[5]),
                    "transfer":       int(counts[6]),
                    "anomalous":      int(counts[7]),
                }

                with open(Path(TRAIN_DIR) / f"{out_name}_meta.json", "w") as f:
                    json.dump(aug_meta, f, indent=2)

                created[cls_idx] += 1

            print(f"  {session_id}: {cls_name}={cls_pct:.1%} — "
                  f"created {AUGMENTATION_MULTIPLIER} augments")

    print(f"\nTotal rare-class augmented files created:")
    for cls_idx, cls_name in RARE_CLASSES.items():
        print(f"  [{cls_idx}] {cls_name}: {created[cls_idx]}")


if __name__ == "__main__":
    np.random.seed(44)
    augment_rare_sessions()
