import numpy as np
import json
from pathlib import Path

TRAIN_DIR = r"C:\Users\Student3\Documents\xrtraceai\preprocess_data\train"
LOCOMOTION_CLASS = 1
AUGMENTATION_MULTIPLIER = 8  # create 8 copies 

def augment_locomotion_sessions():
    brv_files = sorted(Path(TRAIN_DIR).glob("*_brv.npy"))
    created = 0
    
    for fp in brv_files:
        # Skip files that are already synthetic 
        if "SYNTH_" in fp.stem or "LOCO_AUG_" in fp.stem:
            continue
            
        brv = np.load(fp)
        labels_path = str(fp).replace("_brv.npy", "_labels.npy")
        meta_path = str(fp).replace("_brv.npy", "_meta.json")
        
        if not Path(labels_path).exists() or not Path(meta_path).exists():
            continue
            
        labels = np.load(labels_path)
        
        # Check if this session has significant locomotion
        loco_pct = (labels == LOCOMOTION_CLASS).mean()
        if loco_pct < 0.05:  # less than 5% locomotion — skip
            continue
            
        session_id = fp.stem.replace("_brv", "")
        
        # Load the original metadata to use as a template
        with open(meta_path, "r") as f:
            orig_meta = json.load(f)
            
        for aug_idx in range(AUGMENTATION_MULTIPLIER):
            aug_brv = brv.copy()
            
            # Augmentation 1: add small Gaussian noise to all features
            noise_scale = 0.02 * np.random.uniform(0.5, 1.5)
            aug_brv += np.random.normal(0, noise_scale, aug_brv.shape)
            
            # Augmentation 2: scale overall magnitude slightly
            scale = np.random.uniform(0.92, 1.08)
            aug_brv *= scale
            
            # Augmentation 3: time-warp — stretch/compress by ±8%
            T = len(aug_brv)
            warp = np.random.uniform(0.92, 1.08)
            new_T = max(int(T * warp), 60)
            indices = np.linspace(0, T-1, new_T)
            aug_brv = np.array([
                np.interp(indices, np.arange(T), aug_brv[:, f])
                for f in range(aug_brv.shape[1])
            ]).T
            
            # Trim/pad labels to match
            if len(indices) > T:
                aug_labels = np.concatenate([
                    labels, np.full(len(indices)-T, labels[-1])
                ])[:len(indices)]
            else:
                aug_labels = labels[:len(indices)]
                
            out_name = f"LOCO_AUG_{aug_idx:02d}_{session_id}"
            
            # Save the NPY files
            np.save(Path(TRAIN_DIR) / f"{out_name}_brv.npy", aug_brv.astype(np.float32))
            np.save(Path(TRAIN_DIR) / f"{out_name}_labels.npy", aug_labels.astype(np.int8))
            
            # Save the Meta JSON file 
            counts = np.bincount(aug_labels, minlength=4)
            aug_meta = orig_meta.copy()
            aug_meta["session_id"] = out_name
            aug_meta["task_type"] = "LOCOMOTION_AUG"
            aug_meta["class_distribution"] = {
                "idle": int(counts[0]),
                "locomotion": int(counts[1]),
                "object_interaction": int(counts[2]),
                "anomalous": int(counts[3])
            }
            
            with open(Path(TRAIN_DIR) / f"{out_name}_meta.json", "w") as f:
                json.dump(aug_meta, f, indent=2)
            
            
            created += 1
            
        print(f"  {session_id}: loco={loco_pct:.1%} — created {AUGMENTATION_MULTIPLIER} augments")
        
    print(f"\nTotal augmented locomotion files created: {created}")

if __name__ == "__main__":
    augment_locomotion_sessions()