import os
import json
import glob
import matplotlib.pyplot as plt

BASE_DIR = r"C:\Users\Student3\Documents\xrtraceai\preprocess_data"
SPLITS = ["train", "val", "test"]

print("Scanning directories for updated frame counts and generating all graphs...")

# Dictionaries to hold separate stats for each split/view
all_stats = {
    "train_pre": {},
    "train_post": {},
    "val": {},
    "val_post": {},   
    "test": {}
}

global_totals = {
    "train": {"idle": 0, "loco": 0, "obj": 0, "anom": 0, "synth_videos": 0, "total_videos": 0},
    "val": {"idle": 0, "loco": 0, "obj": 0, "anom": 0, "synth_videos": 0, "total_videos": 0},
    "test": {"idle": 0, "loco": 0, "obj": 0, "anom": 0, "synth_videos": 0, "total_videos": 0}
}

for split in SPLITS:
    split_dir = os.path.join(BASE_DIR, split)
    if not os.path.exists(split_dir):
        print(f"  [skip] {split} directory not found.")
        continue
        
    meta_files = glob.glob(os.path.join(split_dir, "*_meta.json"))
    global_totals[split]["total_videos"] = len(meta_files)
    
    for meta_file in meta_files:
        is_synth = "SYNTH_" in os.path.basename(meta_file)
        if is_synth:
            global_totals[split]["synth_videos"] += 1
            
        with open(meta_file, "r") as f:
            meta = json.load(f)
            
        dist = meta.get("class_distribution", {})
        task = meta.get("task_type", "UNKNOWN")
        
        # --- FIX: Re-assign synthetic anomalies to their original tasks ---
        if is_synth and task == "SYNTHETIC_ANOMALY":
            orig_sid = meta.get("session_id", "")
            for prefix in ["SYNTH_HYPER_SPEED_", "SYNTH_ERRATIC_TREMOR_", "SYNTH_AGGRESSIVE_REACH_"]:
                if orig_sid.startswith(prefix):
                    orig_sid = orig_sid.replace(prefix, "")
                    break
                    
            orig_meta_path = os.path.join(split_dir, f"{orig_sid}_meta.json")
            if os.path.exists(orig_meta_path):
                with open(orig_meta_path, "r") as orig_f:
                    orig_meta = json.load(orig_f)
                    task = orig_meta.get("task_type", task)
        # -----------------------------------------------------------------
        
        idle = dist.get("idle", 0)
        loco = dist.get("locomotion", 0)
        obj = dist.get("object_interaction", 0)
        anom = dist.get("anomalous", 0)
        
        # Update global counts (using post-synth numbers for train)
        global_totals[split]["idle"] += idle
        global_totals[split]["loco"] += loco
        global_totals[split]["obj"] += obj
        global_totals[split]["anom"] += anom
        
        # Initialize task dictionaries if missing
        if split == "train":
            if task not in all_stats["train_post"]: all_stats["train_post"][task] = {"idle": 0, "locomotion": 0, "object_interaction": 0, "anomalous": 0}
            if task not in all_stats["train_pre"]: all_stats["train_pre"][task] = {"idle": 0, "locomotion": 0, "object_interaction": 0, "anomalous": 0}
            
            # Add to POST-synth stats
            all_stats["train_post"][task]["idle"] += idle
            all_stats["train_post"][task]["locomotion"] += loco
            all_stats["train_post"][task]["object_interaction"] += obj
            all_stats["train_post"][task]["anomalous"] += anom
            
            # Add to PRE-synth stats 
            if not is_synth:
                all_stats["train_pre"][task]["idle"] += idle
                all_stats["train_pre"][task]["locomotion"] += loco
                all_stats["train_pre"][task]["object_interaction"] += obj
                all_stats["train_pre"][task]["anomalous"] += anom
        elif split == "val":
            if task not in all_stats["val_post"]: all_stats["val_post"][task] = {"idle": 0, "locomotion": 0, "object_interaction": 0, "anomalous": 0}
            if task not in all_stats["val"]: all_stats["val"][task] = {"idle": 0, "locomotion": 0, "object_interaction": 0, "anomalous": 0}
            
            # Add to POST-synth stats
            all_stats["val_post"][task]["idle"] += idle
            all_stats["val_post"][task]["locomotion"] += loco
            all_stats["val_post"][task]["object_interaction"] += obj
            all_stats["val_post"][task]["anomalous"] += anom
            
            # Add to PRE-synth stats 
            if not is_synth:
                all_stats["val"][task]["idle"] += idle
                all_stats["val"][task]["locomotion"] += loco
                all_stats["val"][task]["object_interaction"] += obj
                all_stats["val"][task]["anomalous"] += anom
        
        
        else:
            # Test split
            if task not in all_stats[split]: all_stats[split][task] = {"idle": 0, "locomotion": 0, "object_interaction": 0, "anomalous": 0}
            all_stats[split][task]["idle"] += idle
            all_stats[split][task]["locomotion"] += loco
            all_stats[split][task]["object_interaction"] += obj
            all_stats[split][task]["anomalous"] += anom

print("\n=== FINAL DATASET STATISTICS ===")
for split in SPLITS:
    totals = global_totals[split]
    if totals["total_videos"] == 0: continue
    
    total_frames = totals["idle"] + totals["loco"] + totals["obj"] + totals["anom"]
    
    print(f"\n--- {split.upper()} SPLIT ---")
    print(f"Total Videos: {totals['total_videos']} (Synthetic: {totals['synth_videos']})")
    if total_frames > 0:
        print(f"Idle Frames:           {totals['idle']:,} ({totals['idle']/total_frames*100:.1f}%)")
        print(f"Locomotion Frames:     {totals['loco']:,} ({totals['loco']/total_frames*100:.1f}%)")
        print(f"Object Interaction:    {totals['obj']:,} ({totals['obj']/total_frames*100:.1f}%)")
        print(f"ANOMALOUS FRAMES:      {totals['anom']:,} ({totals['anom']/total_frames*100:.1f}%)")
    else:
        print("No valid labeled frames found (likely masked with -1).")

def generate_distribution_graph(task_stats, title, output_path):
    tasks = sorted(list(task_stats.keys()))
    num_tasks = len(tasks)

    if num_tasks > 0:
        fig, axes = plt.subplots(1, num_tasks, figsize=(3 * num_tasks, 4), sharey=True)
        if num_tasks == 1:
            axes = [axes]
            
        fig.suptitle(title, fontsize=16, fontweight='bold')
        
        classes = ["idle", "loco", "obj_int", "anomalous"]
        colors = ['#5b6777', '#3498db', '#1abc9c', '#e74c3c']
        
        for i, task in enumerate(tasks):
            ax = axes[i]
            stats = task_stats[task]
            
            counts = [stats["idle"], stats["locomotion"], stats["object_interaction"], stats["anomalous"]]
            total = sum(counts)
            pcts = [c / total * 100 if total > 0 else 0 for c in counts]
            
            bars = ax.bar(classes, pcts, color=colors)
            
            # Print total frames in the title. If 0, it confirms masking.
            ax.set_title(f"{task}\n({total:,} frames)", fontsize=10)
            ax.set_ylim(0, 100)
            ax.tick_params(axis='x', rotation=45, labelsize=9)
            
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            
            for bar, pct in zip(bars, pcts):
                if pct > 0:
                    height = bar.get_height()
                    ax.annotate(f'{pct:.1f}%',
                                xy=(bar.get_x() + bar.get_width() / 2, height),
                                xytext=(0, 3),  
                                textcoords="offset points",
                                ha='center', va='bottom', fontsize=8, fontweight='bold')
                    
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"SUCCESS! Graph saved to: {output_path}")
    else:
        print(f" Skipped {title} (No tasks found to graph)")


print("\n=== GENERATING GRAPHS ===")

generate_distribution_graph(
    all_stats["train_pre"], 
    "Class Distribution - TRAIN SPLIT (Pre-Synthetic)", 
    os.path.join(BASE_DIR, "class_dist_train_pre_synth.jpg")
)

generate_distribution_graph(
    all_stats["train_post"], 
    "Class Distribution - TRAIN SPLIT (Post-Synthetic)", 
    os.path.join(BASE_DIR, "class_dist_train_post_synth.jpg")
)

generate_distribution_graph(
    all_stats["val"], 
    "Class Distribution - VAL SPLIT", 
    os.path.join(BASE_DIR, "class_dist_val.jpg")
)
generate_distribution_graph(
    all_stats["val_post"], 
    "Class Distribution - VAL SPLIT (Post-Synthetic)", 
    os.path.join(BASE_DIR, "class_dist_val_post_synth.jpg")
)
generate_distribution_graph(
    all_stats["test"], 
    "Class Distribution - TEST SPLIT (Masked)", 
    os.path.join(BASE_DIR, "class_dist_test.jpg")
)