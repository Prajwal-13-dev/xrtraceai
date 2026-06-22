import os
import json
import glob
import matplotlib.pyplot as plt

TRAIN_DIR = r"C:\Users\Student3\Documents\xrtraceai\preprocess_data\train"
OUTPUT_GRAPH_PRE = r"C:\Users\Student3\Documents\xrtraceai\preprocess_data\class_dist_train.jpg"
OUTPUT_GRAPH_POST = r"C:\Users\Student3\Documents\xrtraceai\preprocess_data\class_dist_train_post_synth.jpg"

print("Scanning train directory for updated frame counts and generating graphs...")
meta_files = glob.glob(os.path.join(TRAIN_DIR, "*_meta.json"))

total_idle = 0
total_loco = 0
total_obj = 0
total_anom = 0
synth_videos = 0

# Dictionaries to hold separate stats for "Before" and "After"
pre_task_stats = {}
post_task_stats = {}

for meta_file in meta_files:
    is_synth = "SYNTH_" in meta_file
    if is_synth:
        synth_videos += 1
        
    with open(meta_file, "r") as f:
        meta = json.load(f)
        
    dist = meta.get("class_distribution", {})
    task = meta.get("task_type", "UNKNOWN")
    
    # --- FIX: Re-assign synthetic anomalies to their original tasks ---
    if is_synth and task == "SYNTHETIC_ANOMALY":
        orig_sid = meta.get("session_id", "")
        # Strip the prefix to get the original session ID
        for prefix in ["SYNTH_HYPER_SPEED_", "SYNTH_ERRATIC_TREMOR_", "SYNTH_AGGRESSIVE_REACH_"]:
            if orig_sid.startswith(prefix):
                orig_sid = orig_sid.replace(prefix, "")
                break
                
        # Look up the original file to find out what task it actually was
        orig_meta_path = os.path.join(TRAIN_DIR, f"{orig_sid}_meta.json")
        if os.path.exists(orig_meta_path):
            with open(orig_meta_path, "r") as orig_f:
                orig_meta = json.load(orig_f)
                task = orig_meta.get("task_type", task) # Fallback to current if missing
    # -----------------------------------------------------------------
    
    idle = dist.get("idle", 0)
    loco = dist.get("locomotion", 0)
    obj = dist.get("object_interaction", 0)
    anom = dist.get("anomalous", 0)
    
    # Global Totals (Post-Synthesis view for terminal output)
    total_idle += idle
    total_loco += loco
    total_obj += obj
    total_anom += anom
    
    # Initialize dictionary keys
    if task not in post_task_stats:
        post_task_stats[task] = {"idle": 0, "locomotion": 0, "object_interaction": 0, "anomalous": 0}
    if task not in pre_task_stats:
        pre_task_stats[task] = {"idle": 0, "locomotion": 0, "object_interaction": 0, "anomalous": 0}
        
    # 1. Add to POST-synth stats (All files)
    post_task_stats[task]["idle"] += idle
    post_task_stats[task]["locomotion"] += loco
    post_task_stats[task]["object_interaction"] += obj
    post_task_stats[task]["anomalous"] += anom
    
    # 2. Add to PRE-synth stats (Original files ONLY)
    if not is_synth:
        pre_task_stats[task]["idle"] += idle
        pre_task_stats[task]["locomotion"] += loco
        pre_task_stats[task]["object_interaction"] += obj
        pre_task_stats[task]["anomalous"] += anom

total_frames = total_idle + total_loco + total_obj + total_anom

print("\n=== FINAL TRAINING SET STATISTICS ===")
print(f"Total Videos (Including Synthetic): {len(meta_files)}")
print(f"Synthetic Videos Generated: {synth_videos}")
print("-" * 40)
print(f"Idle Frames:           {total_idle:,} ({total_idle/total_frames*100:.1f}%)")
print(f"Locomotion Frames:     {total_loco:,} ({total_loco/total_frames*100:.1f}%)")
print(f"Object Interaction:    {total_obj:,} ({total_obj/total_frames*100:.1f}%)")
print(f"ANOMALOUS FRAMES:      {total_anom:,} ({total_anom/total_frames*100:.1f}%)")
print("=====================================\n")

# ============================================================================
# GRAPH GENERATION HELPER FUNCTION
# ============================================================================
def generate_distribution_graph(task_stats, title, output_path):
    tasks = sorted(list(task_stats.keys()))
    # Filter out empty tasks just in case
    tasks = [t for t in tasks if sum(task_stats[t].values()) > 0]
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

# Generate both graphs
print("\nDrawing PRE-synthesis task distribution graph...")
generate_distribution_graph(pre_task_stats, "Class Distribution - TRAIN SPLIT (Pre-Synthetic)", OUTPUT_GRAPH_PRE)

print("Drawing POST-synthesis task distribution graph...")
generate_distribution_graph(post_task_stats, "Class Distribution - TRAIN SPLIT (Post-Synthetic)", OUTPUT_GRAPH_POST)