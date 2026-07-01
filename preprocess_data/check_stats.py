import os
import re
import json
import glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

BASE_DIR = r"C:\Users\Student3\Documents\xrtraceai\preprocess_data"
SPLITS    = ["train", "val", "test"]

CLASS_NAMES  = ["idle", "locomotion", "grasp_release", "assembly",
                "manipulation", "control_action", "transfer", "anomalous"]
CLASS_SHORT  = ["idle", "loco", "grsp", "asm", "manip", "ctrl", "trsf", "anom"]
CLASS_COLORS = ["#6B7280", "#3B82F6", "#10B981", "#F59E0B",
                "#8B5CF6", "#EC4899", "#14B8A6", "#EF4444"]

def empty_dist():
    return {c: 0 for c in CLASS_NAMES}

def strip_synth_prefix(sid):
    sid = re.sub(r'^LOCO_AUG_\d+_', '', sid)
    sid = re.sub(r'^SYNTH_(HYPER_SPEED|ERRATIC_TREMOR|AGGRESSIVE_REACH|SUDDEN_HEAD_JERK|CONTROLLER_DROP)_', '', sid)
    return sid

# ── Data collection ──────────────────────────────────────────────────────────

print("Scanning directories...")

all_stats = {
    "train_pre":  {},
    "train_post": {},
    "val":        {},
    "val_post":   {},
    "test":       {},
}

global_totals = {
    split: {**{c: 0 for c in CLASS_NAMES}, "synth_videos": 0, "total_videos": 0}
    for split in SPLITS
}

for split in SPLITS:
    split_dir = os.path.join(BASE_DIR, split)
    if not os.path.exists(split_dir):
        print(f"  [skip] {split} directory not found.")
        continue

    meta_files = glob.glob(os.path.join(split_dir, "*_meta.json"))
    global_totals[split]["total_videos"] = len(meta_files)

    for meta_file in meta_files:
        basename  = os.path.basename(meta_file)
        is_synth  = "SYNTH_" in basename or "LOCO_AUG_" in basename

        if is_synth:
            global_totals[split]["synth_videos"] += 1

        with open(meta_file) as f:
            meta = json.load(f)

        dist = meta.get("class_distribution", {})
        task = meta.get("task_type", "UNKNOWN")

        # Map synthetic files back to their source task
        if is_synth and task in ("SYNTHETIC_ANOMALY", "LOCOMOTION_AUG"):
            orig_sid       = strip_synth_prefix(meta.get("session_id", ""))
            orig_meta_path = os.path.join(split_dir, f"{orig_sid}_meta.json")
            if os.path.exists(orig_meta_path):
                with open(orig_meta_path) as f2:
                    task = json.load(f2).get("task_type", task)

        counts = {c: dist.get(c, 0) for c in CLASS_NAMES}

        for c in CLASS_NAMES:
            global_totals[split][c] += counts[c]

        if split == "train":
            for bucket in ("train_pre", "train_post"):
                if task not in all_stats[bucket]:
                    all_stats[bucket][task] = empty_dist()
            for c in CLASS_NAMES:
                all_stats["train_post"][task][c] += counts[c]
            if not is_synth:
                for c in CLASS_NAMES:
                    all_stats["train_pre"][task][c] += counts[c]

        elif split == "val":
            for bucket in ("val", "val_post"):
                if task not in all_stats[bucket]:
                    all_stats[bucket][task] = empty_dist()
            for c in CLASS_NAMES:
                all_stats["val_post"][task][c] += counts[c]
            if not is_synth:
                for c in CLASS_NAMES:
                    all_stats["val"][task][c] += counts[c]

        else:
            if task not in all_stats["test"]:
                all_stats["test"][task] = empty_dist()
            for c in CLASS_NAMES:
                all_stats["test"][task][c] += counts[c]

# ── Console summary ──────────────────────────────────────────────────────────

print("\n=== FINAL DATASET STATISTICS ===")
for split in SPLITS:
    totals = global_totals[split]
    if totals["total_videos"] == 0:
        continue
    total_frames = sum(totals[c] for c in CLASS_NAMES)
    print(f"\n--- {split.upper()} SPLIT ---")
    print(f"  Videos : {totals['total_videos']}  (Synthetic/Aug: {totals['synth_videos']})")
    if total_frames > 0:
        for c in CLASS_NAMES:
            pct = totals[c] / total_frames * 100
            print(f"  {c:<22}: {totals[c]:>10,}  ({pct:.1f}%)")
        print(f"  {'TOTAL':<22}: {total_frames:>10,}")
    else:
        print("  No labeled frames (test split is masked).")

# ── Graph generation ─────────────────────────────────────────────────────────

def generate_distribution_graph(task_stats, title, output_path):
    tasks = sorted(task_stats.keys())
    if not tasks:
        print(f"  Skipped '{title}' — no tasks found.")
        return

    num_tasks = len(tasks)
    fig_w     = max(5 * num_tasks, 10)
    fig, axes = plt.subplots(1, num_tasks, figsize=(fig_w, 5), sharey=True)
    if num_tasks == 1:
        axes = [axes]

    fig.suptitle(title, fontsize=13, fontweight='bold', y=1.01)

    x         = np.arange(len(CLASS_SHORT))
    bar_width = 0.65

    for i, task in enumerate(tasks):
        ax     = axes[i]
        stats  = task_stats[task]
        counts = [stats.get(c, 0) for c in CLASS_NAMES]
        total  = sum(counts)
        pcts   = [c / total * 100 if total > 0 else 0 for c in counts]

        bars = ax.bar(x, pcts, width=bar_width, color=CLASS_COLORS, edgecolor='white', linewidth=0.5)

        ax.set_title(f"{task}\n({total:,} frames)", fontsize=9, pad=4)
        ax.set_xticks(x)
        ax.set_xticklabels(CLASS_SHORT, rotation=45, ha='right', fontsize=8)
        ax.set_ylim(0, 108)
        ax.set_ylabel("% of frames" if i == 0 else "", fontsize=9)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.yaxis.grid(True, linestyle='--', alpha=0.4)
        ax.set_axisbelow(True)

        for bar, pct in zip(bars, pcts):
            if pct >= 1.0:
                ax.annotate(
                    f'{pct:.1f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=7, fontweight='bold'
                )

    legend_handles = [Patch(color=col, label=name)
                      for col, name in zip(CLASS_COLORS, CLASS_NAMES)]
    fig.legend(handles=legend_handles, loc='lower center', ncol=4,
               fontsize=8, bbox_to_anchor=(0.5, -0.08), frameon=False)

    plt.tight_layout(rect=[0, 0.08, 1, 1])
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def generate_overall_bar(split_label, totals, output_path):
    total_frames = sum(totals[c] for c in CLASS_NAMES)
    if total_frames == 0:
        print(f"  Skipped overall bar for {split_label} — no frames.")
        return

    pcts = [totals[c] / total_frames * 100 for c in CLASS_NAMES]

    fig, ax = plt.subplots(figsize=(10, 4))
    x   = np.arange(len(CLASS_NAMES))
    bars = ax.bar(x, pcts, color=CLASS_COLORS, edgecolor='white', linewidth=0.6)

    ax.set_title(f"Overall Class Distribution — {split_label}", fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(CLASS_NAMES, rotation=30, ha='right', fontsize=9)
    ax.set_ylabel("% of frames")
    ax.set_ylim(0, max(pcts) * 1.15)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.yaxis.grid(True, linestyle='--', alpha=0.4)
    ax.set_axisbelow(True)

    for bar, pct, c in zip(bars, pcts, CLASS_NAMES):
        ax.annotate(
            f'{pct:.1f}%\n({totals[c]:,})',
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 4), textcoords="offset points",
            ha='center', va='bottom', fontsize=8
        )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


print("\n=== GENERATING GRAPHS ===")

generate_distribution_graph(
    all_stats["train_pre"],
    "Class Distribution — TRAIN (Pre-Synthetic, by Task)",
    os.path.join(BASE_DIR, "class_dist_train_pre_synth.jpg")
)
generate_distribution_graph(
    all_stats["train_post"],
    "Class Distribution — TRAIN (Post-Synthetic, by Task)",
    os.path.join(BASE_DIR, "class_dist_train_post_synth.jpg")
)
generate_distribution_graph(
    all_stats["val"],
    "Class Distribution — VAL (Pre-Synthetic, by Task)",
    os.path.join(BASE_DIR, "class_dist_val.jpg")
)
generate_distribution_graph(
    all_stats["val_post"],
    "Class Distribution — VAL (Post-Synthetic, by Task)",
    os.path.join(BASE_DIR, "class_dist_val_post_synth.jpg")
)
generate_distribution_graph(
    all_stats["test"],
    "Class Distribution — TEST (Labels Masked)",
    os.path.join(BASE_DIR, "class_dist_test.jpg")
)

# Overall single-bar charts per split
generate_overall_bar(
    "TRAIN (Post-Synthetic)",
    global_totals["train"],
    os.path.join(BASE_DIR, "class_dist_train_overall.jpg")
)
generate_overall_bar(
    "VAL (Post-Synthetic)",
    global_totals["val"],
    os.path.join(BASE_DIR, "class_dist_val_overall.jpg")
)

print("\nDone.")
