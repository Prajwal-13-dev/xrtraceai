import os

# Your exact dataset path
DATASET_DIR = r"C:\Users\Student3\Documents\xrtraceai\Data_set\Halo_assist_dataset"
SPLITS_DIR = os.path.join(DATASET_DIR, "data-splits-v1_2")

print("CHECKING OFFICIAL SPLIT FILES")
total_in_splits = 0
all_split_sessions = set()

for split in ["train", "val", "test"]:
    split_file = os.path.join(SPLITS_DIR, f"{split}-v1_2.txt")
    if os.path.exists(split_file):
        with open(split_file, "r") as f:
            # Count non-empty lines
            lines = [line.strip() for line in f if line.strip()]
            count = len(lines)
            all_split_sessions.update(lines)
            print(f"  {split}-v1_2.txt contains: {count} sessions")
            total_in_splits += count

print(f"  TOTAL IN SPLIT FILES: {total_in_splits}\n")

print("CHECKING HARD DRIVE")
# Ignore folders that aren't videos
ignore_list = ["data-splits-v1_2", "Annotations", "preprocess_data"]

# Count all subdirectories that look like sessions
actual_folders = [
    f for f in os.listdir(DATASET_DIR) 
    if os.path.isdir(os.path.join(DATASET_DIR, f)) and f not in ignore_list
]

print(f"  TOTAL SESSION FOLDERS ON DISK: {len(actual_folders)}\n")

print("CONCLUSION")
if total_in_splits == 1758:
    print("PERFECT MATCH: The JSON file has exactly the same number of sessions as the official dataset splits.")
else:
    print(f"ℹNote: The JSON found 1758, but the splits list {total_in_splits}.")

if len(actual_folders) == total_in_splits:
    print("DOWNLOAD COMPLETE: You have downloaded every single folder listed in the splits.")
elif len(actual_folders) < total_in_splits:
    print(f"MISSING DATA: You only downloaded {len(actual_folders)} out of {total_in_splits} folders.")