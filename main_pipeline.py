import numpy as np
import pandas as pd
import os

# ✅ Import Phase 6 & 7
from forensic_analysis.forensic_feature_extractor import process_all_segments
from forensic_analysis.forensic_report_generator import generate_report


# -------------------------
# CONFIG
# -------------------------
FPS = 30

COLUMNS = [
    'Head_Y','Head_Qx','Head_Qy','Head_Qz','Head_Qw',
    'L_PosX','L_PosY','L_PosZ','R_PosX','R_PosY','R_PosZ',
    'Head_Y_Vel','Head_Qx_Vel','Head_Qy_Vel','Head_Qz_Vel','Head_Qw_Vel',
    'L_VelX','L_VelY','L_VelZ','R_VelX','R_VelY','R_VelZ'
]


# -------------------------
# LOAD DATA
# -------------------------
def load_data(npy_path):
    if not os.path.exists(npy_path):
        raise FileNotFoundError(f"❌ File not found: {npy_path}")

    data = np.load(npy_path)

    if data.shape[1] != len(COLUMNS):
        print(f"⚠️ Warning: Expected {len(COLUMNS)} features, got {data.shape[1]}")

    df = pd.DataFrame(data[:, :len(COLUMNS)], columns=COLUMNS)
    return df


# -------------------------
# PHASE 5 (TEMP / DUMMY)
# Replace this when model is ready
# -------------------------
def get_segments(n_frames):
    return [
        {"id": 1, "label": 0, "start": 0, "end": int(n_frames*0.12)},      
        {"id": 2, "label": 1, "start": int(n_frames*0.12)+1, "end": int(n_frames*0.24)},  
        {"id": 3, "label": 2, "start": int(n_frames*0.24)+1, "end": int(n_frames*0.36)},  
        {"id": 4, "label": 3, "start": int(n_frames*0.36)+1, "end": int(n_frames*0.50)},  
        {"id": 5, "label": 4, "start": int(n_frames*0.50)+1, "end": int(n_frames*0.65)},  
        {"id": 6, "label": 5, "start": int(n_frames*0.65)+1, "end": int(n_frames*0.78)},  
        {"id": 7, "label": 6, "start": int(n_frames*0.78)+1, "end": int(n_frames*0.88)},  
        {"id": 8, "label": 7, "start": int(n_frames*0.88)+1, "end": n_frames-1},          
    ]


# -------------------------
# MAIN PIPELINE
# -------------------------
def main():

    # 🔥 CHANGE THIS if needed
    npy_file = "preprocess_data/R073-20July-GoPro_brv.npy"

    print("\n" + "="*60)
    print("🚀 XR FORENSIC ANALYSIS PIPELINE")
    print("="*60)

    # -------------------------
    # STEP 1: LOAD DATA
    # -------------------------
    print("\n📂 Loading data...")
    df = load_data(npy_file)
    print(f"✅ Data loaded successfully")
    print(f"📊 Shape: {df.shape}")

    # -------------------------
    # STEP 2: SEGMENTATION (Phase 5)
    # -------------------------
    print("\n🧠 Generating segments (Phase 5)...")
    segments = get_segments(len(df))
    print(f"✅ {len(segments)} segments created")

    # -------------------------
    # STEP 3: FEATURE EXTRACTION (Phase 6)
    # -------------------------
    print("\n⚙️ Extracting kinematic features (Phase 6)...")
    features = process_all_segments(segments, df)
    print("✅ Feature extraction complete")

    # -------------------------
    # STEP 4: REPORT GENERATION (Phase 7)
    # -------------------------
    print("\n📝 Generating forensic report (Phase 7)...")
    report = generate_report(features, segments)
    print("✅ Report generation complete")

    # -------------------------
    # STEP 5: DISPLAY OUTPUT
    # -------------------------
    print("\n" + "="*60)
    print("🔍 FINAL FORENSIC REPORT")
    print("="*60)

    for r in report:
        print(f"\n🆔 Event {r['event_id']}")
        print("— Detailed:")
        print(r["detailed_report"])
        print("\n— Insight:")
        print(r["forensic_insight"])
        print("-"*60)


# -------------------------
# RUN
# -------------------------
if __name__ == "__main__":
    main()
