import numpy as np
import os

# Set this to the video_name that produced the 32 m/s spike
VIDEO_NAME = "R005-7July-GoPro" 
TRAIN_DIR = r"C:\Users\Student3\Documents\xrtraceai\preprocess_data\train"

brv = np.load(os.path.join(TRAIN_DIR, f"{VIDEO_NAME}_brv.npy"))
# Assuming your layout is: 0:head_y, 1-4:quat_x,y,z,w
# Adjust indices based on your ACTUAL layout if different!
quats = brv[:, 1:5] 
left_vel = np.linalg.norm(brv[:, 16:19], axis=1)

# Find frames where velocity > 10 m/s
spike_indices = np.where(left_vel > 10)[0]

print(f"Analyzing {len(spike_indices)} spikes in {VIDEO_NAME}...")
for idx in spike_indices:
    if idx > 0:
        print(f"\nSpike at Frame {idx}, Velocity: {left_vel[idx]:.2f} m/s")
        print(f"  Frame {idx-1} Quat: {quats[idx-1]}")
        print(f"  Frame {idx}   Quat: {quats[idx]}")
        # Check if Frame idx Quat is roughly the negative of Frame idx-1
        diff = np.abs(quats[idx] - quats[idx-1])
        if np.all(diff > 0.5): # Significant change
            print("  [!] Significant orientation jump detected.")