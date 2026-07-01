import numpy as np

FPS = 30

LABEL_MAP = {
    0: "idle",
    1: "walking",
    2: "grasp_release",
    3: "assembly",
    4: "manipulation",
    5: "control_action",
    6: "other",
    7: "anomalous"
}


# -------------------------
# CORE COMPUTATIONS
# -------------------------
def compute_displacement(x, y, z):
    dx = np.diff(x)
    dy = np.diff(y)
    dz = np.diff(z)
    return np.sum(np.sqrt(dx**2 + dy**2 + dz**2))


def velocity_magnitude(vx, vy, vz):
    return np.sqrt(vx**2 + vy**2 + vz**2)


# -------------------------
# FEATURE EXTRACTION
# -------------------------
def extract_segment_features(segment, df):
    start, end = segment["start"], segment["end"]
    label = segment["label"]

    frames = df.iloc[start:end+1]

    if len(frames) < 2:
        return None

    # Duration
    duration = (end - start) / FPS

    # Displacement
    left_disp = compute_displacement(frames['L_PosX'], frames['L_PosY'], frames['L_PosZ'])
    right_disp = compute_displacement(frames['R_PosX'], frames['R_PosY'], frames['R_PosZ'])
    total_disp = float(left_disp + right_disp)

    # Velocity
    left_vel = velocity_magnitude(frames['L_VelX'], frames['L_VelY'], frames['L_VelZ'])
    right_vel = velocity_magnitude(frames['R_VelX'], frames['R_VelY'], frames['R_VelZ'])

    combined_vel = np.concatenate([left_vel.values, right_vel.values])

    mean_velocity = float(np.mean(combined_vel))
    peak_velocity = float(np.max(combined_vel))
    velocity_std = float(np.std(combined_vel))

    # Movement Intensity
    intensity = total_disp / (duration + 1e-5)

    # Dominant Hand
    if abs(left_disp - right_disp) < 1e-6:
        dominant_hand = "balanced"
        dominance_ratio = 0.5
    else:
        dominant_hand = "left" if left_disp > right_disp else "right"
        dominance_ratio = max(left_disp, right_disp) / (total_disp + 1e-5)

    # Anomaly
    anomaly_flag = 1 if label == 7 else 0

    # Suspicion Score
    suspicion_score = (
        0.4 * peak_velocity +
        0.3 * velocity_std +
        0.3 * intensity
    )

    return {
        "id": segment["id"],
        "label": LABEL_MAP.get(label, str(label)),
        "duration": round(duration, 2),
        "displacement": round(total_disp, 4),
        "mean_velocity": round(mean_velocity, 4),
        "peak_velocity": round(peak_velocity, 4),
        "velocity_std": round(velocity_std, 4),
        "movement_intensity": round(intensity, 4),
        "dominant_hand": dominant_hand,
        "dominance_ratio": round(dominance_ratio, 3),
        "anomaly": anomaly_flag,
        "suspicion_score": round(suspicion_score, 4)
    }


def process_all_segments(segments, df):
    results = []
    for seg in segments:
        f = extract_segment_features(seg, df)
        if f:
            results.append(f)
    return results
