# -------------------------
# CLASSIFIERS
# -------------------------
def classify_velocity(v):
    if v > 1.5:
        return "high velocity"
    elif v > 0.7:
        return "moderate velocity"
    else:
        return "low velocity"


def classify_suspicion(score):
    if score > 2.0:
        return "highly suspicious"
    elif score > 1.0:
        return "moderately suspicious"
    else:
        return "normal"


def format_time(frame, fps=30):
    seconds = frame / fps
    mins = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{mins:02d}:{secs:02d}"


# -------------------------
# ACTIVITY-AWARE RULES
# -------------------------
def activity_context_rules(f):
    label = f["label"]

    if label == "idle":
        if f["peak_velocity"] > 0.7:
            return "unexpected motion during idle state"

    elif label == "locomotion":
        if f["dominant_hand"] != "balanced":
            return "asymmetric movement during locomotion"

    elif label == "grasp_release":
        if f["dominance_ratio"] > 0.8:
            return "one-handed interaction dominance detected"

    elif label == "assembly":
        if f["peak_velocity"] > 1.5:
            return "unusually fast assembly interaction"

    elif label == "manipulation":
        if f["velocity_std"] > 0.8:
            return "unstable manipulation behavior"

    elif label == "control_action":
        if f["movement_intensity"] > 1.5:
            return "forceful control interaction detected"

    elif label == "transfer":
        if f["dominant_hand"] != "balanced":
            return "imbalanced object transfer behavior"

    elif label == "anomalous":
        return "explicit anomaly detected by model"

    return None


# -------------------------
# REASONING ENGINE
# -------------------------
def generate_reasoning(f):
    reasons = []

    if f["peak_velocity"] > 1.5:
        reasons.append("abrupt high-speed motion")

    if f["velocity_std"] > 0.8:
        reasons.append("irregular motion pattern")

    if f["dominance_ratio"] > 0.8:
        reasons.append(f"strong {f['dominant_hand']}-hand dominance")

    if f["movement_intensity"] > 1.5:
        reasons.append("elevated movement intensity")

    context_reason = activity_context_rules(f)
    if context_reason:
        reasons.append(context_reason)

    if not reasons:
        return "movement pattern within expected behavioral range"

    return ", ".join(reasons)


# -------------------------
# DETAILED REPORT
# -------------------------
def generate_detailed_sentence(feature, start_frame):
    velocity_desc = classify_velocity(feature["peak_velocity"])
    suspicion_level = classify_suspicion(feature["suspicion_score"])
    timestamp = format_time(start_frame)

    reasoning = generate_reasoning(feature)

    return (
        f"Event ID: {feature['id']}: During {feature['label']} activity, "
        f"{velocity_desc} with {feature['dominant_hand']} hand dominance was observed. "
        f"The behavior is classified as {suspicion_level}. "
        f"Analysis indicates {reasoning}. "
        f"Timestamp: {timestamp}, Source: HoloLens2."
    )


# -------------------------
# COMPACT INSIGHT
# -------------------------
def generate_compact_insight(f):
    parts = []

    if f["peak_velocity"] > 1.5:
        parts.append("abnormally high")
    elif f["peak_velocity"] > 0.7:
        parts.append("moderate")
    else:
        parts.append("low")

    if f["dominant_hand"] != "balanced":
        parts.append(f"{f['dominant_hand']}-hand")

    parts.append("velocity")

    activity = f["label"]

    if f["label"] == "anomalous":
        conclusion = "confirms anomalous behavior detected in XR interaction"
    elif f["suspicion_score"] > 2:
        conclusion = "suggests irregular interaction behavior"
    elif f["suspicion_score"] > 1:
        conclusion = "indicates potentially unusual behavior"
    else:
        conclusion = "remains within expected behavior patterns"

    return f"{' '.join(parts).capitalize()} during {activity} {conclusion}."


# -------------------------
# FINAL REPORT GENERATOR
# -------------------------
def generate_report(features, segments):
    report = []

    for f, seg in zip(features, segments):
        detailed = generate_detailed_sentence(f, seg["start"])
        compact = generate_compact_insight(f)

        report.append({
            "event_id": f["id"],
            "detailed_report": detailed,
            "forensic_insight": compact
        })

    return report
