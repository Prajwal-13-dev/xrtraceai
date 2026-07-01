"""
Export the trained XRTraceAI model to ONNX for Unity Sentis/Barracuda inference.

Run this after training completes:
    cd models
    python export_onnx.py

Outputs:
    <SAVE_DIR>/xrtrace_model.onnx
    <SAVE_DIR>/export_summary.txt
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model_base import XRAnomalyDetector

# ── Config ────────────────────────────────────────────────────────────────────
# Point this at whichever checkpoint you want to ship
SAVE_DIR   = r"C:\Users\Student3\Documents\xrtraceai\models\train_base_8class_v4"
CHECKPOINT = os.path.join(SAVE_DIR, "best_model.pth")
ONNX_PATH  = os.path.join(SAVE_DIR, "xrtrace_model.onnx")

SEQ_LEN  = 60   # frames in sliding window (2 s at 30 Hz)
BRV_DIM  = 26   # feature dimension
OPSET    = 12   # opset 12 for better LSTM support in Sentis

CLASS_NAMES = [
    "idle", "locomotion", "grasp_release", "assembly",
    "manipulation", "control_action", "transfer", "anomalous"
]

# ── Wrapper: return only softmax probabilities ────────────────────────────────
# Stripping the attention output simplifies the ONNX graph for Barracuda/Sentis.
class XRTraceExportWrapper(nn.Module):
    def __init__(self, base_model: XRAnomalyDetector):
        super().__init__()
        self.model = base_model

    def forward(self, x):
        logits, _ = self.model(x)
        return torch.softmax(logits, dim=-1)   # (batch, 8) — probabilities


# ── Load checkpoint ───────────────────────────────────────────────────────────
if not os.path.exists(CHECKPOINT):
    sys.exit(
        f"ERROR: checkpoint not found at {CHECKPOINT}\n"
        "Train the model first (python models/train_base.py) then re-run this script."
    )

print(f"Loading checkpoint: {CHECKPOINT}")
base_model = XRAnomalyDetector(input_dim=BRV_DIM, num_classes=len(CLASS_NAMES))
ckpt = torch.load(CHECKPOINT, map_location="cpu")
state_dict = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
base_model.load_state_dict(state_dict)
base_model.eval()

export_model = XRTraceExportWrapper(base_model)
export_model.eval()

# ── Verify with a forward pass ────────────────────────────────────────────────
dummy = torch.randn(1, SEQ_LEN, BRV_DIM)
with torch.no_grad():
    probs = export_model(dummy)

assert probs.shape == (1, len(CLASS_NAMES)), f"Unexpected output shape: {probs.shape}"
assert abs(probs.sum().item() - 1.0) < 1e-4, "Probabilities do not sum to 1"
print(f"Forward pass OK — output shape: {probs.shape}, sum={probs.sum().item():.4f}")

# ── Export to ONNX ────────────────────────────────────────────────────────────
print(f"\nExporting to: {ONNX_PATH}")
torch.onnx.export(
    export_model,
    (dummy,),
    ONNX_PATH,
    export_params=True,
    opset_version=OPSET,
    do_constant_folding=True,
    input_names=["brv_sequence"],        # (batch, seq_len, 26)
    output_names=["class_probabilities"],# (batch, 8)
    dynamic_axes={
        "brv_sequence":        {0: "batch"},
        "class_probabilities": {0: "batch"},
    },
    verbose=False,
)
print("ONNX export done.")

# ── Validate with onnxruntime (optional but recommended) ──────────────────────
try:
    import onnxruntime as ort
    sess = ort.InferenceSession(ONNX_PATH, providers=["CPUExecutionProvider"])
    ort_out = sess.run(None, {"brv_sequence": dummy.numpy()})[0]
    max_diff = np.abs(ort_out - probs.detach().numpy()).max()
    print(f"OnnxRuntime validation: max abs diff vs PyTorch = {max_diff:.2e}  (expect < 1e-5)")
    if max_diff > 1e-4:
        print("  WARNING: large numerical difference — check ONNX export carefully.")
    else:
        print("  OK — ONNX output matches PyTorch output.")
except ImportError:
    print("onnxruntime not installed — skipping numerical validation.")
    print("  Install with: pip install onnxruntime")

# ── Write export summary ──────────────────────────────────────────────────────
summary = {
    "checkpoint":     CHECKPOINT,
    "onnx_path":      ONNX_PATH,
    "opset_version":  OPSET,
    "input_name":     "brv_sequence",
    "input_shape":    [1, SEQ_LEN, BRV_DIM],
    "output_name":    "class_probabilities",
    "output_shape":   [1, len(CLASS_NAMES)],
    "class_names":    CLASS_NAMES,
    "class_indices":  {name: i for i, name in enumerate(CLASS_NAMES)},
    "anomalous_class_index": 7,
    "notes": [
        "Input must be z-score normalised using brv_scaler_stats.json before inference.",
        "Sliding window: 60 frames (2 s at 30 Hz). Run inference every 15 frames (0.5 s).",
        "BRV layout: [0:3]=head_pos, [3:7]=head_quat, [7:10]=left_rel, [10:13]=right_rel, [13:26]=velocities.",
        "Use Unity Sentis (com.unity.sentis) >= 1.3 for best ONNX compatibility.",
        "Barracuda is deprecated — prefer Sentis.",
    ]
}

summary_path = os.path.join(SAVE_DIR, "export_summary.json")
with open(summary_path, "w") as f:
    json.dump(summary, f, indent=2)

print(f"\nExport summary: {summary_path}")
print("\n=== FILES TO SEND TO UNITY TEAMMATE ===")
print(f"  1. {ONNX_PATH}")
print(f"  2. <preprocess_data>/brv_scaler_stats.json")
print(f"  3. unity/XRTraceInference.cs")
print(f"  4. unity/BRVBuilder.cs")
print(f"\nAnomaly class index = 7  →  class_probabilities[7] > threshold triggers alert")
