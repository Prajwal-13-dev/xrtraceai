// XRTraceInference.cs
// Runs the XRTraceAI model (xrtrace_model.onnx) every 0.5 s on the BRV buffer
// and raises OnPrediction with the predicted class, confidence, and all probabilities.
//
// REQUIRES: Unity Sentis  (com.unity.sentis >= 1.3)
//   Install via Package Manager → Add package by name → com.unity.sentis
//
// SETUP:
//   1. Import xrtrace_model.onnx into your Unity project Assets folder.
//      Unity auto-imports it as a ModelAsset.
//   2. Copy brv_scaler_stats.json into Assets/StreamingAssets/
//   3. Attach this script to the same GameObject as BRVBuilder.
//   4. Drag the ModelAsset into the modelAsset slot in the Inspector.
//   5. Connect OnPrediction to your HUD script.

using System;
using UnityEngine;
using Unity.Sentis;

public class XRTraceInference : MonoBehaviour
{
    [Header("Model — drag xrtrace_model.onnx here")]
    public ModelAsset modelAsset;

    [Header("Inference settings")]
    [Tooltip("Run inference every N frames (15 = 0.5 s at 30 Hz)")]
    public int inferenceIntervalFrames = 15;

    [Tooltip("Probability threshold above which class 7 triggers anomaly alert")]
    [Range(0f, 1f)]
    public float anomalyThreshold = 0.5f;

    // ── Event: (classIndex, confidence, allProbabilities[8]) ─────────────────
    public event Action<int, float, float[]> OnPrediction;

    // ── Constants ─────────────────────────────────────────────────────────────
    const int NUM_CLASSES = 8;
    const int SEQ_LEN     = 60;
    const int BRV_DIM     = 26;
    const int ANOMALOUS   = 7;

    static readonly string[] CLASS_NAMES = {
        "idle", "locomotion", "grasp_release", "assembly",
        "manipulation", "control_action", "transfer", "anomalous"
    };

    // ── Internal state ─────────────────────────────────────────────────────────
    Model    _runtimeModel;
    IWorker  _worker;
    BRVBuilder _brv;
    int      _frameCounter = 0;
    bool     _modelReady   = false;

    // ── Lifecycle ──────────────────────────────────────────────────────────────
    void Start()
    {
        _brv = GetComponent<BRVBuilder>();
        if (_brv == null)
        {
            Debug.LogError("[XRTraceInference] BRVBuilder component not found on this GameObject.");
            return;
        }

        if (modelAsset == null)
        {
            Debug.LogError("[XRTraceInference] modelAsset is not assigned. Drag xrtrace_model.onnx into the Inspector.");
            return;
        }

        _runtimeModel = ModelLoader.Load(modelAsset);
        _worker       = WorkerFactory.CreateWorker(BackendType.GPUCompute, _runtimeModel);
        _modelReady   = true;

        Debug.Log("[XRTraceInference] Model loaded. Waiting for BRV buffer to fill (2 s)...");
    }

    void Update()
    {
        if (!_modelReady || !_brv.IsReady) return;

        _frameCounter++;
        if (_frameCounter % inferenceIntervalFrames != 0) return;

        RunInference();
    }

    void OnDestroy()
    {
        _worker?.Dispose();
    }

    // ── Inference ──────────────────────────────────────────────────────────────
    void RunInference()
    {
        float[] seq = _brv.GetNormalisedSequence();
        if (seq == null) return;

        // Build Sentis tensor: shape (1, SEQ_LEN, BRV_DIM)
        using var inputTensor = new TensorFloat(new TensorShape(1, SEQ_LEN, BRV_DIM), seq);

        _worker.Execute(inputTensor);

        // Output tensor: shape (1, NUM_CLASSES) — softmax probabilities
        var outputTensor = _worker.PeekOutput("class_probabilities") as TensorFloat;
        if (outputTensor == null)
        {
            Debug.LogError("[XRTraceInference] Could not read output tensor.");
            return;
        }

        outputTensor.MakeReadable();

        float[] probs    = new float[NUM_CLASSES];
        float   maxProb  = -1f;
        int     maxClass = 0;

        for (int c = 0; c < NUM_CLASSES; c++)
        {
            probs[c] = outputTensor[0, c];
            if (probs[c] > maxProb)
            {
                maxProb  = probs[c];
                maxClass = c;
            }
        }

        OnPrediction?.Invoke(maxClass, maxProb, probs);
        LogPrediction(maxClass, maxProb, probs);
    }

    void LogPrediction(int classIdx, float confidence, float[] probs)
    {
        string label  = CLASS_NAMES[classIdx];
        bool   isAnom = classIdx == ANOMALOUS && confidence >= anomalyThreshold;

        string probStr = "";
        for (int c = 0; c < NUM_CLASSES; c++)
            probStr += $"{CLASS_NAMES[c]}:{probs[c]:F2} ";

        if (isAnom)
            Debug.LogWarning($"[XRTraceAI] ANOMALY DETECTED  conf={confidence:F2}  | {probStr}");
        else
            Debug.Log($"[XRTraceAI] {label}  conf={confidence:F2}  | {probStr}");
    }

    // ── Public helpers ─────────────────────────────────────────────────────────

    /// <summary>
    /// Returns true if the last prediction was anomalous above threshold.
    /// Use this to drive visual alerts without subscribing to the event.
    /// </summary>
    public bool IsAnomalous(int classIdx, float confidence)
        => classIdx == ANOMALOUS && confidence >= anomalyThreshold;

    public static string GetClassName(int idx)
        => (idx >= 0 && idx < CLASS_NAMES.Length) ? CLASS_NAMES[idx] : "unknown";
}
