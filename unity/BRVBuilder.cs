// BRVBuilder.cs
// Assembles the 26-feature Behavioural Representation Vector (BRV) from
// HoloLens 2 head and hand tracking data every frame, then returns a
// normalised (1, 60, 26) tensor ready for XRTraceInference.
//
// Feature layout (matches Python extract_data.py):
//   [0:3]   head_pos       (world-space x, y, z)
//   [3:7]   head_quat      (qx, qy, qz, qw)
//   [7:10]  left_hand_rel  (left wrist position relative to head, x, y, z)
//   [10:13] right_hand_rel (right wrist relative to head, x, y, z)
//   [13:26] velocities     (finite-difference × 30 Hz, same order as above)
//
// SETUP:
//   1. Attach this script to a persistent GameObject (e.g. XRTraceManager).
//   2. Set headTransform to Camera.main.transform (or the HMD anchor).
//   3. Call SetHandPositions(leftWristWorld, rightWristWorld) every FixedUpdate
//      from your MRTK / OpenXR hand-tracking component.
//   4. Call GetNormalisedSequence() from XRTraceInference when you need to run the model.

using System;
using System.IO;
using UnityEngine;
using Newtonsoft.Json;   // Unity package: com.unity.nuget.newtonsoft-json

public class BRVBuilder : MonoBehaviour
{
    [Header("Head Tracking")]
    public Transform headTransform;   // assign Camera.main.transform in Inspector

    [Header("Normaliser — place brv_scaler_stats.json in StreamingAssets")]
    public string scalerFileName = "brv_scaler_stats.json";

    // ── Constants ─────────────────────────────────────────────────────────────
    const int BRV_DIM   = 26;
    const int SEQ_LEN   = 60;          // 2 s at 30 Hz
    const float FRAME_RATE = 30f;

    // ── Internal state ─────────────────────────────────────────────────────────
    float[][] _buffer;                 // circular buffer [SEQ_LEN][BRV_DIM]
    int       _writeIdx  = 0;
    int       _fillCount = 0;

    float[]   _prevSpatial;            // previous frame's spatial[13] for velocity diff
    float[]   _scalerMean;
    float[]   _scalerStd;
    bool      _scalerLoaded = false;

    Vector3   _leftWristWorld  = Vector3.zero;
    Vector3   _rightWristWorld = Vector3.zero;
    bool      _handsValid      = false;

    // ── Unity lifecycle ────────────────────────────────────────────────────────
    void Awake()
    {
        _buffer      = new float[SEQ_LEN][];
        for (int i = 0; i < SEQ_LEN; i++)
            _buffer[i] = new float[BRV_DIM];

        _prevSpatial = new float[13];

        LoadScaler();
    }

    void FixedUpdate()
    {
        if (!_handsValid || headTransform == null) return;
        AppendFrame();
    }

    // ── Public API ─────────────────────────────────────────────────────────────

    /// <summary>
    /// Call this every frame from your hand-tracking component.
    /// Positions should be in Unity world space.
    /// </summary>
    public void SetHandPositions(Vector3 leftWrist, Vector3 rightWrist)
    {
        _leftWristWorld  = leftWrist;
        _rightWristWorld = rightWrist;
        _handsValid      = true;
    }

    /// <summary>
    /// Returns true once the buffer has been filled with at least SEQ_LEN frames.
    /// </summary>
    public bool IsReady => _fillCount >= SEQ_LEN;

    /// <summary>
    /// Returns the normalised (1, SEQ_LEN, BRV_DIM) float array for inference.
    /// Returns null if the buffer is not yet full.
    /// </summary>
    public float[] GetNormalisedSequence()
    {
        if (!IsReady) return null;

        float[] seq = new float[SEQ_LEN * BRV_DIM];
        for (int t = 0; t < SEQ_LEN; t++)
        {
            // Read from circular buffer in chronological order
            int srcIdx = (_writeIdx - SEQ_LEN + t + SEQ_LEN) % SEQ_LEN;
            float[] frame = _buffer[srcIdx];

            for (int f = 0; f < BRV_DIM; f++)
            {
                float val = _scalerLoaded
                    ? (frame[f] - _scalerMean[f]) / _scalerStd[f]
                    : frame[f];

                seq[t * BRV_DIM + f] = val;
            }
        }
        return seq;
    }

    // ── Internal helpers ───────────────────────────────────────────────────────

    void AppendFrame()
    {
        float[] spatial = BuildSpatial();
        float[] vel     = ComputeVelocity(spatial);

        float[] frame = _buffer[_writeIdx];

        // Copy spatial [0:13]
        Array.Copy(spatial, 0, frame, 0, 13);
        // Copy velocity [13:26]
        Array.Copy(vel, 0, frame, 13, 13);

        Array.Copy(spatial, _prevSpatial, 13);

        _writeIdx = (_writeIdx + 1) % SEQ_LEN;
        if (_fillCount < SEQ_LEN) _fillCount++;
    }

    float[] BuildSpatial()
    {
        float[] s = new float[13];

        // Head position [0:3]
        Vector3 hp = headTransform.position;
        s[0] = hp.x; s[1] = hp.y; s[2] = hp.z;

        // Head quaternion [3:7]  (qx, qy, qz, qw)
        Quaternion hq = headTransform.rotation;
        s[3] = hq.x; s[4] = hq.y; s[5] = hq.z; s[6] = hq.w;

        // Left hand relative to head [7:10]
        Vector3 leftRel = _leftWristWorld - hp;
        s[7] = leftRel.x; s[8] = leftRel.y; s[9] = leftRel.z;

        // Right hand relative to head [10:13]
        Vector3 rightRel = _rightWristWorld - hp;
        s[10] = rightRel.x; s[11] = rightRel.y; s[12] = rightRel.z;

        return s;
    }

    float[] ComputeVelocity(float[] current)
    {
        float[] vel = new float[13];
        for (int i = 0; i < 13; i++)
            vel[i] = (current[i] - _prevSpatial[i]) * FRAME_RATE;
        return vel;
    }

    void LoadScaler()
    {
        string path = Path.Combine(Application.streamingAssetsPath, scalerFileName);
        if (!File.Exists(path))
        {
            Debug.LogWarning($"[BRVBuilder] Scaler not found at {path}. Running unnormalised.");
            return;
        }

        try
        {
            string json = File.ReadAllText(path);
            var data    = JsonConvert.DeserializeObject<ScalerData>(json);

            _scalerMean   = data.mean;
            _scalerStd    = data.std;
            _scalerLoaded = true;

            Debug.Log($"[BRVBuilder] Scaler loaded — {_scalerMean.Length} features.");
        }
        catch (Exception e)
        {
            Debug.LogError($"[BRVBuilder] Failed to load scaler: {e.Message}");
        }
    }

    [Serializable]
    class ScalerData
    {
        public float[] mean;
        public float[] std;
    }
}
