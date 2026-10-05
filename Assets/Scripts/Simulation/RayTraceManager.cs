using UnityEngine;
using System.Collections.Generic;

[DefaultExecutionOrder(10000)]
public partial class RayTraceManager : MonoBehaviour
{
    public AudioManager audioManager;
    public int sampleRate = 48000;
    public Transform source, listener;
    public List<GameObject> obstacleObjects;

    void Start() { if (!useTakeRenderer) StartLegacy(); }
    void Update() { if (!useTakeRenderer) UpdateLegacy(); }
    void OnDestroy() { DisposeTakeSession(); DisposeLegacy(); }

    public static float[] LoadSample(AudioClip clip,int sampleRate)
    {
        float[] raw = new float[clip.samples * clip.channels];
        if (!clip.GetData(raw, 0))
            throw new System.InvalidOperationException($"Cannot read PCM from {clip.name}. Enable Decompress On Load and preload the clip before recording.");

        float[] mono = new float[clip.samples];
        for (int i = 0; i < clip.samples; i++)
        {
            float sum = 0;
            for (int c = 0; c < clip.channels; c++) sum += raw[i * clip.channels + c];
            mono[i] = sum / clip.channels;
        }

        if (clip.frequency == sampleRate) return mono;

        float ratio = (float)clip.frequency / sampleRate;
        int newLength = Mathf.RoundToInt(clip.samples / ratio);
        float[] resampled = new float[newLength];

        for (int i = 0; i < newLength; i++)
        {
            float srcIdx = i * ratio;
            int idx0 = Mathf.FloorToInt(srcIdx);
            int idx1 = Mathf.Min(idx0 + 1, mono.Length - 1);
            float t = srcIdx - idx0;
            resampled[i] = Mathf.Lerp(mono[idx0], mono[idx1], t);
        }

        return resampled;
    }

}
