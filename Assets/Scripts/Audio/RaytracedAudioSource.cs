using UnityEngine;

public class RaytracedAudioSource : MonoBehaviour
{
    public RayTraceManager rayTraceManager;
    public AudioClip clip;
    public bool streamed;
    [Tooltip("Zero monitors every receiver; 1 excludes receiver zero (local self voice). Fixed for a capture.")]
    public int mutedListenerPlusOne;
    [System.NonSerialized] public SourceStream Stream;
    [Tooltip("Applied on the next Play call.")]
    public bool loop = true;
    [Tooltip("Input gain on Play; use SetGain for recorded live changes.")]
    [Range(0, 2)] public float gain = 1;
    public bool disableUnityAudioSources = true;

    void OnDrawGizmos()
    {
        if (!isActiveAndEnabled) return;
        Gizmos.color = streamed ? new Color(.4f, .9f, 1) : new Color(1, .8f, .4f);
        Gizmos.DrawWireSphere(transform.position, .2f);
#if UNITY_EDITOR
        UnityEditor.Handles.Label(transform.position + Vector3.up * .3f, name);
#endif
    }

    void Awake() => Apply();
    void OnValidate()
    {
        if (rayTraceManager == null) rayTraceManager = FindAnyObjectByType<RayTraceManager>();
    }

    public void Apply()
    {
        OnValidate();
        if (disableUnityAudioSources)
            foreach (var source in GetComponentsInChildren<AudioSource>()) source.enabled = false;
    }

    [ContextMenu("Play Raytraced Audio")]
    public void Play()
    {
        Apply();
        if (rayTraceManager == null) return;
        if (!rayTraceManager.IsLiveSimulation && !rayTraceManager.IsRecordingTake) rayTraceManager.BeginLiveSimulation();
        rayTraceManager.RecordSourceEvent(transform, AcousticTake.VoiceAction.Play, gain, !streamed && loop);
    }

    public void SetGain(float value)
    {
        if (!float.IsFinite(value) || value < 0 || value > 2) throw new System.ArgumentOutOfRangeException(nameof(value));
        if (rayTraceManager != null) rayTraceManager.RecordSourceEvent(transform, AcousticTake.VoiceAction.Gain, value);
        gain = value;
    }

    [ContextMenu("Stop Raytraced Audio")]
    public void Stop() => rayTraceManager?.RecordSourceEvent(transform, AcousticTake.VoiceAction.Stop);
}
