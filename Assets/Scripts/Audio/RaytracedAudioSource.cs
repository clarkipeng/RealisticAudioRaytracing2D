using UnityEngine;

public class RaytracedAudioSource : MonoBehaviour
{
    public RayTraceManager rayTraceManager;
    public AudioClip clip;
    [Tooltip("Applied on the next Play call.")]
    public bool loop = true;
    [Tooltip("Input gain on Play; use SetGain for recorded live changes.")]
    [Range(0,2)] public float gain = 1;
    public bool prepareOnStart = true;
    public bool assignSourceTransform = true;
    public bool assignClip = true;
    public bool disableUnityAudioSources = true;

    void Awake()
    {
        Apply();
    }

    void Start()
    {
        if (prepareOnStart)
            Prepare();
    }

    void OnValidate()
    {
        if (rayTraceManager == null)
            rayTraceManager = FindObjectOfType<RayTraceManager>();
    }

    [ContextMenu("Apply To RayTraceManager")]
    public void Apply()
    {
        if (rayTraceManager == null)
            rayTraceManager = FindObjectOfType<RayTraceManager>();

        if (rayTraceManager == null) return;

        if (disableUnityAudioSources)
        {
            foreach (AudioSource audioSource in GetComponentsInChildren<AudioSource>())
                audioSource.enabled = false;
        }
        if (rayTraceManager.useTakeRenderer) return;

        // Existing scenes keep their selected legacy emitter. Register additional
        // emitters on the take instead of having Awake order choose a global source.
        if (rayTraceManager.source != null && rayTraceManager.source != transform) return;
        if (assignSourceTransform)
            rayTraceManager.source = transform;

        if (assignClip && clip != null)
            rayTraceManager.inputClip = clip;

        rayTraceManager.loopAudio = loop;

    }

    [ContextMenu("Prepare Raytraced Audio")]
    public void Prepare()
    {
        Apply();
        if (rayTraceManager != null && rayTraceManager.source == transform)
            rayTraceManager.PrepareCurrentClip();
    }

    [ContextMenu("Play Raytraced Audio")]
    public void Play()
    {
        Apply();
        if (rayTraceManager != null && rayTraceManager.useTakeRenderer)
        {
            if (!rayTraceManager.IsLiveSimulation) rayTraceManager.BeginLiveSimulation();
            rayTraceManager.RecordSourceEvent(transform,AcousticTake.VoiceAction.Play,gain,loop);
        }
        else if (rayTraceManager != null && rayTraceManager.source == transform)
            rayTraceManager.StartStreaming();
        else if (rayTraceManager != null)
        {
            rayTraceManager.RecordSourceEvent(transform, AcousticTake.VoiceAction.Play, gain, loop);
            Debug.LogWarning("This source's event is recorded, but the legacy realtime renderer only plays its selected emitter. Offline direct rendering supports all recorded sources.", this);
        }
    }

    public void SetGain(float value)
    {
        if(!float.IsFinite(value) || value<0 || value>2) throw new System.ArgumentOutOfRangeException(nameof(value));
        if(rayTraceManager!=null) rayTraceManager.RecordSourceEvent(transform,AcousticTake.VoiceAction.Gain,value);
        gain=value;
    }

    [ContextMenu("Stop Raytraced Audio")]
    public void Stop()
    {
        if (rayTraceManager != null && rayTraceManager.useTakeRenderer)
            rayTraceManager.RecordSourceEvent(transform,AcousticTake.VoiceAction.Stop);
        else if (rayTraceManager != null && rayTraceManager.source == transform)
            rayTraceManager.StopStreaming();
        else if (rayTraceManager != null)
            rayTraceManager.RecordSourceEvent(transform, AcousticTake.VoiceAction.Stop);
    }
}
