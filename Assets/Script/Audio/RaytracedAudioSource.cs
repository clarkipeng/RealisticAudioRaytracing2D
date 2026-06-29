using UnityEngine;

public class RaytracedAudioSource : MonoBehaviour
{
    public RayTraceManager rayTraceManager;
    public AudioClip clip;
    public bool loop = true;
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

        if (assignSourceTransform)
            rayTraceManager.source = transform;

        if (assignClip && clip != null)
            rayTraceManager.inputClip = clip;

        rayTraceManager.loopAudio = loop;

        if (disableUnityAudioSources)
        {
            foreach (AudioSource audioSource in GetComponentsInChildren<AudioSource>())
                audioSource.enabled = false;
        }
    }

    [ContextMenu("Prepare Raytraced Audio")]
    public void Prepare()
    {
        Apply();
        if (rayTraceManager != null)
            rayTraceManager.PrepareCurrentClip();
    }

    [ContextMenu("Play Raytraced Audio")]
    public void Play()
    {
        Apply();
        if (rayTraceManager != null)
            rayTraceManager.StartStreaming();
    }

    [ContextMenu("Stop Raytraced Audio")]
    public void Stop()
    {
        if (rayTraceManager != null)
            rayTraceManager.StopStreaming();
    }
}
