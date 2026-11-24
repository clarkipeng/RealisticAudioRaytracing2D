using UnityEngine;

public class TrainVisualLoader : MonoBehaviour
{
    [Header("Visual Asset")]
    public GameObject trainModelPrefab;
    public string childName = "Runtime Train Visual";

    [Header("Placement")]
    public Vector3 localPosition = Vector3.zero;
    public Vector3 localEulerAngles = Vector3.zero;
    public Vector3 localScale = Vector3.one;
    public bool hideExistingRenderers = true;
    public bool disableUnityAudioSources = true;

    GameObject instance;

    void Start()
    {
        Load();
    }

    [ContextMenu("Reload Train Visual")]
    public void Load()
    {
        if (trainModelPrefab == null)
        {
            Debug.LogWarning("TrainVisualLoader has no trainModelPrefab assigned.");
            return;
        }

        if (instance != null)
        {
            if (Application.isPlaying) Destroy(instance);
            else DestroyImmediate(instance);
        }

        if (hideExistingRenderers)
        {
            foreach (Renderer renderer in GetComponentsInChildren<Renderer>())
            {
                if (renderer.gameObject != gameObject)
                    renderer.enabled = false;
            }
        }

        instance = Instantiate(trainModelPrefab, transform);
        instance.name = childName;
        instance.transform.localPosition = localPosition;
        instance.transform.localRotation = Quaternion.Euler(localEulerAngles);
        instance.transform.localScale = localScale;

        if (disableUnityAudioSources)
        {
            foreach (AudioSource audioSource in instance.GetComponentsInChildren<AudioSource>())
                audioSource.enabled = false;
        }
    }
}
