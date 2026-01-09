using UnityEngine;

[CreateAssetMenu(fileName = "NewAudioMaterial", menuName = "Audio/Audio Material")]
public class AudioMaterial : ScriptableObject
{
    [Range(0f, 1f)]
    [Tooltip("0 = No energy lost, 1 = All energy lost")]
    public float absorption = 0.1f;

    [Range(0f, 1f)]
    [Tooltip("0 = Mirror, 1 = Diffuse")]
    public float scattering = 0.5f;

    [Range(0f, 1f)]
    [Tooltip("0 = Blocked, 1 = Pass through")]
    public float transmission = 0.1f;

    [Range(0.01f, 4f)]
    [Tooltip("IOR 0.1 = Fast (Concrete), 1.0 = Air")]
    public float ior = 0.5f;
}