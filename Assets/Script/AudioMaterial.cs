using UnityEngine;

public class AudioMaterial : MonoBehaviour
{
    [Range(0f, 1f)]
    [Tooltip("0 = No energy lost, 1 = All energy lost")]
    public float absorption = 0.1f;
    [Range(0f, 1f)]
    [Tooltip("0 = Mirror, 1 = Diffuse")]
    public float scattering = 0.5f; // 0=Mirror, 1=Diffuse
    [Range(0f, 1f)]
    [Tooltip("0 = no chance of pass through, 1 = pass through object")]
    public float transmission = 0.1f; // Probability of passing through (0 to 1)

    [Range(0f, 1f)]
    [Tooltip("Inverse of how fast sound travels through, 0 -> fast, 1 -> slower")]
    public float ior = 0.5f;

}