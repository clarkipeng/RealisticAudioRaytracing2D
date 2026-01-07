using UnityEngine;

public class AudioMaterial : MonoBehaviour
{
    [Range(0f, 1f)]
    [Tooltip("0 = Perfect Reflection, 1 = No Reflection")]
    public float absorption = 0.1f;
}