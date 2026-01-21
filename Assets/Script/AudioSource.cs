using UnityEngine;

public class AudioSource : MonoBehaviour
{

    [Header("Audio Settings")]
    public AudioClip inputClip;
    public float inputGain = 1.0f;
    public bool loop = true;    
    public float volume = 1f;
}