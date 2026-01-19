using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Assertions;

public class AudioSource : MonoBehaviour
{
    [Header("Audio Settings")]
    public AudioClip inputClip;
    public float inputGain = 1.0f;
    public bool loop = true;

    // Start is called before the first frame update
    void Start()
    {
        int sampleRate = 48000;
        Assert.IsTrue(sampleRate == inputClip.frequency, $"SampleRate ({sampleRate}) != input frequency ({inputClip.frequency})");
    }

    // Update is called once per frame
    void Update()
    {
        
    }
}
