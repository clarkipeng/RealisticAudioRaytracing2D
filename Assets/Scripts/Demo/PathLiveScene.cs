using System;
using UnityEngine;

// A scene's part in the live showcase (PathLive): the sources that play, and the smoke run's walk. It sits next to the
// scene's SceneSession; scenes without it are not live showcase scenes.
[DisallowMultipleComponent]
public sealed class PathLiveScene : MonoBehaviour
{
    [Tooltip("The sources that stay on. Every other source in the scene is switched off.")]
    public RaytracedAudioSource[] sources = Array.Empty<RaytracedAudioSource>();
    [Tooltip("Those of the sources that play looped from the start; the scene's actors play the rest.")]
    public RaytracedAudioSource[] startLooped = Array.Empty<RaytracedAudioSource>();
    [Tooltip("Rays in flight per source (0: PathLive's default, 8,000). The tracer, the denoiser and the reverb run per " +
             "source, so a scene with more sources needs fewer rays per source; -pathPopulation N overrides it.")]
    [Min(0)] public int population;
    [Tooltip("The smoke run's walk from Player spawn 0: waypoints in order, at floor height.")]
    public Vector3[] route = Array.Empty<Vector3>();
    [Tooltip("The smoke run presses E after reaching this waypoint (-1: never).")]
    public int interactAfter = -1;
    [Tooltip("Seconds the smoke run stands still before the walk and after it.")]
    [Min(0)] public float standSeconds = 5, holdSeconds = 2;

    void OnDrawGizmosSelected()
    {
        Gizmos.color = new Color(.25f, .75f, .67f);
        for (int i = 0; i < route.Length; i++)
        {
            Gizmos.DrawWireSphere(route[i], .3f);
            if (i > 0) Gizmos.DrawLine(route[i - 1], route[i]);
        }
    }
}
