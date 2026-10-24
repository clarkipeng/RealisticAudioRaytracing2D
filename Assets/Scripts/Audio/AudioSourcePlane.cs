using System;
using System.IO;
using UnityEngine;

// A masked rectangle: legacy fixed sites or seeded finite bursts on the audio clock.
[RequireComponent(typeof(RaytracedAudioSource)), DisallowMultipleComponent]
public sealed class AudioSourcePlane : MonoBehaviour
{
    public Vector2 size = new Vector2(10, 10);
    [Tooltip("Readable linear grayscale texture: white emits, black is sheltered. Empty means white.")]
    public Texture2D mask;
    [Min(1), Tooltip("Fixed spatial sites in legacy mode; maximum retained events in burst mode.")]
    public int samples = 8;
    public int seed = 1;
    [Min(0), Tooltip("Zero preserves fixed sites. Otherwise emit one seeded burst per period, in capture-rate samples.")]
    public int burstPeriodSamples;
    [Min(4), Tooltip("Hann-windowed burst length in capture-rate samples. Samples above bounds retained bursts; overflow is an error.")]
    public int burstDurationSamples = 2400;

    public AcousticTake.EmissionPlane Export(int source, AcousticTake.Dimension dimension)
    {
        if (!float.IsFinite(size.x+size.y) || size.x<=0 || size.y<=0)
            throw new InvalidOperationException(name + ": plane dimensions must be finite and positive.");
        var audio = GetComponent<RaytracedAudioSource>();
        if (audio.streamed || !audio.loop || audio.clip == null)
            throw new InvalidOperationException(name + ": a rain plane requires a looping PCM clip.");
        if (mask != null && !mask.isReadable)
            throw new InvalidOperationException(name + ": enable Read/Write on the rain mask.");
        var inverse = Quaternion.Inverse(transform.rotation);
        var pixels = mask == null ? new[] { Color.white } : mask.GetPixels();
        var result = new AcousticTake.EmissionPlane { source = source, samples = samples, seed = seed,
            u = inverse * transform.TransformVector(Vector3.right * size.x),
            v = inverse * transform.TransformVector((dimension == AcousticTake.Dimension.Planar2D ? Vector3.up : Vector3.forward) * size.y),
            width = mask == null ? 1 : mask.width, height = mask == null ? 1 : mask.height,
            mask = Array.ConvertAll(pixels, p => p.r),
            generatorVersion = burstPeriodSamples == 0 ? 0 : 1, burstPeriodSamples = burstPeriodSamples,
            burstDurationSamples = burstPeriodSamples == 0 ? 0 : burstDurationSamples };
        Validate(result);
        return result;
    }

    internal static void Validate(AcousticTake.EmissionPlane p)
    {
        if (p == null || (p.generatorVersion != 0 && p.generatorVersion != 1) ||
            (p.generatorVersion == 0 ? p.burstPeriodSamples != 0 || p.burstDurationSamples != 0 : p.burstPeriodSamples <= 0 || p.burstDurationSamples < 4) || p.samples < 1 || p.width < 1 || p.height < 1 || p.mask == null ||
            (long)p.width * p.height != p.mask.Length || !float.IsFinite(p.u.sqrMagnitude + p.v.sqrMagnitude) ||
            !float.IsFinite(Vector3.Cross(p.u, p.v).sqrMagnitude) || !(Vector3.Cross(p.u, p.v).sqrMagnitude > 0) ||
            Array.Exists(p.mask, x => !float.IsFinite(x) || x < 0 || x > 1))
            throw new InvalidDataException("Invalid emission plane or mask.");
    }

    internal static uint Hash(uint x)
    {
        unchecked { x ^= x >> 16; x *= 0x7feb352d; x ^= x >> 15; x *= 0x846ca68b; return x ^ (x >> 16); }
    }
    static float Fraction(uint x, uint radix)
    {
        float value = 0, scale = 1;
        while (x > 0) { scale /= radix; value += (x % radix) * scale; x /= radix; }
        return value;
    }
    internal static Vector4 Sample(AcousticTake.EmissionPlane p, int index)
    {
        // Shifted Halton prefixes retain existing positions when the sample budget changes.
        float u = Mathf.Repeat(Fraction((uint)index, 2) + (Hash((uint)p.seed) >> 8) / 16777216f, 1);
        float v = Mathf.Repeat(Fraction((uint)index, 3) + (Hash((uint)p.seed ^ 0x9e3779b9u) >> 8) / 16777216f, 1);
        // Nearest mask texel keeps black shelter boundaries black, without filtering light into them.
        float density = p.mask[Math.Min((int)(v * p.height), p.height - 1) * p.width + Math.Min((int)(u * p.width), p.width - 1)];
        Vector3 local = p.u * (u - .5f) + p.v * (v - .5f);
        return new Vector4(local.x, local.y, local.z, Mathf.Sqrt(Vector3.Cross(p.u, p.v).magnitude * density / p.samples));
    }
    internal static int MaximumSamples(AcousticTake.Scene scene)
    {
        int count = 1;
        if (scene.planes != null) foreach (var plane in scene.planes) count = Math.Max(count, plane.samples);
        return count;
    }
    internal static void ValidateRender(AcousticTake.Scene scene, TakeRender.Settings settings)
    {
        if (scene.planes != null && scene.planes.Length > 0 &&
            (settings.transport != TakeRender.Transport.Raytraced || settings.synthesis != TakeRender.Synthesis.Spectral))
            throw new NotSupportedException("Rain planes require raytraced spectral rendering (GPU or CPU-reference synthesis).");
    }
    internal static void ValidateEvent(AcousticTake.Scene scene, AcousticTake.VoiceEvent e)
    {
        if (e.action==AcousticTake.VoiceAction.Play && !e.loop && scene.planes!=null && Array.Exists(scene.planes,p=>p.source==e.sourceIndex))
            throw new InvalidDataException("Rain planes require looping Play events.");
    }

    void OnDrawGizmosSelected()
    {
        var manager = GetComponent<RaytracedAudioSource>().rayTraceManager;
        bool planar = manager != null && manager.takeDimension == AcousticTake.Dimension.Planar2D;
        var previous = Gizmos.matrix; Gizmos.matrix = transform.localToWorldMatrix; Gizmos.color = new Color(.3f, .8f, 1);
        Gizmos.DrawWireCube(Vector3.zero, planar ? new Vector3(size.x, size.y, 0) : new Vector3(size.x, 0, size.y));
        Gizmos.matrix = previous;
    }
}
