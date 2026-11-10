using System;
using System.Globalization;
using UnityEngine;

// Unity -batchmode -executeMethod StaticImpulseTake.Derive
//   -impulseTake <take> -impulseOutput <new take>[;...] -impulseSample <take sample> -impulseListener x,y,z[;...] -impulseSeconds <s>
// (';'-separated outputs and listener points derive one take each)
// A static copy of a take for measuring rendered impulse responses
// (docs/benchmarks/energy-audio-2026-11-09/realism.py): every pose frozen at one
// sample of the take, the listener moved to a fixed point, and source 0 playing a
// single unit sample at time zero. The render of this take is the renderer's own
// binaural impulse response at that point (direct-sound convention: the free-field
// direct sound at the reference distance has amplitude 1), so convolving a dry
// recording with it gives what the renderer plays for a static listener.
public static class StaticImpulseTake
{
    public static void Derive()
    {
        if (!Application.isBatchMode) throw new InvalidOperationException("Offline only.");
        string input = Arg("-impulseTake");
        var outputs = Arg("-impulseOutput").Split(';');
        var points = Arg("-impulseListener").Split(';');
        if (outputs.Length != points.Length) throw new ArgumentException("One listener point per output.");
        long at = long.Parse(Arg("-impulseSample"), CultureInfo.InvariantCulture);
        double seconds = double.Parse(Arg("-impulseSeconds"), CultureInfo.InvariantCulture);
        for (int n = 0; n < outputs.Length; n++)
            Write(input, outputs[n], at, Array.ConvertAll(points[n].Split(','), v => float.Parse(v, CultureInfo.InvariantCulture)), seconds);
    }

    static void Write(string input, string output, long at, float[] p, double seconds)
    {
        using var take = AcousticTake.Open(input);
        var meta = take.Metadata;
        var scene = JsonUtility.FromJson<AcousticTake.Scene>(JsonUtility.ToJson(meta.scene));
        if (scene.sources.Length == 0) throw new ArgumentException("The take has no source.");
        var pcm = new float[scene.clips.Length][];
        for (int c = 0; c < pcm.Length; c++) pcm[c] = c == scene.sources[0].clipIndex ? new[] {1f} : take.ReadClip(c);
        var state = new AcousticTake.State[scene.objects.Length];
        take.Evaluate(Math.Min(Math.Max(0, at), meta.durationSamples), state);
        var listener = scene.listeners[0];
        state[listener].position = new Vector3(p[0], p[1], p[2]);
        for (int i = 0; i < state.Length; i++) state[i].velocity = state[i].angularVelocity = state[i].acceleration = state[i].angularAcceleration = Vector3.zero;
        long length = (long)Math.Ceiling(seconds * meta.sampleRate);
        using (var writer = new AcousticTake.Writer(output, meta.sampleRate, scene, take.Geometry, pcm))
        {
            writer.Append(0, state);
            writer.Append(new AcousticTake.VoiceEvent {sample = 0, sourceIndex = 0, action = AcousticTake.VoiceAction.Play, clipOffset = 0, loop = false, gain = 1});
            writer.Append(length, state);
            writer.Complete();
        }
        Debug.Log($"STATIC_IMPULSE_TAKE {output} sample={at} listener={state[listener].position} seconds={seconds}");
    }

    static string Arg(string name)
    {
        var args = Environment.GetCommandLineArgs();
        int i = Array.IndexOf(args, name);
        if (i < 0 || i + 1 >= args.Length) throw new ArgumentException("Missing " + name);
        return args[i + 1];
    }
}
