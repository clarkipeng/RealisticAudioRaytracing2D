using System;
using System.Collections.Generic;
using System.IO;
using System.Security.Cryptography;
using UnityEngine;

// Portable inputs for replay. These types contain no Unity object references.
public sealed class AcousticTake : IDisposable
{
    public const int Version = 1;
    const int StateBytes = 80; // position, quaternion, four motion vectors, flags
    public enum Dimension { Planar2D, Spatial3D }
    public enum VoiceAction { Play, Stop, Gain }

    [Serializable] public sealed class ObjectDefinition
    {
        public string id, name;
        public int primitiveStart, primitiveCount;
        public Vector3 scale;
    }

    [Serializable] public sealed class SourceDefinition
    {
        public int objectIndex, clipIndex;
    }

    [Serializable] public sealed class ClipDefinition
    {
        public string name, file, sha256;
        public int samples;
    }

    [Serializable] public sealed class MaterialDefinition
    {
        public string name;
        public string interpolation = "linear-hz-clamped-v1";
        public float[] frequencies, absorption, scattering;
    }

    [Serializable] public sealed class Scene
    {
        public Dimension dimension;
        public ObjectDefinition[] objects;
        public SourceDefinition[] sources;
        public int[] listeners;
        public ClipDefinition[] clips;
        public MaterialDefinition[] materials;
        public string geometryHash;
    }

    [Serializable] public sealed class Provenance
    {
        public string solverContract="direct-opaque-v1", dspContract="fractional-delay-hrtf-fir-v2";
        public string csharpModule,shaderName,shaderDependencyHash,unityVersion,buildGuid;
        public static Provenance Capture(ComputeShader shader=null)
        {
            var value=new Provenance {csharpModule=typeof(AcousticTake).Assembly.ManifestModule.ModuleVersionId.ToString(),
                unityVersion=Application.unityVersion,buildGuid=Application.buildGUID,shaderName=shader==null ? null:shader.name};
#if UNITY_EDITOR
            if (shader!=null)
            {
                string path=UnityEditor.AssetDatabase.GetAssetPath(shader);
                if (!string.IsNullOrEmpty(path)) value.shaderDependencyHash=UnityEditor.AssetDatabase.GetAssetDependencyHash(path).ToString();
            }
#endif
            return value;
        }
    }

    [Serializable] public sealed class Manifest
    {
        public int version = Version;
        public Provenance captureProvenance;
        public int sampleRate, frameCount, eventCount;
        public int coalescedObservations;
        public string observationPolicy;
        public long durationSamples;
        public long maxObservationGapSamples;
        public string sceneHash, stateHash, eventHash;
        public string interpolation = "linear-position-shortest-quaternion-v1";
        public string motion = "backward-pose-difference-v1";
        public bool complete;
        public Scene scene;
    }

    public struct Primitive
    {
        // Baked local coordinates: segment a/b in XY, or triangle a/b/c in 3D.
        public Vector3 a, b, c;
        public int objectIndex, materialIndex;
    }

    [Flags] public enum StateFlags : uint
    {
        Active = 1, Geometry = 2, Source = 4, Listener = 8,
        Cut = 16, VelocityValid = 32, AccelerationValid = 64
    }

    public struct State
    {
        public Vector3 position, velocity, angularVelocity, acceleration, angularAcceleration;
        public Quaternion rotation;
        public StateFlags flags;

        public static State Observe(Transform transform, StateFlags roles)
        {
            return new State { position = transform.position, rotation = transform.rotation,
                flags = roles | (transform.gameObject.activeInHierarchy ? StateFlags.Active : 0) };
        }

        public void MeasureMotion(State previous, double seconds)
        {
            flags &= ~(StateFlags.VelocityValid | StateFlags.AccelerationValid);
            velocity = angularVelocity = acceleration = angularAcceleration = Vector3.zero;
            if (seconds <= 0 || (flags & StateFlags.Cut) != 0) return;
            float dt = (float)seconds;
            velocity = (position - previous.position) / dt;
            Quaternion delta = rotation * Quaternion.Inverse(previous.rotation);
            if (delta.w < 0) delta = new Quaternion(-delta.x, -delta.y, -delta.z, -delta.w);
            delta.ToAngleAxis(out float degrees, out Vector3 axis);
            angularVelocity = degrees < 0.0001f ? Vector3.zero : axis * (degrees * Mathf.Deg2Rad / dt);
            flags |= StateFlags.VelocityValid;
            if ((previous.flags & StateFlags.VelocityValid) != 0)
            {
                acceleration = (velocity - previous.velocity) / dt;
                angularAcceleration = (angularVelocity - previous.angularVelocity) / dt;
                flags |= StateFlags.AccelerationValid;
            }
        }
    }

    public struct VoiceEvent
    {
        public long sample, clipOffset;
        public int sourceIndex;
        public VoiceAction action;
        public float gain;
        public bool loop;
    }

    public Manifest Metadata { get; private set; }
    public Primitive[] Geometry { get; private set; }
    public VoiceEvent[] Events { get; private set; }
    public string DirectoryPath { get; private set; }
    public long FrameSample(int index) { return samples[index]; }
    public void ReadStoredFrame(int index, State[] output)
    {
        if (index < 0 || index >= samples.Length || output.Length != left.Length) throw new ArgumentOutOfRangeException(nameof(index));
        ReadFrame(index,output);
    }
    readonly BinaryReader states;
    readonly long[] samples;
    readonly State[] left, right;

    AcousticTake(string directory, Manifest manifest, Primitive[] geometry, VoiceEvent[] events,
        BinaryReader reader, long[] frameSamples)
    {
        DirectoryPath = directory; Metadata = manifest; Geometry = geometry; Events = events;
        states = reader; samples = frameSamples;
        left = new State[manifest.scene.objects.Length]; right = new State[left.Length];
    }

    // Metadata and source events can also describe input-only runs without pose/PCM files.
    public static Manifest ReadManifest(string directory)
    {
        var manifest = JsonUtility.FromJson<Manifest>(File.ReadAllText(Path.Combine(directory, "take.json")));
        if (manifest == null || manifest.version != Version || !manifest.complete ||
            manifest.sampleRate <= 0 || manifest.frameCount <= 0 || manifest.durationSamples < 0 ||
            manifest.scene == null || manifest.scene.objects == null || manifest.eventCount < 0 ||
            manifest.interpolation != "linear-position-shortest-quaternion-v1" ||
            manifest.motion != "backward-pose-difference-v1")
            throw new InvalidDataException("Unsupported or incomplete acoustic take.");
        ValidateScene(manifest.scene);
        if (HashText(JsonUtility.ToJson(manifest.scene)) != manifest.sceneHash)
            throw new InvalidDataException("Scene metadata checksum mismatch.");
        return manifest;
    }

    public static AcousticTake Open(string directory)
    {
        var manifest=ReadManifest(directory);
        Verify(directory, "geometry.bin", manifest.scene.geometryHash);
        Verify(directory, "states.bin", manifest.stateHash);
        foreach (var clip in manifest.scene.clips)
        {
            Verify(directory, clip.file, clip.sha256);
            if (new FileInfo(Path.Combine(directory, clip.file)).Length != (long)clip.samples * 4)
                throw new InvalidDataException("PCM size mismatch.");
        }
        Primitive[] geometry;
        using (var r = new BinaryReader(File.OpenRead(Path.Combine(directory, "geometry.bin"))))
        {
            if (r.BaseStream.Length % 44 != 0) throw new InvalidDataException("Invalid geometry length.");
            geometry = new Primitive[checked((int)(r.BaseStream.Length / 44))];
            for (int i = 0; i < geometry.Length; i++)
            {
                geometry[i] = new Primitive { a = ReadVector(r), b = ReadVector(r), c = ReadVector(r),
                    objectIndex = r.ReadInt32(), materialIndex = r.ReadInt32() };
                if (geometry[i].objectIndex < 0 || geometry[i].objectIndex >= manifest.scene.objects.Length ||
                    geometry[i].materialIndex < 0 || geometry[i].materialIndex >= manifest.scene.materials.Length)
                    throw new InvalidDataException("Invalid primitive index.");
            }
        }
        var events=ReadEvents(directory,manifest);
        var reader = new BinaryReader(File.OpenRead(Path.Combine(directory, "states.bin")));
        try
        {
            long stride = checked(8L + StateBytes * (long)manifest.scene.objects.Length);
            if (reader.BaseStream.Length != stride * manifest.frameCount)
                throw new InvalidDataException("State count/size mismatch.");
            var times = new long[manifest.frameCount];
            for (int i = 0; i < times.Length; i++)
            {
                reader.BaseStream.Position = i * stride;
                times[i] = reader.ReadInt64();
                if ((i == 0 && times[i] != 0) || (i > 0 && times[i] <= times[i - 1]))
                    throw new InvalidDataException("State samples must increase from zero.");
            }
            if (times[times.Length - 1] != manifest.durationSamples)
                throw new InvalidDataException("Missing final state.");
            return new AcousticTake(directory, manifest, geometry, events, reader, times);
        }
        catch { reader.Dispose(); throw; }
    }

    public static VoiceEvent[] ReadEvents(string directory,Manifest manifest)
    {
        Verify(directory,"events.bin",manifest.eventHash);
        using (var r = new BinaryReader(File.OpenRead(Path.Combine(directory, "events.bin"))))
        {
            if (r.BaseStream.Length != (long)manifest.eventCount * 29) throw new InvalidDataException("Invalid event length.");
            var events = new VoiceEvent[manifest.eventCount];
            for (int i = 0; i < events.Length; i++)
            {
                events[i] = new VoiceEvent { sample = r.ReadInt64(), clipOffset = r.ReadInt64(),
                    sourceIndex = r.ReadInt32(), action = (VoiceAction)r.ReadInt32(), gain = r.ReadSingle(), loop = r.ReadBoolean() };
                ValidateEvent(events[i], manifest.scene);
                if (events[i].sample > manifest.durationSamples || (i > 0 && events[i].sample < events[i - 1].sample))
                    throw new InvalidDataException("Unordered or out-of-range event.");
            }
            return events;
        }
    }

    // One reader/evaluator per consumer. Exact stored samples are not re-interpolated.
    public void Evaluate(long sample, State[] output)
    {
        if (sample < 0 || sample > Metadata.durationSamples || output.Length != left.Length)
            throw new ArgumentOutOfRangeException(nameof(sample));
        int index = Array.BinarySearch(samples, sample);
        if (index >= 0) { ReadFrame(index, output); return; }
        int next = ~index;
        ReadFrame(next - 1, left); ReadFrame(next, right);
        Interpolate(sample,samples[next-1],samples[next],Metadata.sampleRate,left,right,output);
    }

    public static void Interpolate(long sample,long leftSample,long rightSample,int sampleRate,State[] left,State[] right,State[] output)
    {
        if (sample<leftSample || sample>rightSample || rightSample<leftSample) throw new ArgumentOutOfRangeException(nameof(sample));
        if (sample==rightSample) { Array.Copy(right,output,output.Length); return; }
        if (sample==leftSample) { Array.Copy(left,output,output.Length); return; }
        float t = (float)((double)(sample - leftSample) / (rightSample - leftSample));
        for (int i = 0; i < output.Length; i++)
        {
            output[i] = left[i]; // Activity and roles are step changes; cuts occur only at their knot.
            output[i].flags = (output[i].flags | StateFlags.VelocityValid) & ~(StateFlags.Cut | StateFlags.AccelerationValid);
            output[i].acceleration = output[i].angularAcceleration = Vector3.zero;
            if ((right[i].flags & StateFlags.Cut) != 0)
            {
                output[i].velocity = output[i].angularVelocity = Vector3.zero;
                continue;
            }
            output[i].position = Vector3.LerpUnclamped(left[i].position, right[i].position, t);
            output[i].rotation = Quaternion.SlerpUnclamped(left[i].rotation, right[i].rotation, t);
            output[i].velocity = (right[i].position - left[i].position) *
                ((float)sampleRate / (rightSample - leftSample));
            var interpolatedMotion=right[i];
            interpolatedMotion.MeasureMotion(left[i],(double)(rightSample-leftSample)/sampleRate);
            output[i].angularVelocity=interpolatedMotion.angularVelocity;
        }
    }

    public bool HasCut(long after,long through)
    {
        int first=Array.BinarySearch(samples,after); first=first>=0 ? first+1:~first;
        for (int f=first; f<samples.Length && samples[f]<=through; f++)
        {
            ReadFrame(f,right);
            foreach (var state in right) if ((state.flags&StateFlags.Cut)!=0) return true;
        }
        return false;
    }

    void ReadFrame(int index, State[] output)
    {
        states.BaseStream.Position = index * (8L + StateBytes * (long)output.Length) + 8;
        for (int i = 0; i < output.Length; i++)
        {
            output[i] = new State { position = ReadVector(states), rotation = new Quaternion(states.ReadSingle(),
                states.ReadSingle(), states.ReadSingle(), states.ReadSingle()), velocity = ReadVector(states),
                angularVelocity = ReadVector(states), acceleration = ReadVector(states), angularAcceleration = ReadVector(states),
                flags = (StateFlags)states.ReadUInt32() };
        }
    }

    public float[] ReadClip(int index)
    {
        var clip = Metadata.scene.clips[index];
        var result = new float[clip.samples];
        using (var r = new BinaryReader(File.OpenRead(Path.Combine(DirectoryPath, clip.file))))
            for (int i = 0; i < result.Length; i++) result[i] = r.ReadSingle();
        return result;
    }

    public void Dispose() { states.Dispose(); }

    public sealed class Writer : IDisposable
    {
        public Manifest Metadata { get; private set; }
        public string DirectoryPath { get; private set; }
        readonly BinaryWriter states, events;
        long lastSample = -1, lastEventSample = -1;
        bool closed, failed;

        public Writer(string directory, int sampleRate, Scene scene, Primitive[] geometry, float[][] pcm)
        {
            if (sampleRate <= 0) throw new ArgumentOutOfRangeException(nameof(sampleRate));
            ValidateScene(scene);
            if (pcm.Length != scene.clips.Length) throw new ArgumentException("Clip count mismatch.");
            if (Directory.Exists(directory)) throw new IOException("Take output directory already exists.");
            DirectoryPath = directory;
            Directory.CreateDirectory(directory);
            using (var w = new BinaryWriter(File.Create(Path.Combine(directory, "geometry.bin"))))
                foreach (var p in geometry)
                { WriteVector(w, p.a); WriteVector(w, p.b); WriteVector(w, p.c); w.Write(p.objectIndex); w.Write(p.materialIndex); }
            scene.geometryHash = HashFile(Path.Combine(directory, "geometry.bin"));
            for (int i = 0; i < pcm.Length; i++)
            {
                scene.clips[i].file = "clip-" + i + ".f32";
                scene.clips[i].samples = pcm[i].Length;
                string file = Path.Combine(directory, scene.clips[i].file);
                using (var w = new BinaryWriter(File.Create(file))) foreach (float value in pcm[i])
                { RequireFinite(value); w.Write(value); }
                scene.clips[i].sha256 = HashFile(file);
            }
            Metadata = new Manifest { scene = scene, sampleRate = sampleRate, captureProvenance=Provenance.Capture(),
                sceneHash = HashText(JsonUtility.ToJson(scene)) };
            File.WriteAllText(Path.Combine(directory, "take.json"), JsonUtility.ToJson(Metadata, true));
            states = new BinaryWriter(File.Create(Path.Combine(directory, "states.bin")));
            events = new BinaryWriter(File.Create(Path.Combine(directory, "events.bin")));
        }

        public bool Append(long sample, State[] frame)
        {
            if (closed) throw new ObjectDisposedException(nameof(Writer));
            if (failed) throw new InvalidOperationException("A frame write failed; discard this incomplete take.");
            if (sample < 0 || sample < lastSample || (lastSample < 0 && sample != 0) || frame.Length != Metadata.scene.objects.Length)
                throw new ArgumentException("Invalid frame timestamp or shape.");
            // Two observations on the same sample are coalesced, never retimed.
            if (sample == lastSample)
            {
                Metadata.coalescedObservations++;
                states.BaseStream.Position -= 8L + StateBytes * (long)frame.Length;
            }
            // A failed overwrite can leave a full-length but partial row. Never finalize it.
            failed = true;
            states.Write(sample);
            foreach (var s in frame)
            {
                WriteVector(states, s.position);
                RequireFinite(s.rotation.x); RequireFinite(s.rotation.y); RequireFinite(s.rotation.z); RequireFinite(s.rotation.w);
                float norm=s.rotation.x*s.rotation.x+s.rotation.y*s.rotation.y+s.rotation.z*s.rotation.z+s.rotation.w*s.rotation.w;
                if (Mathf.Abs(norm-1)>0.001f) throw new InvalidDataException("Pose rotation must be a unit quaternion.");
                states.Write(s.rotation.x); states.Write(s.rotation.y); states.Write(s.rotation.z); states.Write(s.rotation.w);
                WriteVector(states, s.velocity); WriteVector(states, s.angularVelocity);
                WriteVector(states, s.acceleration); WriteVector(states, s.angularAcceleration); states.Write((uint)s.flags);
            }
            bool added = sample != lastSample;
            if (added) Metadata.frameCount++;
            if (lastSample>=0) Metadata.maxObservationGapSamples=Math.Max(Metadata.maxObservationGapSamples,sample-lastSample);
            lastSample = sample;
            failed = false;
            return added;
        }

        public void Append(VoiceEvent e)
        {
            if (closed) throw new ObjectDisposedException(nameof(Writer));
            ValidateEvent(e, Metadata.scene);
            if (e.sample < lastEventSample) throw new ArgumentException("Events must be ordered by sample.");
            events.Write(e.sample); events.Write(e.clipOffset); events.Write(e.sourceIndex);
            events.Write((int)e.action); events.Write(e.gain); events.Write(e.loop);
            lastEventSample = e.sample; Metadata.eventCount++;
        }

        public void Complete()
        {
            if (closed) throw new ObjectDisposedException(nameof(Writer));
            if (failed || lastSample < 0 || lastEventSample > lastSample) throw new InvalidOperationException("Capture requires successful frames and a final state after all events.");
            if (states.BaseStream.Length != Metadata.frameCount*(8L+StateBytes*(long)Metadata.scene.objects.Length))
                throw new InvalidDataException("A frame write failed; capture cannot be completed.");
            Dispose();
            Metadata.durationSamples = lastSample;
            Metadata.stateHash = HashFile(Path.Combine(DirectoryPath, "states.bin"));
            Metadata.eventHash = HashFile(Path.Combine(DirectoryPath, "events.bin"));
            Metadata.complete = true;
            string temp = Path.Combine(DirectoryPath, "take.complete.json");
            File.WriteAllText(temp, JsonUtility.ToJson(Metadata, true));
            File.Replace(temp, Path.Combine(DirectoryPath, "take.json"), null);
        }

        public void Dispose()
        {
            if (closed) return;
            closed = true; states.Dispose(); events.Dispose();
        }
    }

    static void ValidateScene(Scene scene)
    {
        if (scene == null || scene.objects == null || scene.sources == null || scene.listeners == null || scene.clips == null || scene.materials == null ||
            !Enum.IsDefined(typeof(Dimension), scene.dimension)) throw new InvalidDataException("Invalid scene tables.");
        var ids = new HashSet<string>();
        foreach (var obj in scene.objects)
            if (obj == null || string.IsNullOrEmpty(obj.id) || !ids.Add(obj.id)) throw new InvalidDataException("Missing/duplicate object ID.");
        foreach (var source in scene.sources)
            if (source.objectIndex < 0 || source.objectIndex >= scene.objects.Length || source.clipIndex < 0 || source.clipIndex >= scene.clips.Length)
                throw new InvalidDataException("Invalid source binding.");
        foreach (int listener in scene.listeners)
            if (listener < 0 || listener >= scene.objects.Length) throw new InvalidDataException("Invalid listener binding.");
    }

    static void ValidateEvent(VoiceEvent e, Scene scene)
    {
        if (e.sample < 0 || e.clipOffset < 0 || e.sourceIndex < 0 || e.sourceIndex >= scene.sources.Length ||
            !Enum.IsDefined(typeof(VoiceAction), e.action)) throw new InvalidDataException("Invalid audio event.");
        if (e.action == VoiceAction.Play && e.clipOffset > scene.clips[scene.sources[e.sourceIndex].clipIndex].samples)
            throw new InvalidDataException("Clip offset exceeds recorded PCM.");
        RequireFinite(e.gain);
    }

    static void Verify(string directory, string file, string hash)
    {
        if (string.IsNullOrEmpty(file) || Path.GetFileName(file) != file) throw new InvalidDataException("Invalid payload path.");
        if (HashFile(Path.Combine(directory, file)) != hash) throw new InvalidDataException("Checksum mismatch: " + file);
    }
    public static string HashText(string value)
    {
        using (var sha = SHA256.Create()) return Hex(sha.ComputeHash(System.Text.Encoding.UTF8.GetBytes(value)));
    }
    public static string HashFile(string file)
    {
        using (var sha = SHA256.Create()) using (var stream = File.OpenRead(file)) return Hex(sha.ComputeHash(stream));
    }
    static string Hex(byte[] bytes) { return BitConverter.ToString(bytes).Replace("-", "").ToLowerInvariant(); }
    static void RequireFinite(float value)
    {
        if (float.IsNaN(value) || float.IsInfinity(value)) throw new InvalidDataException("Non-finite numeric input.");
    }
    static Vector3 ReadVector(BinaryReader reader) { return new Vector3(reader.ReadSingle(), reader.ReadSingle(), reader.ReadSingle()); }
    static void WriteVector(BinaryWriter writer, Vector3 v)
    { RequireFinite(v.x); RequireFinite(v.y); RequireFinite(v.z); writer.Write(v.x); writer.Write(v.y); writer.Write(v.z); }
}
