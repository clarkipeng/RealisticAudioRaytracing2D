using System;
using System.Collections.Generic;
using System.IO;
using System.Security.Cryptography;
using UnityEngine;

// Portable inputs for replay. These types contain no Unity object references.
public sealed class AcousticTake : IDisposable
{
    public const int Version = 3;
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
        public bool streamed;
        public int mutedListenerPlusOne;
    }

    [Serializable] public sealed class StreamDefinition
    {
        public int sourceIndex;
        public string file,sha256;
        public long samples;
    }
    [Serializable] public sealed class SessionDefinition
    {
        public string scene,sessionId,clientId,mode,build;
        public int slots,localSlot;
        public double serverTimeOrigin;
    }
    [Serializable] public sealed class StreamEpoch
    {
        public int sourceIndex,rate,channels;
        public long sample;
        public string participant,epoch,processing;
        public double remoteTime,uncertaintySeconds,presentationSeconds; // -1 uncertainty means physical timing is not measured.
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
        public float[] transmission,ior,damping;
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
        public string solverContract="direct-retarded-opaque-v2", dspContract="continuous-delay-interpolated-hrtf-fir-v4";
        public string csharpModule,shaderName,shaderDependencyHash,unityVersion,buildGuid;
        public static Provenance Capture(ComputeShader shader=null,TakeRender.Settings settings=null)
        {
            var value=new Provenance {csharpModule=typeof(AcousticTake).Assembly.ManifestModule.ModuleVersionId.ToString(),
                unityVersion=Application.unityVersion,buildGuid=Application.buildGUID,shaderName=shader==null ? null:shader.name};
            if(settings!=null && settings.transport==TakeRender.Transport.Raytraced)
            {value.solverContract="integrated-edge-preview-v7";value.dspContract=settings.synthesis==TakeRender.Synthesis.ImpulseResponse?"partitioned-linear-fir-hrtf-v1":"path-clock-spectral-interpolated-v3";}
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
        public TakeRender.Settings captureSettings;
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
        public StreamDefinition[] streams;
        public StreamEpoch[] streamEpochs;
        public string sourceBlocksHash;
        public SessionDefinition session;
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
        if (manifest == null || (manifest.version < 1 || manifest.version > Version) || !manifest.complete ||
            manifest.sampleRate <= 0 || manifest.frameCount <= 0 || manifest.durationSamples < 0 ||
            manifest.scene == null || manifest.scene.objects == null || manifest.eventCount < 0 ||
            manifest.interpolation != "linear-position-shortest-quaternion-v1" ||
            manifest.motion != "backward-pose-difference-v1")
            throw new InvalidDataException("Unsupported or incomplete acoustic take.");
        ValidateScene(manifest.scene);
        if(manifest.version<3 && Array.Exists(manifest.scene.sources,v=>v.streamed || v.mutedListenerPlusOne!=0))
            throw new InvalidDataException("Legacy takes cannot contain stream or monitoring extensions.");
        if (SceneHash(manifest.scene,manifest.version) != manifest.sceneHash)
            throw new InvalidDataException("Scene metadata checksum mismatch.");
        if(manifest.version==1)foreach(var material in manifest.scene.materials)
            material.transmission=material.ior=material.damping=null; // Version 1 did not authenticate these fields.
        return manifest;
    }

    [Serializable] sealed class LegacySource { public int objectIndex,clipIndex; }
    [Serializable] sealed class VersionTwoScene
    {
        public Dimension dimension;
        public ObjectDefinition[] objects;
        public LegacySource[] sources;
        public int[] listeners;
        public ClipDefinition[] clips;
        public MaterialDefinition[] materials;
        public string geometryHash;
    }
    [Serializable] sealed class VersionOneMaterial
    {
        public string name,interpolation;
        public float[] frequencies,absorption,scattering;
    }
    [Serializable] sealed class VersionOneScene
    {
        public Dimension dimension;
        public ObjectDefinition[] objects;
        public LegacySource[] sources;
        public int[] listeners;
        public ClipDefinition[] clips;
        public VersionOneMaterial[] materials;
        public string geometryHash;
    }
    public static string SceneHash(Scene scene,int version)
    {
        if(version==Version)return HashText(JsonUtility.ToJson(scene));
        var sources=Array.ConvertAll(scene.sources,v=>new LegacySource {objectIndex=v.objectIndex,clipIndex=v.clipIndex});
        if(version==2)return HashText(JsonUtility.ToJson(new VersionTwoScene {dimension=scene.dimension,objects=scene.objects,sources=sources,
            listeners=scene.listeners,clips=scene.clips,materials=scene.materials,geometryHash=scene.geometryHash}));
        if(version!=1)throw new InvalidDataException("Unsupported take version.");
        var materials=new VersionOneMaterial[scene.materials.Length];
        for(int i=0;i<materials.Length;i++)
        {var m=scene.materials[i];materials[i]=new VersionOneMaterial {name=m.name,interpolation=m.interpolation,frequencies=m.frequencies,absorption=m.absorption,scattering=m.scattering};}
        return HashText(JsonUtility.ToJson(new VersionOneScene {dimension=scene.dimension,objects=scene.objects,sources=sources,listeners=scene.listeners,
            clips=scene.clips,materials=materials,geometryHash=scene.geometryHash}));
    }

    public static AcousticTake Open(string directory)
    {
        var manifest=ReadManifest(directory);
        ValidateStreams(directory,manifest);
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

    readonly Dictionary<int,SourceStream> streamReaders=new Dictionary<int,SourceStream>();
    public SourceStream ReadStream(int source)
    {
        if(!streamReaders.TryGetValue(source,out var reader))
        {
            var d=Metadata.streams[source];reader=new SourceStream(Path.Combine(DirectoryPath,d.file),d.samples);streamReaders.Add(source,reader);
        }
        return reader;
    }
    public void Dispose() { states.Dispose();foreach(var stream in streamReaders.Values)stream.Dispose(); }
    internal static SourceStream[] OpenStreams(string directory,Manifest manifest)
    {
        ValidateStreams(directory,manifest);var result=new SourceStream[manifest.scene.sources.Length];
        try
        {
            for(int i=0;i<result.Length;i++)if(manifest.scene.sources[i].streamed)
                result[i]=new SourceStream(Path.Combine(directory,manifest.streams[i].file),manifest.streams[i].samples);
            return result;
        }
        catch {foreach(var stream in result)stream?.Dispose();throw;}
    }
    public static void CopyStreams(string from,string to,Manifest manifest)
    {
        var original=ReadManifest(from);ValidateStreams(from,original);
        manifest.session=original.session;manifest.streamEpochs=original.streamEpochs;
        if(!Array.Exists(original.scene.sources,s=>s.streamed))return;
        manifest.streams=original.streams;manifest.sourceBlocksHash=original.sourceBlocksHash;
        File.Copy(Path.Combine(from,"source-blocks.bin"),Path.Combine(to,"source-blocks.bin"));
        foreach(var stream in original.streams)if(stream!=null && !string.IsNullOrEmpty(stream.file))File.Copy(Path.Combine(from,stream.file),Path.Combine(to,stream.file));
    }
    [Serializable] sealed class SourceIdentity { public StreamDefinition[] streams;public StreamEpoch[] epochs;public string blocks; }
    public static string SourceHash(Manifest manifest) => Array.Exists(manifest.scene.sources,s=>s.streamed)
        ? HashText(JsonUtility.ToJson(new SourceIdentity {streams=manifest.streams,epochs=manifest.streamEpochs,blocks=manifest.sourceBlocksHash})):null;
    internal static void ValidateStreams(string directory,Manifest manifest)
    {
        bool hasStreams=Array.Exists(manifest.scene.sources,s=>s.streamed);
        if(!hasStreams)return;
        if(manifest.version<3 || manifest.streams==null || manifest.streams.Length!=manifest.scene.sources.Length)
            throw new InvalidDataException("Missing source stream table.");
        Verify(directory,"source-blocks.bin",manifest.sourceBlocksHash);
        var lengths=new long[manifest.streams.Length];
        for(int i=0;i<lengths.Length;i++)if(manifest.scene.sources[i].streamed)
        {
            var d=manifest.streams[i];
            if(d==null || d.sourceIndex!=i || d.samples<0)throw new InvalidDataException("Invalid stream descriptor.");
            Verify(directory,d.file,d.sha256);
            if(new FileInfo(Path.Combine(directory,d.file)).Length!=checked(d.samples*4))throw new InvalidDataException("Stream length mismatch.");
        }
        using(var r=new BinaryReader(File.OpenRead(Path.Combine(directory,"source-blocks.bin"))))
        {
            if(r.BaseStream.Length%28!=0)throw new InvalidDataException("Invalid source block log.");
            while(r.BaseStream.Position<r.BaseStream.Length)
            {
                int source=r.ReadInt32();long first=r.ReadInt64();int count=r.ReadInt32();long available=r.ReadInt64();uint status=r.ReadUInt32();
                if(source<0 || source>=lengths.Length || !manifest.scene.sources[source].streamed || count<0 ||
                    first!=lengths[source] || available<0 || status>7)throw new InvalidDataException("Invalid source block span.");
                lengths[source]=checked(first+count);
            }
        }
        for(int i=0;i<lengths.Length;i++)if(manifest.scene.sources[i].streamed && lengths[i]!=manifest.streams[i].samples)
            throw new InvalidDataException("Unaccounted source samples.");
    }

    public sealed class Writer : IDisposable
    {
        public Manifest Metadata { get; private set; }
        public string DirectoryPath { get; private set; }
        readonly BinaryWriter states, events;
        long lastSample = -1, lastEventSample = -1;
        bool closed, failed;

        public Writer(string directory, int sampleRate, Scene scene, Primitive[] geometry, float[][] pcm, bool retainPayloads = true,int version=Version)
        {
            if (sampleRate <= 0) throw new ArgumentOutOfRangeException(nameof(sampleRate));
            if(version<1 || version>Version)throw new InvalidDataException("Unsupported take version.");
            ValidateScene(scene);
            if(version<3 && Array.Exists(scene.sources,s=>s.streamed || s.mutedListenerPlusOne!=0))throw new InvalidDataException("Source streams and monitor masks require take version 3.");
            if (pcm.Length != scene.clips.Length) throw new ArgumentException("Clip count mismatch.");
            if (Directory.Exists(directory)) throw new IOException("Take output directory already exists.");
            DirectoryPath = directory;
            Directory.CreateDirectory(directory);
            scene.geometryHash = Payload("geometry.bin", retainPayloads, w => {
                foreach (var p in geometry)
                { WriteVector(w, p.a); WriteVector(w, p.b); WriteVector(w, p.c); w.Write(p.objectIndex); w.Write(p.materialIndex); }
            });
            for (int i = 0; i < pcm.Length; i++)
            {
                scene.clips[i].file = "clip-" + i + ".f32";
                scene.clips[i].samples = pcm[i].Length;
                scene.clips[i].sha256 = Payload(scene.clips[i].file, retainPayloads, w => {
                    foreach (float value in pcm[i]) { RequireFinite(value); w.Write(value); }
                });
            }
            Metadata = new Manifest {version=version,scene = scene, sampleRate = sampleRate, captureProvenance=Provenance.Capture(),
                sceneHash = SceneHash(scene,version) };
            File.WriteAllText(Path.Combine(directory, "take.json"), JsonUtility.ToJson(Metadata, true));
            if (retainPayloads) states = new BinaryWriter(File.Create(Path.Combine(directory, "states.bin")));
            events = new BinaryWriter(File.Create(Path.Combine(directory, "events.bin")));
        }

        string Payload(string file, bool retain, Action<BinaryWriter> write)
        {
            using (var sha=SHA256.Create())
            using (var stream=retain ? File.Create(Path.Combine(DirectoryPath,file)) : Stream.Null)
            using (var hashing=new CryptoStream(stream,sha,CryptoStreamMode.Write))
            using (var writer=new BinaryWriter(hashing))
            {
                write(writer); writer.Flush(); hashing.FlushFinalBlock(); return Hex(sha.Hash);
            }
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
                if (states!=null) states.BaseStream.Position -= 8L + StateBytes * (long)frame.Length;
            }
            // A failed overwrite can leave a full-length but partial row. Never finalize it.
            failed = true;
            states?.Write(sample);
            foreach (var s in frame)
            {
                WriteVector(states, s.position);
                RequireFinite(s.rotation.x); RequireFinite(s.rotation.y); RequireFinite(s.rotation.z); RequireFinite(s.rotation.w);
                float norm=s.rotation.x*s.rotation.x+s.rotation.y*s.rotation.y+s.rotation.z*s.rotation.z+s.rotation.w*s.rotation.w;
                if (Mathf.Abs(norm-1)>0.001f) throw new InvalidDataException("Pose rotation must be a unit quaternion.");
                states?.Write(s.rotation.x); states?.Write(s.rotation.y); states?.Write(s.rotation.z); states?.Write(s.rotation.w);
                WriteVector(states, s.velocity); WriteVector(states, s.angularVelocity);
                WriteVector(states, s.acceleration); WriteVector(states, s.angularAcceleration); states?.Write((uint)s.flags);
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
            if (states!=null && states.BaseStream.Length != Metadata.frameCount*(8L+StateBytes*(long)Metadata.scene.objects.Length))
                throw new InvalidDataException("A frame write failed; capture cannot be completed.");
            Dispose();
            Metadata.durationSamples = lastSample;
            Metadata.stateHash = states==null ? null:HashFile(Path.Combine(DirectoryPath, "states.bin"));
            Metadata.eventHash = HashFile(Path.Combine(DirectoryPath, "events.bin"));
            Metadata.complete = true;
            string temp = Path.Combine(DirectoryPath, "take.complete.json");
            File.WriteAllText(temp, JsonUtility.ToJson(Metadata, true));
            File.Replace(temp, Path.Combine(DirectoryPath, "take.json"), null);
        }

        // A required companion recorder may fail after the acoustic payload closes.
        // Publish failure as well: a complete acoustic manifest must never mask it.
        public void Invalidate()
        {
            failed=true; Metadata.complete=false;
            File.WriteAllText(Path.Combine(DirectoryPath,"take.json"),JsonUtility.ToJson(Metadata,true));
        }

        public void Dispose()
        {
            if (closed) return;
            closed = true; states?.Dispose(); events.Dispose();
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
            if (source.objectIndex < 0 || source.objectIndex >= scene.objects.Length || (!source.streamed && (source.clipIndex < 0 || source.clipIndex >= scene.clips.Length)) || source.mutedListenerPlusOne<0 || source.mutedListenerPlusOne>scene.listeners.Length)
                throw new InvalidDataException("Invalid source binding.");
        foreach (int listener in scene.listeners)
            if (listener < 0 || listener >= scene.objects.Length) throw new InvalidDataException("Invalid listener binding.");
    }

    static void ValidateEvent(VoiceEvent e, Scene scene)
    {
        if (e.sample < 0 || e.clipOffset < 0 || e.sourceIndex < 0 || e.sourceIndex >= scene.sources.Length ||
            !Enum.IsDefined(typeof(VoiceAction), e.action)) throw new InvalidDataException("Invalid audio event.");
        var source=scene.sources[e.sourceIndex];
        if(source.streamed && (e.clipOffset!=0 || e.loop))throw new InvalidDataException("Streams use absolute samples and cannot loop or restart an offset.");
        if (e.action == VoiceAction.Play && !source.streamed && e.clipOffset > scene.clips[source.clipIndex].samples)
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
    { RequireFinite(v.x); RequireFinite(v.y); RequireFinite(v.z); writer?.Write(v.x); writer?.Write(v.y); writer?.Write(v.z); }
}
