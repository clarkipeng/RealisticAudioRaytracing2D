using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using UnityEngine;

[assembly: System.Runtime.CompilerServices.InternalsVisibleTo("Acoustics.Tests")]

// Small CPU direct-path baseline. Shared numeric input; no scene or physics queries.
// Reflection/energy estimators must pass their own correctness gate before export.
public static partial class TakeRender
{
    public enum ListenerOutput { Mono, Binaural }
    public enum DelayFilter { Linear, WindowedSinc }
    public enum Transport { Direct, Raytraced }
    public enum Synthesis { ImpulseResponse, Spectral }
    public enum AudioBackend { Gpu, CpuReference }
    [Serializable] public sealed class Settings : ISerializationCallbackReceiver
    {
        public int responseStepSamples = 512;
        public float speedOfSound = 343, referenceDistance = 1, minimumDistance = .1f;
        public int tailSamples;
        public int transitionSamples;
        public DelayFilter delayFilter;
        public ListenerOutput listenerOutput;
        public int ChannelsPerListener => listenerOutput==ListenerOutput.Binaural ? 2:1;
        public int sincRadius=16;
        public float maximumDelaySeconds=10;
        public int maximumSignalEvents=4096;
        public int maximumPendingObservations=16;
        public int maximumResponsesPerBatch=64;
        [Tooltip("Use a BVH for fixed geometry. Objects that move use direct intersection tests for the rest of the session.")]
        public bool useBvh;
        public Transport transport;
        public Synthesis synthesis;
        public AudioBackend audioBackend;
        [Min(1)] public int raysPerBin=1,maxBounces=5;
        public int seed;
        [HideInInspector] public int raysPerPair; // Read older serialized total-ray settings.
        public int FrequencyBins => fftSize/2+1;
        public int RaysPerBin => raysPerPair>0 ? checked((raysPerPair-1)/FrequencyBins+1):raysPerBin;
        public void OnBeforeSerialize() { }
        public void OnAfterDeserialize()
        {
            if(raysPerPair>0 && FrequencyBins>0){raysPerBin=RaysPerBin;raysPerPair=0;}
        }
        public int RaysPerPair => checked(FrequencyBins*RaysPerBin);
        public int SpectralSlotsPerPair => checked(FrequencyBins*(diffractionFactor>0?2:1)+RaysPerPair*checked(maxBounces+1));
        [Tooltip("Knife-edge pressure strength: 1 is the model gain, 0 disables diffraction. Lower values reduce shadow sound.")]
        [Min(0)] public float diffractionFactor=1;
        public bool airAbsorption=true;
        [Min(2)] public int fftSize=1024,spectralHop=256;
        public int DspLatencySamples => (transport==Transport.Raytraced ? fftSize:0)+(delayFilter==DelayFilter.WindowedSinc ? sincRadius:0);
        public long HistorySamples(int rate) => checked((long)Math.Ceiling((double)maximumDelaySeconds*rate)+2L*DspLatencySamples+2+(listenerOutput==ListenerOutput.Binaural ? Hrtf.Load(rate).Taps-1:0));
    }

    [Serializable] public sealed class Report
    {
        public string method = "direct-retarded-opaque-v2";
        public string motion = "local-constant-velocity-retarded-direct-and-first-mirror; other-boundaries-frozen-at-solve";
        public int formatVersion=3;
        public string responseLayout = "frames: int64 sample, then source-major/listener-minor float32(delaySeconds,pressureGain,directionX,directionY,directionZ,delayRate); direction is world listener-to-arrival-origin";
        public int channelsPerListener,hrtfTaps;
        public string hrtfDataset,hrtfHash;
        public string sceneHash, takeStateHash, takeEventHash, takeSourceHash, responseHash;
        public string unityVersion = Application.unityVersion;
        public string platform = Application.platform.ToString(), processor = SystemInfo.processorType, graphicsDevice = SystemInfo.graphicsDeviceName;
        public string[] audioHashes;
        public AcousticTake.Provenance provenance;
        public string settingsHash,audioShaderDependencyHash;
        public int dspLatencySamples;
        public long responseSupportSamples;
        public bool tailCoversResponseSupport;
        public long audioUploadBytes,audioReadbackBytes;
        public string backendScope="Transport backend; audio synthesis currently runs on CPU";
        public string traversalCounterScope="Direct opaque diagnostic pair/primitive bound only; excludes spectral rays, bounces, edges and medium queries";
        public long pairPrimitiveTestUpperBound;
        public Settings settings;
        public int sampleRate, sourceCount, listenerCount, responseCount, objectCount, primitiveCount;
        public int responseBatchCapacity;
        public long outputSamples, preparedPcmBytes, workingNumericBytes;
        public double preparationSeconds,wallSeconds, solveSeconds, evaluationSeconds, audioSeconds, responsePreparationSeconds, writeSeconds, maximumSolveSeconds;
        public string backend = "CPU";
        public double uploadSeconds, dispatchReadbackSeconds;
        public double bvhBuildSeconds,bvhUpdateSeconds;
        public int bvhUpdates,bvhStaticPrimitives;
        public string transportDescription;
        public bool complete;
    }

    public const int DirectPathStride=32;
    [System.Runtime.InteropServices.StructLayout(System.Runtime.InteropServices.LayoutKind.Sequential)]
    public struct DirectPath
    {
        public Vector4 direction; // xyz: world listener-to-arrival-origin, w: propagation delay rate (seconds/second)
        public float delaySeconds,pressureGain;
        Vector2 padding;
    }
    internal static void WritePath(BinaryWriter writer,DirectPath path)
    {
        writer.Write(path.delaySeconds); writer.Write(path.pressureGain);
        writer.Write(path.direction.x); writer.Write(path.direction.y); writer.Write(path.direction.z); writer.Write(path.direction.w);
    }

    public static void SolveDirect(AcousticTake.Scene scene, AcousticTake.Primitive[] geometry,
        AcousticTake.State[] state, Settings settings, DirectPath[] output)
        => SolveDirect(scene,geometry,state,settings,output,null);

    internal static void SolveDirect(AcousticTake.Scene scene, AcousticTake.Primitive[] geometry,
        AcousticTake.State[] state, Settings settings, DirectPath[] output,DirectBvh bvh)
    {
        Validate(settings);
        if (state.Length != scene.objects.Length || output.Length != scene.sources.Length * scene.listeners.Length)
            throw new ArgumentException("Direct response shape mismatch.");
        for (int s=0; s<scene.sources.Length; s++) for (int l=0; l<scene.listeners.Length; l++)
        {
            Vector3 origin=state[scene.sources[s].objectIndex].position;
            Vector3 end=state[scene.listeners[l]].position;
            if (scene.dimension == AcousticTake.Dimension.Planar2D) { origin.z=0; end.z=0; }
            var displacement=end-origin;
            var sv=Doppler.Velocity(state[scene.sources[s].objectIndex]);var lv=Doppler.Velocity(state[scene.listeners[l]]);
            if(scene.dimension==AcousticTake.Dimension.Planar2D){sv.z=0;lv.z=0;}
            Doppler.Direct(displacement,sv,lv,settings.speedOfSound,out float delay,out float delayRate,out var direction);
            origin-=sv*delay;displacement=end-origin;
            float distance=displacement.magnitude;
            bool blocked=bvh!=null && bvh.Blocked(origin,end,state);
            if(bvh==null)foreach (var p in geometry)
            {
                var pose=state[p.objectIndex];
                if ((pose.flags & (AcousticTake.StateFlags.Active|AcousticTake.StateFlags.Geometry)) !=
                    (AcousticTake.StateFlags.Active|AcousticTake.StateFlags.Geometry)) continue;
                var a=pose.position+pose.rotation*p.a; var b=pose.position+pose.rotation*p.b;
                if (scene.dimension == AcousticTake.Dimension.Planar2D)
                    blocked=CrossesSegment(origin,end,a,b);
                else blocked=CrossesTriangle(origin,displacement,a,b,pose.position+pose.rotation*p.c);
                if (blocked) break;
            }
            output[s*scene.listeners.Length+l]=new DirectPath { direction=new Vector4(-direction.x,-direction.y,-direction.z,delayRate),delaySeconds=delay,
                pressureGain=blocked ? 0 : settings.referenceDistance/Mathf.Max(distance,settings.minimumDistance) };
        }
        ValidatePaths(output);
    }

    internal static void ValidatePaths(DirectPath[] paths)
    {
        foreach (var p in paths)
            if (p.direction.w>=1 || !float.IsFinite(p.direction.sqrMagnitude) || float.IsNaN(p.delaySeconds) || float.IsInfinity(p.delaySeconds) || float.IsNaN(p.pressureGain) || float.IsInfinity(p.pressureGain))
                throw new InvalidOperationException("Non-finite direct response; check scene scale and solver settings.");
    }

    internal static bool CrossesSegment(Vector3 o, Vector3 end, Vector3 a, Vector3 b)
    {
        var d=end-o; var edge=b-a; var q=a-o;
        double det=(double)d.x*edge.y-(double)d.y*edge.x;
        if (Math.Abs(det)<1e-12) return false;
        double t=((double)q.x*edge.y-(double)q.y*edge.x)/det;
        double u=((double)q.x*d.y-(double)q.y*d.x)/det;
        return t>1e-6 && t<1-1e-6 && u>=0 && u<=1;
    }

    internal static bool CrossesTriangle(Vector3 o, Vector3 direction, Vector3 a, Vector3 b, Vector3 c)
    {
        var e1=b-a; var e2=c-a; var h=Vector3.Cross(direction,e2);
        double det=Vector3.Dot(e1,h);
        if (Math.Abs(det)<1e-12) return false;
        var s=o-a; double u=Vector3.Dot(s,h)/det;
        if (u<0 || u>1) return false;
        var q=Vector3.Cross(s,e1); double v=Vector3.Dot(direction,q)/det;
        if (v<0 || u+v>1) return false;
        double t=Vector3.Dot(e2,q)/det;
        return t>1e-6 && t<1-1e-6;
    }

    // Runs under Editor batch mode as well as from the Inspector. No audio device is required.
    public static Report RenderDirect(AcousticTake take, Settings settings, string outputDirectory, Action<float> progress = null, ComputeShader shader = null)
    {
        Validate(settings);
        var watch=Stopwatch.StartNew();
        if (Directory.Exists(outputDirectory)) throw new IOException("Render output directory already exists.");
        var metadata=take.Metadata; var scene=metadata.scene;
        bool useGpuAudio=GpuAudio.Selected(settings,shader);
        if (scene.sources.Length == 0 || scene.listeners.Length == 0) throw new InvalidOperationException("Rendering requires a recorded source and listener.");
        long length=checked(metadata.durationSamples+1+settings.tailSamples);
        if (length > (uint.MaxValue-36)/(4L*settings.ChannelsPerListener)) throw new InvalidOperationException("Output exceeds the WAV size limit; render a shorter take.");
        Directory.CreateDirectory(outputDirectory);
        var report=new Report {takeSourceHash=AcousticTake.SourceHash(metadata), sceneHash=metadata.sceneHash, takeStateHash=metadata.stateHash, takeEventHash=metadata.eventHash,
            settings=settings, provenance=AcousticTake.Provenance.Capture(shader), settingsHash=AcousticTake.HashText(JsonUtility.ToJson(settings)),
            dspLatencySamples=settings.DspLatencySamples, sampleRate=metadata.sampleRate, sourceCount=scene.sources.Length,
            listenerCount=scene.listeners.Length, objectCount=scene.objects.Length, primitiveCount=take.Geometry.Length, outputSamples=length,channelsPerListener=settings.ChannelsPerListener };
        if(settings.transport==Transport.Raytraced)
        {
            report.method="geometric-spectral-preview-v5";report.provenance.solverContract=report.method;report.provenance.dspContract=settings.synthesis==Synthesis.ImpulseResponse?"partitioned-linear-fir-hrtf-v1":"path-clock-spectral-overlap-save-v1";
            if(useGpuAudio)
            {
                report.provenance.dspContract="gpu-path-filter-overlap-save-v1";
                report.audioShaderDependencyHash=AcousticTake.Provenance.Capture(Resources.Load<ComputeShader>("Acoustics/GpuAudio")).shaderDependencyHash;
            }
            report.responseLayout="Direct opaque diagnostic probes only; indirect and transmitted arrivals are synthesized into WAVs. "+report.responseLayout;
            report.transportDescription="Synthesis="+settings.synthesis+". Frequency-dependent geometric reflection/scattering/transmission/refraction, air/material attenuation and dominant visible knife-edge path with a Fresnel pressure-magnitude approximation; not a validated wave reference. CPU IR uses partitioned linear convolution; GPU IR uses equivalent delayed short filters with overlap-save. Spectral uses continuous path clocks and short filters with overlap-save. First mirrors use retarded image sources with locally constant rigid velocity; other occluders and diffuse/diffraction geometry remain frozen at solve time.";
        }
        if(settings.listenerOutput==ListenerOutput.Binaural)
        {var hrtf=Hrtf.Load(metadata.sampleRate); report.hrtfDataset=Hrtf.Dataset; report.hrtfHash=hrtf.Hash; report.hrtfTaps=hrtf.Taps;}
        string reportPath=Path.Combine(outputDirectory,"run.json");
        File.WriteAllText(reportPath,JsonUtility.ToJson(report,true));
        var state=new AcousticTake.State[scene.objects.Length];
        var response=new DirectPath[scene.sources.Length*scene.listeners.Length];
        var signals=new Signal[scene.sources.Length];
        var pcm=new float[scene.clips.Length][];
        for (int c=0; c<pcm.Length; c++) { pcm[c]=take.ReadClip(c); report.preparedPcmBytes+=(long)pcm[c].Length*4; }
        for (int s=0; s<signals.Length; s++) signals[s]=Signal.FromTake(take,s,scene.sources[s].streamed?null:pcm[scene.sources[s].clipIndex]);
        using var gpuAudio=useGpuAudio?new GpuAudio(scene,metadata.sampleRate,settings,signals):null;
        var mixer=useGpuAudio?null:new Mixer(scene,metadata.sampleRate,settings,signals);
        long initialMixerBytes=gpuAudio?.NumericBytes??mixer.NumericBytes;
        report.backendScope=useGpuAudio?"GPU transport and audio synthesis; CPU source timeline and PCM output":"Transport backend; CPU reference audio synthesis";
        var block=new float[checked(settings.responseStepSamples*scene.listeners.Length*settings.ChannelsPerListener)];
        int batchCapacity=shader==null || !SystemInfo.supportsAsyncGPUReadback ? 1:(int)Math.Min(settings.maximumResponsesPerBatch,(length-1)/settings.responseStepSamples+1);
        var batchStates=batchCapacity==1 ? state:new AcousticTake.State[checked(state.Length*batchCapacity)];
        var batchPaths=batchCapacity==1 ? response:new DirectPath[checked(response.Length*batchCapacity)];
        report.responseBatchCapacity=batchCapacity;
        using (var solver=new TransportSolver(scene,take.Geometry,settings,shader,batchCapacity,metadata.sampleRate,!useGpuAudio))
        {
            report.workingNumericBytes=report.preparedPcmBytes+(long)take.Geometry.Length*44+(long)state.Length*80+(long)response.Length*DirectPathStride+(long)block.Length*4+solver.GpuBufferBytes+solver.CpuBufferBytes+initialMixerBytes+
                (batchCapacity==1?0:(long)batchStates.Length*80+(long)batchPaths.Length*DirectPathStride);
            report.backend=solver.Backend;
            report.preparationSeconds=watch.Elapsed.TotalSeconds;
            var outputs=new List<BinaryWriter>();
            try
            {
                for (int l=0; l<scene.listeners.Length; l++)
                {
                    var writer=new BinaryWriter(File.Create(Path.Combine(outputDirectory,"listener-"+l+".wav")));
                    outputs.Add(writer); WriteWaveHeader(writer,(int)metadata.sampleRate,length,settings.ChannelsPerListener);
                }
                using (var writer=new BinaryWriter(File.Create(Path.Combine(outputDirectory,"responses.bin"))))
                {
                    for (long first=0; first<length; first+=(long)batchCapacity*settings.responseStepSamples)
                    {
                        int frames=(int)Math.Min(batchCapacity,(length-first-1)/settings.responseStepSamples+1);
                        long start=Stopwatch.GetTimestamp();
                        for(int f=0;f<frames;f++)
                        {
                            take.Evaluate(Math.Min(first+(long)f*settings.responseStepSamples,metadata.durationSamples),state);
                            if(first+(long)f*settings.responseStepSamples>metadata.durationSamples)FreezeMotion(state);
                            if(batchCapacity>1)Array.Copy(state,0,batchStates,f*state.Length,state.Length);
                        }
                        report.evaluationSeconds+=Elapsed(start);
                        start=Stopwatch.GetTimestamp();
                        if(batchCapacity==1)solver.Solve(state,response);
                        else
                        {
                            solver.BeginSolve(batchStates,frames);
                            if(!solver.TryComplete(batchPaths,true))throw new InvalidOperationException("Offline response batch did not complete.");
                        }
                        double solveTime=Elapsed(start);
                        report.solveSeconds+=solveTime; report.maximumSolveSeconds=Math.Max(report.maximumSolveSeconds,solveTime);
                        for(int f=0;f<frames;f++)
                        {
                            long sample=first+(long)f*settings.responseStepSamples;
                            if(batchCapacity>1) {Array.Copy(batchStates,f*state.Length,state,0,state.Length);Array.Copy(batchPaths,f*response.Length,response,0,response.Length);}
                            start=Stopwatch.GetTimestamp();
                            bool cut=(settings.transport==Transport.Raytraced || settings.transitionSamples>0) && take.HasCut(sample-settings.responseStepSamples,sample);
                            if(useGpuAudio)gpuAudio.SetResponse(sample,cut,solver.RayBuffer,f*solver.RayPathsPerFrame,scene,state);
                            else mixer.SetResponse(sample,response,state,cut,solver.RayPaths,f*solver.RayPathsPerFrame);
                            double responseTime=Elapsed(start);report.audioSeconds+=responseTime;report.responsePreparationSeconds+=responseTime;
                            start=Stopwatch.GetTimestamp();
                            writer.Write(sample);
                            foreach (var path in response) WritePath(writer,path);
                            report.responseCount++;
                            int count=(int)Math.Min(settings.responseStepSamples,length-sample);
                            report.writeSeconds+=Elapsed(start);
                            start=Stopwatch.GetTimestamp(); if(useGpuAudio)gpuAudio.Render(sample,count,block);else mixer.Render(sample,count,block);
                            report.audioSeconds+=Elapsed(start);
                            start=Stopwatch.GetTimestamp();
                            for (int n=0; n<count; n++) for (int l=0; l<outputs.Count; l++) for(int ear=0;ear<settings.ChannelsPerListener;ear++)
                                outputs[l].Write(block[(n*outputs.Count+l)*settings.ChannelsPerListener+ear]);
                            report.writeSeconds+=Elapsed(start);
                            progress?.Invoke((float)((double)(sample+count)/length));
                        }
                    }
                }
            }
            finally { foreach (var output in outputs) output.Dispose(); }
            report.pairPrimitiveTestUpperBound=solver.PairPrimitiveUpperBound;
            report.uploadSeconds=solver.UploadSeconds; report.dispatchReadbackSeconds=solver.DispatchReadbackSeconds;
            report.bvhBuildSeconds=solver.BvhBuildSeconds;report.bvhUpdateSeconds=solver.BvhUpdateSeconds;report.bvhUpdates=solver.BvhUpdates;report.bvhStaticPrimitives=solver.StaticPrimitiveCount;
        }
        report.workingNumericBytes+=(gpuAudio?.NumericBytes??mixer.NumericBytes)-initialMixerBytes;
        report.responseSupportSamples=gpuAudio?.ResponseSupportSamples??mixer.ResponseSupportSamples;
        report.audioUploadBytes=gpuAudio?.UploadedBytes??0;report.audioReadbackBytes=gpuAudio?.ReadbackBytes??0;
        report.tailCoversResponseSupport=settings.tailSamples>=report.responseSupportSamples;
        report.wallSeconds=watch.Elapsed.TotalSeconds;
        report.responseHash=AcousticTake.HashFile(Path.Combine(outputDirectory,"responses.bin"));
        report.audioHashes=new string[scene.listeners.Length];
        for (int l=0; l<report.audioHashes.Length; l++)
            report.audioHashes[l]=AcousticTake.HashFile(Path.Combine(outputDirectory,"listener-"+l+".wav"));
        report.complete=true;
        File.WriteAllText(Path.Combine(outputDirectory,"run.complete.json"),JsonUtility.ToJson(report,true));
        File.Replace(Path.Combine(outputDirectory,"run.complete.json"),reportPath,null);
        return report;
    }

    // All mutation happens on the producer/control thread. The audio callback receives PCM only.
    public sealed class Signal
    {
        struct Voice { public long sample, start, offset; public bool playing, loop; public float gain; }
        readonly List<Voice> voices=new List<Voice>();
        readonly float[] pcm;
        readonly SourceStream stream;
        public long ReadyEnd => stream==null || stream.Finished?long.MaxValue:stream.End;
        public long EndSample=long.MaxValue;
        readonly List<long> gateSamples=new List<long>();
        readonly List<bool> gateEnabled=new List<bool>();
        readonly int maximumEvents;
        public int RetainedEvents => voices.Count+gateSamples.Count;
        public Signal(float[] clip,int maximumEvents=int.MaxValue) { pcm=clip; this.maximumEvents=maximumEvents; }
        public Signal(SourceStream stream,int maximumEvents=int.MaxValue) {this.stream=stream??throw new ArgumentNullException(nameof(stream));this.maximumEvents=maximumEvents;}
        void ReserveEvent() { if (RetainedEvents>=maximumEvents) throw new InvalidOperationException("Source event history capacity exceeded; increase Maximum Signal Events."); }
        public void Prune(long before)
        {
            stream?.Prune(before);
            int count=0;
            while (count+1<voices.Count && voices[count+1].sample<=before) count++;
            if (count>0) voices.RemoveRange(0,count);
            count=gateSamples.BinarySearch(before); if (count<0) count=~count-1;
            if (count>0) { gateSamples.RemoveRange(0,count); gateEnabled.RemoveRange(0,count); }
        }
        public static Signal FromTake(AcousticTake take,int source,float[] clip)
        {
            var result=take.Metadata.scene.sources[source].streamed?new Signal(take.ReadStream(source)):new Signal(clip);
            result.EndSample=take.Metadata.durationSamples;
            var state=new AcousticTake.State[take.Metadata.scene.objects.Length];
            for (int f=0; f<take.Metadata.frameCount; f++)
            {
                take.ReadStoredFrame(f,state);
                result.SetActive(take.FrameSample(f),Enabled(state[take.Metadata.scene.sources[source].objectIndex],AcousticTake.StateFlags.Source));
            }
            foreach (var e in take.Events) if (e.sourceIndex==source) result.Append(e);
            return result;
        }
        public void SetActive(long sample,bool active)
        {
            int n=gateSamples.Count;
            if (n>0 && sample<gateSamples[n-1]) throw new ArgumentException("Unordered source activity.");
            if (n>0 && sample==gateSamples[n-1]) { gateEnabled[n-1]=active; return; }
            if (n==0 || active!=gateEnabled[n-1]) { ReserveEvent(); gateSamples.Add(sample); gateEnabled.Add(active); }
        }
        public void Append(AcousticTake.VoiceEvent e)
        {
            var current=voices.Count==0 ? new Voice():voices[voices.Count-1];
            if (e.sample<current.sample || e.clipOffset<0 || (stream==null?e.clipOffset>pcm.Length:e.clipOffset!=0 || e.loop) || float.IsNaN(e.gain) || float.IsInfinity(e.gain))
                throw new ArgumentException("Invalid source event.");
            current.sample=e.sample;
            if (e.action==AcousticTake.VoiceAction.Play)
            { current.playing=true; current.start=e.sample; current.offset=e.clipOffset; current.loop=e.loop; current.gain=e.gain; }
            else if (e.action==AcousticTake.VoiceAction.Stop) current.playing=false;
            else if (e.action==AcousticTake.VoiceAction.Gain) current.gain=e.gain;
            else throw new ArgumentException("Unknown voice action.");
            if (voices.Count>0 && voices[voices.Count-1].sample==e.sample) voices[voices.Count-1]=current;
            else { ReserveEvent(); voices.Add(current); }
        }
        public float Read(double sample)
        {
            long i=(long)Math.Floor(sample); float f=(float)(sample-i);
            return Mathf.LerpUnclamped(Read(i),Read(i+1),f);
        }
        internal float Read(long sample)
        {
            if (sample<0 || sample>EndSample || voices.Count==0 || (stream==null && pcm.Length==0)) return 0;
            int gate=gateSamples.BinarySearch(sample); if (gate<0) gate=~gate-1;
            if (gate<0 || !gateEnabled[gate]) return 0;
            int lo=0,hi=voices.Count;
            while (lo<hi) { int mid=lo+(hi-lo)/2; if (voices[mid].sample<=sample) lo=mid+1; else hi=mid; }
            if (lo==0) return 0;
            var v=voices[lo-1]; if (!v.playing) return 0;
            if(stream!=null)return stream.Read(sample)*v.gain;
            long index=sample-v.start+v.offset; if (v.loop) index%=pcm.Length;
            return index<pcm.Length ? pcm[index]*v.gain:0;
        }
    }

    public sealed class Mixer
    {
        readonly AcousticTake.Scene scene;
        readonly int sampleRate,transition,taps,delayTaps,firstTap,latency,channels,maximumBlock;
        readonly float maximumDelay;
        readonly long[] delay,oldDelay;
        readonly float[] weights,oldWeights,delayWeights,input;
        readonly double[] mixed;
        readonly Signal[] signals;
        readonly DirectPath[] current,previous;
        readonly bool[] enabled,oldEnabled;
        readonly Hrtf hrtf;
        readonly SpectralAudio spectral;
        readonly TakeRender.DelayFilter delayFilter;
        readonly int sincRadius;
        readonly Hrtf.Blend[] earBlend,oldEarBlend;
        long responseSample,previousResponseSample;
        bool initialized;
        int fade;
        long directSupport;
        public long ResponseSupportSamples => spectral!=null?spectral.ResponseSupportSamples:directSupport;
        public long NumericBytes => (long)(current.Length+previous.Length)*DirectPathStride+(long)(delay.Length+oldDelay.Length)*8+
            (long)(earBlend.Length+oldEarBlend.Length)*32+(long)(weights.Length+oldWeights.Length+delayWeights.Length+input.Length)*4+(long)mixed.Length*8+enabled.Length+oldEnabled.Length+(hrtf?.NumericBytes ?? 0)+(spectral?.NumericBytes??0);
        public Mixer(AcousticTake.Scene scene,int sampleRate,Settings settings,Signal[] signals)
        {
            Validate(settings); this.scene=scene; this.sampleRate=sampleRate; this.signals=signals; transition=settings.transitionSamples;
            channels=settings.ChannelsPerListener; maximumBlock=settings.responseStepSamples;
            delayFilter=settings.delayFilter;sincRadius=settings.sincRadius;
            earBlend=new Hrtf.Blend[scene.sources.Length*scene.listeners.Length];oldEarBlend=new Hrtf.Blend[earBlend.Length];
            if(channels==2) hrtf=Hrtf.Load(sampleRate);
            current=new DirectPath[scene.sources.Length*scene.listeners.Length]; previous=new DirectPath[current.Length];
            enabled=new bool[scene.listeners.Length]; oldEnabled=new bool[enabled.Length];
            latency=settings.DspLatencySamples; maximumDelay=settings.maximumDelaySeconds;
            delayTaps=settings.delayFilter==DelayFilter.Linear ? 2:checked(settings.sincRadius*2);
            int hrtfTail=hrtf==null ? 0:hrtf.Taps-1;
            taps=checked(delayTaps+hrtfTail);
            firstTap=(settings.delayFilter==DelayFilter.Linear ? 0:1-settings.sincRadius)-hrtfTail;
            weights=new float[checked(current.Length*channels*taps)]; oldWeights=new float[weights.Length];
            delayWeights=new float[delayTaps]; input=new float[checked(maximumBlock+taps-1)];
            mixed=new double[checked(maximumBlock*channels*enabled.Length)];
            delay=new long[current.Length]; oldDelay=new long[delay.Length];
            if(settings.transport==Transport.Raytraced)spectral=new SpectralAudio(scene,sampleRate,settings,signals);
        }
        public void SetResponse(long sample,DirectPath[] paths,AcousticTake.State[] state,bool cut=false) => SetResponse(sample,paths,state,cut,null,0);
        internal void SetResponse(long sample,DirectPath[] paths,AcousticTake.State[] state,bool cut,RayTransport.Path[] arrivals,int offset)
        {
            if(paths.Length!=current.Length || state.Length!=scene.objects.Length || (initialized && sample<responseSample)) throw new ArgumentException("Invalid response order/shape.");
            if(spectral!=null){spectral.SetResponse(sample,cut,arrivals,offset,scene,state);responseSample=sample;initialized=true;return;}
            ValidatePaths(paths);
            foreach(var path in paths) if(path.delaySeconds<0 || path.delaySeconds>maximumDelay)
                throw new InvalidOperationException("Response exceeds Maximum Delay Seconds; increase the declared history horizon.");
            previousResponseSample=responseSample;Array.Copy(earBlend,oldEarBlend,earBlend.Length);
            Array.Copy(current,previous,current.Length); Array.Copy(paths,current,current.Length);
            Array.Copy(delay,oldDelay,delay.Length); Array.Copy(weights,oldWeights,weights.Length); Array.Clear(weights,0,weights.Length);
            for(int pair=0;pair<paths.Length;pair++)
            {
                double samples=(double)paths[pair].delaySeconds*sampleRate;
                delay[pair]=checked((long)Math.Ceiling(samples)+latency);
                directSupport=Math.Max(directSupport,delay[pair]-firstTap);
                double fraction=Math.Ceiling(samples)-samples,sum=0;
                int delayFirst=latency==0 ? 0:1-latency;
                for(int t=0;t<delayTaps;t++)
                {
                    double x=fraction-(delayFirst+t);
                    double weight=latency==0 ? (t==0 ? 1-fraction:fraction):Sinc(x)*Sinc(x/latency);
                    delayWeights[t]=(float)weight; sum+=weight;
                }
                for(int t=0;t<delayTaps;t++) delayWeights[t]/=(float)sum;
                if(hrtf==null) Array.Copy(delayWeights,0,weights,pair*taps,taps);
                else
                {
                    int listener=scene.listeners[pair%scene.listeners.Length];
                    Vector3 local=Quaternion.Inverse(state[listener].rotation)*(Vector3)paths[pair].direction;
                    var direction=hrtf.Interpolate(local);earBlend[pair]=direction;
                    // Compose the fractional-delay FIR and HRIR once per response.
                    for(int ear=0;ear<2;ear++) for(int k=0;k<hrtf.Taps;k++) for(int t=0;t<delayTaps;t++)
                        weights[(pair*2+ear)*taps+hrtf.Taps-1-k+t]+=delayWeights[t]*hrtf.Coefficient(direction,ear,k);
                }
            }
            Array.Copy(enabled,oldEnabled,enabled.Length);
            for(int l=0;l<enabled.Length;l++) enabled[l]=Enabled(state[scene.listeners[l]],AcousticTake.StateFlags.Listener);
            fade=initialized && !cut ? transition:0;
            foreach(var s in state) if((s.flags&AcousticTake.StateFlags.Cut)!=0) fade=0;
            responseSample=sample; initialized=true;
        }
        static double Sinc(double x) => Math.Abs(x)<1e-12 ? 1:Math.Sin(Math.PI*x)/(Math.PI*x);
        static readonly Unity.Profiling.ProfilerMarker mixMarker=new Unity.Profiling.ProfilerMarker("Acoustics.MixPCM");
        public void Render(long sample,int count,float[] interleaved)
        {
            using(mixMarker.Auto())
            {
                if(!initialized || sample<responseSample || count<0 || count>maximumBlock || interleaved.Length<(long)count*enabled.Length*channels) throw new ArgumentException("Invalid mix block.");
                if(spectral!=null){spectral.Render(sample,count,interleaved);return;}
                int length=count*enabled.Length*channels; Array.Clear(mixed,0,length);
                for(int s=0;s<signals.Length;s++) for(int l=0;l<enabled.Length;l++)
                {
                    if(scene.sources[s].mutedListenerPlusOne==l+1)continue;
                    int pair=s*enabled.Length+l;
                    if(enabled[l] && current[pair].pressureGain!=0) Mix(s,l,pair,sample,count,current[pair].pressureGain,delay,weights,false);
                    if(oldEnabled[l] && fade>0 && sample-responseSample<fade && previous[pair].pressureGain!=0)
                        Mix(s,l,pair,sample,count,previous[pair].pressureGain,oldDelay,oldWeights,true);
                }
                for(int i=0;i<length;i++)
                {
                    float value=(float)mixed[i];
                    if(!float.IsFinite(value)) throw new InvalidOperationException("Non-finite mixed audio; render cannot complete.");
                    interleaved[i]=value;
                }
            }
        }
        void Mix(int source,int listener,int pair,long sample,int count,float gain,long[] delays,float[] kernel,bool old)
        {
            var path=old?previous[pair]:current[pair];
            if(path.direction.w!=0)
            {
                double ratio=1-(double)path.direction.w;long anchor=old?previousResponseSample:responseSample;
                var ears=old?oldEarBlend[pair]:earBlend[pair];
                for(int n=0;n<count;n++)
                {
                    float blend=fade==0?1:Mathf.Clamp01((float)(sample+n-responseSample)/fade);if(old)blend=1-blend;
                    if(blend==0)continue;
                    for(int k=0;k<(hrtf?.Taps??1);k++)
                    {
                        double receiver=sample+n-latency-k;
                        double emission=receiver-path.delaySeconds*sampleRate-path.direction.w*(receiver-anchor);
                        float value=Doppler.Read(signals[source],emission,ratio,delayFilter,sincRadius);
                        for(int ear=0;ear<channels;ear++)mixed[(n*enabled.Length+listener)*channels+ear]+=blend*gain*value*(hrtf==null?1:hrtf.Coefficient(ears,ear,k));
                    }
                }
                return;
            }
            long first=sample-delays[pair]+firstTap;
            // Evaluate source events once per needed sample, shared by both ears/FIR taps.
            for(int i=0;i<count+taps-1;i++) input[i]=signals[source].Read(first+i);
            for(int n=0;n<count;n++)
            {
                float blend=fade==0 ? 1:Mathf.Clamp01((float)(sample+n-responseSample)/fade);
                if(old) blend=1-blend;
                if(blend==0) continue;
                for(int ear=0;ear<channels;ear++)
                {
                    double sum=0; int offset=(pair*channels+ear)*taps;
                    for(int k=0;k<taps;k++) sum+=kernel[offset+k]*input[n+k];
                    mixed[(n*enabled.Length+listener)*channels+ear]+=blend*gain*sum;
                }
            }
        }
    }
    internal static void FreezeMotion(AcousticTake.State[] states)
    {
        // The requested export tail holds the last scene pose, so it must also hold its motion.
        for(int i=0;i<states.Length;i++)
        {states[i].velocity=states[i].angularVelocity=Vector3.zero;states[i].flags&=~(AcousticTake.StateFlags.VelocityValid|AcousticTake.StateFlags.AccelerationValid);}
    }
    public static bool Enabled(AcousticTake.State state,AcousticTake.StateFlags role) =>
        (state.flags&(AcousticTake.StateFlags.Active|role))==(AcousticTake.StateFlags.Active|role);
    public static double Elapsed(long start) => (double)(Stopwatch.GetTimestamp()-start)/Stopwatch.Frequency;

    public static void Validate(Settings settings)
    {
        if (settings == null || (settings.listenerOutput!=ListenerOutput.Mono && settings.listenerOutput!=ListenerOutput.Binaural) || settings.responseStepSamples<=0 || settings.tailSamples<0 || settings.transitionSamples<0 || settings.transitionSamples>settings.responseStepSamples ||
            !PositiveFinite(settings.maximumDelaySeconds) || settings.maximumSignalEvents<2 || settings.maximumPendingObservations<1 || settings.maximumResponsesPerBatch<1 ||
            (settings.delayFilter!=DelayFilter.Linear && settings.delayFilter!=DelayFilter.WindowedSinc) || settings.sincRadius<2 || settings.sincRadius>int.MaxValue/2 ||
            !PositiveFinite(settings.speedOfSound) || !PositiveFinite(settings.referenceDistance) || !PositiveFinite(settings.minimumDistance))
            throw new ArgumentException("Invalid render settings.");
        if(settings.audioBackend!=AudioBackend.Gpu && settings.audioBackend!=AudioBackend.CpuReference)throw new ArgumentException("Unknown audio backend.");
        if(settings.synthesis!=Synthesis.Spectral && settings.synthesis!=Synthesis.ImpulseResponse)throw new ArgumentException("Unknown synthesis method.");
        if(settings.transport!=Transport.Direct && settings.transport!=Transport.Raytraced)throw new ArgumentException("Unknown transport model.");
        if(settings.transport==Transport.Raytraced)
        {
            if(settings.fftSize<2 || (settings.fftSize&(settings.fftSize-1))!=0 || settings.spectralHop<1 || settings.spectralHop>settings.fftSize/2 ||
                settings.fftSize%settings.spectralHop!=0 || settings.responseStepSamples%settings.spectralHop!=0 ||
                settings.RaysPerBin<1 || settings.maxBounces<1 || !float.IsFinite(settings.diffractionFactor) || settings.diffractionFactor<0)
                throw new ArgumentException("Raytraced transport needs a power-of-two FFT, a dividing hop no larger than half the FFT, response steps divisible by the hop, and positive ray/bounce counts.");
            checked { _=settings.SpectralSlotsPerPair; }
        }
    }
    static bool PositiveFinite(float value) { return value>0 && !float.IsInfinity(value); }
    internal static void WriteWaveHeader(BinaryWriter writer,int rate,long length,int channels=1)
    {
        writer.Write(System.Text.Encoding.ASCII.GetBytes("RIFF")); writer.Write((uint)(36+length*4*channels));
        writer.Write(System.Text.Encoding.ASCII.GetBytes("WAVEfmt ")); writer.Write(16);
        writer.Write((ushort)3); writer.Write((ushort)channels); writer.Write(rate); writer.Write(checked(rate*4*channels));
        writer.Write((ushort)(4*channels)); writer.Write((ushort)32);
        writer.Write(System.Text.Encoding.ASCII.GetBytes("data")); writer.Write((uint)(length*4*channels));
    }
}
