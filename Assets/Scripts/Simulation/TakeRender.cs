using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using UnityEngine;

// Small CPU direct-path baseline. Shared numeric input; no scene or physics queries.
// Reflection/energy estimators must pass their own correctness gate before export.
public static partial class TakeRender
{
    public enum DelayFilter { Linear, WindowedSinc }
    [Serializable] public sealed class Settings
    {
        public int responseStepSamples = 512;
        public float speedOfSound = 343, referenceDistance = 1, minimumDistance = .1f;
        public int tailSamples;
        public int transitionSamples;
        public DelayFilter delayFilter;
        public int sincRadius=16;
        public float maximumDelaySeconds=10;
        public int maximumSignalEvents=4096;
        public int maximumPendingObservations=16;
        public int maximumResponsesPerBatch=64;
        public int DspLatencySamples => delayFilter==DelayFilter.WindowedSinc ? sincRadius:0;
        public long HistorySamples(int rate) => checked((long)Math.Ceiling((double)maximumDelaySeconds*rate)+2L*DspLatencySamples+2);
    }

    [Serializable] public sealed class Report
    {
        public string method = "direct-opaque-v1";
        public string motion = "frozen-geometry-held-per-response-step";
        public string responseLayout = "frames: int64 sample, then source-major/listener-minor float32(delaySeconds,pressureGain)";
        public string sceneHash, takeStateHash, takeEventHash, responseHash;
        public string unityVersion = Application.unityVersion;
        public string platform = Application.platform.ToString(), processor = SystemInfo.processorType, graphicsDevice = SystemInfo.graphicsDeviceName;
        public string[] audioHashes;
        public AcousticTake.Provenance provenance;
        public string settingsHash;
        public int dspLatencySamples;
        public long pairPrimitiveTestUpperBound;
        public Settings settings;
        public int sampleRate, sourceCount, listenerCount, responseCount, objectCount, primitiveCount;
        public long outputSamples, preparedPcmBytes, workingNumericBytes;
        public double preparationSeconds,wallSeconds, solveSeconds, evaluationSeconds, audioSeconds, writeSeconds, maximumSolveSeconds;
        public string backend = "CPU";
        public double uploadSeconds, dispatchReadbackSeconds;
        public bool complete;
    }

    public struct DirectPath { public float delaySeconds, pressureGain; }

    public static void SolveDirect(AcousticTake.Scene scene, AcousticTake.Primitive[] geometry,
        AcousticTake.State[] state, Settings settings, DirectPath[] output)
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
            float distance=displacement.magnitude;
            bool blocked=false;
            foreach (var p in geometry)
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
            output[s*scene.listeners.Length+l]=new DirectPath { delaySeconds=distance/settings.speedOfSound,
                pressureGain=blocked ? 0 : settings.referenceDistance/Mathf.Max(distance,settings.minimumDistance) };
        }
        ValidatePaths(output);
    }

    internal static void ValidatePaths(DirectPath[] paths)
    {
        foreach (var p in paths)
            if (float.IsNaN(p.delaySeconds) || float.IsInfinity(p.delaySeconds) || float.IsNaN(p.pressureGain) || float.IsInfinity(p.pressureGain))
                throw new InvalidOperationException("Non-finite direct response; check scene scale and solver settings.");
    }

    static bool CrossesSegment(Vector3 o, Vector3 end, Vector3 a, Vector3 b)
    {
        var d=end-o; var edge=b-a; var q=a-o;
        double det=(double)d.x*edge.y-(double)d.y*edge.x;
        if (Math.Abs(det)<1e-12) return false;
        double t=((double)q.x*edge.y-(double)q.y*edge.x)/det;
        double u=((double)q.x*d.y-(double)q.y*d.x)/det;
        return t>1e-6 && t<1-1e-6 && u>=0 && u<=1;
    }

    static bool CrossesTriangle(Vector3 o, Vector3 direction, Vector3 a, Vector3 b, Vector3 c)
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
        if (scene.sources.Length == 0 || scene.listeners.Length == 0) throw new InvalidOperationException("Rendering requires a recorded source and listener.");
        long length=checked(metadata.durationSamples+1+settings.tailSamples);
        if (length > (uint.MaxValue-36)/4) throw new InvalidOperationException("Output exceeds the WAV size limit; render a shorter take.");
        Directory.CreateDirectory(outputDirectory);
        var report=new Report { sceneHash=metadata.sceneHash, takeStateHash=metadata.stateHash, takeEventHash=metadata.eventHash,
            settings=settings, provenance=AcousticTake.Provenance.Capture(shader), settingsHash=AcousticTake.HashText(JsonUtility.ToJson(settings)),
            dspLatencySamples=settings.DspLatencySamples, sampleRate=metadata.sampleRate, sourceCount=scene.sources.Length,
            listenerCount=scene.listeners.Length, objectCount=scene.objects.Length, primitiveCount=take.Geometry.Length, outputSamples=length };
        string reportPath=Path.Combine(outputDirectory,"run.json");
        File.WriteAllText(reportPath,JsonUtility.ToJson(report,true));
        var state=new AcousticTake.State[scene.objects.Length];
        var response=new DirectPath[scene.sources.Length*scene.listeners.Length];
        var signals=new Signal[scene.sources.Length];
        var pcm=new float[scene.clips.Length][];
        for (int c=0; c<pcm.Length; c++) { pcm[c]=take.ReadClip(c); report.preparedPcmBytes+=(long)pcm[c].Length*4; }
        for (int s=0; s<signals.Length; s++) signals[s]=Signal.FromTake(take,s,pcm[scene.sources[s].clipIndex]);
        var mixer=new Mixer(scene,metadata.sampleRate,settings,signals);
        var block=new float[checked(settings.responseStepSamples*scene.listeners.Length)];
        using (var solver=new DirectSolver(scene,take.Geometry,settings,shader))
        {
            report.workingNumericBytes=report.preparedPcmBytes+(long)take.Geometry.Length*44+(long)state.Length*80+(long)response.Length*8+(long)block.Length*4+solver.GpuBufferBytes+solver.CpuBufferBytes+mixer.NumericBytes;
            report.backend=solver.Backend;
            report.preparationSeconds=watch.Elapsed.TotalSeconds;
            var outputs=new List<BinaryWriter>();
            try
            {
                for (int l=0; l<scene.listeners.Length; l++)
                {
                    var writer=new BinaryWriter(File.Create(Path.Combine(outputDirectory,"listener-"+l+".wav")));
                    outputs.Add(writer); WriteWaveHeader(writer,(int)metadata.sampleRate,length);
                }
                using (var writer=new BinaryWriter(File.Create(Path.Combine(outputDirectory,"responses.bin"))))
                {
                    for (long sample=0; sample<length; sample+=settings.responseStepSamples)
                    {
                        long start=Stopwatch.GetTimestamp();
                        take.Evaluate(Math.Min(sample,metadata.durationSamples),state);
                        report.evaluationSeconds+=Elapsed(start);
                        start=Stopwatch.GetTimestamp();
                        solver.Solve(state,response);
                        double solveTime=Elapsed(start);
                        report.solveSeconds+=solveTime; report.maximumSolveSeconds=Math.Max(report.maximumSolveSeconds,solveTime);
                        mixer.SetResponse(sample,response,state,settings.transitionSamples>0 && take.HasCut(sample-settings.responseStepSamples,sample));
                        start=Stopwatch.GetTimestamp();
                        writer.Write(sample);
                        foreach (var path in response) { writer.Write(path.delaySeconds); writer.Write(path.pressureGain); }
                        report.responseCount++;
                        int count=(int)Math.Min(settings.responseStepSamples,length-sample);
                        report.writeSeconds+=Elapsed(start);
                        start=Stopwatch.GetTimestamp(); mixer.Render(sample,count,block);
                        report.audioSeconds+=Elapsed(start);
                        start=Stopwatch.GetTimestamp();
                        for (int n=0; n<count; n++) for (int l=0; l<outputs.Count; l++)
                            outputs[l].Write(block[n*outputs.Count+l]);
                        report.writeSeconds+=Elapsed(start);
                        progress?.Invoke((float)((double)(sample+count)/length));
                    }
                }
            }
            finally { foreach (var output in outputs) output.Dispose(); }
            report.pairPrimitiveTestUpperBound=solver.PairPrimitiveUpperBound;
            report.uploadSeconds=solver.UploadSeconds; report.dispatchReadbackSeconds=solver.DispatchReadbackSeconds;
        }
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
        public long EndSample=long.MaxValue;
        readonly List<long> gateSamples=new List<long>();
        readonly List<bool> gateEnabled=new List<bool>();
        readonly int maximumEvents;
        public int RetainedEvents => voices.Count+gateSamples.Count;
        public Signal(float[] clip,int maximumEvents=int.MaxValue) { pcm=clip; this.maximumEvents=maximumEvents; }
        void ReserveEvent() { if (RetainedEvents>=maximumEvents) throw new InvalidOperationException("Source event history capacity exceeded; increase Maximum Signal Events."); }
        public void Prune(long before)
        {
            int count=0;
            while (count+1<voices.Count && voices[count+1].sample<=before) count++;
            if (count>0) voices.RemoveRange(0,count);
            count=gateSamples.BinarySearch(before); if (count<0) count=~count-1;
            if (count>0) { gateSamples.RemoveRange(0,count); gateEnabled.RemoveRange(0,count); }
        }
        public static Signal FromTake(AcousticTake take,int source,float[] clip)
        {
            var result=new Signal(clip) {EndSample=take.Metadata.durationSamples};
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
            if (e.sample<current.sample || e.clipOffset<0 || e.clipOffset>pcm.Length || float.IsNaN(e.gain) || float.IsInfinity(e.gain))
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
            if (sample<0 || sample>EndSample || voices.Count==0 || pcm.Length==0) return 0;
            int gate=gateSamples.BinarySearch(sample); if (gate<0) gate=~gate-1;
            if (gate<0 || !gateEnabled[gate]) return 0;
            int lo=0,hi=voices.Count;
            while (lo<hi) { int mid=lo+(hi-lo)/2; if (voices[mid].sample<=sample) lo=mid+1; else hi=mid; }
            if (lo==0) return 0;
            var v=voices[lo-1]; if (!v.playing) return 0;
            long index=sample-v.start+v.offset; if (v.loop) index%=pcm.Length;
            return index<pcm.Length ? pcm[index]*v.gain:0;
        }
    }

    public sealed class Mixer
    {
        readonly AcousticTake.Scene scene;
        readonly int sampleRate,transition,taps,firstTap,latency;
        readonly float maximumDelay;
        readonly long[] delay,oldDelay;
        readonly float[] weights,oldWeights;
        readonly Signal[] signals;
        readonly DirectPath[] current,previous;
        readonly bool[] enabled,oldEnabled;
        long responseSample;
        bool initialized;
        int fade;
        public long NumericBytes => (long)(current.Length+previous.Length)*8+(long)(delay.Length+oldDelay.Length)*8+(long)(weights.Length+oldWeights.Length)*4+enabled.Length+oldEnabled.Length;
        public Mixer(AcousticTake.Scene scene,int sampleRate,Settings settings,Signal[] signals)
        {
            Validate(settings); this.scene=scene; this.sampleRate=sampleRate; this.signals=signals; transition=settings.transitionSamples;
            current=new DirectPath[scene.sources.Length*scene.listeners.Length]; previous=new DirectPath[current.Length];
            enabled=new bool[scene.listeners.Length]; oldEnabled=new bool[enabled.Length];
            latency=settings.DspLatencySamples; maximumDelay=settings.maximumDelaySeconds;
            taps=settings.delayFilter==DelayFilter.Linear ? 2:checked(settings.sincRadius*2);
            firstTap=settings.delayFilter==DelayFilter.Linear ? 0:1-settings.sincRadius;
            weights=new float[checked(current.Length*taps)]; oldWeights=new float[weights.Length];
            delay=new long[current.Length]; oldDelay=new long[delay.Length];
        }
        public void SetResponse(long sample,DirectPath[] paths,AcousticTake.State[] state,bool cut=false)
        {
            if (paths.Length!=current.Length || (initialized && sample<responseSample)) throw new ArgumentException("Invalid response order/shape.");
            ValidatePaths(paths);
            foreach (var path in paths) if (path.delaySeconds<0 || path.delaySeconds>maximumDelay)
                throw new InvalidOperationException("Response exceeds Maximum Delay Seconds; increase the declared history horizon.");
            Array.Copy(current,previous,current.Length); Array.Copy(paths,current,current.Length);
            Array.Copy(delay,oldDelay,delay.Length); Array.Copy(weights,oldWeights,weights.Length);
            for (int pair=0;pair<paths.Length;pair++)
            {
                double samples=(double)paths[pair].delaySeconds*sampleRate;
                delay[pair]=checked((long)Math.Ceiling(samples)+latency);
                double fraction=Math.Ceiling(samples)-samples, sum=0;
                for (int t=0;t<taps;t++)
                {
                    double x=fraction-(firstTap+t);
                    double weight=latency==0 ? (t==0 ? 1-fraction:fraction):Sinc(x)*Sinc(x/latency);
                    weights[pair*taps+t]=(float)weight; sum+=weight;
                }
                for (int t=0;t<taps;t++) weights[pair*taps+t]/=(float)sum;
            }
            Array.Copy(enabled,oldEnabled,enabled.Length);
            for (int l=0; l<enabled.Length; l++) enabled[l]=Enabled(state[scene.listeners[l]],AcousticTake.StateFlags.Listener);
            fade=initialized && !cut ? transition:0;
            // Discontinuous pose changes must not smear a previous scene through the new one.
            foreach (var s in state) if ((s.flags&AcousticTake.StateFlags.Cut)!=0) fade=0;
            responseSample=sample; initialized=true;
        }
        static double Sinc(double x) => Math.Abs(x)<1e-12 ? 1:Math.Sin(Math.PI*x)/(Math.PI*x);
        double Read(int source,int pair,long sample,long[] delays,float[] kernel)
        {
            long start=sample-delays[pair]+firstTap; double sum=0;
            for (int t=0;t<taps;t++) sum+=kernel[pair*taps+t]*signals[source].Read(start+t);
            return sum;
        }
        static readonly Unity.Profiling.ProfilerMarker mixMarker=new Unity.Profiling.ProfilerMarker("Acoustics.MixPCM");
        public void Render(long sample,int count,float[] interleaved)
        {
            using (mixMarker.Auto())
            {
                if (!initialized || sample<responseSample || count<0 || interleaved.Length<count*enabled.Length) throw new ArgumentException("Invalid mix block.");
                for (int n=0; n<count; n++)
                {
                    long at=sample+n;
                    float blend=fade==0 ? 1:Mathf.Clamp01((float)(at-responseSample)/fade);
                    for (int l=0; l<enabled.Length; l++)
                    {
                        double sum=0;
                        for (int s=0; s<signals.Length; s++)
                        {
                            int pair=s*enabled.Length+l;
                            var a=previous[pair]; var b=current[pair];
                            if (enabled[l] && blend>0) sum+=blend*b.pressureGain*Read(s,pair,at,delay,weights);
                            if (oldEnabled[l] && blend<1) sum+=(1-blend)*a.pressureGain*Read(s,pair,at,oldDelay,oldWeights);
                        }
                        float value=(float)sum;
                        if (float.IsNaN(value) || float.IsInfinity(value)) throw new InvalidOperationException("Non-finite mixed audio; render cannot complete.");
                        interleaved[n*enabled.Length+l]=value;
                    }
                }
            }
        }
    }
    public static bool Enabled(AcousticTake.State state,AcousticTake.StateFlags role) =>
        (state.flags&(AcousticTake.StateFlags.Active|role))==(AcousticTake.StateFlags.Active|role);
    public static double Elapsed(long start) => (double)(Stopwatch.GetTimestamp()-start)/Stopwatch.Frequency;

    public static void Validate(Settings settings)
    {
        if (settings == null || settings.responseStepSamples<=0 || settings.tailSamples<0 || settings.transitionSamples<0 || settings.transitionSamples>settings.responseStepSamples ||
            !PositiveFinite(settings.maximumDelaySeconds) || settings.maximumSignalEvents<2 || settings.maximumPendingObservations<1 || settings.maximumResponsesPerBatch<1 ||
            (settings.delayFilter!=DelayFilter.Linear && settings.delayFilter!=DelayFilter.WindowedSinc) || settings.sincRadius<2 || settings.sincRadius>int.MaxValue/2 ||
            !PositiveFinite(settings.speedOfSound) || !PositiveFinite(settings.referenceDistance) || !PositiveFinite(settings.minimumDistance))
            throw new ArgumentException("Invalid direct-render settings.");
    }
    static bool PositiveFinite(float value) { return value>0 && !float.IsInfinity(value); }
    internal static void WriteWaveHeader(BinaryWriter writer,int rate,long length)
    {
        writer.Write(System.Text.Encoding.ASCII.GetBytes("RIFF")); writer.Write((uint)(36+length*4));
        writer.Write(System.Text.Encoding.ASCII.GetBytes("WAVEfmt ")); writer.Write(16);
        writer.Write((ushort)3); writer.Write((ushort)1); writer.Write(rate); writer.Write(checked(rate*4));
        writer.Write((ushort)4); writer.Write((ushort)32);
        writer.Write(System.Text.Encoding.ASCII.GetBytes("data")); writer.Write((uint)(length*4));
    }
}
