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
    public enum ReflectionSampling { FirstImages, PrimitiveBaseline }
    // Acoustic method of the transport solver. Hybrid is the renderer (phase-exact early
    // part + energy late field). Pressure and Energy are single-loop kernels (MethodPressure.hlsl,
    // MethodEnergy.hlsl; Energy also renders audio). Path is one energy path tracer
    // (MethodPath.hlsl, docs/benchmarks/path-tracer-2026-11-12): one histogram per solve, rendered
    // by PathReverb with the exact direct arrival.
    public enum Method { Hybrid, Pressure, Energy, Path }
    [Serializable] public sealed class Settings : ISerializationCallbackReceiver
    {
        public int responseStepSamples = 512;
        [Min(0),Tooltip("Offline renders: render only the first this many seconds of the take (plus the tail); zero renders all of it.")]
        public float renderSeconds;
        public float speedOfSound = 343, referenceDistance = 1, minimumDistance = .1f;
        public int tailSamples;
        [Tooltip("Direct/IR output fade. Spectral interpolates consecutive responses at acoustic arrival time.")]
        public int transitionSamples;
        public DelayFilter delayFilter=DelayFilter.WindowedSinc;
        [Tooltip("Mono is suitable for speakers. Binaural applies ear filters for headphones.")]
        public ListenerOutput listenerOutput;
        public int ChannelsPerListener => listenerOutput==ListenerOutput.Binaural ? 2:1;
        public int sincRadius=16;
        public float maximumDelaySeconds=10;
        public int maximumSignalEvents=4096;
        public int maximumPendingObservations=16;
        public int maximumResponsesPerBatch=64;
        [Tooltip("Use local mesh hierarchies with refitted instance bounds for static and moving geometry.")]
        public bool useBvh;
        public Transport transport;
        public Synthesis synthesis=Synthesis.Spectral;
        public AudioBackend audioBackend;
        [Min(1),Tooltip("Rays per frequency bin for the PrimitiveBaseline mirror estimator; also the floor of the First Images candidate budget.")] public int raysPerBin=1;
        [Tooltip("Keep complete first-order mirror routes as delayed audio arrivals. PrimitiveBaseline preserves the old estimator for comparisons.")]
        public ReflectionSampling reflectionSampling;
        [Min(1), Tooltip("Minimum mirror candidates per emission. The candidate budget also grows with rays per bin; small scenes enumerate every primitive.")]
        public int reflectionCandidates=128;
        [Min(0),Tooltip("Complete one/two-interaction diffraction and mixed reflection paths per emission (D, DD, RD, DR, RR). Zero keeps only direct sound and first images in the early part.")]
        public int completeRouteSamples;
        internal int CompleteSamples => maxBounces>0?completeRouteSamples:0;
        [Tooltip("Opt-in level-preserving gains for the sampled early families (first-order images and complete routes), fitted offline; data in Resources/Acoustics/FamilyGains.json. Needs complete routes and First Images. Off renders the unbiased estimator.")]
        public bool familyGains;
        [NonSerialized] internal FamilyGainTable familyGainTable; // Checks only; null reads the shipped table.
        internal FamilyGainTable FamilyGains => familyGains?familyGainTable??FamilyGainTable.Shipped():null;
        internal int CompleteSlots(AcousticTake.Scene scene) => checked(CompleteSamples*(scene.dimension==AcousticTake.Dimension.Planar2D?2:1));
        [Min(0), Tooltip("Maximum interactions per path, early and late. Zero renders direct sound only, including material transmission, with no reflection or diffraction paths.")]
        public int maxBounces=5;
        [Min(1),Tooltip("Energy rays per octave band and source/listener pair for the late field. Its variance falls as one over this count.")]
        public int lateRays=512;
        [Tooltip("Receiver sphere radius (m) for late specular energy. Larger spheres lower variance and smear arrival times by up to 2r/c.")]
        public float lateReceiverRadius=.5f;
        [Min(0),Tooltip("Late-field length (s). Beyond the simulated horizon (maximum delay) each band's envelope continues its fitted exponential decay (LateTail.cs) so a long reverberation fades out instead of stopping. Zero or less than the horizon: no continuation.")]
        public float lateTailSeconds;
        [Min(0),Tooltip("Energy method audio: time constant (s) of the late field's moving average over solves, each with a new walk seed. Averages out the walk's sampling noise for a static or slowly moving listener (it smooths over the distance moved in this time); zero keeps one fixed-seed walk per solve.")]
        public float lateAccumulationSeconds;
        internal const int LateBands=8,LateDirections=6;
        // One late histogram bin per DSP partition. A partition never exceeds the
        // DSP latency (so its block convolution is causal) and divides the response
        // step (so late responses change on block boundaries).
        internal int LatePartition => Math.Min(Math.Min(fftSize,1024),responseStepSamples&-responseStepSamples);
        internal int LateBins(int rate) => maxBounces==0?0:checked((int)Math.Ceiling(maximumDelaySeconds*(double)rate/LatePartition));
        // Synthesis kernel length: the simulated bins, continued to lateTailSeconds (LateTail).
        internal int LateKernelBins(int rate) => Math.Max(LateBins(rate),maxBounces==0?0:checked((int)Math.Ceiling(lateTailSeconds*(double)rate/LatePartition)));
        // Specular order held by the phase-exact early part: first images, plus
        // second-order images in the complete routes' RR family.
        internal int LateEarlyOrder => CompleteSamples>0 && maxBounces>1?2:1;
        public int seed;
        [Tooltip("Hybrid and Energy render audio (Energy: exact discrete arrivals plus the energy walk as the late field). Pressure is a single-loop offline measurement kernel (see MethodTransport).")]
        public Method method;
        [Min(1),Tooltip("Pressure/Energy methods: rays per solve; every ray carries all eight octave bands.")]
        public int methodRays=4096;
        [Min(1e-4f),Tooltip("Pressure method: a specular chain is scored when the ray leaving its last non-specular vertex passes within this distance (m) of the listener image. Energy method: radius (m) of the receiver sphere that scores the walk's specular segments (the diffuse field).")]
        public float specularDetectionRadius=.25f;
        [Range(0,1),Tooltip("Pressure/Energy methods: probability that a path continues through an edge rather than along a sampled direction.")]
        public float methodEdgeProbability=.5f;
        [Range(0,1),Tooltip("Energy method: Russian roulette starts only once a path's throughput (energy fraction over its sampling probability) falls below this floor, so the number of paths stays constant over the first 60 dB of decay. 1 terminates paths at the rate their energy decays, which starves the late tail.")]
        public float methodRouletteFloor=1e-6f;
        [Tooltip("Energy method: enumerate the direct sound, pure specular paths up to methodArrivalOrder and the D/DD/RD/DR edge routes from the source as exact discrete arrivals; the walk then holds everything else (the diffuse field). Off: the walk estimates every path.")]
        public bool methodArrivals=true;
        [Range(0,2),Tooltip("Energy method: highest specular order enumerated as discrete arrivals (bounded by the scene's plane count).")]
        public int methodArrivalOrder=2;
        [Min(1),Tooltip("Energy method audio: discrete arrivals per source/listener pair rendered as delayed paths (the strongest); the rest join the diffuse field.")]
        public int methodArrivalSlots=64;
        [Min(0),Tooltip("Energy method: in scenes with too many planes to pair all of them, enumerate the second-order reflections of this many planes per solve, those most important to the source and listener (area over squared distance); the walk keeps every other second-order path. Zero: second order only when every plane pair fits.")]
        public int methodNearPlanes;
        [Min(1),Tooltip("Pressure/Energy methods: interaction cap per ray (a safety bound; the 2 s output horizon normally ends paths first).")]
        public int methodMaxEvents=1<<16;
        [Min(1e-6f),Tooltip("Path method: width (s) of the histogram's fine time bins, used up to pathFineBins of them; also the impulse spacing of its audio synthesis.")]
        public float pathBinSeconds=.00025f;
        [Min(1),Tooltip("Path method: number of fine bins; after them each bin is 1/pathFineBins of its start time (constant relative resolution).")]
        public int pathFineBins=64;
        [Min(1),Tooltip("Path method: cap on the geometric trials that measure the share of a detection region following a specular chain (expected 1/share; a capped detection is counted and biased low).")]
        public int pathMaxTrials=64;
        [Min(0),Tooltip("Path method: survival in time. A path is alive at time t with probability min(1, this / t), weight-compensated (unbiased), so each histogram bin keeps about as many contributing paths while late work falls sharply. Zero: every path lives to the horizon (roulette on energy only).")]
        public float pathSurvivalSeconds=.1f;
        // Exact arrivals rendered as delayed paths per pair: Energy's slots, Path's direct sound.
        internal int ArrivalSlots => method==Method.Path?1:methodArrivalSlots;
        [HideInInspector] public int raysPerPair; // Read older serialized total-ray settings.
        public int FrequencyBins => fftSize/2+1;
        public int RaysPerBin => raysPerPair>0 ? checked((raysPerPair-1)/FrequencyBins+1):raysPerBin;
        public void OnBeforeSerialize() { }
        public void OnAfterDeserialize()
        {
            if(raysPerPair>0 && FrequencyBins>0){raysPerBin=RaysPerBin;raysPerPair=0;}
        }
        // Sampled mirror rays of the PrimitiveBaseline estimator; First Images enumerate candidates instead.
        public int RaysPerPair => maxBounces==0 || reflectionSampling!=ReflectionSampling.PrimitiveBaseline ? 0:checked(FrequencyBins*RaysPerBin);
        public int PathSlotsPerRay => reflectionSampling==ReflectionSampling.PrimitiveBaseline?1:0;
        // Geometry-free point-source capacity. Use SlotsPerPair(scene) or Layout
        // for real scenes, whose complete mirrors have their own route range.
        public int SpectralSlotsPerPair => checked(FrequencyBins+RaysPerPair*PathSlotsPerRay);
        internal int MirrorCandidates(AcousticTake.Scene scene)
        {
            if(maxBounces==0 || reflectionSampling==ReflectionSampling.PrimitiveBaseline)return 0;
            int primitives=0;foreach(var obj in scene.objects)primitives=checked(primitives+obj.primitiveCount);
            return Math.Min(primitives,Math.Max(reflectionCandidates,RaysPerBin));
        }
        internal int SourceRaysPerBin(AcousticTake.Scene scene,int source)
        {
            var plane=scene.planes==null?null:Array.Find(scene.planes,p=>p.source==source);
            // Every retained event needs an indirect launch. Keep that coverage
            // local to its source instead of inflating unrelated pairs.
            return plane!=null&&plane.generatorVersion!=0?Math.Max(RaysPerBin,plane.samples):RaysPerBin;
        }
        public int SlotsPerPair(AcousticTake.Scene scene)
        {
            int maximum=0;
            for(int s=0;s<scene.sources.Length;s++)
            {
                int sites=scene.planes==null?1:Array.Find(scene.planes,p=>p.source==s)?.samples??1;
                maximum=Math.Max(maximum,checked(FrequencyBins*(sites*(1+MirrorCandidates(scene)+CompleteSlots(scene))+(maxBounces==0?0:SourceRaysPerBin(scene,s)*PathSlotsPerRay))));
            }
            return maximum;
        }
        [Tooltip("Signed rigid-wedge diffraction in the complete early routes: 1 uses the wedge model; 0 disables diffraction. The late energy field has no diffraction.")]
        [Min(0)] public float diffractionFactor=1;
        public bool airAbsorption=true;
        [Min(2)] public int fftSize=1024,spectralHop=256;
        public int DspLatencySamples => (transport==Transport.Raytraced ? fftSize:0)+(delayFilter==DelayFilter.WindowedSinc ? sincRadius:0);
        internal long ResponseSample(long outputSample) => transport==Transport.Raytraced && synthesis==Synthesis.Spectral ? Math.Max(0,outputSample+responseStepSamples-DspLatencySamples):outputSample;
        public long HistorySamples(int rate) => checked((long)Math.Ceiling((double)maximumDelaySeconds*rate)+2L*DspLatencySamples+2+(listenerOutput==ListenerOutput.Binaural ? Hrtf.Load(rate).Taps-1:0));
    }

    [Serializable] public sealed class AudioDiagnostics
    {
        public long responseUpdates,reusedResponses,filterBytes,materialFilters,inputFfts;
        public int pathCapacity,activeGroups,peakGroups;
        // Request-to-callback includes GPU/readback and player-loop scheduling, not just GPU execution.
        public int readbacks;
        public double readbackSeconds,maximumReadbackSeconds,maximumConsumeDelaySeconds;
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
        public string familyGainsHash; // SHA-256 of the gain table text when settings.familyGains is on.
        public int dspLatencySamples;
        public long responseSupportSamples;
        public bool tailCoversResponseSupport;
        public long audioUploadBytes,audioReadbackBytes;
        public AudioDiagnostics gpuAudio;
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
        public int bvhUpdates,bvhPrimitives;
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
        long duration=settings.renderSeconds>0?Math.Min(metadata.durationSamples,(long)Math.Ceiling(settings.renderSeconds*(double)metadata.sampleRate)):metadata.durationSamples;
        long length=checked(duration+1+settings.tailSamples);
        if (length > (uint.MaxValue-36)/(4L*settings.ChannelsPerListener)) throw new InvalidOperationException("Output exceeds the WAV size limit; render a shorter take.");
        Directory.CreateDirectory(outputDirectory);
        var report=new Report {takeSourceHash=AcousticTake.SourceHash(metadata), sceneHash=metadata.sceneHash, takeStateHash=metadata.stateHash, takeEventHash=metadata.eventHash,
            settings=settings, provenance=AcousticTake.Provenance.Capture(shader), settingsHash=AcousticTake.HashText(JsonUtility.ToJson(settings)),
            familyGainsHash=settings.FamilyGains?.hash,
            dspLatencySamples=settings.DspLatencySamples, sampleRate=metadata.sampleRate, sourceCount=scene.sources.Length,
            listenerCount=scene.listeners.Length, objectCount=scene.objects.Length, primitiveCount=take.Geometry.Length, outputSamples=length,channelsPerListener=settings.ChannelsPerListener };
        if(settings.transport==Transport.Raytraced)
        {
            report.method=settings.CompleteSamples>0?"complete-early-routes-late-energy-v14":"first-images-late-energy-v14";report.provenance.solverContract=report.method;report.provenance.dspContract=settings.synthesis==Synthesis.ImpulseResponse?"partitioned-complex-fir-hrtf-late-noise-v3":"path-clock-complex-spectral-late-noise-v5";
            if(useGpuAudio)
            {
                report.provenance.dspContract=settings.synthesis==Synthesis.ImpulseResponse?"gpu-complex-path-filter-overlap-save-late-noise-v3":"gpu-path-clock-complex-spectral-late-noise-v5";
                report.audioShaderDependencyHash=AcousticTake.Provenance.Capture(Resources.Load<ComputeShader>("Acoustics/GpuAudio")).shaderDependencyHash;
            }
            report.responseLayout="Direct opaque diagnostic probes only; indirect and transmitted arrivals are synthesized into WAVs. "+report.responseLayout;
            report.transportDescription="Synthesis="+settings.synthesis+". Phase-exact early sound: retarded direct arrivals with material transmission, first-order image sources (selection="+settings.reflectionSampling+", candidate budget floor="+settings.reflectionCandidates+"; First Images keep each candidate as a separate delayed arrival) and, with complete early-route samples="+settings.CompleteSamples+", the wave-tuned signed one/two-edge integral for D/DD/RD/DR/RR. Late field: incoherent energy from "+settings.lateRays+" rays per octave band (8 bands, 125 Hz-16 kHz) with deterministic absorption/air loss and transmission/diffuse/specular continuations chosen by energy fraction, detected by diffuse rain and a "+settings.lateReceiverRadius+" m receiver sphere into a head-frame six-direction energy histogram; paths held by the early part (direct/transmitted-only and unobstructed specular paths up to order "+settings.LateEarlyOrder+") are excluded and the late field has no diffraction. It is synthesized as seeded band-split noise shaped by the histogram envelope (Polack), with diffuse-field interaural coherence for binaural output, by partitioned convolution of each source's emitted signal. First mirrors use retarded image sources with locally constant rigid velocity; other geometry is frozen at solve time. Not a validated wave reference.";
            if(settings.method==Method.Path)
            {
                report.method=report.provenance.solverContract="path-histogram-v1";
                report.transportDescription="Synthesis="+settings.synthesis+". Path method (MethodPath.hlsl): one energy path tracer with "+settings.methodRays+" rays per solve and a new seed every solve; next-event connections at the source and scattering points (straight, through walls, and through edges), UTD edge rays from the source, and specular chains connected by exact image retrace over a "+settings.specularDetectionRadius+" m detection region whose probability is measured by geometric trials (unbiased). Output: one histogram per solve (six head-frame directions, eight octave bands, "+settings.pathBinSeconds*1000+" ms bins up to "+settings.pathFineBins+" of them, then 1/"+settings.pathFineBins+" relative resolution, to "+settings.maximumDelaySeconds+" s). Audio: the exact direct arrival as a delayed path with Doppler; everything else from the histogram as fixed random-sign impulses per fine slot with amplitude sqrt(energy), band-filtered, with diffuse-field equalised ear band energies and interaural coherence, crossfaded per update (PathReverb, CPU). Not a validated wave reference.";
            }
            if(settings.method==Method.Energy)
            {
                report.method=report.provenance.solverContract="energy-arrivals-diffuse-v1";
                report.transportDescription="Synthesis="+settings.synthesis+". Energy method (MethodEnergy.hlsl): discrete arrivals enumerated exactly from the source (direct sound with transmission, pure specular paths up to order "+settings.methodArrivalOrder+", UTD edge routes D/DD/RD/DR), each with exact delay, band energies, direction and delay rate, rendered as delayed paths with gain sqrt(band energy) and no phase (the strongest "+settings.methodArrivalSlots+" per pair); everything else (scattered at least once, higher specular orders, edge rays) is a "+settings.methodRays+"-ray energy walk scored by a "+settings.specularDetectionRadius+" m receiver sphere into the head-frame late histogram, synthesized as the hybrid's late noise field. Russian roulette floor "+settings.methodRouletteFloor+". Not a validated wave reference.";
            }
            if(settings.maxBounces==0)
                report.transportDescription="Synthesis="+settings.synthesis+". Direct sound only: retarded direct arrivals, occlusion, material transmission/interior travel and air/material attenuation. No reflected, scattered or diffracted paths are allocated or traced. Existing source recordings may contain their own reverberation.";
        }
        if(settings.listenerOutput==ListenerOutput.Binaural)
        {var hrtf=Hrtf.Load(metadata.sampleRate); report.hrtfDataset=Hrtf.Dataset; report.hrtfHash=hrtf.Hash; report.hrtfTaps=hrtf.Taps;}
        string reportPath=Path.Combine(outputDirectory,"run.json");
        File.WriteAllText(reportPath,JsonUtility.ToJson(report,true));
        var state=new AcousticTake.State[scene.objects.Length];
        var headStates=settings.ChannelsPerListener==2?new AcousticTake.State[state.Length]:null;
        var response=new DirectPath[scene.sources.Length*scene.listeners.Length];
        var pcm=new float[scene.clips.Length][];
        for (int c=0; c<pcm.Length; c++) { pcm[c]=take.ReadClip(c); report.preparedPcmBytes+=(long)pcm[c].Length*4; }
        var signals=Signal.FromTake(take,pcm);
        void EvaluateEmission(long birth,AcousticTake.State[] target) => take.Evaluate(Math.Min(birth,metadata.durationSamples),target);
        int batchCapacity=shader==null || !SystemInfo.supportsAsyncGPUReadback ? 1:(int)Math.Min(settings.maximumResponsesPerBatch,(length-1)/settings.responseStepSamples+1);
        using var gpuAudio=useGpuAudio?new GpuAudio(scene,metadata.sampleRate,settings,signals,batchCapacity,transported:true):null;
        var mixer=useGpuAudio?null:new Mixer(scene,metadata.sampleRate,settings,signals);
        long initialMixerBytes=gpuAudio?.NumericBytes??mixer.NumericBytes;
        report.backendScope=useGpuAudio?"GPU transport and audio synthesis; CPU source timeline and PCM output":"Transport backend; CPU reference audio synthesis";
        var block=new float[checked(settings.responseStepSamples*scene.listeners.Length*settings.ChannelsPerListener*(useGpuAudio?batchCapacity:1))];
        var batchStates=batchCapacity==1 ? state:new AcousticTake.State[checked(state.Length*batchCapacity)];
        var batchPaths=batchCapacity==1 ? response:new DirectPath[checked(response.Length*batchCapacity)];
        report.responseBatchCapacity=batchCapacity;
        using (var solver=new TransportSolver(scene,take.Geometry,settings,shader,batchCapacity,metadata.sampleRate,!useGpuAudio,methodAudio:true))
        {
            report.workingNumericBytes=report.preparedPcmBytes+(long)take.Geometry.Length*44+((long)state.Length+(headStates?.Length??0))*80+(long)response.Length*DirectPathStride+(long)block.Length*4+solver.GpuBufferBytes+solver.CpuBufferBytes+initialMixerBytes+
                (batchCapacity==1?0:(long)batchStates.Length*80+(long)batchPaths.Length*DirectPathStride);
            report.backend=solver.Backend;
            report.preparationSeconds=watch.Elapsed.TotalSeconds;
            var outputs=new List<BinaryWriter>();
            void WriteAudio(int count)
            {
                long start=Stopwatch.GetTimestamp();
                for(int n=0;n<count;n++)for(int l=0;l<outputs.Count;l++)for(int ear=0;ear<settings.ChannelsPerListener;ear++)
                    outputs[l].Write(block[(n*outputs.Count+l)*settings.ChannelsPerListener+ear]);
                report.writeSeconds+=Elapsed(start);
            }
            try
            {
                for (int l=0; l<scene.listeners.Length; l++)
                {
                    var writer=new BinaryWriter(File.Create(Path.Combine(outputDirectory,"listener-"+l+".wav")));
                    outputs.Add(writer); WriteWaveHeader(writer,(int)metadata.sampleRate,length,settings.ChannelsPerListener);
                }
                using (var writer=new BinaryWriter(File.Create(Path.Combine(outputDirectory,"responses.bin"))))
                {
                    if(settings.ResponseSample(0)>0)
                    {
                        take.Evaluate(0,state);solver.PrepareEmissions(signals,0,1,EvaluateEmission);solver.Solve(state,response);
                        if(useGpuAudio){gpuAudio.SetResponse(0,true,solver.RayBuffer,0,scene,state,0,late:solver.Late(0));gpuAudio.CompletePreparation();}
                        else mixer.SetResponse(0,response,state,true,solver.RayPaths,0,0,late:solver.Late(0));
                        writer.Write(0L);foreach(var path in response)WritePath(writer,path);report.responseCount++;
                    }
                    for (long first=0; first<length; first+=(long)batchCapacity*settings.responseStepSamples)
                    {
                        int frames=(int)Math.Min(batchCapacity,(length-first-1)/settings.responseStepSamples+1);
                        long start=Stopwatch.GetTimestamp();
                        for(int f=0;f<frames;f++)
                        {
                            long anchor=settings.ResponseSample(first+(long)f*settings.responseStepSamples);
                            take.Evaluate(Math.Min(anchor,metadata.durationSamples),state);
                            if(anchor>metadata.durationSamples)FreezeMotion(state);
                            if(batchCapacity>1)Array.Copy(state,0,batchStates,f*state.Length,state.Length);
                        }
                        report.evaluationSeconds+=Elapsed(start);
                        start=Stopwatch.GetTimestamp();
                        solver.PrepareEmissions(signals,first,frames,EvaluateEmission);
                        if(batchCapacity==1)solver.Solve(state,response);
                        else
                        {
                            solver.BeginSolve(batchStates,frames);
                            if(!solver.TryComplete(batchPaths,true))throw new InvalidOperationException("Offline response batch did not complete.");
                        }
                        double solveTime=Elapsed(start);
                        report.solveSeconds+=solveTime; report.maximumSolveSeconds=Math.Max(report.maximumSolveSeconds,solveTime);
                        int batchSamples=(int)Math.Min(length-first,(long)frames*settings.responseStepSamples);
                        start=Stopwatch.GetTimestamp();gpuAudio?.PrepareInput(first+batchSamples);report.audioSeconds+=Elapsed(start);
                        for(int f=0;f<frames;f++)
                        {
                            long sample=first+(long)f*settings.responseStepSamples;
                            if(batchCapacity>1) {Array.Copy(batchStates,f*state.Length,state,0,state.Length);Array.Copy(batchPaths,f*response.Length,response,0,response.Length);}
                            start=Stopwatch.GetTimestamp();
                            long anchor=settings.ResponseSample(sample);
                            if(headStates!=null){take.Evaluate(Math.Min(sample,metadata.durationSamples),headStates);if(sample>metadata.durationSamples)FreezeMotion(headStates);}
                            bool cut=(settings.transport==Transport.Raytraced || settings.transitionSamples>0) && take.HasCut(settings.ResponseSample(sample-settings.responseStepSamples),anchor);
                            if(useGpuAudio)gpuAudio.SetResponse(sample,cut,solver.RayBuffer,f*solver.RayPathsPerFrame,scene,state,anchor,headStates,solver.Late(f));
                            else mixer.SetResponse(sample,response,state,cut,solver.RayPaths,f*solver.RayPathsPerFrame,anchor,headStates,solver.Late(f));
                            double responseTime=Elapsed(start);report.audioSeconds+=responseTime;report.responsePreparationSeconds+=responseTime;
                            start=Stopwatch.GetTimestamp();
                            writer.Write(anchor);
                            foreach (var path in response) WritePath(writer,path);
                            report.responseCount++;
                            int count=(int)Math.Min(settings.responseStepSamples,length-sample);
                            report.writeSeconds+=Elapsed(start);
                            start=Stopwatch.GetTimestamp(); if(useGpuAudio)gpuAudio.QueueRender(sample,count);else mixer.Render(sample,count,block);
                            report.audioSeconds+=Elapsed(start);
                            if(!useGpuAudio)WriteAudio(count);
                        }
                        if(useGpuAudio)
                        {
                            start=Stopwatch.GetTimestamp();gpuAudio.BeginReadback();gpuAudio.TryComplete(block,true);report.audioSeconds+=Elapsed(start);
                            WriteAudio(batchSamples);
                        }
                        progress?.Invoke((float)((double)(first+batchSamples)/length));
                    }
                }
            }
            finally { foreach (var output in outputs) output.Dispose(); }
            report.pairPrimitiveTestUpperBound=solver.PairPrimitiveUpperBound;
            report.uploadSeconds=solver.UploadSeconds; report.dispatchReadbackSeconds=solver.DispatchReadbackSeconds;
            report.bvhBuildSeconds=solver.BvhBuildSeconds;report.bvhUpdateSeconds=solver.BvhUpdateSeconds;report.bvhUpdates=solver.BvhUpdates;report.bvhPrimitives=solver.BvhPrimitiveCount;
        }
        report.workingNumericBytes+=(gpuAudio?.NumericBytes??mixer.NumericBytes)-initialMixerBytes;
        report.responseSupportSamples=gpuAudio?.ResponseSupportSamples??mixer.ResponseSupportSamples;
        report.audioUploadBytes=gpuAudio?.UploadedBytes??0;report.audioReadbackBytes=gpuAudio?.ReadbackBytes??0;
        report.gpuAudio=gpuAudio?.Diagnostics;
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
        internal struct ClipClock { public uint index; public float gain; public uint epochLow,epochHigh; }
        internal int BurstDuration;
        internal bool OneShot;
        struct Voice { public long sample, start, offset; public bool playing, loop; public float gain; }
        readonly List<Voice> voices=new List<Voice>();
        readonly float[] pcm;
        internal float[] Clip => pcm;
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
        public static Signal[] FromTake(AcousticTake take,float[][] clips)
        {
            var sources=take.Metadata.scene.sources;var result=new Signal[sources.Length];
            for(int i=0;i<result.Length;i++)
            {result[i]=sources[i].streamed?new Signal(take.ReadStream(i)):new Signal(clips[sources[i].clipIndex]);result[i].EndSample=take.Metadata.durationSamples;}
            var state=new AcousticTake.State[take.Metadata.scene.objects.Length];
            for(int f=0;f<take.Metadata.frameCount;f++)
            {
                take.ReadStoredFrame(f,state);
                for(int i=0;i<result.Length;i++)result[i].SetActive(take.FrameSample(f),Enabled(state[sources[i].objectIndex],AcousticTake.StateFlags.Source));
            }
            foreach(var e in take.Events)result[e.sourceIndex].Append(e);
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
        internal IEnumerable<(long start,long end)> Epochs(long from,long to)
        {
            for(int i=0;i<voices.Count;)
            {
                var voice=voices[i++];if(!voice.playing)continue;
                while(i<voices.Count && voices[i].playing && voices[i].start==voice.start)i++;
                long end=Math.Min(i<voices.Count?voices[i].sample:long.MaxValue,EndSample==long.MaxValue?long.MaxValue:EndSample+1);
                if(end>from && voice.start<to)yield return (voice.start,end);
            }
        }
        public float Read(double sample,uint phase=0,long birth=0)
        {
            long i=(long)Math.Floor(sample); float f=(float)(sample-i);
            return Mathf.LerpUnclamped(Read(i,phase,birth),Read(i+1,phase,birth),f);
        }
        internal float Read(long sample,uint phase=0,long birth=0)
        {
            if(!VoiceAt(sample,out var v))return 0;
            if(stream!=null)return stream.Read(sample)*v.gain;
            if((phase&Emission.Burst)!=0)
            {
                long age=sample-birth;
                if(v.start>birth || age<0 || age>=BurstDuration)return 0;
                if(OneShot)return age<pcm.Length?pcm[age]*v.gain:0;
                return pcm[(age+v.offset+(phase&~Emission.Burst))%pcm.Length]*v.gain*Emission.Window(age,BurstDuration);
            }
            var clock=Clock(sample,v);
            if(clock.gain==0)return 0;
            long index=clock.index;
            if(phase!=0)index=(index+((long)phase*pcm.Length>>16))%pcm.Length;
            return pcm[index]*clock.gain;
        }
        internal ClipClock ReadClipClock(long sample)
        {
            if(!VoiceAt(sample,out var v))return default;
            var clock=BurstDuration==0?Clock(sample,v):new ClipClock{index=(uint)v.offset,gain=v.gain};
            clock.epochLow=(uint)v.start;clock.epochHigh=(uint)(v.start>>32);return clock;
        }
        ClipClock Clock(long sample,Voice v)
        {
            long index=sample-v.start+v.offset;if(v.loop)index%=pcm.Length;
            return index<pcm.Length?new ClipClock{index=(uint)index,gain=v.gain}:default;
        }
        bool VoiceAt(long sample,out Voice voice)
        {
            voice=default;
            if (sample<0 || sample>EndSample || voices.Count==0 || (stream==null && pcm.Length==0)) return false;
            int gate=gateSamples.BinarySearch(sample); if (gate<0) gate=~gate-1;
            if (gate<0 || !gateEnabled[gate]) return false;
            int lo=0,hi=voices.Count;
            while (lo<hi) { int mid=lo+(hi-lo)/2; if (voices[mid].sample<=sample) lo=mid+1; else hi=mid; }
            if (lo==0) return false;
            voice=voices[lo-1];return voice.playing;
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
        readonly ImpulseResponseAudio ir;
        readonly LateReverb late;
        readonly PathReverb pathLate;
        readonly TakeRender.DelayFilter delayFilter;
        readonly int sincRadius;
        readonly Hrtf.Blend[] earBlend,oldEarBlend;
        long responseSample,previousResponseSample;
        bool initialized;
        int fade;
        long directSupport;
        readonly long lateSupport;
        public long ResponseSupportSamples => Math.Max(lateSupport,spectral?.ResponseSupportSamples??ir?.ResponseSupportSamples??directSupport);
        public long NumericBytes => (late?.NumericBytes??0)+(pathLate?.NumericBytes??0)+(spectral?.NumericBytes??ir?.NumericBytes??(long)(current.Length+previous.Length)*DirectPathStride+(long)(delay.Length+oldDelay.Length)*8+
            (long)(earBlend.Length+oldEarBlend.Length)*32+(long)(weights.Length+oldWeights.Length+delayWeights.Length+input.Length)*4+(long)mixed.Length*8+enabled.Length+oldEnabled.Length+(hrtf?.NumericBytes ?? 0));
        public Mixer(AcousticTake.Scene scene,int sampleRate,Settings settings,Signal[] signals)
        {
            AudioSourcePlane.ValidateRender(scene,settings);
            Validate(settings); this.scene=scene; this.sampleRate=sampleRate; this.signals=signals; transition=settings.transitionSamples;
            channels=settings.ChannelsPerListener; maximumBlock=settings.responseStepSamples;
            delayFilter=settings.delayFilter;sincRadius=settings.sincRadius;
            if(settings.transport==Transport.Raytraced)
            {
                if(settings.synthesis==Synthesis.ImpulseResponse)ir=new ImpulseResponseAudio(scene,sampleRate,settings,signals);
                else spectral=new SpectralAudio(scene,sampleRate,settings,signals);
                if(settings.maxBounces>0 && settings.method==Method.Path)
                {
                    pathLate=new PathReverb(scene,sampleRate,settings,signals);lateSupport=pathLate.SupportSamples;
                }
                else if(settings.maxBounces>0)
                {
                    Emission.Configure(scene,signals);late=new LateReverb(scene,sampleRate,settings,signals);
                    lateSupport=(long)settings.LateKernelBins(sampleRate)*settings.LatePartition+settings.DspLatencySamples;
                }
                return;
            }
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
        }
        public void SetResponse(long sample,DirectPath[] paths,AcousticTake.State[] state,bool cut=false) => SetResponse(sample,paths,state,cut,null,0);
        // late: the solved frame's late energy; default renders no late field.
        internal void SetResponse(long sample,DirectPath[] paths,AcousticTake.State[] state,bool cut,RayTransport.Path[] arrivals,int offset,long? anchor=null,AcousticTake.State[] headStates=null,RayTransport.LateFrame late=default)
        {
            if(paths.Length!=signals.Length*scene.listeners.Length || state.Length!=scene.objects.Length || (initialized && sample<responseSample)) throw new ArgumentException("Invalid response order/shape.");
            if(spectral!=null || ir!=null)
            {
                if(spectral!=null)spectral.SetResponse(sample,cut,arrivals,offset,scene,state,anchor,headStates);else ir.SetResponse(sample,cut,arrivals,offset,scene,state);
                if(this.late!=null || pathLate!=null)
                {
                    bool lateCut=cut;foreach(var s in state)lateCut|=(s.flags&AcousticTake.StateFlags.Cut)!=0;
                    if(pathLate!=null)pathLate.SetResponse(sample,lateCut,PathReverb.Histogram(late),state);else this.late.SetResponse(sample,lateCut,late,state);
                }
                responseSample=sample;initialized=true;return;
            }
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
                if(!initialized || sample<responseSample || count<0 || count>maximumBlock || interleaved.Length<(long)count*scene.listeners.Length*channels) throw new ArgumentException("Invalid mix block.");
                if(spectral!=null || ir!=null)
                {
                    if(spectral!=null)spectral.Render(sample,count,interleaved);else ir.Render(sample,count,interleaved);
                    late?.Render(sample,count,interleaved);pathLate?.Render(sample,count,interleaved);
                    for(int i=0;i<count*scene.listeners.Length*channels;i++)if(!float.IsFinite(interleaved[i]))throw new InvalidOperationException("Non-finite late audio.");
                    return;
                }
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

    // Audio is rendered by the hybrid or the energy method (discrete arrivals as delayed
    // paths, the diffuse histogram as the late field); Pressure is an offline measurement
    // kernel whose band responses are read from TransportSolver.Method.
    public static void RequireAudioMethod(Settings settings)
    {
        if(settings.method==Method.Pressure)
            throw new NotSupportedException($"Audio rendering uses the hybrid, energy or path method; {settings.method} is an offline measurement kernel.");
        if((settings.method==Method.Energy || settings.method==Method.Path) && settings.transport!=Transport.Raytraced)
            throw new NotSupportedException("The energy and path methods render through raytraced transport.");
        if(settings.method==Method.Path && settings.synthesis!=Synthesis.Spectral)
            throw new NotSupportedException("The path method renders with spectral synthesis.");
    }
    public static void Validate(Settings settings)
    {
        if (settings == null || (settings.listenerOutput!=ListenerOutput.Mono && settings.listenerOutput!=ListenerOutput.Binaural) || settings.responseStepSamples<=0 || settings.tailSamples<0 || settings.transitionSamples<0 || settings.transitionSamples>settings.responseStepSamples ||
            !PositiveFinite(settings.maximumDelaySeconds) || settings.maximumSignalEvents<2 || settings.maximumPendingObservations<1 || settings.maximumResponsesPerBatch<1 ||
            (settings.delayFilter!=DelayFilter.Linear && settings.delayFilter!=DelayFilter.WindowedSinc) || settings.sincRadius<2 || settings.sincRadius>int.MaxValue/2 ||
            !PositiveFinite(settings.speedOfSound) || !PositiveFinite(settings.referenceDistance) || !PositiveFinite(settings.minimumDistance) ||
            !(settings.lateTailSeconds>=0 && settings.lateTailSeconds<=600) || !(settings.lateAccumulationSeconds>=0 && settings.lateAccumulationSeconds<=600))
            throw new ArgumentException("Invalid render settings.");
        if(settings.audioBackend!=AudioBackend.Gpu && settings.audioBackend!=AudioBackend.CpuReference)throw new ArgumentException("Unknown audio backend.");
        if(settings.synthesis!=Synthesis.Spectral && settings.synthesis!=Synthesis.ImpulseResponse)throw new ArgumentException("Unknown synthesis method.");
        if(settings.transport!=Transport.Direct && settings.transport!=Transport.Raytraced)throw new ArgumentException("Unknown transport model.");
        if(settings.method!=Method.Hybrid && settings.method!=Method.Pressure && settings.method!=Method.Energy && settings.method!=Method.Path)throw new ArgumentException("Unknown acoustic method.");
        if(settings.method==Method.Path && (!PositiveFinite(settings.pathBinSeconds) || settings.pathFineBins<1 || settings.pathMaxTrials<1 || !(settings.pathSurvivalSeconds>=0)))throw new ArgumentException("Invalid path method settings.");
        if(settings.method!=Method.Hybrid && (settings.methodRays<1 || !PositiveFinite(settings.specularDetectionRadius) || !(settings.methodEdgeProbability>=0 && settings.methodEdgeProbability<1) || settings.methodMaxEvents<1 || !(settings.methodRouletteFloor>0 && settings.methodRouletteFloor<=1) ||
            settings.methodArrivalOrder<0 || settings.methodArrivalOrder>2 || settings.methodArrivalSlots<1 || settings.methodNearPlanes<0))
            throw new ArgumentException("Invalid single-loop method settings.");
        ValidateFamilyGains(settings);
        if(settings.transport==Transport.Raytraced)
        {
            if(settings.completeRouteSamples<0 || settings.reflectionCandidates<1 || (settings.reflectionSampling!=ReflectionSampling.FirstImages && settings.reflectionSampling!=ReflectionSampling.PrimitiveBaseline))
                throw new ArgumentException("Invalid mirror route sampling settings.");
            if(settings.fftSize<2 || (settings.fftSize&(settings.fftSize-1))!=0 || settings.spectralHop<1 || settings.spectralHop>settings.fftSize/2 ||
                settings.fftSize%settings.spectralHop!=0 || settings.responseStepSamples%settings.spectralHop!=0 ||
                settings.RaysPerBin<1 || settings.maxBounces<0 || !float.IsFinite(settings.diffractionFactor) || settings.diffractionFactor<0 ||
                settings.lateRays<1 || !PositiveFinite(settings.lateReceiverRadius))
                throw new ArgumentException("Raytraced transport needs a power-of-two FFT, a dividing hop no larger than half the FFT, response steps divisible by the hop, positive ray counts, a positive late receiver radius and nonnegative bounce counts (zero is direct sound only).");
            checked { _=settings.SpectralSlotsPerPair; }
        }
    }
    static bool PositiveFinite(float value) { return value>0 && !float.IsInfinity(value); }
    // The gains were fitted for the complete-route families together with first-image
    // mirrors; primitive-baseline mirrors use a different estimator in those slots.
    internal static void ValidateFamilyGains(Settings settings)
    {
        if(settings.familyGains && (settings.transport!=Transport.Raytraced || settings.CompleteSamples==0 || settings.reflectionSampling!=ReflectionSampling.FirstImages))
            throw new ArgumentException("Family gains need raytraced transport with complete routes (completeRouteSamples > 0 with bounces) and First Images reflection sampling.");
    }
    internal static void WriteWaveHeader(BinaryWriter writer,int rate,long length,int channels=1)
    {
        writer.Write(System.Text.Encoding.ASCII.GetBytes("RIFF")); writer.Write((uint)(36+length*4*channels));
        writer.Write(System.Text.Encoding.ASCII.GetBytes("WAVEfmt ")); writer.Write(16);
        writer.Write((ushort)3); writer.Write((ushort)channels); writer.Write(rate); writer.Write(checked(rate*4*channels));
        writer.Write((ushort)(4*channels)); writer.Write((ushort)32);
        writer.Write(System.Text.Encoding.ASCII.GetBytes("data")); writer.Write((uint)(length*4*channels));
    }
}
