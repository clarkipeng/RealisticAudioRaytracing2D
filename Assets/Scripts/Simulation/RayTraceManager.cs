using System;
using System.Collections.Generic;
using System.IO;
using UnityEngine;

[DefaultExecutionOrder(10000)]
public class RayTraceManager : MonoBehaviour
{
    public AudioManager audioManager;
    public int sampleRate = 48000;

    void OnDestroy() => DisposeTakeSession();

    public static float[] LoadSample(AudioClip clip,int sampleRate)
    {
        float[] raw = new float[clip.samples * clip.channels];
        if (!clip.GetData(raw, 0))
            throw new System.InvalidOperationException($"Cannot read PCM from {clip.name}. Enable Decompress On Load and preload the clip before recording.");

        float[] mono = new float[clip.samples];
        for (int i = 0; i < clip.samples; i++)
        {
            float sum = 0;
            for (int c = 0; c < clip.channels; c++) sum += raw[i * clip.channels + c];
            mono[i] = sum / clip.channels;
        }

        if (clip.frequency == sampleRate) return mono;

        float ratio = (float)clip.frequency / sampleRate;
        int newLength = Mathf.RoundToInt(clip.samples / ratio);
        float[] resampled = new float[newLength];

        for (int i = 0; i < newLength; i++)
        {
            float srcIdx = i * ratio;
            int idx0 = Mathf.FloorToInt(srcIdx);
            int idx1 = Mathf.Min(idx0 + 1, mono.Length - 1);
            float t = srcIdx - idx0;
            resampled[i] = Mathf.Lerp(mono[idx0], mono[idx1], t);
        }

        return resampled;
    }


    [Serializable] public sealed class TakeObjectBinding
    {
        public string id;
        public Transform transform;
        public bool geometry;
    }

    public enum SessionMode { Idle, Live, Recording, Replay, Draining }
    public SessionMode Mode => takeWriter!=null ? SessionMode.Recording:captureScene!=null ? SessionMode.Live:
        replayTake!=null ? SessionMode.Replay:monitorQueue!=null ? SessionMode.Draining:SessionMode.Idle;
    void RequireIdle()
    {
        if (Mode!=SessionMode.Idle) throw new InvalidOperationException("Finish the current acoustic session first ("+Mode+").");
    }
    AudioManager sessionAudio;
    AudioManager.PlaybackCapture playbackCapture;

    [Header("Shared acoustic renderer")]
    [UnityEngine.Serialization.FormerlySerializedAs("useTakeRenderer")]
    [Tooltip("Render audio while capturing. Disable for silent data collection.")]
    public bool monitorLive;
    public ComputeShader directShader;
    public TakeRender.Settings directSettings=new TakeRender.Settings();
    [Min(0)] public int monitorListener;
    [Min(1)] public int monitorLeadSamples=4096;
    [Min(1)] public int monitorCapacitySamples=96000;
    public bool IsLiveSimulation => captureScene!=null && liveDirect!=null;
    [Serializable] public sealed class LiveDiagnostics
    {
        public string backend,scope="producer stopped; callback counters sampled at stop";
        public int sampleRate,responseCount,queueCapacity,deviceBufferSamples,maximumPendingObservations,retainedEvents;
        public long pairPrimitiveTestUpperBound,validationCount,coalescedObservations,responseSupportSamples;
        public bool tailCoversResponseSupport;
        public double completionSeconds,maximumCompletionSeconds;
        public long channelMismatchSamples;
        public long observedSamples,leadSamples,underrunSamples,lateSamples,overflowSamples,overBudgetCallbacks,maximumObservationGapSamples;
        public double preparationSeconds,validationSeconds,maximumValidationSeconds,solveSeconds,audioSeconds,responsePreparationSeconds,maximumSolveSeconds,maximumCallbackSeconds;
    }
    public LiveDiagnostics liveDiagnostics=new LiveDiagnostics();
    // Unity hot reload also serializes private fields; session ownership is runtime-only.
    [NonSerialized] AcousticTake.Scene captureScene;
    SceneToTake.Guard captureGuard;
    TakeRender.Live liveDirect;
    AudioManager.PcmQueue monitorQueue;
    int captureRate;

    [Header("Take recording / replay")]
    public AcousticTake.Dimension takeDimension = AcousticTake.Dimension.Spatial3D;
    public Transform takeRoot;
    public AudioMaterial takeDefaultMaterial;
    [Min(.0001f)] public float takeCurveTolerance = .01f;
    public List<TakeObjectBinding> takeObjects = new List<TakeObjectBinding>();
    public List<RaytracedAudioSource> takeSources = new List<RaytracedAudioSource>();
    public List<Transform> takeListeners = new List<Transform>();
    [Tooltip("Additional custom movement scripts to suspend during visual replay.")]
    public Behaviour[] replayMotionWriters = new Behaviour[0];
    public string takePath;
    [Min(0)] public double replaySeconds;
    public bool IsRecordingTake => takeWriter != null;
    public bool RecordingFailed {get;private set;}
    public bool IsReplayingTake => replayTake != null;
    public bool IsTakeReplayPlaying => replayTake != null && replayPlaying;
    public long LiveSample => captureScene!=null ? TakeSampleNow():0;
    public string TakeStatus { get; private set; } = "Idle";
    public event Action<string> RecordingStarted;
    public event Action<long> RecordingObserved;
    public event Action<string> RecordingFinalizing;
    public event Action<string,bool> RecordingStopped;

    AcousticTake.Writer takeWriter;
    SourceStream[] captureStreams;
    SourceStream.Capture streamCapture;
    readonly List<AcousticTake.StreamEpoch> streamEpochs=new List<AcousticTake.StreamEpoch>();
    public AcousticTake.SessionDefinition SessionInfo;
    AcousticTake replayTake;
    [NonSerialized] AcousticTake.State[] takeState, previousTakeState, restoreTakeState;
    [NonSerialized] bool[] restoreActiveSelf;
    int[] replayOrder;
    Transform[] captureSources;
    Transform[] captureObjects;
    double takeClockOrigin;
    int takeOutputRate;
    long previousTakeSample;
    bool replayPlaying;
    bool externalCaptureClock;
    readonly HashSet<int> takeCuts = new HashSet<int>();
    readonly Dictionary<Behaviour, bool> suspendedWriters = new Dictionary<Behaviour, bool>();
    struct BodyState { public bool kinematic; public Vector3 velocity, angularVelocity; }
    readonly Dictionary<Rigidbody, BodyState> suspendedBodies = new Dictionary<Rigidbody, BodyState>();
    readonly Dictionary<Rigidbody2D, bool> suspendedBodies2D = new Dictionary<Rigidbody2D, bool>();

    // Call in edit mode to persist identity in the scene, including source/listener-only objects.
    public void RegisterTakeObjects()
    {
        RequireIdle();
        takeObjects.RemoveAll(b => b == null || b.transform == null);
        if (takeRoot != null)
        {
            foreach (var surface in takeRoot.GetComponentsInChildren<AcousticSurface>(true)) RegisterTakeObject(surface.transform, true);
            foreach (var audio in takeRoot.GetComponentsInChildren<RaytracedAudioSource>(true))
                if (!takeSources.Contains(audio)) takeSources.Add(audio);
        }
        foreach (var audio in takeSources) if (audio != null) RegisterTakeObject(audio.transform, false);
        foreach (var receiver in takeListeners) if (receiver != null) RegisterTakeObject(receiver, false);
        foreach (var binding in takeObjects) if (string.IsNullOrEmpty(binding.id)) binding.id = Guid.NewGuid().ToString("N");
        takeObjects.Sort((a,b) => StringComparer.Ordinal.Compare(a.id,b.id));
        ValidateTakeBindings();
    }

    void RegisterTakeObject(Transform target, bool geometry)
    {
        var existing = takeObjects.Find(b => b.transform == target);
        if (existing != null) { existing.geometry |= geometry; return; }
        takeObjects.Add(new TakeObjectBinding { id = Guid.NewGuid().ToString("N"), transform = target, geometry = geometry });
    }

    void ValidateTakeBindings()
    {
        var ids = new HashSet<string>(); var transforms = new HashSet<Transform>();
        foreach (var binding in takeObjects)
            if (binding == null || binding.transform == null || string.IsNullOrEmpty(binding.id) || !ids.Add(binding.id) || !transforms.Add(binding.transform))
                throw new InvalidOperationException("Take objects need unique, persisted IDs and transforms. Register objects in the Inspector first.");
        if (takeObjects.Count == 0) throw new InvalidOperationException("Register objects before recording.");
    }

    public void BeginTakeRecording(string directory, bool retainPayloads=true, bool monitorOutput=true)
    {
        if (string.IsNullOrEmpty(directory)) throw new ArgumentException("Choose a take directory.");
        BeginCapture(directory,retainPayloads,monitorOutput);
    }
    public void BeginLiveSimulation()
    {
        if (!monitorLive) throw new InvalidOperationException("Enable live monitoring first.");
        BeginCapture(null);
    }
    SourceStream[] replayStreams;
    readonly float[] replaySourceBlock=new float[4096];
    public bool IsSourceReplay => externalCaptureClock || replayTake!=null;
    public bool CanReplayObservation => liveDirect!=null && liveDirect.CanObserve;
    public void BeginInputPlayback(string directory=null)
    {
        BeginLiveSimulation(); externalCaptureClock=true;
        if(directory!=null)replayStreams=AcousticTake.OpenStreams(directory,AcousticTake.ReadManifest(directory));
    }
    void ReplaySourcesThrough(long end)
    {
        if(replayStreams==null)return;
        for(int s=0;s<replayStreams.Length;s++)if(replayStreams[s]!=null)
        {
            var target=captureStreams[s];
            while(target.End<end)
            {
                int count=(int)Math.Min(replaySourceBlock.Length,end-target.End);
                for(int i=0;i<count;i++)replaySourceBlock[i]=replayStreams[s].Read(target.End+i);
                target.Append(replaySourceBlock,0,count,end);
            }
        }
    }
    public void ReplayObservation(long sample,bool final)
    {
        if(!externalCaptureClock)throw new InvalidOperationException("Begin input playback before replaying observations.");
        ReplaySourcesThrough(sample+1);
        if(final)EndCapture(sample);else CaptureTakeState(sample);
    }
    void BeginCapture(string directory, bool retainPayloads=true, bool monitorOutput=true)
    {
        RequireIdle();
        if (Application.isPlaying && !isActiveAndEnabled) throw new InvalidOperationException("The session manager must stay active while capturing.");
        long preparationStart=System.Diagnostics.Stopwatch.GetTimestamp();
        if (monitorLive)
        {
            TakeRender.Validate(directSettings);
            if(directSettings.listenerOutput==TakeRender.ListenerOutput.Binaural && AudioSettings.GetConfiguration().speakerMode!=AudioSpeakerMode.Stereo)
                throw new InvalidOperationException("Binaural monitoring requires a stereo output device/configuration.");
            if (audioManager==null || audioManager.IsCapturingOutput) throw new InvalidOperationException("Assign an available AudioManager before starting a session.");
            // Live output and its recorded inputs share the actual device clock.
            // Silent offline capture keeps the explicitly configured sample rate.
            sampleRate=AudioSettings.outputSampleRate;
            if (directShader!=null && !SystemInfo.supportsAsyncGPUReadback) throw new NotSupportedException("Live GPU tracing requires asynchronous readback support; select CPU on this device.");
        }
        if (sampleRate<=0) throw new InvalidOperationException("Recording requires a positive sample rate.");
        ValidateTakeBindings();
        var sources = new List<Transform>(); var clips = new List<AudioClip>();
        foreach (var audio in takeSources)
        {
            if (audio == null || (!audio.streamed && audio.clip == null) || sources.Contains(audio.transform)) throw new InvalidOperationException("Take sources must be unique and have prepared clips or stream bindings.");
            sources.Add(audio.transform); clips.Add(audio.streamed?null:audio.clip);
        }
        var receivers = new List<Transform>(takeListeners);
        var scene = SceneToTake.Compile(takeObjects, takeDimension, takeCurveTolerance, takeDefaultMaterial,
            sources.ToArray(), clips.ToArray(), receivers.ToArray(), out var geometry);
        var pcm = new float[scene.clips.Length][];
        for (int i=0; i<clips.Count; i++)
        {
            if(scene.sources[i].streamed)continue;
            int clipIndex=scene.sources[i].clipIndex;
            if (pcm[clipIndex] == null) pcm[clipIndex]=LoadSample(clips[i],sampleRate);
        }
        captureSources = sources.ToArray();
        var captureClips=clips.ToArray(); captureObjects=new Transform[takeObjects.Count];
        for (int i=0; i<takeObjects.Count; i++) captureObjects[i]=takeObjects[i].transform;
        takeState = new AcousticTake.State[takeObjects.Count]; previousTakeState = new AcousticTake.State[takeObjects.Count];
        if (monitorLive && (audioManager==null || receivers.Count==0 || sources.Count==0))
            throw new InvalidOperationException("Live monitoring needs an AudioManager, a source and a listener.");
        captureGuard=new SceneToTake.Guard(this,captureSources,captureClips,receivers.ToArray());
        captureRate=sampleRate; captureScene=scene;
        liveDiagnostics=new LiveDiagnostics {sampleRate=sampleRate,leadSamples=monitorLeadSamples,queueCapacity=monitorCapacitySamples};
        AudioSettings.GetDSPBufferSize(out liveDiagnostics.deviceBufferSamples,out var deviceBufferCount);
        takeClockOrigin = AudioSettings.dspTime; takeOutputRate = AudioSettings.outputSampleRate;
        previousTakeSample = -1;
        try
        {
            captureStreams=new SourceStream[scene.sources.Length];streamEpochs.Clear();
            for(int i=0;i<captureStreams.Length;i++)if(scene.sources[i].streamed)
            {
                var audio=captureSources[i].GetComponent<RaytracedAudioSource>();
                captureStreams[i]=audio.Stream=new SourceStream(checked((int)directSettings.HistorySamples(sampleRate)+2*sampleRate+monitorCapacitySamples));
            }
            if (directory!=null)
            {
                RecordingFailed=false;takePath=Path.GetFullPath(directory);
                takeWriter=new AcousticTake.Writer(takePath,sampleRate,scene,geometry,pcm,retainPayloads);
                takeWriter.Metadata.captureProvenance=AcousticTake.Provenance.Capture(monitorLive?directShader:null,monitorLive?directSettings:null);
                takeWriter.Metadata.observationPolicy="first-observation-per-audio-sample";
                takeWriter.Metadata.session=SessionInfo;
                if(Array.Exists(captureStreams,v=>v!=null))streamCapture=new SourceStream.Capture(takePath,captureStreams,takeWriter.Metadata);
            }
            if (monitorLive)
            {
                var settings=JsonUtility.FromJson<TakeRender.Settings>(JsonUtility.ToJson(directSettings));
                if(takeWriter!=null)takeWriter.Metadata.captureSettings=settings;
                sessionAudio=audioManager;
                if (directory!=null && retainPayloads && monitorOutput) playbackCapture=new AudioManager.PlaybackCapture(Path.Combine(takePath,"monitor"),sampleRate,monitorCapacitySamples,
                    liveDiagnostics.deviceBufferSamples,monitorLeadSamples,monitorListener,settings,directShader);
                monitorQueue=new AudioManager.PcmQueue(monitorCapacitySamples,sampleRate,playbackCapture,settings.ChannelsPerListener);
                if (monitorLeadSamples<0 || monitorLeadSamples>monitorCapacitySamples) throw new ArgumentOutOfRangeException(nameof(monitorLeadSamples));
                monitorQueue.Publish(0,null,monitorLeadSamples);
                liveDirect=new TakeRender.Live(scene,geometry,pcm,sampleRate,settings,monitorQueue,monitorLeadSamples,monitorListener,directShader,captureStreams);
                liveDiagnostics.backend=liveDirect.Backend;
                if (playbackCapture!=null) liveDirect.ResponseApplied=playbackCapture.RecordResponse;
            }
            liveDiagnostics.preparationSeconds=TakeRender.Elapsed(preparationStart);
            if(takeWriter!=null) RecordingStarted?.Invoke(takePath);
            takeClockOrigin=AudioSettings.dspTime;
            if (playbackCapture!=null) playbackCapture.Metadata.inputDspOriginSeconds=takeClockOrigin;
            if (monitorQueue!=null) sessionAudio.StartPcmStream(monitorQueue);
            CaptureTakeState(0);
            for(int i=0;i<captureStreams.Length;i++)if(captureStreams[i]!=null)
            {
                var e=new AcousticTake.VoiceEvent {sourceIndex=i,action=AcousticTake.VoiceAction.Play,gain=captureSources[i].GetComponent<RaytracedAudioSource>().gain};
                takeWriter?.Append(e);liveDirect?.Append(e);
            }
            TakeStatus = directory==null ? "Live acoustic simulation":"Recording: " + takePath;
        }
        catch(Exception e) { CloseCapture(false);if(directory!=null)RecordingFailed=true; TakeStatus="Recording failed: "+e.Message; throw; }
    }

    public void EndTakeRecording() { if (takeWriter!=null) EndCapture(); }
    public void StopLiveSimulation() { if (captureScene!=null) EndCapture(); }
    void EndCapture(long? finalSample=null)
    {
        bool complete=false;
        try
        {
            long end=finalSample??TakeSampleNow(); ValidateCapturedScene(); CaptureTakeState(end);
            if(captureStreams!=null)foreach(var stream in captureStreams)if(stream!=null)
            {stream.SilenceUntil(end+1,end,SourceStream.Status.Missing);stream.Finish();}
            liveDirect?.Finish(end,takeState,wait:false); UpdateLiveDiagnostics(end);
            if (takeWriter!=null)
            {
                if (liveDirect!=null) File.WriteAllText(Path.Combine(takePath,"live.json"),JsonUtility.ToJson(liveDiagnostics,true));
                streamCapture?.Complete(takeWriter.Metadata);takeWriter.Metadata.streamEpochs=streamEpochs.ToArray();
                takeWriter.Complete(); RecordingFinalizing?.Invoke(takePath); TakeStatus="Saved: "+takePath;
            }
            else TakeStatus="Live simulation stopped.";
            complete=true;
        }
        catch(Exception e)
        {
            TakeStatus="Recording failed: "+e.Message; takeWriter?.Invalidate(); throw;
        }
        finally { CloseCapture(complete); }
    }
    public void AbortTakeRecording(string reason)
    {
        if(takeWriter==null)return;
        try { takeWriter.Invalidate(); }
        finally { CloseCapture(false); TakeStatus="Recording failed: "+reason; }
    }
    void UpdateLiveDiagnostics(long sample)
    {
        liveDiagnostics.observedSamples=sample;
        if (liveDirect==null) return;
        liveDiagnostics.maximumPendingObservations=liveDirect.MaximumPendingObservations; liveDiagnostics.retainedEvents=liveDirect.RetainedEvents;
        liveDiagnostics.completionSeconds=liveDirect.CompletionSeconds; liveDiagnostics.maximumCompletionSeconds=liveDirect.MaximumCompletionSeconds;
        liveDiagnostics.pairPrimitiveTestUpperBound=liveDirect.PairPrimitiveUpperBound;
        liveDiagnostics.responseSupportSamples=liveDirect.ResponseSupportSamples;
        liveDiagnostics.tailCoversResponseSupport=liveDirect.TailCoversResponseSupport;
        liveDiagnostics.responseCount=liveDirect.ResponseCount; liveDiagnostics.solveSeconds=liveDirect.SolveSeconds;
        liveDiagnostics.audioSeconds=liveDirect.AudioSeconds; liveDiagnostics.responsePreparationSeconds=liveDirect.ResponsePreparationSeconds; liveDiagnostics.maximumSolveSeconds=liveDirect.MaximumSolveSeconds;
        liveDiagnostics.channelMismatchSamples=System.Threading.Interlocked.Read(ref monitorQueue.ChannelMismatchSamples);
        liveDiagnostics.underrunSamples=System.Threading.Interlocked.Read(ref monitorQueue.UnderrunSamples);
        liveDiagnostics.overBudgetCallbacks=System.Threading.Interlocked.Read(ref monitorQueue.OverBudgetCallbacks);
        liveDiagnostics.lateSamples=monitorQueue.LateSamples; liveDiagnostics.overflowSamples=monitorQueue.OverflowSamples;
        liveDiagnostics.maximumCallbackSeconds=(double)System.Threading.Interlocked.Read(ref monitorQueue.MaximumCallbackTicks)/System.Diagnostics.Stopwatch.Frequency;
    }
    void CloseCapture(bool drain)
    {
        string recordedDirectory=takeWriter?.DirectoryPath;
        bool recordedComplete=takeWriter!=null && takeWriter.Metadata.complete;
        if(recordedDirectory!=null)RecordingFailed=!recordedComplete;
        try { if(!drain) {liveDirect?.Dispose();liveDirect=null;} }
        finally
        {
            try {streamCapture?.Dispose();}catch(Exception e) {Debug.LogWarning("Source writer stopped: "+e.Message,this);}
            finally {streamCapture=null;takeWriter?.Dispose();takeWriter=null;}
            if(captureStreams!=null)foreach(var stream in captureStreams)stream?.Dispose();captureStreams=null;
            if (drain && monitorQueue!=null) {if(liveDirect==null || liveDirect.Completed)monitorQueue.Complete();}
            else
            {
                sessionAudio?.StopPcmStream(); playbackCapture?.Dispose();
                monitorQueue=null; playbackCapture=null; sessionAudio=null;
            }
            if(replayStreams!=null)foreach(var stream in replayStreams)stream?.Dispose();replayStreams=null;
            captureScene=null; captureGuard=null; takeCuts.Clear(); externalCaptureClock=false;
            if(recordedDirectory!=null) RecordingStopped?.Invoke(recordedDirectory,recordedComplete);
        }
    }

    long TakeSampleNow()
    {
        if (sampleRate != captureRate || AudioSettings.outputSampleRate != takeOutputRate || AudioSettings.dspTime < takeClockOrigin)
            throw new InvalidOperationException("Audio clock changed; this take is incomplete.");
        return checked((long)Math.Floor((AudioSettings.dspTime - takeClockOrigin) * sampleRate));
    }

    public void MarkTakeDiscontinuity(Transform target)
    {
        for(int i=0;i<takeObjects.Count;i++)
            if(takeObjects[i].transform==target || takeObjects[i].transform.IsChildOf(target))takeCuts.Add(i);
    }

    void ValidateCapturedScene()
    {
        long start=System.Diagnostics.Stopwatch.GetTimestamp(); captureGuard.Validate();
        double elapsed=TakeRender.Elapsed(start); liveDiagnostics.validationCount++; liveDiagnostics.validationSeconds+=elapsed;
        liveDiagnostics.maximumValidationSeconds=Math.Max(liveDiagnostics.maximumValidationSeconds,elapsed);
    }

    void CaptureTakeState(long sample)
    {
        // An accepted knot may already have contributed to live PCM. Duplicate polls cannot rewrite it.
        if (sample==previousTakeSample)
        {
            liveDiagnostics.coalescedObservations++;
            if (takeWriter!=null) takeWriter.Metadata.coalescedObservations++;
            liveDirect?.Pump(); UpdateLiveDiagnostics(sample); return;
        }
        var scene=captureScene;
        ValidateCapturedScene();
        if (takeObjects.Count != scene.objects.Length) throw new InvalidOperationException("Scene membership changed during recording.");
        for (int i=0; i<takeState.Length; i++)
        {
            var binding = takeObjects[i];
            if (binding.transform == null || binding.transform != captureObjects[i] || binding.id != scene.objects[i].id ||
                (binding.transform.lossyScale - scene.objects[i].scale).sqrMagnitude > 1e-10f)
                throw new InvalidOperationException("A recorded object was removed, reordered or resized; start a new take.");
            takeState[i] = SceneToTake.Observe(scene, i, binding.transform);
            if (takeCuts.Contains(i) || (sample == previousTakeSample && (previousTakeState[i].flags & AcousticTake.StateFlags.Cut) != 0))
                takeState[i].flags |= AcousticTake.StateFlags.Cut;
            if (previousTakeSample >= 0 && sample > previousTakeSample)
                takeState[i].MeasureMotion(previousTakeState[i], (double)(sample-previousTakeSample)/sampleRate);
        }
        if (previousTakeSample>=0) liveDiagnostics.maximumObservationGapSamples=Math.Max(liveDiagnostics.maximumObservationGapSamples,sample-previousTakeSample);
        takeWriter?.Append(sample,takeState);
        liveDirect?.Observe(previousTakeSample<0 ? 0:previousTakeSample,sample,previousTakeSample<0 ? takeState:previousTakeState,takeState);
        UpdateLiveDiagnostics(sample);
        Array.Copy(takeState,previousTakeState,takeState.Length);
        previousTakeSample=sample; takeCuts.Clear();
        if(takeWriter!=null) RecordingObserved?.Invoke(sample);
    }

    public void RecordStreamEpoch(RaytracedAudioSource source,AcousticTake.StreamEpoch epoch)
    {
        epoch.sourceIndex=Array.IndexOf(captureSources,source.transform);
        if(epoch.sourceIndex<0 || !source.streamed)throw new InvalidOperationException("Unregistered source stream.");
        streamEpochs.Add(epoch);
    }
    public void RecordSourceEvent(Transform target, AcousticTake.VoiceAction action, float gain = 1, bool repeat = false, long clipOffset = 0)
    {
        if (captureScene == null) return;
        int index=Array.IndexOf(captureSources,target);
        if (index < 0) throw new InvalidOperationException("Source is not registered in this take.");
        var e=new AcousticTake.VoiceEvent { sample=TakeSampleNow(), sourceIndex=index,
            action=action, gain=gain, loop=repeat, clipOffset=clipOffset };
        takeWriter?.Append(e); liveDirect?.Append(e);
    }
    public void AppendReplaySound(AcousticTake.VoiceEvent value)
    {
        if(Mode!=SessionMode.Live) throw new InvalidOperationException("Input playback needs a live audio session.");
        liveDirect.Append(value);
    }

    void LateUpdate()
    {
        try
        {
            if(takeWriter!=null && streamCapture?.Failure is Exception failure)
            {
                RecordingFailed=true;string directory=takeWriter.DirectoryPath;takeWriter.Invalidate();takeWriter.Dispose();takeWriter=null;
                try {streamCapture.Dispose();}catch(Exception) {} // The originating fault is reported below.
                streamCapture=null;RecordingStopped?.Invoke(directory,false);
                TakeStatus="Recording failed: "+failure.Message+" Live playback continues.";Debug.LogError(TakeStatus,this);
            }
            if(Mode==SessionMode.Draining && liveDirect!=null)
            {
                liveDirect.Pump();
                if(liveDirect.Completed) {liveDirect.Dispose();liveDirect=null;monitorQueue.Complete();}
            }
            if (Mode==SessionMode.Draining && monitorQueue.IsDrained && (playbackCapture==null || playbackCapture.Closed))
            { monitorQueue=null; playbackCapture=null; sessionAudio=null; }
            if (captureScene != null)
            {
                if(externalCaptureClock)liveDirect?.Pump();
                else CaptureTakeState(TakeSampleNow());
            }
            if (replayTake != null)
            {
                if (replayPlaying) replaySeconds = AudioSettings.dspTime-takeClockOrigin;
                long clockSample=(long)Math.Round(replaySeconds*replayTake.Metadata.sampleRate);
                long sample=Math.Max(0,Math.Min(clockSample,replayTake.Metadata.durationSamples));
                if(replayAudio!=null)PumpReplayAudio(sample);
                long visual=replayAudio!=null || replayAudioFinished?Math.Max(0,Math.Min(clockSample-monitorLeadSamples,replayTake.Metadata.durationSamples)):sample;
                replayTake.Evaluate(visual,takeState); ApplyTakeState(takeState);
                if (sample == replayTake.Metadata.durationSamples && (monitorQueue==null || monitorQueue.IsDrained))
                { replayTake.Evaluate(sample,takeState);ApplyTakeState(takeState);replayPlaying=false; }
            }
        }
        catch (Exception e)
        {
            if (captureScene != null || liveDirect!=null) CloseCapture(false);
            if (replayTake != null || restoreTakeState != null) StopTakeReplay();
            TakeStatus="Take stopped: "+e.Message;
            Debug.LogError(TakeStatus,this);
        }
    }

    TakeRender.Live replayAudio;
    AcousticTake.State[] replayAudioLeft,replayAudioRight;
    int replayFrame,replayEvent;
    bool replayAudioFinished;
    public void BeginTakeReplay(string directory, bool play = true, bool audio=false)
    {
        if (takeWriter != null) throw new InvalidOperationException("Stop recording before replaying.");
        if (IsLiveSimulation) StopLiveSimulation();
        StopTakeReplay(); RequireIdle(); ValidateTakeBindings();
        var loaded=AcousticTake.Open(directory);
        try
        {
            if (loaded.Metadata.scene.objects.Length != takeObjects.Count) throw new InvalidOperationException("Take object count does not match this scene.");
            for (int i=0; i<takeObjects.Count; i++)
                if (loaded.Metadata.scene.objects[i].id != takeObjects[i].id) throw new InvalidOperationException("Take object IDs do not match this scene.");
            restoreTakeState=new AcousticTake.State[takeObjects.Count];
            restoreActiveSelf=new bool[takeObjects.Count];
            takeState=new AcousticTake.State[takeObjects.Count];
            replayOrder=new int[takeObjects.Count];
            for (int i=0; i<replayOrder.Length; i++) replayOrder[i]=i;
            Array.Sort(replayOrder,(a,b)=>Depth(takeObjects[a].transform).CompareTo(Depth(takeObjects[b].transform)));
            for (int i=0; i<takeObjects.Count; i++)
            {
                var t=takeObjects[i].transform;
                restoreTakeState[i]=AcousticTake.State.Observe(t,0);
                restoreActiveSelf[i]=t.gameObject.activeSelf;
            }
            for (int i=0; i<takeObjects.Count; i++)
            {
                var t=takeObjects[i].transform;
                for (var parent=t; parent != null; parent=parent.parent)
                {
                    Suspend(parent.GetComponent<PlayerController>()); Suspend(parent.GetComponent<MouseLook>());
                    Suspend(parent.GetComponent<Oscillator>()); Suspend(parent.GetComponent<Animator>());
                    var body=parent.GetComponent<Rigidbody>();
                    if (body != null && !suspendedBodies.ContainsKey(body))
                    {
                        suspendedBodies.Add(body,new BodyState {kinematic=body.isKinematic,velocity=body.linearVelocity,angularVelocity=body.angularVelocity});
                        body.isKinematic=true;
                    }
                    var body2=parent.GetComponent<Rigidbody2D>();
                    if (body2 != null && !suspendedBodies2D.ContainsKey(body2)) { suspendedBodies2D.Add(body2,body2.simulated); body2.simulated=false; }
                    if (parent == takeRoot) break;
                }
            }
            foreach (var writer in replayMotionWriters) Suspend(writer);
            replayTake=loaded; replaySeconds=0; replayPlaying=play; takeClockOrigin=AudioSettings.dspTime;
            replayTake.Evaluate(0,takeState); ApplyTakeState(takeState);
            if(audio)BeginReplayAudio();
            TakeStatus=(audio?"Replaying scene and recorded source audio: ":"Replaying recorded poses: ")+directory;
        }
        catch { loaded.Dispose(); StopTakeReplay(); throw; }
    }

    void BeginReplayAudio()
    {
        var meta=replayTake.Metadata;
        if(meta.sampleRate!=AudioSettings.outputSampleRate)throw new InvalidOperationException("Set the output device to the recorded sample rate before auditioning this take. Offline rendering supports its original rate.");
        if(audioManager==null)throw new InvalidOperationException("Replay audio needs an AudioManager.");
        var pcm=new float[meta.scene.clips.Length][];for(int i=0;i<pcm.Length;i++)pcm[i]=replayTake.ReadClip(i);
        var streams=new SourceStream[meta.scene.sources.Length];for(int i=0;i<streams.Length;i++)if(meta.scene.sources[i].streamed)streams[i]=replayTake.ReadStream(i);
        monitorQueue=new AudioManager.PcmQueue(monitorCapacitySamples,meta.sampleRate,null,directSettings.ChannelsPerListener);
        monitorQueue.Publish(0,null,monitorLeadSamples);sessionAudio=audioManager;
        replayAudio=new TakeRender.Live(meta.scene,replayTake.Geometry,pcm,meta.sampleRate,directSettings,monitorQueue,monitorLeadSamples,monitorListener,directShader,streams);
        replayAudioLeft=new AcousticTake.State[meta.scene.objects.Length];replayAudioRight=new AcousticTake.State[meta.scene.objects.Length];
        replayTake.ReadStoredFrame(0,replayAudioLeft);replayFrame=1;replayEvent=0;replayAudioFinished=false;
        takeClockOrigin=AudioSettings.dspTime;sessionAudio.StartPcmStream(monitorQueue);
    }
    void PumpReplayAudio(long sample)
    {
        // Recorded future inputs are available. Fill only the existing bounded queues,
        // instead of waiting until each pose is due and treating a burst as live input.
        replayAudio.Pump();
        long horizon=Math.Min(replayTake.Metadata.durationSamples,sample+monitorCapacitySamples-monitorLeadSamples);
        while(replayEvent<replayTake.Events.Length && replayTake.Events[replayEvent].sample<=horizon)
            replayAudio.Append(replayTake.Events[replayEvent++]);
        while(replayAudio.CanObserve && replayFrame<replayTake.Metadata.frameCount && replayTake.FrameSample(replayFrame)<=horizon)
        {
            replayTake.ReadStoredFrame(replayFrame,replayAudioRight);
            replayAudio.Observe(replayTake.FrameSample(replayFrame-1),replayTake.FrameSample(replayFrame),replayAudioLeft,replayAudioRight);
            var swap=replayAudioLeft;replayAudioLeft=replayAudioRight;replayAudioRight=swap;replayFrame++;
        }
        replayAudio.Pump();
        if(replayFrame==replayTake.Metadata.frameCount && !replayAudio.Finishing)replayAudio.Finish(replayTake.Metadata.durationSamples,replayAudioLeft,wait:false);
        if(replayAudio.Completed)
        {replayAudio.Dispose();replayAudio=null;replayAudioFinished=true;monitorQueue.Complete();}
    }
    public void SetTakeReplayPlaying(bool play)
    {
        if(replayAudio!=null || replayAudioFinished)throw new InvalidOperationException("Source-audio replay plays from the beginning; reopen the take to restart.");
        if (replayTake == null) throw new InvalidOperationException("Open a take before playing it.");
        if (double.IsNaN(replaySeconds) || double.IsInfinity(replaySeconds)) throw new ArgumentException("Invalid replay time.");
        replaySeconds=Math.Max(0,Math.Min(replaySeconds,(double)replayTake.Metadata.durationSamples/replayTake.Metadata.sampleRate));
        takeClockOrigin=AudioSettings.dspTime-replaySeconds;
        replayPlaying=play;
    }

    void Suspend(Behaviour behaviour)
    {
        if (behaviour == null || behaviour == this || suspendedWriters.ContainsKey(behaviour)) return;
        suspendedWriters.Add(behaviour,behaviour.enabled); behaviour.enabled=false;
    }

    void ApplyTakeState(AcousticTake.State[] frame)
    {
        // Parent-first application prevents a child pose from being moved twice.
        foreach (int i in replayOrder)
        {
            var t=takeObjects[i].transform;
            if (t == null) continue;
            t.SetPositionAndRotation(frame[i].position,frame[i].rotation);
            t.gameObject.SetActive((frame[i].flags & AcousticTake.StateFlags.Active) != 0);
        }
    }
    static int Depth(Transform t) { int depth=0; while (t != null) { depth++; t=t.parent; } return depth; }

    public void StopTakeReplay()
    {
        bool wasReplaying = replayTake != null;
        if(replayAudio!=null || replayAudioFinished)
        {
            replayAudio?.Dispose();replayAudio=null;replayAudioFinished=false;
            sessionAudio?.StopPcmStream();monitorQueue=null;sessionAudio=null;
        }
        if (restoreTakeState != null)
        {
            ApplyTakeState(restoreTakeState);
            for (int i=0; i<takeObjects.Count; i++) if (takeObjects[i].transform != null) takeObjects[i].transform.gameObject.SetActive(restoreActiveSelf[i]);
            restoreTakeState=null; restoreActiveSelf=null;
        }
        foreach (var p in suspendedWriters) if (p.Key != null) p.Key.enabled=p.Value;
        foreach (var p in suspendedBodies) if (p.Key != null)
        {
            p.Key.isKinematic=p.Value.kinematic;
            if (!p.Value.kinematic) { p.Key.linearVelocity=p.Value.velocity; p.Key.angularVelocity=p.Value.angularVelocity; }
        }
        foreach (var p in suspendedBodies2D) if (p.Key != null) p.Key.simulated=p.Value;
        suspendedWriters.Clear(); suspendedBodies.Clear(); suspendedBodies2D.Clear();
        replayTake?.Dispose(); replayTake=null; replayPlaying=false;
        if (wasReplaying) TakeStatus="Replay stopped; live scene restored.";
    }

    void OnDisable()
    {
        if (Mode!=SessionMode.Idle) { DisposeTakeSession(); TakeStatus="Session cancelled because its manager was disabled."; }
    }

    void DisposeTakeSession()
    {
        CloseCapture(false);
        StopTakeReplay();
    }
}
