using System;
using System.Collections.Generic;
using System.IO;
using UnityEngine;

public partial class RayTraceManager
{
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
    void RequireLegacyIdle()
    {
        RequireIdle();
        if (useTakeRenderer) throw new InvalidOperationException("This action belongs to the legacy renderer. Use live preview or render a saved take.");
    }
    AudioManager sessionAudio;
    AudioManager.PlaybackCapture playbackCapture;

    [Header("Shared direct renderer")]
    public bool useTakeRenderer;
    public ComputeShader directShader;
    public TakeRender.Settings directSettings=new TakeRender.Settings();
    [Min(0)] public int monitorListener;
    [Min(1)] public int monitorLeadSamples=4096;
    [Min(1)] public int monitorCapacitySamples=96000;
    public bool IsLiveSimulation => liveDirect!=null;
    [Serializable] public sealed class LiveDiagnostics
    {
        public string backend,scope="producer stopped; callback counters sampled at stop";
        public int sampleRate,responseCount,queueCapacity,deviceBufferSamples,maximumPendingObservations,retainedEvents;
        public long pairPrimitiveTestUpperBound,validationCount,coalescedObservations;
        public double completionSeconds,maximumCompletionSeconds;
        public long channelMismatchSamples;
        public long observedSamples,leadSamples,underrunSamples,lateSamples,overflowSamples,overBudgetCallbacks,maximumObservationGapSamples;
        public double preparationSeconds,validationSeconds,maximumValidationSeconds,solveSeconds,audioSeconds,maximumSolveSeconds,maximumCallbackSeconds;
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
    public bool IsReplayingTake => replayTake != null;
    public bool IsTakeReplayPlaying => replayTake != null && replayPlaying;
    public long LiveSample => captureScene!=null ? TakeSampleNow():0;
    public string TakeStatus { get; private set; } = "Idle";
    public event Action<string> RecordingStarted;
    public event Action<long> RecordingObserved;
    public event Action<string,bool> RecordingStopped;

    AcousticTake.Writer takeWriter;
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
        if (obstacleObjects != null) foreach (var obj in obstacleObjects) if (obj != null) RegisterTakeObject(obj.transform, true);
        if (takeRoot != null)
        {
            foreach (var surface in takeRoot.GetComponentsInChildren<AcousticSurface>(true)) RegisterTakeObject(surface.transform, true);
            foreach (var audio in takeRoot.GetComponentsInChildren<RaytracedAudioSource>(true))
                if (!takeSources.Contains(audio)) takeSources.Add(audio);
        }
        if (source != null) RegisterTakeObject(source, false);
        if (listener != null) RegisterTakeObject(listener, false);
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

    public void BeginTakeRecording(string directory, bool retainPayloads=true)
    {
        if (string.IsNullOrEmpty(directory)) throw new ArgumentException("Choose a take directory.");
        BeginCapture(directory,retainPayloads);
    }
    public void BeginLiveSimulation()
    {
        if (!useTakeRenderer) throw new InvalidOperationException("Enable Use Take Renderer first.");
        BeginCapture(null);
    }
    public void BeginInputPlayback() { BeginLiveSimulation(); externalCaptureClock=true; }
    public void ReplayObservation(long sample,bool final)
    {
        if(!externalCaptureClock)throw new InvalidOperationException("Begin input playback before replaying observations.");
        if(final)EndCapture(sample);else CaptureTakeState(sample);
    }
    void BeginCapture(string directory, bool retainPayloads=true)
    {
        RequireIdle();
        if (Application.isPlaying && !isActiveAndEnabled) throw new InvalidOperationException("The session manager must stay active while capturing.");
        long preparationStart=System.Diagnostics.Stopwatch.GetTimestamp();
        if (useTakeRenderer)
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
        if (isStreaming || isPrecomputing || (precomputedPlaybackSource!=null && precomputedPlaybackSource.isPlaying)) throw new InvalidOperationException("Stop legacy playback before recording, then start voices after the take begins.");
        if (sampleRate<=0) throw new InvalidOperationException("Recording requires a positive sample rate.");
        ValidateTakeBindings();
        var sources = new List<Transform>(); var clips = new List<AudioClip>();
        foreach (var audio in takeSources)
        {
            if (audio == null || audio.clip == null || sources.Contains(audio.transform)) throw new InvalidOperationException("Take sources must be unique and have prepared clips.");
            sources.Add(audio.transform); clips.Add(audio.clip);
        }
        if (sources.Count == 0 && source != null && inputClip != null) { sources.Add(source); clips.Add(inputClip); }
        var receivers = new List<Transform>(takeListeners);
        if (receivers.Count == 0 && listener != null) receivers.Add(listener);
        var scene = SceneToTake.Compile(takeObjects, takeDimension, takeCurveTolerance, takeDefaultMaterial,
            sources.ToArray(), clips.ToArray(), receivers.ToArray(), out var geometry);
        var pcm = new float[scene.clips.Length][];
        for (int i=0; i<clips.Count; i++)
        {
            int clipIndex=scene.sources[i].clipIndex;
            if (pcm[clipIndex] == null) pcm[clipIndex]=LoadSample(clips[i],sampleRate);
        }
        captureSources = sources.ToArray();
        var captureClips=clips.ToArray(); captureObjects=new Transform[takeObjects.Count];
        for (int i=0; i<takeObjects.Count; i++) captureObjects[i]=takeObjects[i].transform;
        takeState = new AcousticTake.State[takeObjects.Count]; previousTakeState = new AcousticTake.State[takeObjects.Count];
        if (useTakeRenderer && (audioManager==null || receivers.Count==0 || sources.Count==0))
            throw new InvalidOperationException("Live monitoring needs an AudioManager, a source and a listener.");
        captureGuard=new SceneToTake.Guard(this,captureSources,captureClips,receivers.ToArray());
        captureRate=sampleRate; captureScene=scene;
        liveDiagnostics=new LiveDiagnostics {sampleRate=sampleRate,leadSamples=monitorLeadSamples,queueCapacity=monitorCapacitySamples};
        AudioSettings.GetDSPBufferSize(out liveDiagnostics.deviceBufferSamples,out var deviceBufferCount);
        takeClockOrigin = AudioSettings.dspTime; takeOutputRate = AudioSettings.outputSampleRate;
        previousTakeSample = -1;
        try
        {
            if (directory!=null)
            {
                takePath=Path.GetFullPath(directory);
                takeWriter=new AcousticTake.Writer(takePath,sampleRate,scene,geometry,pcm,retainPayloads);
                takeWriter.Metadata.observationPolicy="first-observation-per-audio-sample";
            }
            if (useTakeRenderer)
            {
                var settings=JsonUtility.FromJson<TakeRender.Settings>(JsonUtility.ToJson(directSettings));
                sessionAudio=audioManager;
                if (directory!=null && retainPayloads) playbackCapture=new AudioManager.PlaybackCapture(Path.Combine(takePath,"monitor"),sampleRate,monitorCapacitySamples,
                    liveDiagnostics.deviceBufferSamples,monitorLeadSamples,monitorListener,settings,directShader);
                monitorQueue=new AudioManager.PcmQueue(monitorCapacitySamples,sampleRate,playbackCapture,settings.ChannelsPerListener);
                if (monitorLeadSamples<0 || monitorLeadSamples>monitorCapacitySamples) throw new ArgumentOutOfRangeException(nameof(monitorLeadSamples));
                monitorQueue.Publish(0,null,monitorLeadSamples);
                liveDirect=new TakeRender.Live(scene,geometry,pcm,sampleRate,settings,monitorQueue,monitorLeadSamples,monitorListener,directShader);
                liveDiagnostics.backend=liveDirect.Backend;
                if (playbackCapture!=null) liveDirect.ResponseApplied=playbackCapture.RecordResponse;
            }
            liveDiagnostics.preparationSeconds=TakeRender.Elapsed(preparationStart);
            if(takeWriter!=null) RecordingStarted?.Invoke(takePath);
            takeClockOrigin=AudioSettings.dspTime;
            if (playbackCapture!=null) playbackCapture.Metadata.inputDspOriginSeconds=takeClockOrigin;
            if (monitorQueue!=null) sessionAudio.StartPcmStream(monitorQueue);
            CaptureTakeState(0);
            TakeStatus = directory==null ? "Live direct simulation":"Recording: " + takePath;
        }
        catch { CloseCapture(false); throw; }
    }

    public void EndTakeRecording() { if (takeWriter!=null) EndCapture(); }
    public void StopLiveSimulation() { if (captureScene!=null) EndCapture(); }
    void EndCapture(long? finalSample=null)
    {
        bool complete=false;
        try
        {
            long end=finalSample??TakeSampleNow(); ValidateCapturedScene(); CaptureTakeState(end);
            liveDirect?.Finish(end,takeState); UpdateLiveDiagnostics(end);
            if (takeWriter!=null)
            {
                if (liveDirect!=null) File.WriteAllText(Path.Combine(takePath,"live.json"),JsonUtility.ToJson(liveDiagnostics,true));
                takeWriter.Complete(); TakeStatus="Saved: "+takePath;
            }
            else TakeStatus="Live simulation stopped.";
            complete=true;
        }
        finally { CloseCapture(complete); }
    }
    void UpdateLiveDiagnostics(long sample)
    {
        liveDiagnostics.observedSamples=sample;
        if (liveDirect==null) return;
        liveDiagnostics.maximumPendingObservations=liveDirect.MaximumPendingObservations; liveDiagnostics.retainedEvents=liveDirect.RetainedEvents;
        liveDiagnostics.completionSeconds=liveDirect.CompletionSeconds; liveDiagnostics.maximumCompletionSeconds=liveDirect.MaximumCompletionSeconds;
        liveDiagnostics.pairPrimitiveTestUpperBound=liveDirect.PairPrimitiveUpperBound;
        liveDiagnostics.responseCount=liveDirect.ResponseCount; liveDiagnostics.solveSeconds=liveDirect.SolveSeconds;
        liveDiagnostics.audioSeconds=liveDirect.AudioSeconds; liveDiagnostics.maximumSolveSeconds=liveDirect.MaximumSolveSeconds;
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
        try { liveDirect?.Dispose(); }
        finally
        {
            liveDirect=null; takeWriter?.Dispose(); takeWriter=null;
            if (drain && monitorQueue!=null) monitorQueue.Complete();
            else
            {
                sessionAudio?.StopPcmStream(); playbackCapture?.Dispose();
                monitorQueue=null; playbackCapture=null; sessionAudio=null;
            }
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
        int index = takeObjects.FindIndex(b => b.transform == target);
        if (index >= 0) takeCuts.Add(index);
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
            var flags = scene.objects[i].primitiveCount > 0 ? AcousticTake.StateFlags.Geometry : 0;
            var surface=binding.transform.GetComponent<AcousticSurface>();
            if (surface != null && !surface.enabled) flags &= ~AcousticTake.StateFlags.Geometry;
            for (int s=0; s<scene.sources.Length; s++) if (scene.sources[s].objectIndex == i)
            {
                var component = binding.transform.GetComponent<RaytracedAudioSource>();
                if (component == null || component.enabled) flags |= AcousticTake.StateFlags.Source;
            }
            if (Array.IndexOf(scene.listeners,i) >= 0) flags |= AcousticTake.StateFlags.Listener;
            if (takeCuts.Contains(i) || (sample == previousTakeSample && (previousTakeState[i].flags & AcousticTake.StateFlags.Cut) != 0))
                flags |= AcousticTake.StateFlags.Cut;
            takeState[i] = AcousticTake.State.Observe(binding.transform, flags);
            if (scene.dimension == AcousticTake.Dimension.Planar2D &&
                (Mathf.Abs(takeState[i].position.z) > .0001f || Mathf.Abs(Vector3.Dot(binding.transform.forward,Vector3.forward)) < .99999f))
                throw new InvalidOperationException("Planar capture uses the XY plane; use a 3D take for out-of-plane objects.");
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

    void RecordLegacyVoice(AcousticTake.VoiceAction action, long offset = 0)
    {
        if (captureScene != null && source != null && Array.IndexOf(captureSources,source) >= 0)
            RecordSourceEvent(source,action,inputGain,loopAudio,offset);
    }

    void LateUpdate()
    {
        try
        {
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
                long sample=(long)Math.Round(replaySeconds*replayTake.Metadata.sampleRate);
                sample=Math.Max(0,Math.Min(sample,replayTake.Metadata.durationSamples));
                replayTake.Evaluate(sample,takeState); ApplyTakeState(takeState);
                if (sample == replayTake.Metadata.durationSamples) replayPlaying=false;
            }
        }
        catch (Exception e)
        {
            if (captureScene != null) CloseCapture(false);
            if (replayTake != null || restoreTakeState != null) StopTakeReplay();
            TakeStatus="Take stopped: "+e.Message;
            Debug.LogError(TakeStatus,this);
        }
    }

    public void BeginTakeReplay(string directory, bool play = true)
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
            StopStreaming();
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
            TakeStatus="Replaying recorded poses (audio renders separately): "+directory;
        }
        catch { loaded.Dispose(); StopTakeReplay(); throw; }
    }

    public void SetTakeReplayPlaying(bool play)
    {
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
        StopStreaming();
    }

    void DisposeTakeSession()
    {
        CloseCapture(false);
        StopTakeReplay();
    }
}
