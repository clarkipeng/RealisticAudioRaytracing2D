using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using UnityEngine;
using UnityEngine.Profiling;

// The playable Church with the path method live (docs/benchmarks/path-denoise-2026-11-12, part 26; build and
// controls in README.md, "Playable Church"). It turns the Church scene's session into: persistent rays under
// population control traced every audio update on the GPU, the persistent-ray denoiser, and the path reverb
// built and convolved on the GPU (GpuAudio.Path); the audio thread only copies finished PCM. One source, the
// organ, plays in a loop; the doors open and close with E (SlidingDoor). Keys: N denoiser on/off, H HUD.
// Added to the Church scene at load by the player build (scripting define PATH_LIVE_CHURCH) or, in the editor,
// by Acoustics > Play Church live (path method). With -pathLiveSmoke DIR on the command line it runs the
// scripted smoke walk instead of reading the player's input: it records the output to DIR/output.wav (the
// session's monitor capture), frame times and audio underruns to DIR/smoke.json, then quits.
[DefaultExecutionOrder(-200)]
public sealed class PathLiveChurch : MonoBehaviour
{
    // 8,000 rays in flight: smoke runs beside the Church's own rendering had no audio underrun at 8,000, but at 11,000 one
    // of two runs lost 1,972 samples when the doors opened, and 16,000 and 23,000 lost more (README part 38).
    // Command line: -pathPopulation N.
    public int population = 8000;
    // The auto-budget (PathPopulationBudget, README part 34): the GPU milliseconds per update the tracer, the denoiser and
    // the reverb may take together; the population then moves within the denoiser's trained range. Zero: a fixed population.
    // Command line: -pathBudget MS. Off by default: it reads the profiler's per-sample GPU times, which only development
    // players and the editor report (README part 38).
    public float budgetMs;
    PathPopulationBudget budget;
    public float horizonSeconds = 6;
    public int leadSamples = 6144;
    public bool denoise = true;
    public bool hud = true;

    SceneSession session;
    RayTraceManager manager;
    string smoke;
    float smokeStart = -1;
    readonly List<float> frameMs = new List<float>();
    readonly List<float> traceMs = new List<float>(), reverbMs = new List<float>(), denoiseMs = new List<float>(), gpuMs = new List<float>(), frameGpuMs = new List<float>();
    readonly FrameTiming[] frameTiming = new FrameTiming[1];
    float frameGpu = float.NaN;
    System.Text.StringBuilder timeline;   // smoke run: per frame with updates, time, phase, population, GPU ms (timeline.csv)
    Recorder traceRecorder, reverbRecorder, denoiseRecorder;
    long lastUpdates;
    float inFlightAt;
    string status = "";
    bool finishing;

    const string Define = "PATH_LIVE_CHURCH";
    public const string EditorFlag = "PathLiveChurch.Play";

    [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.AfterSceneLoad)]
    static void Bootstrap()
    {
        bool wanted = false;
#if PATH_LIVE_CHURCH
        wanted = true;
#endif
#if UNITY_EDITOR
        wanted |= UnityEditor.SessionState.GetBool(EditorFlag, false);
#endif
        if (!wanted) return;
        if (Array.IndexOf(Environment.GetCommandLineArgs(), "-pathLiveSmoke") >= 0 && FindAnyObjectByType<Watchdog>() == null)
        {
            var dog = new GameObject("Path live smoke watchdog"); DontDestroyOnLoad(dog); dog.AddComponent<Watchdog>();
        }
        var session = FindAnyObjectByType<SceneSession>();
        if (session == null || session.gameObject.scene.name != "Church") { Debug.LogWarning($"{Define}: no Church session in the loaded scene."); return; }
        if (session.GetComponent<PathLiveChurch>() == null) session.gameObject.AddComponent<PathLiveChurch>();
    }

    void Awake()
    {
        session = GetComponent<SceneSession>(); manager = session.manager;
        var args = Environment.GetCommandLineArgs();
        int at = Array.IndexOf(args, "-pathLiveSmoke");
        if (at >= 0 && at + 1 < args.Length) smoke = Path.GetFullPath(args[at + 1]);
        int n = Array.IndexOf(args, "-pathPopulation");
        if (n >= 0 && n + 1 < args.Length) population = int.Parse(args[n + 1]);
        int b = Array.IndexOf(args, "-pathBudget");
        if (b >= 0 && b + 1 < args.Length) budgetMs = float.Parse(args[b + 1], System.Globalization.CultureInfo.InvariantCulture);
        Configure();
    }

    // The session's settings for the live path method (the offline Church recipe of church.py, binaural, at the
    // output rate), one source (the organ) and the doors; the weather and voices of the authored scene are off.
    void Configure()
    {
        var organ = manager.takeSources.Where(s => s != null).OrderBy(s => s.name == "Pipe organ" ? 0 : 1)
            .ThenBy(s => (s.transform.position - new Vector3(0, 5, 18)).sqrMagnitude).First();
        organ.loop = true;
        foreach (var source in manager.takeSources.Where(s => s != null && s != organ)) source.gameObject.SetActive(false);
        manager.takeSources = new List<RaytracedAudioSource> {organ};
        session.startSounds = new[] {organ};
        session.actors = session.actors.Where(a => a is SlidingDoor).ToArray();
        session.retainFullReference = smoke != null;   // the smoke run keeps the session's monitor output (output.wav)
        var s = manager.directSettings;
        s.method = TakeRender.Method.Path; s.transport = TakeRender.Transport.Raytraced; s.synthesis = TakeRender.Synthesis.Spectral;
        s.audioBackend = TakeRender.AudioBackend.Gpu; s.listenerOutput = TakeRender.ListenerOutput.Binaural;
        s.pathPersistent = true; s.pathPopulation = population; s.pathDenoise = denoise;
        if (budgetMs > 0) { s.pathPopulationMaximum = 23000; budget = new PathPopulationBudget(budgetMs, population, 4000, 23000); }
        s.pathEdgeSliceUpdates = 4;   // the edge distribution refreshed every 4th update (README part 30)
        s.pathIterationCap = 16;   // the long-chain lottery (README part 28): unbiased, about a fifth off the tracer at equal noise
        s.responseStepSamples = 1024; s.fftSize = 1024; s.spectralHop = 256; s.transitionSamples = 512;
        s.maximumDelaySeconds = horizonSeconds; s.tailSamples = (int)(horizonSeconds * manager.sampleRate);
        s.maximumResponsesPerBatch = 2; s.maximumPendingObservations = 16; s.maxBounces = 64; s.useBvh = true;
        s.methodRays = 512; s.methodRouletteFloor = 1e-6f; s.methodEdgeProbability = .5f; s.specularDetectionRadius = 2;
        s.delayFilter = TakeRender.DelayFilter.WindowedSinc; s.sincRadius = 16; s.airAbsorption = true;
        s.pathHistogramExport = s.pathHistogramImport = "";
        manager.monitorLive = true; manager.monitorLeadSamples = leadSamples;
        manager.monitorCapacitySamples = Math.Max(manager.monitorCapacitySamples, 4 * leadSamples);
        traceRecorder = Recorder.Get(MethodTransport.TraceSample); reverbRecorder = Recorder.Get("Acoustics.GPU.PathKernels");
        denoiseRecorder = Recorder.Get(PathDenoiser.ProfilerSample);
        foreach (var r in new[] {traceRecorder, reverbRecorder, denoiseRecorder}) if (r.isValid) r.enabled = true;
    }

    void Start()
    {
        if (smoke == null) return;
        Directory.CreateDirectory(smoke);
        var spawn = session.gameObject.scene.GetRootGameObjects().SelectMany(r => r.GetComponentsInChildren<Transform>(true)).FirstOrDefault(t => t.name == "Player spawn 0");
        if (spawn != null) { session.player.SetPositionAndRotation(spawn.position, spawn.rotation); Physics.SyncTransforms(); }
        session.ReadPlayerInput = false;
        session.SampleInput += Script;
        manager.audioManager.muteOutput = true;
        Application.runInBackground = true;   // the smoke run never takes focus; without this the player pauses
        // Evidence that the scene is drawn (the smoke run may be windowless, -batchmode): camera renders per frame.
        Camera.onPostRender += _ => cameraRenders++;
        UnityEngine.Rendering.RenderPipelineManager.endCameraRendering += (_, _) => cameraRenders++;
        hud = false;
        // Waypoints from the scene itself: the organ (stopping short of it), the doors' centre from inside, then out
        // through them and along the facade.
        var organ = manager.takeSources[0].transform.position;
        var doors = session.actors.OfType<SlidingDoor>().ToArray();
        var door = doors.Length > 0 ? doors.Aggregate(Vector3.zero, (a, d) => a + d.transform.position) / doors.Length : organ;
        float y = session.player.position.y;
        Vector3 Flat(Vector3 v) => new Vector3(v.x, y, v.z);
        var inward = Flat(organ) - Flat(door); inward.y = 0; inward = inward.sqrMagnitude > 0 ? inward.normalized : Vector3.forward;
        var along = Vector3.Cross(Vector3.up, inward);
        route = new[] {Flat(organ) - inward * 3, Flat(door) + inward * 1.2f, Flat(door) - inward * 8, Flat(door) - inward * 8 + along * 8};
    }
    Vector3[] route;
    long cameraRenders;
    public string ScriptPhase { get; private set; } = "";

    // The smoke walk: stand 5 s, walk to the organ, back to the doors, open them, walk out and along the facade. Each leg
    // steers toward its waypoint (movement in the player's frame) and ends there or after its time limit; ticks of the
    // session's fixed step.
    long firstTick = -1, legStart;
    int leg = -1;
    bool scriptDone, pressed;
    float best = float.MaxValue, bestAt;
    void Script(long tick)
    {
        if (firstTick < 0) { firstTick = tick; legStart = tick; }
        float seconds = (tick - legStart) * Time.fixedDeltaTime;
        var movement = Vector2.zero; bool interact = false;
        void Next() { leg++; legStart = tick; best = float.MaxValue; bestAt = 0; }
        switch (leg)
        {
            case -1: ScriptPhase = "standing"; if (seconds >= 5) Next(); break;
            case 0: case 1: case 3: case 4:
            {
                int target = leg <= 1 ? leg : leg - 1;
                ScriptPhase = new[] {"to the organ", "to the doors", "outside", "along the facade"}[target];
                var to = route[target] - session.player.position; to.y = 0;
                // A leg ends at its waypoint, or when it has not come 10 cm closer for 2 s (blocked: a step, a rail, a door).
                if (to.magnitude < best - .1f) { best = to.magnitude; bestAt = seconds; }
                if (to.magnitude < .4f || seconds - bestAt > 2) { Next(); break; }
                to.Normalize();
                movement = new Vector2(Vector3.Dot(to, session.player.right), Vector3.Dot(to, session.player.forward));
                break;
            }
            case 2: ScriptPhase = "opening the doors"; interact = !pressed; pressed = true; if (seconds >= 3) Next(); break;
            case 5: ScriptPhase = "done"; if (seconds >= 2) scriptDone = true; break;
        }
        session.SubmitInput(movement, Vector2.zero, false, interact);
    }

    void Update()
    {
        if (Input.GetKeyDown(KeyCode.N) && manager.LiveRenderer != null) manager.LiveRenderer.PathDenoising = !manager.LiveRenderer.PathDenoising;
        if (Input.GetKeyDown(KeyCode.H)) hud = !hud;
        var live = manager.LiveRenderer;
        if (live == null) return;
        var method = live.Method;
        long updates = method.AudioUpdates, done = updates - lastUpdates; lastUpdates = updates;
        frameMs.Add(Time.unscaledDeltaTime * 1000);
        // The whole frame's GPU time (the scene's rendering and the audio's compute), which release players report
        // through FrameTimingManager (Player Settings: frame timing stats), unlike the profiler's per-sample GPU times.
        FrameTimingManager.CaptureFrameTimings();
        frameGpu = FrameTimingManager.GetLatestTimings(1, frameTiming) > 0 ? (float)frameTiming[0].gpuFrameTime : float.NaN;
        if (frameGpu > 0) frameGpuMs.Add(frameGpu);
        if (done > 0)
        {
            float gpu = 0;
            if (traceRecorder.isValid && traceRecorder.gpuElapsedNanoseconds > 0) { traceMs.Add(traceRecorder.gpuElapsedNanoseconds * 1e-6f / done); gpu += traceMs[^1]; }
            if (reverbRecorder.isValid && reverbRecorder.gpuElapsedNanoseconds > 0) { reverbMs.Add(reverbRecorder.gpuElapsedNanoseconds * 1e-6f / done); gpu += reverbMs[^1]; }
            if (denoiseRecorder.isValid && denoiseRecorder.gpuElapsedNanoseconds > 0) { denoiseMs.Add(denoiseRecorder.gpuElapsedNanoseconds * 1e-6f / done); gpu += denoiseMs[^1]; }
            if (gpu > 0) gpuMs.Add(gpu);
            if (budget != null && gpu > 0) method.Population = budget.Observe(gpu);
            if (smoke != null && smokeStart >= 0) (timeline ??= new System.Text.StringBuilder("seconds,phase,population,gpuMs,frameMs,frameGpuMs,x,z,doorsOpen,underrunSamples\n")).Append(
                FormattableString.Invariant($"{Time.unscaledTime - smokeStart:F3},{ScriptPhase},{method.Population},{gpu:F3},{Time.unscaledDeltaTime * 1000:F2},{frameGpu:F2},{session.player.position.x:F2},{session.player.position.z:F2},{session.actors.OfType<SlidingDoor>().Count(d => d.IsOpen)},{manager.audioManager.OutputQueue?.UnderrunSamples ?? 0}\n"));
        }
        if (Time.unscaledTime > inFlightAt) { method.RequestRaysInFlight(); inFlightAt = Time.unscaledTime + .5f; }
        var queue = manager.audioManager.OutputQueue;
        string stages = traceMs.Count > 0 ? $"{Recent(traceMs):F1} ms per update (GPU tracer), reverb {Recent(reverbMs):F1} ms, denoiser {Recent(denoiseMs):F2} ms"
                                          : "per-stage GPU times need a development player";
        status = $"path method, persistent rays: {stages}; whole frame on the GPU {Recent(frameGpuMs):F1} ms\n" +
                 $"rays in flight {method.LastRaysInFlight:N0} (population {method.Population:N0}{(budget != null ? $", auto-budget {budgetMs:F1} ms" : "")}, {method.NewRays:N0} new per update)\n" +
                 $"denoiser {(live.HasPathDenoiser ? live.PathDenoising ? "on" : "off" : "not loaded")} (N)   frame {Recent(frameMs):F1} ms   " +
                 $"audio underruns {queue?.UnderrunSamples ?? 0} samples   doors: E   HUD: H";
        if (smoke != null) Smoke(queue);
    }

    static float Recent(List<float> v) { if (v.Count == 0) return float.NaN; int n = Math.Min(30, v.Count); float s = 0; for (int i = v.Count - n; i < v.Count; i++) s += v[i]; return s / n; }

    [Serializable] sealed class SmokeReport
    {
        public string scene, takePath, outputWav;
        public int population, newRaysPerUpdate, sampleRate, leadSamples, frames, raysInFlightLast;
        public long audioUpdates, underrunSamples, overBudgetCallbacks, callbacks, lateSamples, overflowSamples;
        public float seconds, frameMsMedian, frameMsP99, frameMsMax, maximumCallbackMs, traceMsMedian, traceMsP99, reverbMsMedian, denoiseMsMedian, gpuMsMedian, gpuMsP90, budgetMs;
        public bool gpuTimings, batchMode;
        public float cameraRendersPerFrame, frameGpuMsMedian, frameGpuMsP99;
        public bool denoiser;
    }

    void Smoke(AudioManager.PcmQueue queue)
    {
        if (smokeStart < 0 && manager.IsRecordingTake) smokeStart = Time.unscaledTime;
        if (smokeStart < 0 || finishing || !scriptDone && Time.unscaledTime - smokeStart < 120) return;
        finishing = true;
        var live = manager.LiveRenderer; var method = live.Method;
        float P(List<float> v, double q) { if (v.Count == 0) return float.NaN; var s = v.Skip(Math.Min(v.Count - 1, 60)).OrderBy(x => x).ToArray(); return s[Math.Min(s.Length - 1, (int)(q * s.Length))]; }
        var report = new SmokeReport
        {
            scene = gameObject.scene.name, population = method.Population, newRaysPerUpdate = method.NewRays, sampleRate = manager.sampleRate, leadSamples = leadSamples,
            frames = frameMs.Count, raysInFlightLast = method.LastRaysInFlight, audioUpdates = method.AudioUpdates, seconds = Time.unscaledTime - smokeStart,
            underrunSamples = queue.UnderrunSamples, overBudgetCallbacks = queue.OverBudgetCallbacks, callbacks = queue.CallbackCount,
            lateSamples = queue.LateSamples, overflowSamples = queue.OverflowSamples,
            maximumCallbackMs = (float)(queue.MaximumCallbackTicks * 1000.0 / System.Diagnostics.Stopwatch.Frequency),
            frameMsMedian = P(frameMs, .5), frameMsP99 = P(frameMs, .99), frameMsMax = P(frameMs, 1),
            traceMsMedian = P(traceMs, .5), traceMsP99 = P(traceMs, .99), reverbMsMedian = P(reverbMs, .5), denoiseMsMedian = P(denoiseMs, .5),
            gpuMsMedian = P(gpuMs, .5), gpuMsP90 = P(gpuMs, .9), budgetMs = budgetMs, gpuTimings = traceMs.Count > 0,
            batchMode = Application.isBatchMode, frameGpuMsMedian = P(frameGpuMs, .5), frameGpuMsP99 = P(frameGpuMs, .99), cameraRendersPerFrame = frameMs.Count > 0 ? cameraRenders / (float)frameMs.Count : 0,
            denoiser = live.PathDenoising, takePath = manager.takePath, outputWav = Path.Combine(smoke, "output.wav")
        };
        File.WriteAllText(Path.Combine(smoke, "smoke.json"), JsonUtility.ToJson(report, true));
        if (timeline != null) File.WriteAllText(Path.Combine(smoke, "timeline.csv"), timeline.ToString());
        manager.EndTakeRecording();
        StartCoroutine(Quit(report));
    }

    System.Collections.IEnumerator Quit(SmokeReport report)
    {
        // The monitor capture closes on the AudioManager's updates; then its output.wav is copied next to the report.
        float until = Time.unscaledTime + 30;
        while (manager.audioManager.OutputCapture != null && Time.unscaledTime < until) yield return null;
        string wav = Path.Combine(report.takePath ?? "", "monitor", "output.wav");
        if (File.Exists(wav)) File.Copy(wav, report.outputWav, true);
        Debug.Log($"PATH_LIVE_SMOKE {JsonUtility.ToJson(report)}");
        Application.Quit();
    }

    // The smoke run never outlives three minutes, whatever happens to the scene.
    sealed class Watchdog : MonoBehaviour
    {
        void Update() { if (Time.realtimeSinceStartup > 180) { Debug.LogError("PATH_LIVE_SMOKE abandoned after 180 s"); Application.Quit(2); enabled = false; } }
    }

    void OnGUI()
    {
        if (!hud || string.IsNullOrEmpty(status)) return;
        GUI.Label(new Rect(12, 12, 900, 80), status);
    }
}
