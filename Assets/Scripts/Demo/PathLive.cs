using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using UnityEngine;
using UnityEngine.Profiling;
using UnityEngine.SceneManagement;

// The live showcase: the path method running live in each of its scenes (docs/benchmarks/path-denoise-2026-11-12,
// parts 26 and 38; build and controls in README.md, "Live showcase"). In a scene with a PathLiveScene it turns the
// SceneSession into: persistent rays under population control traced every audio update on the GPU, the persistent-ray
// denoiser, and the path reverb built and convolved on the GPU (GpuAudio.Path); the audio thread only copies finished
// PCM. The scene's PathLiveScene names the sources that play. Key: N denoiser on/off (the HUD and the scene picker are
// RaytraceControlPanel's). Added at every scene load by the player build (scripting define PATH_LIVE) or, in the editor,
// by Acoustics > Play live showcase. With -pathLiveSmoke DIR it runs the scene's scripted smoke walk instead of reading
// the player's input, for each scene of -pathLiveScenes A,B (default: the first scene) in turn: the output to
// DIR/<scene>/output.wav (the session's monitor capture), frame times and audio underruns to smoke.json, then the next
// scene or quit. -pathLiveShots also saves a screenshot at each phase of the walk.
[DefaultExecutionOrder(-200)]
public sealed class PathLive : MonoBehaviour
{
    // The showcase's scenes, in the scene picker's order; the build includes these and starts with the first.
    public static readonly (string title, string path)[] Scenes =
    {
        ("Church", "Assets/Scenes/Church.unity"),
        ("Train station", "Assets/Scenes/TrainStation.unity"),
    };
    public static string Title(Scene scene) => Scenes.FirstOrDefault(s => s.path == scene.path).title ?? scene.name;

    // 11,000 rays in flight per source: ten windowed smoke runs of the Church at 11,000 and three at 16,000 had no audio
    // underrun (README part 41; part 38's runs at 11,000 lost samples to main-thread stalls that did not recur).
    // A scene's PathLiveScene.population overrides it, and the command line both: -pathPopulation N.
    public int population = 11000;
    // The auto-budget (PathPopulationBudget, README part 34): the GPU milliseconds per update the tracer, the denoiser and
    // the reverb may take together; the population then moves within the denoiser's trained range. Zero: a fixed population.
    // Command line: -pathBudget MS. Off by default: it reads the profiler's per-sample GPU times, which only development
    // players and the editor report (README part 38).
    public float budgetMs;
    PathPopulationBudget budget;
    public float horizonSeconds = 6;
    public int leadSamples = 6144;
    public bool denoise = true;

    // For the HUD: one line of live figures, and the per-stage GPU times when the player reports them (development only).
    public string Stats { get; private set; } = "";
    public string StageTimes { get; private set; } = "";
    public bool IsSmokeRun => smoke != null;
    public bool SmokeShowsMenu { get; private set; }   // the smoke run's last screenshot is of the menu

    SceneSession session;
    PathLiveScene showcase;
    RayTraceManager manager;
    string smoke;
    bool shots;
    float smokeStart = -1;
    readonly List<float> frameMs = new List<float>();
    readonly List<float> traceMs = new List<float>(), reverbMs = new List<float>(), denoiseMs = new List<float>(), gpuMs = new List<float>(), frameGpuMs = new List<float>();
    readonly FrameTiming[] frameTiming = new FrameTiming[1];
    float frameGpu = float.NaN;
    System.Text.StringBuilder timeline;   // smoke run: per frame with updates, time, phase, population, GPU ms (timeline.csv)
    Recorder traceRecorder, reverbRecorder, denoiseRecorder;
    long lastUpdates;
    float inFlightAt;
    bool finishing;

    const string Define = "PATH_LIVE";
    public const string EditorFlag = "PathLive.Play";
    static readonly Queue<string> smokeScenes = new Queue<string>();

    [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.AfterSceneLoad)]
    static void Bootstrap()
    {
        bool wanted = false;
#if PATH_LIVE
        wanted = true;
#endif
#if UNITY_EDITOR
        wanted |= UnityEditor.SessionState.GetBool(EditorFlag, false);
#endif
        if (!wanted) return;
        var args = Environment.GetCommandLineArgs();
        if (Array.IndexOf(args, "-pathLiveSmoke") >= 0)
        {
            int at = Array.IndexOf(args, "-pathLiveScenes");
            var names = at >= 0 && at + 1 < args.Length ? args[at + 1].Split(',') : new[] {SceneManager.GetActiveScene().name};
            foreach (string name in names)
            {
                var match = Scenes.FirstOrDefault(s => Path.GetFileNameWithoutExtension(s.path) == name.Trim());
                if (match.path == null) { Debug.LogError($"{Define}: no showcase scene {name}"); Application.Quit(2); return; }
                smokeScenes.Enqueue(match.path);
            }
            var dog = new GameObject("Path live smoke watchdog"); DontDestroyOnLoad(dog);
            dog.AddComponent<Watchdog>().seconds = 180 * smokeScenes.Count;
        }
        SceneManager.sceneLoaded += (scene, _) => Attach(scene);
        Attach(SceneManager.GetActiveScene());
        if (smokeScenes.Count > 0 && SceneManager.GetActiveScene().path != smokeScenes.Peek()) SceneManager.LoadScene(smokeScenes.Peek());
    }

    // Runs before the scene's Start methods (sceneLoaded and AfterSceneLoad both follow Awake), so the session opens its
    // take already configured.
    static void Attach(Scene scene)
    {
        var showcase = scene.GetRootGameObjects().SelectMany(r => r.GetComponentsInChildren<PathLiveScene>()).FirstOrDefault();
        var session = showcase != null ? showcase.GetComponent<SceneSession>() : null;
        if (session == null) { Debug.LogWarning($"{Define}: {scene.name} has no PathLiveScene next to its SceneSession."); return; }
        if (session.GetComponent<PathLive>() == null) session.gameObject.AddComponent<PathLive>();
    }

    void Awake()
    {
        session = GetComponent<SceneSession>(); showcase = GetComponent<PathLiveScene>(); manager = session.manager;
        var args = Environment.GetCommandLineArgs();
        int at = Array.IndexOf(args, "-pathLiveSmoke");
        if (at >= 0 && at + 1 < args.Length && smokeScenes.Count > 0 && smokeScenes.Peek() == gameObject.scene.path)
            smoke = Path.Combine(Path.GetFullPath(args[at + 1]), gameObject.scene.name);
        shots = Array.IndexOf(args, "-pathLiveShots") >= 0;
        if (showcase.population > 0) population = showcase.population;
        int n = Array.IndexOf(args, "-pathPopulation");
        if (n >= 0 && n + 1 < args.Length) population = int.Parse(args[n + 1]);
        int b = Array.IndexOf(args, "-pathBudget");
        if (b >= 0 && b + 1 < args.Length) budgetMs = float.Parse(args[b + 1], System.Globalization.CultureInfo.InvariantCulture);
        Configure();
    }

    // The session's settings for the live path method (the offline Church recipe of church.py, binaural, at the
    // output rate), the showcase's sources, and the scene's actors but its scripted sequence (weather, voices).
    void Configure()
    {
        var sources = showcase.sources.Where(s => s != null).Distinct().ToList();
        if (sources.Count == 0) throw new InvalidOperationException($"{gameObject.scene.name}: PathLiveScene names no sources.");
        foreach (var source in manager.takeSources.Where(s => s != null && !sources.Contains(s))) source.gameObject.SetActive(false);
        manager.takeSources = sources;
        session.startSounds = showcase.startLooped.Where(sources.Contains).ToArray();
        foreach (var source in session.startSounds) source.loop = true;
        session.actors = session.actors.Where(a => !(a is SceneSequenceActor)).ToArray();
        session.SaveOnR = false;
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
        // The station's train (smoke and listening runs): -stationTrainSpeed V and -stationTrainAcceleration A (m/s, m/s^2)
        // replace StationDemo's cruise speed and acceleration, before the session initialises its actors.
        foreach (var demo in session.actors.OfType<StationDemo>())
        {
            var arguments = Environment.GetCommandLineArgs();
            int v = Array.IndexOf(arguments, "-stationTrainSpeed"), a = Array.IndexOf(arguments, "-stationTrainAcceleration");
            if (v >= 0 && v + 1 < arguments.Length) demo.trainSpeed = float.Parse(arguments[v + 1], System.Globalization.CultureInfo.InvariantCulture);
            if (a >= 0 && a + 1 < arguments.Length) demo.trainAcceleration = float.Parse(arguments[a + 1], System.Globalization.CultureInfo.InvariantCulture);
        }
        traceRecorder = Recorder.Get(MethodTransport.TraceSample); reverbRecorder = Recorder.Get("Acoustics.GPU.PathKernels");
        denoiseRecorder = Recorder.Get(PathDenoiser.ProfilerSample);
        foreach (var r in new[] {traceRecorder, reverbRecorder, denoiseRecorder}) if (r.isValid) r.enabled = true;
        // Every showcase scene marks where its sounds are (the HUD's H hides the marks with it).
        var view = session.player.GetComponentInChildren<Camera>();
        if (view != null && view.GetComponent<AudioSourceOverlay>() == null) view.gameObject.AddComponent<AudioSourceOverlay>().manager = manager;
    }

    void Start()
    {
        // Every visit starts at the scene's first spawn point, the smoke run's start too.
        var spawn = session.gameObject.scene.GetRootGameObjects().SelectMany(r => r.GetComponentsInChildren<Transform>(true)).FirstOrDefault(t => t.name == "Player spawn 0");
        if (spawn != null) { session.player.SetPositionAndRotation(spawn.position, spawn.rotation); Physics.SyncTransforms(); }
        if (smoke == null) return;
        Directory.CreateDirectory(smoke);
        session.ReadPlayerInput = false;
        session.SampleInput += Script;
        manager.audioManager.muteOutput = true;
        Application.runInBackground = true;   // the smoke run never takes focus; without this the player pauses
        // Evidence that the scene is drawn (the smoke run may be windowless, -batchmode): camera renders per frame.
        Camera.onPostRender += CountRender;
        UnityEngine.Rendering.RenderPipelineManager.endCameraRendering += CountRender;
    }
    void CountRender(Camera _) => cameraRenders++;
    void CountRender(UnityEngine.Rendering.ScriptableRenderContext _, Camera __) => cameraRenders++;
    void OnDestroy()
    {
        Camera.onPostRender -= CountRender;
        UnityEngine.Rendering.RenderPipelineManager.endCameraRendering -= CountRender;
    }
    long cameraRenders;
    public string ScriptPhase { get; private set; } = "";

    // The smoke walk: stand, walk the scene's route (pressing E once after its interactAfter waypoint), then hold. Each
    // leg steers toward its waypoint (movement in the player's frame) and ends there, or when it has not come 10 cm closer
    // for 2 s (blocked: a step, a rail, a closed door); ticks of the session's fixed step.
    long legStart = -1;
    int leg = -1;
    bool scriptDone, pressed;
    float best = float.MaxValue, bestAt;
    void Script(long tick)
    {
        if (legStart < 0) { legStart = tick; Shot(); }
        float seconds = (tick - legStart) * Time.fixedDeltaTime;
        var movement = Vector2.zero; bool interact = false;
        var route = showcase.route;
        void Next() { leg++; legStart = tick; best = float.MaxValue; bestAt = 0; pressed = false; Shot(); }
        if (leg < 0) { ScriptPhase = "standing"; if (seconds >= showcase.standSeconds) Next(); }
        else if (leg < route.Length && pressed) { ScriptPhase = "interacting"; if (seconds >= 3) Next(); }
        else if (leg < route.Length)
        {
            ScriptPhase = $"to waypoint {leg + 1}";
            var to = route[leg] - session.player.position; to.y = 0;
            if (to.magnitude < best - .1f) { best = to.magnitude; bestAt = seconds; }
            if (to.magnitude < .4f || seconds - bestAt > 2)
            {
                if (leg == showcase.interactAfter) { pressed = interact = true; legStart = tick; }
                else Next();
            }
            else
            {
                to.Normalize();
                movement = new Vector2(Vector3.Dot(to, session.player.right), Vector3.Dot(to, session.player.forward));
            }
        }
        else { ScriptPhase = "holding"; if (seconds >= showcase.holdSeconds) scriptDone = true; }
        session.SubmitInput(movement, Vector2.zero, false, interact);
    }
    void Shot() { if (shots) ScreenCapture.CaptureScreenshot(Path.Combine(smoke, $"shot-{leg + 1:D2}.png")); }

    void Update()
    {
        if (Input.GetKeyDown(KeyCode.N) && manager.LiveRenderer != null) manager.LiveRenderer.PathDenoising = !manager.LiveRenderer.PathDenoising;
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
        Stats = $"{(method.LastRaysInFlight >= 0 ? method.LastRaysInFlight.ToString("N0") : "–")} rays in flight{(budget != null ? $" (auto-budget {budgetMs:F1} ms)" : "")}   ·   " +
                $"denoiser {(live.HasPathDenoiser ? live.PathDenoising ? "on" : "off" : "not loaded")}   ·   " +
                $"frame {Ms(Recent(frameMs))}, GPU {Ms(Recent(frameGpuMs))}   ·   audio underruns {queue?.UnderrunSamples ?? 0}";
        StageTimes = traceMs.Count > 0 ? $"GPU per update: tracer {Ms(Recent(traceMs))}, reverb {Ms(Recent(reverbMs))}, denoiser {Recent(denoiseMs):F2} ms" : "";
        if (smoke != null) Smoke(queue);
    }

    static string Ms(float v) => float.IsNaN(v) ? "–" : $"{v:F1} ms";
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
        StartCoroutine(Finish(report));
    }

    System.Collections.IEnumerator Finish(SmokeReport report)
    {
        // The monitor capture closes on the AudioManager's updates and the take drains; then its output.wav is copied
        // next to the report.
        float until = Time.unscaledTime + 30;
        while ((manager.audioManager.OutputCapture != null || manager.Mode == RayTraceManager.SessionMode.Draining) && Time.unscaledTime < until) yield return null;
        string wav = Path.Combine(report.takePath ?? "", "monitor", "output.wav");
        // A capture still open after 30 s (no audio callbacks: no output device) leaves no output.wav; the run goes on.
        try { if (File.Exists(wav)) File.Copy(wav, report.outputWav, true); }
        catch (IOException e) { Debug.LogWarning($"PATH_LIVE_SMOKE no output.wav: {e.Message}"); }
        Debug.Log($"PATH_LIVE_SMOKE {JsonUtility.ToJson(report)}");
        if (shots)
        {
            SmokeShowsMenu = true;
            for (int i = 0; i < 3; i++) yield return null;   // the panel refreshes every 0.1 s
            yield return new WaitForSecondsRealtime(.2f);
            ScreenCapture.CaptureScreenshot(Path.Combine(smoke, "menu.png"));
            yield return null; yield return null;
        }
        smokeScenes.Dequeue();
        // The next scene loads the way the scene picker loads it.
        if (smokeScenes.Count > 0) SceneManager.LoadScene(smokeScenes.Peek());
        else Application.Quit();
    }

    // The smoke run never outlives three minutes per scene, whatever happens to the scenes.
    sealed class Watchdog : MonoBehaviour
    {
        public float seconds = 180;
        void Update() { if (Time.realtimeSinceStartup > seconds) { Debug.LogError($"PATH_LIVE_SMOKE abandoned after {seconds:F0} s"); Application.Quit(2); enabled = false; } }
    }
}
