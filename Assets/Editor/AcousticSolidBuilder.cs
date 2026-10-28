using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEditorInternal;
using UnityEngine;

// Keeps every loaded AcousticSolid's exterior current without a manual step.
// Boundaries are content-addressed by the solid's local-frame source hash, so
// whole-group motion never invalidates them. The component stores the current
// boundary (players and checkouts without Python can capture); Library keeps
// recent results so undo, duplicates and reverted edits never rerun the backend.
[InitializeOnLoad]
public static class AcousticSolidBuilder
{
    public const string PythonPreference = "Acoustics.SolidBakePython";
    public const string PythonVariable = "ACOUSTIC_SOLID_PYTHON";
    const double WorkerSeconds = 120, SettleSeconds = .5;
    const long WorkerBytes = 1L << 30;
    const int CacheEntries = 256;

    public enum State { Checking, Current, Building, Failed, Inactive }
    public sealed class Status
    {
        public readonly State state; public readonly string message;
        public Status(State state, string message) { this.state = state; this.message = message; }
    }

    static readonly Dictionary<AcousticSolid, Status> statuses = new Dictionary<AcousticSolid, Status>();
    // Input-determined or expensive failures are not retried until the input,
    // the Python setting or an explicit Retry changes. Missing Python is retried.
    static readonly Dictionary<string, string> failures = new Dictionary<string, string>();
    static readonly List<(AcousticSolid solid, string input, string hash)> queue = new List<(AcousticSolid, string, string)>();
    static Worker running;
    static bool dirty;
    static double changedAt;

    // Diagnostic: backend processes started this domain. Cache hits start none.
    public static int WorkerLaunches { get; private set; }

    public static string Python
    {
        get
        {
            string value = Environment.GetEnvironmentVariable(PythonVariable);
            return string.IsNullOrEmpty(value) ? EditorPrefs.GetString(PythonPreference, "python3") : value;
        }
    }

    static string ProjectRoot => Directory.GetParent(Application.dataPath).FullName;
    static string CacheFolder => Path.Combine(ProjectRoot, "Library", "AcousticSolidBoundaries");
    public static string CachePath(string hash) => Path.Combine(CacheFolder, hash + ".json");

    static AcousticSolidBuilder()
    {
        AcousticSolid.Regenerate = Build;
        ObjectChangeEvents.changesPublished += (ref ObjectChangeEventStream stream) => Refresh();
        Undo.undoRedoPerformed += Refresh;
        EditorSceneManager.sceneOpened += (_, _) => Refresh();
        // Saved scenes and prefabs carry a current boundary for teammates without Python.
        EditorSceneManager.sceneSaving += (scene, _) => Flush(solid => solid.gameObject.scene == scene);
        PrefabStage.prefabSaving += root => Flush(solid => solid.transform.IsChildOf(root.transform));
        EditorApplication.playModeStateChanged += change =>
        {
            if (change == PlayModeStateChange.ExitingEditMode) Flush(_ => true);
            if (change == PlayModeStateChange.EnteredEditMode) Refresh();
        };
        EditorApplication.update += Tick;
        AssemblyReloadEvents.beforeAssemblyReload += StopWorker;
        EditorApplication.quitting += StopWorker;
        Refresh();
    }

    public static void Refresh() { dirty = true; changedAt = EditorApplication.timeSinceStartup; }

    public static Status Describe(AcousticSolid solid) =>
        statuses.TryGetValue(solid, out var status) ? status : new Status(State.Checking, "Checking acoustic exterior...");

    public static void Retry()
    {
        failures.Clear();
        Refresh(); changedAt -= SettleSeconds;
    }

    static void Tick()
    {
        if (running != null) Poll();
        if (EditorApplication.isPlayingOrWillChangePlaymode || EditorApplication.isCompiling || EditorApplication.isUpdating) return;
        if (dirty && EditorApplication.timeSinceStartup - changedAt >= SettleSeconds) { dirty = false; Scan(); }
        while (running == null && queue.Count > 0)
        {
            var (solid, input, hash) = queue[0]; queue.RemoveAt(0);
            if (solid == null) continue;
            try { running = new Worker(solid, input, hash, background: true); SetStatus(solid, State.Building, "Building acoustic exterior..."); }
            catch (Exception error) { Fail(solid, hash, error); }
        }
    }

    static void Scan()
    {
        foreach (var gone in statuses.Keys.Where(key => key == null).ToList()) statuses.Remove(gone);
        foreach (var solid in Loaded())
        {
            if (!Applies(solid, out string reason)) { Abandon(solid); SetStatus(solid, State.Inactive, reason); continue; }
            string hash, input;
            try { hash = solid.InputHash(Fallback(solid), out input, out _); }
            catch (Exception error) { Abandon(solid); Fail(solid, null, error); continue; }
            if (solid.IsCurrent(hash)) { Abandon(solid); SetStatus(solid, State.Current, Summary(solid.Compiled)); continue; }
            if (running != null && running.solid == solid && running.hash == hash) continue;
            Abandon(solid);
            if (TryCache(solid, hash)) continue;
            if (failures.TryGetValue(hash, out string known)) { SetStatus(solid, State.Failed, known); continue; }
            queue.Add((solid, input, hash)); SetStatus(solid, State.Building, "Queued for exterior regeneration.");
        }
    }

    static void Poll()
    {
        var worker = running;
        try { if (!worker.Wait(0)) return; }
        catch (Exception error) { StopWorker(); Fail(worker.solid, worker.hash, error); return; }
        StopWorker();
        // Applied through the cache, so the solid's current hash is rechecked first.
        Refresh(); changedAt -= SettleSeconds;
    }

    // Synchronous path for capture, save and Play entry. Throws a clear message on failure.
    static void Build(AcousticSolid solid, string input, string hash)
    {
        if (TryCache(solid, hash)) return;
        if (failures.TryGetValue(hash, out string known))
            throw new InvalidOperationException(solid.name + ": " + known);
        Abandon(solid);
        SetStatus(solid, State.Building, "Building acoustic exterior...");
        try
        {
            using (var worker = new Worker(solid, input, hash, background: false))
                while (!worker.Wait(100)) { }
            if (!TryCache(solid, hash)) throw new InvalidOperationException("The regenerated boundary could not be read from " + CachePath(hash) + ".");
        }
        catch (Exception error)
        {
            // The caller reports the exception; only record the failure here.
            Fail(solid, hash, error, log: false);
            throw new InvalidOperationException(solid.name + ": " + error.Message, error);
        }
    }

    static void Flush(Func<AcousticSolid, bool> include)
    {
        foreach (var solid in Loaded().Where(include).ToList())
        {
            if (!Applies(solid, out _)) continue;
            try
            {
                string hash = solid.InputHash(Fallback(solid), out string input, out _);
                if (!solid.IsCurrent(hash)) Build(solid, input, hash);
            }
            catch (Exception error)
            {
                UnityEngine.Debug.LogError(error.Message, solid);
                if (Describe(solid).state != State.Failed) SetStatus(solid, State.Failed, error.Message);
            }
        }
    }

    static bool Applies(AcousticSolid solid, out string reason)
    {
        var surface = solid.GetComponent<AcousticSurface>();
        reason = !solid.enabled ? "Disabled: members contribute their own geometry." :
            surface != null && surface.geometrySource == AcousticSurface.GeometrySource.None ? "Geometry Source is None: members contribute their own geometry." :
            solid.members == null || solid.members.Length == 0 ? "Add fixed child surfaces to Members." : null;
        return reason == null;
    }

    static AudioMaterial Fallback(AcousticSolid solid)
    {
        var manager = solid.GetComponentInParent<RayTraceManager>();
        if (manager == null) manager = UnityEngine.Object.FindFirstObjectByType<RayTraceManager>();
        return manager == null ? null : manager.takeDefaultMaterial;
    }

    static IEnumerable<AcousticSolid> Loaded()
    {
        var stage = PrefabStageUtility.GetCurrentPrefabStage();
        foreach (var solid in Resources.FindObjectsOfTypeAll<AcousticSolid>())
        {
            if (EditorUtility.IsPersistent(solid)) continue;
            var scene = solid.gameObject.scene;
            if (!scene.IsValid() || (EditorSceneManager.IsPreviewScene(scene) && (stage == null || stage.scene != scene))) continue;
            yield return solid;
        }
    }

    static bool TryCache(AcousticSolid solid, string hash)
    {
        string path = CachePath(hash);
        if (!File.Exists(path)) return false;
        AcousticSolid.Boundary data;
        try
        {
            data = JsonUtility.FromJson<AcousticSolid.Boundary>(File.ReadAllText(path));
            solid.Assign(data, hash);
        }
        catch (Exception error) when (error is IOException || error is ArgumentException || error is InvalidOperationException)
        {
            try { File.Delete(path); } catch (IOException) { }
            return false;
        }
        try { File.SetLastWriteTimeUtc(path, DateTime.UtcNow); } catch (IOException) { } // Recency for eviction.
        if (!Application.isPlaying)
        {
            EditorUtility.SetDirty(solid);
            PrefabUtility.RecordPrefabInstancePropertyModifications(solid);
            if (solid.gameObject.scene.IsValid()) EditorSceneManager.MarkSceneDirty(solid.gameObject.scene);
        }
        SetStatus(solid, State.Current, Summary(data));
        return true;
    }

    static string Summary(AcousticSolid.Boundary data) =>
        $"Exterior current: {data.inputTriangles} -> {data.outputTriangles} triangles, {data.components} component(s), {data.backend}.";

    static void Abandon(AcousticSolid solid)
    {
        queue.RemoveAll(item => item.solid == solid);
        if (running != null && running.solid == solid) StopWorker();
    }

    static void StopWorker()
    {
        running?.Dispose(); running = null;
    }

    static void Fail(AcousticSolid solid, string hash, Exception error, bool log = true)
    {
        string message = error.Message;
        if (hash != null && error is WorkerException worker && worker.remember) failures[hash] = message;
        if (solid == null) return;
        var previous = Describe(solid);
        if (log && (previous.state != State.Failed || previous.message != message)) UnityEngine.Debug.LogError(solid.name + ": " + message, solid);
        SetStatus(solid, State.Failed, message);
    }

    static void SetStatus(AcousticSolid solid, State state, string message)
    {
        if (statuses.TryGetValue(solid, out var old) && old.state == state && old.message == message) return;
        statuses[solid] = new Status(state, message);
        InternalEditorUtility.RepaintAllViews();
    }

    sealed class WorkerException : InvalidOperationException
    {
        public readonly bool remember;
        public WorkerException(string message, bool remember) : base(message) { this.remember = remember; }
    }

    static WorkerException NotFound(string python) => new WorkerException(
        $"Solid-boundary Python '{python}' was not found. Create a venv with tools/solid-boundary-requirements.txt, then set " +
        $"Preferences > Acoustics > Solid Boundary Python or the {PythonVariable} environment variable.", false);

    // One low-priority backend process with time, memory and cancellation limits.
    // No shell: quoted arguments are consumed only by the nice/Python process.
    sealed class Worker : IDisposable
    {
        public readonly AcousticSolid solid;
        public readonly string hash;
        readonly string python, folder, output;
        readonly bool background;
        readonly Process process;
        readonly Task<string> stderr;
        readonly Stopwatch timer = Stopwatch.StartNew();
        readonly int progress = -1;
        bool cancelled;

        static string Quote(string value) => "\"" + value.Replace("\\", "\\\\").Replace("\"", "\\\"") + "\"";

        public Worker(AcousticSolid solid, string input, string hash, bool background)
        {
            this.solid = solid; this.hash = hash; this.background = background; python = Python;
            string script = Path.Combine(ProjectRoot, "tools", "solid_boundary.py");
            if (!File.Exists(script)) throw new WorkerException("Missing the solid-boundary backend script " + script + ".", false);
            if (Path.IsPathRooted(python) && !File.Exists(python)) throw NotFound(python);
            folder = Path.Combine(Path.GetTempPath(), "acoustic-solid-" + Guid.NewGuid().ToString("N"));
            Directory.CreateDirectory(folder);
            string request = Path.Combine(folder, "input.json"); output = Path.Combine(folder, "output.json");
            File.WriteAllText(request, input, new UTF8Encoding(false));
            var start = new ProcessStartInfo(python, Quote(script) + " " + Quote(request) + " " + Quote(output))
                { UseShellExecute = false, RedirectStandardOutput = true, RedirectStandardError = true, CreateNoWindow = true, WorkingDirectory = folder };
            if (Application.platform != RuntimePlatform.WindowsEditor)
            { start.Arguments = "-n 19 " + Quote(python) + " " + start.Arguments; start.FileName = "/usr/bin/nice"; }
            start.EnvironmentVariables["OPENBLAS_NUM_THREADS"] = "1";
            start.EnvironmentVariables["OMP_NUM_THREADS"] = "1";
            start.EnvironmentVariables["VECLIB_MAXIMUM_THREADS"] = "1";
            try { process = Process.Start(start); }
            catch (Win32Exception) { Directory.Delete(folder, true); throw NotFound(python); }
            WorkerLaunches++;
            if (Application.platform == RuntimePlatform.WindowsEditor)
                try { process.PriorityClass = ProcessPriorityClass.Idle; } catch (InvalidOperationException) { }
            _ = process.StandardOutput.ReadToEndAsync(); stderr = process.StandardError.ReadToEndAsync();
            if (background)
            {
                progress = Progress.Start("Acoustic solid exterior", solid.name, Progress.Options.Indefinite);
                Progress.RegisterCancelCallback(progress, () => cancelled = true);
            }
        }

        // True once the boundary is in the cache. Throws on failure, budget or cancellation.
        public bool Wait(int milliseconds)
        {
            if (!process.WaitForExit(milliseconds))
            {
                process.Refresh();
                if (timer.Elapsed.TotalSeconds > WorkerSeconds) throw new WorkerException($"Exterior generation exceeded its {WorkerSeconds:0} s budget. Simplify the members or click Retry.", true);
                if (process.WorkingSet64 > WorkerBytes) throw new WorkerException("Exterior generation exceeded its 1 GiB memory budget. Simplify the members or click Retry.", true);
                if (!background && !Application.isBatchMode &&
                    EditorUtility.DisplayCancelableProgressBar("Acoustic solid exterior", "Generating " + solid.name, (float)(timer.Elapsed.TotalSeconds / WorkerSeconds)))
                    cancelled = true;
                if (cancelled) throw new WorkerException("Exterior generation was cancelled. Edit the solid or click Retry.", true);
                return false;
            }
            process.WaitForExit();
            string error = stderr.GetAwaiter().GetResult().Trim();
            switch (process.ExitCode)
            {
                case 0: break;
                case 126: case 127: throw NotFound(python);
                case 3: throw new WorkerException(error + "\nThen click Retry.", false);
                case 4: throw new WorkerException(error, true);
                default: throw new WorkerException($"Solid-boundary backend failed (exit {process.ExitCode}) using '{python}':\n{error}", true);
            }
            var data = JsonUtility.FromJson<AcousticSolid.Boundary>(File.ReadAllText(output));
            if (data == null || data.inputSha256 != hash || data.version != AcousticSolid.BoundaryVersion)
                throw new WorkerException("Backend output does not match the solid's source hash.", true);
            Store(output, hash);
            return true;
        }

        static void Store(string file, string hash)
        {
            Directory.CreateDirectory(CacheFolder);
            string target = CachePath(hash), partial = target + ".partial";
            File.Copy(file, partial, true);
            if (File.Exists(target)) File.Delete(target);
            File.Move(partial, target);
            var entries = new DirectoryInfo(CacheFolder).GetFiles("*.json");
            foreach (var old in entries.OrderByDescending(entry => entry.LastWriteTimeUtc).Skip(CacheEntries))
                try { old.Delete(); } catch (IOException) { }
        }

        public void Dispose()
        {
            if (process != null)
            {
                try { if (!process.HasExited) { process.Kill(); process.WaitForExit(); } }
                catch (Exception error) when (error is InvalidOperationException || error is Win32Exception) { }
                process.Dispose();
            }
            if (progress >= 0) Progress.Remove(progress);
            if (!background && !Application.isBatchMode) EditorUtility.ClearProgressBar();
            try { Directory.Delete(folder, true); } catch (IOException) { }
        }
    }
}
