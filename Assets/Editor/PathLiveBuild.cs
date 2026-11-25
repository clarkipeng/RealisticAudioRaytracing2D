using System;
using System.IO;
using System.Linq;
using UnityEditor;
using UnityEditor.Build.Reporting;
using UnityEditor.SceneManagement;
using UnityEngine;

// The live showcase: the path method live in each of PathLive.Scenes (README.md, "Live showcase").
//   Editor: Acoustics > Play live showcase (path method): the open scene if it is a showcase scene, else the first.
//   Build:  Acoustics > Build live showcase, or batch mode:
//           Unity -batchmode -quit -projectPath . -executeMethod PathLiveBuild.Build [-pathLiveOutput PATH]
//           for the editor's own platform (default Builds/LiveShowcase/LiveShowcase.app on macOS, .exe on Windows,
//           on Direct3D 12). -pathLiveDevelopment builds a development player, whose
//           profiler reports the per-stage GPU times (tracer, reverb, denoiser) that release players leave empty.
public static class PathLiveBuild
{
    static bool Windows => Application.platform == RuntimePlatform.WindowsEditor;
    static string DefaultOutput => Windows ? "Builds/LiveShowcase/LiveShowcase.exe" : "Builds/LiveShowcase/LiveShowcase.app";

    [MenuItem("Acoustics/Play live showcase (path method)")]
    static void Play()
    {
        if (!EditorSceneManager.SaveCurrentModifiedScenesIfUserWantsTo()) return;
        string open = EditorSceneManager.GetActiveScene().path;
        EditorSceneManager.OpenScene(PathLive.Scenes.Any(s => s.path == open) ? open : PathLive.Scenes[0].path);
        SessionState.SetBool(PathLive.EditorFlag, true);
        EditorApplication.playModeStateChanged -= Clear; EditorApplication.playModeStateChanged += Clear;
        EditorApplication.EnterPlaymode();
    }

    static void Clear(PlayModeStateChange change)
    {
        if (change != PlayModeStateChange.EnteredEditMode) return;
        SessionState.EraseBool(PathLive.EditorFlag);
        EditorApplication.playModeStateChanged -= Clear;
    }

    [MenuItem("Acoustics/Build live showcase")]
    public static void Build()
    {
        var args = Environment.GetCommandLineArgs();
        int at = Array.IndexOf(args, "-pathLiveOutput");
        string output = Path.GetFullPath(at >= 0 && at + 1 < args.Length ? args[at + 1] : DefaultOutput);
        var options = new BuildPlayerOptions
        {
            scenes = PathLive.Scenes.Select(s => s.path).ToArray(), locationPathName = output,
            target = Windows ? BuildTarget.StandaloneWindows64 : BuildTarget.StandaloneOSX, targetGroup = BuildTargetGroup.Standalone,
            extraScriptingDefines = new[] {"PATH_LIVE"}, options = args.Contains("-pathLiveDevelopment") ? BuildOptions.Development : BuildOptions.None
        };
        // Frame timing stats: release players report the frame's GPU time through FrameTimingManager (the profiler's
        // per-sample GPU times are development-only); the HUD and the smoke run read it.
        PlayerSettings.enableFrameTimingStats = true;
        // Direct3D 12 on Windows (Unity's automatic choice is Direct3D 11): the API of the inline hardware ray queries.
        if (Windows) { PlayerSettings.SetUseDefaultGraphicsAPIs(BuildTarget.StandaloneWindows64, false); PlayerSettings.SetGraphicsAPIs(BuildTarget.StandaloneWindows64, new[] {UnityEngine.Rendering.GraphicsDeviceType.Direct3D12}); }
        var report = BuildPipeline.BuildPlayer(options);
        var summary = report.summary;
        Debug.Log($"PATH_LIVE_BUILD {summary.result} {output} {summary.totalSize / (1 << 20)} MiB in {summary.totalTime.TotalSeconds:F0} s, {summary.totalErrors} errors");
        if (Application.isBatchMode) EditorApplication.Exit(summary.result == BuildResult.Succeeded ? 0 : 1);
    }
}
