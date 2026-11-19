using System;
using System.IO;
using UnityEditor;
using UnityEditor.Build.Reporting;
using UnityEditor.SceneManagement;
using UnityEngine;

// The playable Church with the path method live (PathLiveChurch; README.md, "Playable Church").
//   Editor: Acoustics > Play Church live (path method).
//   Build:  Acoustics > Build playable Church (macOS), or batch mode:
//           Unity -batchmode -quit -projectPath . -executeMethod PathLiveChurchBuild.Build [-pathLiveOutput DIR]
//           (default Builds/ChurchLive/ChurchLive.app).
public static class PathLiveChurchBuild
{
    const string Scene = "Assets/Scenes/Church.unity";

    [MenuItem("Acoustics/Play Church live (path method)")]
    static void Play()
    {
        if (!EditorSceneManager.SaveCurrentModifiedScenesIfUserWantsTo()) return;
        EditorSceneManager.OpenScene(Scene);
        SessionState.SetBool(PathLiveChurch.EditorFlag, true);
        EditorApplication.playModeStateChanged -= Clear; EditorApplication.playModeStateChanged += Clear;
        EditorApplication.EnterPlaymode();
    }

    static void Clear(PlayModeStateChange change)
    {
        if (change != PlayModeStateChange.EnteredEditMode) return;
        SessionState.EraseBool(PathLiveChurch.EditorFlag);
        EditorApplication.playModeStateChanged -= Clear;
    }

    [MenuItem("Acoustics/Build playable Church (macOS)")]
    public static void Build()
    {
        var args = Environment.GetCommandLineArgs();
        int at = Array.IndexOf(args, "-pathLiveOutput");
        string output = Path.GetFullPath(at >= 0 && at + 1 < args.Length ? args[at + 1] : "Builds/ChurchLive/ChurchLive.app");
        var options = new BuildPlayerOptions
        {
            scenes = new[] {Scene}, locationPathName = output, target = BuildTarget.StandaloneOSX, targetGroup = BuildTargetGroup.Standalone,
            extraScriptingDefines = new[] {"PATH_LIVE_CHURCH"}, options = BuildOptions.None
        };
        var report = BuildPipeline.BuildPlayer(options);
        var summary = report.summary;
        Debug.Log($"PATH_LIVE_BUILD {summary.result} {output} {summary.totalSize / (1 << 20)} MiB in {summary.totalTime.TotalSeconds:F0} s, {summary.totalErrors} errors");
        if (Application.isBatchMode) EditorApplication.Exit(summary.result == BuildResult.Succeeded ? 0 : 1);
    }
}
