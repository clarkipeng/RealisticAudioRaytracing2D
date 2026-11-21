using System;
using System.IO;
using System.Linq;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;

// Still images of scenes for a visual review, without Play mode or a player build:
//   Unity -batchmode -quit -projectPath . -executeMethod SceneStills.Capture -stillsOutput DIR [-stillsScenes Church,TrainStation]
// Per scene: the player's own camera, "Player spawn 0" at eye height, and each waypoint of a showcase scene's smoke walk
// (PathLiveScene.route), looking along the walk.
public static class SceneStills
{
    const int Width = 1280, Height = 720;

    public static void Capture()
    {
        string output = Path.GetFullPath(Argument("-stillsOutput") ?? ".context/polish/stills");
        string[] names = (Argument("-stillsScenes") ?? "").Split(',', StringSplitOptions.RemoveEmptyEntries);
        var scenes = AssetDatabase.FindAssets("t:Scene", new[] {"Assets/Scenes"}).Select(AssetDatabase.GUIDToAssetPath)
            .Where(p => names.Length == 0 || names.Contains(Path.GetFileNameWithoutExtension(p))).ToArray();
        Directory.CreateDirectory(output);
        int failures = 0;
        foreach (string path in scenes)
        {
            try { CaptureScene(path, output); }
            catch (Exception e) { failures++; Debug.LogException(e); }
        }
        Debug.Log($"SCENE_STILLS {scenes.Length - failures}/{scenes.Length} scenes to {output}");
        if (Application.isBatchMode) EditorApplication.Exit(failures == 0 ? 0 : 1);
    }

    static void CaptureScene(string path, string output)
    {
        var scene = EditorSceneManager.OpenScene(path, OpenSceneMode.Single);
        string name = scene.name.Replace(' ', '-');
        var all = scene.GetRootGameObjects().SelectMany(r => r.GetComponentsInChildren<Transform>(true)).ToArray();
        var main = all.Select(t => t.GetComponent<Camera>()).FirstOrDefault(c => c != null && c.isActiveAndEnabled);
        var camera = new GameObject("Stills camera").AddComponent<Camera>();
        if (main != null) camera.CopyFrom(main);
        camera.targetTexture = null;
        if (main != null) Shoot(camera, main.transform.position, main.transform.rotation, Path.Combine(output, $"{name}-player.png"));
        foreach (var spawn in all.Where(t => t.name.StartsWith("Player spawn 0", StringComparison.Ordinal)))
            Shoot(camera, spawn.position + Vector3.up * 1.6f, spawn.rotation, Path.Combine(output, $"{name}-spawn.png"));
        var showcase = all.Select(t => t.GetComponent<PathLiveScene>()).FirstOrDefault(c => c != null);
        var route = showcase != null ? showcase.route : Array.Empty<Vector3>();
        for (int i = 0; i < route.Length; i++)
        {
            var ahead = i + 1 < route.Length ? route[i + 1] - route[i] : route[i] - route[Math.Max(0, i - 1)];
            ahead.y = 0;
            if (ahead.sqrMagnitude < 1e-4f) ahead = Vector3.forward;
            Shoot(camera, route[i] + Vector3.up * 1.6f, Quaternion.LookRotation(ahead), Path.Combine(output, $"{name}-route-{i + 1}.png"));
        }
        UnityEngine.Object.DestroyImmediate(camera.gameObject);
    }

    static void Shoot(Camera camera, Vector3 position, Quaternion rotation, string file)
    {
        camera.transform.SetPositionAndRotation(position, rotation);
        var target = new RenderTexture(Width, Height, 24, RenderTextureFormat.ARGB32) {antiAliasing = 4};
        var image = new Texture2D(Width, Height, TextureFormat.RGB24, false);
        try
        {
            camera.targetTexture = target; camera.Render();
            RenderTexture.active = target; image.ReadPixels(new Rect(0, 0, Width, Height), 0, 0); image.Apply();
            File.WriteAllBytes(file, image.EncodeToPNG());
        }
        finally
        {
            camera.targetTexture = null; RenderTexture.active = null;
            UnityEngine.Object.DestroyImmediate(target); UnityEngine.Object.DestroyImmediate(image);
        }
    }

    static string Argument(string name)
    {
        var args = Environment.GetCommandLineArgs();
        int at = Array.IndexOf(args, name);
        return at >= 0 && at + 1 < args.Length ? args[at + 1] : null;
    }
}
