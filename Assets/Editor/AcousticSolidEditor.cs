using System;
using System.Diagnostics;
using System.IO;
using System.Text;
using UnityEditor;
using UnityEngine;
using Debug = UnityEngine.Debug;

[CustomEditor(typeof(AcousticSolid))]
public sealed class AcousticSolidEditor : Editor
{
    const string PythonPreference = "Acoustics.SolidBakePython";
    public override void OnInspectorGUI()
    {
        DrawDefaultInspector();
        EditorGUILayout.HelpBox("Select fixed child surfaces that form one interior medium. Unlisted surfaces, including moving doors, remain independent. Source renderers are unchanged.", MessageType.Info);
        string python = EditorPrefs.GetString(PythonPreference, "python3");
        string next = EditorGUILayout.TextField("Bake Python", python);
        if (next != python) EditorPrefs.SetString(PythonPreference, next);
        using (new EditorGUI.DisabledScope(Application.isPlaying))
            if (GUILayout.Button("Bake acoustic exterior"))
            {
                string path = EditorUtility.SaveFilePanelInProject("Save acoustic boundary", target.name+"-boundary", "json", "Choose a baked boundary asset.");
                if (string.IsNullOrEmpty(path)) return;
                try
                {
                    var solid = (AcousticSolid)target;
                    var manager = solid.GetComponentInParent<RayTraceManager>();
                    if (manager == null) manager = FindFirstObjectByType<RayTraceManager>();
                    Bake(solid, next, AssetDatabase.GenerateUniqueAssetPath(path), manager == null ? null : manager.takeDefaultMaterial);
                }
                catch (Exception error) { Debug.LogException(error, target); }
            }
    }

    // No shell: quoted arguments are consumed only by the Python/nice process.
    static string Quote(string value) => "\""+value.Replace("\\", "\\\\").Replace("\"", "\\\"")+"\"";
    public static void Bake(AcousticSolid solid, string python, string assetPath, AudioMaterial fallback)
    {
        if (Application.isPlaying) throw new InvalidOperationException("Bake acoustic solids outside Play mode.");
        string project = Directory.GetParent(Application.dataPath).FullName;
        string assetFile = Path.GetFullPath(Path.Combine(project, assetPath));
        if (!assetPath.StartsWith("Assets/", StringComparison.Ordinal) || !assetFile.StartsWith(Application.dataPath+Path.DirectorySeparatorChar, StringComparison.Ordinal) || Path.GetExtension(assetPath) != ".json" || File.Exists(assetFile))
            throw new ArgumentException("Use a new .json asset path under Assets.");
        string input = solid.ExportInput(fallback, out var materials);
        string folder = Path.Combine(Path.GetTempPath(), "acoustic-solid-"+Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(folder);
        try
        {
            string request = Path.Combine(folder, "input.json"), response = Path.Combine(folder, "output.json");
            File.WriteAllText(request, input, new UTF8Encoding(false));
            string script = Path.Combine(project, "tools", "solid_boundary.py");
            if (!File.Exists(script)) throw new FileNotFoundException("Missing offline solid baker.", script);
            var start = new ProcessStartInfo(python, Quote(script)+" "+Quote(request)+" "+Quote(response))
            { UseShellExecute = false, RedirectStandardOutput = true, RedirectStandardError = true, CreateNoWindow = true };
            if (Application.platform != RuntimePlatform.WindowsEditor)
            { start.Arguments = "-n 19 "+Quote(python)+" "+start.Arguments; start.FileName = "/usr/bin/nice"; }
            start.EnvironmentVariables["OPENBLAS_NUM_THREADS"] = "1";
            start.EnvironmentVariables["OMP_NUM_THREADS"] = "1";
            start.EnvironmentVariables["VECLIB_MAXIMUM_THREADS"] = "1";
            using (var process = Process.Start(start))
            {
                var stdout = process.StandardOutput.ReadToEndAsync(); var stderr = process.StandardError.ReadToEndAsync();
                var timer = Stopwatch.StartNew();
                try
                {
                    while (!process.WaitForExit(100))
                    {
                        if (timer.Elapsed.TotalSeconds > 120 || process.WorkingSet64 > 1024L*1024*1024)
                            throw new InvalidOperationException("Solid bake exceeded its 120 s / 1 GiB worker budget.");
                        if (!Application.isBatchMode && EditorUtility.DisplayCancelableProgressBar("Baking acoustic exterior", solid.name, .5f))
                            throw new OperationCanceledException();
                    }
                    if (process.ExitCode != 0) throw new InvalidOperationException("Boundary bake failed. Install tools/solid-boundary-requirements.txt in the selected Python environment.\n"+stderr.GetAwaiter().GetResult());
                    Debug.Log(stdout.GetAwaiter().GetResult(), solid);
                }
                finally
                {
                    if (!process.HasExited) { process.Kill(); process.WaitForExit(); }
                    if (!Application.isBatchMode) EditorUtility.ClearProgressBar();
                }
            }
            var data = JsonUtility.FromJson<AcousticSolid.Boundary>(File.ReadAllText(response));
            if (data == null || data.inputSha256 != AcousticTake.HashText(input)) throw new InvalidOperationException("Bake response does not match source input.");
            if (solid.ExportInput(fallback, out _) != input) throw new InvalidOperationException("Source geometry changed during bake.");
            Directory.CreateDirectory(Path.GetDirectoryName(assetFile)); File.Copy(response, assetFile);
            AssetDatabase.ImportAsset(assetPath, ImportAssetOptions.ForceSynchronousImport);
            Undo.RecordObject(solid, "Assign acoustic solid boundary");
            solid.boundary = AssetDatabase.LoadAssetAtPath<TextAsset>(assetPath); solid.surfaceMaterials = materials;
            solid.ReadValidated(fallback);
            EditorUtility.SetDirty(solid); PrefabUtility.RecordPrefabInstancePropertyModifications(solid);
        }
        finally { Directory.Delete(folder, true); }
    }
}
