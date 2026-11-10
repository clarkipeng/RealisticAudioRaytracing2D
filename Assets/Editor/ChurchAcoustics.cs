using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;

// Batch tools for the Church's acoustic model (docs/benchmarks/energy-audio-2026-11-09).
// Run in the isolated runner only, never in the user's editor:
//   -executeMethod ChurchAcoustics.Inspect -churchOutput DIR            acoustic and visual inventory (inventory.json)
//   -executeMethod ChurchAcoustics.Apply -churchPlan PLAN.json          materials, assignments and interior proxies; saves the scene
//   -executeMethod ChurchAcoustics.CaptureSpeechWalk -churchOutput DIR -churchSpeech SPEECH.f32
//                                                                      the scripted speech walk as a take (DIR/take)
public static class ChurchAcoustics
{
    const string ScenePath = "Assets/Scenes/Church.unity", ProxyRoot = "Acoustic interior proxies";

    [Serializable] public sealed class Box { public string path, name, material, kind; public Vector3 center, size; public int triangles; }
    [Serializable] public sealed class Inventory { public List<Box> acoustic = new(), visual = new(); }
    [Serializable] public sealed class MaterialPlan { public string name; public float[] absorption, scattering; public float[] frequencies; }
    [Serializable] public sealed class Assignment { public string objectName, material; }
    [Serializable] public sealed class Proxy { public string name, material; public Vector3 center, size; public float yaw; }
    [Serializable] public sealed class Plan { public MaterialPlan[] materials; public Assignment[] assignments; public Proxy[] proxies; }

    static string Arg(string name)
    {
        var args = Environment.GetCommandLineArgs(); int i = Array.IndexOf(args, name);
        if (i < 0 || i + 1 >= args.Length) throw new ArgumentException("Missing " + name);
        return args[i + 1];
    }
    static void Batch(Action action)
    {
        try
        {
            if (!Application.isBatchMode) throw new InvalidOperationException("Batch only: run in the isolated runner.");
            action(); EditorApplication.Exit(0);
        }
        catch (Exception e) { Debug.LogException(e); EditorApplication.Exit(1); }
    }
    static string PathOf(Transform t) => t.parent == null ? t.name : PathOf(t.parent) + "/" + t.name;

    public static void Inspect() => Batch(() =>
    {
        string output = Arg("-churchOutput"); Directory.CreateDirectory(output);
        EditorSceneManager.OpenScene(ScenePath);
        var inventory = new Inventory();
        foreach (var surface in UnityEngine.Object.FindObjectsByType<AcousticSurface>(FindObjectsInactive.Include, FindObjectsSortMode.None))
        {
            Bounds bounds = default; int triangles = 0; string kind = "none";
            if (surface.TryGetComponent<Collider>(out var collider)) { bounds = collider.bounds; kind = collider.GetType().Name; }
            if (surface.TryGetComponent<MeshFilter>(out var filter) && filter.sharedMesh != null)
            { if (kind == "none") { bounds = surface.GetComponent<Renderer>()?.bounds ?? default; kind = "Mesh"; } triangles = filter.sharedMesh.triangles.Length / 3; }
            inventory.acoustic.Add(new Box {path = PathOf(surface.transform), name = surface.name, material = surface.material != null ? surface.material.name : "",
                                            kind = kind, center = bounds.center, size = bounds.size, triangles = triangles});
        }
        foreach (var renderer in UnityEngine.Object.FindObjectsByType<MeshRenderer>(FindObjectsInactive.Exclude, FindObjectsSortMode.None))
        {
            string path = PathOf(renderer.transform);
            if (!new[] {"Pew", "Nave arcades", "Arcade", "Column", "Pillar", "Pier", "Organ", "Altar", "Font", "Reading", "Pews and sanctuary", "Visual details", "Window", "Glass", "Rose", "Roof", "Wall", "Gable"}.Any(path.Contains)) continue;
            var filter = renderer.GetComponent<MeshFilter>();
            inventory.visual.Add(new Box {path = path, name = renderer.name, kind = filter != null && filter.sharedMesh != null ? filter.sharedMesh.name : "",
                                          center = renderer.bounds.center, size = renderer.bounds.size,
                                          triangles = filter != null && filter.sharedMesh != null ? filter.sharedMesh.triangles.Length / 3 : 0});
        }
        File.WriteAllText(Path.Combine(output, "inventory.json"), JsonUtility.ToJson(inventory, true));
        // Combined visual meshes whose parts (columns, piers) need splitting: world triangles, 9 floats each.
        foreach (var renderer in UnityEngine.Object.FindObjectsByType<MeshRenderer>(FindObjectsInactive.Exclude, FindObjectsSortMode.None))
        {
            if (renderer.name != "Nave arcades" || !renderer.TryGetComponent<MeshFilter>(out var filter) || filter.sharedMesh == null) continue;
            var mesh = filter.sharedMesh; var vertices = mesh.vertices; var triangles = mesh.triangles; var world = renderer.transform.localToWorldMatrix;
            using var writer = new BinaryWriter(File.Create(Path.Combine(output, renderer.name + ".tris.f32")));
            foreach (int index in triangles) { var v = world.MultiplyPoint3x4(vertices[index]); writer.Write(v.x); writer.Write(v.y); writer.Write(v.z); }
        }
        Debug.Log($"CHURCH_INVENTORY acoustic={inventory.acoustic.Count} visual={inventory.visual.Count}");
    });

    public static void Apply() => Batch(() =>
    {
        var plan = JsonUtility.FromJson<Plan>(File.ReadAllText(Arg("-churchPlan")));
        var materials = new Dictionary<string, AudioMaterial>();
        foreach (var m in plan.materials)
        {
            string asset = "Assets/Materials/" + m.name + ".asset";
            var material = AssetDatabase.LoadAssetAtPath<AudioMaterial>(asset);
            if (material == null) { material = ScriptableObject.CreateInstance<AudioMaterial>(); AssetDatabase.CreateAsset(material, asset); }
            if (m.frequencies.Length != m.absorption.Length || m.frequencies.Length != m.scattering.Length) throw new ArgumentException("Band mismatch in " + m.name);
            material.energyCurve = m.frequencies.Select((f, i) => new AudioMaterial.EnergyKnot {frequencyHz = f, absorption = m.absorption[i], scattering = m.scattering[i]}).ToArray();
            EditorUtility.SetDirty(material);
            materials[m.name] = material;
        }
        AssetDatabase.SaveAssets();
        EditorSceneManager.OpenScene(ScenePath);
        // Opening the scene unloads the in-memory instances; use the saved assets.
        foreach (var name in materials.Keys.ToArray())
            materials[name] = AssetDatabase.LoadAssetAtPath<AudioMaterial>("Assets/Materials/" + name + ".asset") ?? throw new IOException("Material asset not saved: " + name);
        var session = UnityEngine.Object.FindAnyObjectByType<SceneSession>();
        var manager = session.manager;
        int assigned = 0;
        foreach (var a in plan.assignments)
            foreach (var surface in UnityEngine.Object.FindObjectsByType<AcousticSurface>(FindObjectsInactive.Include, FindObjectsSortMode.None))
                if (surface.name == a.objectName)
                {
                    surface.material = materials.TryGetValue(a.material, out var m) ? m : AssetDatabase.LoadAssetAtPath<AudioMaterial>("Assets/Materials/" + a.material + ".asset");
                    if (surface.material == null) throw new ArgumentException("Unknown material " + a.material);
                    surface.submeshMaterials = Array.Empty<AudioMaterial>();
                    EditorUtility.SetDirty(surface); assigned++;
                }
        var root = manager.takeRoot.Find(ProxyRoot);
        if (root != null) UnityEngine.Object.DestroyImmediate(root.gameObject);
        if (plan.proxies.Length > 0) { root = new GameObject(ProxyRoot).transform; root.SetParent(manager.takeRoot, false); }
        foreach (var p in plan.proxies)
        {
            var box = new GameObject(p.name); box.transform.SetParent(root, false);
            box.transform.SetPositionAndRotation(p.center, Quaternion.Euler(0, p.yaw, 0));
            box.isStatic = true;
            var collider = box.AddComponent<BoxCollider>(); collider.size = p.size;
            var surface = box.AddComponent<AcousticSurface>();
            surface.geometrySource = AcousticSurface.GeometrySource.Collider;
            surface.material = materials.TryGetValue(p.material, out var m) ? m : AssetDatabase.LoadAssetAtPath<AudioMaterial>("Assets/Materials/" + p.material + ".asset");
            if (surface.material == null) throw new ArgumentException("Unknown material " + p.material);
        }
        manager.RegisterTakeObjects();
        EditorUtility.SetDirty(manager);
        EditorSceneManager.MarkSceneDirty(manager.gameObject.scene);
        if (!EditorSceneManager.SaveScene(manager.gameObject.scene)) throw new IOException("Church scene was not saved.");
        Debug.Log($"CHURCH_APPLY materials={materials.Count} assigned={assigned} proxies={plan.proxies.Length} takeObjects={manager.takeObjects.Count}");
    });

    // The scripted speech walk of the 15 October movement probe (.context/church-motion):
    // stand, turn the head (3-5 s), walk to the organ and back, open the doors, walk out.
    const int Rate = 44100, Ticks = 900;
    const float Step = .02f;
    static Vector2 Look(float t) => t < 3 || t > 5 ? Vector2.zero : new Vector2(90, 20) * (float)Math.Sin(Math.PI * (t - 3));
    public static void CaptureSpeechWalk() => Batch(() =>
    {
        string output = Arg("-churchOutput");
        var bytes = File.ReadAllBytes(Arg("-churchSpeech"));
        var speech = new float[bytes.Length / 4]; Buffer.BlockCopy(bytes, 0, speech, 0, bytes.Length);
        AudioListener.volume = 0;
        EditorSceneManager.OpenScene(ScenePath);
        var session = UnityEngine.Object.FindAnyObjectByType<SceneSession>();
        using var rig = new SceneReplay.Rig(session);
        var manager = session.manager;
        // The organ (the 15 October probe's source at (0, 5, 18)).
        var organ = UnityEngine.Object.FindObjectsByType<RaytracedAudioSource>(FindObjectsInactive.Include, FindObjectsSortMode.None)
            .OrderBy(a => a.name == "Pipe organ" ? 0 : 1).ThenBy(a => (a.transform.position - new Vector3(0, 5, 18)).sqrMagnitude).First();
        Debug.Log($"CHURCH_CAPTURE source={organ.name} at {organ.transform.position}");
        var scene = SceneToTake.Compile(rig.bindings, manager.takeDimension, manager.takeCurveTolerance, manager.takeDefaultMaterial,
            new[] {rig.copies[organ.transform]}, new[] {organ.clip}, new[] {rig.head}, out var geometry);
        scene.clips[0].name = "Synthetic speech";
        // Start where the October probe started: the first player spawn, inside the nave (the scene's
        // authored player now stands outside for the weather walk).
        var spawn = rig.scene.GetRootGameObjects().SelectMany(r => r.GetComponentsInChildren<Transform>(true)).First(t => t.name == "Player spawn 0");
        rig.demo.player.SetPositionAndRotation(spawn.position, spawn.rotation);
        Physics.SyncTransforms();
        var frames = new List<AcousticTake.State[]>();
        for (int tick = 0; tick <= Ticks; tick++)
        {
            if (tick > 0)
            {
                int i = tick - 1;
                var input = new SceneRecording.Input
                {
                    movement = i >= 250 && i < 350 ? Vector2.up : i >= 350 && i < 500 || i >= 580 && i < 680 ? Vector2.down : i >= 680 && i < 780 ? Vector2.right : Vector2.zero,
                    look = Look(tick * Step) - Look(i * Step), interact = i == 500
                };
                rig.demo.Step(input, Step); rig.scene.GetPhysicsScene().Simulate(Step);
                if (rig.demo.player.position.y < .7f) throw new Exception("Fell through floor.");
            }
            var states = rig.bindings.Select((b, i) => SceneToTake.Observe(scene, i, b.transform)).ToArray();
            if (tick > 0) for (int i = 0; i < states.Length; i++) states[i].MeasureMotion(frames[tick - 1][i], Step);
            frames.Add(states);
        }
        if (rig.demo.player.position.z > -39) throw new Exception("Did not exit the church: " + rig.demo.player.position);
        using (var writer = new AcousticTake.Writer(Path.Combine(output, "take"), Rate, scene, geometry, new[] {speech}))
        {
            for (int tick = 0; tick <= Ticks; tick++) writer.Append(tick * 882L, frames[tick]);
            writer.Append(new AcousticTake.VoiceEvent {action = AcousticTake.VoiceAction.Play, gain = 1});
            writer.Append(new AcousticTake.VoiceEvent {sample = Ticks * 882L, action = AcousticTake.VoiceAction.Stop});
            writer.Complete();
        }
        Debug.Log($"CHURCH_CAPTURE objects={scene.objects.Length} triangles={geometry.Length} final={rig.demo.player.position}");
    });
}
