using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;

// The train station's acoustic materials (docs/benchmarks/path-denoise-2026-11-12, part 43): published absorption,
// scattering and transmission per octave band for each kind of surface, assigned by the surface's visual material.
// Run in the isolated runner only, never in the user's editor (tools/station_acoustics.py):
//   -executeMethod StationAcoustics.Inspect -stationOutput DIR      inventory of the take's geometry (inventory.json)
//   -executeMethod StationAcoustics.Apply -stationPlan PLAN.json    material assets and assignments; saves the scene
public static class StationAcoustics
{
    const string ScenePath = "Assets/Scenes/TrainStation.unity";

    [Serializable] public sealed class Item { public string path, name, surface, source; public string[] visual; public int submeshes, triangles; public Vector3 center, size; }
    [Serializable] public sealed class Inventory { public List<Item> geometry = new(); public string fallback; }
    // Bands as ChurchAcoustics: absorption and scattering per knot (frequencies), transmission eight octave bands
    // 125 Hz - 16 kHz, ior optional (1: a partition). The note is the material's sources, kept in the plan and README.
    [Serializable] public sealed class MaterialPlan { public string name, note; public float[] frequencies, absorption, scattering, transmission, ior; }
    // A visual material (by name) and the acoustic material its triangles get; objects by name override it.
    [Serializable] public sealed class Rule { public string visual, material; }
    // trackBed: the material of the ballast bed under the train's loop (StationLoop), which the scene draws but did not
    // model acoustically: thin proxy slabs on the bed's footprint, as the Church's pew blocks.
    [Serializable] public sealed class Plan { public MaterialPlan[] materials; public Rule[] visuals; public Rule[] objects; public string fallback, trackBed; }
    const string TrackBedRoot = "Acoustic track bed";

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
    static RayTraceManager Manager() => UnityEngine.Object.FindAnyObjectByType<SceneSession>().manager;

    public static void Inspect() => Batch(() =>
    {
        string output = Arg("-stationOutput"); Directory.CreateDirectory(output);
        EditorSceneManager.OpenScene(ScenePath);
        var manager = Manager(); var inventory = new Inventory {fallback = manager.takeDefaultMaterial != null ? manager.takeDefaultMaterial.name : ""};
        foreach (var binding in manager.takeObjects.Where(b => b.geometry && b.transform != null))
        {
            var t = binding.transform; var surface = t.GetComponent<AcousticSurface>(); var renderer = t.GetComponent<Renderer>(); var filter = t.GetComponent<MeshFilter>();
            Bounds bounds = renderer != null ? renderer.bounds : t.TryGetComponent<Collider>(out var c) ? c.bounds : new Bounds(t.position, Vector3.zero);
            string source = surface == null ? "Auto" : surface.geometrySource.ToString();
            if ((surface == null || surface.geometrySource == AcousticSurface.GeometrySource.Auto) && filter == null) source += t.GetComponent<Collider>() != null ? " (collider)" : " (none)";
            inventory.geometry.Add(new Item
            {
                path = PathOf(t), name = t.name, surface = surface != null && surface.material != null ? surface.material.name : "", source = source,
                visual = renderer != null ? renderer.sharedMaterials.Select(m => m != null ? m.name : "").ToArray() : Array.Empty<string>(),
                submeshes = filter != null && filter.sharedMesh != null ? filter.sharedMesh.subMeshCount : 0,
                triangles = filter != null && filter.sharedMesh != null ? filter.sharedMesh.triangles.Length / 3 : 0, center = bounds.center, size = bounds.size
            });
        }
        File.WriteAllText(Path.Combine(output, "inventory.json"), JsonUtility.ToJson(inventory, true));
        Debug.Log($"STATION_INVENTORY geometry={inventory.geometry.Count}");
    });

    public static void Apply() => Batch(() =>
    {
        var plan = JsonUtility.FromJson<Plan>(File.ReadAllText(Arg("-stationPlan")));
        var materials = new Dictionary<string, AudioMaterial>();
        foreach (var m in plan.materials)
        {
            string asset = "Assets/Materials/" + m.name + ".asset";
            var material = AssetDatabase.LoadAssetAtPath<AudioMaterial>(asset);
            if (material == null) { material = ScriptableObject.CreateInstance<AudioMaterial>(); AssetDatabase.CreateAsset(material, asset); }
            if (m.frequencies.Length != m.absorption.Length || m.frequencies.Length != m.scattering.Length) throw new ArgumentException("Band mismatch in " + m.name);
            material.source = m.note ?? "";
            material.energyCurve = m.frequencies.Select((f, i) => new AudioMaterial.EnergyKnot {frequencyHz = f, absorption = m.absorption[i], scattering = m.scattering[i]}).ToArray();
            if (m.transmission == null || m.transmission.Length != 8) throw new ArgumentException("Transmission needs eight octave bands in " + m.name);
            // As ChurchAcoustics: band overrides carry the transmission; the derived index and damping are kept.
            material.overrideBands = false;
            var derived = material.GetShaderData();
            material.iorBands = new[] {derived.iorLow.x, derived.iorLow.y, derived.iorLow.z, derived.iorLow.w, derived.iorHigh.x, derived.iorHigh.y, derived.iorHigh.z, derived.iorHigh.w};
            material.dampingBands = new[] {derived.dampingLow.x, derived.dampingLow.y, derived.dampingLow.z, derived.dampingLow.w, derived.dampingHigh.x, derived.dampingHigh.y, derived.dampingHigh.z, derived.dampingHigh.w};
            material.transmissionBands = (float[])m.transmission.Clone();
            if (m.ior != null && m.ior.Length == 8) material.iorBands = (float[])m.ior.Clone();
            material.overrideBands = true;
            EditorUtility.SetDirty(material);
            materials[m.name] = material;
        }
        AssetDatabase.SaveAssets();
        EditorSceneManager.OpenScene(ScenePath);
        foreach (var name in materials.Keys.ToArray())
            materials[name] = AssetDatabase.LoadAssetAtPath<AudioMaterial>("Assets/Materials/" + name + ".asset") ?? throw new IOException("Material asset not saved: " + name);
        AudioMaterial Named(string name) => materials.TryGetValue(name, out var m) ? m : throw new ArgumentException("Unknown material " + name);
        var byVisual = plan.visuals.ToDictionary(r => r.visual, r => Named(r.material));
        var byObject = plan.objects.ToDictionary(r => r.visual, r => Named(r.material));
        var manager = Manager();
        manager.takeDefaultMaterial = Named(plan.fallback);
        int assigned = 0, bySubmesh = 0, unmatched = 0;
        foreach (var binding in manager.takeObjects.Where(b => b.geometry && b.transform != null))
        {
            var t = binding.transform; var renderer = t.GetComponent<Renderer>();
            var surface = t.GetComponent<AcousticSurface>(); if (surface == null) surface = t.gameObject.AddComponent<AcousticSurface>();   // (Unity's null: not ??)
            var visuals = renderer != null ? renderer.sharedMaterials.Select(m => m != null ? m.name : "").ToArray() : Array.Empty<string>();
            if (byObject.TryGetValue(t.name, out var named)) { surface.material = named; surface.submeshMaterials = Array.Empty<AudioMaterial>(); }
            else
            {
                var slots = visuals.Select(v => byVisual.TryGetValue(v, out var m) ? m : null).ToArray();
                unmatched += slots.Count(s => s == null);
                // The object's material: its first matched slot (or the fallback); per submesh where they differ.
                surface.material = slots.FirstOrDefault(s => s != null) ?? manager.takeDefaultMaterial;
                surface.submeshMaterials = slots.Distinct().Count() > 1 ? slots : Array.Empty<AudioMaterial>();
                if (surface.submeshMaterials.Length > 0) bySubmesh++;
                if (slots.Any(s => s == null)) Debug.LogWarning($"STATION_APPLY no rule for {PathOf(t)}: {string.Join(", ", visuals)} (fallback {surface.material.name})");
            }
            EditorUtility.SetDirty(surface); assigned++;
        }
        int slabs = TrackBed(manager, Named(plan.trackBed));
        manager.RegisterTakeObjects();
        EditorUtility.SetDirty(manager);
        EditorSceneManager.MarkSceneDirty(manager.gameObject.scene);
        if (!EditorSceneManager.SaveScene(manager.gameObject.scene)) throw new IOException("Station scene was not saved.");
        Debug.Log($"STATION_APPLY materials={materials.Count} assigned={assigned} perSubmesh={bySubmesh} unmatchedSlots={unmatched} trackBedSlabs={slabs} takeObjects={manager.takeObjects.Count}");
    });

    // The ballast bed of the loop track (TakeDemo.StationLoop: 4.2 m wide, from 0.34 to 0.25 m below the route's
    // origin): straight slabs on the straights, and on each curve chords of at most 3 m, each as long as its chord.
    static int TrackBed(RayTraceManager manager, AudioMaterial ballast)
    {
        var old = manager.takeRoot.Find(TrackBedRoot);
        if (old != null) UnityEngine.Object.DestroyImmediate(old.gameObject);
        var route = UnityEngine.Object.FindAnyObjectByType<StationLoop>(); if (route == null) throw new InvalidOperationException("No StationLoop in the station.");
        var root = new GameObject(TrackBedRoot).transform; root.SetParent(manager.takeRoot, false);
        double arc = Math.PI * route.radius; int chords = (int)Math.Ceiling(arc / 3);
        var pieces = new List<(double from, double to)> {(0, route.straightLength)};
        for (int i = 0; i < chords; i++) pieces.Add((route.straightLength + arc * i / chords, route.straightLength + arc * (i + 1) / chords));
        pieces.Add((route.straightLength + arc, 2 * route.straightLength + arc));
        for (int i = 0; i < chords; i++) pieces.Add((2 * route.straightLength + arc + arc * i / chords, 2 * route.straightLength + arc + arc * (i + 1) / chords));
        int count = 0;
        foreach (var (from, to) in pieces)
        {
            var a = route.PoseAt(from).position; var b = route.PoseAt(to).position; var along = b - a;
            var slab = new GameObject($"Track bed {count++}"); slab.transform.SetParent(root, true);
            var centre = (a + b) * .5f + route.transform.rotation * new Vector3(0, -.295f, 0);
            slab.transform.SetPositionAndRotation(centre, Quaternion.LookRotation(along.normalized, route.transform.up));
            slab.isStatic = true;
            var box = slab.AddComponent<BoxCollider>(); box.size = new Vector3(4.2f, .09f, along.magnitude);
            var surface = slab.AddComponent<AcousticSurface>(); surface.geometrySource = AcousticSurface.GeometrySource.Collider; surface.material = ballast;
        }
        return count;
    }
}
