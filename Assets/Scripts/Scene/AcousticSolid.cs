using System;
using System.Collections.Generic;
using UnityEngine;

// One rigid acoustic boundary; source renderers and independently moving objects
// keep their own transforms. The Editor regenerates the exterior automatically
// with an offline Boolean backend and stores it here, keyed by the source hash.
[DisallowMultipleComponent, RequireComponent(typeof(AcousticSurface))]
public sealed class AcousticSolid : MonoBehaviour
{
    public const int BoundaryVersion = 1;
    [Tooltip("Fixed child surfaces, in precedence order. Earlier members own coincident exterior material overlaps.")]
    public AcousticSurface[] members = Array.Empty<AcousticSurface>();
    [Min(.000001f)] public float curveTolerance = .01f;
    // Generated data. Serialized so players and checkouts without Python can capture.
    [SerializeField, HideInInspector] Boundary compiled;

    // Installed by the Editor: make the boundary for (solid, input, hash) current or throw.
    public static Action<AcousticSolid, string, string> Regenerate;

    [Serializable] public sealed class Body { public float[] vertices; public int[] triangles, materials; }
    [Serializable] public sealed class Input { public int version = BoundaryVersion; public Body[] bodies; }
    [Serializable] public sealed class Boundary
    {
        public int version, inputTriangles, outputTriangles, components;
        public string inputSha256, backend;
        public float[] vertices;
        public int[] triangles, materials;
        public double volume, area, tolerance, seconds;
    }

    public Boundary Compiled => compiled != null && !string.IsNullOrEmpty(compiled.inputSha256) ? compiled : null;
    // Material assets for boundary slots, derived from the members on every resolve.
    public AudioMaterial[] SurfaceMaterials { get; private set; } = Array.Empty<AudioMaterial>();
    public bool IsCurrent(string inputHash) => Compiled != null && compiled.version == BoundaryVersion && compiled.inputSha256 == inputHash;

    public Matrix4x4 MemberFrame(AcousticSurface member)
    {
        if (member == null || member.transform == transform || !member.transform.IsChildOf(transform))
            throw new InvalidOperationException(name + ": solid members must be child surfaces.");
        if (!member.enabled || member.GetComponent<AcousticSolid>() != null)
            throw new InvalidOperationException(name + ": use enabled, ungrouped source surfaces.");
        var result = Matrix4x4.identity;
        for (var t = member.transform; t != transform; t = t.parent)
        {
            if (!t.gameObject.activeSelf) throw new InvalidOperationException(name + ": solid members must stay active relative to their group.");
            result = Matrix4x4.TRS(t.localPosition, t.localRotation, t.localScale) * result;
        }
        return result;
    }

    public string ExportInput(AudioMaterial fallback, out AudioMaterial[] materials)
    {
        if (members == null || members.Length == 0) throw new InvalidOperationException(name + ": select fixed child surfaces for the solid.");
        var bindings = new List<RayTraceManager.TakeObjectBinding>();
        var frames = new Matrix4x4[members.Length]; var unique = new HashSet<AcousticSurface>();
        for (int i = 0; i < members.Length; i++)
        {
            frames[i] = MemberFrame(members[i]);
            if (!unique.Add(members[i])) throw new InvalidOperationException(name + ": duplicate solid member.");
            bindings.Add(new RayTraceManager.TakeObjectBinding { id = i.ToString(), transform = members[i].transform, geometry = true });
        }
        var assets = new List<AudioMaterial>();
        var scene = SceneToTake.Compile(bindings, AcousticTake.Dimension.Spatial3D, curveTolerance, fallback,
            Array.Empty<Transform>(), Array.Empty<AudioClip>(), Array.Empty<Transform>(), out var geometry,
            authoringCoordinates: true, materialAssets: assets, useSolidBakes: false);
        materials = assets.ToArray(); ValidateInterior(scene.materials);
        var input = new Input { bodies = new Body[members.Length] };
        for (int i = 0; i < members.Length; i++)
        {
            var obj = scene.objects[i]; int count = obj.primitiveCount;
            if (count == 0) throw new InvalidOperationException(name + ": a solid member has no geometry.");
            var body = new Body { vertices = new float[count * 9], triangles = new int[count * 3], materials = new int[count] };
            for (int t = 0; t < count; t++)
            {
                var p = geometry[obj.primitiveStart + t]; body.materials[t] = p.materialIndex;
                var points = new[] { p.a, p.b, p.c };
                for (int v = 0; v < 3; v++)
                {
                    var point = frames[i].MultiplyPoint3x4(points[v]); int at = 9*t+3*v;
                    body.vertices[at] = point.x; body.vertices[at+1] = point.y; body.vertices[at+2] = point.z;
                    body.triangles[3*t+v] = 3*t+v;
                }
            }
            input.bodies[i] = body;
        }
        return JsonUtility.ToJson(input);
    }

    // The existing take represents one interior medium per closed owner. Surface
    // coefficients may vary freely; merging different interiors would change physics.
    static void ValidateInterior(AcousticTake.MaterialDefinition[] materials)
    {
        var knots = new HashSet<float> { 0 };
        foreach (var material in materials) foreach (float hz in material.frequencies) knots.Add(hz);
        float At(AcousticTake.MaterialDefinition material, float[] values, float hz, float fallback)
        {
            if (values == null || values.Length == 0) return fallback;
            int hi = 0; while (hi < values.Length-1 && material.frequencies[hi] < hz) hi++;
            int lo = Math.Max(0, hi-1);
            return Mathf.Lerp(values[lo], values[hi], lo == hi ? 0 : Mathf.InverseLerp(material.frequencies[lo], material.frequencies[hi], hz));
        }
        var first = materials[0];
        foreach (var other in materials) foreach (float hz in knots)
            if (At(first, first.ior, hz, 1) != At(other, other.ior, hz, 1) || At(first, first.damping, hz, 0) != At(other, other.damping, hz, 0))
                throw new InvalidOperationException("A baked solid must share interior IOR/damping. Keep different interior media in separate groups; surface materials may differ.");
    }

    public string InputHash(AudioMaterial fallback, out string input, out AudioMaterial[] materials)
    {
        input = ExportInput(fallback, out materials);
        return AcousticTake.HashText(input);
    }

    // Returns the boundary for the current members, regenerating it in the Editor
    // when stale. Never returns geometry built from a different source hash.
    public Boundary Resolve(AudioMaterial fallback)
    {
        string hash = InputHash(fallback, out string input, out var materials);
        if (!IsCurrent(hash))
        {
            if (Regenerate == null)
                throw new InvalidOperationException(name + ": the acoustic solid boundary is missing or stale and cannot be regenerated here. " +
                    "Open and save the scene in the Unity Editor with the solid-boundary Python configured, or set the root Geometry Source to None to use the separate members.");
            Regenerate(this, input, hash);
            if (!IsCurrent(hash)) throw new InvalidOperationException(name + ": acoustic solid regeneration did not produce the current boundary.");
        }
        Check(compiled, hash, materials.Length);
        SurfaceMaterials = materials;
        return compiled;
    }

    public void Assign(Boundary data, string inputHash)
    {
        Check(data, inputHash, int.MaxValue);
        compiled = data;
    }

    static void Check(Boundary data, string inputHash, int materialCount)
    {
        if (data == null || data.version != BoundaryVersion || data.inputSha256 != inputHash)
            throw new InvalidOperationException("Boundary data does not match the solid's source hash.");
        if (data.vertices == null || data.vertices.Length % 3 != 0 || data.triangles == null || data.triangles.Length % 3 != 0 ||
            data.materials == null || data.materials.Length*3 != data.triangles.Length || data.materials.Length == 0 || !(data.volume > 0))
            throw new InvalidOperationException("Invalid boundary data.");
        foreach (float v in data.vertices) if (!float.IsFinite(v)) throw new InvalidOperationException("Non-finite boundary vertex.");
        foreach (int v in data.triangles) if (v < 0 || v >= data.vertices.Length/3) throw new InvalidOperationException("Invalid boundary vertex index.");
        foreach (int m in data.materials) if (m < 0 || m >= materialCount) throw new InvalidOperationException("Invalid boundary material index.");
    }

    public sealed class Snapshot
    {
        readonly AcousticSolid solid; readonly bool enabled, grouped; readonly float tolerance;
        readonly Boundary boundary;
        readonly AcousticSurface[] members; readonly AudioMaterial[] materials; readonly Matrix4x4[] frames;
        public Snapshot(AcousticSolid solid, bool geometry)
        {
            this.solid = solid; enabled = solid.enabled; tolerance = solid.curveTolerance;
            grouped = enabled && geometry;
            boundary = solid.compiled;
            members = (AcousticSurface[])solid.members.Clone(); materials = (AudioMaterial[])solid.SurfaceMaterials.Clone();
            frames = new Matrix4x4[members.Length];
            if (grouped) for (int i = 0; i < members.Length; i++) frames[i] = solid.MemberFrame(members[i]);
        }
        public void Validate()
        {
            void Stale() => throw new InvalidOperationException("Acoustic solid membership, boundary or relative motion changed; start a new take.");
            // Reference comparison: any regenerated or reloaded boundary is a new object.
            if (solid == null || solid.enabled != enabled || solid.curveTolerance != tolerance || !ReferenceEquals(solid.compiled, boundary) ||
                solid.members.Length != members.Length || solid.SurfaceMaterials.Length != materials.Length) Stale();
            for (int i = 0; i < materials.Length; i++) if (solid.SurfaceMaterials[i] != materials[i]) Stale();
            for (int i = 0; i < members.Length; i++)
            {
                if (solid.members[i] != members[i]) Stale();
                if (grouped)
                {
                    var now = solid.MemberFrame(members[i]);
                    // Frames are evaluated only from local hierarchy transforms;
                    // whole-group rigid motion cannot introduce cancellation error.
                    if (now != frames[i]) Stale();
                }
            }
        }
    }
}
