using System;
using System.Collections.Generic;
using UnityEngine;

// One rigid acoustic boundary; source renderers and independently moving objects
// keep their own transforms. The Boolean operation runs offline in the Editor.
[DisallowMultipleComponent, RequireComponent(typeof(AcousticSurface))]
public sealed class AcousticSolid : MonoBehaviour
{
    [Tooltip("Fixed child surfaces, in precedence order. Earlier members own coincident exterior material overlaps.")]
    public AcousticSurface[] members = Array.Empty<AcousticSurface>();
    [Min(.000001f)] public float curveTolerance = .01f;
    public TextAsset boundary;
    [HideInInspector] public AudioMaterial[] surfaceMaterials = Array.Empty<AudioMaterial>();

    [Serializable] public sealed class Body { public float[] vertices; public int[] triangles, materials; }
    [Serializable] public sealed class Input { public int version = 1; public Body[] bodies; }
    [Serializable] public sealed class Boundary
    {
        public int version, inputTriangles, outputTriangles, components;
        public string inputSha256, backend;
        public float[] vertices;
        public int[] triangles, materials;
        public double volume, area, tolerance, seconds;
    }

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
        if (members == null || members.Length == 0) throw new InvalidOperationException(name + ": select fixed child surfaces to bake.");
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

    public Boundary ReadValidated(AudioMaterial fallback)
    {
        if (boundary == null) throw new InvalidOperationException(name + ": bake the acoustic solid first.");
        var data = JsonUtility.FromJson<Boundary>(boundary.text);
        string input = ExportInput(fallback, out var materials);
        if (data == null || data.version != 1 || data.inputSha256 != AcousticTake.HashText(input) ||
            materials.Length != surfaceMaterials.Length)
            throw new InvalidOperationException(name + ": acoustic solid bake is stale; rebake changed source geometry.");
        for (int i = 0; i < materials.Length; i++) if (materials[i] != surfaceMaterials[i])
            throw new InvalidOperationException(name + ": acoustic solid material assignments changed; rebake.");
        if (data.vertices == null || data.vertices.Length % 3 != 0 || data.triangles == null || data.triangles.Length % 3 != 0 ||
            data.materials == null || data.materials.Length*3 != data.triangles.Length || !(data.volume > 0))
            throw new InvalidOperationException(name + ": invalid boundary data.");
        foreach (float v in data.vertices) if (!float.IsFinite(v)) throw new InvalidOperationException("Non-finite boundary vertex.");
        foreach (int v in data.triangles) if (v < 0 || v >= data.vertices.Length/3) throw new InvalidOperationException("Invalid boundary vertex index.");
        foreach (int m in data.materials) if (m < 0 || m >= materials.Length) throw new InvalidOperationException("Invalid boundary material index.");
        return data;
    }

    public sealed class Snapshot
    {
        readonly AcousticSolid solid; readonly bool enabled; readonly float tolerance;
        readonly TextAsset boundary; readonly byte[] bytes;
        readonly AcousticSurface[] members; readonly AudioMaterial[] materials; readonly Matrix4x4[] frames;
        public Snapshot(AcousticSolid solid)
        {
            this.solid = solid; enabled = solid.enabled; tolerance = solid.curveTolerance;
            boundary = solid.boundary; bytes = boundary == null ? null : boundary.bytes;
            members = (AcousticSurface[])solid.members.Clone(); materials = (AudioMaterial[])solid.surfaceMaterials.Clone();
            frames = new Matrix4x4[members.Length];
            if (enabled) for (int i = 0; i < members.Length; i++) frames[i] = solid.MemberFrame(members[i]);
        }
        public void Validate()
        {
            void Stale() => throw new InvalidOperationException("Acoustic solid membership, bake or relative motion changed; stop and rebake.");
            if (solid == null || solid.enabled != enabled || solid.curveTolerance != tolerance || solid.boundary != boundary ||
                solid.members.Length != members.Length || solid.surfaceMaterials.Length != materials.Length) Stale();
            if (boundary != null)
            {
                var data = boundary.GetData<byte>();
                if (data.Length != bytes.Length) Stale();
                for (int i = 0; i < bytes.Length; i++) if (data[i] != bytes[i]) Stale();
            }
            for (int i = 0; i < materials.Length; i++) if (solid.surfaceMaterials[i] != materials[i]) Stale();
            for (int i = 0; i < members.Length; i++)
            {
                if (solid.members[i] != members[i]) Stale();
                if (enabled)
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
