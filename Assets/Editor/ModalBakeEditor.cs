using System;
using System.Collections.Generic;
using System.IO;
using UnityEditor;
using UnityEngine;

// Export mechanical inputs without changing the scene, colliders or tracing geometry.
public static class ModalBakeEditor
{
    [Serializable] public sealed class Material
    {
        public double density, young, poisson, loss;
    }
    [Serializable] public sealed class Geometry
    {
        public string kind;
        public double[] size, center, basis, vertices;
        public int[] triangles;
    }
    [Serializable] public sealed class Input
    {
        public int version = 1;
        public string name, support = "free";
        public Geometry geometry;
        public Material material;
    }

    [MenuItem("Tools/Acoustics/Modal Bake/Export selected mesh")]
    static void ExportMesh() => Export(false);
    [MenuItem("Tools/Acoustics/Modal Bake/Export selected box collider")]
    static void ExportBox() => Export(true);

    static void Export(bool box)
    {
        try
        {
            var input = Describe(Selection.activeGameObject, box);
            string path = EditorUtility.SaveFilePanel("Export solid vibration inputs", ".context", input.name+"-modal", "json");
            if (string.IsNullOrEmpty(path)) return;
            File.WriteAllText(path, JsonUtility.ToJson(input, true)+"\n");
            Debug.Log("Modal inputs exported to "+path+". Support defaults to free; set mounting in the JSON before baking. Use tools/modal_bake.py fit/bake.");
        }
        catch (Exception error) { Debug.LogException(error); }
    }

    public static Input Describe(GameObject target, bool box)
    {
        if (target == null) throw new ArgumentException("Select one physical object.");
        var surface = target.GetComponent<AcousticSurface>();
        var material = surface == null ? null : surface.material;
        if (material == null) throw new ArgumentException("Assign the object's AcousticSurface material before modal export.");
        if (surface.submeshMaterials != null)
            foreach (var other in surface.submeshMaterials)
                if (other != null && other != material)
                    throw new ArgumentException("Modal export currently requires one uniform material.");
        if (!(material.density > 0) || !float.IsFinite(material.density) ||
            !(material.modalYoungModulus > 0) || !float.IsFinite(material.modalYoungModulus) ||
            !(material.modalPoissonRatio > -1 && material.modalPoissonRatio < .5) ||
            !(material.modalLossFactor >= 0 && material.modalLossFactor < 2))
            throw new ArgumentException("Invalid physical solid material properties.");
        var input = new Input { name=target.name, material=new Material {
            density=material.density, young=material.modalYoungModulus,
            poisson=material.modalPoissonRatio, loss=material.modalLossFactor } };
        // Bake scale/rotation into metre coordinates relative to the object's world origin.
        var transform = target.transform;
        if (box)
        {
            var collider = target.GetComponent<BoxCollider>();
            if (collider == null) throw new ArgumentException("A BoxCollider is required for primitive export.");
            var x=transform.TransformVector(Vector3.right); var y=transform.TransformVector(Vector3.up); var z=transform.TransformVector(Vector3.forward);
            var size=Vector3.Scale(collider.size,new Vector3(x.magnitude,y.magnitude,z.magnitude));
            if (!(size.x>0 && size.y>0 && size.z>0)) throw new ArgumentException("Box dimensions must be positive.");
            x.Normalize(); y.Normalize(); z.Normalize();
            if (Mathf.Abs(Vector3.Dot(x,y))>1e-5 || Mathf.Abs(Vector3.Dot(x,z))>1e-5 || Mathf.Abs(Vector3.Dot(y,z))>1e-5)
                throw new ArgumentException("Sheared transforms need mesh export.");
            // A reflected box describes the same solid with a right-handed local basis.
            z=Vector3.Cross(x,y).normalized;
            input.geometry = new Geometry { kind="box", size=Array(size), center=Array(transform.TransformVector(collider.center)),
                basis=new double[] {x.x,y.x,z.x,x.y,y.y,z.y,x.z,y.z,z.z} };
        }
        else
        {
            var filter=target.GetComponent<MeshFilter>();
            var collider=target.GetComponent<MeshCollider>();
            var mesh=filter!=null ? filter.sharedMesh : collider!=null ? collider.sharedMesh : null;
            if (mesh==null || !mesh.isReadable) throw new ArgumentException("A readable MeshFilter or MeshCollider is required.");
            var vertices=mesh.vertices; var positions=new double[vertices.Length*3];
            for (int i=0;i<vertices.Length;i++)
            {
                var p=transform.TransformVector(vertices[i]);
                if (!float.IsFinite(p.x) || !float.IsFinite(p.y) || !float.IsFinite(p.z)) throw new ArgumentException("Non-finite mesh position.");
                positions[3*i]=p.x; positions[3*i+1]=p.y; positions[3*i+2]=p.z;
            }
            var triangles=new List<int>();
            for (int i=0;i<mesh.subMeshCount;i++)
            {
                if (mesh.GetTopology(i)!=MeshTopology.Triangles) throw new ArgumentException("Triangulate the solid before export.");
                triangles.AddRange(mesh.GetTriangles(i));
            }
            input.geometry=new Geometry {kind="mesh", vertices=positions, triangles=triangles.ToArray()};
        }
        return input;
    }

    static double[] Array(Vector3 value) => new double[] {value.x,value.y,value.z};
}
