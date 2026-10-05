using UnityEngine;

public class AcousticSurface : MonoBehaviour
{
    public enum GeometrySource { Auto, Mesh, Collider, None }
    public GeometrySource geometrySource;
    public AudioMaterial material;
    public AudioMaterial[] submeshMaterials;
}
