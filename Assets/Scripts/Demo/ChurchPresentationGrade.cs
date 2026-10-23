using UnityEngine;

[ExecuteInEditMode]
[RequireComponent(typeof(Camera))]
public sealed class ChurchPresentationGrade : MonoBehaviour
{
    public Shader shader;
    [Range(0f, 2f)] public float contrast = 1.08f;
    [Range(-1f, 1f)] public float exposure = .25f;
    [Range(0f, 1f)] public float saturation = .94f;
    [Range(0f, 1f)] public float vignette = .18f;

    Material material;

    void OnDisable()
    {
        if (material != null)
        {
            if (Application.isPlaying) Destroy(material);
            else DestroyImmediate(material);
            material = null;
        }
    }

    void OnRenderImage(RenderTexture source, RenderTexture destination)
    {
        if (shader == null || !shader.isSupported)
        {
            Graphics.Blit(source, destination);
            return;
        }
        if (material == null) material = new Material(shader) { hideFlags = HideFlags.HideAndDontSave };
        material.SetFloat("_Contrast", contrast);
        material.SetFloat("_Exposure", exposure);
        material.SetFloat("_Saturation", saturation);
        material.SetFloat("_Vignette", vignette);
        Graphics.Blit(source, destination, material);
    }
}
