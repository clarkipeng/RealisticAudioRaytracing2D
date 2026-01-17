using UnityEngine;

public struct AudioMatData
{
    public float absorption;
    public float scattering;
    public float transmission;
    public float ior;
    public float damping;
}
[CreateAssetMenu(fileName = "NewAudioMaterial", menuName = "Audio/Audio Material")]
public class AudioMaterial : ScriptableObject
{
    [Header("Physical Properties")]
    [Tooltip("Surface texture. 0 = Polished Mirror, 1 = Rough/Diffuse.")]
    [Range(0f, 1f)]
    public float roughness = 0.5f;

    [Tooltip("Density in kg/m^3. Air=1.2, Wood=700, Concrete=2400, Steel=8000.")]
    [Min(0f)]
    public float density = 2400f;

    [Tooltip("How 'open' the material structure is. 0 = Solid/Watertight, 1 = Open Foam/Cloth.")]
    [Range(0f, 1f)]
    public float porosity = 0.1f;

    [Tooltip("Stiffness. 0 = Rubber/Cloth (Slow Sound), 1 = Steel/Glass (Fast Sound).")]
    [Range(0f, 1f)]
    public float rigidity = 0.9f;

    public AudioMatData GetShaderData()
    {
        AudioMatData data = new AudioMatData();

        float speedOfSoundAir = 343f;
        float densityAir = 1.225f;
        float impedanceAir = densityAir * speedOfSoundAir;

        float speedOfSoundMat = Mathf.Lerp(50f, 6000f, rigidity);
        float impedanceMat = density * speedOfSoundMat; // Z_mat

        // Reflection Coefficient (R)
        // R = ((Z2 - Z1) / (Z2 + Z1))^2
        float R = Mathf.Pow((impedanceMat - impedanceAir) / (impedanceMat + impedanceAir), 2.0f);
        data.transmission = 1.0f - R;

        data.scattering = roughness;
        data.absorption = Mathf.Clamp01(porosity + (roughness * 0.1f));
        data.ior = speedOfSoundAir / speedOfSoundMat;
        data.damping = (1.0f - rigidity) * 0.5f + porosity * 0.5f;

        return data;
    }
}