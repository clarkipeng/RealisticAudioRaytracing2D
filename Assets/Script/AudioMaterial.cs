using UnityEngine;

public struct AudioMatData
{
    public Vector4 absorptionLow;
    public Vector4 absorptionHigh;
    public Vector4 scatteringLow;
    public Vector4 scatteringHigh;
    public Vector4 transmissionLow;
    public Vector4 transmissionHigh;
    public Vector4 iorLow;
    public Vector4 iorHigh;
    public Vector4 dampingLow;
    public Vector4 dampingHigh;
}
[CreateAssetMenu(fileName = "NewAudioMaterial", menuName = "Audio/Audio Material")]
public class AudioMaterial : ScriptableObject
{
    const int BandCount = 8;
    static readonly float[] OctaveBandsHz = { 125f, 250f, 500f, 1000f, 2000f, 4000f, 8000f, 16000f };

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

    public bool overrideTrans = false;

    [Header("Octave Band Overrides")]
    [Tooltip("Use explicit 125, 250, 500, 1k, 2k, 4k, 8k, 16k Hz coefficients instead of deriving them from the physical controls.")]
    public bool overrideBands = false;

    [Range(0f, 1f)] public float[] absorptionBands = DefaultArray(0.1f);
    [Range(0f, 1f)] public float[] scatteringBands = DefaultArray(0.5f);
    [Range(0f, 1f)] public float[] transmissionBands = DefaultArray(0.05f);
    [Min(0.001f)] public float[] iorBands = DefaultArray(343f / 5405f);
    [Min(0f)] public float[] dampingBands = DefaultArray(0.05f);

    public AudioMatData GetShaderData()
    {
        float speedOfSoundAir = 343f;
        float densityAir = 1.225f;
        float impedanceAir = densityAir * speedOfSoundAir;

        float speedOfSoundMat = Mathf.Lerp(50f, 6000f, rigidity);
        float impedanceMat = density * speedOfSoundMat; // Z_mat

        // Normal-incidence energy reflection coefficient:
        // R = ((Z2 - Z1) / (Z2 + Z1))^2.
        float R = Mathf.Pow((impedanceMat - impedanceAir) / (impedanceMat + impedanceAir), 2.0f);
        float baseTransmission = overrideTrans ? 0.5f : 1.0f - R;

        float baseAbsorption = Mathf.Clamp01(porosity + (roughness * 0.1f));
        float baseDamping = (1.0f - rigidity) * 0.5f + porosity * 0.5f;
        float baseIor = speedOfSoundAir / speedOfSoundMat;

        float[] absorption = new float[BandCount];
        float[] scattering = new float[BandCount];
        float[] transmission = new float[BandCount];
        float[] ior = new float[BandCount];
        float[] damping = new float[BandCount];

        if (overrideBands)
        {
            FillFromOverride(absorption, absorptionBands, baseAbsorption, clamp01: true);
            FillFromOverride(scattering, scatteringBands, roughness, clamp01: true);
            FillFromOverride(transmission, transmissionBands, baseTransmission, clamp01: true);
            FillFromOverride(ior, iorBands, baseIor, clamp01: false);
            FillFromOverride(damping, dampingBands, baseDamping, clamp01: false);
        }
        else
        {
            for (int i = 0; i < BandCount; i++)
            {
                float logT = i / (float)(BandCount - 1);
                float highFrequencyRise = Mathf.Pow(logT, 1.35f);
                float lowFrequencyMassBlocking = 1.0f - (0.45f * logT);
                float porousHighBandAbsorption = porosity * highFrequencyRise * 0.75f;

                absorption[i] = Mathf.Clamp01(baseAbsorption * (0.65f + 0.7f * highFrequencyRise) + porousHighBandAbsorption);
                scattering[i] = Mathf.Clamp01(roughness * (0.75f + 0.35f * highFrequencyRise));
                transmission[i] = Mathf.Clamp01(baseTransmission * lowFrequencyMassBlocking * (1.0f - porosity * highFrequencyRise * 0.45f));
                ior[i] = Mathf.Max(0.001f, baseIor * (1.0f + (OctaveBandsHz[i] / 10000f) * 0.1f));
                damping[i] = Mathf.Max(0f, baseDamping * (0.55f + 1.25f * highFrequencyRise));
            }
        }

        return new AudioMatData {
            absorptionLow = PackLow(absorption),
            absorptionHigh = PackHigh(absorption),
            scatteringLow = PackLow(scattering),
            scatteringHigh = PackHigh(scattering),
            transmissionLow = PackLow(transmission),
            transmissionHigh = PackHigh(transmission),
            iorLow = PackLow(ior),
            iorHigh = PackHigh(ior),
            dampingLow = PackLow(damping),
            dampingHigh = PackHigh(damping)
        };
    }

    void OnValidate()
    {
        EnsureBandArray(ref absorptionBands, 0.1f);
        EnsureBandArray(ref scatteringBands, roughness);
        EnsureBandArray(ref transmissionBands, overrideTrans ? 0.5f : 0.05f);
        EnsureBandArray(ref iorBands, 343f / Mathf.Lerp(50f, 6000f, rigidity));
        EnsureBandArray(ref dampingBands, (1.0f - rigidity) * 0.5f + porosity * 0.5f);
    }

    static float[] DefaultArray(float value)
    {
        float[] values = new float[BandCount];
        for (int i = 0; i < values.Length; i++) values[i] = value;
        return values;
    }

    static void EnsureBandArray(ref float[] values, float fill)
    {
        if (values != null && values.Length == BandCount) return;

        float[] resized = new float[BandCount];
        for (int i = 0; i < resized.Length; i++)
            resized[i] = values != null && i < values.Length ? values[i] : fill;
        values = resized;
    }

    static void FillFromOverride(float[] target, float[] source, float fallback, bool clamp01)
    {
        for (int i = 0; i < target.Length; i++)
        {
            float value = source != null && i < source.Length ? source[i] : fallback;
            target[i] = clamp01 ? Mathf.Clamp01(value) : Mathf.Max(0f, value);
        }
    }

    static Vector4 PackLow(float[] values) => new Vector4(values[0], values[1], values[2], values[3]);
    static Vector4 PackHigh(float[] values) => new Vector4(values[4], values[5], values[6], values[7]);
}
