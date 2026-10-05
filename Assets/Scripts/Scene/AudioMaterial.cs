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
    [System.Serializable] public struct EnergyKnot
    {
        [Min(0)] public float frequencyHz;
        [Range(0, 1)] public float absorption, scattering;
    }

    [Tooltip("Optional ordered energy coefficients at any frequency knots. Empty preserves the legacy material. Linear interpolation in Hz, clamped at endpoints.")]
    public EnergyKnot[] energyCurve = new EnergyKnot[0];

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
    [Tooltip("Acoustic index = air speed / material speed. Values below 1 mean faster sound inside a closed solid; open sheets have no interior distance.")]
    [Min(0.001f)] public float[] iorBands = DefaultArray(343f / 5405f);
    [Tooltip("Interior pressure attenuation in nepers per metre: gain = exp(-damping * distance).")]
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

        if (energyCurve != null && energyCurve.Length > 0)
        {
            ValidateEnergyCurve();
            for (int i = 0; i < BandCount; i++)
            {
                int upper = 0;
                while (upper < energyCurve.Length - 1 && energyCurve[upper].frequencyHz < OctaveBandsHz[i]) upper++;
                var a = energyCurve[Mathf.Max(0, upper - 1)]; var b = energyCurve[upper];
                float t = a.frequencyHz == b.frequencyHz ? 0 : Mathf.InverseLerp(a.frequencyHz, b.frequencyHz, OctaveBandsHz[i]);
                absorption[i] = Mathf.Lerp(a.absorption, b.absorption, t);
                scattering[i] = Mathf.Lerp(a.scattering, b.scattering, t);
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

    public AcousticTake.MaterialDefinition ExportEnergyCurve()
    {
        var result = new AcousticTake.MaterialDefinition { name = name };
        if (energyCurve != null && energyCurve.Length > 0)
        {
            ValidateEnergyCurve();
            result.frequencies = new float[energyCurve.Length];
            result.absorption = new float[energyCurve.Length]; result.scattering = new float[energyCurve.Length];
            for (int i = 0; i < energyCurve.Length; i++)
            {
                result.frequencies[i] = energyCurve[i].frequencyHz;
                result.absorption[i] = energyCurve[i].absorption; result.scattering[i] = energyCurve[i].scattering;
            }
        }
        else
        {
            var legacy = GetShaderData();
            result.frequencies = (float[])OctaveBandsHz.Clone();
            result.absorption = Unpack(legacy.absorptionLow, legacy.absorptionHigh);
            result.scattering = Unpack(legacy.scatteringLow, legacy.scatteringHigh);
        }
        var transport=GetShaderData();
        result.transmission=new float[result.frequencies.Length];result.ior=new float[result.frequencies.Length];result.damping=new float[result.frequencies.Length];
        for(int i=0;i<result.frequencies.Length;i++)
        {
            float band=Mathf.Clamp(Mathf.Log(Mathf.Max(125,result.frequencies[i])/125,2),0,7);
            int a=(int)band,b=Mathf.Min(a+1,7);float t=band-a;
            float Sample(Vector4 lo,Vector4 hi) => Mathf.Lerp(a<4?lo[a]:hi[a-4],b<4?lo[b]:hi[b-4],t);
            result.transmission[i]=Sample(transport.transmissionLow,transport.transmissionHigh);
            result.ior[i]=Sample(transport.iorLow,transport.iorHigh);result.damping[i]=Sample(transport.dampingLow,transport.dampingHigh);
        }
        return result;
    }

    void ValidateEnergyCurve()
    {
        for (int i = 0; i < energyCurve.Length; i++)
        {
            var p = energyCurve[i];
            if (!(p.frequencyHz >= 0) || float.IsInfinity(p.frequencyHz) ||
                (i > 0 && p.frequencyHz <= energyCurve[i - 1].frequencyHz) ||
                !(p.absorption >= 0 && p.absorption <= 1) || !(p.scattering >= 0 && p.scattering <= 1))
                throw new System.ArgumentException(name + ": energy knots require increasing finite frequencies and coefficients in [0, 1].");
        }
    }

    static float[] Unpack(Vector4 lo, Vector4 hi) => new[] { lo.x, lo.y, lo.z, lo.w, hi.x, hi.y, hi.z, hi.w };

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
