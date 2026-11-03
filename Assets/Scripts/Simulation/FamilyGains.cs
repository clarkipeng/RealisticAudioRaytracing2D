using System;
using UnityEngine;

// Opt-in per-path-family gains (TakeRender.Settings.familyGains). A level-preserving
// shrinkage of the sampled early families, fitted offline on the audio-DLSS dataset:
// docs/benchmarks/family-gains-2026-11-03. The data lives in
// Resources/Acoustics/FamilyGains.json, never in code. A path of family f is scaled by
// (1 + k_f * rho)^-power, where rho is that family's candidate routes per sample at run
// time, so every budget and scene size has a defined gain. Enumerated, deterministic
// families (rho <= 1 with small k) stay near 1; subsampled ones are shrunk by about
// 1/sqrt(rho), which keeps their expected energy instead of their spikes.
[Serializable]
internal sealed class FamilyGainTable
{
    internal const string Schema="acoustic-family-gains-v1";
    // Shader order: first-order images, then the complete-route families D, DD, RD, DR, RR.
    internal static readonly string[] Families={"R","D","DD","RD","DR","RR"};
    public string schema, method, formula;
    public string[] families;
    public float[] k, fittedRhoMin, fittedRhoMax;
    public float power;
    public string dataset, datasetSha256, selection, selectionSha256, report, fittedAt;
    [NonSerialized] internal string hash;

    internal static FamilyGainTable Shipped()
    {
        var asset=Resources.Load<TextAsset>("Acoustics/FamilyGains");
        if(asset==null)throw new InvalidOperationException("Resources/Acoustics/FamilyGains.json is missing.");
        return Parse(asset.text);
    }
    internal static FamilyGainTable Parse(string json)
    {
        var table=JsonUtility.FromJson<FamilyGainTable>(json);
        table.hash=AcousticTake.HashText(json);
        table.Validate();
        return table;
    }
    // Every gain equals one; used to check that the enabled path changes nothing else.
    internal static FamilyGainTable Unit()
    {
        var table=new FamilyGainTable {schema=Schema,method="unit",families=(string[])Families.Clone(),k=new float[Families.Length],power=.5f};
        table.hash=AcousticTake.HashText(JsonUtility.ToJson(table));table.Validate();return table;
    }
    void Validate()
    {
        if(schema!=Schema || families==null || families.Length!=Families.Length || k==null || k.Length!=Families.Length)
            throw new ArgumentException("Family gain table must list k for "+string.Join(", ",Families)+" in that order.");
        for(int i=0;i<Families.Length;i++)
            if(families[i]!=Families[i] || !float.IsFinite(k[i]) || k[i]<0)
                throw new ArgumentException("Family gain k must be finite and nonnegative, in order "+string.Join(", ",Families)+".");
        if(!(power>0 && power<=1))throw new ArgumentException("Family gain power must lie in (0, 1].");
    }
    // Reference evaluation of the shader's law, for checks and diagnostics.
    internal float Gain(int family,float rho)
    {
        float x=k[family]*rho;
        return x==0?1:Mathf.Pow(1+x,-power);
    }
}
