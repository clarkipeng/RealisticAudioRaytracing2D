using System;
using System.Collections.Generic;
using System.IO;
using System.Security.Cryptography;
using UnityEngine;

// Loaded/resampled on the control thread. No Unity access or allocation on the audio callback.
public sealed class Hrtf
{
    public const string Dataset="MIT-KEMAR-compact-v1";
    static readonly Dictionary<int,Hrtf> cache=new Dictionary<int,Hrtf>();
    readonly Vector3[] directions;
    readonly float[] filters;
    public int Taps {get;private set;}
    public string Hash {get;private set;}
    public long NumericBytes => (long)directions.Length*12+(long)filters.Length*4;
    public static Hrtf Load(int rate)
    {
        if(rate<=0) throw new ArgumentOutOfRangeException(nameof(rate));
        if(!cache.TryGetValue(rate,out var value)) {value=new Hrtf(rate); cache.Add(rate,value);}
        return value;
    }
    Hrtf(int rate)
    {
        var asset=Resources.Load<TextAsset>("Acoustics/Kemar");
        if(asset==null) throw new InvalidOperationException("Missing bundled KEMAR HRTF data.");
        var bytes=asset.bytes;
        using(var sha=SHA256.Create()) Hash=BitConverter.ToString(sha.ComputeHash(bytes)).Replace("-","").ToLowerInvariant();
        using(var reader=new BinaryReader(new MemoryStream(bytes,false)))
        {
            if(new string(reader.ReadChars(4))!="KMR1") throw new InvalidDataException("Unsupported HRTF format.");
            int nativeRate=reader.ReadInt32(),count=reader.ReadInt32(),nativeTaps=reader.ReadInt32();
            if(nativeRate<=0 || count<=0 || nativeTaps<=0 || bytes.LongLength!=16L+count*(12L+8L*nativeTaps)) throw new InvalidDataException("Invalid HRTF dimensions.");
            Taps=checked((int)Math.Ceiling((double)nativeTaps*rate/nativeRate));
            directions=new Vector3[count]; filters=new float[checked(count*2*Taps)]; var input=new float[nativeTaps];
            for(int i=0;i<count;i++)
            {
                directions[i]=new Vector3(reader.ReadSingle(),reader.ReadSingle(),reader.ReadSingle());
                if(!float.IsFinite(directions[i].sqrMagnitude) || Mathf.Abs(directions[i].sqrMagnitude-1)>1e-4f) throw new InvalidDataException("Invalid HRTF direction.");
                for(int ear=0;ear<2;ear++)
                {
                    for(int k=0;k<nativeTaps;k++) {input[k]=reader.ReadSingle(); if(!float.IsFinite(input[k])) throw new InvalidDataException("Non-finite HRTF.");}
                    for(int k=0;k<Taps;k++) filters[(i*2+ear)*Taps+k]=rate==nativeRate ? input[k]:Resample(input,(double)k*nativeRate/rate,(double)nativeRate/rate);
                }
            }
        }
    }
    // Symmetric windowed-sinc coefficient conversion; retain filter time origin and DC scale.
    static float Resample(float[] input,double at,double ratio)
    {
        const int radius=24;
        double cutoff=Math.Min(1,1/ratio),sum=0;
        int extent=(int)Math.Ceiling(radius/cutoff);
        for(int k=Math.Max(0,(int)Math.Ceiling(at-extent));k<=Math.Min(input.Length-1,(int)Math.Floor(at+extent));k++)
        {
            double x=(at-k)*cutoff;
            if(Math.Abs(x)<radius) sum+=input[k]*Sinc(x)*Sinc(x/radius)*cutoff;
        }
        return (float)(sum*ratio);
    }
    static double Sinc(double x) => Math.Abs(x)<1e-12 ? 1:Math.Sin(Math.PI*x)/(Math.PI*x);
    public int Nearest(Vector3 direction)
    {
        // Coincident source/head has no direction; use the documented frontal convention.
        if(direction.sqrMagnitude<1e-12f) direction=Vector3.forward;
        int index=0; float best=float.NegativeInfinity;
        for(int i=0;i<directions.Length;i++) {float dot=Vector3.Dot(direction,directions[i]); if(dot>best) {best=dot; index=i;}}
        return index;
    }
    public float Coefficient(int direction,int ear,int tap) => filters[(direction*2+ear)*Taps+tap];
    public int DirectionCount => directions.Length;
    public Vector3 Direction(int index) => directions[index];
}
