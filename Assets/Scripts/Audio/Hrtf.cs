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
    readonly float[] elevations;
    readonly int[][] rings;
    readonly float[][] azimuths;
    public struct Blend
    {
        public int a,b,c,d;
        public Vector4 weights;
    }
    public int Taps {get;private set;}
    public string Hash {get;private set;}
    public long NumericBytes => (long)directions.Length*20+(long)(filters.Length+elevations.Length)*4;
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
        // Use the measured elevation rings and their individual azimuth grids.
        // Circular interpolation also covers the rear seam; no angular snapping.
        var groups=new SortedDictionary<float,List<int>>();
        for(int i=0;i<directions.Length;i++)
        {
            float elevation=Mathf.Asin(Mathf.Clamp(directions[i].y,-1,1));
            if(!groups.TryGetValue(elevation,out var ring))groups.Add(elevation,ring=new List<int>());
            ring.Add(i);
        }
        elevations=new float[groups.Count];rings=new int[groups.Count][];azimuths=new float[groups.Count][];
        int row=0;
        foreach(var group in groups)
        {
            group.Value.Sort((a,b)=>Azimuth(directions[a]).CompareTo(Azimuth(directions[b])));
            elevations[row]=group.Key;rings[row]=group.Value.ToArray();azimuths[row]=new float[group.Value.Count];
            for(int i=0;i<rings[row].Length;i++)azimuths[row][i]=Azimuth(directions[rings[row][i]]);
            row++;
        }
    }
    static float Azimuth(Vector3 direction)=>Mathf.Atan2(direction.x,direction.z);
    public Blend Interpolate(Vector3 direction)
    {
        direction=direction.sqrMagnitude<1e-12f?Vector3.forward:direction.normalized;
        float elevation=Mathf.Asin(Mathf.Clamp(direction.y,-1,1)),azimuth=Azimuth(direction);
        int upper=Array.BinarySearch(elevations,elevation);if(upper<0)upper=~upper;
        upper=Math.Min(upper,elevations.Length-1);int lower=Math.Max(0,upper-1);
        float t=lower==upper?0:Mathf.InverseLerp(elevations[lower],elevations[upper],elevation);
        Around(lower,azimuth,out int a,out int b,out float x);Around(upper,azimuth,out int c,out int d,out float y);
        return new Blend {a=a,b=b,c=c,d=d,weights=new Vector4((1-t)*(1-x),(1-t)*x,t*(1-y),t*y)};
    }
    void Around(int row,float angle,out int a,out int b,out float t)
    {
        var grid=azimuths[row];int upper=Array.BinarySearch(grid,angle);if(upper<0)upper=~upper;
        int lower=(upper+grid.Length-1)%grid.Length;upper%=grid.Length;
        a=rings[row][lower];b=rings[row][upper];
        float span=Mathf.Repeat(grid[upper]-grid[lower],2*Mathf.PI);
        t=span==0?0:Mathf.Clamp01(Mathf.Repeat(angle-grid[lower],2*Mathf.PI)/span);
    }
    public float Coefficient(Blend blend,int ear,int tap)=>blend.weights.x*Coefficient(blend.a,ear,tap)+
        blend.weights.y*Coefficient(blend.b,ear,tap)+blend.weights.z*Coefficient(blend.c,ear,tap)+blend.weights.w*Coefficient(blend.d,ear,tap);
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
