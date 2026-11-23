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
        this.rate=rate;
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
        DiffuseFieldEqualize(filters,directions,Taps,rate);
        Hash+=":diffuse-field-v1";
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
    // Diffuse-field equalisation (standard practice for binaural rendering over headphones,
    // e.g. Larcher, Jot & Vandernoot, AES 104th Convention 1998; the MIT KEMAR set is
    // distributed with a "diffuse" variant equalised the same way): every HRIR is divided by
    // the diffuse-field average of the set, the solid-angle-weighted mean power over all
    // directions and both ears, smoothed in third octaves. The KEMAR's ear-canal and concha
    // resonances (about +4 dB at 2-3 kHz in a diffuse field) then no longer colour every
    // sound, while the interaural and directional differences, which the same filter
    // multiplies for every direction and ear, are unchanged. The correction is minimum phase
    // (no added delay), 0 dB below 150 Hz and at 200-400 Hz (the renderer's level reference),
    // and limited to +/-12 dB.
    static void DiffuseFieldEqualize(float[] filters,Vector3[] directions,int taps,int rate)
    {
        int n=1024;while(n<8*taps)n*=2;
        int count=directions.Length;var power=new double[n/2+1];double total=0;var data=new Vector2[n];
        // Equal-area weights: the measured rings are evenly spaced in elevation, so a direction's
        // solid angle is about cos(elevation) / (directions in its ring).
        var ring=new Dictionary<int,int>();
        foreach(var d in directions){int key=Mathf.RoundToInt(Mathf.Asin(Mathf.Clamp(d.y,-1,1))*Mathf.Rad2Deg);ring[key]=ring.TryGetValue(key,out int c)?c+1:1;}
        for(int i=0;i<count;i++)
        {
            float elevation=Mathf.Asin(Mathf.Clamp(directions[i].y,-1,1));
            double weight=Math.Max(.05,Math.Cos(elevation))/ring[Mathf.RoundToInt(elevation*Mathf.Rad2Deg)];
            for(int ear=0;ear<2;ear++)
            {
                Array.Clear(data,0,n);for(int t=0;t<taps;t++)data[t].x=filters[(i*2+ear)*taps+t];
                SpectralAudio.Fft(data,false);
                for(int k=0;k<=n/2;k++)power[k]+=weight*(data[k].x*(double)data[k].x+data[k].y*(double)data[k].y);
            }
            total+=2*weight;
        }
        for(int k=0;k<=n/2;k++)power[k]/=total;
        double Hz(int k)=>(double)k*rate/n;
        var smooth=new double[n/2+1];
        for(int k=1;k<=n/2;k++)
        {
            double f=Hz(k),lo=f/Math.Pow(2,1/6.0),hi=f*Math.Pow(2,1/6.0),sum=0;int m=0;
            for(int j=Math.Max(1,(int)Math.Ceiling(lo*n/rate));j<=Math.Min(n/2,(int)Math.Floor(hi*n/rate));j++){sum+=power[j];m++;}
            smooth[k]=m>0?sum/m:power[k];
        }
        smooth[0]=smooth[1];
        double reference=0;int r=0;for(int k=1;k<=n/2;k++)if(Hz(k)>=200 && Hz(k)<=400){reference+=smooth[k];r++;}
        reference=r>0?reference/r:smooth[1];
        var log=new Vector2[n];
        for(int k=0;k<=n/2;k++)
        {
            double f=Math.Min(Hz(k),18000),gain=Math.Sqrt(reference/Math.Max(1e-30,smooth[Math.Min(n/2,(int)Math.Round(f*n/rate))]));
            gain=Math.Min(4,Math.Max(.25,gain));
            double blend=f<=150?0:f>=300?1:Math.Log(f/150)/Math.Log(2);    // 0 dB below 150 Hz
            double value=blend*Math.Log(gain);
            log[k]=new Vector2((float)value,0);if(k>0 && k<n/2)log[n-k]=log[k];
        }
        // Minimum phase from the real cepstrum (fold the anti-causal part onto the causal one).
        SpectralAudio.Fft(log,true);
        for(int k=1;k<n/2;k++){log[k]*=2;log[n-k]=Vector2.zero;}
        SpectralAudio.Fft(log,false);
        var eq=new Vector2[n];
        for(int k=0;k<n;k++){double m=Math.Exp(log[k].x);eq[k]=new Vector2((float)(m*Math.Cos(log[k].y)),(float)(m*Math.Sin(log[k].y)));}
        for(int i=0;i<count;i++)for(int ear=0;ear<2;ear++)
        {
            Array.Clear(data,0,n);for(int t=0;t<taps;t++)data[t].x=filters[(i*2+ear)*taps+t];
            SpectralAudio.Fft(data,false);
            for(int k=0;k<n;k++){var a=data[k];var b=eq[k];data[k]=new Vector2(a.x*b.x-a.y*b.y,a.x*b.y+a.y*b.x);}
            SpectralAudio.Fft(data,true);
            for(int t=0;t<taps;t++)filters[(i*2+ear)*taps+t]=data[t].x;
        }
    }
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
    internal static Vector3 HeadVelocity(AcousticTake.State state) =>
        (state.flags&(AcousticTake.StateFlags.VelocityValid|AcousticTake.StateFlags.Cut))==AcousticTake.StateFlags.VelocityValid?state.angularVelocity:Vector3.zero;
    internal static Vector2[] Spectra(Hrtf hrtf,int fft)
    {
        int directions=hrtf?.DirectionCount??1,channels=hrtf==null?1:2,bins=fft/2+1;
        var spectra=new Vector2[checked(directions*channels*bins)];var data=new Vector2[fft];
        for(int direction=0;direction<directions;direction++)for(int ear=0;ear<channels;ear++)
        {
            Array.Clear(data,0,fft);for(int t=0;t<(hrtf?.Taps??1);t++)data[t].x=hrtf==null?1:hrtf.Coefficient(direction,ear,t);
            SpectralAudio.Fft(data,false);Array.Copy(data,0,spectra,(direction*channels+ear)*bins,bins);
        }
        return spectra;
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

    // Interaural time difference (README part 42), the set's own: per measured direction, the lag of the peak of the
    // cross-correlation of its left and right HRIRs (the broadband interaural cross-correlation estimator; Katz &
    // Noisternig, JASA 135, 2014, compare it with the others), searched within +-1 ms (beyond any head's ITD), between
    // samples by band-limited interpolation of the correlation (PeakLag). Seconds, positive when the left ear hears later
    // (a source on the right). Over a grid of head-frame directions, every 2 degrees of azimuth (atan2(x, z)) and
    // elevation (asin y), each point the blend of the measured directions the direct path uses there (Interpolate), so
    // an echo arriving from a direction gets the ITD the direct path has from it; read bilinearly (InterauralDelay).
    public const int ItdAzimuths=181,ItdElevations=91;
    public const float ItdStepDegrees=2;
    float[] itdGrid;
    public float[] InterauralDelayGrid {get {if(itdGrid==null)itdGrid=BuildItdGrid();return itdGrid;}}
    readonly int rate;
    // The lag, in samples, of the peak of a correlation r(lag) of two band-limited signals within +-lags: the largest
    // sample, then the maximum of r interpolated between samples (a Lanczos-windowed sinc of radius 8, which is exact
    // for a band-limited r up to the window's ripple) within a sample of it, on a 1/64-sample grid.
    public static double PeakLag(Func<int,double> correlation,int lags)
    {
        const int radius=8;int reach=lags+radius+1;
        var r=new double[2*reach+1];for(int l=-reach;l<=reach;l++)r[l+reach]=correlation(l);
        int best=-lags;for(int l=-lags;l<=lags;l++)if(r[l+reach]>r[best+reach])best=l;
        double peakLag=best,peak=double.NegativeInfinity;
        for(int step=-64;step<=64;step++)
        {
            double tau=best+step/64.0,sum=0;
            for(int l=(int)Math.Floor(tau)-radius+1;l<=(int)Math.Floor(tau)+radius;l++)
            {double x=tau-l;sum+=r[l+reach]*Sinc(x)*Sinc(x/radius);}
            if(sum>peak){peak=sum;peakLag=tau;}
        }
        return peakLag;
    }
    float[] BuildItdGrid()
    {
        int lags=(int)Math.Ceiling(.001*rate);var measured=new float[directions.Length];
        for(int i=0;i<directions.Length;i++)
            measured[i]=(float)(PeakLag(lag=>{double sum=0;for(int t=Math.Max(0,-lag);t<Taps && t+lag<Taps;t++)sum+=(double)Coefficient(i,0,t+lag)*Coefficient(i,1,t);return sum;},lags)/rate);
        var grid=new float[ItdAzimuths*ItdElevations];
        for(int e=0;e<ItdElevations;e++)for(int a=0;a<ItdAzimuths;a++)
        {
            float azimuth=(-180+a*ItdStepDegrees)*Mathf.Deg2Rad,elevation=(-90+e*ItdStepDegrees)*Mathf.Deg2Rad;
            var blend=Interpolate(new Vector3(Mathf.Cos(elevation)*Mathf.Sin(azimuth),Mathf.Sin(elevation),Mathf.Cos(elevation)*Mathf.Cos(azimuth)));
            grid[e*ItdAzimuths+a]=blend.weights.x*measured[blend.a]+blend.weights.y*measured[blend.b]+blend.weights.z*measured[blend.c]+blend.weights.w*measured[blend.d];
        }
        return grid;
    }
    // The ITD (seconds) from a head-frame direction, bilinear in the grid (GpuAudio.compute PathInterauralDelay is the same).
    public static float InterauralDelay(float[] grid,Vector3 direction)
    {
        if(!(direction.sqrMagnitude>0))return 0;
        direction.Normalize();
        float fa=(Mathf.Atan2(direction.x,direction.z)*Mathf.Rad2Deg+180)/ItdStepDegrees,fe=(Mathf.Asin(Mathf.Clamp(direction.y,-1,1))*Mathf.Rad2Deg+90)/ItdStepDegrees;
        int a=Mathf.Clamp((int)Mathf.Floor(fa),0,ItdAzimuths-2),e=Mathf.Clamp((int)Mathf.Floor(fe),0,ItdElevations-2);
        float ta=Mathf.Clamp01(fa-a),te=Mathf.Clamp01(fe-e);
        float low=grid[e*ItdAzimuths+a]*(1-ta)+grid[e*ItdAzimuths+a+1]*ta,high=grid[(e+1)*ItdAzimuths+a]*(1-ta)+grid[(e+1)*ItdAzimuths+a+1]*ta;
        return low*(1-te)+high*te;
    }
    public Vector3 Direction(int index) => directions[index];
}
