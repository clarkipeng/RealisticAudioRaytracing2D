using System;
using System.Collections.Generic;
using UnityEngine;

// Fixed noise for late-field synthesis (Polack 1993), shared by the CPU and GPU
// audio backends. Each output channel owns one long seeded white noise, cut into
// partitions and stored as the spectra of its zero-padded partitions, ready for
// uniform partitioned convolution. Binaural channels mix two independent noises to
// the dataset HRTF's diffuse-field interaural coherence (Borss & Martin 2009). The
// noise never changes, so envelope updates never re-randomise the tail.
// Spectral shaping: the transport delivers eight octave-band energies per partition;
// the late kernel of a partition is the noise spectrum times a smooth amplitude
// envelope A(f) = sum_b sqrt(E_b / P) w_b(f), with w_b the Catmull-Rom (cubic Hermite)
// interpolation weights in log2 frequency through the band centres (EnvelopeWeight;
// constant below 125 Hz and above 16 kHz, negative overshoot clamped to zero). The
// earlier synthesis summed eight octave-filtered noises, i.e. interpolated the amplitude
// linearly between centres, which put a kink at every octave; the cubic envelope is
// smooth across frequency with the same band values, and the basis is eight times smaller.
internal sealed class LateBasis
{
    internal const int Bands=TakeRender.Settings.LateBands,Directions=TakeRender.Settings.LateDirections;
    internal readonly int Partition,Partitions,Channels,Bins;
    internal readonly Vector2[] Spectra;   // [channel][partition][bin], bins = Partition + 1
    internal readonly float[] EnvelopeWeights; // [bin][band]: EnvelopeWeight at each partition-FFT bin
    internal readonly float[] EarWeights;  // [direction][channel][band]: band energy of each head-frame direction
    internal long NumericBytes => Spectra.LongLength*8+EarWeights.Length*4L;
    static readonly Dictionary<(int,int,int,int),LateBasis> cache=new Dictionary<(int,int,int,int),LateBasis>();
    static readonly Vector3[] axes={Vector3.right,Vector3.left,Vector3.up,Vector3.down,Vector3.forward,Vector3.back};

    internal static LateBasis Get(int rate,int partition,int bins,int channels)
    {
        lock(cache)
        {
            var key=(rate,partition,bins,channels);
            if(!cache.TryGetValue(key,out var basis))cache.Add(key,basis=new LateBasis(rate,partition,bins,channels));
            return basis;
        }
    }
    internal static float BandCentre(int band) => 125f*(1<<band);
    // Catmull-Rom weight of band b's amplitude at frequency hz (weights sum to one).
    internal static float EnvelopeWeight(int band,double hz)
    {
        double x=hz<=125?0:Math.Min(Bands-1,Math.Log(hz/125,2));
        int i=Math.Min(Bands-2,(int)Math.Floor(x));double t=x-i;
        double t2=t*t,t3=t2*t;
        double w0=.5*(-t+2*t2-t3),w1=.5*(2-5*t2+3*t3),w2=.5*(t+4*t2-3*t3),w3=.5*(-t2+t3);
        double w=0;
        void Add(int index,double value){if(Math.Max(0,Math.Min(Bands-1,index))==band)w+=value;}
        Add(i-1,w0);Add(i,w1);Add(i+1,w2);Add(i+2,w3);
        return (float)w;
    }
    // Band responses sum to one at every frequency.
    internal static float BandWeight(int band,double hz)
    {
        double position=hz<=125?0:Math.Min(Bands-1,Math.Log(hz/125,2));
        double distance=Math.Abs(position-band);
        return distance>=1?0:(float)(1-distance);
    }

    LateBasis(int rate,int partition,int bins,int channels)
    {
        if(rate<=0 || partition<1 || (partition&(partition-1))!=0 || bins<1 || (channels!=1 && channels!=2))
            throw new ArgumentException("Invalid late synthesis shape.");
        Partition=partition;Channels=channels;Bins=partition+1;
        int length=Math.Max(2*partition,1);while(length<(long)bins*partition)length=checked(length*2);
        Partitions=length/partition;
        var hrtf=channels==2?Hrtf.Load(rate):null;
        EarWeights=Weights(hrtf,rate);
        var coherence=Coherence(hrtf,rate,length);
        Spectra=new Vector2[checked(channels*Partitions*Bins)];
        EnvelopeWeights=new float[Bins*Bands];
        for(int k=0;k<Bins;k++)for(int b=0;b<Bands;b++)EnvelopeWeights[k*Bands+b]=EnvelopeWeight(b,(double)k*rate/(2*partition));
        var noise=new double[2][];
        for(int n=0;n<(channels==2?2:1);n++)noise[n]=WhiteSpectrum(length,0x5e1f00d+(uint)n*0x9e3779b9u);
        var re=new double[length];var im=new double[length];var segment=new Vector2[2*partition];
        for(int c=0;c<channels;c++)
        {
            // Hermitian spectrum of this channel's noise, then one inverse transform.
            Array.Clear(re,0,length);Array.Clear(im,0,length);
            for(int k=0;k<=length/2;k++)
            {
                const float weight=1;
                double xr=noise[0][2*k],xi=noise[0][2*k+1];
                if(channels==2)
                {
                    double g=coherence[k],a=Math.Sqrt((1+g)/2),d=Math.Sqrt((1-g)/2)*(c==0?1:-1);
                    xr=a*xr+d*noise[1][2*k];xi=a*xi+d*noise[1][2*k+1];
                }
                re[k]=weight*xr;im[k]=weight*xi;
                if(k>0 && k<length/2){re[length-k]=re[k];im[length-k]=-im[k];}
            }
            Transform(re,im,true);
            for(int p=0;p<Partitions;p++)
            {
                Array.Clear(segment,0,segment.Length);
                for(int n=0;n<partition;n++)segment[n].x=(float)re[p*partition+n];
                ImpulseResponseAudio.Fft(segment,false);
                Array.Copy(segment,0,Spectra,(c*Partitions+p)*Bins,Bins);
            }
        }
    }
    // Complex Gaussian spectrum of unit-variance real white noise (counter hashes, Box-Muller).
    static double[] WhiteSpectrum(int length,uint seed)
    {
        var spectrum=new double[2*(length/2+1)];
        double Uniform(uint index,uint stream)=>(AudioSourcePlane.Hash(AudioSourcePlane.Hash(seed^stream)^index)+.5)/4294967296.0;
        for(int k=0;k<=length/2;k++)
        {
            double radius=Math.Sqrt(-2*Math.Log(Uniform((uint)k,1))),angle=2*Math.PI*Uniform((uint)k,2);
            bool real=k==0 || k==length/2;
            double scale=Math.Sqrt(real?length:length/2.0);
            spectrum[2*k]=scale*radius*Math.Cos(angle);spectrum[2*k+1]=real?0:scale*radius*Math.Sin(angle);
        }
        return spectrum;
    }
    // Per head-frame direction bin, the mean HRTF energy of each ear in each band,
    // weighted toward the bin's axis by cos^2 over its hemisphere. Mono is flat.
    static float[] Weights(Hrtf hrtf,int rate)
    {
        var weights=new float[Directions*2*Bands];
        if(hrtf==null){Array.Fill(weights,1);return weights;}
        int size=Size(hrtf),bins=size/2+1;var spectra=Hrtf.Spectra(hrtf,size);
        for(int d=0;d<Directions;d++)for(int ear=0;ear<2;ear++)for(int b=0;b<Bands;b++)
        {
            double sum=0,total=0;
            for(int i=0;i<hrtf.DirectionCount;i++)
            {
                double w=Math.Max(0,Vector3.Dot(hrtf.Direction(i),axes[d]));w*=w;if(w==0)continue;
                double energy=0,count=0;
                for(int k=0;k<bins;k++)
                {
                    float band=BandWeight(b,(double)k*rate/size);if(band==0)continue;
                    var h=spectra[(i*2+ear)*bins+k];energy+=band*(h.x*h.x+h.y*h.y);count+=band;
                }
                sum+=w*energy/Math.Max(1e-12,count);total+=w;
            }
            weights[(d*2+ear)*Bands+b]=(float)(sum/Math.Max(1e-12,total));
        }
        return weights;
    }
    // Diffuse-field interaural coherence of the HRTF set, interpolated to the noise grid.
    internal static double[] Coherence(Hrtf hrtf,int rate,int length)
    {
        var result=new double[length/2+1];
        if(hrtf==null)return result;
        int size=Size(hrtf),bins=size/2+1;var spectra=Hrtf.Spectra(hrtf,size);var native=new double[bins];
        for(int k=0;k<bins;k++)
        {
            double cross=0,left=0,right=0;
            for(int i=0;i<hrtf.DirectionCount;i++)
            {
                var l=spectra[(i*2)*bins+k];var r=spectra[(i*2+1)*bins+k];
                cross+=l.x*r.x+l.y*r.y;left+=l.x*l.x+l.y*l.y;right+=r.x*r.x+r.y*r.y;
            }
            native[k]=left>0 && right>0?Math.Max(-1,Math.Min(1,cross/Math.Sqrt(left*right))):1;
        }
        for(int k=0;k<result.Length;k++)
        {
            double at=(double)k*size/length;int lower=Math.Min(bins-1,(int)at),upper=Math.Min(bins-1,lower+1);
            result[k]=native[lower]+(native[upper]-native[lower])*(at-lower);
        }
        return result;
    }
    static int Size(Hrtf hrtf){int size=512;while(size<hrtf.Taps)size*=2;return size;}
    // Double-precision radix-2 transform for the long noise; inverse divides by length.
    static void Transform(double[] re,double[] im,bool inverse)
    {
        int n=re.Length;
        for(int i=1,j=0;i<n;i++)
        {
            int bit=n>>1;for(;(j&bit)!=0;bit>>=1)j^=bit;j^=bit;
            if(i<j){(re[i],re[j])=(re[j],re[i]);(im[i],im[j])=(im[j],im[i]);}
        }
        for(int size=2;size<=n;size*=2)
        {
            double step=(inverse?2:-2)*Math.PI/size;int half=size/2;
            var cos=new double[half];var sin=new double[half];
            for(int k=0;k<half;k++){cos[k]=Math.Cos(step*k);sin[k]=Math.Sin(step*k);}
            for(int first=0;first<n;first+=size)for(int k=0;k<half;k++)
            {
                int a=first+k,b=a+half;
                double xr=re[b]*cos[k]-im[b]*sin[k],xi=re[b]*sin[k]+im[b]*cos[k];
                re[b]=re[a]-xr;im[b]=im[a]-xi;re[a]+=xr;im[a]+=xi;
            }
        }
        if(inverse)for(int i=0;i<n;i++){re[i]/=n;im[i]/=n;}
    }
}
