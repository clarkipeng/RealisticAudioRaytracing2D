using System;
using System.Collections.Generic;
using UnityEngine;

// Band and ear tables of the path reverb (PathReverb, GpuAudio.Path): the octave bands' centres and responses,
// each head-frame direction's band energy at each ear (EarWeights), and the HRTF set's diffuse-field interaural
// coherence (Borss & Martin 2009), which sets the binaural noise's interaural mix.
internal static class LateBasis
{
    internal const int Bands=TakeRender.Settings.LateBands,Directions=TakeRender.Settings.LateDirections;
    static readonly Dictionary<(int,int),float[]> cache=new Dictionary<(int,int),float[]>();
    static readonly Vector3[] axes={Vector3.right,Vector3.left,Vector3.up,Vector3.down,Vector3.forward,Vector3.back};

    // [direction][channel][band]: band energy of each head-frame direction (mono: flat), cached per rate and channels.
    internal static float[] EarWeights(int rate,int channels)
    {
        if(rate<=0 || (channels!=1 && channels!=2))throw new ArgumentException("Invalid late synthesis shape.");
        lock(cache)
        {
            var key=(rate,channels);
            if(!cache.TryGetValue(key,out var weights))cache.Add(key,weights=Weights(channels==2?Hrtf.Load(rate):null,rate));
            return weights;
        }
    }
    internal static float BandCentre(int band) => 125f*(1<<band);
    // Band responses sum to one at every frequency.
    internal static float BandWeight(int band,double hz)
    {
        double position=hz<=125?0:Math.Min(Bands-1,Math.Log(hz/125,2));
        double distance=Math.Abs(position-band);
        return distance>=1?0:(float)(1-distance);
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
}
