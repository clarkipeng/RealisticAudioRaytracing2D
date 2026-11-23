using System;
using System.Collections.Generic;
using UnityEngine;

// Source realization uses integer audio time and counter random numbers. Transport
// quality, listeners and submission boundaries never enter this key.
internal static class Emission
{
    internal const float SurfaceOffset=.001f; // Air side, above the transport ray epsilon; part of generator v3.
    internal const uint Burst = 0x80000000u;
    internal static bool HasBursts(AcousticTake.Scene scene) => scene.planes != null && Array.Exists(scene.planes,p=>p.generatorVersion!=0);
    internal static uint SourceKey(string id, int seed)
    {
        uint key=2166136261u;
        unchecked { foreach(char c in id) { key=(key^(byte)c)*16777619u; key=(key^(byte)(c>>8))*16777619u; } }
        return AudioSourcePlane.Hash(key^(uint)seed);
    }
    internal static uint Random(uint key,long epoch,long index,uint property)
    {
        uint h=AudioSourcePlane.Hash(key^property);
        h=AudioSourcePlane.Hash(h^(uint)epoch);h=AudioSourcePlane.Hash(h^(uint)(epoch>>32));
        h=AudioSourcePlane.Hash(h^(uint)index);return AudioSourcePlane.Hash(h^(uint)(index>>32));
    }
    internal static int Integer(uint value,int count) => (int)(((ulong)value*(uint)count)>>32);
    internal static long Birth(uint key,long epoch,long index,int period) => checked(epoch+index*period+Integer(Random(key,epoch,index,0),period));
    internal static float Window(long age,int duration) => age<=0 || age>=duration-1 ? 0 : (float)(.5-.5*Math.Cos(2*Math.PI*age/(duration-1)));
    internal static void Configure(AcousticTake.Scene scene,TakeRender.Signal[] signals)
    {
        if(scene.planes!=null)foreach(var p in scene.planes)
        {
            signals[p.source].BurstDuration=p.burstDurationSamples;
            signals[p.source].OneShot=p.generatorVersion>=2;
            if(p.generatorVersion>=2 && signals[p.source].Clip.Length!=p.burstDurationSamples)
                throw new InvalidOperationException("One-shot duration must match the complete decoded clip.");
        }
    }

    // Cell mass is density times physical area. A CDF avoids rejection loops on
    // sparse masks and preserves black cells exactly, including an all-black map.
    internal sealed class PlaneSampler
    {
        readonly AcousticTake.EmissionPlane plane;
        readonly ulong[] cdf;
        readonly double sum;
        internal readonly double Mass;
        internal PlaneSampler(AcousticTake.EmissionPlane plane)
        {
            AudioSourcePlane.Validate(plane);this.plane=plane;cdf=new ulong[plane.mask.Length];
            int remaining=0;foreach(float density in plane.mask){sum+=density;if(density>0)remaining++;}
            Mass=sum*Vector3.Cross(plane.u,plane.v).magnitude/cdf.Length;
            double cumulative=0;ulong previous=0;
            for(int i=0;i<cdf.Length;i++)
            {
                cumulative+=plane.mask[i];
                if(plane.mask[i]>0)
                {
                    remaining--;
                    // Reserve at least one integer outcome for every positive cell.
                    // The returned PDF includes this quantization, not the ideal CDF.
                    ulong ideal=(ulong)Math.Round(cumulative/sum*4294967296.0,MidpointRounding.ToEven);
                    previous=Math.Min(4294967296UL-(uint)remaining,Math.Max(previous+1,ideal));
                }
                cdf[i]=previous;
            }
        }
        internal Vector4 Sample(uint cell,uint x,uint y,out double pdf)
        {
            if(sum==0){pdf=0;return Vector4.zero;}
            int lo=0,hi=cdf.Length-1;
            while(lo<hi){int mid=lo+(hi-lo)/2;if(cdf[mid]<=cell)lo=mid+1;else hi=mid;}
            // 24-bit subcell coordinates keep conversion to float strictly below one.
            float u=((lo%plane.width)+(x>>8)/16777216f)/plane.width;
            float v=((lo/plane.width)+(y>>8)/16777216f)/plane.height;
            pdf=(cdf[lo]-(lo==0?0:cdf[lo-1]))/4294967296.0*plane.mask.Length/Vector3.Cross(plane.u,plane.v).magnitude;
            Vector3 position=plane.u*(u-.5f)+plane.v*(v-.5f);
            return new Vector4(position.x,position.y,position.z,plane.mask[lo]);
        }
    }
}
