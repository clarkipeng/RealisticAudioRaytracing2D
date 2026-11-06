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

    // Dense bounded storage, owned by one transport submission. Every path copies
    // birth/clip offset into its existing 48-byte record, so DSP banks own their
    // clocks even after this staging array is overwritten by the next batch.
    internal sealed class Batch
    {
        internal struct Range { public int first,count,seed,mode; }
        internal struct Clock { public uint phase,low,high,reserved; }
        internal readonly Range[] Ranges;
        internal readonly Vector4[] Points;
        internal readonly Clock[] Clocks;
        // Stable emission sites for the late energy field, independent of the burst
        // schedule: a point source's origin, a legacy plane's fixed sites, or fixed
        // draws from a burst plane's density (projected impacts use the first
        // prepared pose). w is the site's energy weight, mean 1 over its source.
        internal struct LateRange { public int first,count,world,reserved; }
        internal readonly Vector4[] LateSites;
        internal readonly LateRange[] LateRanges;
        internal const int LateSitesPerPlane=64;
        bool latePlaced;
        readonly AcousticTake.Scene scene;
        readonly TakeRender.Settings settings;
        readonly PlaneSampler[] samplers;
        readonly AcousticTake.EmissionPlane[] planes;
        readonly uint[] keys;
        readonly float[] weights;
        struct Placement { internal Vector3 position; internal bool valid; }
        readonly Dictionary<long,Placement>[] positions;
        readonly DirectBvh surfaces;
        readonly List<long> expired=new List<long>();
        readonly AcousticTake.State[] pose;
        readonly long history;
        internal readonly int PointsPerFrame,FrameStride;
        internal long NumericBytes => (long)Ranges.Length*16+(long)Points.Length*32+(long)pose.Length*80+keys.Length*8L+(surfaces?.NumericBytes??0);
        internal Batch(AcousticTake.Scene scene,TakeRender.Settings settings,int rate,int frames,AcousticTake.Primitive[] geometry=null)
        {
            this.scene=scene;this.settings=settings;FrameStride=HasBursts(scene)?scene.sources.Length:0;if(FrameStride==0)frames=1;history=settings.HistorySamples(rate)+2L*settings.responseStepSamples;
            planes=new AcousticTake.EmissionPlane[scene.sources.Length];samplers=new PlaneSampler[planes.Length];keys=new uint[planes.Length];weights=new float[planes.Length];Array.Fill(weights,float.NaN);
            positions=new Dictionary<long,Placement>[planes.Length];
            if(scene.planes!=null && Array.Exists(scene.planes,p=>p.generatorVersion==3))
            {
                if(scene.dimension!=AcousticTake.Dimension.Spatial3D || geometry==null)throw new InvalidOperationException("Projected impacts require spatial numeric geometry.");
                surfaces=new DirectBvh(scene,geometry);
            }
            if(scene.planes!=null)foreach(var p in scene.planes)
            {
                planes[p.source]=p;
                if(p.generatorVersion!=0){samplers[p.source]=new PlaneSampler(p);positions[p.source]=new Dictionary<long,Placement>();keys[p.source]=SourceKey(scene.objects[scene.sources[p.source].objectIndex].id,p.seed);}
            }
            for(int s=0;s<planes.Length;s++)PointsPerFrame=checked(PointsPerFrame+(planes[s]?.samples??1));
            LateRanges=new LateRange[planes.Length];int lateCapacity=0;
            for(int s=0;s<planes.Length;s++)
            {
                var p=planes[s];int capacity=Math.Min(settings.lateRays,p==null?1:p.generatorVersion==0?p.samples:LateSitesPerPlane);
                LateRanges[s]=new LateRange{first=lateCapacity,world=p!=null&&p.generatorVersion==3?1:0};lateCapacity=checked(lateCapacity+capacity);
            }
            LateSites=new Vector4[lateCapacity];
            for(int s=0;s<planes.Length;s++)if(planes[s]==null || planes[s].generatorVersion!=3)PlaceLateSites(s,null);
            Points=new Vector4[checked(PointsPerFrame*frames)];Clocks=new Clock[Points.Length];Ranges=new Range[checked(planes.Length*frames)];pose=new AcousticTake.State[scene.objects.Length];
            for(int f=0;f<frames;f++)for(int s=0,at=f*PointsPerFrame;s<planes.Length;s++)
            {
                var p=planes[s];int count=p?.samples??1;
                Ranges[f*planes.Length+s]=new Range{first=at,count=p!=null&&p.generatorVersion!=0?0:count,seed=p?.seed??0,mode=p==null?0:p.generatorVersion==0?1:2};
                for(int n=0;n<count;n++)Points[at+n]=p==null?new Vector4(0,0,0,1):p.generatorVersion==0?AudioSourcePlane.Sample(p,n):default;
                at+=count;
            }
        }
        // Fills one source's late sites. Projected impacts land on the surface below
        // their plane point at the given pose; misses are not sites.
        bool PlaceLateSites(int s,AcousticTake.State[] state)
        {
            var p=planes[s];var range=LateRanges[s];int capacity=(s+1<LateRanges.Length?LateRanges[s+1].first:LateSites.Length)-range.first;
            var sites=new List<Vector4>();
            if(p==null)sites.Add(new Vector4(0,0,0,1));
            else if(p.generatorVersion==0)for(int n=0;n<capacity;n++){var site=AudioSourcePlane.Sample(p,n);if(site.w>0)sites.Add(new Vector4(site.x,site.y,site.z,site.w*site.w));}
            else if(samplers[s].Mass>0)for(int n=0;n<capacity;n++)
            {
                var local=samplers[s].Sample(Random(keys[s],-1,n,16),Random(keys[s],-1,n,17),Random(keys[s],-1,n,18),out double pdf);
                if(pdf<=0 || local.w<=0)continue;
                Vector3 position=local;
                if(p.generatorVersion==3)
                {
                    var pose=state[scene.sources[s].objectIndex];
                    if(!surfaces.Raycast(pose.position+pose.rotation*position,Vector3.down,state,out var hit))continue;
                    position=hit.position+hit.normal*SurfaceOffset;
                }
                sites.Add(new Vector4(position.x,position.y,position.z,(float)(local.w/pdf)));
            }
            double mean=0;foreach(var site in sites)mean+=site.w;mean/=Math.Max(1,sites.Count);
            bool changed=range.count!=sites.Count;
            for(int n=0;n<sites.Count;n++)
            {
                var site=sites[n];site.w=(float)(site.w/mean);
                changed|=!site.Equals(LateSites[range.first+n]);LateSites[range.first+n]=site;
            }
            range.count=sites.Count;LateRanges[s]=range;return changed;
        }
        // Returns whether the late sites changed (projected impacts follow the pose).
        internal bool Prepare(TakeRender.Signal[] signals,long first,int frames,Action<long,AcousticTake.State[]> evaluate)
        {
            Configure(scene,signals);
            // Projected late sites land once, at the first prepared pose; impacts keep their own births.
            bool lateChanged=false;
            for(int s=0;s<planes.Length;s++)if(planes[s]!=null && planes[s].generatorVersion==3 && !latePlaced)
            {evaluate(first,pose);surfaces.Update(pose);lateChanged|=PlaceLateSites(s,pose);}
            latePlaced=true;
            for(int s=0;s<planes.Length;s++)if(positions[s]!=null)
            {
                expired.Clear();foreach(long birth in positions[s].Keys)if(birth<first-history-planes[s].burstDurationSamples)expired.Add(birth);
                foreach(long birth in expired)positions[s].Remove(birth);
            }
            for(int f=0;f<frames;f++)for(int s=0;s<planes.Length;s++)
            {
                var p=planes[s];if(p==null || p.generatorVersion==0)continue;
                long output=checked(first+(long)f*settings.responseStepSamples),from=Math.Max(0,output-history),to=checked(output+2L*settings.responseStepSamples);
                int row=f*planes.Length+s;var range=Ranges[row];range.count=0;
                var sampler=samplers[s];var signal=signals[s];
                if(sampler.Mass>0)
                {
                    if(float.IsNaN(weights[s]))
                    {
                        double energy=0;foreach(float value in signal.Clip)energy+=(double)value*value;
                        double power=energy/signal.Clip.Length;
                        if(!double.IsFinite(power))throw new InvalidOperationException("Non-finite burst clip.");
                        double windowEnergy=3.0*(p.burstDurationSamples-1)/8;
                        weights[s]=power==0?0:(float)Math.Sqrt(p.burstPeriodSamples/(p.generatorVersion>=2?energy:power*windowEnergy));
                    }
                    float weight=weights[s];
                    if(!float.IsFinite(weight))throw new InvalidOperationException("Burst clip normalization exceeds finite pressure range.");
                    if(weight==0){Ranges[row]=range;continue;}
                    foreach(var epoch in signal.Epochs(from-p.burstDurationSamples,to))
                    {
                        long end=Math.Min(to,epoch.end),index=Math.Max(0,(Math.Max(0,from-p.burstDurationSamples-epoch.start))/p.burstPeriodSamples);
                        for(;index<=(end-1-epoch.start)/p.burstPeriodSamples && end>epoch.start;index++)
                        {
                            long birth=Birth(keys[s],epoch.start,index,p.burstPeriodSamples);
                            if(birth>=end || birth<=from-p.burstDurationSamples)continue;
                            uint key=keys[s];var local=sampler.Sample(Random(key,epoch.start,index,1),Random(key,epoch.start,index,2),Random(key,epoch.start,index,3),out double pdf);
                            if(!positions[s].TryGetValue(birth,out var placement))
                            {
                                evaluate(birth,pose);var state=pose[scene.sources[s].objectIndex];
                                placement=new Placement{position=state.position+state.rotation*(Vector3)local,valid=true};
                                if(p.generatorVersion==3)
                                {
                                    surfaces.Update(pose);
                                    placement.valid=surfaces.Raycast(placement.position,Vector3.down,pose,out var hit);
                                    if(placement.valid)placement.position=hit.position+hit.normal*SurfaceOffset;
                                }
                                positions[s].Add(birth,placement);
                            }
                            if(!placement.valid)continue; // A miss consumes rainfall, not an impact slot; never resample.
                            if(range.count>=p.samples || (settings.maxBounces>0 && range.count>=settings.SourceRaysPerBin(scene,s)))
                                throw new InvalidOperationException("Seeded burst coverage exceeds plane Samples or, with indirect transport enabled, Rays Per Bin. Increase the relevant bounds or reduce the delay horizon/event rate.");
                            var world=placement.position;
                            int at=range.first+range.count++;float amplitude=(float)(weight*Math.Sqrt(local.w/pdf));
                            if(!float.IsFinite(amplitude))throw new InvalidOperationException("Burst source weight exceeds finite pressure range.");
                            Points[at]=new Vector4(world.x,world.y,world.z,amplitude);
                            Clocks[at]=new Clock{phase=p.generatorVersion>=2?Burst:Burst|(uint)Integer(Random(key,epoch.start,index,4),signal.Clip.Length),low=(uint)birth,high=(uint)(birth>>32)};
                        }
                    }
                }
                Ranges[row]=range;
            }
            return lateChanged;
        }
    }
}
