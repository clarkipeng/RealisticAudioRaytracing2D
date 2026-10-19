using System;
using System.Collections.Generic;
using UnityEngine;

// Path-wise spectral rendering. Propagation is a continuous source-time read,
// outside the short frequency/ear filter. Overlap-save discards circular wrap;
// no scene IR or propagation-length filter partitions are constructed here.
internal sealed class SpectralAudio
{
    sealed class PathFilter
    {
        public readonly Vector2[] spectrum;
        public int pair;
        public uint identity,bin;
        public PathFilter other;
        public Vector3 direction;
        public double delay,rate;
        public long sample;
        public PathFilter(int count){spectrum=new Vector2[count];}
    }
    sealed class Response
    {
        public readonly List<PathFilter> paths=new List<PathFilter>();
        public int count;
        public readonly float[] block;
        public long sample;
        public readonly Dictionary<(int,uint,uint),PathFilter> index=new Dictionary<(int,uint,uint),PathFilter>();
        public Response(int count){block=new float[count];}
    }
    readonly int size,hop,fft,bins,rate,channels,listeners,pairs,slots,latencyOffset,radius;
    readonly TakeRender.DelayFilter delayFilter;
    readonly double maximumDelay;
    readonly TakeRender.Signal[] signals;
    readonly Hrtf hrtf;
    readonly Vector2[] native,pressure,ears,input,sum;
    readonly int[] heads,next;
    readonly Dictionary<(float,Vector4,uint),int> groups;
    Response current,previous;
    long cursor;
    bool cut;
    long headSample;
    readonly Quaternion[] headRotations;
    readonly Vector3[] headAngular;
    bool initialized,hasPrevious;
    public int ResponseSupportSamples {get;private set;}
    public long NumericBytes => (native.LongLength+pressure.Length+ears.Length+input.Length+sum.Length)*8+
        (heads.LongLength+next.Length+current.block.Length+previous.block.Length)*4+listeners*28L+
        (current.paths.Count+previous.paths.Count)*((long)bins*8+60);
    public SpectralAudio(AcousticTake.Scene scene,int rate,TakeRender.Settings settings,TakeRender.Signal[] signals)
    {
        this.rate=rate;this.signals=signals;maximumDelay=settings.maximumDelaySeconds*(double)rate;size=settings.fftSize;hop=settings.spectralHop;channels=settings.ChannelsPerListener;
        listeners=scene.listeners.Length;pairs=signals.Length*listeners;slots=settings.SpectralSlotsPerPair;
        hrtf=channels==2?Hrtf.Load(rate):null;delayFilter=settings.delayFilter;radius=settings.sincRadius;
        latencyOffset=size/2+(delayFilter==TakeRender.DelayFilter.WindowedSinc?radius:0);
        fft=1;while(fft<size+(hrtf?.Taps??1)+hop-2)fft=checked(fft*2);bins=fft/2+1;
        native=new Vector2[size];pressure=new Vector2[fft];ears=Hrtf.Spectra(hrtf,fft);input=new Vector2[fft];sum=new Vector2[checked(2*listeners*channels*bins)];
        heads=new int[slots];next=new int[slots];groups=new Dictionary<(float,Vector4,uint),int>(slots);
        headRotations=new Quaternion[listeners];headAngular=new Vector3[listeners];
        current=new Response(listeners*channels*hop);previous=new Response(current.block.Length);
    }
    public void SetResponse(long sample,bool cut,RayTransport.Path[] paths,int offset,AcousticTake.Scene scene,AcousticTake.State[] states,long? anchor=null,AcousticTake.State[] headStates=null)
    {
        if(paths==null || offset<0 || paths.Length-offset<checked(pairs*slots))throw new ArgumentException("Missing directional ray response.");
        foreach(var state in states)cut|=(state.flags&AcousticTake.StateFlags.Cut)!=0;
        hasPrevious=initialized;this.cut=cut;headSample=sample;headStates??=states;
        (current,previous)=(previous,current);current.count=0;current.index.Clear();
        current.sample=anchor??sample;
        for(int i=0;i<previous.count;i++)previous.paths[i].other=null;
        for(int l=0;l<listeners;l++)
        {var state=headStates[scene.listeners[l]];headRotations[l]=state.rotation;headAngular[l]=hrtf==null?Vector3.zero:Hrtf.HeadVelocity(state);}
        for(int pair=0;pair<pairs;pair++)
        {
            int listener=pair%listeners;if(scene.sources[pair/listeners].mutedListenerPlusOne==listener+1 || !TakeRender.Enabled(states[scene.listeners[listener]],AcousticTake.StateFlags.Listener))continue;
            int basePath=offset+pair*slots,count=0;groups.Clear();
            for(int slot=0;slot<slots;slot++)
            {
                var path=paths[basePath+slot];if(path.pressureGain==0 && path.pressureImag==0)continue;
                if(path.frequencyBin>size/2 || !float.IsFinite(path.pressureGain) || !float.IsFinite(path.pressureImag) || !float.IsFinite(path.delaySeconds) || path.delaySeconds<0 ||
                    !float.IsFinite(path.direction.sqrMagnitude) || path.direction.w>=1)throw new InvalidOperationException("Invalid or noncausal spectral arrival at pair "+pair+", slot "+slot+": delay="+path.delaySeconds+", rate="+path.direction.w);
                var key=(path.delaySeconds,path.direction,path.identity);
                if(!groups.TryGetValue(key,out int group)){group=count++;groups.Add(key,group);heads[group]=-1;}
                next[slot]=heads[group];heads[group]=slot;
            }
            for(int group=0;group<count;group++)
            {
                if(current.count==current.paths.Count)current.paths.Add(new PathFilter(bins));
                var filter=current.paths[current.count++];int slot=heads[group];var path=paths[basePath+slot];
                filter.pair=pair;filter.direction=path.direction;filter.rate=path.direction.w;filter.delay=path.delaySeconds*(double)rate;filter.sample=current.sample;filter.identity=path.identity;filter.bin=path.frequencyBin;filter.other=null;
                Array.Clear(native,0,size);
                for(int member=slot;member>=0;member=next[member])
                {var value=paths[basePath+member];native[value.frequencyBin]+=new Vector2(value.pressureGain,value.pressureImag);filter.bin=Math.Min(filter.bin,value.frequencyBin);}
                var key=(pair,filter.identity,filter.bin);
                if(!current.index.TryAdd(key,filter))current.index[key]=null;
                // The solve supplies the retarded-time anchor. Carrying an
                // approximate slope between solves can accumulate nonphysical drift.
                if(filter.delay<0 || filter.delay>maximumDelay)throw new InvalidOperationException("Propagation clock is outside the declared delay horizon; retrace the path or increase Maximum Delay Seconds.");
                native[0].y=native[size/2].y=0;
                for(int k=1;k<size/2;k++)native[size-k]=new Vector2(native[k].x,-native[k].y);ImpulseResponseAudio.Fft(native,true);
                Array.Clear(pressure,0,fft);
                for(int n=0;n<size;n++)pressure[n].x=native[(n+size/2)%size].x;
                ImpulseResponseAudio.Fft(pressure,false);
                Array.Copy(pressure,filter.spectrum,bins);
                ResponseSupportSamples=Math.Max(ResponseSupportSamples,checked((int)Math.Ceiling(filter.delay)+latencyOffset+size+(hrtf?.Taps??1)+radius));
            }
        }
        for(int i=0;i<current.count;i++)
        {
            var filter=current.paths[i];var key=(filter.pair,filter.identity,filter.bin);
            if(!cut && current.index[key]==filter && previous.index.TryGetValue(key,out var old) && old!=null &&
                Doppler.Continuous(current.sample-previous.sample,old.delay,filter.delay,old.rate,filter.rate))
            {filter.other=old;old.other=filter;}
        }
        initialized=true;
    }
    public void Render(long sample,int count,float[] pcm)
    {
        if(sample!=cursor)throw new InvalidOperationException("Spectral output must be consumed in order from initialization.");
        for(int n=0;n<count;n++,cursor++)
        {
            if(cursor%hop==0)
            {
                Synthesize(current,cursor);
                if(previous.count>0)Synthesize(previous,cursor);
            }
            int at=(int)(cursor%hop);float blend=Blend(cursor);
            for(int channel=0;channel<listeners*channels;channel++)
            {
                int index=channel*hop+at;float value=current.block[index]*blend+(blend<1?previous.block[index]*(1-blend):0);
                if(!float.IsFinite(value))throw new InvalidOperationException("Non-finite spectral audio.");
                pcm[n*listeners*channels+channel]=value;
            }
        }
    }
    float Blend(long sample)
    {
        if(!hasPrevious || current.sample<=previous.sample)return 1;
        double time=sample-latencyOffset-size/2;
        return cut?(time<current.sample?0:1):Mathf.Clamp01((float)((time-previous.sample)/(current.sample-previous.sample)));
    }
    void Synthesize(Response response,long sample)
    {
        Array.Clear(sum,0,sum.Length);
        for(int p=0;p<response.count;p++)
        {
            var filter=response.paths[p];int source=filter.pair/listeners,listener=filter.pair%listeners;
            var left=response==current?filter.other:filter;var right=response==current?filter:filter.other;
            for(int n=0;n<fft;n++)
            {
                double receiver=sample-(fft-hop)+n-latencyOffset;
                double delay=filter.delay+filter.rate*(receiver-filter.sample),slope=filter.rate;
                if(left!=null && right!=null)Doppler.Clock(receiver-previous.sample,current.sample-previous.sample,left.delay,right.delay,left.rate,right.rate,out delay,out slope);
                input[n]=new Vector2(Doppler.Read(signals[source],receiver-delay,1-slope,delayFilter,radius),0);
            }
            ImpulseResponseAudio.Fft(input,false);
            var angular=headAngular[listener];int endpoints=angular.sqrMagnitude>0?2:1;
            for(int end=0;end<endpoints;end++)
            {
                // World angular velocity advances the output-time pose; source
                // delay and FFT latency are already accounted for in input.
                var rotation=headRotations[listener];
                if(endpoints==2)rotation=Quaternion.AngleAxis(angular.magnitude*(sample-headSample+end*hop)/rate*Mathf.Rad2Deg,angular)*rotation;
                var blend=hrtf==null?default:hrtf.Interpolate(Quaternion.Inverse(rotation)*filter.direction);
                for(int channel=0;channel<channels;channel++)for(int k=0;k<bins;k++)
                {
                    var ear=hrtf==null?Vector2.right:
                        ears[(blend.a*channels+channel)*bins+k]*blend.weights.x+ears[(blend.b*channels+channel)*bins+k]*blend.weights.y+
                        ears[(blend.c*channels+channel)*bins+k]*blend.weights.z+ears[(blend.d*channels+channel)*bins+k]*blend.weights.w;
                    var value=Multiply(input[k],Multiply(filter.spectrum[k],ear));
                    sum[((end*listeners+listener)*channels+channel)*bins+k]+=value;
                }
            }
        }
        Array.Clear(response.block,0,response.block.Length);
        for(int channel=0;channel<listeners*channels;channel++)
        {
            int endpoints=headAngular[channel/channels].sqrMagnitude>0?2:1;
            for(int end=0;end<endpoints;end++)
            {
                Array.Copy(sum,(end*listeners*channels+channel)*bins,input,0,bins);input[0].y=input[fft/2].y=0;
                for(int k=1;k<fft/2;k++)input[fft-k]=new Vector2(input[k].x,-input[k].y);
                ImpulseResponseAudio.Fft(input,true);
                for(int n=0;n<hop;n++)response.block[channel*hop+n]+=input[fft-hop+n].x*(endpoints==1?1:end==0?1-n/(float)hop:n/(float)hop);
            }
        }
    }
    static Vector2 Multiply(Vector2 a,Vector2 b)=>new Vector2(a.x*b.x-a.y*b.y,a.x*b.y+a.y*b.x);
}
