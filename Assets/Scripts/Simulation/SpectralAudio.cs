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
        public double delay,rate;
        public long sample;
        public PathFilter(int count){spectrum=new Vector2[count];}
    }
    sealed class Response
    {
        public readonly List<PathFilter> paths=new List<PathFilter>();
        public int count;
        public readonly float[] block;
        public Response(int count){block=new float[count];}
    }
    readonly int size,hop,fft,bins,rate,channels,listeners,pairs,slots,transition,latencyOffset,radius;
    readonly TakeRender.DelayFilter delayFilter;
    readonly double maximumDelay;
    readonly TakeRender.Signal[] signals;
    readonly Hrtf hrtf;
    readonly Vector2[] native,pressure,ear,input,sum;
    readonly int[] heads,next;
    readonly RayTransport.Path[] lastPaths;
    readonly Quaternion[] lastHeads;
    readonly bool[] lastEnabled;
    readonly Dictionary<(float,Vector4,uint),int> groups;
    readonly ImpulseResponseAudio ir;
    Response current,previous;
    long cursor,responseSample;
    int fade;
    bool initialized;
    public int ResponseSupportSamples {get;private set;}
    public long NumericBytes => ir?.NumericBytes ?? (native.LongLength+pressure.Length+ear.Length+input.Length+sum.Length)*8+
        (heads.LongLength+next.Length+current.block.Length+previous.block.Length)*4+lastPaths.LongLength*32+lastHeads.LongLength*16+lastEnabled.Length+
        (current.paths.Count+previous.paths.Count)*(long)channels*bins*8;
    public SpectralAudio(AcousticTake.Scene scene,int rate,TakeRender.Settings settings,TakeRender.Signal[] signals)
    {
        if(settings.synthesis==TakeRender.Synthesis.ImpulseResponse){ir=new ImpulseResponseAudio(scene,rate,settings,signals);return;}
        this.rate=rate;this.signals=signals;maximumDelay=settings.maximumDelaySeconds*(double)rate;size=settings.fftSize;hop=settings.spectralHop;channels=settings.ChannelsPerListener;
        listeners=scene.listeners.Length;pairs=signals.Length*listeners;slots=settings.SpectralSlotsPerPair;transition=settings.transitionSamples;
        hrtf=channels==2?Hrtf.Load(rate):null;delayFilter=settings.delayFilter;radius=settings.sincRadius;
        latencyOffset=size/2+(delayFilter==TakeRender.DelayFilter.WindowedSinc?radius:0);
        fft=1;while(fft<size+(hrtf?.Taps??1)+hop-2)fft=checked(fft*2);bins=fft/2+1;
        native=new Vector2[size];pressure=new Vector2[fft];ear=new Vector2[fft];input=new Vector2[fft];sum=new Vector2[checked(listeners*channels*bins)];
        lastPaths=new RayTransport.Path[pairs*slots];lastHeads=new Quaternion[listeners];lastEnabled=new bool[listeners];
        heads=new int[slots];next=new int[slots];groups=new Dictionary<(float,Vector4,uint),int>(slots);
        current=new Response(listeners*channels*hop);previous=new Response(current.block.Length);
    }
    public void SetResponse(long sample,bool cut,RayTransport.Path[] paths,int offset,AcousticTake.Scene scene,AcousticTake.State[] states)
    {
        if(ir!=null){ir.SetResponse(sample,cut,paths,offset,scene,states);ResponseSupportSamples=ir.ResponseSupportSamples;return;}
        if(paths==null || offset<0 || paths.Length-offset<checked(pairs*slots))throw new ArgumentException("Missing directional ray response.");
        foreach(var state in states)cut|=(state.flags&AcousticTake.StateFlags.Cut)!=0;
        bool unchanged=initialized && !cut;
        for(int i=0;i<lastPaths.Length && unchanged;i++)
        {
            var a=lastPaths[i];var b=paths[offset+i];
            unchanged=a.delaySeconds==b.delaySeconds && a.pressureGain==b.pressureGain && a.frequencyBin==b.frequencyBin && a.identity==b.identity && a.direction.Equals(b.direction);
        }
        for(int l=0;l<listeners;l++)
        {
            var state=states[scene.listeners[l]];bool enabled=TakeRender.Enabled(state,AcousticTake.StateFlags.Listener);
            unchanged&=lastHeads[l].Equals(state.rotation) && lastEnabled[l]==enabled;
            lastHeads[l]=state.rotation;lastEnabled[l]=enabled;
        }
        if(unchanged){fade=0;responseSample=sample;return;}
        Array.Copy(paths,offset,lastPaths,0,lastPaths.Length);
        (current,previous)=(previous,current);current.count=0;fade=initialized && !cut?transition:0;
        for(int pair=0;pair<pairs;pair++)
        {
            int listener=pair%listeners;if(scene.sources[pair/listeners].mutedListenerPlusOne==listener+1 || !TakeRender.Enabled(states[scene.listeners[listener]],AcousticTake.StateFlags.Listener))continue;
            var rotation=Quaternion.Inverse(states[scene.listeners[listener]].rotation);
            int basePath=offset+pair*slots,count=0;groups.Clear();
            for(int slot=0;slot<slots;slot++)
            {
                var path=paths[basePath+slot];if(path.pressureGain==0)continue;
                if(path.frequencyBin>size/2 || !float.IsFinite(path.pressureGain) || !float.IsFinite(path.delaySeconds) || path.delaySeconds<0 ||
                    !float.IsFinite(path.direction.sqrMagnitude) || path.direction.w>=1)throw new InvalidOperationException("Invalid or noncausal spectral arrival at pair "+pair+", slot "+slot+": delay="+path.delaySeconds+", rate="+path.direction.w);
                var key=(path.delaySeconds,path.direction,path.identity);
                if(!groups.TryGetValue(key,out int group)){group=count++;groups.Add(key,group);heads[group]=-1;}
                next[slot]=heads[group];heads[group]=slot;
            }
            for(int group=0;group<count;group++)
            {
                if(current.count==current.paths.Count)current.paths.Add(new PathFilter(channels*bins));
                var filter=current.paths[current.count++];int slot=heads[group];var path=paths[basePath+slot];
                filter.pair=pair;filter.rate=path.direction.w;filter.delay=path.delaySeconds*(double)rate;filter.sample=sample;
                Array.Clear(native,0,size);
                for(int member=slot;member>=0;member=next[member])
                {var value=paths[basePath+member];native[value.frequencyBin].x+=value.pressureGain;}
                // The solve supplies the retarded-time anchor. Carrying an
                // approximate slope between solves can accumulate nonphysical drift.
                if(filter.delay<0 || filter.delay>maximumDelay)throw new InvalidOperationException("Propagation clock is outside the declared delay horizon; retrace the path or increase Maximum Delay Seconds.");
                for(int k=1;k<size/2;k++)native[size-k]=native[k];ImpulseResponseAudio.Fft(native,true);
                Array.Clear(pressure,0,fft);
                for(int n=0;n<size;n++)pressure[n].x=native[(n+size/2)%size].x;
                ImpulseResponseAudio.Fft(pressure,false);
                var blend=hrtf==null?default:hrtf.Interpolate(rotation*(Vector3)path.direction);
                for(int channel=0;channel<channels;channel++)
                {
                    Array.Clear(ear,0,fft);
                    for(int t=0;t<(hrtf?.Taps??1);t++)ear[t].x=hrtf==null?1:hrtf.Coefficient(blend,channel,t);
                    ImpulseResponseAudio.Fft(ear,false);
                    for(int k=0;k<bins;k++){var a=pressure[k];var b=ear[k];filter.spectrum[channel*bins+k]=new Vector2(a.x*b.x-a.y*b.y,a.x*b.y+a.y*b.x);}
                }
                ResponseSupportSamples=Math.Max(ResponseSupportSamples,checked((int)Math.Ceiling(filter.delay)+latencyOffset+size+(hrtf?.Taps??1)+radius));
            }
        }
        responseSample=sample;initialized=true;
    }
    public void Render(long sample,int count,float[] pcm)
    {
        if(ir!=null){ir.Render(sample,count,pcm);return;}
        if(sample!=cursor)throw new InvalidOperationException("Spectral output must be consumed in order from initialization.");
        for(int n=0;n<count;n++,cursor++)
        {
            if(cursor%hop==0)
            {
                Synthesize(current,cursor);
                if(fade>0 && cursor-responseSample<fade)Synthesize(previous,cursor);
            }
            int at=(int)(cursor%hop);float blend=fade==0?1:Mathf.Clamp01((float)(cursor-responseSample)/fade);
            for(int channel=0;channel<listeners*channels;channel++)
            {
                int index=channel*hop+at;float value=current.block[index]*blend+(blend<1?previous.block[index]*(1-blend):0);
                if(!float.IsFinite(value))throw new InvalidOperationException("Non-finite spectral audio.");
                pcm[n*listeners*channels+channel]=value;
            }
        }
    }
    void Synthesize(Response response,long sample)
    {
        Array.Clear(sum,0,sum.Length);
        for(int p=0;p<response.count;p++)
        {
            var filter=response.paths[p];int source=filter.pair/listeners,listener=filter.pair%listeners;
            double ratio=1-filter.rate;
            for(int n=0;n<fft;n++)
            {
                double receiver=sample-(fft-hop)+n-latencyOffset;
                double emission=receiver-filter.delay-filter.rate*(receiver-filter.sample);
                input[n]=new Vector2(Doppler.Read(signals[source],emission,ratio,delayFilter,radius),0);
            }
            ImpulseResponseAudio.Fft(input,false);
            for(int channel=0;channel<channels;channel++)for(int k=0;k<bins;k++)
            {
                var a=input[k];var b=filter.spectrum[channel*bins+k];int index=(listener*channels+channel)*bins+k;
                sum[index].x+=a.x*b.x-a.y*b.y;sum[index].y+=a.x*b.y+a.y*b.x;
            }
        }
        for(int channel=0;channel<listeners*channels;channel++)
        {
            Array.Copy(sum,channel*bins,input,0,bins);input[0].y=input[fft/2].y=0;
            for(int k=1;k<fft/2;k++)input[fft-k]=new Vector2(input[k].x,-input[k].y);
            ImpulseResponseAudio.Fft(input,true);
            for(int n=0;n<hop;n++)response.block[channel*hop+n]=input[fft-hop+n].x;
        }
    }
}
