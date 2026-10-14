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
        public readonly Quaternion[] heads;
        public readonly Vector3[] angular;
        public long sample;
        public Response(int count,int listeners){block=new float[count];heads=new Quaternion[listeners];angular=new Vector3[listeners];}
    }
    readonly int size,hop,fft,bins,rate,channels,listeners,pairs,slots,transition,latencyOffset,radius;
    readonly TakeRender.DelayFilter delayFilter;
    readonly double maximumDelay;
    readonly TakeRender.Signal[] signals;
    readonly Hrtf hrtf;
    readonly Vector2[] native,pressure,ears,input,sum;
    readonly int[] heads,next;
    readonly RayTransport.Path[] lastPaths;
    readonly bool[] lastEnabled;
    readonly Dictionary<(float,Vector4,uint),int> groups;
    readonly ImpulseResponseAudio ir;
    Response current,previous;
    long cursor,responseSample;
    int fade;
    bool initialized;
    public int ResponseSupportSamples {get;private set;}
    public long NumericBytes => ir?.NumericBytes ?? (native.LongLength+pressure.Length+ears.Length+input.Length+sum.Length)*8+
        (heads.LongLength+next.Length+current.block.Length+previous.block.Length)*4+lastPaths.LongLength*32+listeners*56L+lastEnabled.Length+
        (current.paths.Count+previous.paths.Count)*((long)bins*8+12);
    public SpectralAudio(AcousticTake.Scene scene,int rate,TakeRender.Settings settings,TakeRender.Signal[] signals)
    {
        if(settings.synthesis==TakeRender.Synthesis.ImpulseResponse){ir=new ImpulseResponseAudio(scene,rate,settings,signals);return;}
        this.rate=rate;this.signals=signals;maximumDelay=settings.maximumDelaySeconds*(double)rate;size=settings.fftSize;hop=settings.spectralHop;channels=settings.ChannelsPerListener;
        listeners=scene.listeners.Length;pairs=signals.Length*listeners;slots=settings.SpectralSlotsPerPair;transition=settings.transitionSamples;
        hrtf=channels==2?Hrtf.Load(rate):null;delayFilter=settings.delayFilter;radius=settings.sincRadius;
        latencyOffset=size/2+(delayFilter==TakeRender.DelayFilter.WindowedSinc?radius:0);
        fft=1;while(fft<size+(hrtf?.Taps??1)+hop-2)fft=checked(fft*2);bins=fft/2+1;
        native=new Vector2[size];pressure=new Vector2[fft];ears=Hrtf.Spectra(hrtf,fft);input=new Vector2[fft];sum=new Vector2[checked(2*listeners*channels*bins)];
        lastPaths=new RayTransport.Path[pairs*slots];lastEnabled=new bool[listeners];
        heads=new int[slots];next=new int[slots];groups=new Dictionary<(float,Vector4,uint),int>(slots);
        current=new Response(listeners*channels*hop,listeners);previous=new Response(current.block.Length,listeners);
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
            var angular=hrtf==null?Vector3.zero:Hrtf.HeadVelocity(state);
            unchanged&=current.heads[l].Equals(state.rotation) && current.angular[l].Equals(angular) && angular.sqrMagnitude==0 && lastEnabled[l]==enabled;
            lastEnabled[l]=enabled;
        }
        if(unchanged){fade=0;responseSample=sample;return;}
        Array.Copy(paths,offset,lastPaths,0,lastPaths.Length);
        (current,previous)=(previous,current);current.count=0;fade=initialized && !cut?transition:0;
        current.sample=sample;
        for(int l=0;l<listeners;l++)
        {var state=states[scene.listeners[l]];current.heads[l]=state.rotation;current.angular[l]=hrtf==null?Vector3.zero:Hrtf.HeadVelocity(state);}
        for(int pair=0;pair<pairs;pair++)
        {
            int listener=pair%listeners;if(scene.sources[pair/listeners].mutedListenerPlusOne==listener+1 || !TakeRender.Enabled(states[scene.listeners[listener]],AcousticTake.StateFlags.Listener))continue;
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
                if(current.count==current.paths.Count)current.paths.Add(new PathFilter(bins));
                var filter=current.paths[current.count++];int slot=heads[group];var path=paths[basePath+slot];
                filter.pair=pair;filter.direction=path.direction;filter.rate=path.direction.w;filter.delay=path.delaySeconds*(double)rate;filter.sample=sample;
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
                Array.Copy(pressure,filter.spectrum,bins);
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
            var angular=response.angular[listener];int endpoints=angular.sqrMagnitude>0?2:1;
            for(int end=0;end<endpoints;end++)
            {
                // World angular velocity advances the output-time pose; source
                // delay and FFT latency are already accounted for in input.
                var rotation=response.heads[listener];
                if(endpoints==2)rotation=Quaternion.AngleAxis(angular.magnitude*(sample-response.sample+end*hop)/rate*Mathf.Rad2Deg,angular)*rotation;
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
            int endpoints=response.angular[channel/channels].sqrMagnitude>0?2:1;
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
