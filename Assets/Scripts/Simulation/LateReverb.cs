using System;
using UnityEngine;

// CPU reference of the late-field synthesis; GpuAudio.Late mirrors it.
// Each pair's late impulse response is LateBasis noise whose partition-k band
// gains are sqrt(E / P), E being the ear's histogram energy of that band and
// partition (direction bins weighted by the HRTF band energy). It convolves the
// source's emitted signal (a point source's signal, or a plane's emission sites
// summed with their amplitudes) by uniform partitioned overlap-save, delayed by
// the declared DSP latency so a late arrival keeps the delay origin of the early
// paths. A new response crossfades from the previous one over one response step.
internal sealed class LateReverb
{
    readonly int partition,bins,count,channels,listeners,pairs,latency,step,rate;
    readonly AcousticTake.Scene scene;
    readonly TakeRender.Signal[] signals;
    // Allocated with the first late field, so renders without one never build the noise basis.
    LateBasis basis;
    int[] offsets;                     // per pair: first noise partition (decorrelates pairs)
    Vector2[] kernels;                 // [bank][pair][channel][partition][bin]
    Vector2[] inputs;                  // [source][ring partition][bin]
    readonly bool[] active;            // [bank][pair]
    readonly float[] blocks;           // [bank][listener*channels][partition]
    readonly Vector2[] work,sum;
    readonly float[] dry,gains;
    // Emission rows of plane sources at the latest response: amplitude, clip phase, birth.
    readonly float[][] amplitudes;readonly uint[][] phases;readonly long[][] births;readonly int[] sites;
    int bank,computedBlock=-1;
    long cursor,swap;
    bool initialized,blend;
    internal long NumericBytes => ((kernels?.LongLength??0)+(inputs?.LongLength??0)+work.Length+sum.Length)*8+(blocks.Length+dry.Length+gains.Length)*4L+(basis?.NumericBytes??0);

    internal LateReverb(AcousticTake.Scene scene,int rate,TakeRender.Settings settings,TakeRender.Signal[] signals)
    {
        this.scene=scene;this.signals=signals;partition=settings.LatePartition;count=settings.LateBins(rate);bins=partition+1;
        channels=settings.ChannelsPerListener;listeners=scene.listeners.Length;pairs=signals.Length*listeners;
        latency=settings.DspLatencySamples;step=settings.responseStepSamples;this.rate=rate;
        active=new bool[2*pairs];blocks=new float[2*listeners*channels*partition];
        work=new Vector2[2*partition];sum=new Vector2[bins];dry=new float[2*partition];gains=new float[LateBasis.Bands];
        amplitudes=new float[signals.Length][];phases=new uint[signals.Length][];births=new long[signals.Length][];sites=new int[signals.Length];
    }
    internal void SetResponse(long sample,bool cut,RayTransport.LateFrame late,AcousticTake.State[] states)
    {
        if(initialized && sample<swap)throw new ArgumentException("Late responses must be ordered.");
        if(late.Valid && sample%partition!=0)throw new ArgumentException("Late responses change on partition boundaries.");
        blend=initialized && !cut;bank^=1;swap=sample;initialized=true;computedBlock=-1;
        if(late.Valid && basis==null)
        {
            basis=LateBasis.Get(rate,partition,Math.Max(1,count),channels);
            offsets=new int[pairs];for(int p=0;p<pairs;p++)offsets[p]=(int)(AudioSourcePlane.Hash((uint)p*0x9e3779b9u+0x7f4a7c15u)%(uint)basis.Partitions);
            kernels=new Vector2[checked(2*pairs*channels*count*bins)];inputs=new Vector2[checked(signals.Length*count*bins)];
        }
        int at=bank*pairs*channels*count*bins;
        if(kernels!=null)Array.Clear(kernels,at,pairs*channels*count*bins);
        for(int pair=0;pair<pairs;pair++)
        {
            int listener=pair%listeners;
            active[bank*pairs+pair]=late.Valid && scene.sources[pair/listeners].mutedListenerPlusOne!=listener+1 &&
                TakeRender.Enabled(states[scene.listeners[listener]],AcousticTake.StateFlags.Listener);
            if(!active[bank*pairs+pair])continue;
            for(int c=0;c<channels;c++)for(int k=0;k<count;k++)
            {
                bool any=false;
                for(int b=0;b<LateBasis.Bands;b++)
                {
                    double energy=0;
                    for(int d=0;d<LateBasis.Directions;d++)energy+=basis.EarWeights[(d*2+c)*LateBasis.Bands+b]*late.Energy(pair,d,b,k);
                    gains[b]=(float)Math.Sqrt(Math.Max(0,energy)/partition);any|=gains[b]>0;
                }
                if(!any)continue;
                int kernel=at+((pair*channels+c)*count+k)*bins,noise=(offsets[pair]+k)%basis.Partitions;
                for(int b=0;b<LateBasis.Bands;b++)
                {
                    if(gains[b]==0)continue;int source=((c*LateBasis.Bands+b)*basis.Partitions+noise)*bins;
                    for(int i=0;i<bins;i++)kernels[kernel+i]+=basis.Spectra[source+i]*gains[b];
                }
            }
        }
        if(late.Valid)for(int s=0;s<signals.Length;s++)
        {
            // Plane emission rows: legacy fixed sites (clip phase from the plane seed) or retained bursts.
            var batch=late.Transport.Emissions;var range=batch.Ranges[late.Frame*batch.FrameStride+s];
            sites[s]=range.mode==0?0:range.count;if(sites[s]==0)continue;
            if(amplitudes[s]==null || amplitudes[s].Length<sites[s]){amplitudes[s]=new float[sites[s]];phases[s]=new uint[sites[s]];births[s]=new long[sites[s]];}
            for(int n=0;n<sites[s];n++)
            {
                var point=batch.Points[range.first+n];var clock=batch.Clocks[range.first+n];amplitudes[s][n]=point.w;
                phases[s][n]=range.mode==2?clock.phase:AudioSourcePlane.Hash((uint)range.seed^((uint)n*0x85ebca6bu+0x9e3779b9u))>>16;
                births[s][n]=range.mode==2?(long)(((ulong)clock.high<<32)|clock.low):0;
            }
        }
    }
    float Dry(int source,long sample)
    {
        if(sample<0)return 0;
        if(sites[source]==0)return signals[source].Read(sample);
        float value=0;int duration=signals[source].BurstDuration;
        for(int n=0;n<sites[source];n++)
        {
            long birth=births[source][n];
            if(duration>0 && (sample<birth || sample>=birth+duration))continue;
            value+=amplitudes[source][n]*signals[source].Read(sample,phases[source][n],birth);
        }
        return value;
    }
    // Adds the late field to interleaved listener/channel PCM.
    internal void Render(long sample,int length,float[] pcm)
    {
        if(!initialized || sample!=cursor)throw new InvalidOperationException("Late output must follow its first response in order.");
        for(int n=0;n<length;n++,cursor++)
        {
            int block=(int)(cursor/partition),at=(int)(cursor%partition);
            if(block!=computedBlock)Compute(block);
            float weight=blend?Mathf.Clamp01((float)(cursor-swap)/step):1;
            for(int channel=0;channel<listeners*channels;channel++)
            {
                float value=blocks[(bank*listeners*channels+channel)*partition+at]*weight;
                if(weight<1)value+=blocks[((bank^1)*listeners*channels+channel)*partition+at]*(1-weight);
                pcm[n*listeners*channels+channel]+=value;
            }
        }
    }
    void Compute(int block)
    {
        computedBlock=block;
        if(kernels==null){Array.Clear(blocks,0,blocks.Length);return;}long start=(long)block*partition;int ring=block%Math.Max(1,count);
        for(int s=0;s<signals.Length;s++)
        {
            if(count==0)continue;
            bool used=false;for(int l=0;l<listeners;l++)used|=active[bank*pairs+s*listeners+l]||active[(bank^1)*pairs+s*listeners+l];
            // An unheard source contributes silence to every later block that reads this slot.
            if(!used){Array.Clear(inputs,(s*count+ring)*bins,bins);continue;}
            for(int n=0;n<2*partition;n++)work[n]=new Vector2(Dry(s,start-partition-latency+n),0);
            ImpulseResponseAudio.Fft(work,false);Array.Copy(work,0,inputs,(s*count+ring)*bins,bins);
        }
        bool previous=blend && start<swap+step;
        for(int which=0;which<(previous?2:1);which++)
        {
            int b=bank^which;
            for(int l=0;l<listeners;l++)for(int c=0;c<channels;c++)
            {
                Array.Clear(sum,0,bins);
                for(int s=0;s<signals.Length;s++)
                {
                    int pair=s*listeners+l;if(!active[b*pairs+pair])continue;
                    for(int k=0;k<count && k<=block;k++)
                    {
                        int input=(s*count+(block-k)%count)*bins,kernel=(((b*pairs+pair)*channels+c)*count+k)*bins;
                        for(int i=0;i<bins;i++){var x=inputs[input+i];var h=kernels[kernel+i];sum[i]+=new Vector2(x.x*h.x-x.y*h.y,x.x*h.y+x.y*h.x);}
                    }
                }
                for(int i=0;i<bins;i++)work[i]=sum[i];work[0].y=work[partition].y=0;
                for(int i=1;i<partition;i++)work[2*partition-i]=new Vector2(sum[i].x,-sum[i].y);
                ImpulseResponseAudio.Fft(work,true);
                for(int n=0;n<partition;n++)blocks[(b*listeners*channels+l*channels+c)*partition+n]=work[partition+n].x;
            }
        }
    }
}
