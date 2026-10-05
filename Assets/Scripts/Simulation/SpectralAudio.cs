using System;
using UnityEngine;

// Causal, normalized WOLA synthesis for the geometric spectral preview. Each
// arrival retains its delay/frequency/direction until the listener's ear filters
// are applied. Source spectra are shared across listeners and retained in a ring.
internal sealed class SpectralAudio
{
    readonly int size,hop,bins,rate,channels,listeners,pairs,slots,historyFrames;
    readonly TakeRender.Signal[] signals;
    readonly Vector2[][] history;
    readonly long[] historyKeys;
    readonly Vector2[] work,output;
    readonly Vector2[][] earFilters;
    readonly float[] window,overlap;
    const int FrequencySupport=9; // Truncated Dirichlet interpolation for fractional Doppler bins.
    readonly Vector2[] transfers;
    readonly long[] delays;
    readonly int[] firstBins,counts;
    long responseSample;bool hasResponse;
    readonly bool[] enabled;
    readonly RayTransport.Path[] paths;
    long cursor;
    public long NumericBytes {get;}
    public SpectralAudio(AcousticTake.Scene scene,int rate,TakeRender.Settings settings,TakeRender.Signal[] signals)
    {
        this.rate=rate;this.signals=signals;size=settings.fftSize;hop=settings.spectralHop;bins=size/2+1;
        channels=settings.ChannelsPerListener;listeners=scene.listeners.Length;pairs=scene.sources.Length*listeners;
        slots=checked(bins+settings.raysPerPair*settings.maxBounces);
        paths=new RayTransport.Path[checked(pairs*slots)];transfers=new Vector2[checked(paths.Length*channels*FrequencySupport)];
        delays=new long[paths.Length];firstBins=new int[paths.Length];counts=new int[paths.Length];enabled=new bool[listeners];
        historyFrames=checked((int)Math.Ceiling(settings.maximumDelaySeconds*rate/hop)+2);
        historyKeys=new long[historyFrames];Array.Fill(historyKeys,long.MinValue);
        history=new Vector2[signals.Length][];for(int i=0;i<history.Length;i++)history[i]=new Vector2[checked(historyFrames*bins)];
        work=new Vector2[size];output=new Vector2[checked(listeners*channels*size)];window=new float[size];overlap=new float[output.Length];
        for(int i=0;i<size;i++)window[i]=Mathf.Sqrt((1-Mathf.Cos(2*Mathf.PI*i/size))*hop/size);
        var hrtf=channels==2?Hrtf.Load(rate):null;
        earFilters=new Vector2[hrtf==null?1:checked(hrtf.DirectionCount*2)][];
        for(int ear=0;ear<earFilters.Length;ear++)
        {
            Array.Clear(work,0,size);
            if(hrtf==null)work[0]=Vector2.right;
            else
            {
                if(hrtf.Taps>size)throw new ArgumentException("FFT must contain the complete HRTF filter.");
                for(int i=0;i<hrtf.Taps;i++)work[i].x=hrtf.Coefficient(ear/2,ear%2,i);
            }
            Fft(work,false);earFilters[ear]=new Vector2[bins];Array.Copy(work,earFilters[ear],bins);
        }
        NumericBytes=paths.Length*32L+enabled.Length+historyFrames*8L+signals.Length*(long)historyFrames*bins*8+
            work.Length*8L+output.Length*8L+(window.Length+overlap.Length)*4L+transfers.Length*8L+paths.Length*16L+earFilters.Length*(long)bins*8;
    }
    public void SetResponse(long sample,bool cut,RayTransport.Path[] response,int offset,AcousticTake.Scene scene,AcousticTake.State[] state)
    {
        if(response==null || offset<0 || response.Length-offset<paths.Length)throw new ArgumentException("Missing directional ray response.");
        foreach(var pose in state)cut|=(pose.flags&AcousticTake.StateFlags.Cut)!=0;
        double seconds=(double)(sample-responseSample)/rate;
        for(int i=0;i<paths.Length;i++)
        {
            var next=response[offset+i];var previous=paths[i];
            float ratio=hasResponse && !cut && seconds>0 && previous.pressureGain!=0 && next.pressureGain!=0 && previous.frequencyBin==next.frequencyBin ?
                (float)(1-(next.delaySeconds-previous.delaySeconds)/seconds):1;
            paths[i]=next;counts[i]=0;
            if(next.pressureGain==0)continue;
            int bin=checked((int)next.frequencyBin);
            if(bin>=bins || !float.IsFinite(next.pressureGain) || !float.IsFinite(next.delaySeconds) || next.delaySeconds<0)
                throw new InvalidOperationException("Invalid spectral arrival.");
            if(!float.IsFinite(ratio) || ratio<=0)continue;
            double delay=(double)next.delaySeconds*rate;delays[i]=(long)Math.Floor(delay/hop);
            double phase=-2*Math.PI*bin*(delay-delays[i]*hop)/size;
            var amplitude=new Vector2((float)Math.Cos(phase)*next.pressureGain,(float)Math.Sin(phase)*next.pressureGain);
            double target=bin*(double)ratio;
            if(target>=bins)continue; // Do not alias energy above Nyquist back into the audible range.
            int center=(int)Math.Floor(target),first=ratio==1?bin:Math.Max(0,center-4),last=ratio==1?bin:Math.Min(bins-1,center+4);
            firstBins[i]=first;counts[i]=last-first+1;
            for(int at=first;at<=last;at++)
            {
                double delta=target-at;
                double magnitude=Math.Abs(delta)<1e-10?1:Math.Sin(Math.PI*delta)/(size*Math.Sin(Math.PI*delta/size));
                double angle=Math.PI*delta*(size-1)/size;
                var weight=Multiply(amplitude,new Vector2((float)(magnitude*Math.Cos(angle)),(float)(magnitude*Math.Sin(angle))));
                for(int ear=0;ear<channels;ear++)
                {
                    int filter=channels==1?0:checked((int)next.hrtfIndex*2+ear);
                    transfers[(i*channels+ear)*FrequencySupport+at-first]=Multiply(weight,earFilters[filter][at]);
                }
            }
        }
        responseSample=sample;hasResponse=true;
        for(int i=0;i<listeners;i++)enabled[i]=TakeRender.Enabled(state[scene.listeners[i]],AcousticTake.StateFlags.Listener);
    }
    public void Render(long sample,int count,float[] pcm)
    {
        if(sample!=cursor)throw new InvalidOperationException("Spectral output must be consumed in order from initialization.");
        for(int n=0;n<count;n++,cursor++)
        {
            if(cursor%hop==0)Synthesize(cursor);
            int at=(int)(cursor%size);
            for(int l=0;l<listeners;l++)for(int ear=0;ear<channels;ear++)
            {
                int index=(l*channels+ear)*size+at;
                float value=enabled[l]?overlap[index]:0;
                if(!float.IsFinite(value))throw new InvalidOperationException("Non-finite spectral audio.");
                pcm[(n*listeners+l)*channels+ear]=value;overlap[index]=0;
            }
        }
    }
    void Synthesize(long sample)
    {
        long frame=sample/hop;int ring=(int)(frame%historyFrames);
        for(int s=0;s<signals.Length;s++)
        {
            for(int i=0;i<size;i++)work[i]=new Vector2(signals[s].Read(sample-size+i)*window[i],0);
            Fft(work,false);Array.Copy(work,0,history[s],ring*bins,bins);
        }
        historyKeys[ring]=frame;Array.Clear(output,0,output.Length);
        for(int pair=0;pair<pairs;pair++)
        {
            int source=pair/listeners,listener=pair%listeners;if(!enabled[listener])continue;
            for(int i=0;i<slots;i++)
            {
                int index=pair*slots+i;if(counts[index]==0)continue;
                int bin=(int)paths[index].frequencyBin;long inputFrame=frame-delays[index];
                if(inputFrame<0)continue;
                int inputSlot=(int)(inputFrame%historyFrames);if(historyKeys[inputSlot]!=inputFrame)continue;
                var value=history[source][inputSlot*bins+bin];
                for(int ear=0;ear<channels;ear++)for(int k=0;k<counts[index];k++)
                    output[(listener*channels+ear)*size+firstBins[index]+k]+=Multiply(value,transfers[(index*channels+ear)*FrequencySupport+k]);
            }
        }
        for(int c=0;c<listeners*channels;c++)
        {
            Array.Copy(output,c*size,work,0,bins);work[0].y=work[size/2].y=0;
            for(int k=1;k<size/2;k++)work[size-k]=new Vector2(work[k].x,-work[k].y);
            Fft(work,true);
            for(int i=0;i<size;i++)overlap[c*size+(int)((sample+i)%size)]+=work[i].x*window[i];
        }
    }
    static Vector2 Multiply(Vector2 a,Vector2 b)=>new Vector2(a.x*b.x-a.y*b.y,a.x*b.y+a.y*b.x);
    internal static void Fft(Vector2[] data,bool inverse)
    {
        int n=data.Length;
        for(int i=1,j=0;i<n;i++)
        {
            int bit=n>>1;for(; (j&bit)!=0;bit>>=1)j^=bit;j^=bit;
            if(i<j){var value=data[i];data[i]=data[j];data[j]=value;}
        }
        for(int length=2;length<=n;length*=2)
        {
            double angle=(inverse?2:-2)*Math.PI/length;var step=new Vector2((float)Math.Cos(angle),(float)Math.Sin(angle));
            for(int first=0;first<n;first+=length)
            {
                var w=Vector2.right;
                for(int j=0;j<length/2;j++)
                {
                    var a=data[first+j];var b=Multiply(data[first+j+length/2],w);
                    data[first+j]=a+b;data[first+j+length/2]=a-b;w=Multiply(w,step);
                }
            }
            if(length==n)break;
        }
        if(inverse)for(int i=0;i<n;i++)data[i]/=n;
    }
}
