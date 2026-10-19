using System;
using System.Collections.Generic;
using UnityEngine;

// Uniform partitioned linear convolution. Acoustic bins define a finite, centered
// pressure FIR; propagation and the complete HRIR are composed in time, never
// wrapped inside an STFT window. Input spectra live in a sample-keyed circular buffer.
internal sealed class ImpulseResponseAudio
{
    sealed class Response
    {
        public readonly Vector2[] kernels;
        public readonly int[] active,counts;
        public readonly float[] overlap,block;
        public bool reset=true;
        public Response(int pairs,int listeners,int channels,int partitions,int bins,int hop)
        {
            kernels=new Vector2[checked(pairs*channels*partitions*bins)];
            active=new int[checked(pairs*partitions)];counts=new int[pairs];
            overlap=new float[checked(listeners*channels*hop)];block=new float[overlap.Length];
        }
        public long Bytes=>kernels.LongLength*8+(active.LongLength+counts.Length+overlap.Length+block.Length)*4;
    }
    readonly int size,hop,fft,bins,rate,channels,listeners,pairs,slots,partitions,irLength,hrirTaps,delayRadius,latencyOffset;
    readonly TakeRender.Signal[] signals;
    readonly Vector2[][] history;
    readonly long[] historyKeys;
    readonly Vector2[] work,sum,native,pressureWork,earWork;
    readonly Dictionary<(float,Vector4),int> groups;
    readonly int[] groupHeads,next;
    readonly RayTransport.Path[] lastPaths;
    readonly Quaternion[] lastHeads;
    readonly bool[] lastEnabled;
    readonly double[] impulse,cosine,sine,prefixReal,prefixImag;
    readonly float[] delayWeights;
    readonly Hrtf hrtf;
    readonly int transition;
    int fade;
    Response current,previous;
    long cursor,responseSample;
    bool initialized;
    public long NumericBytes {get;}
    public int ResponseSupportSamples {get;private set;}
    public ImpulseResponseAudio(AcousticTake.Scene scene,int rate,TakeRender.Settings settings,TakeRender.Signal[] signals)
    {
        this.rate=rate;this.signals=signals;size=settings.fftSize;hop=settings.spectralHop;fft=checked(2*hop);bins=hop+1;
        channels=settings.ChannelsPerListener;listeners=scene.listeners.Length;pairs=checked(signals.Length*listeners);
        slots=settings.SpectralSlotsPerPair;transition=settings.transitionSamples;
        hrtf=channels==2?Hrtf.Load(rate):null;hrirTaps=hrtf?.Taps??1;
        delayRadius=settings.delayFilter==TakeRender.DelayFilter.WindowedSinc?settings.sincRadius:0;
        delayWeights=new float[delayRadius==0?2:checked(2*delayRadius)];
        // The block clock contributes hop samples; centering contributes size/2.
        // Keep the declared size (+ fractional FIR latency) end-to-end DSP delay.
        latencyOffset=size/2-hop;
        irLength=checked((int)Math.Ceiling(settings.maximumDelaySeconds*(double)rate)+latencyOffset+size+hrirTaps+2*delayRadius+2);
        partitions=checked((irLength+hop-1)/hop);
        historyKeys=new long[partitions+1];Array.Fill(historyKeys,long.MinValue);
        history=new Vector2[signals.Length][];
        for(int s=0;s<signals.Length;s++)history[s]=new Vector2[checked(historyKeys.Length*bins)];
        current=new Response(pairs,listeners,channels,partitions,bins,hop);
        previous=new Response(pairs,listeners,channels,partitions,bins,hop);
        work=new Vector2[fft];sum=new Vector2[checked(listeners*channels*bins)];impulse=new double[checked(channels*irLength)];
        cosine=new double[checked((size/2+1)*size)];sine=new double[cosine.Length];
        for(int b=0;b<=size/2;b++)for(int n=0;n<size;n++)
        {double angle=2*Math.PI*b*n/size;cosine[b*size+n]=Math.Cos(angle);sine[b*size+n]=Math.Sin(angle);}
        groups=new Dictionary<(float,Vector4),int>(slots);groupHeads=new int[slots];next=new int[slots];
        lastPaths=new RayTransport.Path[checked(pairs*slots)];lastHeads=new Quaternion[listeners];lastEnabled=new bool[listeners];
        native=new Vector2[size];int padded=1;while(padded<size+hrirTaps-1)padded=checked(padded*2);
        pressureWork=new Vector2[padded];earWork=new Vector2[padded];
        prefixReal=new double[checked(channels*(hrirTaps+1))];prefixImag=new double[prefixReal.Length];
        NumericBytes=lastPaths.LongLength*RayTransport.PathStride+lastHeads.LongLength*16+lastEnabled.Length+(groupHeads.LongLength+next.LongLength)*4+
            (native.LongLength+pressureWork.LongLength+earWork.LongLength)*8+current.Bytes+previous.Bytes+historyKeys.LongLength*8+signals.Length*(long)historyKeys.Length*bins*8+
            (work.LongLength+sum.LongLength)*8+(impulse.LongLength+cosine.LongLength+sine.LongLength+prefixReal.LongLength+prefixImag.LongLength)*8+delayWeights.LongLength*4;
    }
    public void SetResponse(long sample,bool cut,RayTransport.Path[] paths,int offset,AcousticTake.Scene scene,AcousticTake.State[] states)
    {
        if(paths==null || offset<0 || paths.Length-offset<checked(pairs*slots))throw new ArgumentException("Missing directional ray response.");
        foreach(var state in states)cut|=(state.flags&AcousticTake.StateFlags.Cut)!=0;
        bool unchanged=initialized;
        for(int i=0;i<lastPaths.Length && unchanged;i++)
        {
            var a=lastPaths[i];var b=paths[offset+i];
            unchanged=a.pressureGain==b.pressureGain && a.pressureImag==b.pressureImag && a.delaySeconds==b.delaySeconds && a.frequencyBin==b.frequencyBin && a.direction.Equals(b.direction);
        }
        for(int l=0;l<listeners;l++)
        {
            var state=states[scene.listeners[l]];bool enabled=TakeRender.Enabled(state,AcousticTake.StateFlags.Listener);
            unchanged&=lastHeads[l].Equals(state.rotation) && lastEnabled[l]==enabled;
            lastHeads[l]=state.rotation;lastEnabled[l]=enabled;
        }
        if(unchanged){fade=0;responseSample=sample;return;}
        Array.Copy(paths,offset,lastPaths,0,lastPaths.Length);
        (current,previous)=(previous,current);fade=initialized && !cut?transition:0;
        Array.Clear(current.counts,0,pairs);current.reset=true;
        for(int pair=0;pair<pairs;pair++)
        {
            int listener=pair%listeners;
            if(scene.sources[pair/listeners].mutedListenerPlusOne==listener+1 || !TakeRender.Enabled(states[scene.listeners[listener]],AcousticTake.StateFlags.Listener))continue;
            Array.Clear(impulse,0,impulse.Length);
            var rotation=Quaternion.Inverse(states[scene.listeners[listener]].rotation);
            int first=irLength,last=0,groupCount=0;groups.Clear();
            int pathOffset=offset+pair*slots;
            for(int slot=0;slot<slots;slot++)
            {
                var path=paths[pathOffset+slot];if(path.pressureGain==0 && path.pressureImag==0)continue;
                if(path.frequencyBin>size/2 || !float.IsFinite(path.pressureGain) || !float.IsFinite(path.pressureImag) || !float.IsFinite(path.delaySeconds) || path.delaySeconds<0 || !float.IsFinite(path.direction.sqrMagnitude))
                    throw new InvalidOperationException("Invalid spectral arrival.");
                var key=(path.delaySeconds,path.direction);
                if(!groups.TryGetValue(key,out int group)){group=groupCount++;groups.Add(key,group);groupHeads[group]=-1;}
                next[slot]=groupHeads[group];groupHeads[group]=slot;
            }
            for(int group=0;group<groupCount;group++)
            {
                int slot=groupHeads[group];var path=paths[pathOffset+slot];
                double delay=path.delaySeconds*(double)rate;int whole=checked((int)Math.Floor(delay));
                int start=checked(whole+latencyOffset+(delayRadius==0?0:1));
                int support=checked(size+hrirTaps+delayWeights.Length-2);
                if(start<0 || start>irLength-support)throw new InvalidOperationException("Arrival exceeds declared spectral history.");
                FractionalDelay(delay-whole);
                int bin=(int)path.frequencyBin,basis=bin*size;
                var blend=hrtf==null?default:hrtf.Interpolate(rotation*(Vector3)path.direction);
                if(next[slot]>=0)
                {
                    // Coalesce the same physical delay/direction across frequencies.
                    // One short FFT and linear HRIR convolution replaces per-bin FIR work.
                    Array.Clear(native,0,size);
                    for(int member=slot;member>=0;member=next[member])
                    {var arrival=paths[pathOffset+member];native[arrival.frequencyBin]+=new Vector2(arrival.pressureGain,arrival.pressureImag);}
                    native[0].y=native[size/2].y=0;
                    for(int k=1;k<size/2;k++)native[size-k]=new Vector2(native[k].x,-native[k].y);
                    Fft(native,true);Array.Clear(pressureWork,0,pressureWork.Length);
                    for(int n=0;n<size;n++)pressureWork[n].x=native[(n+size/2)%size].x;
                    Fft(pressureWork,false);
                    for(int ear=0;ear<channels;ear++)
                    {
                        Array.Clear(earWork,0,earWork.Length);
                        for(int t=0;t<hrirTaps;t++)earWork[t].x=hrtf==null?1:hrtf.Coefficient(blend,ear,t);
                        Fft(earWork,false);
                        for(int k=0;k<earWork.Length;k++){var a=pressureWork[k];var b=earWork[k];earWork[k]=new Vector2(a.x*b.x-a.y*b.y,a.x*b.y+a.y*b.x);}
                        Fft(earWork,true);
                        for(int n=0;n<size+hrirTaps-1;n++)for(int t=0;t<delayWeights.Length;t++)
                            impulse[ear*irLength+start+n+t]+=earWork[n].x*delayWeights[t];
                    }
                }
                else
                {
                    for(int ear=0;ear<channels;ear++)
                    {
                        int p=ear*(hrirTaps+1);prefixReal[p]=prefixImag[p]=0;
                        for(int t=0;t<hrirTaps;t++)
                        {
                            double h=hrtf==null?1:hrtf.Coefficient(blend,ear,t);
                            prefixReal[p+t+1]=prefixReal[p+t]+h*cosine[basis+t%size];
                            prefixImag[p+t+1]=prefixImag[p+t]-h*sine[basis+t%size];
                        }
                    }
                    double scale=(bin==0 || bin==size/2?1.0:2.0)/size, gain=path.pressureGain*scale, imaginary=(bin==0 || bin==size/2?0:path.pressureImag*scale);
                    for(int n=0;n<size+hrirTaps-1;n++)
                    {
                        int a=Math.Max(0,n-size+1),b=Math.Min(n+1,hrirTaps),phase=(n+size/2)%size;
                        double re=cosine[basis+phase]*gain-sine[basis+phase]*imaginary,im=sine[basis+phase]*gain+cosine[basis+phase]*imaginary;
                        for(int ear=0;ear<channels;ear++)
                        {
                            int p=ear*(hrirTaps+1);
                            double value=re*(prefixReal[p+b]-prefixReal[p+a])-im*(prefixImag[p+b]-prefixImag[p+a]);
                            for(int t=0;t<delayWeights.Length;t++)impulse[ear*irLength+start+n+t]+=value*delayWeights[t];
                        }
                    }
                }
                first=Math.Min(first,start);last=Math.Max(last,start+support);
            }
            if(last>0)ResponseSupportSamples=Math.Max(ResponseSupportSamples,last+hop);
            for(int partition=first/hop;partition*hop<last;partition++)
            {
                bool nonzero=false;
                for(int ear=0;ear<channels;ear++)
                {
                    Array.Clear(work,0,fft);bool earNonzero=false;
                    for(int n=0;n<hop && partition*hop+n<irLength;n++)
                    {float value=(float)impulse[ear*irLength+partition*hop+n];work[n].x=value;earNonzero|=value!=0;}
                    // Sparse propagation delays leave many empty partitions. Their FFT is exactly zero.
                    if(earNonzero)Fft(work,false);nonzero|=earNonzero;
                    Array.Copy(work,0,current.kernels,((pair*channels+ear)*partitions+partition)*bins,bins);
                }
                if(nonzero)current.active[pair*partitions+current.counts[pair]++]=partition;
            }
        }
        responseSample=sample;initialized=true;
    }
    void FractionalDelay(double fraction)
    {
        if(delayRadius==0){delayWeights[0]=(float)(1-fraction);delayWeights[1]=(float)fraction;return;}
        double sum=0;
        for(int t=0;t<delayWeights.Length;t++)
        {
            double x=1-delayRadius+t-fraction;
            double value=Math.Abs(x)<1e-12?1:Math.Sin(Math.PI*x)/(Math.PI*x)*Math.Sin(Math.PI*x/delayRadius)/(Math.PI*x/delayRadius);
            delayWeights[t]=(float)value;sum+=value;
        }
        for(int t=0;t<delayWeights.Length;t++)delayWeights[t]/=(float)sum;
    }
    public void Render(long sample,int count,float[] pcm)
    {
        if(sample!=cursor)throw new InvalidOperationException("Spectral output must be consumed in order from initialization.");
        for(int n=0;n<count;n++,cursor++)
        {
            if(cursor%hop==0)Synthesize(cursor);
            int at=(int)(cursor%hop);float blend=fade==0?1:Mathf.Clamp01((float)(cursor-responseSample)/fade);
            for(int channel=0;channel<listeners*channels;channel++)
            {
                int index=channel*hop+at;
                float value=current.block[index]*blend+(blend<1?previous.block[index]*(1-blend):0);
                if(!float.IsFinite(value))throw new InvalidOperationException("Non-finite spectral audio.");
                pcm[n*listeners*channels+channel]=value;
            }
        }
    }
    void Synthesize(long sample)
    {
        long frame=sample/hop;int ring=(int)(frame%historyKeys.Length);
        for(int s=0;s<signals.Length;s++)
        {
            Array.Clear(work,0,fft);
            for(int n=0;n<hop;n++)work[n].x=signals[s].Read(sample-hop+n);
            Fft(work,false);Array.Copy(work,0,history[s],ring*bins,bins);
        }
        historyKeys[ring]=frame;
        Synthesize(current,frame);
        if(fade>0 && sample-responseSample<fade)Synthesize(previous,frame);
    }
    void Synthesize(Response response,long frame)
    {
        if(response.reset)
        {
            Convolve(response,frame-1);
            for(int c=0;c<listeners*channels;c++)
            {Inverse(c);for(int n=0;n<hop;n++)response.overlap[c*hop+n]=work[hop+n].x;}
            response.reset=false;
        }
        Convolve(response,frame);
        for(int c=0;c<listeners*channels;c++)
        {
            Inverse(c);
            for(int n=0;n<hop;n++)
            {int at=c*hop+n;response.block[at]=work[n].x+response.overlap[at];response.overlap[at]=work[hop+n].x;}
        }
    }
    void Inverse(int channel)
    {
        Array.Copy(sum,channel*bins,work,0,bins);work[0].y=work[hop].y=0;
        for(int k=1;k<hop;k++)work[fft-k]=new Vector2(work[k].x,-work[k].y);
        Fft(work,true);
    }
    void Convolve(Response response,long frame)
    {
        Array.Clear(sum,0,sum.Length);
        for(int pair=0;pair<pairs;pair++)
        {
            int source=pair/listeners,listener=pair%listeners;
            for(int p=0;p<response.counts[pair];p++)
            {
                int partition=response.active[pair*partitions+p];long inputFrame=frame-partition;if(inputFrame<0)continue;
                int ring=(int)(inputFrame%historyKeys.Length);if(historyKeys[ring]!=inputFrame)continue;
                for(int ear=0;ear<channels;ear++)
                {
                    int output=(listener*channels+ear)*bins,kernel=((pair*channels+ear)*partitions+partition)*bins,input=ring*bins;
                    for(int k=0;k<bins;k++)
                    {var a=history[source][input+k];var b=response.kernels[kernel+k];sum[output+k]+=new Vector2(a.x*b.x-a.y*b.y,a.x*b.y+a.y*b.x);}
                }
            }
        }
    }
    internal static void Fft(Vector2[] data,bool inverse)
    {
        int n=data.Length;
        for(int i=1,j=0;i<n;i++)
        {
            int bit=n>>1;for(;(j&bit)!=0;bit>>=1)j^=bit;j^=bit;
            if(i<j){var value=data[i];data[i]=data[j];data[j]=value;}
        }
        for(int length=2;length<=n;length*=2)
        {
            double angle=(inverse?2:-2)*Math.PI/length;var step=new Vector2((float)Math.Cos(angle),(float)Math.Sin(angle));
            for(int first=0;first<n;first+=length)
            {
                var w=Vector2.right;
                for(int j=0;j<length/2;j++)
                {var a=data[first+j];var b=data[first+j+length/2];b=new Vector2(b.x*w.x-b.y*w.y,b.x*w.y+b.y*w.x);data[first+j]=a+b;data[first+j+length/2]=a-b;w=new Vector2(w.x*step.x-w.y*step.y,w.x*step.y+w.y*step.x);}
            }
            if(length==n)break;
        }
        if(inverse)for(int i=0;i<n;i++)data[i]/=n;
    }
}
