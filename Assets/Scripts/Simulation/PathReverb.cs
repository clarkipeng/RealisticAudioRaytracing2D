using System;
using System.Collections.Generic;
using UnityEngine;

// Audio from the Path method's histogram (docs/benchmarks/path-tracer-2026-11-12), CPU.
// The histogram is the only input: per head-frame direction, octave band and time bin, the
// energy arriving at the listener (the exact direct arrival is rendered separately as a path).
//
// Impulse response. Time is cut into slots of the histogram's fine width (0.25 ms). Each slot
// holds one impulse at a fixed random position inside it, with a fixed random sign; its energy
// is the histogram energy that falls in the slot (a bin spreads its energy uniformly over its
// span, so a fine bin is one impulse with amplitude sqrt(E) and a wide late bin is a train of
// impulses sharing it). Phase is random: a listening test found exact phase inaudible here.
// Bands: the impulse is band-filtered with amplitude-complementary weights cos^2 in log
// frequency between neighbouring band centres (they sum to one, so one impulse with equal band
// energies stays one flat click), i.e. each band of the source signal is convolved with its own
// impulse train and the bands are summed. The band filters are linear-phase FIRs applied to the
// source with P/2 samples of lookahead (inside the DSP latency), so the trains stay exact.
// Ears: per channel the six directions combine with the existing diffuse-field equalised HRTF
// band energies (LateBasis.EarWeights; interaural level only, no interaural delay), and two
// independent sign sequences are mixed per band to the HRTF set's diffuse-field interaural
// coherence. Positions and signs are fixed per pair, so an update changes amplitudes only.
// Updates crossfade sample by sample over one response step (two kernel banks), as LateReverb.
// Convolution: uniform partitioned overlap-save; each partition's kernel is a weighted sum of
// fixed spectra of its impulse groups (no FFT per update).
internal sealed class PathReverb
{
    const int Bands=TakeRender.Settings.LateBands,Directions=TakeRender.Settings.LateDirections;
    readonly int partition,bins,count,channels,listeners,pairs,sources,latency,step,rate,lookahead;
    readonly AcousticTake.Scene scene;
    readonly TakeRender.Signal[] signals;
    readonly PathBins histogram;
    readonly float[] earWeights;   // [direction][channel][band]
    readonly float[] alpha,beta;   // per band: sign mixing for the interaural coherence
    readonly Vector2[] filters;    // [band][2P] spectra of the band FIRs (delay P/2)
    // Impulse groups: runs of slots in one partition with the same bin weights.
    struct Group { public int partition,first,terms; } // terms: entries in groupBins/groupWeights
    readonly List<Group>[] groups; // per pair
    readonly List<int>[] groupBins;readonly List<float>[] groupWeights;
    readonly Vector2[][] spectra;  // per pair: [group][sign sequence 2][bin]
    readonly Vector2[] kernels;    // [bank][pair][channel][band][partition][bin]
    readonly Vector2[] inputs;     // [source][band][ring partition][bin]
    readonly float[] halves;       // [source][band][P]: the latest band-filtered half block
    readonly long[] halfBlock;     // per source: block of the stored half
    readonly bool[] active;        // [bank][pair]
    readonly float[] blocks;       // [bank][listener*channels][P]
    readonly Vector2[] work,wide,sum;
    readonly double[] energy;      // [channel][band][histogram bin] of one pair
    int bank,computedBlock=-1;
    long cursor,swap;
    bool initialized,blend;
    internal long NumericBytes { get { long b=(kernels.LongLength+inputs.LongLength+work.Length+wide.Length+sum.Length)*8+(halves.Length+blocks.Length)*4L+energy.Length*8L; foreach(var s in spectra)b+=s.LongLength*8; return b; } }
    internal long SupportSamples => (long)count*partition+latency;

    internal PathReverb(AcousticTake.Scene scene,int rate,TakeRender.Settings settings,TakeRender.Signal[] signals)
    {
        this.scene=scene;this.signals=signals;this.rate=rate;
        partition=settings.LatePartition;bins=partition+1;lookahead=partition/2;
        histogram=new PathBins(settings);
        count=checked((int)Math.Ceiling(histogram.Horizon*rate/partition)+1);
        channels=settings.ChannelsPerListener;listeners=scene.listeners.Length;sources=signals.Length;pairs=sources*listeners;
        latency=settings.DspLatencySamples;step=settings.responseStepSamples;
        if(lookahead>latency)throw new InvalidOperationException("The band filters' lookahead exceeds the DSP latency.");
        earWeights=LateBasis.Get(rate,partition,1,channels).EarWeights;
        alpha=new float[Bands];beta=new float[Bands];
        var coherence=channels==2?LateBasis.Coherence(Hrtf.Load(rate),rate,2*partition):null;
        for(int b=0;b<Bands;b++)
        {
            double g=coherence==null?1:coherence[Math.Min(partition,(int)Math.Round(LateBasis.BandCentre(b)*2*partition/rate))];
            alpha[b]=(float)Math.Sqrt((1+g)/2);beta[b]=(float)Math.Sqrt((1-g)/2);
        }
        filters=BandFilters(rate,partition);
        groups=new List<Group>[pairs];groupBins=new List<int>[pairs];groupWeights=new List<float>[pairs];spectra=new Vector2[pairs][];
        for(int p=0;p<pairs;p++)BuildGroups(p);
        kernels=new Vector2[checked(2L*pairs*channels*Bands*count*bins)];
        inputs=new Vector2[checked((long)sources*Bands*count*bins)];
        halves=new float[sources*Bands*partition];halfBlock=new long[sources];Array.Fill(halfBlock,long.MinValue);
        active=new bool[2*pairs];blocks=new float[2*listeners*channels*partition];
        work=new Vector2[2*partition];wide=new Vector2[2*partition];sum=new Vector2[bins];
        energy=new double[channels*Bands*histogram.Count];
    }
    // Amplitude-complementary band weights: cos^2(pi/2 log2(f/fc)) between neighbouring centres,
    // flat below 125 Hz and above 16 kHz; sum over bands is one at every frequency.
    internal static double BandWeight(int band,double hz)
    {
        double x=hz<=125?0:Math.Min(Bands-1,Math.Log(hz/125,2)),d=Math.Abs(x-band);
        if(d>=1)return 0;
        double c=Math.Cos(.5*Math.PI*d);return c*c;
    }
    // Linear-phase FIRs (P + 1 taps, delay P/2, Hann window) of the band weights, as 2P spectra.
    // Windowing keeps their sum an exact delayed unit impulse (the window is one at the centre).
    static Vector2[] BandFilters(int rate,int partition)
    {
        int n=2*partition,half=partition/2;var result=new Vector2[Bands*n];var x=new Vector2[n];
        for(int b=0;b<Bands;b++)
        {
            for(int k=0;k<n;k++){int f=k<=n/2?k:n-k;x[k]=new Vector2((float)BandWeight(b,(double)f*rate/n),0);}
            ImpulseResponseAudio.Fft(x,true);
            var taps=new float[n];
            for(int t=-half;t<=half;t++){double w=.5*(1+Math.Cos(Math.PI*t/(half+1)));taps[t+half]=(float)(x[(t+n)%n].x*w);}
            for(int k=0;k<n;k++)x[k]=new Vector2(k<taps.Length?taps[k]:0,0);
            ImpulseResponseAudio.Fft(x,false);
            Array.Copy(x,0,result,b*n,n);
        }
        return result;
    }
    static uint Hash(uint x) => AudioSourcePlane.Hash(x);
    // Slot j covers [j, j + 1) x FineSeconds; its impulse is at a fixed random sample inside it.
    void BuildGroups(int pair)
    {
        var list=new List<Group>();var gb=new List<int>();var gw=new List<float>();
        var trains=new List<(int partition,int[] at,float[] s1,float[] s2)>();
        double fine=histogram.FineSeconds;int slots=(int)Math.Ceiling(histogram.Horizon/fine);
        uint seed=Hash((uint)pair*0x9e3779b9u+0x51ed27u);
        int currentPartition=-1;var key=new List<(int,float)>();var at=new List<int>();var s1=new List<float>();var s2=new List<float>();
        void Close()
        {
            if(at.Count==0)return;
            list.Add(new Group{partition=currentPartition,first=gb.Count,terms=key.Count});
            foreach(var (bin,weight) in key){gb.Add(bin);gw.Add(weight);}
            trains.Add((currentPartition,at.ToArray(),s1.ToArray(),s2.ToArray()));
            at.Clear();s1.Clear();s2.Clear();
        }
        var terms=new List<(int,float)>();
        for(int j=0;j<slots;j++)
        {
            double a=j*fine,b=Math.Min(histogram.Horizon,(j+1)*fine);
            long first=(long)Math.Ceiling(a*rate),last=(long)Math.Ceiling(b*rate);
            if(last<=first)continue;
            uint h=Hash(seed^(uint)j*0x85ebca6bu);
            long sample=first+(long)((Hash(h)>>8)*(1.0/16777216.0)*(last-first));
            int part=(int)(sample/partition);
            if(part>=count)break;
            // Slot energy = sum over bins of E_bin x (overlap / bin width).
            terms.Clear();
            for(int k=histogram.FirstOverlap(a);k<histogram.Count && histogram.Start(k)<b;k++)
            {
                double w=Math.Max(0,Math.Min(histogram.End(k),b)-Math.Max(histogram.Start(k),a))/histogram.Width(k);
                if(w>0)terms.Add((k,(float)w));
            }
            bool same=part==currentPartition && terms.Count==key.Count;
            for(int i=0;same && i<terms.Count;i++)same=terms[i].Item1==key[i].Item1 && Math.Abs(terms[i].Item2-key[i].Item2)<=1e-6f*terms[i].Item2;
            if(!same){Close();currentPartition=part;key.Clear();key.AddRange(terms);}
            at.Add((int)(sample-(long)part*partition));
            s1.Add((Hash(h^0x68bc21ebu)&1)==0?1:-1);s2.Add((Hash(h^0x02e5be93u)&1)==0?1:-1);
        }
        Close();
        // Fixed spectra of each group's two sign sequences (zero-padded to 2P).
        var spectrum=new Vector2[checked(trains.Count*2*bins)];var x=new Vector2[2*partition];
        for(int g=0;g<trains.Count;g++)for(int q=0;q<2;q++)
        {
            Array.Clear(x,0,x.Length);var t=trains[g];
            for(int i=0;i<t.at.Length;i++)x[t.at[i]].x+=q==0?t.s1[i]:t.s2[i];
            ImpulseResponseAudio.Fft(x,false);Array.Copy(x,0,spectrum,(g*2+q)*bins,bins);
        }
        groups[pair]=list;groupBins[pair]=gb;groupWeights[pair]=gw;spectra[pair]=spectrum;
    }

    // The histogram of one solved frame (MethodTransport), or null.
    internal static Func<int,int,int,int,double> Histogram(RayTransport.LateFrame late)
    {
        if(!(late.Transport is MethodTransport m) || !m.IsPath)return null;
        int frame=late.Frame;return (pair,d,b,k)=>m.HistogramEnergy(frame,pair,d,b,k);
    }
    internal static Func<int,int,uint> Counts(RayTransport.LateFrame late)
    {
        if(!(late.Transport is MethodTransport m) || !m.IsPath)return null;
        int frame=late.Frame;return (pair,k)=>m.HistogramCount(frame,pair,k);
    }
    // Offline experiments (settings.pathHistogramExport / pathHistogramImport): every valid response's
    // histograms [pair][direction][band][bin] (float32), counts [pair][bin] (uint32) and its sample (int64)
    // are appended to <dir>/histogram.f32, counts.u32 and samples.i64; an import replaces the tracer's
    // histogram of the i-th valid response by the i-th record of <dir>/histogram.f32 (e.g. denoised).
    System.IO.BinaryWriter exportHistogram,exportCounts,exportSamples;
    System.IO.FileStream import;
    long importRecord;
    internal void Configure(string exportDirectory,string importDirectory)
    {
        if(!string.IsNullOrEmpty(exportDirectory))
        {
            System.IO.Directory.CreateDirectory(exportDirectory);
            exportHistogram=new System.IO.BinaryWriter(System.IO.File.Create(System.IO.Path.Combine(exportDirectory,"histogram.f32")));
            exportCounts=new System.IO.BinaryWriter(System.IO.File.Create(System.IO.Path.Combine(exportDirectory,"counts.u32")));
            exportSamples=new System.IO.BinaryWriter(System.IO.File.Create(System.IO.Path.Combine(exportDirectory,"samples.i64")));
        }
        if(!string.IsNullOrEmpty(importDirectory))import=System.IO.File.OpenRead(System.IO.Path.Combine(importDirectory,"histogram.f32"));
    }
    internal void Close(){exportHistogram?.Dispose();exportCounts?.Dispose();exportSamples?.Dispose();import?.Dispose();exportHistogram=null;exportCounts=null;exportSamples=null;import=null;}
    Func<int,int,int,int,double> Exchange(long sample,Func<int,int,int,int,double> histogram,Func<int,int,uint> counts)
    {
        if(histogram==null)return null;
        int B=this.histogram.Count;
        if(exportHistogram!=null)
        {
            for(int p=0;p<pairs;p++)for(int d=0;d<Directions;d++)for(int b=0;b<Bands;b++)for(int k=0;k<B;k++)exportHistogram.Write((float)histogram(p,d,b,k));
            for(int p=0;p<pairs;p++)for(int k=0;k<B;k++)exportCounts.Write(counts==null?0u:counts(p,k));
            exportSamples.Write(sample);
            exportHistogram.Flush();exportCounts.Flush();exportSamples.Flush(); // the mixer has no end-of-render hook
        }
        if(import==null)return histogram;
        var record=new float[pairs*Directions*Bands*B];var bytes=new byte[record.Length*4];
        import.Seek(importRecord++*bytes.Length,System.IO.SeekOrigin.Begin);
        if(import.Read(bytes,0,bytes.Length)!=bytes.Length)throw new InvalidOperationException("The imported histograms end before the render.");
        Buffer.BlockCopy(bytes,0,record,0,bytes.Length);
        return (p,d,b,k)=>record[((p*Directions+d)*Bands+b)*B+k];
    }
    // histogram(pair, direction, band, bin): energy in direct-sound units; null renders silence.
    internal void SetResponse(long sample,bool cut,Func<int,int,int,int,double> histogram,AcousticTake.State[] states,Func<int,int,uint> counts=null)
    {
        histogram=Exchange(sample,histogram,counts);
        if(initialized && sample<swap)throw new ArgumentException("Path responses must be ordered.");
        bool valid=histogram!=null;
        if(valid && sample%partition!=0)throw new ArgumentException("Path responses change on partition boundaries.");
        blend=initialized && !cut;bank^=1;swap=sample;initialized=true;computedBlock=-1;
        long bankSize=(long)pairs*channels*Bands*count*bins;
        Array.Clear(kernels,(int)(bank*bankSize),(int)bankSize);
        int B=this.histogram.Count;
        for(int pair=0;pair<pairs;pair++)
        {
            int listener=pair%listeners;
            active[bank*pairs+pair]=valid && scene.sources[pair/listeners].mutedListenerPlusOne!=listener+1 &&
                TakeRender.Enabled(states[scene.listeners[listener]],AcousticTake.StateFlags.Listener);
            if(!active[bank*pairs+pair])continue;
            Array.Clear(energy,0,energy.Length);
            for(int d=0;d<Directions;d++)for(int b=0;b<Bands;b++)for(int k=0;k<B;k++)
            {
                double e=histogram(pair,d,b,k);if(!(e>0))continue;
                for(int c=0;c<channels;c++)energy[(c*Bands+b)*B+k]+=earWeights[(d*2+c)*Bands+b]*e;
            }
            var list=groups[pair];var gb=groupBins[pair];var gw=groupWeights[pair];var spectrum=spectra[pair];
            for(int c=0;c<channels;c++)for(int b=0;b<Bands;b++)
            {
                float a1=alpha[b],a2=channels==2?(c==0?beta[b]:-beta[b]):0;
                long baseKernel=(((long)(bank*pairs+pair)*channels+c)*Bands+b)*count*bins;
                for(int g=0;g<list.Count;g++)
                {
                    var group=list[g];double slotEnergy=0;
                    for(int i=0;i<group.terms;i++)slotEnergy+=energy[(c*Bands+b)*B+gb[group.first+i]]*gw[group.first+i];
                    if(!(slotEnergy>0))continue;
                    float amplitude=(float)Math.Sqrt(slotEnergy);
                    float w1=amplitude*a1,w2=amplitude*a2;
                    long kernel=baseKernel+(long)group.partition*bins;int v1=(g*2)*bins,v2=v1+bins;
                    for(int i=0;i<bins;i++)kernels[kernel+i]+=spectrum[v1+i]*w1+spectrum[v2+i]*w2;
                }
            }
        }
    }
    // Band-filtered source half block (P samples) for UPOLS block `block`: x_b over
    // [block P - latency, block P + P - latency), from x read P/2 samples ahead.
    void Half(int s,long block)
    {
        long start=block*partition-latency+lookahead-partition; // overlap-save window of the FIR
        for(int n=0;n<2*partition;n++)work[n]=new Vector2(start+n<0?0:signals[s].Read(start+n),0);
        ImpulseResponseAudio.Fft(work,false);
        for(int b=0;b<Bands;b++)
        {
            for(int i=0;i<2*partition;i++){var x=work[i];var h=filters[b*2*partition+i];wide[i]=new Vector2(x.x*h.x-x.y*h.y,x.x*h.y+x.y*h.x);}
            ImpulseResponseAudio.Fft(wide,true);
            for(int n=0;n<partition;n++)halves[(s*Bands+b)*partition+n]=wide[partition+n].x;
        }
        halfBlock[s]=block;
    }
    void Compute(int block)
    {
        computedBlock=block;int ring=block%count;
        for(int s=0;s<sources;s++)
        {
            bool used=false;for(int l=0;l<listeners;l++)used|=active[bank*pairs+s*listeners+l]||active[(bank^1)*pairs+s*listeners+l];
            if(!used){for(int b=0;b<Bands;b++)Array.Clear(inputs,((s*Bands+b)*count+ring)*bins,bins);halfBlock[s]=long.MinValue;continue;}
            bool continuous=halfBlock[s]==block-1;
            var old=new float[Bands*partition];
            if(continuous)Array.Copy(halves,s*Bands*partition,old,0,old.Length);
            else{Half(s,block-1);Array.Copy(halves,s*Bands*partition,old,0,old.Length);}
            Half(s,block);
            for(int b=0;b<Bands;b++)
            {
                for(int n=0;n<partition;n++){work[n]=new Vector2(old[b*partition+n],0);work[partition+n]=new Vector2(halves[(s*Bands+b)*partition+n],0);}
                ImpulseResponseAudio.Fft(work,false);
                Array.Copy(work,0,inputs,((s*Bands+b)*count+ring)*bins,bins);
            }
        }
        long start=(long)block*partition;
        bool both=blend && start<swap+step;
        for(int which=0;which<(both?2:1);which++)
        {
            int b0=bank^which;
            for(int l=0;l<listeners;l++)for(int c=0;c<channels;c++)
            {
                Array.Clear(sum,0,bins);
                for(int s=0;s<sources;s++)
                {
                    int pair=s*listeners+l;if(!active[b0*pairs+pair])continue;
                    for(int band=0;band<Bands;band++)
                    {
                        long kernelBase=(((long)(b0*pairs+pair)*channels+c)*Bands+band)*count*bins;
                        for(int k=0;k<count && k<=block;k++)
                        {
                            int input=((s*Bands+band)*count+(block-k)%count)*bins;long kernel=kernelBase+(long)k*bins;
                            for(int i=0;i<bins;i++){var x=inputs[input+i];var h=kernels[kernel+i];sum[i]+=new Vector2(x.x*h.x-x.y*h.y,x.x*h.y+x.y*h.x);}
                        }
                    }
                }
                for(int i=0;i<bins;i++)work[i]=sum[i];work[0].y=work[partition].y=0;
                for(int i=1;i<partition;i++)work[2*partition-i]=new Vector2(sum[i].x,-sum[i].y);
                ImpulseResponseAudio.Fft(work,true);
                for(int n=0;n<partition;n++)blocks[(b0*listeners*channels+l*channels+c)*partition+n]=work[partition+n].x;
            }
        }
    }
    // Adds the reverberation to interleaved listener/channel PCM.
    internal void Render(long sample,int length,float[] pcm)
    {
        if(!initialized || sample!=cursor)throw new InvalidOperationException("Path output must follow its first response in order.");
        for(int n=0;n<length;n++,cursor++)
        {
            int block=(int)(cursor/partition),at=(int)(cursor%partition);
            if(block!=computedBlock)Compute(block);
            float weight=blend?Mathf.Clamp01((float)(cursor-swap)/step):1;
            bool both=blend && block*(long)partition<swap+step;
            for(int channel=0;channel<listeners*channels;channel++)
            {
                float value=blocks[(bank*listeners*channels+channel)*partition+at]*weight;
                if(both && weight<1)value+=blocks[((bank^1)*listeners*channels+channel)*partition+at]*(1-weight);
                pcm[n*listeners*channels+channel]+=value;
            }
        }
    }
}
