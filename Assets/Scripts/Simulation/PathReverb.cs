using System;
using System.Collections.Generic;
using UnityEngine;

// Audio from the Path method's histogram (docs/benchmarks/path-tracer-2026-11-12), CPU.
// The histogram is the only input: per head-frame direction, octave band and time bin, the
// energy arriving at the listener (the exact direct arrival is rendered separately as a path).
//
// Impulse response. Time is cut into slots of the histogram's fine width (0.25 ms). Each slot
// holds a fixed noise-like sequence of unit energy (Gaussian per sample, see SlotSequences), scaled
// by sqrt of the histogram energy that falls in the slot (a bin spreads its energy uniformly over its
// span): an isolated arrival is a 0.25 ms burst, an empty bin is silent, and a filled late field is
// Gaussian noise, as dense as real reverberation (one impulse per slot measured an echo density
// of 0.3 against about 0.85 for real halls). Phase is random: a listening test found exact phase
// inaudible here.
// Bands: the impulse is band-filtered with amplitude-complementary weights cos^2 in log
// frequency between neighbouring band centres (they sum to one, so one impulse with equal band
// energies stays one flat click), i.e. each band of the source signal is convolved with its own
// impulse train and the bands are summed. The band filters are linear-phase FIRs applied to the
// source with P/2 samples of lookahead (inside the DSP latency), so the trains stay exact.
// Ears: per channel the six directions combine with the existing diffuse-field equalised HRTF
// band energies (LateBasis.EarWeights; interaural level only, no interaural delay), and two
// independent (orthogonal) slot sequences are mixed per band to the HRTF set's diffuse-field interaural
// coherence. Positions and signs are fixed per pair, so an update changes amplitudes only.
// Updates crossfade sample by sample over one response step (two kernel banks), with
// a power correction: the two outputs share signs but their amplitudes differ (independent noisy
// estimates), so a linear crossfade loses energy (about 1.2 dB at 512 rays in the Church). Over each
// block the blended output is scaled so its power is the linear interpolation of the two outputs'
// powers, which is what the histograms say; equal responses keep a gain of one.
// Convolution: uniform partitioned overlap-save; each partition's kernel is a weighted sum of
// fixed spectra of its impulse groups (no FFT per update).
// Discrete echoes (docs/method.md, "Discrete echoes, Doppler and ITD"; README parts 40, 42 and 45). In the early part,
// where echoes are discrete, a bin is one echo: an impulse at its mean delay tau with the bin's band energies per ear
// (the slots' noise would give it a new random phase at every update), read from the band-filtered input with the
// direct path's interpolation filter. A bin is one echo while its contributions share one arrival, i.e. one slowness
// vector: at least two of them (one cannot show it); one delay rate, its spread no larger than its mean (variance <=
// mean^2; the rates are all zero while nothing moves, which passes); and one direction, by the same rule on the arrival
// direction: its spread 1 - |u|^2 no larger than |u|^2, u the energy-weighted mean direction. The early part ends at the
// first bin that holds energy and is not one arrival: it pools arrivals of different shifts or directions, which one
// impulse cannot represent, and it and every later bin stay in the convolution.
// Doppler (part 40): the tracer's rate channel gives each bin the energy-weighted mean and mean square of its
// contributions' delay rates and their mean delay. An echo slides at its mean rate: its delay is tau + r (t - t_ref),
// which gives it its own pitch shift. A walk that does not compute the rate channel (nothing moved: the static walk)
// gives no mean delay within the bin; the echo then sits at the bin's centre, the centroid of the uniform spread the
// slots give a bin.
// Interaural delay (part 42): an echo is one arrival, so it has one direction, and its ears hear it at different times,
// whether or not anything moves (part 45). The direction is read from the bin's six direction channels: the tracer
// splits each arrival's energy over the axes by its squared components, so for one arrival u_axis = sqrt(E_+axis / E) -
// sqrt(E_-axis / E) exactly, |u| = 1 (for several, an energy-weighted mean; arrivals on opposite sides of an axis
// shorten u). The HRTF set's own ITD there (Hrtf.InterauralDelay, the direct path's blend) splits the echo's delay
// symmetrically: the left ear's is tau + ITD / 2, the right's tau - ITD / 2. The dense part keeps the diffuse field's
// interaural coherence, which already holds the ITDs of every direction, so it gets none (it would count them twice).
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
    internal struct Group { public int partition,first,terms; } // terms: entries in groupBins/groupWeights
    readonly List<Group>[] groups; // per pair
    readonly List<int>[] groupBins;readonly List<float>[] groupWeights;
    readonly Vector2[][] spectra;  // per pair: [group][slot sequence 2][bin]
    readonly float[][] mixes;      // per pair: [group][band][W11, W12, W22], each band's exact-energy mix of the two sequences
    readonly Vector2[] kernels;    // [bank][pair][channel][band][partition][bin]
    readonly Vector2[] inputs;     // [source][band][ring partition][bin]
    readonly float[] halves;       // [source][band][P]: the latest band-filtered half block
    readonly long[] halfBlock;     // per source: block of the stored half
    readonly bool[] active;        // [bank][pair]
    readonly float[] blocks;       // [bank][listener*channels][P]
    readonly double[] powers;      // [listener*channels][3]: new, old and cross power of a blended block
    readonly Vector2[] work,wide,sum;
    readonly double[] energy;      // [channel][band][histogram bin] of one pair
    int bank,computedBlock=-1;
    long cursor,swap;
    bool initialized,blend;
    // Echoes: per bank which bins are echoes and their slide rates [bank][pair][bin], their mean delays in samples and
    // their per-ear band amplitudes [bank][pair][bin][channel][band], the output sample at which the bank's echoes sit
    // at their mean delays, and per source and band a ring of the band-filtered input.
    readonly bool[] echoes;
    readonly float[] slideRates,slideDelays,slideAmplitudes;   // slideDelays: [bank][pair][bin][channel], with the ITD
    readonly float[] itdGrid;   // binaural: the HRTF set's ITD over head-frame directions (Hrtf.InterauralDelayGrid)
    readonly long[] slideReference=new long[2];
    readonly bool[] sliding=new bool[2];
    readonly float[] history; // [source][band][historyLength]
    readonly int historyLength,sincRadius;
    readonly bool sinc,doppler;   // doppler: settings.pathReflectionDoppler (off: the rates are ignored)
    long historyEnd=long.MinValue;   // next band block to compute
    readonly double[] rawEnergy,moment1,moment2;readonly uint[] binCounts;
    // Room modes (RoomModes.cs): per update and pair inside the bank's room, the histogram's band energies scaled by g_b
    // and the modal difference dh added to every band's kernel (the band filters sum to a delayed unit impulse, so the
    // band-split input convolved with dh in each band is the input convolved with dh), the same for both ears (the head
    // is small against the wavelengths below the crossover's top).
    readonly RoomModes modes;
    readonly double[] modeAmplitudes,modeResponse,unitShare={1,1,1,1,1,1,1,1};
    readonly Vector2[] modePiece;
    readonly float minimumDistance;
    internal int ModalPairs;   // diagnostics: pairs given the modes by the latest response
    internal RoomModes Modes => modes;
    internal int EchoBins;   // diagnostics: the latest response's echoes
    internal long NumericBytes { get { long b=(kernels.LongLength+inputs.LongLength+work.Length+wide.Length+sum.Length)*8+(halves.Length+blocks.Length)*4L+energy.Length*8L; foreach(var s in spectra)b+=s.LongLength*8; foreach(var m in mixes)b+=m.LongLength*4; return b; } }
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
        earWeights=LateBasis.EarWeights(rate,channels);
        alpha=new float[Bands];beta=new float[Bands];
        var coherence=channels==2?LateBasis.Coherence(Hrtf.Load(rate),rate,2*partition):null;
        for(int b=0;b<Bands;b++)
        {
            double g=coherence==null?1:coherence[Math.Min(partition,(int)Math.Round(LateBasis.BandCentre(b)*2*partition/rate))];
            alpha[b]=(float)Math.Sqrt((1+g)/2);beta[b]=(float)Math.Sqrt((1-g)/2);
        }
        filters=BandFilters(rate,partition);
        groups=new List<Group>[pairs];groupBins=new List<int>[pairs];groupWeights=new List<float>[pairs];spectra=new Vector2[pairs][];mixes=new float[pairs][];
        for(int p=0;p<pairs;p++)BuildGroups(p);
        kernels=new Vector2[checked(2L*pairs*channels*Bands*count*bins)];
        inputs=new Vector2[checked((long)sources*Bands*count*bins)];
        halves=new float[sources*Bands*partition];halfBlock=new long[sources];Array.Fill(halfBlock,long.MinValue);
        active=new bool[2*pairs];blocks=new float[2*listeners*channels*partition];powers=new double[3*listeners*channels];
        work=new Vector2[2*partition];wide=new Vector2[2*partition];sum=new Vector2[bins];
        energy=new double[channels*Bands*histogram.Count];
        sincRadius=settings.sincRadius;sinc=settings.delayFilter==TakeRender.DelayFilter.WindowedSinc;doppler=settings.pathReflectionDoppler;
        int B=histogram.Count;
        echoes=new bool[2*pairs*B];slideRates=new float[2*pairs*B];slideDelays=new float[2*pairs*B*channels];itdGrid=channels==2?Hrtf.Load(rate).InterauralDelayGrid:null;slideAmplitudes=new float[checked(2*pairs*B*channels*Bands)];
        // History: the horizon, one step of drift either way, the interpolation's reach and two blocks.
        int need=checked(count*partition+2*step+2*(sincRadius+2)*4+4*partition);
        historyLength=1;while(historyLength<need)historyLength<<=1;
        history=new float[checked(sources*Bands*historyLength)];
        rawEnergy=new double[B];moment1=new double[B];moment2=new double[B];binCounts=new uint[B];
        modes=RoomModes.ForScene(scene,settings,rate,count*partition);minimumDistance=settings.minimumDistance;
        if(modes!=null){modeAmplitudes=new double[modes.Bank.Modes];modeResponse=new double[count*partition];modePiece=new Vector2[2*partition];}
    }
    // The early part of one pair's response: per bin whether it is one echo, and its slide rate. energy(k): the bin's raw
    // energy over every direction and band; m1, m2: its rate moments (null: no rates, all zero); count: its
    // contributions; direction: its mean arrival direction (EchoDirection, not normalised). In time order a bin with
    // energy is an echo while its contributions share one arrival: at least two, one rate (variance <= mean^2) and one
    // direction (1 - |u|^2 <= |u|^2); the first that is not ends the early part.
    internal static void SelectEchoes(int bins,Func<int,double> energy,Func<int,double> m1,Func<int,double> m2,Func<int,uint> count,Func<int,Vector3> direction,bool[] echoes,float[] rates,int offset)
    {
        bool open=true;
        for(int k=0;k<bins;k++)
        {
            echoes[offset+k]=false;rates[offset+k]=0;double e=energy(k);
            if(!open || !(e>0))continue;
            double mean=m1==null?0:m1(k)/e,variance=m1==null?0:m2(k)/e-mean*mean,u2=direction(k).sqrMagnitude;
            if(count(k)>=2 && variance<=mean*mean && 1-u2<=u2){echoes[offset+k]=true;rates[offset+k]=(float)mean;}else open=false;
        }
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
    internal static Vector2[] BandFilters(int rate,int partition)
    {
        int n=2*partition,half=partition/2;var result=new Vector2[Bands*n];var x=new Vector2[n];
        for(int b=0;b<Bands;b++)
        {
            for(int k=0;k<n;k++){int f=k<=n/2?k:n-k;x[k]=new Vector2((float)BandWeight(b,(double)f*rate/n),0);}
            SpectralAudio.Fft(x,true);
            var taps=new float[n];
            for(int t=-half;t<=half;t++){double w=.5*(1+Math.Cos(Math.PI*t/(half+1)));taps[t+half]=(float)(x[(t+n)%n].x*w);}
            for(int k=0;k<n;k++)x[k]=new Vector2(k<taps.Length?taps[k]:0,0);
            SpectralAudio.Fft(x,false);
            Array.Copy(x,0,result,b*n,n);
        }
        return result;
    }
    static uint Hash(uint x) => AudioSourcePlane.Hash(x);
    // Slot j covers [j, j + 1) x FineSeconds, samples [first, last). Its content is fixed per pair: two
    // unit-energy sequences (one per ear), noise-like (Gaussian per sample, so the late field is as dense
    // as real reverberation), with the slot's energy exact: s1 = m 1/M + r1 (area m = +-1, so a band whose
    // period is long against the slot sees exactly the slot's energy, as from one impulse) and s2 = r2,
    // with r1, r2 orthogonal to the constant and to each other (Gram-Schmidt), so the two ears' mix keeps
    // the slot's energy exactly at any coherence. Slots of one or two samples fall back to impulses.
    static double Gaussian(uint h)
    {
        double u1=((Hash(h)>>8)+.5)*(1.0/16777216.0),u2=((Hash(h^0x9e3779b9u)>>8)+.5)*(1.0/16777216.0);
        return Math.Sqrt(-2*Math.Log(u1))*Math.Cos(2*Math.PI*u2);
    }
    static void SlotSequences(uint h,int m,double[] s1,double[] s2)
    {
        double sign=(Hash(h^0x68bc21ebu)&1)==0?1:-1,sign2=(Hash(h^0x02e5be93u)&1)==0?1:-1;
        if(m<3)
        {
            Array.Clear(s1,0,m);Array.Clear(s2,0,m);
            int at=(int)((Hash(h)>>8)*(1.0/16777216.0)*m);s1[at]=sign;
            if(m==2)s2[1-at]=sign2;else s2[0]=sign2;
            return;
        }
        // r1, r2: Gaussian, minus their means (orthogonal to the constant), r2 also orthogonal to r1.
        double mean1=0,mean2=0;
        for(int n=0;n<m;n++){s1[n]=Gaussian(h^(uint)(2*n+1)*0x85ebca6bu);s2[n]=Gaussian(h^(uint)(2*n+2)*0xc2b2ae35u);mean1+=s1[n];mean2+=s2[n];}
        mean1/=m;mean2/=m;double e1=0,dot=0;
        for(int n=0;n<m;n++){s1[n]-=mean1;s2[n]-=mean2;e1+=s1[n]*s1[n];dot+=s1[n]*s2[n];}
        double e2=0;for(int n=0;n<m;n++){s2[n]-=dot/e1*s1[n];e2+=s2[n]*s2[n];}
        double a=Math.Sqrt((1-1.0/m)/e1),b=1/Math.Sqrt(e2);
        for(int n=0;n<m;n++){s1[n]=sign/m+a*s1[n];s2[n]*=b;}
    }
    // Slot layout of one pair (shared with GpuAudio.Path): impulse groups, their bin weights, the fixed spectra of
    // their two sequences (CPU convolution only) and per band the exact-energy mix; with `samples`, per sample of
    // the kernel ([count][P]) its group (-1: none) and the two sequences' values (GPU synthesis).
    internal sealed class Slots
    {
        public List<Group> groups;public List<int> bins;public List<float> weights;public Vector2[] spectra;public float[] mix;
        public int[] sampleGroup;public float[] s1,s2;
    }
    void BuildGroups(int pair)
    {
        var slots=BuildSlots(pair,histogram,rate,partition,count,filters,true,false);
        groups[pair]=slots.groups;groupBins[pair]=slots.bins;groupWeights[pair]=slots.weights;spectra[pair]=slots.spectra;mixes[pair]=slots.mix;
    }
    internal static Slots BuildSlots(int pair,PathBins histogram,int rate,int partition,int count,Vector2[] filters,bool keepSpectra,bool samples)
    {
        int bins=partition+1;
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
        var q1=new double[64];var q2=new double[64];
        for(int j=0;j<slots;j++)
        {
            double a=j*fine,b=Math.Min(histogram.Horizon,(j+1)*fine);
            long first=(long)Math.Ceiling(a*rate),last=(long)Math.Ceiling(b*rate);
            if(last<=first)continue;
            if((int)(first/partition)>=count)break;
            int m=(int)(last-first);
            if(q1.Length<m){q1=new double[m];q2=new double[m];}
            SlotSequences(Hash(seed^(uint)j*0x85ebca6bu),m,q1,q2);
            // Slot energy = sum over bins of E_bin x (overlap / bin width).
            terms.Clear();
            for(int k=histogram.FirstOverlap(a);k<histogram.Count && histogram.Start(k)<b;k++)
            {
                double w=Math.Max(0,Math.Min(histogram.End(k),b)-Math.Max(histogram.Start(k),a))/histogram.Width(k);
                if(w>0)terms.Add((k,(float)w));
            }
            for(int n=0;n<m;n++)
            {
                long sample=first+n;int part=(int)(sample/partition);
                if(part>=count)break;
                bool same=part==currentPartition && terms.Count==key.Count;
                for(int i=0;same && i<terms.Count;i++)same=terms[i].Item1==key[i].Item1 && Math.Abs(terms[i].Item2-key[i].Item2)<=1e-6f*terms[i].Item2;
                if(!same){Close();currentPartition=part;key.Clear();key.AddRange(terms);}
                at.Add((int)(sample-(long)part*partition));
                s1.Add((float)q1[n]);s2.Add((float)q2[n]);
            }
        }
        Close();
        // Fixed spectra of each group's two slot sequences (zero-padded to 2P).
        var spectrum=new Vector2[checked(trains.Count*2*bins)];var x=new Vector2[2*partition];var y=new Vector2[2*partition];
        var mix=new float[trains.Count*Bands*3];int n2=2*partition;
        for(int g=0;g<trains.Count;g++)
        {
            var t=trains[g];double e1=0,e2=0;
            Array.Clear(x,0,n2);Array.Clear(y,0,n2);
            for(int i=0;i<t.at.Length;i++){x[t.at[i]].x+=t.s1[i];y[t.at[i]].x+=t.s2[i];e1+=t.s1[i]*t.s1[i];e2+=t.s2[i]*t.s2[i];}
            SpectralAudio.Fft(x,false);SpectralAudio.Fft(y,false);
            Array.Copy(x,0,spectrum,(g*2)*bins,bins);Array.Copy(y,0,spectrum,(g*2+1)*bins,bins);
            // Exact band energy: the band-filtered pair (h_b * s1, h_b * s2) is made orthonormal at the
            // energy an impulse train of the same broadband energy would give, W = sqrt(target) G^(-1/2),
            // G its Gram matrix (2P points: the linear convolution fits, so these sums are exact).
            for(int b=0;b<Bands;b++)
            {
                double a=0,c=0,d=0,reference=0;
                for(int k=0;k<n2;k++)
                {
                    var h=filters[b*n2+k];double h2=h.x*h.x+h.y*h.y;var u=x[k];var v=y[k];
                    a+=h2*(u.x*u.x+u.y*u.y);d+=h2*(v.x*v.x+v.y*v.y);c+=h2*(u.x*v.x+u.y*v.y);reference+=h2;
                }
                a/=n2;c/=n2;d/=n2;reference/=n2;
                double target=.5*(e1+e2)*reference,det=a*d-c*c;int o=(g*Bands+b)*3;
                if(det>1e-9*a*d && a>0 && d>0)
                {
                    double sd=Math.Sqrt(det),tr=Math.Sqrt(a+d+2*sd),scale=Math.Sqrt(target)/(sd*tr);
                    mix[o]=(float)((d+sd)*scale);mix[o+1]=(float)(-c*scale);mix[o+2]=(float)((a+sd)*scale);
                }
                else if(a>0){mix[o]=(float)Math.Sqrt(target/a);mix[o+1]=0;mix[o+2]=0;}
            }
        }
        var result=new Slots{groups=list,bins=gb,weights=gw,spectra=keepSpectra?spectrum:null,mix=mix};
        if(samples)
        {
            result.sampleGroup=new int[checked(count*partition)];Array.Fill(result.sampleGroup,-1);
            result.s1=new float[result.sampleGroup.Length];result.s2=new float[result.sampleGroup.Length];
            for(int g=0;g<trains.Count;g++)
            {
                var t=trains[g];int origin=t.partition*partition;
                for(int i=0;i<t.at.Length;i++){result.sampleGroup[origin+t.at[i]]=g;result.s1[origin+t.at[i]]=t.s1[i];result.s2[origin+t.at[i]]=t.s2[i];}
            }
        }
        return result;
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
    // The rate channel of one solved frame (MethodTransport.RateMoment: pair, moment, bin), or null when the walk did not
    // compute it (nothing moved: MethodTransport.Doppler).
    internal static Func<int,int,int,double> Rates(RayTransport.LateFrame late)
    {
        if(!(late.Transport is MethodTransport m) || !m.IsPath || !m.Doppler)return null;
        int frame=late.Frame;return (pair,moment,k)=>m.RateMoment(frame,pair,moment,k);
    }
    // counts (pair, bin): the bins' contributions; null renders every bin through the slots (no echoes).
    // rates (pair, moment, bin): the rate channel (Doppler on reflections); null: nothing slides, and echoes sit at their
    // bins' centres.
    internal void SetResponse(long sample,bool cut,Func<int,int,int,int,double> histogram,AcousticTake.State[] states,Func<int,int,uint> counts=null,Func<int,int,int,double> rates=null)
    {
        var raw=histogram;
        histogram=Exchange(sample,histogram,counts);
        if(initialized && sample<swap)throw new ArgumentException("Path responses must be ordered.");
        bool valid=histogram!=null;
        if(valid && sample%partition!=0)throw new ArgumentException("Path responses change on partition boundaries.");
        blend=initialized && !cut;bank^=1;swap=sample;initialized=true;computedBlock=-1;
        slideReference[bank]=sample+step;sliding[bank]=false;EchoBins=0;
        Array.Clear(echoes,bank*pairs*this.histogram.Count,pairs*this.histogram.Count);
        Array.Clear(slideRates,bank*pairs*this.histogram.Count,pairs*this.histogram.Count);
        long bankSize=(long)pairs*channels*Bands*count*bins;
        Array.Clear(kernels,(int)(bank*bankSize),(int)bankSize);
        int B=this.histogram.Count;
        bool room=valid && modes!=null && modes.Applies(states);ModalPairs=0;
        for(int pair=0;pair<pairs;pair++)
        {
            int listener=pair%listeners;
            active[bank*pairs+pair]=valid && scene.sources[pair/listeners].mutedListenerPlusOne!=listener+1 &&
                TakeRender.Enabled(states[scene.listeners[listener]],AcousticTake.StateFlags.Listener);
            if(!active[bank*pairs+pair])continue;
            double direct=0,delay=0;
            bool modal=room && modes.Pair(states[scene.sources[pair/listeners].objectIndex].position,states[scene.listeners[listener]].position,minimumDistance,modeAmplitudes,out direct,out delay);
            var share=modal?modes.Share:unitShare;
            Array.Clear(energy,0,energy.Length);
            for(int d=0;d<Directions;d++)for(int b=0;b<Bands;b++)for(int k=0;k<B;k++)
            {
                double e=histogram(pair,d,b,k)*share[b];if(!(e>0))continue;
                for(int c=0;c<channels;c++)energy[(c*Bands+b)*B+k]+=earWeights[(d*2+c)*Bands+b]*e;
            }
            var list=groups[pair];var gb=groupBins[pair];var gw=groupWeights[pair];var spectrum=spectra[pair];var mix=mixes[pair];
            int bins0=(bank*pairs+pair)*B;
            if(counts!=null && raw!=null)
            {
                bool slide=doppler && rates!=null;
                for(int k=0;k<B;k++)
                {
                    double e=0;for(int d=0;d<Directions;d++)for(int b=0;b<Bands;b++){double x=raw(pair,d,b,k);if(x>0)e+=x;}
                    rawEnergy[k]=e;moment1[k]=slide?rates(pair,0,k):0;moment2[k]=slide?rates(pair,1,k):0;binCounts[k]=counts(pair,k);
                }
                Vector3 Direction(int k)=>EchoDirection(d=>{double x=0;for(int b=0;b<Bands;b++){double v=raw(pair,d,b,k);if(v>0)x+=v;}return x;});
                SelectEchoes(B,k=>rawEnergy[k],slide?k=>moment1[k]:null,k=>moment2[k],k=>binCounts[k],Direction,echoes,slideRates,bins0);
                // An echo sits at its contributions' mean delay (without the rate channel, its bin's centre), with the
                // bin's ear energies per band.
                for(int k=0;k<B;k++)
                {
                    if(!echoes[bins0+k])continue;
                    double offset=rates!=null?rates(pair,2,k)/rawEnergy[k]:.5*this.histogram.Width(k);   // (referred to the update's time: it may lie a little outside the bin)
                    float tau=(float)((this.histogram.Start(k)+offset)*rate),half=itdGrid==null?0:.5f*rate*Hrtf.InterauralDelay(itdGrid,Direction(k));
                    for(int c=0;c<channels;c++)slideDelays[(bins0+k)*channels+c]=channels==2?(c==0?tau+half:tau-half):tau;
                    sliding[bank]=true;EchoBins++;
                    for(int c=0;c<channels;c++)for(int b=0;b<Bands;b++)
                    {double e=energy[(c*Bands+b)*B+k];slideAmplitudes[((bins0+k)*channels+c)*Bands+b]=e>0?(float)Math.Sqrt(e):0;}
                }
            }
            for(int c=0;c<channels;c++)for(int b=0;b<Bands;b++)
            {
                float a1=alpha[b],a2=channels==2?(c==0?beta[b]:-beta[b]):0;
                long baseKernel=(((long)(bank*pairs+pair)*channels+c)*Bands+b)*count*bins;
                for(int g=0;g<list.Count;g++)
                {
                    var group=list[g];double slotEnergy=0;
                    // (An echo's energy is its impulse's, SlideBlock; the kernel keeps the rest.)
                    for(int i=0;i<group.terms;i++){int k=gb[group.first+i];if(!echoes[bins0+k])slotEnergy+=energy[(c*Bands+b)*B+k]*gw[group.first+i];}
                    if(!(slotEnergy>0))continue;
                    float amplitude=(float)Math.Sqrt(slotEnergy);
                    int o=(g*Bands+b)*3;
                    float w1=amplitude*(a1*mix[o]+a2*mix[o+1]),w2=amplitude*(a1*mix[o+1]+a2*mix[o+2]);
                    long kernel=baseKernel+(long)group.partition*bins;int v1=(g*2)*bins,v2=v1+bins;
                    for(int i=0;i<bins;i++)kernels[kernel+i]+=spectrum[v1+i]*w1+spectrum[v2+i]*w2;
                }
            }
            if(modal){AddModes(pair,direct,delay);ModalPairs++;}
        }
    }
    // dh of one pair into every band's and ear's kernel, partition by partition (each P samples zero-padded to 2P).
    void AddModes(int pair,double direct,double delay)
    {
        modes.Synthesize(modeAmplitudes,direct,delay,modeResponse);
        for(int q=0;q<count;q++)
        {
            bool any=false;
            for(int n=0;n<partition;n++){float v=(float)modeResponse[q*partition+n];modePiece[n]=new Vector2(v,0);modePiece[partition+n]=Vector2.zero;any|=v!=0;}
            if(!any)continue;
            SpectralAudio.Fft(modePiece,false);
            for(int c=0;c<channels;c++)for(int b=0;b<Bands;b++)
            {
                long kernel=(((long)(bank*pairs+pair)*channels+c)*Bands+b)*count*bins+(long)q*bins;
                for(int i=0;i<bins;i++)kernels[kernel+i]+=modePiece[i];
            }
        }
    }
    // An echo's head-frame arrival direction from its bin's energy per direction channel (+x, -x, +y, -y, +z, -z):
    // exact for one arrival, whose energy the tracer splits by its squared components (PathDeposit). Zero if none.
    internal static Vector3 EchoDirection(Func<int,double> energy)
    {
        double e0=energy(0),e1=energy(1),e2=energy(2),e3=energy(3),e4=energy(4),e5=energy(5),total=e0+e1+e2+e3+e4+e5;
        if(!(total>0))return Vector3.zero;
        return new Vector3((float)((Math.Sqrt(e0)-Math.Sqrt(e1))/Math.Sqrt(total)),(float)((Math.Sqrt(e2)-Math.Sqrt(e3))/Math.Sqrt(total)),(float)((Math.Sqrt(e4)-Math.Sqrt(e5))/Math.Sqrt(total)));
    }
    // Adds bank b0's echoes to its output block: per sample a time-varying delay tau + r (t - t_ref) into the
    // band-filtered input, read with the direct path's interpolation filter. t_ref: the end of the bank's crossfade in,
    // where the response's state is (TakeRender.Live: the update's anchor is its sample + step - latency in take time,
    // and the output lags take time by the latency).
    void SlideBlock(int b0,int block)
    {
        EnsureHistory(block);
        var acc=new double[Bands];int mask=historyLength-1,B=histogram.Count;
        for(int l=0;l<listeners;l++)for(int s=0;s<sources;s++)
        {
            int pair=s*listeners+l;if(!active[b0*pairs+pair])continue;
            int bins0=(b0*pairs+pair)*B;
            for(int k=0;k<B;k++)
            {
                if(!echoes[bins0+k])continue;double r=slideRates[bins0+k];
                // Read speed 1 - r: reading faster than real time lowers the filter's cutoff (Doppler.Read).
                double cutoff=sinc?Math.Min(1,1/(1-r)):1;int extent=sinc?(int)Math.Ceiling(sincRadius/cutoff):1;
                for(int c=0;c<channels;c++)for(int n=0;n<partition;n++)
                {
                    long t=(long)block*partition+n;
                    // (The filter reads `extent` samples past its centre and the band input reaches t - latency, so a delay
                    // is at least that: only an echo within centimetres of its source is held there.)
                    double u=t-latency-Math.Max(extent,slideDelays[(bins0+k)*channels+c]+r*(t-slideReference[b0]));long center=(long)Math.Floor(u);double phase=u-center;
                    Array.Clear(acc,0,Bands);
                    if(!sinc)for(int b=0;b<Bands;b++){int row=(s*Bands+b)*historyLength;acc[b]=history[row+(int)(center&mask)]*(1-phase)+history[row+(int)((center+1)&mask)]*phase;}
                    else
                    {
                        double sum=0;
                        for(int j=1-extent;j<=extent;j++)
                        {
                            double x=(j-phase)*cutoff;if(Math.Abs(x)>=sincRadius)continue;
                            double w=SincValue(x)*SincValue(x/sincRadius)*cutoff;sum+=w;int index=(int)((center+j)&mask);
                            for(int b=0;b<Bands;b++)acc[b]+=w*history[(s*Bands+b)*historyLength+index];
                        }
                        for(int b=0;b<Bands;b++)acc[b]/=sum;
                    }
                    {
                        double value=0;int a=((bins0+k)*channels+c)*Bands;
                        for(int b=0;b<Bands;b++)value+=slideAmplitudes[a+b]*acc[b];
                        blocks[(b0*listeners*channels+l*channels+c)*partition+n]+=(float)value;
                    }
                }
            }
        }
    }
    static double SincValue(double x)=>Math.Abs(x)<1e-12?1:Math.Sin(Math.PI*x)/(Math.PI*x);
    // The band-filtered input of every source through band block `last` (block j: input samples [j P - latency, +P)),
    // as Half computes it.
    void EnsureHistory(long last)
    {
        long next=historyEnd,oldest=last+2-historyLength/partition;
        if(next==long.MinValue || next<oldest){Array.Clear(history,0,history.Length);next=Math.Max(oldest,next==long.MinValue?oldest:next);}
        var buffer=new Vector2[2*partition];var band=new Vector2[2*partition];int mask=historyLength-1;
        for(;next<=last;next++)
            for(int s=0;s<sources;s++)
            {
                long start=next*partition-latency+lookahead-partition;
                for(int n=0;n<2*partition;n++)buffer[n]=new Vector2(start+n<0?0:signals[s].Read(start+n),0);
                SpectralAudio.Fft(buffer,false);
                for(int b=0;b<Bands;b++)
                {
                    for(int i=0;i<2*partition;i++){var x=buffer[i];var h=filters[b*2*partition+i];band[i]=new Vector2(x.x*h.x-x.y*h.y,x.x*h.y+x.y*h.x);}
                    SpectralAudio.Fft(band,true);
                    long first=next*partition-latency;
                    for(int n=0;n<partition;n++)history[(s*Bands+b)*historyLength+(int)((first+n)&mask)]=band[partition+n].x;
                }
            }
        historyEnd=next;
    }
    // Band-filtered source half block (P samples) for UPOLS block `block`: x_b over
    // [block P - latency, block P + P - latency), from x read P/2 samples ahead.
    void Half(int s,long block)
    {
        long start=block*partition-latency+lookahead-partition; // overlap-save window of the FIR
        for(int n=0;n<2*partition;n++)work[n]=new Vector2(start+n<0?0:signals[s].Read(start+n),0);
        SpectralAudio.Fft(work,false);
        for(int b=0;b<Bands;b++)
        {
            for(int i=0;i<2*partition;i++){var x=work[i];var h=filters[b*2*partition+i];wide[i]=new Vector2(x.x*h.x-x.y*h.y,x.x*h.y+x.y*h.x);}
            SpectralAudio.Fft(wide,true);
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
                SpectralAudio.Fft(work,false);
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
                SpectralAudio.Fft(work,true);
                for(int n=0;n<partition;n++)blocks[(b0*listeners*channels+l*channels+c)*partition+n]=work[partition+n].x;
            }
            if(sliding[b0])SlideBlock(b0,block);
        }
        if(!both)return;
        for(int channel=0;channel<listeners*channels;channel++)
        {
            int a=(bank*listeners*channels+channel)*partition,o=((bank^1)*listeners*channels+channel)*partition;
            double pa=0,po=0,cross=0;
            for(int n=0;n<partition;n++){double x=blocks[a+n],y=blocks[o+n];pa+=x*x;po+=y*y;cross+=x*y;}
            powers[3*channel]=pa;powers[3*channel+1]=po;powers[3*channel+2]=cross;
        }
    }
    // Gain that gives the blend w new + (1 - w) old the power w P_new + (1 - w) P_old over the block.
    // Bounded at 4 (the blend's power can only vanish if the two outputs cancel, which shared signs
    // and non-negative amplitudes make unlikely).
    float BlendGain(int channel,float weight)
    {
        double pa=powers[3*channel],po=powers[3*channel+1],cross=powers[3*channel+2],w=weight,v=1-w;
        double want=w*pa+v*po,have=w*w*pa+v*v*po+2*w*v*cross;
        if(!(want>0))return 1;
        return have>want/16?(float)Math.Sqrt(want/have):4f;
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
                if(both && weight<1)value=(value+blocks[((bank^1)*listeners*channels+channel)*partition+at]*(1-weight))*BlendGain(channel,weight);
                pcm[n*listeners*channels+channel]+=value;
            }
        }
    }
}
