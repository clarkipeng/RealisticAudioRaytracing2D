using System;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;

// GPU synthesis of the Path method's histogram reverb (PathReverb.cs is the CPU reference, with the same slot
// sequences, band filters, ear weights, interaural mix and power-corrected crossfade). PathReverb filters the
// source into eight bands and convolves each with its band's kernel; here the band filters are folded into one
// kernel per channel (convolution is linear, so the output is the same), which is built on the GPU for every
// update and convolved by the late field's partitioned overlap-save. Optionally each pair's histogram first
// goes through the persistent-ray denoiser (PathDenoiser), recorded into the same command buffer.
internal sealed partial class GpuAudio
{
    int pathGroups,pathFilter,pathPieces,pathPowers,pathSlideSelect,pathBandInput,pathSlide;
    // Doppler on reflections (PathReverb, README part 40): the band-input ring's length, and per bank the output sample
    // at which its slots sit at their histogram delays (the end of its crossfade in).
    int pathHistoryLength;
    readonly long[] pathSlideReference=new long[2];
    int pathFftCount,pathSpectralCount;
    ComputeBuffer pathFftBuffer,pathAllBuffer;
    internal bool PathGroupSpectraOff;   // profiling only (MeasurePathLiveBudget): every partition through the band FFTs
    internal int PathKernelSkip;         // profiling only (MeasurePathLiveBudget): skip PathGroups (1), the band-FFT partitions (2), the spectral ones (4), PathPieces (8)
    // Partitions with at most this many impulse groups are summed from the groups' spectra; -1: where that takes fewer
    // operations than the band FFTs (SpectralGroupLimit). Profiling only (part 29's rule was 2); set before the first response.
    internal int PathSpectralGroupLimit=-1;
    internal bool FftSingleStages { set => shader.SetInt("AudioFftSingle",value?1:0); }   // profiling only: one FFT stage per pass, as before part 37
    internal string PathPartitionGroups = ""; // diagnostics: how many partitions hold 0, 1, 2, ... impulse groups (all pairs)
    bool pathMode,lateInitializedOnce;
    int pathBins,pathGroupCount;
    ComputeBuffer pathDenoised;
    PathDenoiser[] denoisers;
    public bool PathDenoising {get;set;}
    public bool HasPathDenoiser => denoisers!=null;

    void PreparePath(TakeRender.Settings settings,int rate)
    {
        pathMode=settings.method==TakeRender.Method.Path;
        foreach(var name in new[]{"PathHistogram","PathEarWeights","PathTermWeights","PathMix","LatePowers"})Buffer(name,1,4);
        foreach(var name in new[]{"PathAlphaBeta","PathSampleValues","PathFilters","PathCoefficients","PathFiltered"})Buffer(name,1,8);
        foreach(var name in new[]{"PathSlideRates","PathSlideDelays","PathSlideAmplitudes","PathSlideEnd","PathBandHistory","PathBinCounts","PathBinStarts","PathItd"})Buffer(name,1,4);
        Buffer("PathGroupInfo",1,16);Buffer("PathTermBins",1,4);Buffer("PathSampleGroup",1,4);Buffer("PathFftPartitions",1,4);Buffer("PathSpectralPartitions",1,16);Buffer("PathSpectralGroups",1,4);Buffer("PathGroupSpectra",1,8);
        Set("LateAdvance",0);Set("LatePower",0);Set("PathFloat",0);
        if(!pathMode)return;
        var bins=new PathBins(settings);pathBins=bins.Count;
        latePartition=settings.LatePartition;lateCount=checked((int)Math.Ceiling(bins.Horizon*rate/latePartition)+1);lateSimulated=pathBins;
        lateSupport=checked(lateCount*latePartition+settings.DspLatencySamples);
    }

    // Slot layouts and constant tables, with the first response.
    void EnsurePath()
    {
        if(lateReady)return;
        int bins=latePartition+1,lanes=listeners*channels,size=2*latePartition;
        var histogram=new PathBins(settings);
        var filters=PathReverb.BandFilters(lateRate,latePartition);
        var info=new List<uint>();var termBins=new List<uint>();var termWeights=new List<float>();var mix=new List<float>();
        var fftPartitions=new List<uint>();var spectral=new List<uint>();var spectralGroups=new List<uint>();var groupSpectra=new List<Vector2>();
        int limit=PathSpectralGroupLimit>=0?PathSpectralGroupLimit:SpectralGroupLimit(latePartition,channels);
        var sampleGroup=new int[checked(pairs*lateCount*latePartition)];var sampleValues=new Vector2[sampleGroup.Length];
        for(int pair=0;pair<pairs;pair++)
        {
            var slots=PathReverb.BuildSlots(pair,histogram,lateRate,latePartition,lateCount,filters,true,true);
            int first=info.Count/4;
            // Partitions with at most `limit` impulse groups are summed from the groups' spectra (PathSpectralPartition).
            var members=new List<int>[lateCount];for(int q=0;q<lateCount;q++)members[q]=new List<int>();
            for(int g=0;g<slots.groups.Count;g++)members[slots.groups[g].partition].Add(g);
            for(int q=0;q<lateCount;q++)
            {
                uint entry=(uint)(pair*lateCount+q);
                if(members[q].Count>limit){fftPartitions.Add(entry);continue;}
                spectral.Add(entry);spectral.Add((uint)spectralGroups.Count);spectral.Add((uint)members[q].Count);spectral.Add((uint)(groupSpectra.Count/(2*bins)));
                foreach(int g in members[q]){spectralGroups.Add((uint)(first+g));groupSpectra.AddRange(new ArraySegment<Vector2>(slots.spectra,g*2*bins,2*bins));}
            }
            var census=new SortedDictionary<int,int>();foreach(var m in members)census[m.Count]=census.TryGetValue(m.Count,out int c)?c+1:1;
            PathPartitionGroups+=string.Join(" ",census.Select(kv=>$"{kv.Key}:{kv.Value}"))+$" (groups {slots.groups.Count}, terms {slots.bins.Count}); ";
            foreach(var g in slots.groups){info.Add((uint)pair);info.Add((uint)g.partition);info.Add((uint)(termBins.Count+g.first));info.Add((uint)g.terms);}
            for(int i=0;i<slots.bins.Count;i++){termBins.Add((uint)slots.bins[i]);termWeights.Add(slots.weights[i]);}
            mix.AddRange(slots.mix);
            int at=pair*lateCount*latePartition;
            for(int n=0;n<slots.sampleGroup.Length;n++)
            {
                sampleGroup[at+n]=slots.sampleGroup[n]<0?-1:first+slots.sampleGroup[n];
                sampleValues[at+n]=new Vector2(slots.s1[n],slots.s2[n]);
            }
        }
        pathGroupCount=info.Count/4;
        var groupInfo=info.ToArray();
        Buffer("PathGroupInfo",pathGroupCount,16).SetData(groupInfo);
        Buffer("PathTermBins",Math.Max(1,termBins.Count),4).SetData(termBins.Count>0?termBins.ToArray():new uint[1]);
        Buffer("PathTermWeights",Math.Max(1,termWeights.Count),4).SetData(termWeights.Count>0?termWeights.ToArray():new float[1]);
        Buffer("PathMix",Math.Max(1,mix.Count),4).SetData(mix.Count>0?mix.ToArray():new float[1]);
        Buffer("PathSampleGroup",sampleGroup.Length,4).SetData(sampleGroup);
        Buffer("PathSampleValues",sampleValues.Length,8).SetData(sampleValues);
        Buffer("PathFilters",filters.Length,8).SetData(filters);
        pathFftCount=fftPartitions.Count;pathSpectralCount=spectral.Count/4;
        pathFftBuffer=Buffer("PathFftPartitions",Math.Max(1,pathFftCount),4);pathFftBuffer.SetData(pathFftCount>0?fftPartitions.ToArray():new uint[1]);
        var all=new uint[pairs*lateCount];for(int i=0;i<all.Length;i++)all[i]=(uint)i;
        pathAllBuffer=new ComputeBuffer(all.Length,4);owned.Add(pathAllBuffer);pathAllBuffer.SetData(all);
        Buffer("PathSpectralPartitions",Math.Max(1,pathSpectralCount),16).SetData(pathSpectralCount>0?spectral.ToArray():new uint[4]);
        Buffer("PathSpectralGroups",Math.Max(1,spectralGroups.Count),4).SetData(spectralGroups.Count>0?spectralGroups.ToArray():new uint[1]);
        Buffer("PathGroupSpectra",Math.Max(1,groupSpectra.Count),8).SetData(groupSpectra.Count>0?groupSpectra.ToArray():new Vector2[1]);
        var ears=LateBasis.Get(lateRate,latePartition,1,channels).EarWeights;
        Buffer("PathEarWeights",ears.Length,4).SetData(ears);
        var alphaBeta=new Vector2[TakeRender.Settings.LateBands];
        var coherence=channels==2?LateBasis.Coherence(Hrtf.Load(lateRate),lateRate,2*latePartition):null;
        for(int b=0;b<alphaBeta.Length;b++)
        {
            double g=coherence==null?1:coherence[Math.Min(latePartition,(int)Math.Round(LateBasis.BandCentre(b)*2*latePartition/lateRate))];
            alphaBeta[b]=new Vector2((float)Math.Sqrt((1+g)/2),(float)Math.Sqrt((1-g)/2));
        }
        Buffer("PathAlphaBeta",alphaBeta.Length,8).SetData(alphaBeta);
        Buffer("PathCoefficients",checked(channels*TakeRender.Settings.LateBands*pathGroupCount),8);
        Buffer("PathFiltered",checked(pairs*lateCount*size),8);
        Buffer("LatePowers",3*lanes,4);
        Buffer("LateKernels",checked(2*pairs*channels*lateCount*bins),8).SetData(new Vector2[2*pairs*channels*lateCount*bins]);
        Buffer("LateInputs",checked(lateSources*lateCount*bins),8).SetData(new Vector2[lateSources*lateCount*bins]);
        Buffer("LateSums",checked(2*lanes*LateTiles*bins),8);
        Buffer("LateBlocks",checked(2*lanes*latePartition),4).SetData(new float[2*lanes*latePartition]);
        Set("LatePartition",latePartition);Set("LateCount",lateCount);Set("LateSimulated",lateSimulated);Set("LateBins",bins);
        Set("LateTiles",LateTiles);Set("LateStep",settings.responseStepSamples);
        Set("LateAdvance",latePartition/2);Set("LatePower",1);Set("PathGroupTotal",pathGroupCount);Set("PathBinsCount",pathBins);
        // Doppler: the slid bins and groups, their weights, and the band-filtered input over the horizon, one step of
        // drift either way, the interpolation's reach and two blocks (PathReverb's ring).
        int need=checked(lateCount*latePartition+2*settings.responseStepSamples+8*(settings.sincRadius+2)+4*latePartition);pathHistoryLength=1;while(pathHistoryLength<need)pathHistoryLength<<=1;
        Buffer("PathSlideRates",checked(2*pairs*pathBins),4).SetData(new float[2*pairs*pathBins]);
        Buffer("PathSlideDelays",checked(2*pairs*pathBins*channels),4).SetData(new float[2*pairs*pathBins*channels]);
        // The echoes' interaural delay (PathReverb, README part 42): the HRTF set's ITD over head-frame directions.
        var itd=channels==2?Hrtf.Load(lateRate).InterauralDelayGrid:new float[1];
        Buffer("PathItd",itd.Length,4).SetData(itd);
        Buffer("PathSlideAmplitudes",checked(2*pairs*pathBins*channels*TakeRender.Settings.LateBands),4);
        Buffer("PathSlideEnd",2*pairs,4).SetData(new uint[2*pairs]);
        Buffer("PathBandHistory",checked(lateSources*TakeRender.Settings.LateBands*pathHistoryLength),4).SetData(new float[lateSources*TakeRender.Settings.LateBands*pathHistoryLength]);
        var starts=new float[pathBins];for(int k=0;k<pathBins;k++)starts[k]=(float)(histogram.Start(k)*lateRate);
        Buffer("PathBinStarts",pathBins,4).SetData(starts);
        Set("PathHistoryLength",pathHistoryLength);Set("PathSincRadius",settings.sincRadius);Set("PathSincFilter",settings.delayFilter==TakeRender.DelayFilter.WindowedSinc?1:0);
        lateReady=true;
    }

    // The most impulse groups for which summing a partition from its groups' spectra takes no more operations than the
    // band FFTs: G groups x 8 bands x channels complex multiply-adds per bin (P + 1 bins) against 8 forward transforms
    // (P log2 2P butterflies each) and their 8 x 2P filter products; the inverse transform is common to both.
    // P = 1,024, two ears: 6.
    internal static int SpectralGroupLimit(int partition,int channels)
    {
        double transforms=partition*Math.Log(2.0*partition,2)+2.0*partition;
        return (int)Math.Floor(transforms/(channels*(partition+1.0)));
    }

    // The persistent-ray denoiser, one per pair, writing into the histogram the reverb reads (PathFloat).
    public void EnablePathDenoiser(ComputeShader denoise,byte[] weights,string header)
    {
        if(!pathMode)throw new InvalidOperationException("The denoiser applies to the path method.");
        if(denoisers!=null)foreach(var d in denoisers)d?.Dispose();   // (replacing the bundled weights: measurements)
        var bins=new PathBins(settings);var starts=new double[bins.Count];for(int k=0;k<starts.Length;k++)starts[k]=bins.Start(k);
        pathDenoised=Buffer("PathHistogram",checked(pairs*PathDenoiser.Cells*bins.Count),4);
        denoisers=new PathDenoiser[pairs];
        for(int p=0;p<pairs;p++)denoisers[p]=new PathDenoiser(denoise,weights,header,starts,pathDenoised,p*PathDenoiser.Cells*bins.Count);
        PathDenoising=true;
    }

    void SetPathResponse(long sample,bool cut,RayTransport.LateFrame late)
    {
        var method=late.Transport as MethodTransport;
        bool valid=method!=null && method.IsPath;
        if(!lateReady && !valid)return;
        EnsurePath();
        if(lateInitialized && sample<lateSwap)throw new ArgumentException("Path responses must be ordered.");
        if(valid && sample%latePartition!=0)throw new ArgumentException("Path responses change on partition boundaries.");
        lateBank^=1;lateBlend=lateInitialized && !cut;lateSwap=sample;lateInitialized=true;
        pathSlideReference[lateBank]=sample+settings.responseStepSamples;
        int fixedOffset=valid?late.Frame*method.PathUintsPerFrame/2:0;
        bool denoise=valid && denoisers!=null && PathDenoising;
        if(denoisers!=null && valid)
        {
            // Every update goes through the denoisers (they keep a history), used or not; a cut restarts them.
            for(int p=0;p<pairs;p++)
            {
                if(cut)denoisers[p].Reset();
                denoisers[p].Record(commands,method.LateBuffer,fixedOffset+p*PathDenoiser.Cells*pathBins,method.CountsBuffer,(late.Frame*pairs+p)*pathBins,method.LateRays);
            }
        }
        commands.SetComputeBufferParam(shader,pathGroups,"LateHistogram",valid?method.LateBuffer:lateEmpty);
        commands.SetComputeBufferParam(shader,pathSlideSelect,"LateHistogram",valid?method.LateBuffer:lateEmpty);
        if(valid)commands.SetComputeBufferParam(shader,pathSlideSelect,"PathBinCounts",method.CountsBuffer);
        Set("PathRateOffset",valid?method.PathRateOffset(late.Frame):0);Set("PathCountOffset",valid?late.Frame*pairs*pathBins:0);
        Set("LateValid",valid?1:0);Set("LateBank",lateBank);Set("LateBlend",lateBlend?1:0);Clock("LateSwap",sample);
        Set("PathFloat",denoise?1:0);Set("PathFixedOffset",fixedOffset);Set("LateRays",valid?method.LateRays:1);
        commands.BeginSample("Acoustics.GPU.PathKernels");
        Set("PathDoppler",settings.pathReflectionDoppler?1:0);
        commands.DispatchCompute(shader,pathSlideSelect,(pairs+63)/64,1,1);
        if((PathKernelSkip&1)==0)
            commands.DispatchCompute(shader,pathGroups,(Math.Max(pairs,checked(channels*TakeRender.Settings.LateBands*pathGroupCount))+63)/64,1,1);
        if(!SkipPathKernels || !lateInitializedOnce)
        {
            // One dispatch: the band-FFT partitions, then the spectral ones (PathFilter).
            int fft=PathGroupSpectraOff?pairs*lateCount:(PathKernelSkip&2)!=0?0:pathFftCount,spectralCount=PathGroupSpectraOff||(PathKernelSkip&4)!=0?0:pathSpectralCount;
            commands.SetComputeBufferParam(shader,pathFilter,"PathFftPartitions",PathGroupSpectraOff?pathAllBuffer:pathFftBuffer);
            Set("PathFftDispatch",fft);Set("PathSpectralFirst",0);
            if(fft+spectralCount>0)commands.DispatchCompute(shader,pathFilter,fft+spectralCount,1,1);
            if((PathKernelSkip&8)==0)commands.DispatchCompute(shader,pathPieces,checked(pairs*lateCount),1,1);
        }
        lateInitializedOnce=true;
        commands.EndSample("Acoustics.GPU.PathKernels");
    }

    void DisposePath()
    {
        if(denoisers!=null)foreach(var d in denoisers)d?.Dispose();
        denoisers=null;
    }
}
