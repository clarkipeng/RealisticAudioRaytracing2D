using System;
using System.Collections.Generic;
using UnityEngine;

// GPU synthesis of the Path method's histogram reverb (PathReverb.cs is the CPU reference, with the same slot
// sequences, band filters, ear weights, interaural mix and power-corrected crossfade). PathReverb filters the
// source into eight bands and convolves each with its band's kernel; here the band filters are folded into one
// kernel per channel (convolution is linear, so the output is the same), which is built on the GPU for every
// update and convolved by the late field's partitioned overlap-save. Optionally each pair's histogram first
// goes through the persistent-ray denoiser (PathDenoiser), recorded into the same command buffer.
internal sealed partial class GpuAudio
{
    int pathEar,pathGroups,pathBand,pathPieces,pathPowers;
    bool pathMode,lateInitializedOnce;
    int pathBins,pathGroupCount;
    ComputeBuffer pathDenoised;
    PathDenoiser[] denoisers;
    public bool PathDenoising {get;set;}
    public bool HasPathDenoiser => denoisers!=null;

    void PreparePath(TakeRender.Settings settings,int rate)
    {
        pathMode=settings.method==TakeRender.Method.Path;
        foreach(var name in new[]{"PathHistogram","PathEarWeights","PathTermWeights","PathMix","PathEnergy","LatePowers"})Buffer(name,1,4);
        foreach(var name in new[]{"PathAlphaBeta","PathSampleValues","PathFilters","PathCoefficients","PathFiltered"})Buffer(name,1,8);
        Buffer("PathGroupInfo",1,16);Buffer("PathTermBins",1,4);Buffer("PathSampleGroup",1,4);
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
        var sampleGroup=new int[checked(pairs*lateCount*latePartition)];var sampleValues=new Vector2[sampleGroup.Length];
        for(int pair=0;pair<pairs;pair++)
        {
            var slots=PathReverb.BuildSlots(pair,histogram,lateRate,latePartition,lateCount,filters,false,true);
            int first=info.Count/4;
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
        Buffer("PathEnergy",checked(pairs*channels*TakeRender.Settings.LateBands*pathBins),4);
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
        lateReady=true;
    }

    // The persistent-ray denoiser, one per pair, writing into the histogram the reverb reads (PathFloat).
    public void EnablePathDenoiser(ComputeShader denoise,byte[] weights,string header)
    {
        if(!pathMode)throw new InvalidOperationException("The denoiser applies to the path method.");
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
        commands.SetComputeBufferParam(shader,pathEar,"LateHistogram",valid?method.LateBuffer:lateEmpty);
        Set("LateValid",valid?1:0);Set("LateBank",lateBank);Set("LateBlend",lateBlend?1:0);Clock("LateSwap",sample);
        Set("PathFloat",denoise?1:0);Set("PathFixedOffset",fixedOffset);Set("LateRays",valid?method.LateRays:1);
        commands.BeginSample("Acoustics.GPU.PathKernels");
        commands.DispatchCompute(shader,pathEar,(checked(pairs*channels*TakeRender.Settings.LateBands*pathBins)+63)/64,1,1);
        commands.DispatchCompute(shader,pathGroups,(checked(channels*TakeRender.Settings.LateBands*pathGroupCount)+63)/64,1,1);
        if(!SkipPathKernels || !lateInitializedOnce)
        {
            commands.DispatchCompute(shader,pathBand,checked(pairs*lateCount),1,1);
            commands.DispatchCompute(shader,pathPieces,checked(pairs*lateCount),1,1);
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
