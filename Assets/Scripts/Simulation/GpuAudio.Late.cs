using System;
using UnityEngine;

// GPU late-field synthesis; LateReverb.cs is the CPU reference with the same
// basis, envelopes, partitioned convolution and crossfade.
internal sealed partial class GpuAudio
{
    const int LateTiles=16;
    int lateEnvelope,lateExtend,lateBuild,lateInput,lateSum,lateInverse,lateWrite;
    int latePartition,lateCount,lateSimulated,lateBank,lateSupport,lateSources;
    long lateSwap;
    bool lateBlend,lateInitialized,lateReady;
    int lateRate;
    ComputeBuffer lateEmpty,lateEmptyRows;

    // Placeholders keep every late buffer bound; the basis and banks are allocated
    // with the first late field, so renders without one never build them.
    void PrepareLate(AcousticTake.Scene scene,int rate)
    {
        lateSources=signals.Length;lateRate=rate;
        lateEmpty=Buffer("LateHistogram",1,4);Buffer("LateEmissionClocks",1,16);Buffer("LateEmissionPoints",1,16);
        lateEmptyRows=Buffer("LateEmissionRanges",Math.Max(1,signals.Length),16);lateEmptyRows.SetData(new uint[Math.Max(1,signals.Length)*4]);
        foreach(var name in new[]{"LateBasisSpectra","LateKernels","LateInputs","LateSums"})Buffer(name,1,8);
        foreach(var name in new[]{"LateEarWeights","LateGains","LateBlocks"})Buffer(name,1,4);
        Buffer("LateOffsets",pairs,4).SetData(new uint[pairs]);Buffer("LateActive",2*pairs,4).SetData(new uint[2*pairs]);
        bool late=settings.transport==TakeRender.Transport.Raytraced && settings.maxBounces>0;
        latePartition=settings.LatePartition;lateCount=late?settings.LateKernelBins(rate):0;lateSimulated=late?settings.LateBins(rate):0;
        if(late)lateSupport=checked(lateCount*latePartition+settings.DspLatencySamples);
    }
    void EnsureLate()
    {
        if(lateReady)return;
        int bins=latePartition+1,lanes=listeners*channels;
        var basis=LateBasis.Get(lateRate,latePartition,lateCount,channels);
        Buffer("LateBasisSpectra",basis.Spectra.Length,8).SetData(basis.Spectra);
        Buffer("LateEarWeights",basis.EarWeights.Length,4).SetData(basis.EarWeights);
        var offsets=new uint[pairs];
        for(int p=0;p<pairs;p++)offsets[p]=AudioSourcePlane.Hash((uint)p*0x9e3779b9u+0x7f4a7c15u)%(uint)basis.Partitions;
        Buffer("LateOffsets",pairs,4).SetData(offsets);
        Buffer("LateGains",checked(pairs*channels*LateBasis.Bands*lateCount),4);
        Buffer("LateKernels",checked(2*pairs*channels*lateCount*bins),8).SetData(new Vector2[2*pairs*channels*lateCount*bins]);
        Buffer("LateInputs",checked(lateSources*lateCount*bins),8).SetData(new Vector2[lateSources*lateCount*bins]);
        Buffer("LateSums",checked(2*lanes*LateTiles*bins),8);
        Buffer("LateBlocks",checked(2*lanes*latePartition),4).SetData(new float[2*lanes*latePartition]);
        Set("LatePartition",latePartition);Set("LateCount",lateCount);Set("LateSimulated",lateSimulated);Set("LateBasisCount",basis.Partitions);Set("LateBins",bins);
        Set("LateTiles",LateTiles);Set("LateStep",settings.responseStepSamples);
        lateReady=true;
    }
    void SetLateResponse(long sample,bool cut,RayTransport.LateFrame late)
    {
        if(lateCount==0 || (!lateReady && !late.Valid))return;
        EnsureLate();
        if(lateInitialized && sample<lateSwap)throw new ArgumentException("Late responses must be ordered.");
        if(late.Valid && sample%latePartition!=0)throw new ArgumentException("Late responses change on partition boundaries.");
        lateBank^=1;lateBlend=lateInitialized && !cut;lateSwap=sample;lateInitialized=true;
        bool valid=late.Valid;var transport=late.Transport;
        foreach(int k in new[]{lateEnvelope,lateInput})
        {
            commands.SetComputeBufferParam(shader,k,"LateHistogram",valid?transport.LateBuffer:lateEmpty);
            // A late field without emission rows (energy method: point sources) reads the empty rows.
            commands.SetComputeBufferParam(shader,k,"LateEmissionRanges",valid?transport.EmissionRanges??lateEmptyRows:lateEmptyRows);
            commands.SetComputeBufferParam(shader,k,"LateEmissionPoints",valid?transport.EmissionPoints??lateEmpty:lateEmpty);
            commands.SetComputeBufferParam(shader,k,"LateEmissionClocks",valid?transport.EmissionClocks??lateEmpty:lateEmpty);
        }
        Set("LateValid",valid?1:0);Set("LateBank",lateBank);Set("LateBlend",lateBlend?1:0);Clock("LateSwap",sample);
        Set("LateHistogramOffset",valid?late.Frame*transport.LateUintsPerFrame:0);Set("LateRays",valid?transport.LateRays:1);
        Set("LateEmissionRow",valid && transport.Emissions!=null?late.Frame*transport.Emissions.FrameStride:0);
        commands.BeginSample("Acoustics.GPU.LateKernels");
        commands.DispatchCompute(shader,lateEnvelope,(checked(pairs*channels*LateBasis.Bands*lateCount)+63)/64,1,1);
        if(lateCount>lateSimulated)commands.DispatchCompute(shader,lateExtend,(pairs*LateBasis.Bands+63)/64,1,1);
        commands.DispatchCompute(shader,lateBuild,(checked(pairs*channels*lateCount*(latePartition+1))+63)/64,1,1);
        commands.EndSample("Acoustics.GPU.LateKernels");
    }
    // At each partition boundary: the sources' input spectra, then the current (and,
    // while fading, the previous) bank's block output.
    void QueueLateBlock(long now)
    {
        if(!lateReady || now%latePartition!=0)return;
        long block=now/latePartition;Set("LateBlock",checked((int)block));
        commands.BeginSample("Acoustics.GPU.Late");
        commands.DispatchCompute(shader,lateInput,lateSources,1,1);
        for(int which=0;which<(lateBlend && now<lateSwap+settings.responseStepSamples?2:1);which++)
        {
            Set("LateWhich",which);
            commands.DispatchCompute(shader,lateSum,(latePartition+1+63)/64,LateTiles,listeners*channels);
            commands.DispatchCompute(shader,lateInverse,listeners*channels,1,1);
        }
        commands.EndSample("Acoustics.GPU.Late");
    }
}
