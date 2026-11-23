using System;
using UnityEngine;

// GPU partitioned convolution of the path reverb (GpuAudio.Path builds its kernels; PathReverb.cs is the CPU
// reference with the same partitions and crossfade): each source's input spectra, the banks' block sums and
// inverse transforms, and the PCM write with the power-corrected crossfade.
internal sealed partial class GpuAudio
{
    const int LateTiles=16;
    int lateInput,lateSum,lateInverse,lateWrite;
    int latePartition,lateCount,lateSimulated,lateBank,lateSupport,lateSources;
    long lateSwap;
    bool lateBlend,lateInitialized,lateReady;
    int lateRate;
    ComputeBuffer lateEmpty;

    // Placeholders keep every late buffer bound; the banks are allocated with the first path response
    // (GpuAudio.Path), so renders without one never build them.
    void PrepareLate(int rate)
    {
        lateSources=signals.Length;lateRate=rate;
        lateEmpty=Buffer("LateHistogram",1,4);
        foreach(var name in new[]{"LateKernels","LateInputs","LateSums"})Buffer(name,1,8);
        Buffer("LateBlocks",1,4);
        Buffer("LateActive",2*pairs,4).SetData(new uint[2*pairs]);
        latePartition=settings.LatePartition;
    }
    // At each partition boundary: the sources' input spectra, then the current (and,
    // while fading, the previous) bank's block output.
    void QueueLateBlock(long now)
    {
        if(!lateReady || now%latePartition!=0 || SkipLateConvolution)return;
        long block=now/latePartition;Set("LateBlock",checked((int)block));
        commands.BeginSample("Acoustics.GPU.Late");
        commands.DispatchCompute(shader,lateInput,lateSources,1,1);
        // The block's band-filtered input into the ring (PathBandInput), and each bank's slid slots added to
        // its block (PathSlide; README part 40).
        Set("PathHistoryAt",(int)((now-settings.DspLatencySamples)&(pathHistoryLength-1)));commands.DispatchCompute(shader,pathBandInput,lateSources*TakeRender.Settings.LateBands,1,1);
        for(int which=0;which<(lateBlend && now<lateSwap+settings.responseStepSamples?2:1);which++)
        {
            Set("LateWhich",which);
            commands.DispatchCompute(shader,lateSum,(latePartition+1+63)/64,LateTiles,listeners*channels);
            commands.DispatchCompute(shader,lateInverse,listeners*channels,1,1);
            Set("PathSlideOffset",checked((int)(now-pathSlideReference[lateBank^which])));commands.DispatchCompute(shader,pathSlide,(listeners*channels*latePartition+63)/64,1,1);
        }
        if(lateBlend && now<lateSwap+settings.responseStepSamples)commands.DispatchCompute(shader,pathPowers,listeners*channels,1,1);
        commands.EndSample("Acoustics.GPU.Late");
    }
}
