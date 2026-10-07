using NUnit.Framework;
using System;
using System.Collections;
using UnityEngine.TestTools;
using UnityEngine;

public partial class TakeChecks
{
    [TestCase(false,TakeRender.Synthesis.ImpulseResponse),TestCase(true,TakeRender.Synthesis.ImpulseResponse)]
    [TestCase(false,TakeRender.Synthesis.Spectral),TestCase(true,TakeRender.Synthesis.Spectral),Category("GPU")]
    public void ResponseChangesMatchContinuousOutputCrossfade(bool accelerated,TakeRender.Synthesis method)
    {
        RequireGpu();const int rate=8192,step=128,change=step*3;
        var scene=Scene(2);var before=States(2);var after=States(2);
        after[1].rotation=Quaternion.Euler(0,90,0);
        var settings=SpectralSettings();settings.synthesis=method;settings.raysPerBin=1;settings.maxBounces=1;
        settings.listenerOutput=TakeRender.ListenerOutput.Binaural;settings.transitionSamples=64;
        var clip=new float[256];for(int i=0;i<clip.Length;i++)clip[i]=.2f*Mathf.Sin(2*Mathf.PI*i/32+.37f);
        var signal=new TakeRender.Signal(clip);signal.SetActive(0,true);
        signal.Append(new AcousticTake.VoiceEvent{gain=1,loop=true});var signals=new[]{signal};
        var oldPaths=new RayTransport.Path[2*settings.SpectralSlotsPerPair];
        for(int listener=0;listener<2;listener++)for(int bin=0;bin<settings.FrequencyBins;bin++)
            oldPaths[listener*settings.SpectralSlotsPerPair+bin]=new RayTransport.Path
                {frequencyBin=(uint)bin,pressureGain=1,direction=Vector3.right};
        var newPaths=(RayTransport.Path[])oldPaths.Clone();
        for(int bin=0;bin<settings.FrequencyBins;bin++){newPaths[bin].pressureGain=.25f;newPaths[bin].delaySeconds=3f/rate;}
        var direct=new TakeRender.DirectPath[2];
        var oldResponse=new TakeRender.Mixer(scene,rate,settings,signals);
        var newResponse=new TakeRender.Mixer(scene,rate,settings,signals);
        oldResponse.SetResponse(0,direct,before,false,oldPaths,0);newResponse.SetResponse(0,direct,after,false,newPaths,0);
        using var gpu=accelerated?new GpuAudio(scene,rate,settings,signals):null;
        var cpu=accelerated?null:new TakeRender.Mixer(scene,rate,settings,signals);
        using var arrivals=new ComputeBuffer(oldPaths.Length,32);
        var a=new float[step*4];var b=new float[a.Length];var actual=new float[a.Length];double error=0,energy=0;
        for(int sample=0;sample<step*7;sample+=step)
        {
            var paths=sample<change?oldPaths:newPaths;var states=sample<change?before:after;
            arrivals.SetData(paths);
            if(accelerated)gpu.SetResponse(sample,false,arrivals,0,scene,states);else cpu.SetResponse(sample,direct,states,false,paths,0);
            oldResponse.Render(sample,step,a);newResponse.Render(sample,step,b);
            int offset=0;
            foreach(int count in new[]{13,step-13})
            {
                if(accelerated)gpu.Render(sample+offset,count,actual);else cpu.Render(sample+offset,count,actual);
                for(int n=0;n<count;n++)for(int channel=0;channel<4;channel++)
                {
                    float blend=Mathf.Clamp01((sample+offset+n-change)/(float)settings.transitionSamples);
                    int index=(offset+n)*4+channel;float expected=Mathf.Lerp(a[index],b[index],blend);
                    error=Math.Max(error,Math.Abs(actual[n*4+channel]-expected));energy+=expected*expected;
                }
                offset+=count;
            }
        }
        Assert.That(energy,Is.GreaterThan(.01));
        Assert.That(error,Is.LessThan(2e-5),"Response/head changes must follow the sample-wise fade, including partial hops and the unchanged second listener.");
    }

    [TestCase(TakeRender.Synthesis.ImpulseResponse, TakeRender.ListenerOutput.Mono, TakeRender.DelayFilter.Linear, 128, 1)]
    [TestCase(TakeRender.Synthesis.ImpulseResponse, TakeRender.ListenerOutput.Mono, TakeRender.DelayFilter.WindowedSinc, 128, 1)]
    [TestCase(TakeRender.Synthesis.ImpulseResponse, TakeRender.ListenerOutput.Binaural, TakeRender.DelayFilter.Linear, 128, 1)]
    [TestCase(TakeRender.Synthesis.ImpulseResponse, TakeRender.ListenerOutput.Binaural, TakeRender.DelayFilter.WindowedSinc, 128, 1)]
    [TestCase(TakeRender.Synthesis.Spectral, TakeRender.ListenerOutput.Mono, TakeRender.DelayFilter.Linear, 128, 1)]
    [TestCase(TakeRender.Synthesis.Spectral, TakeRender.ListenerOutput.Mono, TakeRender.DelayFilter.WindowedSinc, 128, 1)]
    [TestCase(TakeRender.Synthesis.Spectral, TakeRender.ListenerOutput.Binaural, TakeRender.DelayFilter.Linear, 128, 1)]
    [TestCase(TakeRender.Synthesis.Spectral, TakeRender.ListenerOutput.Binaural, TakeRender.DelayFilter.WindowedSinc, 128, 1)]
    [TestCase(TakeRender.Synthesis.Spectral, TakeRender.ListenerOutput.Mono, TakeRender.DelayFilter.Linear, 2048, 1)]
    [TestCase(TakeRender.Synthesis.Spectral, TakeRender.ListenerOutput.Binaural, TakeRender.DelayFilter.Linear, 2048, 1)]
    [TestCase(TakeRender.Synthesis.Spectral, TakeRender.ListenerOutput.Mono, TakeRender.DelayFilter.Linear, 128, 32)]
    [Category("GPU")]
    public void GpuAudioParity(TakeRender.Synthesis method,TakeRender.ListenerOutput ears,TakeRender.DelayFilter delay,int size,int rays)
    {
        var shader=RequireGpu();
        const int rate=8192;var scene=Scene(2);var state=States(2);
        scene.sources=new[]{scene.sources[0],new AcousticTake.SourceDefinition{objectIndex=2,clipIndex=0}};
        state[2].flags|=AcousticTake.StateFlags.Source;state[0].position=new Vector3(-2,0,0);
        var settings=new TakeRender.Settings{transport=TakeRender.Transport.Raytraced,synthesis=method,listenerOutput=ears,delayFilter=delay,
            fftSize=size,spectralHop=size/4,responseStepSamples=size,transitionSamples=size/4,maximumDelaySeconds=.2f,airAbsorption=true,
            raysPerBin=rays,maxBounces=1,speedOfSound=343,minimumDistance=1};
        var clip=new float[257];for(int n=0;n<clip.Length;n++)clip[n]=.2f*Mathf.Sin(n*.37f)+.1f*Mathf.Cos(n*.93f);
        var signals=new[]{new TakeRender.Signal(clip),new TakeRender.Signal(clip)};
        for(int s=0;s<2;s++){signals[s].SetActive(0,true);signals[s].Append(new AcousticTake.VoiceEvent{action=AcousticTake.VoiceAction.Play,gain=1-s*.3f,clipOffset=s*31,loop=true});}
        using var gpu=new GpuAudio(scene,rate,settings,signals);var cpu=new TakeRender.Mixer(scene,rate,settings,signals);
        using var solver=new TransportSolver(scene,Array.Empty<AcousticTake.Primitive>(),settings,shader,sampleRate:rate);
        var direct=new TakeRender.DirectPath[4];var a=new float[size*2*settings.ChannelsPerListener];var b=new float[a.Length];
        double peak=0,error=0,energy=0;
        for(int frame=0;frame<12;frame++)
        {
            long sample=frame*(long)size;bool cut=frame==7;
            if(frame<5 || frame>=7){state[1].position.x=1+frame*.07f;state[1].rotation=Quaternion.Euler(0,frame*19,0);}
            state[0].velocity=Vector3.right*12;state[0].flags|=AcousticTake.StateFlags.VelocityValid;
            if(frame==8)state[1].flags&=~AcousticTake.StateFlags.Active;
            if(frame==9)state[1].flags|=AcousticTake.StateFlags.Active;
            if(frame==9)signals[0].Append(new AcousticTake.VoiceEvent{sample=sample+17,action=AcousticTake.VoiceAction.Stop});
            solver.Solve(state,direct);var paths=solver.RayPaths;
            cpu.SetResponse(sample,direct,state,cut,paths,0);gpu.SetResponse(sample,cut,solver.RayBuffer,0,scene,state);
            // Non-hop-aligned delivery exercises the saved output block and source ring.
            int at=0;foreach(int count in new[]{13,size-13})
            {
                cpu.Render(sample+at,count,a);gpu.Render(sample+at,count,b);
                for(int n=0;n<count*2*settings.ChannelsPerListener;n++)
                {double d=a[n]-b[n];peak=Math.Max(peak,Math.Abs(d));error+=d*d;energy+=a[n]*a[n];}at+=count;
            }
        }
        Check(energy>1e-6,"GPU parity fixture contains audible output.");
        double relative=Math.Sqrt(error/energy);
        Check(peak<2e-4 && relative<5e-4,"GPU/CPU "+method+" "+ears+" "+delay+" FFT "+size+" peak="+peak+" relative="+relative);
        Assert.That(gpu.Diagnostics.responseUpdates,Is.EqualTo(12),"Partial PCM reads must not count a response twice.");
        Assert.That(gpu.Diagnostics.reusedResponses,Is.EqualTo(2),"The two stationary responses reuse filters; turns, cuts and activity changes rebuild them.");
        Assert.That(gpu.Diagnostics.peakGroups,Is.InRange(1,gpu.Diagnostics.pathCapacity));
    }
    [UnityTest, Category("GPU"), Category("Integration")]
    public IEnumerator GpuReadbackOwnsOutputUntilConsumed()
    {
        RequireGpu();
        var scene=Scene();var state=States();
        var settings=SpectralSettings();settings.audioBackend=TakeRender.AudioBackend.Gpu;
        settings.raysPerBin=1;settings.maxBounces=1;settings.airAbsorption=false;
        settings.responseStepSamples=settings.DspLatencySamples+settings.spectralHop;
        var signal=new TakeRender.Signal(new[]{1f});signal.SetActive(0,true);
        signal.Append(new AcousticTake.VoiceEvent {action=AcousticTake.VoiceAction.Play,gain=1,loop=true});
        var paths=new RayTransport.Path[settings.SpectralSlotsPerPair];
        for(int bin=0;bin<settings.FrequencyBins;bin++)paths[bin]=new RayTransport.Path {frequencyBin=(uint)bin,pressureGain=1};
        using var buffer=new ComputeBuffer(paths.Length,32);buffer.SetData(paths);
        using var gpu=new GpuAudio(scene,8192,settings,new[]{signal});
        var pcm=new float[settings.responseStepSamples];
        for(int batch=0;batch<2;batch++)
        {
            long sample=batch*(long)pcm.Length;
            gpu.SetResponse(sample,false,buffer,0,scene,state);gpu.QueueRender(sample,pcm.Length);gpu.BeginReadback();
            Assert.Throws<InvalidOperationException>(()=>gpu.QueueRender(sample+pcm.Length,pcm.Length));
            Assert.Throws<InvalidOperationException>(()=>gpu.SetResponse(sample+pcm.Length,false,buffer,0,scene,state));
            Assert.Throws<InvalidOperationException>(gpu.BeginReadback);
            Assert.Throws<InvalidOperationException>(()=>gpu.PrepareInput(sample+pcm.Length));
            double deadline=UnityEditor.EditorApplication.timeSinceStartup+10;
            while(!gpu.TryComplete(pcm))
            {
                Assert.That(UnityEditor.EditorApplication.timeSinceStartup,Is.LessThan(deadline),"GPU readback timed out.");
                yield return null;
            }
            Assert.That(gpu.IsPending,Is.False);
            Assert.Throws<InvalidOperationException>(()=>gpu.TryComplete(pcm));
            double energy=0;foreach(float sampleValue in pcm){Assert.That(float.IsFinite(sampleValue),Is.True);energy+=sampleValue*sampleValue;}
            Assert.That(energy,Is.GreaterThan(.01),"Both independently owned batches must contain audio.");
        }
    }

    [Test,Category("GPU")]
    public void GpuBatchUploadsOnlyFinalizedSourceSamples()
    {
        RequireGpu();var scene=Scene();var state=States();var settings=SpectralSettings();
        settings.raysPerBin=1;settings.maxBounces=1;
        int step=settings.responseStepSamples;
        using var stream=new SourceStream(step*2);
        var signal=new TakeRender.Signal(stream);signal.SetActive(0,true);
        signal.Append(new AcousticTake.VoiceEvent{gain=1});
        var paths=new RayTransport.Path[settings.SpectralSlotsPerPair];
        for(int bin=0;bin<settings.FrequencyBins;bin++)paths[bin]=new RayTransport.Path{frequencyBin=(uint)bin,pressureGain=1};
        using var arrivals=new ComputeBuffer(paths.Length,32);arrivals.SetData(paths);
        using var gpu=new GpuAudio(scene,8192,settings,new[]{signal},maximumResponses:2);
        gpu.SetResponse(0,false,arrivals,0,scene,state);
        long before=gpu.UploadedBytes;
        Assert.Throws<InvalidOperationException>(()=>gpu.QueueRender(0,step));
        var input=new float[step];Array.Fill(input,.25f);stream.Append(input,0,step,step);
        Assert.Throws<InvalidOperationException>(()=>gpu.PrepareInput(step*2));
        Assert.That(gpu.UploadedBytes,Is.EqualTo(before),"Reject incomplete batches before uploading any samples.");
        stream.Append(input,0,step,step*2);gpu.PrepareInput(step*2);
        stream.Prune(step*2); // Queued work owns the uploaded copy, not the source history.
        gpu.QueueRender(0,step);gpu.QueueRender(step,step);gpu.BeginReadback();
        var pcm=new float[step*2];Assert.That(gpu.TryComplete(pcm,true),Is.True);
        Assert.That(gpu.UploadedBytes-before,Is.EqualTo(step*2*4),"Upload the batch once, even when rendered in pieces.");
        double energy=0;foreach(float sample in pcm){Assert.That(float.IsFinite(sample),Is.True);energy+=sample*sample;}
        Assert.That(energy,Is.GreaterThan(.01));
    }

    [Test,Category("GPU")]
    public void UnchangedLargeResponseDoesNotRunFilterFfts()
    {
        RequireGpu();var scene=Scene();var state=States();var settings=SpectralSettings();
        settings.fftSize=2048;settings.spectralHop=512;settings.responseStepSamples=2048;settings.raysPerBin=1;settings.maxBounces=1;
        var paths=new RayTransport.Path[settings.SpectralSlotsPerPair];
        paths[0]=new RayTransport.Path{pressureGain=1};
        using var arrivals=new ComputeBuffer(paths.Length,32);arrivals.SetData(paths);
        using var gpu=new GpuAudio(scene,8192,settings,new[]{new TakeRender.Signal(new[]{0f})});
        gpu.SetResponse(0,false,arrivals,0,scene,state);gpu.Render(0,2048,new float[2048]);
        var fields=new[]{"work","transform"};var buffers=new ComputeBuffer[fields.Length];
        var sentinel=new Vector2[8];for(int i=0;i<sentinel.Length;i++)sentinel[i]=new Vector2(i+1,-i-1);
        for(int i=0;i<fields.Length;i++)
        {
            buffers[i]=(ComputeBuffer)typeof(GpuAudio).GetField(fields[i],System.Reflection.BindingFlags.Instance|System.Reflection.BindingFlags.NonPublic).GetValue(gpu);
            buffers[i].SetData(sentinel,0,0,sentinel.Length);
        }
        gpu.SetResponse(2048,false,arrivals,0,scene,state);
        foreach(var buffer in buffers)
        {var actual=new Vector2[sentinel.Length];buffer.GetData(actual,0,0,actual.Length);Assert.That(actual,Is.EqualTo(sentinel),"Neither inverse nor forward filter FFT may touch scratch for an unchanged response.");}
    }

}
