using NUnit.Framework;
using System;
using System.Collections;
using UnityEngine.TestTools;
using UnityEngine;

public partial class TakeChecks
{
    [TestCase(128,TakeRender.ListenerOutput.Mono),TestCase(128,TakeRender.ListenerOutput.Binaural)]
    [TestCase(2048,TakeRender.ListenerOutput.Binaural),Category("GPU")]
    public void GpuPathReductionMatchesReference(int size,TakeRender.ListenerOutput ears)
    {
        RequireGpu();const int rate=8192;var scene=Scene(2);var state=States(2);
        scene.sources=new[]{scene.sources[0],new AcousticTake.SourceDefinition{objectIndex=2,clipIndex=0}};
        var settings=SpectralSettings();settings.fftSize=size;settings.spectralHop=size/4;settings.responseStepSamples=size;
        settings.raysPerBin=1;settings.maxBounces=1;settings.listenerOutput=ears;
        var clip=new float[257];for(int i=0;i<clip.Length;i++)clip[i]=.2f*Mathf.Sin(i*.37f);
        var signal=new TakeRender.Signal(clip);signal.SetActive(0,true);signal.Append(new AcousticTake.VoiceEvent{gain=1,loop=true});
        var signals=new[]{signal,signal};var paths=new RayTransport.Path[4*settings.SpectralSlotsPerPair];
        using var arrivals=new ComputeBuffer(paths.Length,RayTransport.PathStride);
        using var gpu=new GpuAudio(scene,rate,settings,signals);var cpu=new TakeRender.Mixer(scene,rate,settings,signals);
        var a=new float[size*2*settings.ChannelsPerListener];var b=new float[a.Length];double error=0,energy=0;
        for(int frame=0;frame<6;frame++)
        {
            // Unequal, non-tile-sized path sets, cancellation, and a fully empty response.
            Array.Clear(paths,0,paths.Length);
            if(frame!=3)for(int pair=0;pair<4;pair++)for(int p=0;p<33-pair;p++)
                paths[pair*settings.SpectralSlotsPerPair+p]=new RayTransport.Path{identity=(uint)p,frequencyBin=(uint)(p%17),
                    pressureGain=(p%2==0?.5f:-.3f)/(pair+1),pressureImag=.2f/(pair+1),delaySeconds=(p+frame)/(float)rate,direction=new Vector4(1,0,0,.01f)};
            state[1].rotation=Quaternion.Euler(0,frame*19,0);state[1].angularVelocity=Vector3.up;
            state[1].flags|=AcousticTake.StateFlags.VelocityValid;
            arrivals.SetData(paths);long sample=frame*(long)size;
            cpu.SetResponse(sample,new TakeRender.DirectPath[4],state,false,paths,0);gpu.SetResponse(sample,false,arrivals,0,scene,state);
            cpu.Render(sample,size,a);gpu.Render(sample,size,b);
            for(int n=0;n<a.Length;n++){double d=a[n]-b[n];error+=d*d;energy+=a[n]*a[n];}
        }
        Assert.That(energy,Is.GreaterThan(1e-6));
        Assert.That(Math.Sqrt(error/energy),Is.LessThan(5e-4));
    }

    [TestCase(128),TestCase(2048),Category("GPU")]
    public void AllMutedPairsNeedNoMaterialFilters(int size)
    {
        RequireGpu();var scene=Scene();scene.sources[0].mutedListenerPlusOne=1;
        var settings=SpectralSettings();settings.fftSize=size;settings.spectralHop=size/4;settings.responseStepSamples=size;
        settings.raysPerBin=1;settings.maxBounces=1;
        var paths=new RayTransport.Path[settings.SpectralSlotsPerPair];paths[0]=new RayTransport.Path{pressureGain=1};
        using var arrivals=new ComputeBuffer(paths.Length,RayTransport.PathStride);arrivals.SetData(paths);
        using var gpu=new GpuAudio(scene,8192,settings,new[]{new TakeRender.Signal(new[]{1f})});
        var pcm=new float[size];gpu.SetResponse(0,false,arrivals,0,scene,States());gpu.Render(0,size,pcm);
        Assert.That(pcm,Is.All.Zero);Assert.That(gpu.Diagnostics.activeGroups,Is.Zero);
        Assert.That(gpu.Diagnostics.materialFilters,Is.Zero);Assert.That(gpu.Diagnostics.filterBytes,Is.EqualTo(8),"Only the required empty-buffer sentinel remains.");
    }

    [Test, Category("GPU")]
    public void LiveGpuKeepsBatchingWhenPlaybackFallsBehind()
    {
        var shader=RequireGpu();const int rate=8192,lead=1024,end=4096;
        var settings=SpectralSettings();settings.audioBackend=TakeRender.AudioBackend.Gpu;settings.maximumResponsesPerBatch=8;
        var queue=new AudioManager.PcmQueue(8192,rate);queue.Publish(0,null,lead);
        queue.Consume(new float[lead+end],1); // Simulate a stalled producer without changing sample identities.
        using var live=new TakeRender.Live(Scene(),Array.Empty<AcousticTake.Primitive>(),new[]{new[]{.1f}},rate,settings,queue,lead,0,shader);
        var state=States();live.Prepare(state);live.Append(new AcousticTake.VoiceEvent{gain=1,loop=true});
        live.Observe(0,end,state,state);live.Pump(true);
        Assert.That(live.Cursor,Is.EqualTo(end));
        Assert.That(live.GpuDiagnostics.readbacks,Is.EqualTo(end/lead),"A missed deadline must not collapse catch-up work into one-response readbacks.");
    }

    [TestCase(false,128,AcousticTake.StateFlags.VelocityValid),TestCase(true,128,AcousticTake.StateFlags.VelocityValid)]
    [TestCase(true,2048,AcousticTake.StateFlags.VelocityValid)]
    [TestCase(true,128,(AcousticTake.StateFlags)0),TestCase(true,128,AcousticTake.StateFlags.VelocityValid|AcousticTake.StateFlags.Cut),Category("GPU")]
    public void HeadPoseMatchesSampleTimedHrir(bool accelerated,int size,AcousticTake.StateFlags motion)
    {
        RequireGpu();const int rate=8192,step=256,delay=31;const float speed=360;
        var scene=Scene();var states=States();var settings=SpectralSettings();
        settings.fftSize=size;settings.spectralHop=16;settings.responseStepSamples=step;settings.transitionSamples=64;
        settings.listenerOutput=TakeRender.ListenerOutput.Binaural;settings.raysPerBin=1;settings.maxBounces=1;
        var initial=Quaternion.Euler(15,170,-20);var axis=new Vector3(.3f,1,.2f).normalized;var direction=Vector3.forward;
        float turn=motion==AcousticTake.StateFlags.VelocityValid?speed:0;
        var clip=new float[256];
        for(int n=0;n<clip.Length;n++)clip[n]=.2f*Mathf.Sin(2*Mathf.PI*n/8);
        var signal=new TakeRender.Signal(clip);signal.SetActive(0,true);signal.Append(new AcousticTake.VoiceEvent{gain=1,loop=true});
        var paths=new RayTransport.Path[settings.SpectralSlotsPerPair];
        for(int bin=0;bin<settings.FrequencyBins;bin++)paths[bin]=new RayTransport.Path{direction=direction,frequencyBin=(uint)bin,pressureGain=1,delaySeconds=delay/(float)rate};
        using var arrivals=new ComputeBuffer(paths.Length,RayTransport.PathStride);arrivals.SetData(paths);
        using var gpu=accelerated?new GpuAudio(scene,rate,settings,new[]{signal}):null;
        var cpu=accelerated?null:new TakeRender.Mixer(scene,rate,settings,new[]{signal});var direct=new TakeRender.DirectPath[1];
        var hrtf=Hrtf.Load(rate);var pcm=new float[step*2];double error=0,energy=0;
        for(int sample=0;sample<rate;sample+=step)
        {
            states[1].rotation=Quaternion.AngleAxis(turn*sample/rate,axis)*initial;
            states[1].angularVelocity=axis*(speed*Mathf.Deg2Rad);states[1].flags|=motion;
            if(accelerated)gpu.SetResponse(sample,false,arrivals,0,scene,states);else cpu.SetResponse(sample,direct,states,false,paths,0);
            int offset=0;
            foreach(int count in new[]{13,step-13})
            {
                if(accelerated)gpu.Render(sample+offset,count,pcm);else cpu.Render(sample+offset,count,pcm);
                for(int n=0;n<count;n++)
                {
                    int at=sample+offset+n;if(at<size+delay+hrtf.Taps)continue;
                    var rotation=Quaternion.AngleAxis(turn*at/rate,axis)*initial;
                    var blend=hrtf.Interpolate(Quaternion.Inverse(rotation)*direction);
                    for(int ear=0;ear<2;ear++)
                    {
                        double expected=0;
                        for(int tap=0;tap<hrtf.Taps;tap++)expected+=hrtf.Coefficient(blend,ear,tap)*.2*Math.Sin(2*Math.PI*(at-settings.DspLatencySamples-delay-tap)/8);
                        double d=pcm[n*2+ear]-expected;error+=d*d;energy+=expected*expected;
                    }
                }
                offset+=count;
            }
        }
        Assert.That(energy,Is.GreaterThan(1));
        Assert.That(Math.Sqrt(error/energy),Is.LessThan(.01),"World-space rotation must follow the output sample clock, including response fades and partial hops.");
    }

    [TestCase(false,TakeRender.Synthesis.ImpulseResponse),TestCase(true,TakeRender.Synthesis.ImpulseResponse)]
    [Category("GPU")]
    public void ResponseChangesMatchContinuousOutputCrossfade(bool accelerated,TakeRender.Synthesis method)
    {
        RequireGpu();const int rate=8192,step=128,change=step*3;
        var scene=Scene(2);var before=States(2);var after=States(2);
        after[1].rotation=Quaternion.Euler(0,90,0);
        var settings=SpectralSettings();settings.synthesis=method;settings.raysPerBin=1;settings.maxBounces=1;
        settings.listenerOutput=TakeRender.ListenerOutput.Binaural;settings.transitionSamples=45; // Ends inside an FFT hop.
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
        using var arrivals=new ComputeBuffer(oldPaths.Length,RayTransport.PathStride);
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
        using var buffer=new ComputeBuffer(paths.Length,RayTransport.PathStride);buffer.SetData(paths);
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
        using var arrivals=new ComputeBuffer(paths.Length,RayTransport.PathStride);arrivals.SetData(paths);
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

    [TestCase(128,true,0),TestCase(2048,true,0),TestCase(128,true,.125f),TestCase(2048,true,.125f),TestCase(2048,false,0),Category("GPU")]
    public void HeadUpdatesReuseMaterialsAndOnlyIdenticalInputClocks(int size,bool turn,float delayRate)
    {
        RequireGpu();var scene=Scene();var state=States();var settings=SpectralSettings();
        settings.fftSize=size;settings.spectralHop=size/4;settings.responseStepSamples=size;settings.transitionSamples=size/2;settings.raysPerBin=1;settings.maxBounces=1;settings.listenerOutput=TakeRender.ListenerOutput.Binaural;
        var paths=new RayTransport.Path[settings.SpectralSlotsPerPair];
        paths[0]=new RayTransport.Path{pressureGain=1,direction=new Vector4(1,0,0,delayRate)};
        using var arrivals=new ComputeBuffer(paths.Length,RayTransport.PathStride);arrivals.SetData(paths);
        var clip=new float[256];for(int i=0;i<clip.Length;i++)clip[i]=Mathf.Sin(i*.3f);
        var signal=new TakeRender.Signal(clip);signal.SetActive(0,true);signal.Append(new AcousticTake.VoiceEvent{gain=1,loop=true});
        var signals=new[]{signal};using var gpu=new GpuAudio(scene,8192,settings,signals);var cpu=new TakeRender.Mixer(scene,8192,settings,signals);
        var actualPcm=new float[size*2];var expectedPcm=new float[size*2];var direct=new TakeRender.DirectPath[1];
        gpu.SetResponse(0,false,arrivals,0,scene,state);gpu.Render(0,size,actualPcm);
        cpu.SetResponse(0,direct,state,false,paths,0);cpu.Render(0,size,expectedPcm);
        long filters=gpu.Diagnostics.materialFilters,ffts=gpu.Diagnostics.inputFfts;
        if(turn){state[1].rotation=Quaternion.Euler(0,90,0);state[1].angularVelocity=Vector3.up;state[1].flags|=AcousticTake.StateFlags.VelocityValid;}
        var fields=new[]{"work","transform"};var buffers=new ComputeBuffer[fields.Length];
        var sentinel=new Vector2[8];for(int i=0;i<sentinel.Length;i++)sentinel[i]=new Vector2(i+1,-i-1);
        for(int i=0;i<fields.Length;i++)
        {
            buffers[i]=(ComputeBuffer)typeof(GpuAudio).GetField(fields[i],System.Reflection.BindingFlags.Instance|System.Reflection.BindingFlags.NonPublic).GetValue(gpu);
            buffers[i].SetData(sentinel,0,0,sentinel.Length);
        }
        gpu.SetResponse(size,false,arrivals,0,scene,state);
        foreach(var buffer in buffers)
        {var actual=new Vector2[sentinel.Length];buffer.GetData(actual,0,0,actual.Length);Assert.That(actual,Is.EqualTo(sentinel),"Head updates must not rebuild acoustic filters.");}
        gpu.Render(size,size,actualPcm);cpu.SetResponse(size,direct,state,false,paths,0);cpu.Render(size,size,expectedPcm);
        Assert.That(gpu.Diagnostics.materialFilters,Is.EqualTo(filters));
        Assert.That(gpu.Diagnostics.inputFfts-ffts,Is.EqualTo(4),"Matched responses share the same interpolated source clock, including head turns.");
        double error=0;for(int i=0;i<actualPcm.Length;i++)error=Math.Max(error,Math.Abs(actualPcm[i]-expectedPcm[i]));
        Assert.That(error,Is.LessThan(2e-5));
    }

    [TestCase(128),TestCase(2048),Category("GPU")]
    public void UnchangedPairsRetainFiltersDuringOtherPairsFades(int size)
    {
        RequireGpu();const int rate=8192;var scene=Scene(2);var state=States(2);scene.listeners=new[]{1};
        scene.sources=new[]{scene.sources[0],new AcousticTake.SourceDefinition{objectIndex=2,clipIndex=0}};
        var settings=SpectralSettings();settings.fftSize=size;settings.spectralHop=size/4;
        settings.responseStepSamples=size;settings.transitionSamples=size/2;settings.raysPerBin=1;settings.maxBounces=1;
        var clip=new float[257];for(int i=0;i<clip.Length;i++)clip[i]=Mathf.Sin(i*.37f)*.2f;
        var signal=new TakeRender.Signal(clip);signal.SetActive(0,true);signal.Append(new AcousticTake.VoiceEvent{gain=1,loop=true});
        var signals=new[]{signal,signal};var paths=new RayTransport.Path[2*settings.SpectralSlotsPerPair];
        paths[0]=new RayTransport.Path{frequencyBin=4,pressureGain=1,direction=new Vector4(1,0,0,0)};
        using var arrivals=new ComputeBuffer(paths.Length,RayTransport.PathStride);using var gpu=new GpuAudio(scene,rate,settings,signals);
        var cpu=new TakeRender.Mixer(scene,rate,settings,signals);var actual=new float[size];var expected=new float[size];
        double error=0,energy=0;
        for(int frame=0;frame<5;frame++)
        {
            paths[settings.SpectralSlotsPerPair]=new RayTransport.Path{frequencyBin=4,pressureGain=1f/(frame+1),direction=new Vector4(0,0,1,.01f)};
            if(frame==2)state[1].flags&=~AcousticTake.StateFlags.Active;else state[1].flags|=AcousticTake.StateFlags.Active;
            arrivals.SetData(paths);long sample=frame*(long)size,filters=gpu.Diagnostics.materialFilters;
            cpu.SetResponse(sample,new TakeRender.DirectPath[2],state,false,paths,0);gpu.SetResponse(sample,false,arrivals,0,scene,state);
            cpu.Render(sample,size,expected);gpu.Render(sample,size,actual);
            Assert.That(gpu.Diagnostics.materialFilters-filters,Is.EqualTo(frame==0 || frame==3?2:frame==2?0:1));
            for(int n=0;n<size;n++){double d=expected[n]-actual[n];error+=d*d;energy+=expected[n]*expected[n];}
        }
        Assert.That(energy,Is.GreaterThan(1e-6));Assert.That(Math.Sqrt(error/energy),Is.LessThan(5e-4));
    }

}
