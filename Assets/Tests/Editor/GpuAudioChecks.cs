using NUnit.Framework;
using System;
using System.Collections;
using UnityEngine.TestTools;
using UnityEngine;

public partial class TakeChecks
{
    [TestCase(TakeRender.Synthesis.ImpulseResponse, TakeRender.ListenerOutput.Mono, TakeRender.DelayFilter.Linear, 128, 1)]
    [TestCase(TakeRender.Synthesis.ImpulseResponse, TakeRender.ListenerOutput.Mono, TakeRender.DelayFilter.WindowedSinc, 128, 1)]
    [TestCase(TakeRender.Synthesis.ImpulseResponse, TakeRender.ListenerOutput.Binaural, TakeRender.DelayFilter.Linear, 128, 1)]
    [TestCase(TakeRender.Synthesis.ImpulseResponse, TakeRender.ListenerOutput.Binaural, TakeRender.DelayFilter.WindowedSinc, 128, 1)]
    [TestCase(TakeRender.Synthesis.Spectral, TakeRender.ListenerOutput.Mono, TakeRender.DelayFilter.Linear, 128, 1)]
    [TestCase(TakeRender.Synthesis.Spectral, TakeRender.ListenerOutput.Mono, TakeRender.DelayFilter.WindowedSinc, 128, 1)]
    [TestCase(TakeRender.Synthesis.Spectral, TakeRender.ListenerOutput.Binaural, TakeRender.DelayFilter.Linear, 128, 1)]
    [TestCase(TakeRender.Synthesis.Spectral, TakeRender.ListenerOutput.Binaural, TakeRender.DelayFilter.WindowedSinc, 128, 1)]
    [TestCase(TakeRender.Synthesis.Spectral, TakeRender.ListenerOutput.Mono, TakeRender.DelayFilter.Linear, 2048, 1)]
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
        using var solver=new DirectSolver(scene,Array.Empty<AcousticTake.Primitive>(),settings,shader,sampleRate:rate);
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

}
