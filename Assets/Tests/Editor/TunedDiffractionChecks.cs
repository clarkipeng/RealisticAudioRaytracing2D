using System;
using System.Collections.Generic;
using System.IO;
using NUnit.Framework;
using UnityEngine;
public partial class TakeChecks
{
    public static IEnumerable<string> CompleteWaveCases()
    {
        var file=Environment.GetEnvironmentVariable("ACOUSTIC_COMPLETE_WAVE_CASES");
        return string.IsNullOrEmpty(file)?new[]{""}:File.ReadAllLines(file);
    }
    [TestCaseSource(nameof(CompleteWaveCases)),Category("QualityMeasurement"),Timeout(1800000)]
    public void MeasureRuntimeCompleteRoutes(string directory)
    {
        if(string.IsNullOrEmpty(directory))Assert.Ignore("Set ACOUSTIC_COMPLETE_WAVE_CASES to measurement inputs.");
        var previous=Environment.GetEnvironmentVariable("ACOUSTIC_WAVE_OUTPUT");
        try {Environment.SetEnvironmentVariable("ACOUSTIC_WAVE_OUTPUT",directory);MeasureWaveField();}
        finally {Environment.SetEnvironmentVariable("ACOUSTIC_WAVE_OUTPUT",previous);}
    }
    [TestCase(false),TestCase(true),Category("GPU"),Category("TunedDiffraction")]
    public void TunedRoutesMatchIndependentHalfPlane(bool planar)
    {
        KnifeEdgeFixture(planar,out var scene,out var geometry,out var state);
        var settings=EdgeSettings();settings.fftSize=16;settings.spectralHop=4;settings.raysPerBin=2;
        settings.completeRouteSamples=65536;settings.useBvh=true;
        using var solver=new TransportSolver(scene,geometry,settings,RequireGpu(),sampleRate:1000);
        solver.Solve(state,new TakeRender.DirectPath[1]);
        int[] bins={0,1,4,8};double[] re={.156429101689,.112431573523,.004442267995,-.066639430815};
        double[] im={0,.020944020787,.083451251431,-.016598353630};
        for(int i=0;i<bins.Length;i++)
            Assert.That((EdgeResponse(solver.RayPaths,bins[i],bins[i]*62.5f)-new System.Numerics.Complex(re[i],im[i])).Magnitude,
                Is.LessThan(.004),"Independent signed half-plane quadrature, bin "+bins[i]);
    }
    [TestCase(false),TestCase(true),Category("GPU"),Category("TunedDiffraction")]
    public void TunedRoutesReachTheFarSideOfASolid(bool planar)
    {
        KnifeEdgeFixture(planar,out var scene,out _,out var state);
        var geometry=TransportBox(scene.dimension,2,0,1);scene.objects[2].primitiveCount=geometry.Length;
        state[0].position=Vector3.left*3;state[1].position=Vector3.right*3;
        var settings=EdgeSettings();settings.fftSize=8;settings.spectralHop=2;settings.raysPerBin=2;
        settings.maxBounces=2;settings.completeRouteSamples=65536;settings.useBvh=true;
        using var solver=new TransportSolver(scene,geometry,settings,RequireGpu(),sampleRate:1000);
        solver.Solve(state,new TakeRender.DirectPath[1]);
        foreach(var path in solver.RayPaths)if(path.pressureGain!=0)
            Assert.That(path.delaySeconds,Is.GreaterThanOrEqualTo((2*Mathf.Sqrt(5)+2)/343-1e-6f),"No route may shortcut the solid.");
        int[] bins=planar?new[]{0}:new[]{0,1,4};
        double[] re=planar?new[]{.199807}:new[]{.160990476380,-.132356150111,-.065193018566};
        double[] im=planar?new[]{0d}:new[]{0d,-.077646144425,.063496034679};
        for(int i=0;i<bins.Length;i++)
            Assert.That((EdgeResponse(solver.RayPaths,bins[i],bins[i]*125)-new System.Numerics.Complex(re[i],im[i])).Magnitude,
                Is.LessThan(.012),"Independent EDtoolbox second-order field, bin "+bins[i]);
    }
    [TestCase(false,TakeRender.Synthesis.Spectral),TestCase(true,TakeRender.Synthesis.Spectral)]
    [TestCase(false,TakeRender.Synthesis.ImpulseResponse),TestCase(true,TakeRender.Synthesis.ImpulseResponse)]
    [Category("GPU"),Category("TunedDiffraction")]
    public void TunedDiffractionProducesMatchingCpuGpuAudio(bool planar,TakeRender.Synthesis method)
    {
        KnifeEdgeFixture(planar,out var scene,out var geometry,out var state);
        var settings=EdgeSettings();settings.raysPerBin=2;settings.completeRouteSamples=128;
        settings.synthesis=method;settings.listenerOutput=TakeRender.ListenerOutput.Binaural;settings.transitionSamples=32;
        var clip=new float[257];for(int i=0;i<clip.Length;i++)clip[i]=.2f*Mathf.Sin(i*.17f);
        var signal=new TakeRender.Signal(clip);signal.SetActive(0,true);signal.Append(new AcousticTake.VoiceEvent{gain=1,loop=true});
        var signals=new[]{signal};var cpu=new TakeRender.Mixer(scene,8000,settings,signals);
        using var gpu=new GpuAudio(scene,8000,settings,signals,transported:true);
        using var solver=new TransportSolver(scene,geometry,settings,RequireGpu(),sampleRate:8000);
        var direct=new TakeRender.DirectPath[1];var a=new float[256];var b=new float[256];double error=0,energy=0,peak=0;
        for(int frame=0;frame<8;frame++)
        {
            long sample=frame*128;state[2].position.y=frame*.02f;state[1].rotation=Quaternion.Euler(0,frame*11,0);
            solver.Solve(state,direct);Assert.That(direct[0].pressureGain,Is.Zero);
            cpu.SetResponse(sample,direct,state,false,solver.RayPaths,0);gpu.SetResponse(sample,false,solver.RayBuffer,0,scene,state);
            cpu.Render(sample,128,a);gpu.Render(sample,128,b);
            for(int i=0;i<a.Length;i++){double d=a[i]-b[i];error+=d*d;energy+=a[i]*a[i];peak=Math.Max(peak,Math.Abs(d));}
        }
        Assert.That(energy,Is.GreaterThan(1e-6));Assert.That(peak,Is.LessThan(2e-4));Assert.That(Math.Sqrt(error/energy),Is.LessThan(5e-4));
    }

    [Test,Category("GPU"),Category("TunedDiffraction")]
    public void TunedBurstAudioPreservesBirthsAcrossBatchingAndLive()
    {
        KnifeEdgeFixture(false,out var scene,out var geometry,out var state);scene.planes=new[]{BurstPlane()};
        const int rate=8192;var pcm=new float[65];for(int i=0;i<pcm.Length;i++)pcm[i]=Mathf.Sin(i*.29f);
        var events=new[]{new AcousticTake.VoiceEvent{gain=.2f,loop=true},
            new AcousticTake.VoiceEvent{sample=73,action=AcousticTake.VoiceAction.Gain,gain=.3f},
            new AcousticTake.VoiceEvent{sample=201,action=AcousticTake.VoiceAction.Stop},
            new AcousticTake.VoiceEvent{sample=299,gain=.2f,loop=true}};
        long[] times={0,53,129,270,411,512};string path=Path.Combine(directory,"diffracted-bursts");
        using(var writer=new AcousticTake.Writer(path,rate,scene,geometry,new[]{pcm}))
        {
            foreach(long time in times){state[0].position.x=-2+time*.0001f;writer.Append(time,state);}
            foreach(var e in events)writer.Append(e);writer.Complete();
        }
        using var take=AcousticTake.Open(path);var settings=BurstSettings();settings.raysPerBin=2;settings.completeRouteSamples=64;
        var shader=RequireGpu();float[] expected=null;
        foreach(var mode in new[]{(TakeRender.AudioBackend.CpuReference,1),(TakeRender.AudioBackend.Gpu,1),(TakeRender.AudioBackend.Gpu,3)})
        {
            settings.audioBackend=mode.Item1;settings.maximumResponsesPerBatch=mode.Item2;
            string output=Path.Combine(directory,"diffracted-"+mode.Item1+mode.Item2);
            TakeRender.RenderDirect(take,settings,output,null,shader);var actual=ReadWave(Path.Combine(output,"listener-0.wav"));
            if(expected==null)
            {expected=actual;double energy=0;foreach(float value in actual)energy+=value*value;Assert.That(energy,Is.GreaterThan(1e-6));}
            else AssertBurstAudio(expected,actual);
        }
        var rendered=new List<float>();Action<long,float[],int> collect=(_,block,count)=>{for(int i=0;i<count;i++)rendered.Add(block[i]);};
        var left=States(2);var right=States(2);take.ReadStoredFrame(0,left);
        using(var live=new TakeRender.Live(scene,take.Geometry,new[]{pcm},rate,settings,null,0,0,shader))
        {
            live.Prepare(left);live.Append(events[0]);live.Observe(0,0,left,left,collect);int e=1;
            for(int f=1;f<times.Length;f++)
            {
                while(e<events.Length && events[e].sample<=times[f])live.Append(events[e++]);
                take.ReadStoredFrame(f,right);live.Observe(times[f-1],times[f],left,right,collect);Array.Copy(right,left,left.Length);
            }
            live.Finish(times[times.Length-1],left,collect);Assert.That(live.Completed,Is.True);
        }
        AssertBurstAudio(expected,rendered.ToArray());
    }

    [TestCase(false),TestCase(true),Category("GPU"),Category("TunedDiffraction")]
    public void TunedEarlyRoutesPreserveExistingDeeperInteractions(bool planar)
    {
        KnifeEdgeFixture(planar,out var scene,out _,out var state);
        var geometry=TransportBox(scene.dimension,2,0,1);scene.objects[2].primitiveCount=geometry.Length;
        scene.materials=new[]{new AcousticTake.MaterialDefinition{frequencies=new[]{0f,500f},absorption=new[]{.2f,.3f},
            scattering=new[]{.2f,.35f},transmission=new[]{.1f,.2f},ior=new[]{1f,1.1f}}};
        state[0].position=new Vector3(-3,.4f,0);state[1].position=new Vector3(3,-.2f,0);
        var settings=EdgeSettings();settings.fftSize=8;settings.spectralHop=2;settings.raysPerBin=128;
        settings.maxBounces=32;settings.maximumDelaySeconds=1.25f;settings.useBvh=true;
        RayTransport.Path[] before;
        using(var baseline=new TransportSolver(scene,geometry,settings,RequireGpu(),sampleRate:1000))
        {baseline.Solve(state,new TakeRender.DirectPath[1]);before=(RayTransport.Path[])baseline.RayPaths.Clone();}
        settings.completeRouteSamples=64;
        using var solver=new TransportSolver(scene,geometry,settings,RequireGpu(),sampleRate:1000);
        solver.Solve(state,new TakeRender.DirectPath[1]);int first=settings.FrequencyBins*(1+settings.MirrorCandidates(scene));
        for(int ray=0;ray<settings.RaysPerPair;ray++)for(int bounce=2;bounce<32;bounce++)
        {
            int at=first+ray*settings.PathSlotsPerRay+bounce;
            Assert.That(solver.RayPaths[at],Is.EqualTo(before[at]),"Later interaction "+at+" (ray "+ray+", bounce "+bounce+"): before="+JsonUtility.ToJson(before[at])+" after="+JsonUtility.ToJson(solver.RayPaths[at]));
        }
    }
    [Test,Category("GPU"),Category("TunedDiffraction")]
    public void TunedRoutesAppearInRayCapture()
    {
        KnifeEdgeFixture(false,out var scene,out var geometry,out var state);
        var settings=EdgeSettings();settings.raysPerBin=2;settings.completeRouteSamples=128;settings.fftSize=8;settings.spectralHop=2;
        using var solver=new TransportSolver(scene,geometry,settings,RequireGpu(),sampleRate:1000);
        var segments=solver.CaptureRays(state,0,1);int first=geometry.Length+1+settings.RaysPerBin*(2+2*settings.maxBounces)+2*settings.MirrorCandidates(scene);
        int diffracted=0,arrivals=0;
        for(int i=first;i<segments.Length;i++)
        {
            if(segments[i].kind==TransportSolver.RayEvent.Diffraction)diffracted++;
            if(segments[i].kind==TransportSolver.RayEvent.Listener)arrivals++;
        }
        Assert.That(diffracted,Is.GreaterThan(0));Assert.That(arrivals,Is.EqualTo(diffracted));
    }

}
