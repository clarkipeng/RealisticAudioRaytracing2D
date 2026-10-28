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

    [TestCase(-1.005f,0),TestCase(-1.005f,17),TestCase(-.995f,0),TestCase(-.995f,17),TestCase(-2f,0),TestCase(-2f,17)]
    [Category("GPU"),Category("FiniteAdaptation")]
    public void FiniteDiffractionKeepsThePeakAtUnequalEndpointHeights(float y,int seed)
    {
        KnifeEdgeFixture(false,out var scene,out var geometry,out var state);
        state[0].position=new Vector3(-2,1,-8);state[1].position=new Vector3(2,y,4);
        var settings=EdgeSettings();settings.fftSize=8;settings.spectralHop=2;settings.raysPerBin=2;
        settings.completeRouteSamples=2048;settings.useBvh=true;settings.seed=seed;
        using var solver=new TransportSolver(scene,geometry,settings,RequireGpu(),sampleRate:1000);
        solver.Solve(state,new TakeRender.DirectPath[1]);
        // Independent double-precision quadrature of the physical finite edge,
        // with the 0.1 s horizon. Refinement changes every component by <8e-9.
        double[] re=y<-1.5f?new[]{.06843164673970846,.012774721943881809,-.03883484010540754,.032912210471587426}:
            y<-1?new[]{.07121761059868985,-.01884081741135529,-.031237589211197396,-.017125893790192675}:
            new[]{.07124448416276889,-.019135875840134972,-.03092320874570661,-.017968330874092296};
        double[] im=y<-1.5f?new[]{0d,.045102900198930984,.012316053724628951,-.010943643929545006}:
            y<-1?new[]{0d,.0502251388012103,-.03841724891942468,.04297588280604822}:
            new[]{0d,.05019485235594404,-.03880526670515018,.04279319803337289};
        int[] bins={0,1,2,4};
        for(int i=0;i<bins.Length;i++)
        {
            var actual=EdgeResponse(solver.RayPaths,bins[i],bins[i]*125);
            Assert.That((actual-new System.Numerics.Complex(re[i],im[i])).Magnitude,Is.LessThan(.003),
                "Unequal-height finite edge, bin "+bins[i]+", value="+actual);
        }
    }

    [TestCase(false,0f),TestCase(true,0f),TestCase(false,.1f),TestCase(true,.1f)]
    [Category("GPU"),Category("FiniteAdaptation")]
    public void JoinedMeshObjectsKeepTheSameSolidFaceMultiplicity(bool rotated,float gap)
    {
        var scene=Scene(3);scene.listeners=new[]{1};
        scene.materials=new[]{new AcousticTake.MaterialDefinition{frequencies=new[]{0f},absorption=new[]{0f},scattering=new[]{0f},transmission=new[]{0f}}};
        var state=States(3);state[0].position=Vector3.left*3;state[1].position=Vector3.right*3;
        var geometry=new List<AcousticTake.Primitive>();
        for(int owner=2;owner<=3;owner++)
        {
            // An off-centre seam/gap catches checks that examine only the midpoint.
            float left=owner==2?-1:-.8f+gap,right=owner==2?-.8f:1;
            state[owner].position=Vector3.right*((left+right)*.5f);state[owner].flags=AcousticTake.StateFlags.Active|AcousticTake.StateFlags.Geometry;
            scene.objects[owner].primitiveStart=geometry.Count;
            foreach(var original in TransportBox(scene.dimension,owner,0,1))
            {var face=original;float half=(right-left)*.5f;face.a.x*=half;face.b.x*=half;face.c.x*=half;geometry.Add(face);}
            scene.objects[owner].primitiveCount=geometry.Count-scene.objects[owner].primitiveStart;
        }
        var rotation=rotated?Quaternion.Euler(23,41,17):Quaternion.identity;
        var translation=rotated?new Vector3(4,-2,1):Vector3.zero;
        for(int i=0;i<state.Length;i++){state[i].position=rotation*state[i].position+translation;state[i].rotation=rotation;}
        var settings=EdgeSettings();settings.fftSize=8;settings.spectralHop=2;settings.raysPerBin=2;
        settings.completeRouteSamples=65536;settings.maxBounces=2;settings.useBvh=true;
        using var solver=new TransportSolver(scene,geometry.ToArray(),settings,RequireGpu(),sampleRate:1000);
        var segments=solver.CaptureRays(state,0,0);
        int start=geometry.Count+1+settings.RaysPerBin*(2+2*settings.maxBounces)+2*settings.MirrorCandidates(scene);
        double actual=0;var inverse=Quaternion.Inverse(rotation);
        for(int i=0;i<settings.CompleteSamples;i++)
        {
            var first=segments[start+3*i];var second=segments[start+3*i+1];
            if(first.kind!=TransportSolver.RayEvent.Diffraction || second.kind!=TransportSolver.RayEvent.Diffraction)continue;
            var a=inverse*((Vector3)first.end-translation);var b=inverse*((Vector3)second.end-translation);
            if(Mathf.Abs(a.x+1)<1e-4f && Mathf.Abs(b.x-1)<1e-4f)actual+=segments[start+3*i+2].pressure;
        }
        // Same four outer-edge integrals as the independently integrated cube.
        // A genuine gap removes the shared-solid-face half multiplicity.
        double expected=.160990476380*(gap==0?1:2);
        Assert.That(actual,Is.EqualTo(expected).Within(expected*.075),
            "Outer-edge field must follow continuous solid coverage, independent of mesh ownership and pose.");
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
