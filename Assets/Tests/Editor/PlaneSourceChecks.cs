using NUnit.Framework;
using System;
using System.IO;
using System.Linq;
using UnityEngine;

public partial class TakeChecks
{
    static AcousticTake.EmissionPlane Plane(int count=8) => new AcousticTake.EmissionPlane {
        source=0,samples=count,seed=19,u=Vector3.right*4,v=Vector3.forward*2,width=1,height=1,mask=new[]{1f} };

    [Test,Category("Core")]
    public void PlaneMaskWeightsPowerAndPreservesSamplePrefixes()
    {
        var plane=Plane(64);plane.width=2;plane.mask=new[]{0f,.25f};
        var first=Enumerable.Range(0,64).Select(i=>AudioSourcePlane.Sample(plane,i)).ToArray();
        Assert.That(first.Sum(p=>p.w*p.w),Is.EqualTo(1).Within(1e-5));
        foreach(var p in first){Assert.That(Math.Abs(p.x),Is.LessThan(2));Assert.That(Math.Abs(p.z),Is.LessThan(1));if(p.x<0)Assert.That(p.w,Is.Zero);}
        plane.samples=256;
        Assert.That(Enumerable.Range(0,256).Sum(i=>Math.Pow(AudioSourcePlane.Sample(plane,i).w,2)),Is.EqualTo(1).Within(1e-5));
        for(int i=0;i<first.Length;i++)Assert.That((Vector3)AudioSourcePlane.Sample(plane,i),Is.EqualTo((Vector3)first[i]));
        plane.mask[1]=float.NaN;Assert.Throws<InvalidDataException>(()=>AudioSourcePlane.Validate(plane));
    }

    [Test,Category("Core")]
    public void PlaneClipOffsetsPreserveEventTimesAndLoopPhase()
    {
        var pcm=Enumerable.Range(0,101).Select(i=>i/101f).ToArray();var signal=new TakeRender.Signal(pcm);
        signal.SetActive(0,true);signal.Append(new AcousticTake.VoiceEvent{sample=10,gain=.5f,loop=true,clipOffset=17});
        signal.Append(new AcousticTake.VoiceEvent{sample=40,action=AcousticTake.VoiceAction.Gain,gain=.25f});
        signal.Append(new AcousticTake.VoiceEvent{sample=210,action=AcousticTake.VoiceAction.Stop});
        foreach(uint phase in new[]{0u,1u,32768u,65535u})for(long n=0;n<240;n++)
        {
            float expected=n<10 || n>=210?0:pcm[(n-10+17+((long)phase*pcm.Length>>16))%pcm.Length]*(n<40?.5f:.25f);
            Assert.That(signal.Read((double)n,phase),Is.EqualTo(expected));
        }
    }

    [Test,Category("Core")]
    public void PlaneTakeAuthenticatesMaskAndRejectsLegacyExtension()
    {
        var scene=Scene();scene.planes=new[]{Plane()};string file=Path.Combine(directory,"plane");
        using(var writer=new AcousticTake.Writer(file,8000,scene,Array.Empty<AcousticTake.Primitive>(),new[]{new[]{.1f,.2f}}))
        {writer.Append(0,States());writer.Append(100,States());writer.Append(new AcousticTake.VoiceEvent{gain=1,loop=true});writer.Complete();}
        using(var take=AcousticTake.Open(file))Assert.That(take.Metadata.scene.planes[0].samples,Is.EqualTo(8));
        Assert.Throws<InvalidDataException>(()=>new AcousticTake.Writer(Path.Combine(directory,"legacy"),8000,scene,Array.Empty<AcousticTake.Primitive>(),new[]{new[]{.1f}},version:4));
        string manifest=Path.Combine(file,"take.json");var metadata=JsonUtility.FromJson<AcousticTake.Manifest>(File.ReadAllText(manifest));
        metadata.scene.planes[0].mask[0]=.5f;File.WriteAllText(manifest,JsonUtility.ToJson(metadata));
        Assert.Throws<InvalidDataException>(()=>AcousticTake.ReadManifest(file));
        Assert.Throws<NotSupportedException>(()=>AudioSourcePlane.ValidateRender(scene,new TakeRender.Settings()));
        Assert.Throws<InvalidDataException>(()=>AudioSourcePlane.ValidateEvent(scene,new AcousticTake.VoiceEvent()));
    }

    [TestCase(AcousticTake.Dimension.Planar2D),TestCase(AcousticTake.Dimension.Spatial3D),Category("GPU")]
    public void PlaneReflectionsKeepEachSiteWaveformAndGeometricDelay(AcousticTake.Dimension dimension)
    {
        var scene=Scene(2);scene.dimension=dimension;scene.listeners=new[]{1};scene.objects[2].primitiveCount=1;
        scene.planes=new[]{Plane(4)};var plane=scene.planes[0];plane.u=Vector3.right*.5f;plane.v=(dimension==AcousticTake.Dimension.Planar2D?Vector3.up:Vector3.forward)*.5f;
        scene.materials=new[]{new AcousticTake.MaterialDefinition{frequencies=new[]{0f},absorption=new[]{0f},scattering=new[]{0f},transmission=new[]{0f}}};
        var geometry=new[]{new AcousticTake.Primitive{objectIndex=2,a=new Vector3(-10,0,-10),b=new Vector3(10,0,-10),c=new Vector3(0,0,10)}};
        var states=States(2);states[0].position=new Vector3(-1,1,0);states[1].position=new Vector3(1,1,0);
        states[2].position=Vector3.zero;states[2].flags=AcousticTake.StateFlags.Active|AcousticTake.StateFlags.Geometry;
        var settings=SpectralSettings();settings.raysPerBin=4;settings.maxBounces=1;settings.speedOfSound=343;
        using var solver=new TransportSolver(scene,geometry,settings,RequireGpu(),sampleRate:8192);
        solver.Solve(states,new TakeRender.DirectPath[1]);var paths=solver.RayPaths;int bins=settings.FrequencyBins;
        var phases=new System.Collections.Generic.HashSet<uint>();
        for(int r=0;r<4;r++)
        {
            var reflected=paths[4*bins+r*2];phases.Add(reflected.emissionPhase);
            int site=Enumerable.Range(0,4).Single(i=>paths[i*bins].emissionPhase==reflected.emissionPhase);
            var sample=AudioSourcePlane.Sample(plane,site);var image=states[0].position+(Vector3)sample;image.y=-image.y;
            float length=Vector3.Distance(image,states[1].position);
            Assert.That(reflected.pressureGain,Is.EqualTo(sample.w/length).Within(1e-5));
            Assert.That(reflected.delaySeconds,Is.EqualTo(length/343).Within(1e-6));
        }
        Assert.That(phases.Count,Is.EqualTo(4));
    }

    [Test,Category("GPU")]
    public void PlaneDirectPathsUseMaskedRotatedPositionsAndSharedSourcePhases()
    {
        var scene=Scene(2);scene.planes=new[]{Plane(8)};scene.planes[0].width=2;scene.planes[0].mask=new[]{0f,1f};
        var states=States(2);states[0].rotation=Quaternion.Euler(0,71,0);states[0].position=new Vector3(2,3,4);
        states[1].position=new Vector3(4,4,-2);states[2].position=new Vector3(-3,2,7);
        var settings=SpectralSettings();settings.airAbsorption=false;settings.raysPerBin=1;settings.maxBounces=1;
        using var solver=new TransportSolver(scene,Array.Empty<AcousticTake.Primitive>(),settings,RequireGpu(),sampleRate:8000);
        solver.Solve(states,new TakeRender.DirectPath[2]);var paths=solver.RayPaths;int slots=settings.SlotsPerPair(scene),bins=settings.FrequencyBins;
        for(int l=0;l<2;l++)for(int j=0;j<8;j++)for(int k=0;k<bins;k++)
        {
            var sample=AudioSourcePlane.Sample(scene.planes[0],j);var path=paths[l*slots+j*bins+k];
            if(sample.w==0){Assert.That(path.pressureGain,Is.Zero);continue;}
            Vector3 position=states[0].position+states[0].rotation*(Vector3)sample;
            float distance=Vector3.Distance(position,states[l+1].position);
            Assert.That(path.delaySeconds,Is.EqualTo(distance/settings.speedOfSound).Within(1e-6));
            Assert.That(path.pressureGain,Is.EqualTo(sample.w/distance).Within(1e-5));
            Assert.That(path.emissionPhase,Is.EqualTo(paths[j*bins].emissionPhase));
        }
        // The ray budget does not change the direct spatial quadrature or audio offsets.
        var saved=(RayTransport.Path[])paths.Clone();settings.raysPerBin=3;
        using var more=new TransportSolver(scene,Array.Empty<AcousticTake.Primitive>(),settings,RequireGpu(),sampleRate:8000);
        more.Solve(states,new TakeRender.DirectPath[2]);
        for(int l=0;l<2;l++)for(int j=0;j<8*bins;j++)
        {Assert.That(more.RayPaths[l*settings.SlotsPerPair(scene)+j].pressureGain,Is.EqualTo(saved[l*slots+j].pressureGain));Assert.That(more.RayPaths[l*settings.SlotsPerPair(scene)+j].emissionPhase,Is.EqualTo(saved[l*slots+j].emissionPhase));}
    }

    [TestCase(128,TakeRender.ListenerOutput.Mono,TakeRender.DelayFilter.Linear)]
    [TestCase(128,TakeRender.ListenerOutput.Binaural,TakeRender.DelayFilter.Linear)]
    [TestCase(2048,TakeRender.ListenerOutput.Mono,TakeRender.DelayFilter.Linear)]
    [TestCase(128,TakeRender.ListenerOutput.Mono,TakeRender.DelayFilter.WindowedSinc),Category("GPU")]
    public void PlaneGpuSynthesisMatchesCpuWithMovementAndSourceEvents(int size,TakeRender.ListenerOutput ears,TakeRender.DelayFilter filter)
    {
        const int rate=8192;var scene=Scene();scene.planes=new[]{Plane(4)};var states=States();
        var settings=SpectralSettings();settings.fftSize=size;settings.spectralHop=size/4;settings.responseStepSamples=size;
        settings.raysPerBin=1;settings.maxBounces=1;settings.listenerOutput=ears;settings.delayFilter=filter;
        var pcm=new float[65537];var random=new System.Random(109);for(int i=0;i<pcm.Length;i++)pcm[i]=.2f*(float)(random.NextDouble()*2-1);
        var signal=new TakeRender.Signal(pcm);signal.SetActive(0,true);signal.Append(new AcousticTake.VoiceEvent{gain=.6f,loop=true,clipOffset=pcm.Length-37});
        signal.Append(new AcousticTake.VoiceEvent{sample=size*3,action=AcousticTake.VoiceAction.Gain,gain=.3f});
        signal.Append(new AcousticTake.VoiceEvent{sample=size*5,action=AcousticTake.VoiceAction.Stop});
        using var solver=new TransportSolver(scene,Array.Empty<AcousticTake.Primitive>(),settings,RequireGpu(),sampleRate:rate);
        var signals=new[]{signal};using var gpu=new GpuAudio(scene,rate,settings,signals);var cpu=new TakeRender.Mixer(scene,rate,settings,signals);
        var direct=new TakeRender.DirectPath[1];var a=new float[size*settings.ChannelsPerListener];var b=new float[a.Length];double error=0,energy=0;
        for(int frame=0;frame<8;frame++)
        {
            states[0].position=new Vector3(frame*.02f,1,0);states[1].position=new Vector3(0,2,4+frame*.03f);states[1].rotation=Quaternion.Euler(0,frame*13,0);
            solver.Solve(states,direct);long sample=frame*(long)size;
            cpu.SetResponse(sample,direct,states,false,solver.RayPaths,0);gpu.SetResponse(sample,false,solver.RayBuffer,0,scene,states);
            cpu.Render(sample,size,a);gpu.Render(sample,size,b);
            for(int i=0;i<a.Length;i++){Assert.That(float.IsFinite(b[i]),Is.True);double d=a[i]-b[i];error+=d*d;energy+=a[i]*a[i];}
        }
        Assert.That(energy,Is.GreaterThan(1e-6));Assert.That(Math.Sqrt(error/energy),Is.LessThan(.001));
        Assert.That(signal.Read(size*8d,32768),Is.Zero);
    }
}
