using System;
using NUnit.Framework;
using UnityEngine;

public partial class TakeChecks
{
    [TestCase(false),TestCase(true),Category("GPU"),Category("CompleteRoutes")]
    public void BudgetedFirstImagesPreservePrimitivePdf(bool planar)
    {
        KnifeEdgeFixture(planar,out var scene,out _,out var states);
        var geometry=TransportBox(scene.dimension,2,0,2);scene.objects[2].primitiveCount=geometry.Length;
        states[0].position=new Vector3(-.3f,.2f,0);states[1].position=new Vector3(.4f,.1f,0);
        states[2].position=Vector3.zero;
        var settings=SpectralSettings();settings.fftSize=8;settings.spectralHop=2;
        settings.raysPerBin=2;settings.reflectionCandidates=1;settings.maxBounces=1;settings.seed=17;
        settings.reflectionSampling=TakeRender.ReflectionSampling.PrimitiveBaseline;
        RayTransport.Path[] before;
        using(var baseline=new TransportSolver(scene,geometry,settings,RequireGpu(),sampleRate:1000))
        {baseline.Solve(states,new TakeRender.DirectPath[1]);before=(RayTransport.Path[])baseline.RayPaths.Clone();}
        settings.reflectionSampling=TakeRender.ReflectionSampling.FirstImages;
        Assert.That(settings.MirrorCandidates(scene),Is.EqualTo(2));
        Assert.That(geometry.Length,Is.GreaterThan(2));
        using var integrated=new TransportSolver(scene,geometry,settings,RequireGpu(),sampleRate:1000);
        integrated.Solve(states,new TakeRender.DirectPath[1]);int audible=0;
        for(int bin=0;bin<settings.FrequencyBins;bin++)for(int ray=0;ray<2;ray++)
        {
            var path=integrated.RayPaths[settings.FrequencyBins+bin*2+ray];
            Assert.That(path,Is.EqualTo(before[settings.FrequencyBins+(bin*2+ray)*2]));
            if(path.pressureGain!=0 || path.pressureImag!=0)audible++;
        }
        Assert.That(audible,Is.GreaterThan(0),"The bounded candidate comparison must include a nonzero reflection.");
    }

    [Test,Category("GPU"),Category("CompleteRoutes")]
    public void PackedGeometryLimitRejectsInvalidArrivalContract()
    {
        RequireGpu();var scene=Scene();var settings=SpectralSettings();settings.maxBounces=0;
        var signal=new TakeRender.Signal(new[]{.1f});signal.SetActive(0,true);signal.Append(new AcousticTake.VoiceEvent{gain=1,loop=true});
        var paths=new RayTransport.Path[settings.FrequencyBins];
        for(int bin=0;bin<paths.Length;bin++)paths[bin]=new RayTransport.Path{frequencyBin=(uint)bin,pressureGain=1,delaySeconds=bin/8192f};
        using var buffer=new ComputeBuffer(paths.Length,RayTransport.PathStride);buffer.SetData(paths);
        using var gpu=new GpuAudio(scene,8192,settings,new[]{signal},transported:true);
        gpu.SetResponse(0,false,buffer,0,scene,States());
        Assert.That(new RayTransport.Layout(scene,settings).GroupLimits[0],Is.EqualTo(1));
        InvalidOperationException error=null;
        try{gpu.Render(0,128,new float[128]);}catch(InvalidOperationException value){error=value;}
        Assert.That(error,Is.Not.Null);
        Assert.That(error.Message,Does.Contain("failed (5)"),"An invalid sharing contract must fail before writing outside packed filter storage.");
    }

    [TestCase(false),TestCase(true),Category("GPU"),Category("CompleteRoutes")]
    public void CompleteMirrorsPreserveAllThirtyTwoWalkInteractions(bool planar)
    {
        KnifeEdgeFixture(planar,out var scene,out _,out var states);
        var geometry=TransportBox(scene.dimension,2,0,1);scene.objects[2].primitiveCount=geometry.Length;
        scene.materials=new[]{new AcousticTake.MaterialDefinition{frequencies=new[]{0f,500f},absorption=new[]{.2f,.3f},
            scattering=new[]{.2f,.35f},transmission=new[]{.1f,.2f},ior=new[]{1f,1.1f}}};
        states[0].position=new Vector3(-3,.4f,0);states[1].position=new Vector3(3,-.2f,0);
        var settings=EdgeSettings();settings.fftSize=8;settings.spectralHop=2;settings.raysPerBin=128;settings.maxBounces=32;
        settings.maximumDelaySeconds=1.25f;settings.useBvh=true;
        settings.reflectionSampling=TakeRender.ReflectionSampling.PrimitiveBaseline;
        RayTransport.Path[] before;
        using(var baseline=new TransportSolver(scene,geometry,settings,RequireGpu(),sampleRate:1000))
        {baseline.Solve(states,new TakeRender.DirectPath[1]);before=(RayTransport.Path[])baseline.RayPaths.Clone();}
        settings.reflectionSampling=TakeRender.ReflectionSampling.FirstImages;
        using var integrated=new TransportSolver(scene,geometry,settings,RequireGpu(),sampleRate:1000);
        integrated.Solve(states,new TakeRender.DirectPath[1]);var after=integrated.RayPaths;
        int first=settings.FrequencyBins*(1+settings.MirrorCandidates(scene));
        for(int bin=0;bin<settings.FrequencyBins;bin++)Assert.That(after[bin],Is.EqualTo(before[bin]));
        for(int ray=0;ray<settings.RaysPerPair;ray++)for(int bounce=0;bounce<settings.maxBounces;bounce++)
            Assert.That(after[first+ray*settings.maxBounces+bounce],
                Is.EqualTo(before[settings.FrequencyBins+ray*(settings.maxBounces+1)+1+bounce]),
                "Mirror integration must preserve every existing surface/edge path, including its identity and timing.");
    }

    [TestCase(false),TestCase(true),Category("GPU"),Category("CompleteRoutes")]
    public void CompleteMirrorsMatchIndependentRoomImagesAtOneRay(bool planar)
    {
        var scene=Scene(2);scene.listeners=new[]{1};scene.dimension=planar?AcousticTake.Dimension.Planar2D:AcousticTake.Dimension.Spatial3D;
        var half=new Vector3(3,2,4);var geometry=TransportBox(scene.dimension,2,0,1);
        for(int i=0;i<geometry.Length;i++)
        {geometry[i].a=Vector3.Scale(geometry[i].a,half);geometry[i].b=Vector3.Scale(geometry[i].b,half);geometry[i].c=Vector3.Scale(geometry[i].c,half);}
        scene.objects[2].primitiveCount=geometry.Length;
        scene.materials=new[]{new AcousticTake.MaterialDefinition{frequencies=new[]{0f},absorption=new[]{.35f},scattering=new[]{0f},transmission=new[]{0f}}};
        var states=States(2);states[0].position=new Vector3(-.5f,.4f,planar?0:.3f);states[1].position=new Vector3(.6f,.2f,planar?0:-.3f);
        states[2].position=Vector3.zero;states[2].flags=AcousticTake.StateFlags.Active|AcousticTake.StateFlags.Geometry;
        var settings=SpectralSettings();settings.fftSize=8;settings.spectralHop=2;settings.speedOfSound=343;
        settings.raysPerBin=1;settings.maxBounces=1;settings.minimumDistance=.01f;
        foreach(int seed in new[]{0,17})
        {
            settings.seed=seed;
            using var solver=new TransportSolver(scene,geometry,settings,RequireGpu(),sampleRate:1000);
            solver.Solve(states,new TakeRender.DirectPath[1]);
            for(int bin=0;bin<settings.FrequencyBins;bin++)
            {
                double waveNumber=2*Math.PI*(bin*125)/343;
                double distance=Vector3.Distance(states[0].position,states[1].position);
                var expected=System.Numerics.Complex.Exp(new System.Numerics.Complex(0,-waveNumber*distance))/distance;
                for(int axis=0;axis<(planar?2:3);axis++)foreach(int side in new[]{-1,1})
                {
                    var image=states[0].position;image[axis]=2*side*half[axis]-image[axis];
                    distance=Vector3.Distance(image,states[1].position);
                    expected+=Math.Sqrt(.65)*System.Numerics.Complex.Exp(new System.Numerics.Complex(0,-waveNumber*distance))/distance;
                }
                Assert.That((EdgeResponse(solver.RayPaths,bin,bin*125)-expected).Magnitude,Is.LessThan(2e-5),"Every first image is retained at one ray; seed cannot change the field.");
            }
        }
    }

    [TestCase(128,TakeRender.Synthesis.Spectral,false,false)]
    [TestCase(128,TakeRender.Synthesis.ImpulseResponse,false,false)]
    [TestCase(2048,TakeRender.Synthesis.Spectral,false,false)]
    [TestCase(128,TakeRender.Synthesis.Spectral,true,false)]
    [TestCase(128,TakeRender.Synthesis.Spectral,false,true)]
    [Category("GPU"),Category("CompleteRoutes")]
    public void PackedRouteFiltersPreservePcm(int size,TakeRender.Synthesis method,bool dispersive,bool spectralBranches)
    {
        var scene=Scene(3);scene.listeners=new[]{1,2};
        scene.materials=new[]{new AcousticTake.MaterialDefinition{frequencies=new[]{0f,4096f},absorption=new[]{.2f,.4f},
            scattering=new[]{.15f,spectralBranches?.25f:.15f},transmission=new[]{0f,0f},ior=new[]{1f,dispersive?1.2f:1f}}};
        var geometry=size<=128?TransportBox(scene.dimension,3,0,2):new[]{new AcousticTake.Primitive{
            objectIndex=3,a=new Vector3(2,-10,-10),b=new Vector3(2,10,-10),c=new Vector3(2,0,10)}};
        scene.objects[3].primitiveCount=geometry.Length;
        var states=States(3);states[0].position=new Vector3(-.3f,.2f,0);states[1].position=new Vector3(.4f,.1f,.2f);states[2].position=new Vector3(-.7f,.3f,.4f);
        states[3].position=Vector3.zero;states[3].flags=AcousticTake.StateFlags.Active|AcousticTake.StateFlags.Geometry;
        var settings=SpectralSettings();settings.fftSize=size;settings.spectralHop=size/4;settings.responseStepSamples=size;
        settings.speedOfSound=343;settings.minimumDistance=.01f;settings.raysPerBin=2;settings.maxBounces=size<=128?4:1;
        settings.synthesis=method;settings.listenerOutput=TakeRender.ListenerOutput.Binaural;
        var pcm=new float[257];for(int i=0;i<pcm.Length;i++)pcm[i]=.1f*Mathf.Sin(i*.371f);
        var signal=new TakeRender.Signal(pcm);signal.SetActive(0,true);signal.Append(new AcousticTake.VoiceEvent{gain=1,loop=true});
        var signals=new[]{signal};
        using var packed=new GpuAudio(scene,8192,settings,signals,transported:true);
        using var full=new GpuAudio(scene,8192,settings,signals);
        var cpu=new TakeRender.Mixer(scene,8192,settings,signals);
        using var solver=new TransportSolver(scene,geometry,settings,RequireGpu(),sampleRate:8192);
        var direct=new TakeRender.DirectPath[2];var a=new float[size*4];var b=new float[a.Length];var c=new float[a.Length];
        double error=0,energy=0;
        for(int frame=0;frame<3;frame++)
        {
            states[1].position.x+=.013f;states[1].rotation=Quaternion.Euler(0,frame*13,0);
            states[2].rotation=Quaternion.Euler(0,-frame*7,0);
            solver.Solve(states,direct);long sample=frame*(long)size;
            packed.SetResponse(sample,frame==2,solver.RayBuffer,0,scene,states);
            full.SetResponse(sample,frame==2,solver.RayBuffer,0,scene,states);
            cpu.SetResponse(sample,direct,states,frame==2,solver.RayPaths,0);
            packed.Render(sample,size,a);full.Render(sample,size,b);cpu.Render(sample,size,c);
            for(int i=0;i<a.Length;i++)
            {
                Assert.That(a[i],Is.EqualTo(b[i]).Within(1e-6),"Storage packing must not change synthesis.");
                double d=a[i]-c[i];error+=d*d;energy+=c[i]*c[i];
            }
        }
        Assert.That(energy,Is.GreaterThan(1e-6));Assert.That(Math.Sqrt(error/energy),Is.LessThan(5e-4));
        if(dispersive)Assert.That(packed.Diagnostics.filterBytes,Is.EqualTo(full.Diagnostics.filterBytes),"Dispersive geometry needs the conservative frequency capacity.");
        else Assert.That(packed.Diagnostics.filterBytes,Is.LessThan(full.Diagnostics.filterBytes));
    }
    // Each edge-to-edge leg multiplies a walk weight by a factor whose absolute mean
    // exceeds one in box scenes, so edge runs stop at the complete-route order:
    // none with complete routes (D/DD are integrated there), one leg (DD) without.
    [TestCase(true),TestCase(false),Category("GPU"),Category("CompleteRoutes")]
    public void WalkEdgeRunsStopAtTheCompleteRouteOrder(bool complete)
    {
        var walls=new[]{(new Vector3(0,0,0),new Vector3(.2f,1.5f,1f)),(new Vector3(0,0,-2.5f),new Vector3(3,2,.2f))};
        var scene=Scene(1+walls.Length);scene.listeners=new[]{1};
        scene.materials=new[]{new AcousticTake.MaterialDefinition {name="rigid",frequencies=new[]{0f},absorption=new[]{0f},scattering=new[]{0f},transmission=new[]{0f}}};
        var state=States(1+walls.Length);state[0].position=new Vector3(-2,1,.3f);state[1].position=new Vector3(2,.8f,-.2f);
        var geometry=new System.Collections.Generic.List<AcousticTake.Primitive>();
        for(int w=0;w<walls.Length;w++)
        {
            int owner=w+2;state[owner].position=walls[w].Item1;state[owner].flags=AcousticTake.StateFlags.Active|AcousticTake.StateFlags.Geometry;
            scene.objects[owner].primitiveStart=geometry.Count;var mesh=TransportBox(scene.dimension,owner,0,1);scene.objects[owner].primitiveCount=mesh.Length;
            foreach(var triangle in mesh)
            {var p=triangle;p.a=Vector3.Scale(p.a,walls[w].Item2);p.b=Vector3.Scale(p.b,walls[w].Item2);p.c=Vector3.Scale(p.c,walls[w].Item2);geometry.Add(p);}
        }
        var settings=EdgeSettings();settings.useBvh=true;settings.airAbsorption=false;settings.fftSize=8;settings.spectralHop=4;
        settings.completeRouteSamples=complete?256:0;settings.raysPerBin=512;settings.maxBounces=8;settings.seed=29;
        settings.maximumDelaySeconds=.1f;settings.minimumDistance=.01f;
        using var solver=new TransportSolver(scene,geometry.ToArray(),settings,RequireGpu(),sampleRate:1000);
        var segments=solver.CaptureRays(state,0,1);
        int per=2+2*settings.maxBounces,longest=0,edges=0;
        for(int ray=0;ray<settings.RaysPerBin;ray++)
            for(int bounce=0,run=0;bounce<settings.maxBounces;bounce++)
            {
                var kind=segments[geometry.Count+1+ray*per+2+2*bounce].kind;
                run=kind==TransportSolver.RayEvent.Diffraction?run+1:0;edges+=run>0?1:0;longest=Math.Max(longest,run);
            }
        Assert.That(edges,Is.GreaterThan(50),"The fixture must sample walk edges.");
        Assert.That(longest,Is.EqualTo(complete?1:2));
        if(!complete)return;
        // Rigid walls never scatter, so with complete routes nothing is left for the walk.
        solver.Solve(state,new TakeRender.DirectPath[1]);
        int walk=settings.FrequencyBins*(1+settings.MirrorCandidates(scene));
        for(int i=walk;i<walk+settings.RaysPerPair*settings.PathSlotsPerRay;i++)
            Assert.That(solver.RayPaths[i].pressureGain==0 && solver.RayPaths[i].pressureImag==0,Is.True,"Rigid walk arrival "+i);
    }
}
