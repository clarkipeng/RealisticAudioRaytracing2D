using NUnit.Framework;
using System;
using System.Collections.Generic;
using System.IO;
using UnityEngine;

public partial class TakeChecks
{
    TakeRender.Settings SpectralSettings() => new TakeRender.Settings {transport=TakeRender.Transport.Raytraced,synthesis=TakeRender.Synthesis.Spectral,audioBackend=TakeRender.AudioBackend.CpuReference,
        responseStepSamples=128,fftSize=128,spectralHop=32,delayFilter=TakeRender.DelayFilter.Linear,raysPerBin=4,maxBounces=3,maximumDelaySeconds=.1f,
        maximumResponsesPerBatch=4,speedOfSound=8000,minimumDistance=1,airAbsorption=false,diffractionFactor=0,tailSamples=256};

    [TestCase(false,false),TestCase(false,true),TestCase(true,false),TestCase(true,true),Category("GPU"),Category("RayDebug")]
    public void RayDebugPreservesArrivals(bool planar,bool bvh)
    {
        var shader=RequireGpu();KnifeEdgeFixture(planar,out var scene,out var geometry,out var state);
        var settings=EdgeSettings();settings.raysPerBin=32;settings.maxBounces=3;settings.useBvh=bvh;
        // Two pair slots ensure the capture selects the requested pair, rather than always pair zero.
        scene.listeners=new[]{0,1};state[0].flags|=AcousticTake.StateFlags.Listener;
        using var solver=new TransportSolver(scene,geometry,settings,shader,sampleRate:8000);
        var direct=new TakeRender.DirectPath[2];solver.Solve(state,direct);
        var expected=(RayTransport.Path[])solver.RayPaths.Clone();
        int bin=10, pairOffset=settings.SpectralSlotsPerPair;
        var segments=solver.CaptureRays(state,1,bin);
        int arrivals=0,edges=0;
        foreach(var s in segments)
        {
            if(s.kind==TransportSolver.RayEvent.None)continue;
            Check(float.IsFinite(s.end.w) && s.end.w>=s.start.w,"Capture has ordered finite propagation times");
            if(s.kind==TransportSolver.RayEvent.Diffraction)edges++;
            if(s.kind!=TransportSolver.RayEvent.Listener)continue;
            arrivals++;Near(Vector3.Distance(s.end,state[1].position),0,"Captured connection ends at selected listener");
            int first=s.path==0?pairOffset+bin:pairOffset+settings.FrequencyBins+
                (bin*settings.RaysPerBin+(int)(s.path-1)/2)*(settings.maxBounces+1);
            bool matches=false;
            int count=s.path==0?1:settings.maxBounces+1;
            for(int i=first;i<first+count;i++)
                matches|=Mathf.Abs(expected[i].delaySeconds-s.end.w)<1e-6f &&
                    Mathf.Abs(expected[i].pressureGain-s.pressure)<1e-6f;
            Check(matches,"Debug listener segment matches the uninstrumented audio arrival");
        }
        Check(arrivals>0 && edges>0,"Capture includes nonzero diffraction connections");
        int start=pairOffset+settings.FrequencyBins+bin*settings.RaysPerBin*(settings.maxBounces+1);
        for(int i=start;i<start+settings.RaysPerBin*(settings.maxBounces+1);i++)
        {
            var actual=solver.RayPaths[i];
            Check(actual.identity==expected[i].identity,"Debugging does not change path sampling");
            Near(actual.pressureGain,expected[i].pressureGain,"Debugging preserves signed gain",1e-6f);
            Near(actual.delaySeconds,expected[i].delaySeconds,"Debugging preserves delay",1e-6f);
        }
        solver.Solve(state,direct);
        Assert.That(solver.RayPaths,Is.EqualTo(expected),"Normal rendering is restored after capture");
    }

    [Test,Category("GPU"),Category("RayDebug")]
    public void RayDebugUsesMaterialTravelTimes()
    {
        var shader=RequireGpu();var scene=Scene(2);scene.dimension=AcousticTake.Dimension.Planar2D;scene.listeners=new[]{1};
        scene.materials=new[]{new AcousticTake.MaterialDefinition {name="slow",frequencies=new[]{0f},
            absorption=new[]{0f},scattering=new[]{0f},transmission=new[]{1f},ior=new[]{2f},damping=new[]{0f}}};
        var geometry=TransportBox(scene.dimension,2,0,1);scene.objects[2].primitiveCount=geometry.Length;
        var state=States(2);state[0].position=Vector3.left*2;state[1].position=Vector3.right*2;state[2].position=Vector3.zero;
        state[2].flags=AcousticTake.StateFlags.Active|AcousticTake.StateFlags.Geometry;
        var settings=SpectralSettings();settings.speedOfSound=1000;
        using var solver=new TransportSolver(scene,geometry,settings,shader,sampleRate:8000);
        var segments=solver.CaptureRays(state,0,1);
        Near(segments[0].end.w,.001f,"Air leg takes one millisecond",1e-7f);
        Near(segments[1].start.w,.001f,"Material starts when air leg ends",1e-7f);
        Near(segments[1].end.w,.005f,"Two metres in half-speed material takes four milliseconds",1e-7f);
        Near(segments[2].end.w,.006f,"Final air leg restores air speed",1e-7f);
        Check(segments[2].kind==TransportSolver.RayEvent.Listener,"Direct capture reaches listener");
        scene.sources[0].mutedListenerPlusOne=1;
        foreach(var segment in solver.CaptureRays(state,0,1))
            Check(segment.kind==TransportSolver.RayEvent.None,"Muted pair leaves no stale segments");
    }

    [Test,Category("GPU"),Category("RayDebug")]
    public void RayDebugCapturesMirrorGeometry()
    {
        var shader=RequireGpu();var scene=Scene(2);scene.dimension=AcousticTake.Dimension.Planar2D;scene.listeners=new[]{1};
        scene.materials=new[]{new AcousticTake.MaterialDefinition {frequencies=new[]{0f},absorption=new[]{0f},
            scattering=new[]{0f},transmission=new[]{0f}}};scene.objects[2].primitiveCount=1;
        var geometry=new[]{new AcousticTake.Primitive {objectIndex=2,a=new Vector3(-3,0,0),b=new Vector3(3,0,0)}};
        var state=States(2);state[0].position=new Vector3(-1,1,0);state[1].position=new Vector3(1,1,0);
        state[2].position=Vector3.zero;state[2].flags=AcousticTake.StateFlags.Active|AcousticTake.StateFlags.Geometry;
        var settings=SpectralSettings();settings.speedOfSound=1000;settings.raysPerBin=1;
        using var solver=new TransportSolver(scene,geometry,settings,shader,sampleRate:8000);
        var segments=solver.CaptureRays(state,0,1);
        Near(Vector3.Distance(segments[2].start,state[0].position),0,"Mirror starts at emitter");
        Near(Vector3.Distance(segments[2].end,Vector3.zero),0,"Mirror hits image-source reflection point");
        Near(segments[2].end.w,Mathf.Sqrt(2)/1000,"Mirror first leg delay",1e-7f);
        Near(segments[3].end.w,2*Mathf.Sqrt(2)/1000,"Mirror full path delay",1e-7f);
        Check(segments[2].kind==TransportSolver.RayEvent.Reflection && segments[3].kind==TransportSolver.RayEvent.Listener,
            "Mirror capture distinguishes surface interaction and listener connection");
    }

    [Test, Category("Core")]
    public void TransportMaterials()
    {
        var scene=Scene();scene.materials=new[]{new AcousticTake.MaterialDefinition {name="glass",frequencies=new[]{100f,5000f},
            absorption=new[]{.1f,.2f},scattering=new[]{.3f,.4f},transmission=new[]{.8f,.6f},ior=new[]{.1f,.2f},damping=new[]{.01f,.02f}}};
        foreach(int version in new[]{1,2,3,4})
        {
            string path=Path.Combine(directory,"version-"+version);
            using(var writer=new AcousticTake.Writer(path,8000,scene,Array.Empty<AcousticTake.Primitive>(),new[]{new[]{1f}},version:version))
            {writer.Append(0,States());writer.Complete();}
            using var take=AcousticTake.Open(path);
            Check(take.Metadata.version==version,"Versioned material read.");
            if(version==1)Check(take.Metadata.scene.materials[0].transmission==null,"V1 does not authenticate new coefficients.");
            else Near(take.Metadata.scene.materials[0].transmission[1],.6f,"V2 transmission round trip");
        }
    }
    [TestCase(0f,0f),TestCase(.5f,0f),TestCase(2f,0f),TestCase(8f,0f)]
    [TestCase(.4f,.3f),TestCase(.4f,-.3f),Category("GPU")]
    public void ImpedanceReflectionMatchesPlaneWave(float real,float imaginary)
    {
        foreach(var dimension in new[]{AcousticTake.Dimension.Planar2D,AcousticTake.Dimension.Spatial3D})
        foreach(float separation in new[]{0f,1f,8f})
        {
            var scene=Scene(2);scene.dimension=dimension;scene.listeners=new[]{1};scene.objects[2].primitiveCount=1;
            scene.materials=new[]{new AcousticTake.MaterialDefinition {frequencies=new[]{0f,125f},
                admittanceReal=new[]{real,real},admittanceImag=new[]{0f,imaginary}}};
            var state=States(2);state[0].position=new Vector3(-1,-separation,0);state[1].position=new Vector3(-1,separation,0);
            state[2].position=Vector3.zero;state[2].flags=AcousticTake.StateFlags.Active|AcousticTake.StateFlags.Geometry;
            var geometry=new[]{new AcousticTake.Primitive {objectIndex=2,a=new Vector3(0,-10,-10),b=new Vector3(0,10,-10),c=new Vector3(0,0,10)}};
            var settings=EdgeSettings();settings.fftSize=8;settings.spectralHop=4;settings.raysPerBin=1;settings.maxBounces=1;settings.diffractionFactor=0;settings.airAbsorption=false;
            using var solver=new TransportSolver(scene,geometry,settings,RequireGpu(),sampleRate:1000);
            solver.Solve(state,new TakeRender.DirectPath[1]);var mirror=solver.RayPaths[settings.FrequencyBins+2];
            var y=new System.Numerics.Complex(real,imaginary);double distance=2*Math.Sqrt(1+separation*separation),cosine=2/distance;
            var expected=(cosine-y)/(cosine+y)/distance;
            Assert.That(mirror.delaySeconds,Is.EqualTo(distance/343).Within(1e-7));
            Assert.That(mirror.pressureGain,Is.EqualTo(expected.Real).Within(1e-6));
            Assert.That(mirror.pressureImag,Is.EqualTo(expected.Imaginary).Within(1e-6));
        }
    }
    [TestCase(false), TestCase(true), Category("GPU")]
    public void SurfaceBranchWeightsPreservePressure(bool planar)
    {
        var scene=Scene(2);scene.listeners=new[]{1};scene.objects[2].primitiveCount=1;
        scene.dimension=planar?AcousticTake.Dimension.Planar2D:AcousticTake.Dimension.Spatial3D;
        scene.materials=new[]{new AcousticTake.MaterialDefinition {frequencies=new[]{0f},
            absorption=new[]{.25f},transmission=new[]{.25f},scattering=new[]{.4f}}};
        var state=States(2);state[0].position=new Vector3(-1,0,0);state[1].position=new Vector3(-2,0,0);
        state[2].position=Vector3.zero;state[2].flags=AcousticTake.StateFlags.Active|AcousticTake.StateFlags.Geometry;
        var geometry=new[]{new AcousticTake.Primitive {objectIndex=2,
            a=new Vector3(0,-1000,-1000),b=new Vector3(0,1000,-1000),c=new Vector3(0,0,1000)}};
        var settings=EdgeSettings();settings.fftSize=8;settings.spectralHop=4;settings.raysPerBin=8192;
        settings.maxBounces=1;settings.diffractionFactor=0;settings.airAbsorption=false;
        using var solver=new TransportSolver(scene,geometry,settings,RequireGpu(),sampleRate:1000);
        var sums=new double[3];int hits=0;double[] energy={.25,.75*.6,.75*.4};
        foreach(var segment in solver.CaptureRays(state,0,1))
        {
            if(segment.path<2 || segment.path%2!=0)continue; // Exclude separately connected mirrors.
            int branch=segment.kind==TransportSolver.RayEvent.Transmission?0:
                segment.kind==TransportSolver.RayEvent.Reflection?1:segment.kind==TransportSolver.RayEvent.Scatter?2:-1;
            if(branch<0)continue;
            Assert.That(segment.pressure*settings.raysPerBin,Is.EqualTo(Math.Sqrt(.75/energy[branch])).Within(1e-5),
                "Surviving pressure is coefficient divided by branch probability.");
            hits++;sums[branch]+=segment.pressure;
        }
        Assert.That(hits,Is.GreaterThan(2000));
        TestContext.WriteLine($"hits={hits}; pressure T/R/S={sums[0]*settings.raysPerBin/hits}, {sums[1]*settings.raysPerBin/hits}, {sums[2]*settings.raysPerBin/hits}");
        for(int i=0;i<3;i++)Assert.That(sums[i]*settings.raysPerBin/hits,
            Is.EqualTo(Math.Sqrt(.75*energy[i])).Within(.035),"Each sampled branch estimates its pressure coefficient.");
    }

    [Test, Category("GPU")]
    public void TransportGeometry()
    {
        var shader=RequireGpu();
        var scene=Scene(2);scene.listeners=new[]{1};scene.objects[2].primitiveCount=1;
        scene.materials=new[]{new AcousticTake.MaterialDefinition {name="wall",frequencies=new[]{0f},absorption=new[]{0f},scattering=new[]{.5f},transmission=new[]{0f},ior=new[]{1f},damping=new[]{0f}}};
        var state=States(2);state[0].position=new Vector3(-1,0,0);state[1].position=new Vector3(1,0,0);state[2].position=Vector3.zero;
        state[2].flags=AcousticTake.StateFlags.Active|AcousticTake.StateFlags.Geometry;
        var geometry=new[]{new AcousticTake.Primitive {objectIndex=2,a=new Vector3(0,-2,-2),b=new Vector3(0,2,-2),c=new Vector3(0,0,2)}};
        var settings=SpectralSettings();settings.speedOfSound=343;
        var direct=new TakeRender.DirectPath[1];
        foreach(var dimension in new[]{AcousticTake.Dimension.Spatial3D,AcousticTake.Dimension.Planar2D})
        {
            scene.dimension=dimension;
            using(var solver=new TransportSolver(scene,geometry,settings,shader,sampleRate:8000))
            {
                solver.Solve(state,direct);Near(direct[0].pressureGain,0,"Opaque diagnostic is blocked");
                var rays=solver.RayPaths;Near(rays[10].pressureGain,0,"Opaque spectral direct is blocked");
            }
            scene.materials[0].transmission[0]=1;
            using(var solver=new TransportSolver(scene,geometry,settings,shader,sampleRate:8000))
            {
                solver.Solve(state,direct);var rays=solver.RayPaths;
                Near(rays[10].pressureGain,.5f,"Transparent wall passes pressure");
            }
            scene.materials[0].transmission[0]=0;
            {
                settings.diffractionFactor=.5f;
                using var edgeSolver=new TransportSolver(scene,geometry,settings,shader,sampleRate:8000);
                edgeSolver.Solve(state,direct);var edgeRays=edgeSolver.RayPaths;double edgeEnergy=0;
                for(int i=65;i<edgeRays.Length;i++)edgeEnergy+=Math.Pow(edgeRays[i].pressureGain,2);
                Check(edgeEnergy>0,"Wavelength edge paths carry sound around an opaque geometry");settings.diffractionFactor=0;
            }
            state[1].position=new Vector3(-1,1,0);
            RayTransport.Path[] baseline=null;
            foreach(bool bvh in new[]{false,true})
            {
                settings.useBvh=bvh;
                using var solver=new TransportSolver(scene,geometry,settings,shader,2,8000);
                solver.Solve(state,direct);var rays=solver.RayPaths;double energy=0;
                for(int i=65;i<settings.SpectralSlotsPerPair;i++)
                {
                    float value=rays[i].pressureGain;Check(float.IsFinite(value),"Finite indirect pressure");energy+=value*value;
                    if(baseline!=null)Near(value,baseline[i].pressureGain,"BVH indirect agreement",1e-5f);
                }
                Check(energy>0,"Reflective wall produces indirect arrivals in "+dimension);
                if(!bvh)baseline=(RayTransport.Path[])rays.Clone();
                // A moving wall retains its local hierarchy and remains acoustically visible.
                state[2].position=Vector3.right;solver.Solve(state,direct);state[2].position=Vector3.zero;
                if(bvh)Check(solver.BvhPrimitiveCount==geometry.Length,"Moving reflector retains its mesh tree");
            }
            state[1].position=Vector3.right;settings.useBvh=false;
            scene.materials[0].absorption[0]=1;
            using(var solver=new TransportSolver(scene,geometry,settings,shader,sampleRate:8000))
            {
                solver.Solve(state,direct);var rays=solver.RayPaths;
                for(int i=0;i<rays.Length;i++)Near(rays[i].pressureGain,0,"Absorber cannot create reflected pressure");
            }
            scene.materials[0].absorption[0]=0;
        }
        // World-space arrival direction survives transport; ear interpolation uses
        // each listener's orientation in the shared DSP, rather than GPU snapping.
        scene.dimension=AcousticTake.Dimension.Spatial3D;scene.objects[2].primitiveCount=0;
        foreach(float turn in new[]{0f,90f,180f,270f})
        {
            state[1].rotation=Quaternion.Euler(0,turn,0);
            using var solver=new TransportSolver(scene,Array.Empty<AcousticTake.Primitive>(),settings,shader,sampleRate:8000);
            solver.Solve(state,direct);var rays=solver.RayPaths;
            Near(Vector3.Distance((Vector3)rays[10].direction,(Vector3)direct[0].direction),0,"Transport retains world arrival direction");
        }
    }
    AcousticTake.Primitive[] TransportBox(AcousticTake.Dimension dimension,int owner,int material,float half)
    {
        var geometry=new List<AcousticTake.Primitive>();
        if(dimension==AcousticTake.Dimension.Planar2D)
        {
            var v=new[]{new Vector3(-half,-half,0),new Vector3(half,-half,0),new Vector3(half,half,0),new Vector3(-half,half,0)};
            for(int i=0;i<4;i++)geometry.Add(new AcousticTake.Primitive {objectIndex=owner,materialIndex=material,a=v[i],b=v[(i+1)%4]});
        }
        else
        {
            var v=new Vector3[8];for(int i=0;i<8;i++)v[i]=new Vector3((i&1)==0?-half:half,(i&2)==0?-half:half,(i&4)==0?-half:half);
            int[] indices={0,2,3,0,3,1,4,5,7,4,7,6,0,1,5,0,5,4,2,6,7,2,7,3,0,4,6,0,6,2,1,3,7,1,7,5};
            for(int i=0;i<indices.Length;i+=3)geometry.Add(new AcousticTake.Primitive {objectIndex=owner,materialIndex=material,a=v[indices[i]],b=v[indices[i+1]],c=v[indices[i+2]]});
        }
        return geometry.ToArray();
    }
    [Test, Category("GPU")]
    public void TransportMedia()
    {
        var shader=RequireGpu();
        foreach(var dimension in new[]{AcousticTake.Dimension.Planar2D,AcousticTake.Dimension.Spatial3D})foreach(bool bvh in new[]{false,true})
        {
            var scene=Scene(3);scene.dimension=dimension;scene.listeners=new[]{1};
            scene.materials=new[]{
                new AcousticTake.MaterialDefinition {frequencies=new[]{0f},absorption=new[]{0f},scattering=new[]{0f},transmission=new[]{1f},ior=new[]{.5f},damping=new[]{.1f}},
                new AcousticTake.MaterialDefinition {frequencies=new[]{0f},absorption=new[]{0f},scattering=new[]{0f},transmission=new[]{1f},ior=new[]{2f},damping=new[]{.3f}}};
            var outer=TransportBox(dimension,2,0,2);var inner=TransportBox(dimension,3,1,1);
            var geometry=new List<AcousticTake.Primitive>(outer);geometry.AddRange(inner);
            scene.objects[2].primitiveCount=outer.Length;scene.objects[3].primitiveStart=outer.Length;scene.objects[3].primitiveCount=inner.Length;
            var state=States(3);state[2].position=state[3].position=Vector3.zero;
            state[2].flags=state[3].flags=AcousticTake.StateFlags.Active|AcousticTake.StateFlags.Geometry;
            var settings=SpectralSettings();settings.speedOfSound=1000;settings.useBvh=bvh;settings.maxBounces=1;
            void Direct(float source,float listener,float time,float attenuation,string name)
            {
                state[0].position=Vector3.right*source;state[1].position=Vector3.right*listener;
                using var solver=new TransportSolver(scene,geometry.ToArray(),settings,shader,sampleRate:8000);
                solver.Solve(state,new TakeRender.DirectPath[1]);var rays=solver.RayPaths;
                Near(rays[10].delaySeconds,time,name+" travel time "+dimension,1e-7f);
                Near(rays[10].pressureGain,Mathf.Exp(-attenuation)/Mathf.Max(1,Mathf.Abs(listener-source)),name+" pressure loss",1e-5f);
            }
            Direct(-3,3,.007f,.8f,"Nested slabs restore outer medium");
            Direct(0,3,.0035f,.4f,"Source starts inside inner solid");
            Direct(-3,0,.0035f,.4f,"Listener ends inside inner solid");
            Direct(-.5f,.5f,.002f,.3f,"Both endpoints inside solid");
            state[3].flags=0;Direct(-3,3,.004f,.4f,"Inactive inner medium is absent");
            state[2].position=Vector3.right*5;Direct(-3,3,.006f,0,"Moved volume no longer intersects path");
            state[2].position=Vector3.zero;
            // Reverse all winding: geometry still bounds the same physical medium.
            for(int i=0;i<geometry.Count;i++){var p=geometry[i];(p.a,p.b)=(p.b,p.a);geometry[i]=p;}
            Direct(-3,3,.004f,.4f,"Reversed closed winding");
            // A sheet has zero thickness and must not turn the rest of space into its material.
            geometry.RemoveRange(1,geometry.Count-1);scene.objects[2].primitiveCount=1;scene.objects[3].primitiveCount=0;
            geometry[0]=new AcousticTake.Primitive {objectIndex=2,a=new Vector3(0,-4,-4),b=new Vector3(0,4,-4),c=new Vector3(0,0,4)};
            Direct(-3,3,.006f,0,"Open sheet leaves air speed unchanged");
            // A reflected path entirely inside one solid uses its speed and damping
            // for both legs, including the final connection to the listener.
            geometry=new List<AcousticTake.Primitive>(TransportBox(dimension,2,0,1));scene.objects[2].primitiveCount=geometry.Count;
            scene.materials[0].transmission[0]=0;state[0].position=state[1].position=Vector3.zero;
            using var reflected=new TransportSolver(scene,geometry.ToArray(),settings,shader,sampleRate:8000);
            reflected.Solve(state,new TakeRender.DirectPath[1]);var paths=reflected.RayPaths;int audible=0;
            for(int i=65;i<paths.Length;i++)
            {
                float gain=paths[i].pressureGain;if(gain==0)continue;audible++;
                var direction=(Vector3)paths[i].direction;float cosine=Mathf.Max(Mathf.Abs(direction.x),Mathf.Abs(direction.y),Mathf.Abs(direction.z)),distance=1/cosine;
                Near(paths[i].delaySeconds,distance/1000,"Reflected interior travel time",1e-7f);
                Check(float.IsFinite(gain) && gain>=0,"Finite attenuated interior reflection");
            }
            Check(audible>0,"Interior reflection fixture exercised actual rays");
        }
    }
    [TestCase(false),TestCase(true),Category("GPU")]
    public void MediumBvhParity(bool planar)
    {
        var shader=RequireGpu();var scene=Scene(4);scene.dimension=planar?AcousticTake.Dimension.Planar2D:AcousticTake.Dimension.Spatial3D;scene.listeners=new[]{1};
        scene.materials=new[]{
            new AcousticTake.MaterialDefinition {frequencies=new[]{0f},transmission=new[]{1f},ior=new[]{.5f},damping=new[]{.1f}},
            new AcousticTake.MaterialDefinition {frequencies=new[]{0f},transmission=new[]{1f},ior=new[]{2f},damping=new[]{.3f}}};
        var geometry=new List<AcousticTake.Primitive>();var state=States(4);
        for(int owner=2;owner<=4;owner++)
        {
            var box=TransportBox(scene.dimension,owner,owner==2?0:1,owner==4?.7f:2);
            scene.objects[owner].primitiveStart=geometry.Count;scene.objects[owner].primitiveCount=box.Length;
            foreach(var triangle in box){var t=triangle;if(owner==3)(t.a,t.b)=(t.b,t.a);geometry.Add(t);}
            state[owner].position=Vector3.zero;state[owner].flags=AcousticTake.StateFlags.Active|AcousticTake.StateFlags.Geometry;
        }
        TakeRender.Settings Settings(bool accelerated)
        {var s=EdgeSettings();s.fftSize=4;s.spectralHop=2;s.raysPerBin=4;s.maxBounces=2;s.diffractionFactor=0;s.airAbsorption=false;s.useBvh=accelerated;return s;}
        using var linear=new TransportSolver(scene,geometry.ToArray(),Settings(false),shader,3,sampleRate:500);
        using var bvh=new TransportSolver(scene,geometry.ToArray(),Settings(true),shader,3,sampleRate:500);
        var frames=new AcousticTake.State[3*state.Length];var a=new TakeRender.DirectPath[3];var b=new TakeRender.DirectPath[3];
        for(int frame=0;frame<12;frame++)
        {
            state[0].position=new Vector3(-3.5f+frame*.4f,.1f,0);state[1].position=new Vector3(3,.3f,planar?0:.2f);
            state[2].rotation=Quaternion.Euler(planar?0:frame*3,planar?0:frame*5,frame*7);
            state[3].position=new Vector3(.4f,frame%3==0?5:0,0);state[3].rotation=Quaternion.Euler(0,0,-frame*11);
            state[4].flags=frame%4==0?0:AcousticTake.StateFlags.Active|AcousticTake.StateFlags.Geometry;
            Array.Copy(state,0,frames,(frame%3)*state.Length,state.Length);
            if(frame%3!=2)continue;
            linear.BeginSolve(frames,3);bvh.BeginSolve(frames,3);Assert.That(linear.TryComplete(a,true)&&bvh.TryComplete(b,true),Is.True);
            var expected=linear.RayPaths;var actual=bvh.RayPaths;Assert.That(actual.Length,Is.EqualTo(expected.Length));
            for(int i=0;i<actual.Length;i++)
            {
                Near(actual[i].delaySeconds,expected[i].delaySeconds,"BVH medium query preserves delay through moving nested/overlapping solids",1e-6f);
                Near(actual[i].pressureGain,expected[i].pressureGain,"BVH medium query preserves attenuation and equal-volume owner priority",1e-5f);
            }
        }
    }
    [TestCase(false),TestCase(true),Category("GPU")]
    public void TransportNativeVolumes(bool bvh)
    {
        var shader=RequireGpu();
        var material=ScriptableObject.CreateInstance<AudioMaterial>();
        try
        {
            foreach(var type in new[]{typeof(BoxCollider),typeof(SphereCollider),typeof(CapsuleCollider),typeof(BoxCollider2D),typeof(CircleCollider2D),typeof(CapsuleCollider2D)})
            {
                var obj=new GameObject("Volume fixture");
                try
                {
                    obj.AddComponent(type);var dimension=typeof(Collider2D).IsAssignableFrom(type)?AcousticTake.Dimension.Planar2D:AcousticTake.Dimension.Spatial3D;
                    var bindings=new List<RayTraceManager.TakeObjectBinding> {new RayTraceManager.TakeObjectBinding {id="solid",transform=obj.transform,geometry=true}};
                    foreach(float scale in new[]{1f,-1f})
                    {
                        obj.transform.localScale=new Vector3(scale,1,1);
                        // Curved 3D collider export requires uniform absolute scale.
                        var exported=SceneToTake.Compile(bindings,dimension,.01f,material,Array.Empty<Transform>(),Array.Empty<AudioClip>(),Array.Empty<Transform>(),out var geometry);
                        var scene=Scene(2);scene.dimension=dimension;scene.listeners=new[]{1};scene.objects[2].primitiveCount=geometry.Length;
                        for(int i=0;i<geometry.Length;i++)geometry[i].objectIndex=2;
                        scene.materials=new[]{new AcousticTake.MaterialDefinition {frequencies=new[]{0f,4000f},absorption=new[]{0f,0f},scattering=new[]{0f,0f},
                            transmission=new[]{1f,1f},ior=new[]{.5f,1.5f},damping=new[]{.1f,.1f}}};
                        var state=States(2);state[0].position=Vector3.down*.1f;state[1].position=Vector3.up*.1f;state[2].position=Vector3.zero;
                        state[2].flags=AcousticTake.StateFlags.Active|AcousticTake.StateFlags.Geometry;
                        var settings=SpectralSettings();settings.speedOfSound=1000;settings.useBvh=bvh;
                        using var solver=new TransportSolver(scene,geometry,settings,shader,sampleRate:8000);
                        solver.Solve(state,new TakeRender.DirectPath[1]);var paths=solver.RayPaths;
                        foreach(int bin in new[]{0,16,32,64})
                        {
                            Near(paths[bin].delaySeconds,.2f*(.5f+bin/64f)/1000,"Native "+type.Name+" volume, mirrored scale and frequency-dependent speed",1e-7f);
                            Near(paths[bin].pressureGain,Mathf.Exp(-.02f),"Native volume damping",1e-5f);
                        }
                    }
                }
                finally {UnityEngine.Object.DestroyImmediate(obj);}
            }
        }
        finally {UnityEngine.Object.DestroyImmediate(material);}
    }
    // Independent time-domain oracle: no FFT/window formula in the expected output.
    [Test, Pairwise, Category("Core")]
    public void SpectralConvolutionChecks(
        [Values(64,128)] int fft,[Values(2,4)] int divisor,[Values(false,true)] bool ears,
        [Values(0f,1f,31f,32f,63f,129f,31.25f)] float delay,
        [Values(TakeRender.DelayFilter.Linear,TakeRender.DelayFilter.WindowedSinc)] TakeRender.DelayFilter filter,
        [Values(TakeRender.Synthesis.Spectral,TakeRender.Synthesis.ImpulseResponse)] TakeRender.Synthesis method)
    {
        var scene=Scene();var state=States();
        int hop=fft/divisor;
        var settings=SpectralSettings();settings.fftSize=fft;settings.spectralHop=hop;settings.responseStepSamples=128;
        settings.synthesis=method;settings.delayFilter=filter;settings.raysPerBin=1;settings.maxBounces=1;settings.listenerOutput=ears?TakeRender.ListenerOutput.Binaural:TakeRender.ListenerOutput.Mono;
        var pcm=new float[256];pcm[31]=1;pcm[127]=-.3f;
        var signal=new TakeRender.Signal(pcm);signal.SetActive(0,true);signal.Append(new AcousticTake.VoiceEvent{action=AcousticTake.VoiceAction.Play,gain=1});
        var mixer=new TakeRender.Mixer(scene,8192,settings,new[]{signal});
        var arrivals=new RayTransport.Path[settings.SpectralSlotsPerPair];
        for(int b=0;b<settings.FrequencyBins;b++)
            arrivals[b]=new RayTransport.Path {frequencyBin=(uint)b,pressureGain=1,delaySeconds=delay/8192,direction=Vector3.forward};
        var output=new float[768*settings.ChannelsPerListener];var block=new float[128*settings.ChannelsPerListener];
        for(int at=0;at<768;at+=128)
        {
            mixer.SetResponse((long)at,new TakeRender.DirectPath[1],state,false,arrivals,0);
            // Changing callback chunk boundaries must not change convolution.
            for(int n=0;n<128;n+=16){mixer.Render(at+n,16,block);Array.Copy(block,0,output,(at+n)*settings.ChannelsPerListener,16*settings.ChannelsPerListener);}
        }
        var expected=new float[output.Length];var hrtf=ears?Hrtf.Load(8192):null;var blend=ears?hrtf.Interpolate(Vector3.forward):default;
        int whole=(int)delay;double fraction=delay-whole;
        int begin=filter==TakeRender.DelayFilter.Linear?0:1-settings.sincRadius;
        int end=filter==TakeRender.DelayFilter.Linear?1:settings.sincRadius;
        var taps=new double[end-begin+1];double normalization=0;
        for(int k=begin;k<=end;k++)
        {
            double x=k-fraction;
            double value=filter==TakeRender.DelayFilter.Linear?(k==0?1-fraction:fraction):
                Math.Abs(x)<1e-12?1:Math.Sin(Math.PI*x)/(Math.PI*x)*Math.Sin(Math.PI*x/settings.sincRadius)/(Math.PI*x/settings.sincRadius);
            taps[k-begin]=value;normalization+=value;
        }
        for(int n=0;n<pcm.Length;n++)for(int ear=0;ear<settings.ChannelsPerListener;ear++)for(int t=0;t<(hrtf?.Taps??1);t++)
        for(int k=begin;k<=end;k++)
        {
            int index=n+settings.DspLatencySamples+whole+t+k;double value=pcm[n]*(ears?hrtf.Coefficient(blend,ear,t):1);
            if(index>=0 && index<768)expected[index*settings.ChannelsPerListener+ear]+=(float)(value*taps[k-begin]/normalization);
        }
        float error=0;for(int i=0;i<output.Length;i++)error=Mathf.Max(error,Mathf.Abs(output[i]-expected[i]));
        Near(error,0,"Independent delayed FIR oracle: FFT "+fft+" hop "+hop+" ears "+ears+" delay "+delay+" filter "+filter,2e-5f);
    }

    [Test, Category("Core")]
    public void ReplacedArrivalKeepsPendingAudio()
    {
        var scene=Scene();var state=States();
        // Reusing an indirect slot for an unrelated longer arrival must not discard it as negative velocity.
        var changedSettings=SpectralSettings();changedSettings.raysPerBin=1;changedSettings.maxBounces=1;
        var pulse=new float[768];pulse[400]=1;var changedSignal=new TakeRender.Signal(pulse);
        changedSignal.SetActive(0,true);changedSignal.Append(new AcousticTake.VoiceEvent{action=AcousticTake.VoiceAction.Play,gain=1});
        var changedMixer=new TakeRender.Mixer(scene,8192,changedSettings,new[]{changedSignal});
        var changedPaths=new RayTransport.Path[changedSettings.SpectralSlotsPerPair];var changedOutput=new float[1024];
        foreach(int at in new[]{0,128})
        {
            for(int bin=0;bin<changedSettings.FrequencyBins;bin++)
                changedPaths[changedSettings.FrequencyBins+bin*(changedSettings.maxBounces+1)]=
                    new RayTransport.Path {frequencyBin=(uint)bin,pressureGain=1,delaySeconds=at==0?0:256f/8192};
            changedMixer.SetResponse((long)at,new TakeRender.DirectPath[1],state,false,changedPaths,0);
            int length=at==0?128:1024;var chunk=new float[128];
            for(int n=0;n<length;n+=128){changedMixer.Render(at+n,128,chunk);Array.Copy(chunk,0,changedOutput,n,128);}
        }
        Near(changedOutput[400+changedSettings.DspLatencySamples+256-128],1,"Unrelated replacement arrival remains audible",2e-5f);
    }

    [Test, Category("Core")]
    public void LongHrirConvolvesFullResponse()
    {
        var scene=Scene();var state=States();
        // A single-bin finite response with HRIR longer than the acoustic grid exercises the ungrouped path.
        var shortGrid=SpectralSettings();shortGrid.fftSize=64;shortGrid.spectralHop=16;shortGrid.raysPerBin=1;
        shortGrid.listenerOutput=TakeRender.ListenerOutput.Binaural;
        var earSignal=new TakeRender.Signal(new[]{1f});earSignal.SetActive(0,true);
        earSignal.Append(new AcousticTake.VoiceEvent{action=AcousticTake.VoiceAction.Play,gain=1});
        var earMixer=new TakeRender.Mixer(scene,44100,shortGrid,new[]{earSignal});
        var earPaths=new RayTransport.Path[shortGrid.SpectralSlotsPerPair];
        earPaths[7]=new RayTransport.Path {frequencyBin=7,pressureGain=.75f,delaySeconds=31.25f/44100,direction=Vector3.right};
        earMixer.SetResponse(0L,new TakeRender.DirectPath[1],state,false,earPaths,0);
        var earOutput=new float[1024];var earBlock=new float[256];
        for(int n=0;n<512;n+=128){earMixer.Render(n,128,earBlock);Array.Copy(earBlock,0,earOutput,n*2,256);}
        var earExpected=new double[1024];var longHrtf=Hrtf.Load(44100);var right=longHrtf.Interpolate(Vector3.right);
        double exactDelay=(double)(31.25f/44100)*44100;int baseDelay=(int)Math.Floor(exactDelay);double sub=exactDelay-baseDelay;
        for(int n=0;n<64;n++)for(int t=0;t<longHrtf.Taps;t++)for(int ear=0;ear<2;ear++)
        {
            double value=.75*2/64*Math.Cos(2*Math.PI*7*(n-32)/64)*longHrtf.Coefficient(right,ear,t);
            int at=baseDelay+32+n+t;earExpected[at*2+ear]+=value*(1-sub);earExpected[(at+1)*2+ear]+=value*sub;
        }
        double earError=0;for(int n=0;n<earOutput.Length;n++)earError=Math.Max(earError,Math.Abs(earOutput[n]-earExpected[n]));
        Near((float)earError,0,"Full long HRIR times finite single-bin response",2e-5f);
    }

    [Test, Category("Core")]
    public void ReceiverDisableCompletesFade()
    {
        var scene=Scene();var state=States();
        // Disabling a receiver fades output once. A subsequent disabled response cannot clip pending windows.
        var options=SpectralSettings();options.transitionSamples=64;options.raysPerBin=1;options.maxBounces=1;
        var constant=new TakeRender.Signal(new[]{1f});constant.SetActive(0,true);constant.Append(new AcousticTake.VoiceEvent{action=AcousticTake.VoiceAction.Play,gain=1,loop=true});
        var renderer=new TakeRender.Mixer(scene,8192,options,new[]{constant});var paths=new RayTransport.Path[options.SpectralSlotsPerPair];
        for(int b=0;b<options.FrequencyBins;b++)paths[b]=new RayTransport.Path {frequencyBin=(uint)b,pressureGain=1};
        var samples=new float[128];
        for(int at=0;at<768;at+=128)
        {
            state[1].flags=at>=512?0:AcousticTake.StateFlags.Active|AcousticTake.StateFlags.Listener;
            renderer.SetResponse((long)at,new TakeRender.DirectPath[1],state,false,paths,0);renderer.Render(at,128,samples);
            if(at>=512)for(int n=0;n<128;n++)Near(samples[n],Mathf.Clamp01(1-(at+n-512)/128f),"Receiver activity follows the acoustic response interval",2e-5f);
        }
    }
    [Test, Category("GPU")]
    public void MirrorChecks()
    {
        var shader=RequireGpu();
        foreach(var dimension in new[]{AcousticTake.Dimension.Spatial3D,AcousticTake.Dimension.Planar2D})
        {
            var scene=Scene(2);scene.dimension=dimension;scene.listeners=new[]{1};scene.objects[2].primitiveCount=1;
            scene.materials=new[]{new AcousticTake.MaterialDefinition{frequencies=new[]{0f},absorption=new[]{0f},scattering=new[]{0f},transmission=new[]{0f},ior=new[]{1f},damping=new[]{0f}}};
            var geometry=new[]{new AcousticTake.Primitive{objectIndex=2,a=new Vector3(0,-10,-10),b=new Vector3(0,10,-10),c=new Vector3(0,0,10)}};
            var state=States(2);state[0].position=new Vector3(-1,-1,0);state[1].position=new Vector3(-1,1,0);
            state[2].position=Vector3.zero;state[2].flags=AcousticTake.StateFlags.Active|AcousticTake.StateFlags.Geometry;
            var settings=SpectralSettings();settings.speedOfSound=343;settings.raysPerBin=1;settings.maxBounces=1;
            foreach(float scattering in new[]{0f,1f})
            {
                scene.materials[0].scattering[0]=scattering;
                using var solver=new TransportSolver(scene,geometry,settings,shader,sampleRate:8192);
                solver.Solve(state,new TakeRender.DirectPath[1]);var rays=solver.RayPaths;
                for(int bin=0;bin<settings.FrequencyBins;bin++)
                {
                    int at=settings.FrequencyBins+bin*(settings.maxBounces+1);
                    Near(rays[at].pressureGain,scattering==0?1/Mathf.Sqrt(8):0,"Image-source specular gain",1e-5f);
                    if(scattering==0)Near(rays[at].delaySeconds,Mathf.Sqrt(8)/343,"Image-source specular time",1e-6f);
                }
            }
        }
        // A square's diagonal belongs to two triangles, but the analytic mirror path is counted once.
        var square=Scene(2);square.listeners=new[]{1};square.objects[2].primitiveCount=2;
        square.materials=new[]{new AcousticTake.MaterialDefinition{frequencies=new[]{0f},absorption=new[]{0f},scattering=new[]{0f},transmission=new[]{0f},ior=new[]{1f},damping=new[]{0f}}};
        var faces=new[]{new AcousticTake.Primitive{objectIndex=2,a=new Vector3(0,-10,-10),b=new Vector3(0,10,-10),c=new Vector3(0,10,10)},
            new AcousticTake.Primitive{objectIndex=2,a=new Vector3(0,-10,-10),b=new Vector3(0,10,10),c=new Vector3(0,-10,10)}};
        var poses=States(2);poses[0].position=new Vector3(-1,-1,0);poses[1].position=new Vector3(-1,1,0);
        poses[2].position=Vector3.zero;poses[2].flags=AcousticTake.StateFlags.Active|AcousticTake.StateFlags.Geometry;
        foreach(int budget in new[]{2,4})foreach(int seed in new[]{0,17})
        {
            var settings=SpectralSettings();settings.speedOfSound=343;settings.raysPerBin=budget;settings.seed=seed;settings.maxBounces=1;
            using var solver=new TransportSolver(square,faces,settings,shader,sampleRate:8192);
            solver.Solve(poses,new TakeRender.DirectPath[1]);var rays=solver.RayPaths;
            for(int bin=0;bin<settings.FrequencyBins;bin++)
            {
                float gain=0;for(int ray=0;ray<budget;ray++)gain+=rays[settings.FrequencyBins+(bin*budget+ray)*2].pressureGain;
                Near(gain,1/Mathf.Sqrt(8),"Shared-edge mirror gain is invariant over complete primitive strata",1e-5f);
            }
        }
        foreach(int fft in new[]{128,1024,2048})foreach(int rays in new[]{1,2,3})
        {
            var settings=SpectralSettings();settings.fftSize=fft;settings.raysPerBin=rays;
            Check(settings.RaysPerPair==settings.FrequencyBins*rays,"Every configured bin gets its requested samples");
        }
    }
    [TestCase(false),TestCase(true),Category("GPU")]
    public void SpectralTransitions(bool accelerated)
    {
        var shader=RequireGpu();var scene=Scene();var state=States();state[1].position=Vector3.zero;
        foreach(bool cut in new[]{false,true})
        {
            var settings=SpectralSettings();settings.responseStepSamples=512;
            var signal=new TakeRender.Signal(new[]{1f});signal.SetActive(0,true);signal.Append(new AcousticTake.VoiceEvent{gain=1,loop=true});
            var mixer=accelerated?null:new TakeRender.Mixer(scene,8000,settings,new[]{signal});
            using var gpu=accelerated?new GpuAudio(scene,8000,settings,new[]{signal}):null;
            using var solver=new TransportSolver(scene,Array.Empty<AcousticTake.Primitive>(),settings,shader,sampleRate:8000);
            var direct=new TakeRender.DirectPath[1];solver.Solve(state,direct);var rays=solver.RayPaths;
            using var arrivals=new ComputeBuffer(rays.Length,RayTransport.PathStride);var output=new float[512];
            for(int sample=0;sample<1536;sample+=512)
            {
                if(sample>0)Array.Clear(rays,0,rays.Length);arrivals.SetData(rays);
                long anchor=settings.ResponseSample(sample);
                if(accelerated){gpu.SetResponse(sample,cut && sample==512,arrivals,0,scene,state,anchor);gpu.Render(sample,512,output);}
                else {mixer.SetResponse(sample,direct,state,cut && sample==512,rays,0,anchor);mixer.Render(sample,512,output);}
                if(sample==0)continue;
                for(int n=0;n<512;n++)Near(output[n],sample==1024?0:cut?1:1-n/512f,"Acoustic gain follows arrival time; a cut never anticipates its knot.",2e-5f);
            }
        }
    }
    [Test, Category("GPU")]
    public void SpectralHeadTurns()
    {
        var shader=RequireGpu();
        var scene=Scene();var state=States();state[0].position=Vector3.forward;state[1].position=Vector3.zero;
        var settings=SpectralSettings();settings.listenerOutput=TakeRender.ListenerOutput.Binaural;settings.responseStepSamples=512;
        float[] Render(float turn)
        {
            state[1].rotation=Quaternion.Euler(0,turn,0);
            var signal=new TakeRender.Signal(new[]{1f});signal.SetActive(0,true);signal.Append(new AcousticTake.VoiceEvent {action=AcousticTake.VoiceAction.Play,gain=1});
            var mixer=new TakeRender.Mixer(scene,8000,settings,new[]{signal});
            using var solver=new TransportSolver(scene,Array.Empty<AcousticTake.Primitive>(),settings,shader,sampleRate:8000);
            var direct=new TakeRender.DirectPath[1];solver.Solve(state,direct);

            mixer.SetResponse(0L,direct,state,false,solver.RayPaths,0);var output=new float[1024];mixer.Render(0,512,output);return output;
        }
        var a=Render(2.499f);var b=Render(2.501f);float jump=0,signalPeak=0;
        for(int i=0;i<a.Length;i++){jump=Mathf.Max(jump,Mathf.Abs(a[i]-b[i]));signalPeak=Mathf.Max(signalPeak,Mathf.Abs(a[i]));}
        Check(signalPeak>.001f && jump<signalPeak*.001f,"Spectral ear output is continuous across the old nearest-direction boundary");
        a=Render(90);b=Render(-90);float symmetry=0;
        for(int i=0;i<a.Length;i+=2){symmetry=Mathf.Max(symmetry,Mathf.Abs(a[i]-b[i+1]));symmetry=Mathf.Max(symmetry,Mathf.Abs(a[i+1]-b[i]));}
        Near(symmetry,0,"Spectral head rotation preserves left/right convention",1e-6f);
    }
    [Test, Category("GPU")]
    public void SpectralIdentity()
    {
        var shader=RequireGpu();
        foreach(int fft in new[]{128,256})foreach(int divisor in new[]{2,4})
        {
            var settings=SpectralSettings();settings.fftSize=fft;settings.spectralHop=fft/divisor;settings.responseStepSamples=256;settings.tailSamples=512;
            var scene=Scene();var state=States();state[1].position=Vector3.zero;
            var pcm=new float[2048];for(int i=0;i<pcm.Length;i++)pcm[i]=.2f*Mathf.Sin(2*Mathf.PI*503*i/8000)+.1f*Mathf.Cos(2*Mathf.PI*1107*i/8000);
            string path=Path.Combine(directory,"identity-"+fft+"-"+divisor);
            using(var writer=new AcousticTake.Writer(path,8000,scene,Array.Empty<AcousticTake.Primitive>(),new[]{pcm}))
            {writer.Append(0,state);writer.Append(2047,state);writer.Append(new AcousticTake.VoiceEvent {sample=0,gain=1,action=AcousticTake.VoiceAction.Play});writer.Complete();}
            using var take=AcousticTake.Open(path);var report=TakeRender.RenderDirect(take,settings,path+"-render",shader:shader);
            var output=ReadWave(Path.Combine(path+"-render","listener-0.wav"));float error=0;
            for(int i=0;i<output.Length;i++){float expected=i>=fft && i<fft+pcm.Length?pcm[i-fft]:0;error=Mathf.Max(error,Mathf.Abs(output[i]-expected));}
            Near(error,0,"Partitioned convolution identity gain, latency and complete tail",2e-5f);Check(report.dspLatencySamples==fft,"Declared FFT latency");
        }
    }
    [Test, Category("GPU")]
    public void SpectralDoppler()
    {
        var shader=RequireGpu();
        foreach(int motion in new[]{-1,1})
        {
            var scene=Scene();var state=States();state[1].position=new Vector3(motion>0?10:80,0,0);
            float initial=state[1].position.x;var pcm=new float[256];
            for(int i=0;i<pcm.Length;i++)pcm[i]=Mathf.Sin(2*Mathf.PI*32*i/256);
            string path=Path.Combine(directory,"doppler-"+motion);
            using(var writer=new AcousticTake.Writer(path,8000,scene,Array.Empty<AcousticTake.Primitive>(),new[]{pcm}))
            {
                writer.Append(0,state);state[1].position.x=initial+motion*68.6f;writer.Append(16000,state);
                writer.Append(new AcousticTake.VoiceEvent {sample=0,gain=1,action=AcousticTake.VoiceAction.Play,loop=true});writer.Complete();
            }
            using var take=AcousticTake.Open(path);var settings=SpectralSettings();
            settings.fftSize=256;settings.responseStepSamples=32;settings.transitionSamples=32;settings.speedOfSound=343;settings.maximumDelaySeconds=1;
            TakeRender.RenderDirect(take,settings,path+"-render",shader:shader);
            var output=ReadWave(Path.Combine(path+"-render","listener-0.wav"));int crossings=0,expected=1000-motion*100;
            double wantedReal=0,wantedImag=0,originalReal=0,originalImag=0;
            for(int n=4001;n<12000;n++)
            {
                if(output[n-1]<0 && output[n]>=0)crossings++;
                double window=.5-.5*Math.Cos(2*Math.PI*(n-4000)/7999),value=output[n]*window;
                wantedReal+=value*Math.Cos(2*Math.PI*expected*n/8000);wantedImag+=value*Math.Sin(2*Math.PI*expected*n/8000);
                originalReal+=value*Math.Cos(2*Math.PI*1000*n/8000);originalImag+=value*Math.Sin(2*Math.PI*1000*n/8000);
            }
            Check(Math.Abs(crossings-expected)<=2,"Approaching/receding listener Doppler frequency: "+crossings+" expected "+expected);
            Check(wantedReal*wantedReal+wantedImag*wantedImag>100*(originalReal*originalReal+originalImag*originalImag),"Doppler tone dominates its unshifted bin");
        }
    }
    [TestCase(AcousticTake.Dimension.Planar2D, TakeRender.AudioBackend.CpuReference, TakeRender.Synthesis.Spectral)]
    [TestCase(AcousticTake.Dimension.Planar2D, TakeRender.AudioBackend.Gpu, TakeRender.Synthesis.ImpulseResponse)]
    [TestCase(AcousticTake.Dimension.Planar2D, TakeRender.AudioBackend.Gpu, TakeRender.Synthesis.Spectral)]
    [TestCase(AcousticTake.Dimension.Spatial3D, TakeRender.AudioBackend.CpuReference, TakeRender.Synthesis.Spectral)]
    [TestCase(AcousticTake.Dimension.Spatial3D, TakeRender.AudioBackend.Gpu, TakeRender.Synthesis.ImpulseResponse)]
    [TestCase(AcousticTake.Dimension.Spatial3D, TakeRender.AudioBackend.Gpu, TakeRender.Synthesis.Spectral)]
    [Category("GPU"), Category("Integration")]
    public void SpectralStreaming(AcousticTake.Dimension dimension,TakeRender.AudioBackend backend,TakeRender.Synthesis method)
    {
        var shader=RequireGpu();
        var scene=Scene(3);scene.dimension=dimension;
        scene.sources=new[]{new AcousticTake.SourceDefinition {objectIndex=0,clipIndex=0},new AcousticTake.SourceDefinition {objectIndex=1,clipIndex=0}};scene.listeners=new[]{2,3};
        var state=States(3);state[1].flags=AcousticTake.StateFlags.Active|AcousticTake.StateFlags.Source;
        var pcm=new float[256];for(int i=0;i<pcm.Length;i++)pcm[i]=.2f*Mathf.Sin(2*Mathf.PI*5*i/256);
        var events=new[]{new AcousticTake.VoiceEvent {sample=0,gain=.5f,action=AcousticTake.VoiceAction.Play,loop=true},
            new AcousticTake.VoiceEvent {sample=97,sourceIndex=1,gain=.3f,action=AcousticTake.VoiceAction.Play,loop=true},
            new AcousticTake.VoiceEvent {sample=611,sourceIndex=0,action=AcousticTake.VoiceAction.Stop}};
        long[] observations={0,61,137,321,577,699,811,1024};string path=Path.Combine(directory,"live-"+dimension+"-"+backend+"-"+method);
        using(var writer=new AcousticTake.Writer(path,8000,scene,Array.Empty<AcousticTake.Primitive>(),new[]{pcm}))
        {
            foreach(long t in observations)
            {
                state[2].position=new Vector3(2+t*.001f+(t>=577?3:0),0,0);state[2].rotation=Quaternion.Euler(0,t*.05f,0);
                state[2].flags&=~AcousticTake.StateFlags.Cut;if(t==577)state[2].flags|=AcousticTake.StateFlags.Cut;writer.Append(t,state);
            }
            foreach(var e in events)writer.Append(e);writer.Complete();
        }
        using var take=AcousticTake.Open(path);var settings=SpectralSettings();settings.audioBackend=backend;settings.synthesis=method;settings.airAbsorption=true;settings.listenerOutput=TakeRender.ListenerOutput.Binaural;settings.transitionSamples=64;
        var report=TakeRender.RenderDirect(take,settings,path+"-render",shader:shader);
        if(backend==TakeRender.AudioBackend.Gpu)Assert.That(report.gpuAudio.readbacks,Is.EqualTo((report.responseCount-(settings.ResponseSample(0)>0?1:0)+report.responseBatchCapacity-1)/report.responseBatchCapacity));
        var samples=new[]{new List<float>(),new List<float>()};Action<long,float[],int> collect=(at,block,count)=>{for(int n=0;n<count;n++)for(int l=0;l<2;l++)for(int ear=0;ear<2;ear++)samples[l].Add(block[n*4+l*2+ear]);};
        var left=new AcousticTake.State[4];var right=new AcousticTake.State[4];take.ReadStoredFrame(0,left);
        using(var live=new TakeRender.Live(scene,take.Geometry,new[]{pcm},8000,settings,null,0,0,shader))
        {
            live.Observe(0,0,left,left,collect);int eventIndex=0;
            for(int f=1;f<observations.Length;f++)
            {while(eventIndex<events.Length && events[eventIndex].sample<=observations[f])live.Append(events[eventIndex++]);take.ReadStoredFrame(f,right);live.Observe(observations[f-1],observations[f],left,right,collect);Array.Copy(right,left,4);}
            live.Finish(1024,left,collect);
        }
        for(int l=0;l<2;l++)
        {
            var expected=ReadWave(Path.Combine(path+"-render","listener-"+l+".wav"));Check(expected.Length==samples[l].Count,"Spectral timelines match");float error=0;
            for(int i=0;i<expected.Length;i++)error=Mathf.Max(error,Mathf.Abs(expected[i]-samples[l][i]));Near(error,0,"Spectral live/offline agreement",1e-5f);
        }
        settings.maximumResponsesPerBatch=1;TakeRender.RenderDirect(take,settings,path+"-single",shader:shader);
        Check(AcousticTake.HashFile(Path.Combine(path+"-single","responses.bin"))==report.responseHash,"GPU batch size preserves diagnostic paths and timestamps");
        for(int l=0;l<2;l++)Check(AcousticTake.HashFile(Path.Combine(path+"-single","listener-"+l+".wav"))==report.audioHashes[l],"GPU batch size preserves spectral PCM");
    }
    void KnifeEdgeFixture(bool planar,out AcousticTake.Scene scene,out AcousticTake.Primitive[] geometry,out AcousticTake.State[] state)
    {
        scene=Scene(2);scene.listeners=new[]{1};scene.dimension=planar?AcousticTake.Dimension.Planar2D:AcousticTake.Dimension.Spatial3D;
        scene.materials=new[]{new AcousticTake.MaterialDefinition {frequencies=new[]{0f},absorption=new[]{0f},transmission=new[]{0f},scattering=new[]{0f}}};
        var a=new Vector3(0,-100,planar?0:-100);var b=new Vector3(0,0,planar?0:-100);var c=new Vector3(0,0,100);var d=new Vector3(0,-100,100);
        geometry=planar?new[]{new AcousticTake.Primitive {objectIndex=2,a=a,b=b}}:
            new[]{new AcousticTake.Primitive {objectIndex=2,a=a,b=b,c=c},new AcousticTake.Primitive {objectIndex=2,a=a,b=c,c=d}};
        scene.objects[2].primitiveCount=geometry.Length;
        state=States(2);state[0].position=new Vector3(-2,1,0);state[1].position=new Vector3(2,-2,0);
        state[2].position=Vector3.zero;state[2].flags=AcousticTake.StateFlags.Active|AcousticTake.StateFlags.Geometry;
    }
    TakeRender.Settings EdgeSettings()
    {
        var settings=SpectralSettings();settings.diffractionFactor=1;settings.speedOfSound=343;settings.raysPerBin=1;settings.maxBounces=1;return settings;
    }
    static System.Numerics.Complex EdgeResponse(RayTransport.Path[] paths,int bin,float frequency)
    {
        var value=System.Numerics.Complex.Zero;
        foreach(var p in paths)if(p.frequencyBin==bin)
            value+=new System.Numerics.Complex(p.pressureGain,p.pressureImag)*System.Numerics.Complex.Exp(new System.Numerics.Complex(0,-2*Math.PI*frequency*p.delaySeconds));
        return value;
    }
    [TestCase(-100f,.0001f,false),TestCase(100f,.0001f,false)]
    [TestCase(0f,.0001f,false),TestCase(.5f,.1f,false),TestCase(0f,.1f,true),Category("GPU"),Category("Diffraction")]
    public void EdgeHeightSamplesMatchCauchyDistribution(float center,float scale,bool planar)
    {
        RequireGpu();var shader=UnityEditor.AssetDatabase.LoadAssetAtPath<ComputeShader>("Assets/Tests/Editor/EdgeSampling.compute");
        using var buffer=new ComputeBuffer(64,8);var samples=new Vector2[64];
        int kernel=shader.FindKernel("Sample");shader.SetVector("Shape",new Vector4(center,scale,1,planar?1:0));
        shader.SetBuffer(kernel,"Samples",buffer);shader.Dispatch(kernel,1,1,1);buffer.GetData(samples);
        // Independent double-precision CDF and normalized density. These far-tail
        // intervals collapse to zero with float atan(hi)-atan(lo).
        double lower=planar?-Math.PI/2:Math.Atan(-(double)center/scale);
        double span=planar?Math.PI:Math.Atan((1d-center)/scale)-lower;
        for(int i=0;i<samples.Length;i++)
        {
            double delta=(double)samples[i].x-center,quantile=(Math.Atan(delta/scale)-lower)/span;
            double density=scale/(((double)scale*scale+delta*delta)*span);
            double u=i==0?1e-7f:i==63?1-1e-7f:(i+.5)/samples.Length;
            Assert.That(quantile,Is.EqualTo(u).Within(1e-4),"CDF quantile "+i);
            Assert.That(samples[i].y/density,Is.EqualTo(1).Within(1e-4),"Proposal density "+i);
            if(!planar)Assert.That(samples[i].x,Is.InRange(0f,1f));
        }
    }

    // Deterministic double-precision quadrature of Svensson et al. (2009),
    // equations 1-6, truncated at the same 0.1-second horizon. These constants
    // include complex phase, not just pressure magnitude or the old ITU fit.
    [TestCase(false,false),TestCase(false,true),TestCase(true,false),TestCase(true,true),Category("GPU"),Category("Diffraction")]
    public void EdgeIntegralMatchesRigidHalfPlane(bool planar,bool bvh)
    {
        KnifeEdgeFixture(planar,out var scene,out var geometry,out var state);var settings=EdgeSettings();
        settings.useBvh=bvh;settings.fftSize=16;settings.spectralHop=4;settings.raysPerBin=65536;
        using var solver=new TransportSolver(scene,geometry,settings,RequireGpu(),sampleRate:1000);solver.Solve(state,new TakeRender.DirectPath[1]);
        int[] bins={0,1,4,8};double[] real={.156429101689,.112431573523,.004442267995,-.066639430815};
        double[] imaginary={0,.020944020787,.083451251431,-.016598353630};
        for(int i=0;i<bins.Length;i++)
        {
            var actual=EdgeResponse(solver.RayPaths,bins[i],bins[i]*62.5f);
            Assert.That((actual-new System.Numerics.Complex(real[i],imaginary[i])).Magnitude,Is.LessThan(.004),"Complex rigid half-plane response, bin "+bins[i]);
        }
    }
    [TestCase(false),TestCase(true),Category("GPU"),Category("Diffraction")]
    public void EdgeShadowBoundaryRetainsPhaseAndContinuity(bool planar)
    {
        KnifeEdgeFixture(planar,out var scene,out var geometry,out var state);var settings=EdgeSettings();
        settings.fftSize=4;settings.spectralHop=1;settings.raysPerBin=65536;
        using var solver=new TransportSolver(scene,geometry,settings,RequireGpu(),sampleRate:1000);
        var expected=new System.Numerics.Complex(-.022105881,-.127143985);
        foreach(float dy in new[]{-.005f,0,.005f})
        {
            state[1].position.y=-1+dy;solver.Solve(state,new TakeRender.DirectPath[1]);
            Assert.That((EdgeResponse(solver.RayPaths,1,250)-expected).Magnitude,Is.LessThan(.006),"The complex total field stays continuous across the geometrical shadow.");
        }
    }
    [TestCase(false,false),TestCase(false,true),TestCase(true,false),TestCase(true,true),Category("GPU"),Category("Diffraction")]
    public void EdgeChainsMatchRigidSecondOrder(bool planar,bool transformed)
    {
        KnifeEdgeFixture(planar,out var scene,out _,out var state);var settings=EdgeSettings();
        settings.fftSize=8;settings.spectralHop=2;settings.raysPerBin=65536;settings.useBvh=true;settings.seed=17;
        var geometry=TransportBox(scene.dimension,2,0,1);scene.objects[2].primitiveCount=geometry.Length;
        state[0].position=Vector3.left*3;state[1].position=Vector3.right*3;
        if(transformed)
        {
            var rotation=planar?Quaternion.Euler(0,0,37):Quaternion.Euler(12,37,9);var translation=new Vector3(4,-2,3);
            state[2].rotation=rotation;state[2].position=translation;
            for(int i=0;i<2;i++)state[i].position=rotation*state[i].position+translation;
        }
        foreach(int order in new[]{1,2})
        {
            settings.maxBounces=order;
            using var solver=new TransportSolver(scene,geometry,settings,RequireGpu(),sampleRate:1000);solver.Solve(state,new TakeRender.DirectPath[1]);
            int arrivals=0;
            foreach(var p in solver.RayPaths)if(p.pressureGain!=0)
            {
                arrivals++;Assert.That(float.IsFinite(p.pressureGain),Is.True);
                Assert.That(p.delaySeconds,Is.GreaterThanOrEqualTo((2*Mathf.Sqrt(5)+2)/343-1e-6f),"No path can shortcut the solid.");
            }
            if(order==1)Assert.That(arrivals,Is.Zero);else Assert.That(arrivals,Is.GreaterThan(0),"Two edge events share the ordinary bounce loop.");
            if(order!=2)continue;
            // EDtoolbox EDwedge2nd (67e53c4f), continuous double quadrature:
            // a shared solid face contributes 1/2, independently of triangle seams.
            // 3D: four finite edge pairs. 2D: two infinite pairs, same 0.1 s horizon.
            int[] bins=planar?new[]{0}:new[]{0,1,4};
            double[] real=planar?new[]{.199807}:new[]{.160990476380,-.132356150111,-.065193018566};
            double[] imag=planar?new[]{0d}:new[]{0d,-.077646144425,.063496034679};
            for(int i=0;i<bins.Length;i++)
                Assert.That((EdgeResponse(solver.RayPaths,bins[i],bins[i]*125)-new System.Numerics.Complex(real[i],imag[i])).Magnitude,
                    Is.LessThan(.012),"Second-order complex pressure, bin "+bins[i]);
        }
    }
    [Test,Category("GPU"),Category("Diffraction")]
    public void MovingEdgeUsesLocalVelocityAndCurrentPose()
    {
        KnifeEdgeFixture(false,out var scene,out var geometry,out var state);var settings=EdgeSettings();
        settings.raysPerBin=64;settings.useBvh=true;
        state[2].flags|=AcousticTake.StateFlags.VelocityValid;state[2].velocity=Vector3.up;
        using var solver=new TransportSolver(scene,geometry,settings,RequireGpu(),sampleRate:8000);
        foreach(float y in new[]{0f,.01f})
        {
            state[2].position.y=y;solver.Solve(state,new TakeRender.DirectPath[1]);int count=0;
            foreach(var p in solver.RayPaths)if(p.frequencyBin==4 && p.pressureGain!=0)
            {
                var outward=(Vector3)p.direction;var edge=state[1].position-outward*(state[1].position.x/outward.x);
                Assert.That(edge.y,Is.EqualTo(y).Within(1e-4));
                var incoming=(edge-state[0].position).normalized;var outgoing=-outward;
                float ratio=(343-Vector3.Dot(state[2].velocity,incoming))/(343-Vector3.Dot(state[2].velocity,outgoing));
                Assert.That(p.direction.w,Is.EqualTo(1-ratio).Within(2e-6));
                Assert.That(p.delaySeconds,Is.EqualTo((Vector3.Distance(edge,state[0].position)+Vector3.Distance(edge,state[1].position))/343).Within(1e-6));count++;
            }
            Assert.That(count,Is.GreaterThan(0));
        }
        Assert.That(solver.BvhPrimitiveCount,Is.EqualTo(geometry.Length));
    }
    [Test,Category("GPU"),Category("Diffraction")]
    public void TransparentGeometryDoesNotGenerateEdgeSound()
    {
        KnifeEdgeFixture(false,out var scene,out var geometry,out var state);var settings=EdgeSettings();
        scene.materials[0].transmission[0]=1;
        using var solver=new TransportSolver(scene,geometry,settings,RequireGpu(),sampleRate:8000);solver.Solve(state,new TakeRender.DirectPath[1]);
        Assert.That(solver.RayPaths[4].pressureGain,Is.EqualTo(1/Vector3.Distance(state[0].position,state[1].position)).Within(1e-6));
        for(int i=settings.FrequencyBins;i<solver.RayPaths.Length;i++)Assert.That(solver.RayPaths[i].pressureGain,Is.Zero);
    }
    [TestCase(TakeRender.Synthesis.ImpulseResponse,64,false),TestCase(TakeRender.Synthesis.Spectral,64,false)]
    [TestCase(TakeRender.Synthesis.ImpulseResponse,64,true),TestCase(TakeRender.Synthesis.Spectral,64,true)]
    [TestCase(TakeRender.Synthesis.Spectral,2048,false),Category("GPU")]
    public void ComplexArrivalsPreserveMaterialPhase(TakeRender.Synthesis method,int size,bool singleBin)
    {
        RequireGpu();var scene=Scene();var state=States();var settings=SpectralSettings();
        settings.synthesis=method;settings.fftSize=size;settings.spectralHop=size/4;settings.responseStepSamples=size*2;settings.raysPerBin=1;settings.maxBounces=1;
        var signal=new TakeRender.Signal(new[]{1f});signal.SetActive(0,true);
        signal.Append(new AcousticTake.VoiceEvent{action=AcousticTake.VoiceAction.Play,gain=1});
        var paths=new RayTransport.Path[settings.SpectralSlotsPerPair];
        for(int bin=0;bin<settings.FrequencyBins;bin++)
        {
            if(singleBin && bin!=3)continue;
            double phase=-2*Math.PI*bin*3/size;
            paths[bin]=new RayTransport.Path {frequencyBin=(uint)bin,delaySeconds=5f/8192,
                pressureGain=singleBin?0:(float)Math.Cos(phase),pressureImag=singleBin?1:(float)Math.Sin(phase)};
        }
        using var buffer=new ComputeBuffer(paths.Length,RayTransport.PathStride);buffer.SetData(paths);
        using var gpu=new GpuAudio(scene,8192,settings,new[]{signal});var cpu=new TakeRender.Mixer(scene,8192,settings,new[]{signal});
        cpu.SetResponse(0,new TakeRender.DirectPath[1],state,false,paths,0);gpu.SetResponse(0,false,buffer,0,scene,state);
        var a=new float[size*2];var b=new float[a.Length];cpu.Render(0,a.Length,a);gpu.Render(0,b.Length,b);
        for(int n=0;n<a.Length;n++)
        {
            int relative=n-settings.DspLatencySamples-5;
            double expected=singleBin?(relative>=-size/2 && relative<size/2?-2*Math.Sin(2*Math.PI*3*relative/size)/size:0):(relative==3?1:0);
            // Includes single-precision FFT roundoff at the 2048-point size.
            Assert.That(a[n],Is.EqualTo(expected).Within(1e-4),"CPU sample "+n);
            Assert.That(b[n],Is.EqualTo(expected).Within(1e-4),"GPU sample "+n);
        }
    }
    [TestCase(TakeRender.Synthesis.ImpulseResponse),TestCase(TakeRender.Synthesis.Spectral),Category("GPU"),Category("Diffraction")]
    public void SignedArrivalsPreserveDelayedCancellation(TakeRender.Synthesis method)
    {
        RequireGpu();var scene=Scene();var state=States();var settings=SpectralSettings();
        settings.synthesis=method;settings.fftSize=64;settings.spectralHop=16;settings.raysPerBin=1;settings.maxBounces=1;
        var signal=new TakeRender.Signal(new[]{1f});signal.SetActive(0,true);
        signal.Append(new AcousticTake.VoiceEvent{action=AcousticTake.VoiceAction.Play,gain=1});
        var paths=new RayTransport.Path[settings.SpectralSlotsPerPair];
        for(int bin=0;bin<settings.FrequencyBins;bin++)
        {
            paths[bin]=new RayTransport.Path{frequencyBin=(uint)bin,pressureGain=.25f,delaySeconds=5f/8192};
            paths[settings.FrequencyBins+bin]=new RayTransport.Path{frequencyBin=(uint)bin,pressureGain=-.75f,delaySeconds=9f/8192,identity=1};
        }
        using var buffer=new ComputeBuffer(paths.Length,RayTransport.PathStride);buffer.SetData(paths);
        using var gpu=new GpuAudio(scene,8192,settings,new[]{signal});var cpu=new TakeRender.Mixer(scene,8192,settings,new[]{signal});
        cpu.SetResponse(0,new TakeRender.DirectPath[1],state,false,paths,0);gpu.SetResponse(0,false,buffer,0,scene,state);
        var a=new float[128];var b=new float[128];cpu.Render(0,a.Length,a);gpu.Render(0,b.Length,b);
        for(int n=0;n<a.Length;n++)
        {
            float expected=n==settings.DspLatencySamples+5?.25f:n==settings.DspLatencySamples+9?-.75f:0;
            Assert.That(a[n],Is.EqualTo(expected).Within(2e-5));Assert.That(b[n],Is.EqualTo(expected).Within(2e-5));
        }
    }
    [TestCase(TakeRender.Synthesis.ImpulseResponse),TestCase(TakeRender.Synthesis.Spectral),Category("GPU"),Category("Diffraction")]
    public void DiffractedAudioMatchesCpuReference(TakeRender.Synthesis method)
    {
        KnifeEdgeFixture(false,out var scene,out var geometry,out var state);var settings=EdgeSettings();
        settings.raysPerBin=16;settings.synthesis=method;settings.listenerOutput=TakeRender.ListenerOutput.Binaural;settings.transitionSamples=32;
        var clip=new float[257];for(int i=0;i<clip.Length;i++)clip[i]=.2f*Mathf.Sin(i*.17f);
        var signal=new TakeRender.Signal(clip);signal.SetActive(0,true);signal.Append(new AcousticTake.VoiceEvent{action=AcousticTake.VoiceAction.Play,gain=1,loop=true});
        var signals=new[]{signal};var cpu=new TakeRender.Mixer(scene,8000,settings,signals);
        using var gpu=new GpuAudio(scene,8000,settings,signals);
        using var solver=new TransportSolver(scene,geometry,settings,RequireGpu(),sampleRate:8000);
        var direct=new TakeRender.DirectPath[1];var a=new float[256];var b=new float[256];double error=0,energy=0,peak=0;
        for(int frame=0;frame<8;frame++)
        {
            long sample=frame*128;state[2].position.y=frame*.02f;state[1].rotation=Quaternion.Euler(0,frame*11,0);
            solver.Solve(state,direct);Assert.That(direct[0].pressureGain,Is.Zero,"The fixture must depend on diffraction.");
            cpu.SetResponse(sample,direct,state,false,solver.RayPaths,0);gpu.SetResponse(sample,false,solver.RayBuffer,0,scene,state);
            cpu.Render(sample,128,a);gpu.Render(sample,128,b);
            for(int i=0;i<a.Length;i++){double d=a[i]-b[i];error+=d*d;energy+=a[i]*a[i];peak=Math.Max(peak,Math.Abs(d));}
        }
        Assert.That(energy,Is.GreaterThan(1e-6));Assert.That(peak,Is.LessThan(2e-4));Assert.That(Math.Sqrt(error/energy),Is.LessThan(5e-4));
    }
    [TestCase(false,false),TestCase(false,true),TestCase(true,false),TestCase(true,true),Category("GPU")]
    public void ZeroBouncesPreservesDirectImpulseAndOcclusionWithoutReflectedTail(bool planar,bool bvh)
    {
        var shader=RequireGpu();const int rate=8192,count=512;
        var scene=Scene(2);scene.dimension=planar?AcousticTake.Dimension.Planar2D:AcousticTake.Dimension.Spatial3D;
        scene.listeners=new[]{1};scene.objects[2].primitiveCount=1;
        scene.materials=new[]{new AcousticTake.MaterialDefinition {frequencies=new[]{0f},absorption=new[]{0f},
            scattering=new[]{0f},transmission=new[]{0f},ior=new[]{1f},damping=new[]{0f}}};
        var geometry=new[]{new AcousticTake.Primitive {objectIndex=2,a=new Vector3(0,-10,-10),b=new Vector3(0,10,-10),c=new Vector3(0,0,10)}};
        var state=States(2);state[0].position=new Vector3(-1,-1,0);state[1].position=new Vector3(-1,1,0);
        state[2].position=Vector3.zero;state[2].flags=AcousticTake.StateFlags.Active|AcousticTake.StateFlags.Geometry;
        var settings=SpectralSettings();settings.speedOfSound=512;settings.maxBounces=0;settings.useBvh=bvh;settings.responseStepSamples=count;
        settings.diffractionFactor=1; // Zero bounces must also suppress the experimental edge branch.
        TakeRender.Validate(settings);
        var signal=new TakeRender.Signal(new[]{1f});signal.SetActive(0,true);signal.Append(new AcousticTake.VoiceEvent{gain=1});
        using(var solver=new TransportSolver(scene,geometry,settings,shader,sampleRate:rate))
        {
            var direct=new TakeRender.DirectPath[1];solver.Solve(state,direct);
            Assert.That(solver.RayPaths.Length,Is.EqualTo(settings.FrequencyBins),"Only direct arrivals own storage.");
            foreach(var path in solver.RayPaths)
            {Near(path.pressureGain,.5f,"Direct pressure");Near(path.delaySeconds,2f/512,"Direct delay");}
            var cpu=new TakeRender.Mixer(scene,rate,settings,new[]{signal});
            using var gpu=new GpuAudio(scene,rate,settings,new[]{signal},maximumResponses:4);
            var expected=new float[count];var actual=new float[count];
            cpu.SetResponse(0,direct,state,false,solver.RayPaths,0);gpu.SetResponse(0,false,solver.RayBuffer,0,scene,state);
            cpu.Render(0,count,expected);gpu.Render(0,count,actual);
            int onset=settings.DspLatencySamples+32;
            for(int i=0;i<count;i++)
            {Near(expected[i],i==onset?.5f:0,"Analytic dry impulse",2e-6f);Near(actual[i],expected[i],"GPU dry impulse",2e-6f);}
            Assert.That(gpu.Diagnostics.pathCapacity,Is.EqualTo(settings.FrequencyBins));
            state[1].position=new Vector3(1,1,0);solver.Solve(state,direct);
            foreach(var path in solver.RayPaths)Near(path.pressureGain,0,"Opaque direct occlusion");
        }
        // Transmission remains available on the same direct-only path.
        scene.materials[0].transmission[0]=.25f;
        using(var solver=new TransportSolver(scene,geometry,settings,shader,sampleRate:rate))
        {
            solver.Solve(state,new TakeRender.DirectPath[1]);
            // Material transmission is an energy fraction; pressure uses its square root.
            foreach(var path in solver.RayPaths)Near(path.pressureGain,.5f/Mathf.Sqrt(8),"Transmitted direct pressure",1e-5f);
        }
        // Positive control proves the fixture can produce an audible reflected tail.
        scene.materials[0].transmission[0]=0;state[1].position=new Vector3(-1,1,0);settings.maxBounces=1;settings.diffractionFactor=0;
        using(var solver=new TransportSolver(scene,geometry,settings,shader,sampleRate:rate))
        {
            var direct=new TakeRender.DirectPath[1];solver.Solve(state,direct);
            Assert.That(solver.RayPaths[settings.FrequencyBins].pressureGain,Is.GreaterThan(0));
            var mixer=new TakeRender.Mixer(scene,rate,settings,new[]{signal});var wet=new float[count];
            mixer.SetResponse(0,direct,state,false,solver.RayPaths,0);mixer.Render(0,count,wet);
            double energy=0;for(int i=settings.DspLatencySamples+36;i<count;i++)energy+=(double)wet[i]*wet[i];
            Assert.That(energy,Is.GreaterThan(.001),"Reflection-enabled control must ring after the dry impulse.");
        }
    }

    [Test, Category("GPU")]
    public void GpuOnlyTransportRetainsOwnershipUntilAudioCompletes()
    {
        KnifeEdgeFixture(false, out var scene, out var geometry, out var state);
        using var solver = new TransportSolver(scene, geometry, EdgeSettings(), RequireGpu(), sampleRate:8000, readbackPaths:false);
        solver.BeginSolve(state, diagnostics:false);
        Assert.That(solver.IsPending, Is.True);
        Assert.That(solver.PairPrimitiveUpperBound, Is.Zero, "GPU playback does not trace diagnostic probes.");
        Assert.Throws<InvalidOperationException>(() => solver.BeginSolve(state));
        Assert.Throws<InvalidOperationException>(() => solver.TryComplete(new TakeRender.DirectPath[1]));
        // Stand in for the downstream PCM readback on the same GPU queue.
        var consumer = UnityEngine.Rendering.AsyncGPUReadback.Request(solver.RayBuffer);
        consumer.WaitForCompletion();
        Assert.That(consumer.hasError, Is.False);
        solver.CompleteAfterGpuAudio();
        Assert.That(solver.IsPending, Is.False);
        Assert.Throws<InvalidOperationException>(() => solver.CompleteAfterGpuAudio());
        solver.BeginSolve(state, diagnostics:false); // Disposal must also fence a cancelled batch.
    }
}
