using System;
using System.Runtime.InteropServices;
using NUnit.Framework;
using UnityEngine;

// Opt-in per-family gains (FamilyGains.cs; docs/benchmarks/family-gains-2026-11-03):
// gains of one change no bit, each sampled family is scaled by its own law and
// nothing else, the law sees the routes-per-sample the fit assumed, and CPU and
// GPU audio stay in parity with gains on.
public partial class TakeChecks
{
    static byte[] PathBytes(RayTransport.Path[] paths) => MemoryMarshal.AsBytes(paths.AsSpan()).ToArray();

    [TestCase(false,256),TestCase(false,4096),TestCase(true,256),Category("GPU"),Category("CompleteRoutes")]
    public void FamilyGainsOfOneReproduceEveryPath(bool planar,int samples)
    {
        var shader=RequireGpu();
        AcousticTake.Scene scene;AcousticTake.Primitive[] geometry;AcousticTake.State[] state;
        if(planar){KnifeEdgeFixture(true,out scene,out geometry,out state);}
        else TwoWallFixture(out scene,out geometry,out state);
        var settings=TwoWallSettings(samples);settings.reflectionCandidates=4;
        byte[] Solve(bool gains)
        {
            settings.familyGains=gains;settings.familyGainTable=gains?FamilyGainTable.Unit():null;
            using var solver=new TransportSolver(scene,geometry,settings,shader,sampleRate:1000);
            solver.Solve(state,new TakeRender.DirectPath[1]);
            Assert.That(solver.FamilyGains!=null,Is.EqualTo(gains));
            return PathBytes(solver.RayPaths);
        }
        var off=Solve(false);var unit=Solve(true);
        Assert.That(unit,Is.EqualTo(off),"Unit family gains must reproduce every path bit for bit.");
        settings.familyGains=true;settings.familyGainTable=null;
        using var gained=new TransportSolver(scene,geometry,settings,shader,sampleRate:1000);
        gained.Solve(state,new TakeRender.DirectPath[1]);
        Assert.That(PathBytes(gained.RayPaths),Is.Not.EqualTo(off),"The shipped gains must change the sampled families.");
    }

    // Two boxes: 24 triangles and 24 edges, so 2,328 complete candidates. 256 samples
    // subsample every family (rho = size * 5 / 256); 4,096 enumerate them (rho = 2,328 / 4,096).
    [TestCase(256),TestCase(4096),Category("GPU"),Category("CompleteRoutes")]
    public void FamilyGainsScaleOnlyTheirFamily(int samples)
    {
        var shader=RequireGpu();
        TwoWallFixture(out var scene,out var geometry,out var state);
        var settings=TwoWallSettings(samples);settings.reflectionCandidates=20;
        RayTransport.Path[] off;int[] families;
        using(var plain=new TransportSolver(scene,geometry,settings,shader,sampleRate:1000))
        {plain.Solve(state,new TakeRender.DirectPath[1]);off=(RayTransport.Path[])plain.RayPaths.Clone();families=plain.CompleteSampleFamilies();}
        settings.familyGains=true;
        using var solver=new TransportSolver(scene,geometry,settings,shader,sampleRate:1000);
        solver.Solve(state,new TakeRender.DirectPath[1]);var on=solver.RayPaths;var table=solver.FamilyGains;
        Assert.That(table.hash,Is.EqualTo(FamilyGainTable.Shipped().hash));

        float[] expected=samples==256?new[]{1.2f,24*5/256f,576*5/256f,576*5/256f,576*5/256f,576*5/256f}:new[]{1.2f,2328/4096f,2328/4096f,2328/4096f,2328/4096f,2328/4096f};
        for(int f=0;f<6;f++)Assert.That(solver.FamilyRoutesPerSample(f),Is.EqualTo(expected[f]).Within(1e-6f*expected[f]),"rho of "+FamilyGainTable.Families[f]);

        int bins=settings.FrequencyBins,mirrors=bins*(1+settings.MirrorCandidates(scene));
        int completeStart=mirrors+settings.RaysPerPair*settings.PathSlotsPerRay;
        Assert.That(off.Length,Is.EqualTo(completeStart+samples*bins));
        var gained=new int[6];
        for(int i=0;i<off.Length;i++)
        {
            int family=i<bins||(i>=mirrors&&i<completeStart)?-1:i<mirrors?0:families[(i-completeStart)/bins]<0?-1:1+families[(i-completeStart)/bins];
            float gain=family<0?1:table.Gain(family,solver.FamilyRoutesPerSample(family));
            var a=off[i];var b=on[i];
            Assert.That(b.pressureGain,Is.EqualTo(a.pressureGain*gain).Within(1e-5f*Mathf.Abs(a.pressureGain)),"Path "+i+" real part");
            Assert.That(b.pressureImag,Is.EqualTo(a.pressureImag*gain).Within(1e-5f*Mathf.Abs(a.pressureImag)),"Path "+i+" imaginary part");
            a.pressureGain=b.pressureGain;a.pressureImag=b.pressureImag;
            Assert.That(PathBytes(new[]{b}),Is.EqualTo(PathBytes(new[]{a})),"A gain changes only the pressure of path "+i);
            if(family>=0 && (a.pressureGain!=0 || a.pressureImag!=0)){Assert.That(gain,Is.LessThan(1));gained[family]++;}
        }
        Assert.That(gained[0],Is.GreaterThan(0),"The fixture must contain a sampled first-order image.");
        int complete=0;for(int f=1;f<6;f++)complete+=gained[f]>0?1:0;
        // Subsampled at 256 samples, most drawn DD/RD/DR routes are invalid in this fixture.
        Assert.That(complete,Is.GreaterThanOrEqualTo(samples>=4096?3:2),"The fixture must gain several complete-route families.");
    }

    [Test,Category("GPU"),Category("CompleteRoutes")]
    public void FamilyGainsKeepCpuAndGpuAudioInParity()
    {
        var shader=RequireGpu();const int rate=8192,size=128;
        TwoWallFixture(out var scene,out var geometry,out var state);
        var settings=new TakeRender.Settings{transport=TakeRender.Transport.Raytraced,synthesis=TakeRender.Synthesis.Spectral,delayFilter=TakeRender.DelayFilter.Linear,
            fftSize=size,spectralHop=size/4,responseStepSamples=size,transitionSamples=size/4,maximumDelaySeconds=.1f,airAbsorption=true,useBvh=true,
            raysPerBin=2,maxBounces=2,completeRouteSamples=256,diffractionFactor=1,speedOfSound=343,minimumDistance=.1f,familyGains=true};
        TakeRender.Validate(settings);
        var clip=new float[257];for(int n=0;n<clip.Length;n++)clip[n]=.2f*Mathf.Sin(n*.37f)+.1f*Mathf.Cos(n*.93f);
        var signals=new[]{new TakeRender.Signal(clip)};signals[0].SetActive(0,true);
        signals[0].Append(new AcousticTake.VoiceEvent{action=AcousticTake.VoiceAction.Play,gain=1,loop=true});
        using var gpu=new GpuAudio(scene,rate,settings,signals);var cpu=new TakeRender.Mixer(scene,rate,settings,signals);
        using var solver=new TransportSolver(scene,geometry,settings,shader,sampleRate:rate);
        Assert.That(solver.FamilyGains,Is.Not.Null);
        var direct=new TakeRender.DirectPath[1];var a=new float[size];var b=new float[size];
        double peak=0,error=0,energy=0;
        for(int frame=0;frame<8;frame++)
        {
            long sample=frame*(long)size;
            state[1].position=new Vector3(2-frame*.05f,.8f,-.2f+frame*.03f);
            solver.Solve(state,direct);
            cpu.SetResponse(sample,direct,state,false,solver.RayPaths,0);gpu.SetResponse(sample,false,solver.RayBuffer,0,scene,state);
            cpu.Render(sample,size,a);gpu.Render(sample,size,b);
            for(int n=0;n<size;n++){double d=a[n]-b[n];peak=Math.Max(peak,Math.Abs(d));error+=d*d;energy+=a[n]*a[n];}
        }
        Check(energy>1e-6,"The gained fixture contains audible output.");
        double relative=Math.Sqrt(error/energy);
        Check(peak<2e-4 && relative<5e-4,"GPU/CPU with family gains: peak="+peak+" relative="+relative);
    }

    // Opt-in cost measurement: synchronous solves at the live budget (2 rays per bin,
    // 256 complete samples, 32 bounces), gains off and on, alternated in one process.
    [Test, Category("QualityMeasurement"), Timeout(int.MaxValue)]
    public void MeasureFamilyGainCost()
    {
        string output=Environment.GetEnvironmentVariable("ACOUSTIC_FAMILY_GAIN_COST");
        if(string.IsNullOrEmpty(output))Assert.Ignore("Set ACOUSTIC_FAMILY_GAIN_COST to an output JSON path.");
        var shader=RequireGpu();
        TwoWallFixture(out var scene,out var geometry,out var state);
        var rows=new System.Collections.Generic.List<FamilyGainCost>();
        foreach(int fft in new[]{64,1024})
        {
            var settings=TwoWallSettings(256);settings.maxBounces=32;settings.fftSize=fft;settings.spectralHop=fft/4;
            settings.maximumDelaySeconds=1.25f;settings.responseStepSamples=fft;
            var off=TwoWallSettings(256);off.maxBounces=32;off.fftSize=fft;off.spectralHop=fft/4;off.maximumDelaySeconds=1.25f;off.responseStepSamples=fft;
            settings.familyGains=true;
            using var plain=new TransportSolver(scene,geometry,off,shader,sampleRate:8000);
            using var gained=new TransportSolver(scene,geometry,settings,shader,sampleRate:8000);
            var direct=new TakeRender.DirectPath[1];const int warm=20,count=300;
            var a=new double[count];var b=new double[count];
            for(int i=-warm;i<count;i++)
            {
                state[1].position=new Vector3(2,.8f+.001f*(i&7),-.2f);
                long t=System.Diagnostics.Stopwatch.GetTimestamp();plain.Solve(state,direct);double x=TakeRender.Elapsed(t);
                t=System.Diagnostics.Stopwatch.GetTimestamp();gained.Solve(state,direct);double y=TakeRender.Elapsed(t);
                if(i>=0){a[i]=x;b[i]=y;}
            }
            var d=new double[count];for(int i=0;i<count;i++)d[i]=b[i]-a[i];
            Array.Sort(a);Array.Sort(b);Array.Sort(d);
            rows.Add(new FamilyGainCost{fftSize=fft,bins=settings.FrequencyBins,pathsPerSolve=plain.RayPaths.Length,solves=count,
                medianMsOff=1e3*a[count/2],medianMsOn=1e3*b[count/2],medianPairedDifferenceMs=1e3*d[count/2],
                p10PairedDifferenceMs=1e3*d[count/10],p90PairedDifferenceMs=1e3*d[count*9/10]});
        }
        System.IO.File.WriteAllText(output,JsonUtility.ToJson(new FamilyGainCosts{device=SystemInfo.graphicsDeviceName,cases=rows.ToArray()},true));
    }
    [Serializable] sealed class FamilyGainCost
    {
        public int fftSize,bins,pathsPerSolve,solves;
        public double medianMsOff,medianMsOn,medianPairedDifferenceMs,p10PairedDifferenceMs,p90PairedDifferenceMs;
    }
    [Serializable] sealed class FamilyGainCosts { public string device; public FamilyGainCost[] cases; }

    [Test]
    public void FamilyGainsNeedCompleteRoutesAndFirstImages()
    {
        var settings=TwoWallSettings(256);settings.familyGains=true;
        Assert.DoesNotThrow(()=>TakeRender.ValidateFamilyGains(settings));
        settings.completeRouteSamples=0;Assert.Throws<ArgumentException>(()=>TakeRender.ValidateFamilyGains(settings),"No complete routes");
        settings.completeRouteSamples=256;settings.maxBounces=0;Assert.Throws<ArgumentException>(()=>TakeRender.ValidateFamilyGains(settings),"Direct only");
        settings.maxBounces=4;settings.reflectionSampling=TakeRender.ReflectionSampling.PrimitiveBaseline;
        Assert.Throws<ArgumentException>(()=>TakeRender.ValidateFamilyGains(settings),"Primitive-baseline mirrors");
        settings.reflectionSampling=TakeRender.ReflectionSampling.FirstImages;settings.transport=TakeRender.Transport.Direct;
        Assert.Throws<ArgumentException>(()=>TakeRender.ValidateFamilyGains(settings),"Direct transport");
        settings.familyGains=false;settings.completeRouteSamples=0;
        Assert.DoesNotThrow(()=>TakeRender.ValidateFamilyGains(settings),"Off needs nothing");
    }

    [Test]
    public void ShippedFamilyGainTableIsProvenancedData()
    {
        var table=FamilyGainTable.Shipped();
        Assert.That(table.families,Is.EqualTo(FamilyGainTable.Families));
        Assert.That(table.datasetSha256,Has.Length.EqualTo(64));Assert.That(table.selectionSha256,Has.Length.EqualTo(64));
        Assert.That(table.method,Is.EqualTo("rho-energy"));Assert.That(table.power,Is.EqualTo(.5f));
        for(int f=0;f<FamilyGainTable.Families.Length;f++)
        {
            Assert.That(table.Gain(f,0),Is.EqualTo(1f),"No sampled routes, no gain.");
            Assert.That(table.Gain(f,10),Is.LessThanOrEqualTo(table.Gain(f,1)),"More routes per sample never raises the gain.");
            Assert.That(table.fittedRhoMin[f],Is.LessThanOrEqualTo(table.fittedRhoMax[f]));
            Assert.That(FamilyGainTable.Unit().Gain(f,1e6f),Is.EqualTo(1f));
        }
        Assert.Throws<ArgumentException>(()=>FamilyGainTable.Parse(JsonUtility.ToJson(new FamilyGainTable{schema=FamilyGainTable.Schema,families=FamilyGainTable.Families,k=new[]{1f,1,1,1,1,-1},power=.5f})));
    }
}
