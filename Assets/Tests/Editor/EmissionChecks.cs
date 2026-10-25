using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using NUnit.Framework;
using UnityEngine;

public partial class TakeChecks
{
    static AcousticTake.EmissionPlane BurstPlane(int capacity=16)
    {
        var p=Plane(capacity);p.generatorVersion=1;p.burstPeriodSamples=96;p.burstDurationSamples=40;
        p.u=Vector3.right*.5f;p.v=Vector3.forward*.5f;return p;
    }
    static TakeRender.Settings BurstSettings(int batch=1) => new TakeRender.Settings {
        transport=TakeRender.Transport.Raytraced,synthesis=TakeRender.Synthesis.Spectral,
        fftSize=64,spectralHop=16,responseStepSamples=64,raysPerBin=17,maxBounces=1,
        maximumDelaySeconds=.04f,maximumResponsesPerBatch=batch,tailSamples=600,
        speedOfSound=512,airAbsorption=false,delayFilter=TakeRender.DelayFilter.Linear };
    static TakeRender.Signal BurstSignal(float[] pcm,long start=0)
    {
        var signal=new TakeRender.Signal(pcm);signal.SetActive(0,true);
        signal.Append(new AcousticTake.VoiceEvent{sample=start,gain=.2f,loop=true});return signal;
    }

    [Test,Category("Core")]
    public void EmissionVersionOneIntegerGoldenVectors()
    {
        uint key=Emission.SourceKey("rain/source-1",19);Assert.That(key,Is.EqualTo(864995555u));
        Assert.That(Emission.Random(key,0,0,0),Is.EqualTo(70076207u));
        Assert.That(Emission.Random(key,1L<<33,1L<<40,4),Is.EqualTo(3632954017u));
        Assert.That(Emission.Random(key,(1L<<48)+17,(1L<<32)+5,2),Is.EqualTo(2000707616u));
        Assert.That(Emission.Random(key,19,10001,3),Is.EqualTo(1522221143u));
    }

    [Test,Category("Core")]
    public void EmissionWeightedMaskHasCorrectMassPdfAndEmptySupport()
    {
        var p=BurstPlane();p.u=Vector3.right*4;p.v=Vector3.forward*2;p.width=3;p.mask=new[]{0f,.25f,1f};
        var sampler=new Emission.PlaneSampler(p);Assert.That(sampler.Mass,Is.EqualTo(8*1.25/3).Within(1e-6));
        int middle=0;const int count=20000;
        for(int i=0;i<count;i++)
        {
            var position=sampler.Sample(Emission.Random(2,0,i,1),Emission.Random(2,0,i,2),Emission.Random(2,0,i,3),out double pdf);
            Assert.That(position.x,Is.GreaterThanOrEqualTo(-2f/3));
            bool gray=position.x<2f/3;if(gray)middle++;
            Assert.That(pdf,Is.EqualTo((gray?.25:1)/sampler.Mass).Within(1e-9));
        }
        Assert.That(middle/(double)count,Is.EqualTo(.2).Within(.015));
        p.mask=new[]{float.Epsilon,0f,1f};sampler=new Emission.PlaneSampler(p);
        var tiny=sampler.Sample(0,0,0,out var tinyPdf);Assert.That(tiny.w,Is.EqualTo(float.Epsilon));Assert.That(tinyPdf,Is.GreaterThan(0));
        p.mask=new float[3];sampler=new Emission.PlaneSampler(p);sampler.Sample(0,0,0,out var empty);
        Assert.That(sampler.Mass,Is.Zero);Assert.That(empty,Is.Zero);
    }

    [Test,Category("Core")]
    public void EmissionClockKeepsWindowClipOffsetGainAndRestartSeparate()
    {
        var signal=BurstSignal(new[]{1f,2f,3f,4f,5f},100);signal.BurstDuration=16;
        signal.Append(new AcousticTake.VoiceEvent{sample=108,action=AcousticTake.VoiceAction.Gain,gain=.5f});
        signal.Append(new AcousticTake.VoiceEvent{sample=113,action=AcousticTake.VoiceAction.Stop});
        signal.Append(new AcousticTake.VoiceEvent{sample=115,gain=1,loop=true});
        for(long sample=95;sample<140;sample++)
        {
            long age=sample-102;float expected=sample>=102 && sample<113 ? new[]{1f,2f,3f,4f,5f}[(age+3)%5]*(sample<108?.2f:.5f)*Emission.Window(age,16):0;
            Assert.That(signal.Read((double)sample,Emission.Burst|3,102),Is.EqualTo(expected).Within(1e-7));
        }
        Assert.That(signal.Epochs(0,200).ToArray(),Is.EqualTo(new[]{(100L,113L),(115L,long.MaxValue)}));
        Assert.That(Emission.Window(0,16),Is.Zero);Assert.That(Emission.Window(15,16),Is.Zero);
    }

    [Test,Category("Core")]
    public void EmissionBatchPreservesBirthsAcrossPartitionQualityAndLongClocks()
    {
        var scene=Scene();scene.planes=new[]{BurstPlane(64)};var p=scene.planes[0];p.burstPeriodSamples=13;
        var settings=BurstSettings(3);settings.raysPerBin=64;
        long epoch=(1L<<33)+7;var signals=new[]{BurstSignal(new[]{-1f,1f},epoch)};
        void Pose(long sample,AcousticTake.State[] output) {Array.Copy(States(),output,output.Length);output[0].position=new Vector3((sample-epoch)*.01f,0,0);}
        var batch=new Emission.Batch(scene,settings,8192,3);batch.Prepare(signals,epoch,3,Pose);
        var single=new Emission.Batch(scene,settings,8192,1);
        for(int frame=0;frame<3;frame++)
        {
            single.Prepare(signals,epoch+frame*64,1,Pose);var r=batch.Ranges[frame];Assert.That(single.Ranges[0].count,Is.EqualTo(r.count));
            for(int n=0;n<r.count;n++)
            {
                var a=batch.Clocks[r.first+n];var b=single.Clocks[n];
                Assert.That((a.phase,a.low,a.high),Is.EqualTo((b.phase,b.low,b.high)));Assert.That(batch.Points[r.first+n],Is.EqualTo(single.Points[n]));
            }
        }
        Assert.That(Emission.Random(1,epoch,3,0),Is.Not.EqualTo(Emission.Random(1,epoch,3+(1L<<32),0)));
        Assert.That(Emission.SourceKey("source-a",1),Is.Not.EqualTo(Emission.SourceKey("source-b",1)));
        settings.raysPerBin=1;Assert.Throws<InvalidOperationException>(()=>single.Prepare(signals,epoch,1,Pose));
    }

    [Test,Category("Core")]
    public void EmissionTakeAuthenticatesGeneratorAndPreservesVersionFive()
    {
        var scene=Scene();scene.planes=new[]{Plane()};string legacy=Path.Combine(directory,"v5");
        using(var writer=new AcousticTake.Writer(legacy,8192,scene,Array.Empty<AcousticTake.Primitive>(),new[]{new[]{-1f,1f}},version:5))
        {writer.Append(0,States());writer.Append(500,States());writer.Complete();}
        using(var take=AcousticTake.Open(legacy))Assert.That(take.Metadata.scene.planes[0].generatorVersion,Is.Zero);
        scene.planes[0]=BurstPlane();
        Assert.Throws<InvalidDataException>(()=>new AcousticTake.Writer(Path.Combine(directory,"bad-v5"),8192,scene,Array.Empty<AcousticTake.Primitive>(),new[]{new[]{1f}},version:5));
        string path=Path.Combine(directory,"bursts");
        using(var writer=new AcousticTake.Writer(path,8192,scene,Array.Empty<AcousticTake.Primitive>(),new[]{new[]{1f}}))
        {writer.Append(0,States());writer.Append(500,States());writer.Complete();}
        string file=Path.Combine(path,"take.json");var manifest=JsonUtility.FromJson<AcousticTake.Manifest>(File.ReadAllText(file));
        manifest.scene.planes[0].burstPeriodSamples++;File.WriteAllText(file,JsonUtility.ToJson(manifest));
        Assert.Throws<InvalidDataException>(()=>AcousticTake.ReadManifest(path));
    }

    [TestCase(TakeRender.DelayFilter.Linear,64,1),TestCase(TakeRender.DelayFilter.WindowedSinc,64,1)]
    [TestCase(TakeRender.DelayFilter.Linear,128,1),TestCase(TakeRender.DelayFilter.WindowedSinc,128,1)]
    [TestCase(TakeRender.DelayFilter.Linear,64,0),TestCase(TakeRender.DelayFilter.WindowedSinc,64,0)]
    [TestCase(TakeRender.DelayFilter.Linear,128,0),TestCase(TakeRender.DelayFilter.WindowedSinc,128,0),Category("GPU")]
    public void EmissionOfflineCpuGpuBatchAndLiveAgreeWithShortBursts(TakeRender.DelayFilter filter,int step,int bounces)
    {
        var shader=RequireGpu();const int rate=8192;var scene=Scene();scene.planes=new[]{BurstPlane()};
        var pcm=new float[257];var random=new System.Random(37);for(int i=0;i<pcm.Length;i++)pcm[i]=(float)(random.NextDouble()*2-1);
        var events=new[]{new AcousticTake.VoiceEvent{gain=.2f,loop=true},
            new AcousticTake.VoiceEvent{sample=311,action=AcousticTake.VoiceAction.Gain,gain=.1f},
            new AcousticTake.VoiceEvent{sample=499,action=AcousticTake.VoiceAction.Stop},
            new AcousticTake.VoiceEvent{sample=573,gain=.3f,loop=true,clipOffset=51}};
        long[] observations={0,19,137,298,443,601,787,1000};
        string takePath=Path.Combine(directory,"burst-take");var states=States();states[1].position=new Vector3(0,1,2);
        using(var writer=new AcousticTake.Writer(takePath,rate,scene,Array.Empty<AcousticTake.Primitive>(),new[]{pcm}))
        {
            foreach(long sample in observations){states[0].position=new Vector3(sample*.0001f,0,0);writer.Append(sample,states);}
            foreach(var e in events)writer.Append(e);writer.Complete();
        }
        using var take=AcousticTake.Open(takePath);var settings=BurstSettings();settings.delayFilter=filter;settings.responseStepSamples=step;
        settings.maxBounces=bounces;if(bounces==0)settings.raysPerBin=1; // All direct events still receive coverage.
        float[] expected=null;
        foreach(var mode in new[]{(TakeRender.AudioBackend.CpuReference,1),(TakeRender.AudioBackend.Gpu,1),(TakeRender.AudioBackend.Gpu,3)})
        {
            settings.audioBackend=mode.Item1;settings.maximumResponsesPerBatch=mode.Item2;
            string output=Path.Combine(directory,"render-"+mode.Item1+mode.Item2);TakeRender.RenderDirect(take,settings,output,null,shader);
            var actual=ReadWave(Path.Combine(output,"listener-0.wav"));
            if(expected==null){expected=actual;Assert.That(expected.Sum(x=>(double)x*x),Is.GreaterThan(1e-5));}
            else AssertBurstAudio(expected,actual);
        }
        // Independent flat-response oracle: sum windowed PCM at each frozen
        // site's physical delay, removing only the declared renderer latency.
        var signals=TakeRender.Signal.FromTake(take,new[]{pcm});var sourceBatch=new Emission.Batch(scene,settings,rate,1);
        var sites=new Dictionary<long,(Vector4 point,Emission.Batch.Clock clock)>();
        void Pose(long sample,AcousticTake.State[] target)=>take.Evaluate(Math.Min(sample,1000),target);
        for(long sample=0;sample<expected.Length;sample+=64)
        {
            sourceBatch.Prepare(signals,sample,1,Pose);
            for(int n=0;n<sourceBatch.Ranges[0].count;n++)
            {
                var clock=sourceBatch.Clocks[n];long birth=(long)(((ulong)clock.high<<32)|clock.low);
                sites[birth]=(sourceBatch.Points[n],clock);
            }
        }
        var analytic=new float[expected.Length];
        for(int n=0;n<analytic.Length;n++)foreach(var site in sites)
        {
            double distance=Vector3.Distance((Vector3)site.Value.point,states[1].position),delay=distance*rate/settings.speedOfSound;
            analytic[n]+=site.Value.point.w/(float)distance*Doppler.Read(signals[0],n-settings.DspLatencySamples-delay,1,filter,settings.sincRadius,site.Value.clock.phase,site.Key);
        }
        AssertBurstAudio(analytic,expected);
        var rendered=new List<float>();Action<long,float[],int> collect=(sample,block,count)=>{for(int n=0;n<count;n++)rendered.Add(block[n]);};
        var left=States();var right=States();take.ReadStoredFrame(0,left);
        using(var live=new TakeRender.Live(scene,take.Geometry,new[]{pcm},rate,settings,null,0,0,shader))
        {
            live.Prepare(left);live.Append(events[0]);live.Observe(0,0,left,left,collect);int e=1;
            for(int f=1;f<observations.Length;f++)
            {
                while(e<events.Length && events[e].sample<=observations[f])live.Append(events[e++]);
                take.ReadStoredFrame(f,right);live.Observe(observations[f-1],observations[f],left,right,collect);Array.Copy(right,left,left.Length);
            }
            live.Finish(1000,left,collect);Assert.That(live.Completed,Is.True);
        }
        AssertBurstAudio(expected,rendered.ToArray());
    }
    [TestCase(AcousticTake.Dimension.Planar2D,false),TestCase(AcousticTake.Dimension.Spatial3D,false)]
    [TestCase(AcousticTake.Dimension.Planar2D,true),TestCase(AcousticTake.Dimension.Spatial3D,true),Category("GPU")]
    public void EmissionUnevenRayAllocationPreservesEachMirrorWeight(AcousticTake.Dimension dimension,bool bvh)
    {
        var scene=Scene(2);scene.dimension=dimension;scene.listeners=new[]{1};scene.objects[2].primitiveCount=1;
        var plane=BurstPlane();plane.burstPeriodSamples=32;if(dimension==AcousticTake.Dimension.Planar2D)plane.v=Vector3.up*.5f;
        scene.planes=new[]{plane};scene.materials=new[]{new AcousticTake.MaterialDefinition{frequencies=new[]{0f},absorption=new[]{0f},scattering=new[]{0f},transmission=new[]{0f}}};
        var geometry=new[]{new AcousticTake.Primitive{objectIndex=2,a=new Vector3(-10,0,-10),b=new Vector3(10,0,-10),c=new Vector3(0,0,10)}};
        var states=States(2);states[0].position=new Vector3(-1,1,0);states[1].position=new Vector3(1,1,0);
        states[2].position=Vector3.zero;states[2].flags=AcousticTake.StateFlags.Active|AcousticTake.StateFlags.Geometry;
        states[0].velocity=Vector3.right*10;states[0].flags|=AcousticTake.StateFlags.VelocityValid;
        var settings=BurstSettings();settings.useBvh=bvh;var signals=new[]{BurstSignal(new[]{-1f,1f})};
        void Pose(long sample,AcousticTake.State[] output)=>Array.Copy(states,output,states.Length);
        var batch=new Emission.Batch(scene,settings,8192,1);batch.Prepare(signals,0,1,Pose);
        Assert.That(batch.Ranges[0].count,Is.EqualTo(4));
        using var solver=new TransportSolver(scene,geometry,settings,RequireGpu(),sampleRate:8192);
        solver.PrepareEmissions(signals,0,1,Pose);solver.Solve(states,new TakeRender.DirectPath[1]);
        var sums=new Dictionary<long,float>();int first=plane.samples*settings.FrequencyBins;
        for(int ray=0;ray<settings.RaysPerBin;ray++)
        {
            var path=solver.RayPaths[first+ray*2];Assert.That(path.emissionPhase&Emission.Burst,Is.Not.Zero);
            Assert.That(path.direction.w,Is.Zero);sums.TryGetValue(path.Birth,out float old);sums[path.Birth]=old+path.pressureGain;
        }
        Assert.That(sums.Count,Is.EqualTo(4));
        for(int n=0;n<4;n++)
        {
            var point=batch.Points[n];Vector3 image=point;image.y=-image.y;
            var clock=batch.Clocks[n];long birth=(long)(((ulong)clock.high<<32)|clock.low);
            Assert.That(sums[birth],Is.EqualTo(point.w/Vector3.Distance(image,states[1].position)).Within(1e-5));
        }
    }

    [TestCase(128,2),TestCase(2048,1),Category("GPU")]
    public void EmissionPackedDspKeepsDistinctBirthsAndPointSources(int size,int listenerCount)
    {
        RequireGpu();var scene=Scene(listenerCount+1);scene.listeners=Enumerable.Range(2,listenerCount).ToArray();
        scene.sources=new[]{scene.sources[0],new AcousticTake.SourceDefinition{objectIndex=1,clipIndex=0,mutedListenerPlusOne=2}};
        var plane=BurstPlane(2);plane.burstDurationSamples=size*4;scene.planes=new[]{plane};
        var settings=BurstSettings();settings.fftSize=size;settings.responseStepSamples=size;settings.spectralHop=size/4;settings.raysPerBin=2;
        var state=States(listenerCount+1);state[1].flags=AcousticTake.StateFlags.Active|AcousticTake.StateFlags.Source;
        var clip=Enumerable.Range(0,257).Select(i=>Mathf.Sin(i*.31f)).ToArray();var signals=new[]{BurstSignal(clip),BurstSignal(clip)};
        var layout=new RayTransport.Layout(scene,settings);int pairs=scene.sources.Length*listenerCount;
        Assert.That(layout.Capacity,Is.LessThan(pairs*layout.MaximumSlots));
        var paths=new RayTransport.Path[layout.Capacity];
        for(int pair=0;pair<pairs;pair++)for(int site=0;site<(pair<listenerCount?2:1);site++)for(int bin=0;bin<settings.FrequencyBins;bin++)
            paths[layout.Ranges[pair].x+site*settings.FrequencyBins+bin]=new RayTransport.Path{direction=Vector3.forward,pressureGain=.1f,
                frequencyBin=(uint)bin,identity=7,emissionPhase=pair<listenerCount?Emission.Burst|7:0,birthLow=pair<listenerCount?(uint)(site*size/2):0};
        using var buffer=new ComputeBuffer(paths.Length,RayTransport.PathStride);
        using var gpu=new GpuAudio(scene,8192,settings,signals);var cpu=new TakeRender.Mixer(scene,8192,settings,signals);
        Assert.That(gpu.Diagnostics.pathCapacity,Is.EqualTo(layout.Capacity));
        var a=new float[size*listenerCount];var b=new float[a.Length];var direct=new TakeRender.DirectPath[pairs];
        var expected=new List<float>();var actual=new List<float>();
        for(int frame=0;frame<3;frame++)
        {
            // The two births have identical geometry/topology/clip offset. Reorder
            // them to catch accidental grouping or matching by slot/hash alone.
            if(frame==1)for(int pair=0;pair<listenerCount;pair++)for(int bin=0;bin<settings.FrequencyBins;bin++)
            {int first=layout.Ranges[pair].x+bin,second=first+settings.FrequencyBins;(paths[first],paths[second])=(paths[second],paths[first]);}
            buffer.SetData(paths);long sample=(long)frame*size;
            cpu.SetResponse(sample,direct,state,false,paths,0);gpu.SetResponse(sample,false,buffer,0,scene,state);
            cpu.Render(sample,size,a);gpu.Render(sample,size,b);expected.AddRange(a);actual.AddRange(b);
        }
        AssertBurstAudio(expected.ToArray(),actual.ToArray());
    }

    [Test,Category("Core")]
    public void EmissionCandidateIntervalsMatchExhaustiveBirthEnumeration()
    {
        var random=new System.Random(1701);var scene=Scene();var settings=BurstSettings();settings.raysPerBin=1024;
        for(int trial=0;trial<100;trial++)
        {
            var plane=BurstPlane(1024);plane.burstPeriodSamples=random.Next(1,200);plane.burstDurationSamples=random.Next(4,300);scene.planes=new[]{plane};
            long epoch=random.Next(0,500),stop=epoch+random.Next(1,4000),output=random.Next(0,4000);
            var signal=BurstSignal(new[]{-1f,1f},epoch);signal.Append(new AcousticTake.VoiceEvent{sample=stop,action=AcousticTake.VoiceAction.Stop});
            var batch=new Emission.Batch(scene,settings,8192,1);batch.Prepare(new[]{signal},output,1,(sample,pose)=>Array.Copy(States(),pose,pose.Length));
            long from=Math.Max(0,output-settings.HistorySamples(8192)-2*settings.responseStepSamples),to=output+2*settings.responseStepSamples;
            uint key=Emission.SourceKey(scene.objects[0].id,plane.seed);var expected=new List<long>();
            for(long index=0;epoch+index*plane.burstPeriodSamples<Math.Min(stop,to);index++)
            {long birth=Emission.Birth(key,epoch,index,plane.burstPeriodSamples);if(birth<Math.Min(stop,to) && birth+plane.burstDurationSamples>from)expected.Add(birth);}
            var actual=batch.Clocks.Take(batch.Ranges[0].count).Select(c=>(long)(((ulong)c.high<<32)|c.low)).ToArray();
            Assert.That(actual,Is.EqualTo(expected),"trial "+trial);
        }
    }

    [Test,Category("GPU")]
    public void EmissionPackedTransportKeepsFrameSourceAndListenerOffsets()
    {
        var scene=Scene(3);scene.sources=new[]{scene.sources[0],new AcousticTake.SourceDefinition{objectIndex=1,clipIndex=0}};scene.listeners=new[]{2,3};scene.planes=new[]{Plane(3)};
        var settings=BurstSettings(2);settings.raysPerBin=1;var layout=new RayTransport.Layout(scene,settings);
        var states=new AcousticTake.State[scene.objects.Length*2];
        for(int f=0;f<2;f++)
        {
            var frame=States(3);frame[0].position=new Vector3(f*.5f,1,0);frame[0].rotation=Quaternion.Euler(0,f*30,0);
            frame[1].position=new Vector3(-2,1,0);frame[1].flags=AcousticTake.StateFlags.Active|AcousticTake.StateFlags.Source;
            frame[2].position=new Vector3(0,2,4);frame[3].position=new Vector3(3,2,1);Array.Copy(frame,0,states,f*frame.Length,frame.Length);
        }
        using var solver=new TransportSolver(scene,Array.Empty<AcousticTake.Primitive>(),settings,RequireGpu(),2,8192);
        solver.BeginSolve(states,2);Assert.That(solver.TryComplete(new TakeRender.DirectPath[8],true),Is.True);
        Assert.That(solver.RayPathsPerFrame,Is.EqualTo(layout.Capacity));
        for(int f=0;f<2;f++)for(int pair=0;pair<4;pair++)for(int site=0;site<(pair<2?3:1);site++)
        {
            var source=states[f*4+pair/2];var listener=states[f*4+scene.listeners[pair%2]];
            var local=pair<2?AudioSourcePlane.Sample(scene.planes[0],site):new Vector4(0,0,0,1);
            float distance=Vector3.Distance(source.position+source.rotation*(Vector3)local,listener.position);
            var path=solver.RayPaths[f*layout.Capacity+layout.Ranges[pair].x+site*settings.FrequencyBins];
            Assert.That(path.delaySeconds,Is.EqualTo(distance/settings.speedOfSound).Within(1e-6));
            Assert.That(path.pressureGain,Is.EqualTo(local.w/distance).Within(1e-5));
        }
    }

    static void AssertBurstAudio(float[] expected,float[] actual)
    {
        Assert.That(actual.Length,Is.EqualTo(expected.Length));double error=0,energy=0;
        for(int i=0;i<expected.Length;i++){double d=actual[i]-expected[i];error+=d*d;energy+=(double)expected[i]*expected[i];}
        Assert.That(Math.Sqrt(error/energy),Is.LessThan(.001));
    }
}
