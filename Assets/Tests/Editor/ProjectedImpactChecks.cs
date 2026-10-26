using System;
using System.IO;
using System.Linq;
using NUnit.Framework;
using UnityEngine;

public partial class TakeChecks
{
    static AcousticTake.Primitive[] ImpactQuad(int owner,float y,float radius) => new[] {
        new AcousticTake.Primitive{objectIndex=owner,a=new Vector3(-radius,y,-radius),b=new Vector3(radius,y,radius),c=new Vector3(radius,y,-radius)},
        new AcousticTake.Primitive{objectIndex=owner,a=new Vector3(-radius,y,-radius),b=new Vector3(-radius,y,radius),c=new Vector3(radius,y,radius)} };

    [Test,Category("Core")]
    public void ProjectedImpactFirstSurfaceHandlesRoofsWindingMotionAndFlags()
    {
        var scene=Scene(3);var states=States(3);
        for(int i=0;i<states.Length;i++){states[i].position=Vector3.zero;states[i].flags=AcousticTake.StateFlags.Active|AcousticTake.StateFlags.Geometry;}
        var geometry=ImpactQuad(2,2,1).Concat(ImpactQuad(3,0,10)).ToArray();
        // Double-sided hits: reversing a roof face must not leak rain to the floor.
        (geometry[0].b,geometry[0].c)=(geometry[0].c,geometry[0].b);
        var bvh=new DirectBvh(scene,geometry);bvh.Update(states);
        Assert.That(bvh.Raycast(new Vector3(.3f,8,.2f),Vector3.down,states,out var roof),Is.True);
        Assert.That(geometry[roof.primitive].objectIndex,Is.EqualTo(2));Assert.That(roof.position.y,Is.EqualTo(2).Within(1e-6));Assert.That(roof.normal.y,Is.GreaterThan(.99));
        Assert.That(bvh.Raycast(new Vector3(2,8,0),Vector3.down,states,out var ground),Is.True);Assert.That(ground.position.y,Is.Zero);
        Assert.That(bvh.Raycast(new Vector3(20,8,0),Vector3.down,states,out _),Is.False);
        // A rotated/moving roof is queried in its actual pose, not its initial bounds.
        states[2].position=new Vector3(0,1,0);states[2].rotation=Quaternion.Euler(0,0,15);bvh.Update(states);
        Assert.That(bvh.Raycast(new Vector3(0,8,0),Vector3.down,states,out roof),Is.True);
        Assert.That(roof.position.y,Is.EqualTo(1+2/Mathf.Cos(Mathf.PI/12)).Within(1e-5));
        foreach(var flags in new[]{AcousticTake.StateFlags.Active,AcousticTake.StateFlags.Geometry})
        {states[2].flags=flags;bvh.Update(states);Assert.That(bvh.Raycast(new Vector3(0,8,0),Vector3.down,states,out ground),Is.True);Assert.That(ground.position.y,Is.Zero);}
        var empty=new DirectBvh(scene,Array.Empty<AcousticTake.Primitive>());empty.Update(states);
        Assert.That(empty.Raycast(Vector3.up,Vector3.down,states,out _),Is.False);
    }

    [Test,Category("Core")]
    public void ProjectedImpactBirthsFreezeAndKeepEnergyAcrossBatches()
    {
        var scene=Scene(2);scene.listeners=new[]{1};var plane=BurstPlane(64);plane.generatorVersion=3;plane.burstDurationSamples=4;plane.burstPeriodSamples=13;scene.planes=new[]{plane};
        var geometry=ImpactQuad(2,0,10);var settings=BurstSettings(3);var signals=new[]{BurstSignal(new[]{.2f,-.4f,.1f,0f})};
        int evaluations=0;
        void Pose(long sample,AcousticTake.State[] output)
        {
            evaluations++;Array.Copy(States(2),output,output.Length);output[0].position=new Vector3(0,10,0);
            output[2].position=new Vector3(0,2+sample*.001f,0);output[2].flags=AcousticTake.StateFlags.Active|AcousticTake.StateFlags.Geometry;
        }
        var batch=new Emission.Batch(scene,settings,8192,3,geometry);batch.Prepare(signals,0,3,Pose);
        int count=evaluations;batch.Prepare(signals,0,3,Pose);Assert.That(evaluations,Is.EqualTo(count),"Retained births must not be recast.");
        var single=new Emission.Batch(scene,settings,8192,1,geometry);double energy=.21;
        for(int f=0;f<3;f++)
        {
            settings.raysPerBin=1;single.Prepare(signals,f*64,1,Pose);var range=batch.Ranges[f];Assert.That(range.count,Is.GreaterThan(0));Assert.That(single.Ranges[0].count,Is.EqualTo(range.count));
            for(int n=0;n<range.count;n++)
            {
                int at=range.first+n;long birth=(long)(((ulong)batch.Clocks[at].high<<32)|batch.Clocks[at].low);
                Assert.That(batch.Points[at],Is.EqualTo(single.Points[n]));Assert.That(batch.Clocks[at].phase,Is.EqualTo(Emission.Burst));
                Assert.That(batch.Points[at].y,Is.EqualTo(2+birth*.001f+Emission.SurfaceOffset).Within(2e-6));
                Assert.That(batch.Points[at].w*batch.Points[at].w*energy/plane.burstPeriodSamples,Is.EqualTo(new Emission.PlaneSampler(plane).Mass).Within(1e-6));
            }
        }
        // Misses remain silent even with fewer impact slots than sampled rainfall events.
        plane.samples=1;var misses=new Emission.Batch(scene,settings,8192,1,Array.Empty<AcousticTake.Primitive>());
        misses.Prepare(signals,0,1,Pose);Assert.That(misses.Ranges[0].count,Is.Zero);
        count=evaluations;misses.Prepare(signals,0,1,Pose);Assert.That(evaluations,Is.EqualTo(count));
    }

    [Test,Category("Core")]
    public void ProjectedImpactGeneratorIsAuthenticatedAndRequiresSpatialOneShots()
    {
        var scene=Scene();var plane=BurstPlane();plane.generatorVersion=3;plane.burstDurationSamples=4;scene.planes=new[]{plane};
        string path=Path.Combine(directory,"surface");
        using(var writer=new AcousticTake.Writer(path,8192,scene,Array.Empty<AcousticTake.Primitive>(),new[]{new[]{1f,0f,0f,0f}}))
        {writer.Append(0,States());writer.Append(100,States());writer.Complete();}
        using(var take=AcousticTake.Open(path))Assert.That(take.Metadata.scene.planes[0].generatorVersion,Is.EqualTo(3));
        string file=Path.Combine(path,"take.json");var meta=JsonUtility.FromJson<AcousticTake.Manifest>(File.ReadAllText(file));meta.scene.planes[0].generatorVersion=2;
        File.WriteAllText(file,JsonUtility.ToJson(meta));Assert.Throws<InvalidDataException>(()=>AcousticTake.ReadManifest(path));
        Assert.Throws<InvalidDataException>(()=>AudioSourcePlane.ValidateEvent(scene,new AcousticTake.VoiceEvent{loop=true,clipOffset=1}));
        scene.dimension=AcousticTake.Dimension.Planar2D;
        Assert.Throws<InvalidDataException>(()=>new AcousticTake.Writer(Path.Combine(directory,"planar"),8192,scene,Array.Empty<AcousticTake.Primitive>(),new[]{new[]{1f,0f,0f,0f}}));
    }

    [TestCase(false),TestCase(true),Category("GPU")]
    public void ProjectedImpactRoofUsesMaterialTransmission(bool useBvh)
    {
        var scene=Scene(2);scene.listeners=new[]{1};scene.objects[2].primitiveCount=2;
        var plane=BurstPlane();plane.generatorVersion=3;plane.burstDurationSamples=4;scene.planes=new[]{plane};
        var material=new AcousticTake.MaterialDefinition{frequencies=new[]{0f},absorption=new[]{0f},transmission=new[]{.25f},ior=new[]{1f}};scene.materials=new[]{material};
        var geometry=ImpactQuad(2,2,10);var state=States(2);state[0].position=new Vector3(0,5,0);state[1].position=Vector3.up;
        state[2].position=Vector3.zero;state[2].flags=AcousticTake.StateFlags.Active|AcousticTake.StateFlags.Geometry;
        var settings=BurstSettings();settings.maxBounces=0;settings.useBvh=useBvh;
        var signals=new[]{BurstSignal(new[]{0f,1f,-1f,0f})};
        void Pose(long sample,AcousticTake.State[] output)=>Array.Copy(state,output,state.Length);
        var batch=new Emission.Batch(scene,settings,8192,1,geometry);batch.Prepare(signals,0,1,Pose);
        foreach(float transmission in new[]{.25f,0f})
        {
            material.transmission[0]=transmission;
            using var solver=new TransportSolver(scene,geometry,settings,RequireGpu(),sampleRate:8192);
            solver.PrepareEmissions(signals,0,1,Pose);solver.Solve(state,new TakeRender.DirectPath[1]);
            Assert.That(batch.Ranges[0].count,Is.GreaterThan(0));
            for(int n=0;n<batch.Ranges[0].count;n++)
            {
                var point=batch.Points[n];float expected=point.w*Mathf.Sqrt(transmission)/Vector3.Distance(point,state[1].position);
                foreach(int bin in new[]{0,settings.FrequencyBins-1})
                    Assert.That(solver.RayPaths[n*settings.FrequencyBins+bin].pressureGain,Is.EqualTo(expected).Within(1e-4));
            }
        }
    }
}
