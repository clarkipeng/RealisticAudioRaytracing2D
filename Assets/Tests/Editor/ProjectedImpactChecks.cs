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

}
