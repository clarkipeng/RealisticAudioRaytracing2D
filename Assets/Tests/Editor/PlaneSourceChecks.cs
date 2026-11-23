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



}
