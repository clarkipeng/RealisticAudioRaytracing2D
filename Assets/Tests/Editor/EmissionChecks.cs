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
    static TakeRender.Signal BurstSignal(float[] pcm,long start=0)
    {
        var signal=new TakeRender.Signal(pcm);signal.SetActive(0,true);
        signal.Append(new AcousticTake.VoiceEvent{sample=start,gain=.2f,loop=true});return signal;
    }

    [Test,Category("Core")]
    public void ImpactReadsTheWholeClipOnceWithoutExcerptOrAddedWindow()
    {
        var pcm=new[]{.4f,-.8f,.2f,.1f,0f};var signal=BurstSignal(pcm,100);
        signal.BurstDuration=pcm.Length;signal.OneShot=true;
        for(long sample=98;sample<112;sample++)
        {
            long age=sample-102;float expected=age>=0&&age<pcm.Length?pcm[age]*.2f:0;
            Assert.That(signal.Read(sample,Emission.Burst,102),Is.EqualTo(expected));
        }
        signal.Append(new AcousticTake.VoiceEvent{sample=104,action=AcousticTake.VoiceAction.Stop});
        Assert.That(signal.Read(104L,Emission.Burst,102),Is.Zero);
        signal.Append(new AcousticTake.VoiceEvent{sample=105,gain=1,loop=true});
        Assert.That(signal.Read(105L,Emission.Burst,102),Is.Zero);
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








}
