using System;
using System.Collections.Generic;
using System.Linq;
using System.IO;
using NUnit.Framework;
using UnityEditor.SceneManagement;
using UnityEngine;

public sealed class SceneSequenceChecks
{
    SceneSequence sequence;
    [SetUp] public void Create()
    {
        sequence=ScriptableObject.CreateInstance<SceneSequence>();sequence.duration=5;
        sequence.cues=new[]{new SceneSequence.Cue{id="call",target="bird",time=0,duration=1.003f,gain=.7f},
            new SceneSequence.Cue{id="strike",target="bell",time=.123f,duration=2,gain=1}};
        sequence.weather=new[]{new SceneSequence.WeatherKey{time=0},new SceneSequence.WeatherKey{time=2,rain=1,cloud=1},
            new SceneSequence.WeatherKey{time=4}};
        sequence.levels=new[]{new SceneSequence.LevelTrack{id="shower",target="rain",start=0,end=4,gain=.8f}};
    }
    [TearDown] public void Destroy()=>UnityEngine.Object.DestroyImmediate(sequence);
    [Test] public void CommandsDoNotDependOnObservationPartitionOrRepeatedFrames()
    {
        var run=new SceneSequence.Run(sequence,48000);var a=new List<SceneSequence.Run.Command>();
        foreach(var sample in new long[]{0,0,256,850,5000,5000,19000,70000,240000})run.Advance(sample,a.Add);
        CollectionAssert.AreEqual(run.Commands,a);
        Assert.That(a.Count(c=>c.audio.action==AcousticTake.VoiceAction.Play),Is.EqualTo(3));
        Assert.That(a.Single(c=>c.occurrence=="strike"&&c.audio.action==AcousticTake.VoiceAction.Play).audio.sample,Is.EqualTo(5904));
        Assert.That(a.Single(c=>c.occurrence=="call"&&c.audio.action==AcousticTake.VoiceAction.Stop).audio.sample,Is.EqualTo(48144));
        Assert.Throws<InvalidOperationException>(()=>run.Advance(0,a.Add));
        run.Reset();var b=new List<SceneSequence.Run.Command>();run.Advance(240000,b.Add);CollectionAssert.AreEqual(a,b);
    }
    [Test] public void WeatherAndWetnessCanBeEvaluatedInAnyOrder()
    {
        Assert.That(sequence.At(1).rain,Is.EqualTo(.5f));
        Assert.That(sequence.At(1).wetness,Is.EqualTo(.25*.065).Within(1e-6));
        Assert.That(sequence.At(4).wetness,Is.EqualTo(2*.065).Within(1e-6));
        Assert.That(sequence.At(100).wetness,Is.EqualTo(sequence.At(4).wetness));
        Assert.That(sequence.At(1).rain,Is.EqualTo(.5f));
    }
    [Test] public void SourceConflictsAndDuplicateOccurrencesFailBeforeCapture()
    {
        var cues=sequence.cues;var c=cues[1];c.target="bird";cues[1]=c;
        Assert.Throws<InvalidOperationException>(sequence.Validate);
        c.target="bell";c.id="call";cues[1]=c;Assert.Throws<InvalidOperationException>(sequence.Validate);
    }
    [Test] public void ActorFreezesAuthoringAndResumesAnActivityOnANewAudioClock()
    {
        var root=new GameObject("Sequence test");AudioClip clip=null;
        try
        {
            sequence.cues=Array.Empty<SceneSequence.Cue>();
            var actor=root.AddComponent<SceneSequenceActor>();actor.sequence=sequence;
            var source=new GameObject("rain");source.transform.parent=root.transform;
            clip=AudioClip.Create("loop",48000,1,48000,false);source.AddComponent<RaytracedAudioSource>().clip=clip;
            actor.bindings=new[]{new SceneSequenceActor.Binding{id="rain",target=source.transform}};actor.Initialize();
            sequence.weather[1].rain=.1f;
            Assert.That(actor.SettingsUnchanged,Is.False);
            Assert.That(actor.Definition.At(2).rain,Is.EqualTo(1));
            actor.Apply(2.5);var emitted=new List<AcousticTake.VoiceEvent>();actor.BeginAudio(48000,(_,e)=>emitted.Add(e));
            Assert.That(emitted.Count,Is.EqualTo(1));Assert.That(emitted[0].sample,Is.Zero);
            Assert.That(emitted[0].clipOffset,Is.EqualTo(24000));Assert.That(emitted[0].gain,Is.EqualTo(.6f).Within(1e-6));
            actor.Apply(4);actor.Drain((_,e)=>emitted.Add(e));
            Assert.That(emitted.Last().action,Is.EqualTo(AcousticTake.VoiceAction.Stop));
            Assert.That(emitted.Last().sample,Is.EqualTo(72000));
            int count=emitted.Count;actor.Drain((_,e)=>emitted.Add(e));Assert.That(emitted.Count,Is.EqualTo(count));
        }
        finally {UnityEngine.Object.DestroyImmediate(root);if(clip!=null)UnityEngine.Object.DestroyImmediate(clip);}
    }
    [Test] public void IsolatedChurchReplayRemapsSequenceBindingsAndMovesTheSameBird()
    {
        var scene=EditorSceneManager.OpenPreviewScene("Assets/Scenes/Church.unity");
        try
        {
            var session=scene.GetRootGameObjects().SelectMany(r=>r.GetComponentsInChildren<SceneSession>()).Single();
            var actor=session.actors.OfType<SceneSequenceActor>().Single();actor.Initialize();
            using var rig=new SceneReplay.Rig(session);var copy=rig.demo.actors.OfType<SceneSequenceActor>().Single();
            Assert.That(copy.Target("bird"),Is.Not.SameAs(actor.Target("bird")));
            for(int tick=0;tick<1500;tick++)
            {actor.Step(default,session.player,.02f);rig.demo.Step(default,.02f);}
            Assert.That(Vector3.Distance(actor.Target("bird").position,copy.Target("bird").position),Is.Zero);
            Assert.That(actor.SettingsJson,Is.EqualTo(copy.SettingsJson));
            Assert.That(session.manager.directSettings.maxBounces,Is.EqualTo(32));
        }
        finally {EditorSceneManager.ClosePreviewScene(scene);}
    }
    [Test] public void SilentCaptureStoresCueSamplesAndRejectsRetroactiveCommands()
    {
        var scene=EditorSceneManager.OpenPreviewScene("Assets/Scenes/Church.unity");
        string path=Path.Combine(Path.GetTempPath(),"church-sequence-"+Guid.NewGuid().ToString("N"));
        try
        {
            var session=scene.GetRootGameObjects().SelectMany(r=>r.GetComponentsInChildren<SceneSession>()).Single();
            session.InitializeSimulation();var manager=session.manager;manager.monitorLive=false;
            var actor=session.actors.OfType<SceneSequenceActor>().Single();manager.BeginTakeRecording(path);actor.BeginAudio(manager.sampleRate,manager.RecordSourceEventAt);
            for(int i=0;i<250;i++){session.Step(default,.02f);manager.CaptureSimulationObservation();}
            Assert.Throws<InvalidOperationException>(()=>manager.RecordSourceEventAt(actor.Target("bell"),new AcousticTake.VoiceEvent{sample=0,action=AcousticTake.VoiceAction.Stop}));
            manager.CaptureSimulationObservation(true);
            using var take=AcousticTake.Open(path);
            int bell=manager.takeSources.FindIndex(s=>s.transform==actor.Target("bell"));
            Assert.That(take.Events.Single(e=>e.sourceIndex==bell).sample,Is.EqualTo(4L*manager.sampleRate));
            Assert.That(take.Metadata.durationSamples,Is.EqualTo(5L*manager.sampleRate));
        }
        finally {EditorSceneManager.ClosePreviewScene(scene);if(Directory.Exists(path))Directory.Delete(path,true);}
    }
}
