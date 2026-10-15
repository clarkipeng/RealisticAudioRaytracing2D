using UnityEngine.TestTools;
using UnityEditor.SceneManagement;
using System.Threading;
using System.Collections;
using NUnit.Framework;
using System;
using System.IO;
using System.Linq;
using UnityEditor;
using UnityEngine;

public partial class TakeChecks
{
    [Test, Category("Core")]
    public void SceneMotionKeepsItsSimulationTimestamp()
    {
        var scene=EditorSceneManager.OpenPreviewScene("Assets/Scenes/TrainStation.unity");
        try
        {
            var authored=scene.GetRootGameObjects().SelectMany(r=>r.GetComponentsInChildren<SceneSession>(true)).Single();
            using var rig=new SceneReplay.Rig(authored);var simulation=rig.demo;
            simulation.SubmitInput(Vector2.up,Vector2.zero);var before=simulation.player.position;long previous=0;
            foreach(long sample in new long[]{0,1024,1024,5120,9216,14336,44100,88200})
            {
                long stamp=simulation.AdvanceTo(sample,0,44100);
                Assert.That(stamp,Is.InRange(Math.Max(0,sample-882),sample),"A pose must not render future, unobserved source events.");
                if(stamp>previous)Assert.That((simulation.player.position.z-before.z)*44100/(stamp-previous),Is.EqualTo(6).Within(.0002),
                    "Steady walking must not gain artificial Doppler from uneven/repeated audio observations.");
                before=simulation.player.position;previous=stamp;
            }
            long tick=simulation.Tick;
            Assert.That(simulation.AdvanceTo(0,tick,44100),Is.Zero,"A new capture retains the scene pose with a new sample origin.");
            Assert.That(simulation.Tick,Is.EqualTo(tick));
        }
        finally {EditorSceneManager.ClosePreviewScene(scene);}
    }
    [Test, Category("Core")]
    public void StationDoorSealsClosedOpening()
    {
        var preview=EditorSceneManager.OpenPreviewScene("Assets/Scenes/TrainStation.unity");
        try
        {
            var manager=preview.GetRootGameObjects().SelectMany(r=>r.GetComponentsInChildren<RayTraceManager>(true)).Single();
            var scene=SceneToTake.Compile(manager.takeObjects,manager.takeDimension,manager.takeCurveTolerance,manager.takeDefaultMaterial,
                manager.takeSources.Select(s=>s.transform).ToArray(),manager.takeSources.Select(s=>s.clip).ToArray(),manager.takeListeners.ToArray(),out var geometry);
            var states=manager.takeObjects.Select((b,i)=>SceneToTake.Observe(scene,i,b.transform)).ToArray();
            int door=manager.takeObjects.FindIndex(b=>b.transform.name=="Sliding door");
            int bell=manager.takeSources.FindIndex(s=>s.clip!=null && s.clip.name=="station_signal");
            states[scene.listeners[0]].position=new Vector3(4,1.73f,.16f);
            var paths=new TakeRender.DirectPath[scene.sources.Length*scene.listeners.Length];
            TakeRender.SolveDirect(scene,geometry,states,manager.directSettings,paths);
            Assert.That(paths[bell*scene.listeners.Length].pressureGain,Is.Zero,"An oblique ray must not slip behind the closed door.");
            states[door].position+=manager.takeObjects[door].transform.GetComponent<SlidingDoor>().openOffset;
            TakeRender.SolveDirect(scene,geometry,states,manager.directSettings,paths);
            Assert.That(paths[bell*scene.listeners.Length].pressureGain,Is.GreaterThan(0),"Opening the door restores this path.");
        }
        finally {EditorSceneManager.ClosePreviewScene(preview);}
    }

    [Test, Category("Core")]
    public void CompactPayloads()
    {
        string compact=Path.Combine(directory,"compact-payloads");
        string reference=Path.Combine(directory,"reference-payloads");
        foreach(bool retain in new[]{false,true})
        using(var writer=new AcousticTake.Writer(retain?reference:compact,1000,Scene(),
            new[]{new AcousticTake.Primitive {a=Vector3.zero,b=Vector3.right,c=Vector3.up}},new[]{new[]{1f,2f,3f}},retain))
        {
            writer.Append(0,States());writer.Append(0,States());writer.Append(80,States());
            writer.Append(new AcousticTake.VoiceEvent {action=AcousticTake.VoiceAction.Play,gain=1});writer.Complete();
        }
        var a=AcousticTake.ReadManifest(compact);var b=AcousticTake.ReadManifest(reference);
        Check(Directory.GetFiles(compact).Length==2,"Compact take writes metadata/events only; station adds input/timing files.");
        Check(a.sceneHash==b.sceneHash && a.eventHash==b.eventHash,"Omitted payloads retain exactly the same identity hashes.");
        Check(a.frameCount==2 && a.durationSamples==80 && string.IsNullOrEmpty(a.stateHash),"Compact timing/counts remain valid without pose serialization.");
        using(var writer=new AcousticTake.Writer(Path.Combine(directory,"invalid-compact"),1000,Scene(),
            new AcousticTake.Primitive[0],new[]{new[]{1f}},false))
        {
            var invalid=States();invalid[0].position.x=float.NaN;
            Reject(()=>writer.Append(0,invalid),"Compact capture still rejects invalid scene state.");
            Reject(writer.Complete,"A failed compact observation cannot finalize.");
        }
    }

    [Test, Category("Core")]
    public void StationActivation()
    {
        var authoredScene=UnityEditor.SceneManagement.EditorSceneManager.OpenPreviewScene("Assets/Scenes/TrainStation.unity");
        try
        {
            var station=authoredScene.GetRootGameObjects().SelectMany(root=>root.GetComponentsInChildren<SceneSession>(true)).Single();
            station.InitializeSimulation();
            var manager=station.manager;var source=manager.takeSources[0];source.enabled=false;
            var surface=manager.takeObjects.Select(b=>b.transform.GetComponent<AcousticSurface>()).First(value=>value!=null);surface.enabled=false;
            var scene=SceneToTake.Compile(manager.takeObjects,manager.takeDimension,manager.takeCurveTolerance,manager.takeDefaultMaterial,
                manager.takeSources.Select(s=>s.transform).ToArray(),manager.takeSources.Select(s=>s.clip).ToArray(),manager.takeListeners.ToArray(),out _);
            using(var rig=new SceneReplay.Rig(station))
            {
                var observe=typeof(SceneReplay).GetMethod("Observe",System.Reflection.BindingFlags.Static|System.Reflection.BindingFlags.NonPublic);
                var states=(AcousticTake.State[])observe.Invoke(null,new object[]{rig,scene});
                int sourceIndex=manager.takeObjects.FindIndex(b=>b.transform==source.transform),surfaceIndex=manager.takeObjects.FindIndex(b=>b.transform==surface.transform);
                Check((states[sourceIndex].flags&AcousticTake.StateFlags.Source)==0,"Reconstruction preserves disabled initial source");
                Check((states[surfaceIndex].flags&AcousticTake.StateFlags.Geometry)==0,"Reconstruction preserves disabled initial surface");
            }
            string path=Path.Combine(directory,"activation-guard");Directory.CreateDirectory(path);
            using(var recorder=new SceneRecording(station,path))
            {
                recorder.Observe(0,0);source.enabled=true;
                Reject(()=>recorder.Observe(1,0),"Compact input capture rejects unrecorded activation changes");
            }
        }
        finally {UnityEditor.SceneManagement.EditorSceneManager.ClosePreviewScene(authoredScene);}
    }

    [TestCase("duplicate-id"), TestCase("order"), TestCase("settings"), Category("Core")]
    public void SceneActorsKeepRecordedIdentity(string change)
    {
        var player=new GameObject("Player").transform; player.gameObject.AddComponent<CharacterController>();
        player.gameObject.AddComponent<PlayerController>();
        var head=new GameObject("Head").AddComponent<MouseLook>(); head.transform.SetParent(player); head.playerBody=player;
        var session=new GameObject("Session").AddComponent<SceneSession>(); session.player=player;
        session.manager=session.gameObject.AddComponent<RayTraceManager>();
        var a=new GameObject("Door A").AddComponent<SlidingDoor>(); a.id="a";
        var b=new GameObject("Door B").AddComponent<SlidingDoor>(); b.id="b";
        session.actors=new SceneActor[]{a,b};
        if(change=="duplicate-id") {b.id=a.id;Assert.Throws<InvalidOperationException>(session.InitializeSimulation);return;}
        session.InitializeSimulation();
        string path=Path.Combine(directory,"actor-identity");Directory.CreateDirectory(path);
        using(var recorder=new SceneRecording(session,path))
        {
            recorder.BeforeTick(0,default,Time.fixedDeltaTime);
            if(change=="order") session.actors=new SceneActor[]{b,a}; else a.speed+=1;
            Assert.Throws<InvalidOperationException>(()=>recorder.BeforeTick(1,default,Time.fixedDeltaTime));
        }
    }

    [Test, Category("Core")]
    public void BatchJobsRequireExplicitComparison()
    {
        var job=TakeBatch.ParseJob("{\"take\":\"input\",\"output\":\"output\"}");
        Check(job.comparison==null && job.referenceCheck==null && job.settings.responseStepSamples==512,"A render job cannot implicitly request comparison/reference work.");
        job=TakeBatch.ParseJob("{\"take\":\"input\",\"output\":\"output\",\"comparison\":{\"responseStepSamples\":128}}");
        Check(job.comparison.responseStepSamples==128 && job.referenceCheck==null,"A comparison does not implicitly request a finer reference.");
        job=TakeBatch.ParseJob("{\"take\":\"input\",\"output\":\"output\",\"comparison\":null,\"referenceCheck\":null}");
        Check(job.comparison==null && job.referenceCheck==null,"Explicit null optional jobs remain disabled.");
        job=TakeBatch.ParseJob("{\"take\":\"input\",\"output\":\"output\",\"comparison\":{\"responseStepSamples\":128},\"referenceCheck\":{\"responseStepSamples\":64}}");
        Check(job.referenceCheck.responseStepSamples==64,"Explicit reference settings are retained.");
        Reject(()=>TakeBatch.ParseJob("{\"take\":\"input\",\"output\":\"output\",\"referenceCheck\":{}}"),"Reference-only jobs cannot silently discard the requested check.");
    }

    [Test, Category("Core")]
    public void ProjectIdentityTracksAuthoredInputs()
    {
        string project=Path.Combine(directory,"project-identity"),copyProject=project+"-copy";
        foreach(string root in new[]{project,copyProject})
            foreach(string folder in new[]{"Assets","Packages","ProjectSettings"})
            { Directory.CreateDirectory(Path.Combine(root,folder));File.WriteAllText(Path.Combine(root,folder,"input"),folder); }
        string identity=SceneRecording.HashProject(project);
        Check(identity==SceneRecording.HashProject(copyProject),"Project identity survives a different checkout directory.");
        Directory.CreateDirectory(Path.Combine(copyProject,"Library"));File.WriteAllText(Path.Combine(copyProject,"Library","local"),"rebuild");
        Check(identity==SceneRecording.HashProject(copyProject),"Generated Library contents do not affect replay identity.");
        File.WriteAllText(Path.Combine(copyProject,"Assets","input"),"changed");
        Check(identity!=SceneRecording.HashProject(copyProject),"Changed asset/code bytes invalidate replay identity.");
        File.WriteAllText(Path.Combine(copyProject,"Assets","input"),"Assets");File.Move(Path.Combine(copyProject,"Assets","input"),Path.Combine(copyProject,"Assets","renamed"));
        Check(identity!=SceneRecording.HashProject(copyProject),"Renamed project inputs invalidate replay identity.");
        File.Delete(Path.Combine(copyProject,"Assets","renamed"));
        Check(identity!=SceneRecording.HashProject(copyProject),"Missing project inputs invalidate replay identity.");
    }

    [TestCase("station-input-from-init-v2"), TestCase("scene-input-from-init-v1"), Category("Core")]
    public void InputLogRoundTripAndValidation(string schema)
    {
        string path=Path.Combine(directory,"station-inputs");
        using(var writer=Writer("station-inputs",Scene()))
        {
            writer.Append(0,States());writer.Append(30,States());writer.Append(80,States());
            writer.Append(new AcousticTake.VoiceEvent {action=AcousticTake.VoiceAction.Play,gain=.2f,loop=true});
            writer.Append(new AcousticTake.VoiceEvent {sample=30,action=AcousticTake.VoiceAction.Gain,gain=.4f});
            writer.Complete();
        }
        var metadata=new SceneRecording.Manifest {schema=schema,complete=true,stepSeconds=.02f,ticks=4,inputCount=3,observationCount=3};
        void Inputs(float x=0,byte flags=1)
        {
            using(var writer=new BinaryWriter(File.Create(Path.Combine(path,"input.bin"))))
            {
                writer.Write(0L);writer.Write(x);writer.Write(1f);writer.Write(12f);writer.Write(-3f);writer.Write(flags);
                writer.Write(2L);writer.Write(1f);writer.Write(0f);writer.Write(0f);writer.Write(0f);writer.Write((byte)2);
                writer.Write(3L);writer.Write(0f);writer.Write(0f);writer.Write(0f);writer.Write(0f);writer.Write((byte)0);
            }
        }
        void Observations(long start=0,long middle=2,long end=80,long finalTick=4)
        {
            using(var writer=new BinaryWriter(File.Create(Path.Combine(path,"input-observations.bin"))))
            {writer.Write(start);writer.Write(0L);writer.Write(30L);writer.Write(middle);writer.Write(end);writer.Write(finalTick);}
        }
        void Header()
        {
            metadata.takeHash=AcousticTake.HashFile(Path.Combine(path,"take.json"));
            metadata.inputHash=AcousticTake.HashFile(Path.Combine(path,"input.bin"));
            metadata.observationHash=AcousticTake.HashFile(Path.Combine(path,"input-observations.bin"));
            File.WriteAllText(Path.Combine(path,"input.json"),JsonUtility.ToJson(metadata));
        }
        void Load() {using(var input=new SceneRecording.Playback(path)){} }
        void ReadObservations()
        {using(var input=new SceneRecording.Playback(path))while(input.NextObservation(long.MaxValue,out _,out _)){} }
        Inputs();Observations();Header();
        // These are deliberately absent: playback must use scene initialization and inputs.
        foreach(string file in new[]{"states.bin","geometry.bin","clip-0.f32"})File.Delete(Path.Combine(path,file));
        using(var input=new SceneRecording.Playback(path))
        {
            Check(input.Events.Length==2 && input.Events[1].sample==30,"Input-only replay retains source sample timing.");
            Check(input.NextObservation(0,out long sample,out long tick) && sample==0 && tick==0,"Replay starts at tick/sample zero.");
            Check(!input.NextObservation(29,out _,out _),"Replay waits for the recorded observation time.");
            Check(input.NextObservation(30,out sample,out tick) && tick==2,"Replay follows stored timing rather than frame duration.");
            var first=input.Read(0);var held=input.Read(1);var interact=input.Read(2);var stop=input.Read(3);
            Check(first.jump && first.look==new Vector2(12,-3),"Jump and mouse delta survive input serialization.");
            Check(held.movement==Vector2.up && !held.jump && held.look==Vector2.zero,"Held movement persists without repeating pulses.");
            Check(interact.interact && interact.movement==Vector2.right && !stop.interact && stop.movement==Vector2.zero,"Interaction and release occur on their own ticks.");
            Reject(()=>input.Read(0),"Replay cannot seek backwards.");
            Check(input.NextObservation(80,out sample,out tick) && tick==4 && !input.NextObservation(long.MaxValue,out _,out _),"Final observation terminates replay.");
        }
        Check(!File.ReadAllText(Path.Combine(path,"input.json")).Contains("\"initial\""),"Input format contains no world snapshot.");
        string exported=Path.Combine(directory,"exported-inputs");SceneRecording.ExportInputs(path,exported);
        Check(Directory.GetFiles(exported).Length==5,"Compact export only includes metadata, input/timing and source events.");
        using(var copy=new SceneRecording.Playback(exported))Check(copy.Read(0).jump,"Export remains playable without reference files.");
        bool rejected=false;try {SceneRecording.ExportInputs(path,exported);}catch(IOException){rejected=true;}
        Check(rejected,"Never overwrite an existing export.");
        var bytes=File.ReadAllBytes(Path.Combine(path,"input.bin"));bytes[0]^=1;File.WriteAllBytes(Path.Combine(path,"input.bin"),bytes);
        Reject(Load,"Reject altered input payload.");
        Inputs(float.NaN);Header();Reject(()=>{using(var input=new SceneRecording.Playback(path))input.Read(0);},"Reject non-finite input even with a matching checksum.");
        Inputs(0,4);Header();Reject(()=>{using(var input=new SceneRecording.Playback(path))input.Read(0);},"Reject unknown button bits.");
        Inputs();Observations(start:1);Header();Reject(ReadObservations,"Reject an omitted initialization observation.");
        Observations(middle:-1);Header();Reject(ReadObservations,"Reject backward observation ticks.");
        Observations(end:79);Header();Reject(ReadObservations,"Reject a clock map missing the audio endpoint.");
        Observations(finalTick:3);Header();Reject(ReadObservations,"Reject a clock map missing final input ticks.");
        Observations();metadata.complete=false;Header();Reject(Load,"Reject incomplete input recordings.");
        metadata.complete=true;metadata.observationCount=0;Header();Reject(Load,"Reject empty clock maps.");
        metadata.observationCount=3;metadata.inputCount=2;Header();Reject(Load,"Reject incorrect input counts.");
    }
}

// Real scene lifecycle and device-clock coverage. Numeric contracts live in TakeChecks.
public abstract class SceneEndToEndTests
{
    protected SceneSession session;
    protected abstract string ScenePath {get;}
    string recording;
    bool previousBackground;

    [UnitySetUp]
    public IEnumerator OpenSession()
    {
        if(!SystemInfo.supportsComputeShaders || !SystemInfo.supportsAsyncGPUReadback)
            Assert.Ignore("Station tests require a GPU with asynchronous readback.");
        if(AudioSettings.GetConfiguration().speakerMode!=AudioSpeakerMode.Stereo)
            Assert.Ignore("Station tests require a stereo audio device.");
        // Configure the loaded copy before Start can run. Test Runner restores
        // the original scene; these fixture settings are never saved to disk.
        EditorSceneManager.OpenScene(ScenePath,OpenSceneMode.Single);
        session=UnityEngine.Object.FindAnyObjectByType<SceneSession>();
        Assert.That(session,Is.Not.Null);
        session.enabled=false;
        session.player.GetComponent<PlayerController>().enabled=false;
        session.player.GetComponentInChildren<MouseLook>().enabled=false;
        EditorApplication.ExecuteMenuItem("Window/General/Game");
        yield return new EnterPlayMode();
        previousBackground=Application.runInBackground;
        Application.runInBackground=true;
        session=UnityEngine.Object.FindAnyObjectByType<SceneSession>();
        Assert.That(session,Is.Not.Null);
        var walker=session.player.GetComponent<PlayerController>();walker.externalClock=true;walker.enabled=true;
        var look=session.player.GetComponentInChildren<MouseLook>();look.externalClock=true;look.enabled=true;
        session.ReadPlayerInput=false;
    }

    protected IEnumerator StartSession(bool reference=false)
    {
        session.retainFullReference=reference;session.enabled=true;
        yield return Until(()=>session.manager.IsRecordingTake,"session capture startup");
        recording=session.manager.takePath;
        session.SetWalking(false);
    }

    [UnityTearDown]
    public IEnumerator CloseSession()
    {
        // Exiting Play disposes the session even when an assertion/timeout interrupted it.
        if(Application.isPlaying)
        {
            if(session==null)session=UnityEngine.Object.FindAnyObjectByType<SceneSession>();
            if(session!=null && session.manager.IsRecordingTake)
            {
                // Keep the path even if a test failed during startup.
                recording=session.manager.takePath;
                try {session.manager.EndTakeRecording();}
                catch(Exception e) {Debug.LogException(e);}
            }
            Application.runInBackground=previousBackground;
            yield return new ExitPlayMode();
        }
        if(!string.IsNullOrEmpty(recording) && Directory.Exists(recording))Directory.Delete(recording,true);
    }

    protected static IEnumerator Until(Func<bool> ready,string operation,double seconds=30)
    {
        double end=Time.realtimeSinceStartupAsDouble+seconds;
        while(!ready())
        {
            Assert.That(Time.realtimeSinceStartupAsDouble,Is.LessThan(end),"Timed out waiting for "+operation);
            yield return null;
        }
    }

    protected IEnumerator VerifyReplay()
    {
        session.manager.EndTakeRecording();
        yield return Until(()=>session.manager.Mode==RayTraceManager.SessionMode.Idle,"capture drain");

        string compact=Path.Combine(recording,"input-only");
        SceneRecording.ExportInputs(recording,compact);
        string rebuilt=SceneReplay.Reconstruct(compact,Path.Combine(compact,"rebuilt"));
        using(var original=AcousticTake.Open(recording))
        using(var replay=AcousticTake.Open(rebuilt))
        {
            Assert.That(replay.Metadata.stateHash,Is.EqualTo(original.Metadata.stateHash),"Recorded observations must reconstruct exactly.");
            Assert.That(replay.Metadata.eventHash,Is.EqualTo(original.Metadata.eventHash),"Source events must retain sample timing.");
        }

        var previous=session;
        session.ReplayInputs(compact);
        yield return Until(()=>
        {
            var next=UnityEngine.Object.FindAnyObjectByType<SceneSession>();
            if(next==null || next==previous || !next.IsInputReplay)return false;
            session=next;session.SetWalking(false);return true;
        },"fresh-scene input replay");
        using(var take=AcousticTake.Open(recording))
        {
            yield return Until(()=>session.InputReplayFinished && session.manager.Mode==RayTraceManager.SessionMode.Idle,
                "input replay completion",(double)take.Metadata.durationSamples/take.Metadata.sampleRate+30);
            var expected=new AcousticTake.State[take.Metadata.scene.objects.Length];
            take.ReadStoredFrame(take.Metadata.frameCount-1,expected);
            for(int i=0;i<expected.Length;i++)
            {
                var actual=session.manager.takeObjects[i].transform;
                Assert.That(actual.position,Is.EqualTo(expected[i].position),"Final replay position: "+take.Metadata.scene.objects[i].id);
                Assert.That(Quaternion.Angle(actual.rotation,expected[i].rotation),Is.LessThan(.05f),"Final replay orientation: "+take.Metadata.scene.objects[i].id);
            }
        }
    }

}

public sealed class StationEndToEndTests : SceneEndToEndTests
{
    protected override string ScenePath => "Assets/Scenes/TrainStation.unity";
    [UnityTest, Category("EndToEnd"), Category("GPU")]
    public IEnumerator RecordingReconstructsAndReplays()
    {
        yield return StartSession(true);
        // Feed exact simulation ticks: a frame-based wait can overshoot several
        // physics ticks under GPU load and walk past the carriage entrance.
        long start=session.Tick;
        session.SampleInput=tick=>session.SubmitInput(tick-start<57?Vector2.up:tick-start<152?Vector2.right:Vector2.zero,Vector2.zero);
        yield return Until(()=>session.Tick>=start+157,"walking aboard");
        Assert.That(session.Walker.Ground!=null && session.Walker.Ground.IsChildOf(session.GetComponent<StationDemo>().carriage),Is.True,
            "The scripted player must actually walk aboard before testing moving-platform replay.");
        yield return Until(()=>session.GetComponent<StationDemo>().CurrentTrainSpeed>2,"train departure",session.GetComponent<StationDemo>().stationDwell+15);
        start=session.Tick;
        session.SampleInput=tick=>session.SubmitInput(tick-start>=60 && tick-start<72?Vector2.up:Vector2.zero,
            tick==start?new Vector2(20,-15):tick==start+60?new Vector2(-20,15):Vector2.zero,tick==start);
        yield return Until(()=>session.Tick>=start+107,"onboard motion");
        session.SampleInput=null;
        yield return VerifyReplay();
    }

    [UnityTest, Category("EndToEnd"), Category("GPU"), Category("Realtime")]
    [Explicit("Measures real device deadlines under session graphics load. Run explicitly on an otherwise idle target machine.")]
    public IEnumerator StationDeliversAudioBeforeItsDeadline()
    {
        yield return StartSession();
        var manager=session.manager;
        Assert.That(manager.directSettings.audioBackend,Is.EqualTo(TakeRender.AudioBackend.Gpu));
        Assert.That(manager.directSettings.transport,Is.EqualTo(TakeRender.Transport.Raytraced));
        Assert.That(manager.liveDiagnostics.backend,Does.StartWith("GPU-"));
        var queue=manager.audioManager.OutputQueue;
        Assert.That(queue,Is.Not.Null);
        yield return Until(()=>queue.ReadSample>=manager.sampleRate*2L,"two seconds of device warmup");
        long start=queue.ReadSample,callbacks=Interlocked.Read(ref queue.CallbackCount);
        long underruns=Interlocked.Read(ref queue.UnderrunSamples),late=queue.LateSamples,overflow=queue.OverflowSamples;
        long mismatch=Interlocked.Read(ref queue.ChannelMismatchSamples),overBudget=Interlocked.Read(ref queue.OverBudgetCallbacks);
        long end=start+manager.sampleRate*20L;
        double timeout=Time.realtimeSinceStartupAsDouble+40;
        while(queue.ReadSample<end)
        {
            Assert.That(Time.realtimeSinceStartupAsDouble,Is.LessThan(timeout),"Audio device stopped advancing.");
            session.SubmitInput(session.Tick%200<100?Vector2.up:Vector2.zero,new Vector2(.2f,0));
            yield return null;
        }
        manager.EndTakeRecording();long stopTick=session.Tick;
        yield return Until(()=>manager.Mode==RayTraceManager.SessionMode.Idle,"recording tail and output drain");
        Assert.That(session.Tick,Is.GreaterThan(stopTick),"Saving must not freeze scene controls during the output tail.");
        long missed=Interlocked.Read(ref queue.UnderrunSamples)-underruns;
        long tardy=queue.LateSamples-late,overflowed=queue.OverflowSamples-overflow;
        long mismatched=Interlocked.Read(ref queue.ChannelMismatchSamples)-mismatch;
        long slow=Interlocked.Read(ref queue.OverBudgetCallbacks)-overBudget;
        TestContext.WriteLine("Device frames="+(queue.ReadSample-start)+", underrun="+missed+", late="+tardy+
            ", overflow="+overflowed+", channel mismatch="+mismatched+", over-budget callbacks="+slow+
            ", lead="+manager.monitorLeadSamples+", hardware="+SystemInfo.graphicsDeviceName);
        TestContext.WriteLine("Producer diagnostics: "+JsonUtility.ToJson(manager.liveDiagnostics));
        Assert.That(Interlocked.Read(ref queue.CallbackCount),Is.GreaterThan(callbacks),"The real audio callback must run.");
        Assert.That(manager.liveDiagnostics.responseCount,Is.GreaterThan(0),"The producer must trace responses.");
        Assert.That(session.GetComponent<StationDemo>().CurrentTrainSpeed,Is.GreaterThan(0),"The test must include a moving train.");
        Assert.That(new[]{missed,tardy,overflowed,mismatched,slow},Is.All.Zero,
            "Audio delivery missed its contract. Underrun and late counters overlap; do not add them.");
    }
}

public sealed class ChurchEndToEndTests : SceneEndToEndTests
{
    protected override string ScenePath => "Assets/Scenes/Church.unity";
    [UnityTest, Category("EndToEnd"), Category("GPU")]
    public IEnumerator ChurchWalksThroughDoorAndReplays()
    {
        yield return StartSession(true);
        Assert.That(session.GetComponent<StationDemo>(),Is.Null,"The shared session must work without a train.");
        var door=(SlidingDoor)session.actors.Single();
        long start=session.Tick;
        // 171 ticks down the clear central aisle, then approach the east doorway.
        session.SampleInput=tick=>session.SubmitInput(tick-start<171?Vector2.up:tick-start<207?Vector2.right:Vector2.zero,Vector2.zero);
        yield return Until(()=>session.Tick>=start+211,"door approach");
        Assert.That(session.player.position.z,Is.InRange(7.8f,8.2f));
        Assert.That(session.Walker.Ground,Is.Not.Null,"The player must stay on the floor.");
        start=session.Tick;
        session.SampleInput=tick=>session.SubmitInput(tick-start>=85 && tick-start<129?Vector2.right:Vector2.zero,
            tick==start+150?new Vector2(25,10):Vector2.zero,interact:tick==start);
        yield return Until(()=>session.Tick>=start+155,"door opening and side-room entry");
        Assert.That(door.IsOpen,Is.True);
        Assert.That(session.player.position.x,Is.GreaterThan(8),"The player must pass the collider opening into the room.");
        Assert.That(session.player.position.y,Is.InRange(-.1f,.1f),"The player must not fall through the floor.");
        session.SampleInput=null;
        yield return VerifyReplay();
    }
}
