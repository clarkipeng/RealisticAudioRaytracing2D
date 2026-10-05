using System;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using UnityEditor;
using UnityEngine;

public static partial class TakeChecks
{
    const string StationTestKey="Acoustics.StationLifecycleCheck";
    [MenuItem("Tools/Acoustics/Test station recording and replay")]
    public static void StartStationLifecycleCheck()
    {
        if(Application.isPlaying)throw new InvalidOperationException("Run this check from Edit mode in the saved station scene.");
        var station=UnityEngine.Object.FindAnyObjectByType<StationDemo>();
        if(station==null || station.gameObject.scene.isDirty || string.IsNullOrEmpty(station.gameObject.scene.path))
            throw new InvalidOperationException("Open and save the station scene before running this check.");
        File.Delete("Recordings/station-lifecycle-result.txt");
        SessionState.SetBool(StationTestKey,true);EditorApplication.isPlaying=true;
    }
    [InitializeOnLoadMethod]
    static void InstallStationLifecycleCheck()
    {
        EditorApplication.playModeStateChanged+=state=> {
            if(state!=PlayModeStateChange.EnteredPlayMode || !SessionState.GetBool(StationTestKey,false))return;
            SessionState.EraseBool(StationTestKey);StationLifecycleCheck();
        };
    }
    static async Task StationTicks(StationDemo station,int count)
    {long end=station.Tick+count;while(station.Tick<end)await Task.Delay(5);}
    static async void StationLifecycleCheck()
    {
        string result="Recordings/station-lifecycle-result.txt";
        try
        {
            StationDemo d;
            do {await Task.Delay(5);d=UnityEngine.Object.FindAnyObjectByType<StationDemo>();}while(d==null || !d.manager.IsRecordingTake);
            string path=d.manager.takePath;d.ReadPlayerInput=false;
            d.SubmitInput(Vector2.up,Vector2.zero);await StationTicks(d,57);
            d.SubmitInput(Vector2.right,Vector2.zero);await StationTicks(d,95);
            d.SubmitInput(Vector2.zero,Vector2.zero);await StationTicks(d,5);
            if(d.Walker.Ground==null || !d.Walker.Ground.IsChildOf(d.carriage))throw new Exception("Player did not walk aboard.");
            while(d.CurrentTrainSpeed<2)await Task.Delay(5);
            d.SubmitInput(Vector2.zero,new Vector2(20,-15),true);await StationTicks(d,60);
            d.SubmitInput(Vector2.up,new Vector2(-20,15));await StationTicks(d,12);
            d.SubmitInput(Vector2.zero,Vector2.zero);await StationTicks(d,35);
            d.SaveRecording();while(d.Busy)await Task.Delay(5);
            string compact=Path.Combine(path,"input-only");StationRecording.ExportInputs(path,compact);
            long bytes=Directory.GetFiles(compact).Sum(f=>new FileInfo(f).Length);
            string rebuilt=await CompactReplayChecks.VerifyLive(compact);
            using(var a=AcousticTake.Open(path))using(var b=AcousticTake.Open(rebuilt))
            {
                if(a.Metadata.stateHash!=b.Metadata.stateHash || a.Metadata.eventHash!=b.Metadata.eventHash)throw new Exception("Input reconstruction drifted.");
                var settings=new TakeRender.Settings {responseStepSamples=256,listenerOutput=TakeRender.ListenerOutput.Binaural};
                TakeRender.RenderDirect(a,settings,Path.Combine(path,"check-audio"));TakeRender.RenderDirect(b,settings,Path.Combine(rebuilt,"check-audio"));
            }
            var audio=TakeComparison.CompareWave(Path.Combine(path,"check-audio/listener-0.wav"),Path.Combine(rebuilt,"check-audio/listener-0.wav"));
            if(!audio.exactMatch || audio.referenceRms<=0)throw new Exception("Reconstructed audio differs.");
            File.WriteAllText(Path.Combine(path,"check-audio.json"),JsonUtility.ToJson(audio,true));
            d.ReplayInputs(compact);
            do {await Task.Delay(5);d=UnityEngine.Object.FindAnyObjectByType<StationDemo>();}while(d==null || !d.IsInputReplay);
            var telemetry=new System.Text.StringBuilder("sample,tick,read,written,underrun\n");
            while(!d.InputReplayFinished || d.manager.Mode!=RayTraceManager.SessionMode.Idle)
            {
                var queue=d.manager.audioManager.OutputQueue;
                if(queue!=null)telemetry.AppendLine(d.manager.LiveSample+","+d.Tick+","+queue.ReadSample+","+queue.WrittenSample+","+queue.UnderrunSamples);
                await Task.Delay(5);
            }
            File.WriteAllText(Path.Combine(path,"replay-timing.csv"),telemetry.ToString());
            using(var take=AcousticTake.Open(path))
            {
                var states=new AcousticTake.State[take.Metadata.scene.objects.Length];take.ReadStoredFrame(take.Metadata.frameCount-1,states);
                for(int i=0;i<states.Length;i++)if(!states[i].position.Equals(d.manager.takeObjects[i].transform.position))throw new Exception("Scene replay drifted.");
            }
            var diagnostics=d.manager.liveDiagnostics;
            File.WriteAllText(Path.Combine(path,"replay-live.json"),JsonUtility.ToJson(diagnostics,true));
            File.WriteAllText(result,"PASS: states/events, stereo reconstruction and input-only scene replay. Export bytes: "+bytes+
                ". Replay underrun/late/overflow samples: "+diagnostics.underrunSamples+" / "+diagnostics.lateSamples+" / "+diagnostics.overflowSamples+"\n"+path);
        }
        catch(Exception e) {Directory.CreateDirectory("Recordings");File.WriteAllText(result,"FAIL: "+e);Debug.LogException(e);}
        finally {EditorApplication.isPlaying=false;}
    }

    static void StationInputs()
    {
        string path=Path.Combine(directory,"station-inputs");
        using(var writer=Writer("station-inputs",Scene()))
        {
            writer.Append(0,States());writer.Append(30,States());writer.Append(80,States());
            writer.Append(new AcousticTake.VoiceEvent {action=AcousticTake.VoiceAction.Play,gain=.2f,loop=true});
            writer.Append(new AcousticTake.VoiceEvent {sample=30,action=AcousticTake.VoiceAction.Gain,gain=.4f});
            writer.Complete();
        }
        var metadata=new StationRecording.Manifest {complete=true,stepSeconds=.02f,ticks=4,inputCount=3,observationCount=3};
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
        void Load() {using(var input=new StationRecording.Playback(path)){} }
        void ReadObservations()
        {using(var input=new StationRecording.Playback(path))while(input.NextObservation(long.MaxValue,out _,out _)){} }
        Inputs();Observations();Header();
        // These are deliberately absent: playback must use scene initialization and inputs.
        foreach(string file in new[]{"states.bin","geometry.bin","clip-0.f32"})File.Delete(Path.Combine(path,file));
        using(var input=new StationRecording.Playback(path))
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
        string exported=Path.Combine(directory,"exported-inputs");StationRecording.ExportInputs(path,exported);
        Check(Directory.GetFiles(exported).Length==5,"Compact export only includes metadata, input/timing and source events.");
        using(var copy=new StationRecording.Playback(exported))Check(copy.Read(0).jump,"Export remains playable without reference files.");
        bool rejected=false;try {StationRecording.ExportInputs(path,exported);}catch(IOException){rejected=true;}
        Check(rejected,"Never overwrite an existing export.");
        var bytes=File.ReadAllBytes(Path.Combine(path,"input.bin"));bytes[0]^=1;File.WriteAllBytes(Path.Combine(path,"input.bin"),bytes);
        Reject(Load,"Reject altered input payload.");
        Inputs(float.NaN);Header();Reject(()=>{using(var input=new StationRecording.Playback(path))input.Read(0);},"Reject non-finite input even with a matching checksum.");
        Inputs(0,4);Header();Reject(()=>{using(var input=new StationRecording.Playback(path))input.Read(0);},"Reject unknown button bits.");
        Inputs();Observations(start:1);Header();Reject(ReadObservations,"Reject an omitted initialization observation.");
        Observations(middle:-1);Header();Reject(ReadObservations,"Reject backward observation ticks.");
        Observations(end:79);Header();Reject(ReadObservations,"Reject a clock map missing the audio endpoint.");
        Observations(finalTick:3);Header();Reject(ReadObservations,"Reject a clock map missing final input ticks.");
        Observations();metadata.complete=false;Header();Reject(Load,"Reject incomplete input recordings.");
        metadata.complete=true;metadata.observationCount=0;Header();Reject(Load,"Reject empty clock maps.");
        metadata.observationCount=3;metadata.inputCount=2;Header();Reject(Load,"Reject incorrect input counts.");
    }
}
