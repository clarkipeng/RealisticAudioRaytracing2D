#if UNITY_EDITOR || DEVELOPMENT_BUILD
using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Reflection;
using UnityEngine;

// Opt-in two-process integration probe. No device capture; the voice stimulus is explicitly synthetic.
[DefaultExecutionOrder(-35)]
public sealed class MultiplayerSmoke : MonoBehaviour
{
    [Serializable] sealed class Result
    {
        public string role,take,sourceHash,sceneHash,stateHash,eventHash,status="running",error;
        public int receivedPackets,concealedFrames,latePackets,frames,nonzeroStreams,responseStepSamples;
        public float remoteMetres,sharedMetres,replayPositionError,replayAngleError;
        public long underrunSamples,lateSamples,callbacks;
        public bool replayOffline,microphoneClosed;
    }
    static string role,output;
    Result result;
    SceneSession session;
    MultiplayerSession network;
    MethodInfo voiceState,voiceSamples;
    readonly float[] pcm=new float[960];
    bool speaking,feeding,interacted,requested;
    long encodedSamples,voiceSourceStart;
    double speakingTime;
    readonly Dictionary<Transform,Vector3> remoteInitial=new Dictionary<Transform,Vector3>();
    double start;
    Transform shared;
    Vector3 sharedInitial;
    readonly List<string> errors=new List<string>();
    [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.BeforeSceneLoad)]
    static void Initialize()
    {
        var args=Environment.GetCommandLineArgs();
        string Arg(string name)=>Array.Find(args,a=>a.StartsWith(name+"=",StringComparison.Ordinal))?.Substring(name.Length+1);
        role=Arg("--acoustics-smoke");output=Arg("--acoustics-result");
        if(role==null)return;
        if((role!="host" && role!="client") || string.IsNullOrEmpty(output))throw new ArgumentException("Smoke test needs host/client and result path.");
        MultiplayerSession.PendingMode=role;MultiplayerSession.PendingAddress="127.0.0.1";
        string step=Arg("--acoustics-smoke-step");
        if(step!=null)UnityEngine.SceneManagement.SceneManager.sceneLoaded+=(scene,mode)=>FindAnyObjectByType<RayTraceManager>().directSettings.responseStepSamples=int.Parse(step);
        var go=new GameObject("Multiplayer integration probe");DontDestroyOnLoad(go);go.AddComponent<MultiplayerSmoke>();
    }
    IEnumerator Start()
    {
        result=new Result {role=role};Application.runInBackground=true;Application.targetFrameRate=60;
        Application.logMessageReceived+=Log;
        yield return Wait(()=> (session=FindAnyObjectByType<SceneSession>())!=null && session.manager.IsRecordingTake,"network capture startup",25);
        if(errors.Count>0) {Finish();yield break;}
        network=session.multiplayer;network.microphone.enabled=false;session.ReadPlayerInput=false;session.SetWalking(false);
        voiceState=typeof(MultiplayerSession).GetMethod("LocalVoiceState",BindingFlags.NonPublic|BindingFlags.Instance);
        voiceSamples=typeof(MultiplayerSession).GetMethod("LocalVoiceSamples",BindingFlags.NonPublic|BindingFlags.Instance);
        shared=network.sharedObjects.First();sharedInitial=shared.position;
        start=AudioSettings.dspTime;feeding=true;result.take=session.manager.takePath;result.responseStepSamples=session.manager.directSettings.responseStepSamples;
        double seconds=role=="host"?18:11;
        while(AudioSettings.dspTime-start<seconds && errors.Count==0)yield return null;
        StopVoice();feeding=false;
        result.receivedPackets=network.ReceivedPackets;result.concealedFrames=network.ConcealedFrames;result.latePackets=network.LatePackets;
        ReadCounters();
        session.SaveRecording();yield return Wait(()=>!session.Busy,"capture finalization",15);
        if(errors.Count>0) {Finish();yield break;}
        try
        {
            using var take=AcousticTake.Open(result.take);result.sourceHash=AcousticTake.SourceHash(take.Metadata);
            result.sceneHash=take.Metadata.sceneHash;result.stateHash=take.Metadata.stateHash;result.eventHash=take.Metadata.eventHash;result.frames=take.Metadata.frameCount;
            for(int s=0;s<take.Metadata.scene.sources.Length;s++)if(take.Metadata.scene.sources[s].streamed)
            {
                var source=take.ReadStream(s);double energy=0;for(long n=0;n<source.End;n++)energy+=source.Read(n)*source.Read(n);
                if(energy>1)result.nonzeroStreams++;
            }
            if(result.receivedPackets<20 || result.nonzeroStreams<2)throw new Exception("Both peers must receive and save audible voice.");
            if(result.remoteMetres<.1f || result.sharedMetres<.1f)throw new Exception("Peer movement and the shared actor must both replicate.");
            session.ReplayInputs(result.take);
        }
        catch(Exception e) {errors.Add(e.ToString());Finish();yield break;}
        var old=session;
        yield return Wait(()=> (session=FindAnyObjectByType<SceneSession>())!=null && session!=old && session.manager.IsReplayingTake,"offline source replay",20);
        if(errors.Count>0) {Finish();yield break;}
        result.replayOffline=!session.multiplayer.Connected;result.microphoneClosed=!session.multiplayer.microphone.Active;
        yield return Wait(()=>!session.manager.IsTakeReplayPlaying,"replay completion",seconds+20);
        if(errors.Count==0)
        {
            using var take=AcousticTake.Open(result.take);var states=new AcousticTake.State[take.Metadata.scene.objects.Length];take.Evaluate(take.Metadata.durationSamples,states);
            for(int i=0;i<states.Length;i++)
            {
                var actual=session.manager.takeObjects[i].transform;
                result.replayPositionError=Math.Max(result.replayPositionError,Vector3.Distance(states[i].position,actual.position));
                result.replayAngleError=Math.Max(result.replayAngleError,Quaternion.Angle(states[i].rotation,actual.rotation));
            }
            if(result.replayPositionError>1e-5f || result.replayAngleError>.05f || !result.replayOffline || !result.microphoneClosed)errors.Add("Replay final state/network/device isolation failed.");
        }
        Finish();
    }
    void Update()
    {
        if(!feeding || session==null)return;
        ReadCounters();
        if(!session.manager.IsLiveSimulation)return;
        double elapsed=AudioSettings.dspTime-start;var manager=session.manager;var stream=network.microphone.source.Stream;
        bool speak=elapsed>=1 && elapsed<6;
        if(speak && !speaking) {voiceState.Invoke(network,new object[]{true,network.microphone.Clock()});speaking=true;encodedSamples=0;speakingTime=AudioSettings.dspTime;voiceSourceStart=stream.End;}
        if(!speak)StopVoice();
        long end=manager.LiveSample;
        while(stream.End<end)
        {
            int count=(int)Math.Min(pcm.Length,end-stream.End);
            if(speaking)
            {
                for(int i=0;i<count;i++)pcm[i]=.15f*Mathf.Sin((float)(2*Math.PI*(role=="host"?440:660)*(stream.End-voiceSourceStart+i)/manager.sampleRate));
                stream.Append(pcm,0,count,end);
            }
            else stream.SilenceUntil(end,end);
        }
        if(speaking)
        {
            long count48=(long)((AudioSettings.dspTime-speakingTime)*48000);
            while(encodedSamples<count48)
            {
                int count=(int)Math.Min(pcm.Length,count48-encodedSamples);
                for(int i=0;i<count;i++)pcm[i]=.15f*Mathf.Sin((float)(2*Math.PI*(role=="host"?440:660)*(encodedSamples+i)/48000));
                voiceSamples.Invoke(network,new object[]{pcm,count});encodedSamples+=count;
            }
        }
        if(role=="client" && elapsed>2 && !interacted)
        {
            var door=session.actors.OfType<SlidingDoor>().FirstOrDefault();
            if(door!=null) {network.TeleportPlayer(new Vector3(door.transform.position.x-1,0,door.transform.position.z),session.player.rotation);}
            interacted=true;
        }
        if(role=="client" && elapsed>3 && !requested) {network.RequestInteraction();requested=true;}
        session.SubmitInput(elapsed>4 && elapsed<5?new Vector2(0,.3f):Vector2.zero,new Vector2(.1f,0));
        result.sharedMetres=Math.Max(result.sharedMetres,Vector3.Distance(shared.position,sharedInitial));
        foreach(var voice in network.remoteVoices)if(voice.gameObject.activeInHierarchy)
        {
            var root=voice.transform.parent;
            if(!remoteInitial.ContainsKey(root))remoteInitial.Add(root,root.position);
            result.remoteMetres=Math.Max(result.remoteMetres,Vector3.Distance(root.position,remoteInitial[root]));
        }
    }
    void ReadCounters()
    {
        var queue=session.manager.audioManager.OutputQueue;if(queue==null)return;
        result.underrunSamples=Math.Max(result.underrunSamples,queue.UnderrunSamples);result.lateSamples=Math.Max(result.lateSamples,queue.LateSamples);
        result.callbacks=Math.Max(result.callbacks,queue.CallbackCount);
    }
    void StopVoice() {if(speaking) {voiceState.Invoke(network,new object[]{false,network.microphone.Clock()});speaking=false;}}
    IEnumerator Wait(Func<bool> ready,string operation,double timeout)
    {
        double deadline=Time.realtimeSinceStartupAsDouble+timeout;
        while(!ready() && errors.Count==0)
        {if(Time.realtimeSinceStartupAsDouble>deadline) {errors.Add("Timed out: "+operation);yield break;}yield return null;}
    }
    void Log(string message,string stack,LogType type) {if(type==LogType.Error || type==LogType.Exception || type==LogType.Assert)errors.Add(message+"\n"+stack);}
    void Finish()
    {
        result.status=errors.Count==0?"passed":"failed";result.error=string.Join("\n",errors);
        Directory.CreateDirectory(Path.GetDirectoryName(output));File.WriteAllText(output,JsonUtility.ToJson(result,true));
        Application.logMessageReceived-=Log;Application.Quit(errors.Count==0?0:1);
    }
}
#endif
