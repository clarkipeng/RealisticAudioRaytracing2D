using System;
using System.Collections;
using System.IO;
using UnityEngine;
using UnityEngine.EventSystems;

// One initialized fixed-step session for live input and isolated reconstruction.
[DefaultExecutionOrder(-20)]
public sealed class SceneSession : MonoBehaviour
{
    public RayTraceManager manager;
    public Transform player;
    public MultiplayerSession multiplayer;
    public SceneActor[] actors=Array.Empty<SceneActor>();
    public RaytracedAudioSource[] startSounds=Array.Empty<RaytracedAudioSource>();
    [Tooltip("Also save full poses, geometry, source PCM and monitored audio for verification.")]
    public bool retainFullReference;
    public bool Busy {get;private set;}
    PlayerController walker;
    MouseLook look;
    SceneRecording recording;
    SceneRecording.Playback playback;
    static string pendingReplay;
    int nextSound;
    SceneRecording.Input pending;
    bool ready;
    public long Tick {get;private set;}
    public bool ReadPlayerInput {get;set;}=true;
    public Action<long> SampleInput;
    public PlayerController Walker => walker;
    public MouseLook Look => look;
    public bool IsInputReplay => playback!=null || manager.IsReplayingTake;
    public bool InputReplayFinished => playback!=null ? Tick==playback.Metadata.ticks : manager.IsReplayingTake && !manager.IsTakeReplayPlaying;

    // Called once on a fresh scene, both for live capture and replay.
    public void InitializeSimulation()
    {
        ConfigureControls(); Tick=0; pending=default;
        var ids=new System.Collections.Generic.HashSet<string>(StringComparer.Ordinal);
        foreach(var actor in actors)
        {
            if(actor==null || !actor.isActiveAndEnabled || actor.gameObject.scene!=gameObject.scene ||
                string.IsNullOrWhiteSpace(actor.id) || !ids.Add(actor.id))
                throw new InvalidOperationException("Scene actors need unique persisted IDs and active bindings in this scene.");
            if(!(actor is StationDemo) && !(actor is SlidingDoor))
                throw new NotSupportedException("Add an isolated replay adapter before recording actor "+actor.GetType().Name);
            actor.Initialize();
        }
        walker.Initialize(); look.Initialize(); Physics.SyncTransforms();
    }
    void ConfigureControls()
    {
        walker=player.GetComponent<PlayerController>(); look=player.GetComponentInChildren<MouseLook>();
        walker.externalClock=look.externalClock=true;
    }

    IEnumerator Start()
    {
        InitializeSimulation();
        string replay=pendingReplay; pendingReplay=null;
        if(multiplayer!=null)
        {
            multiplayer.Prepare(replay==null?null:AcousticTake.ReadManifest(replay).session);
            if(replay==null) {yield return multiplayer.Connect();if(!multiplayer.Ready)yield break;}
        }
        manager.RecordingStarted+=BeginInputRecording;
        manager.RecordingObserved+=ObserveInputRecording;
        manager.RecordingFinalizing+=FinalizeInputRecording;
        manager.RecordingStopped+=EndInputRecording;
        manager.simulation=this;
        Application.runInBackground=true;
        // Presentation must not hold the main thread that submits acoustic work.
        QualitySettings.vSyncCount=0;Application.targetFrameRate=60;
        SetWalking(replay==null);
        yield return null; // Let the audio components finish Start before opening the session.
        ready=Try(()=> {
            if(replay!=null)
            {
                if(File.Exists(Path.Combine(replay,"states.bin")) && AcousticTake.ReadManifest(replay).session?.mode=="lan-client-presented-v1")
                    manager.BeginTakeReplay(replay,true,true);
                else
                {
                    playback=new SceneRecording.Playback(replay); playback.ValidateScene(this);
                    manager.BeginInputPlayback(replay);
                }
                manager.takePath=replay; ReadPlayerInput=false;
            }
            else
            {
                string path=Path.Combine(RecordingsDirectory,gameObject.scene.name.ToLowerInvariant()+"-"+DateTime.UtcNow.ToString("yyyyMMdd-HHmmss-fff")+"-"+Guid.NewGuid().ToString("N"));
                manager.SessionInfo=multiplayer?.Description();
                manager.BeginTakeRecording(path,retainFullReference || (multiplayer!=null && multiplayer.Connected),retainFullReference); multiplayer?.CaptureStarted();StartSceneSounds();
            }
        });
    }

    void Update()
    {
        if(Input.GetKeyDown(KeyCode.Escape)) SetWalking(false);
        if(!ready || manager.IsReplayingTake || IsInputReplay) return;
        if(ReadPlayerInput)
        {
            bool accept=Cursor.lockState==CursorLockMode.Locked;
            SubmitInput(accept ? new Vector2(Input.GetAxis("Horizontal"),Input.GetAxis("Vertical")):Vector2.zero,
                accept ? look.ReadDelta():Vector2.zero,accept && Input.GetButtonDown("Jump"),accept && Input.GetKeyDown(KeyCode.E));
        }
        manager.AdvanceSimulation();
        if(Cursor.lockState==CursorLockMode.Locked && Input.GetKeyDown(KeyCode.R)) SaveRecording();
    }
    public void SubmitInput(Vector2 movement,Vector2 lookDegrees,bool jump=false,bool interact=false)
    {
        pending.movement=movement; pending.look+=lookDegrees; pending.jump|=jump; pending.interact|=interact;
    }
    internal long AdvanceTo(long sample,long origin,int rate)
    {
        long At(long tick)=>checked((long)Math.Round((tick-origin)*(double)Time.fixedDeltaTime*rate));
        // Pose knots and the simulation that produced them share the sample clock.
        // Catch up fixed steps without attaching a frame's DSP timestamp to an older pose.
        while(At(Tick+1)<=sample)
        {
            SampleInput?.Invoke(Tick);
            var input=pending;pending.look=Vector2.zero;pending.jump=pending.interact=false;
            Step(input,Time.fixedDeltaTime);
        }
        return At(Tick);
    }
    public void Step(SceneRecording.Input input,float seconds)
    {
        if(walker==null) ConfigureControls();
        if(recording!=null)
        {
            try {recording.BeforeTick(Tick,input,seconds);}
            catch(Exception e) {manager.AbortTakeRecording(e.Message); Debug.LogException(e,this);}
        }
        bool moved=false;
        if(multiplayer==null || multiplayer.ControlsWorld)
            foreach(var actor in actors)moved|=actor.Step(input,player,seconds);
        else if(input.interact)multiplayer.RequestInteraction();
        if(moved) Physics.SyncTransforms();
        look.Simulate(input.look); walker.Simulate(input.movement,input.jump,seconds); Tick++;
    }

    void BeginInputRecording(string directory)
    {
        if(multiplayer==null || !multiplayer.Connected)recording=new SceneRecording(this,directory);
    }
    void ObserveInputRecording(long sample) => recording?.Observe(sample,Tick);
    void FinalizeInputRecording(string directory)
    {
        if(recording!=null)recording.Complete(true);
        else if(multiplayer==null || manager.SessionInfo?.mode!="lan-client-presented-v1")throw new InvalidOperationException("The required scene input recording is missing.");
    }
    void EndInputRecording(string directory,bool complete) {recording?.Dispose(); recording=null;}
    void OnDestroy()
    {
        recording?.Dispose(); playback?.Dispose();
        if(manager!=null) { manager.RecordingStarted-=BeginInputRecording; manager.RecordingObserved-=ObserveInputRecording; manager.RecordingFinalizing-=FinalizeInputRecording; manager.RecordingStopped-=EndInputRecording; }
        if(walker!=null) walker.externalClock=false;
        if(look!=null) look.externalClock=false;
    }
    void OnApplicationQuit()
    {
        if(manager!=null && manager.IsRecordingTake) Try(manager.EndTakeRecording);
    }

    public void SetWalking(bool value)
    {
        Cursor.lockState=value?CursorLockMode.Locked:CursorLockMode.None;
        Cursor.visible=!value;
        if(value && EventSystem.current!=null) EventSystem.current.SetSelectedGameObject(null);
    }
    void StartSceneSounds()
    {
        foreach(var source in startSounds)
        {
            long offset=source.loop?(long)((multiplayer?.SceneSoundOffset??0)*manager.sampleRate)%Math.Max(1,(long)(source.clip.length*manager.sampleRate)):0;
            manager.RecordSourceEvent(source.transform,AcousticTake.VoiceAction.Play,source.gain,source.loop,offset);
        }
    }
    void StartLiveAudio()
    {
        manager.BeginLiveSimulation();
        multiplayer?.CaptureStarted();
        StartSceneSounds();
    }
    public static string RecordingsDirectory=>Application.isEditor?Path.GetFullPath(Path.Combine(Application.dataPath,"..","Recordings")):Path.Combine(Application.persistentDataPath,"Recordings");
    public void SaveRecording()
    {
        if(Busy || !manager.IsRecordingTake || manager.IsReplayingTake || IsInputReplay) return;
        StartCoroutine(AfterStop(StartLiveAudio));
    }
    // Reloading the authored scene gives replay the same initialization as recording.
    public void ReplayInputs(string path)
    {
        if(Busy || manager.IsRecordingTake) throw new InvalidOperationException("Save the run before replaying.");
        var metadata=AcousticTake.ReadManifest(path);
        if(metadata.session?.mode=="lan-client-presented-v1")
        {
            if(metadata.session.scene!=gameObject.scene.path)throw new InvalidDataException("Open the take's recorded scene first: "+metadata.session.scene);
            using(var take=AcousticTake.Open(path)) {}
        }
        else using(var input=new SceneRecording.Playback(path))input.ValidateScene(this);
        StartCoroutine(AfterStop(()=> {
            multiplayer?.Shutdown();pendingReplay=Path.GetFullPath(path);
            ReloadScene();
        }));
    }
    public void RestartScene()
    {
        if(Busy)return;
        StartCoroutine(AfterStop(()=> {multiplayer?.Shutdown();ReloadScene();}));
    }
    void ReloadScene()
    {
#if UNITY_EDITOR
            UnityEditor.SceneManagement.EditorSceneManager.LoadSceneInPlayMode(gameObject.scene.path,
                new UnityEngine.SceneManagement.LoadSceneParameters(UnityEngine.SceneManagement.LoadSceneMode.Single));
#else
            UnityEngine.SceneManagement.SceneManager.LoadScene(gameObject.scene.path);
#endif
    }
    void LateUpdate()
    {
        // UI processes pointer input in Update before we decide whether the scene was clicked.
        if((Input.GetMouseButtonDown(0) || Input.GetMouseButtonDown(1)) && !manager.IsReplayingTake && !IsInputReplay &&
            (EventSystem.current==null || !EventSystem.current.IsPointerOverGameObject())) SetWalking(true);
        if(!ready || playback==null || !manager.IsLiveSimulation) return;
        if(!Try(()=> {
            long now=manager.LiveSample;
            while(nextSound<playback.Events.Length && playback.Events[nextSound].sample<=now)
                manager.AppendReplaySound(playback.Events[nextSound++]);
            while(manager.CanReplayObservation && playback.NextObservation(now,out long sample,out long tick))
            {
                while(Tick<tick) Step(playback.Read(Tick),playback.Metadata.stepSeconds);
                manager.ReplayObservation(sample,InputReplayFinished && sample==playback.TakeMetadata.durationSamples);
            }
        })) { ready=false; manager.StopLiveSimulation(); }
    }
    IEnumerator AfterStop(Action action)
    {
        Busy=true;
        bool stopped=Try(()=> {
            multiplayer?.microphone.StopCapture();
            if(manager.IsRecordingTake) manager.EndTakeRecording();
            else if(manager.IsLiveSimulation) manager.StopLiveSimulation();
        });
        if(stopped)
        {
            while(manager.Mode==RayTraceManager.SessionMode.Draining) yield return null;
            Try(action);
        }
        Busy=false;
    }
    bool Try(Action action)
    {
        try {action(); return true;}
        catch(Exception e) {Debug.LogException(e,this); return false;}
    }

    void OnDrawGizmosSelected()
    {
        if(Camera.current!=null && Camera.current.cameraType!=CameraType.SceneView) return;
        Gizmos.color=new Color(.1f,.9f,.85f,.65f);
        if(manager==null) return;
        foreach(var binding in manager.takeObjects)
        {
            if(!binding.geometry || binding.transform==null) continue;
            var box=binding.transform.GetComponent<BoxCollider>();
            if(box==null) continue;
            Gizmos.matrix=box.transform.localToWorldMatrix; Gizmos.DrawWireCube(box.center,box.size);
        }
        Gizmos.matrix=Matrix4x4.identity;
    }
}

// Supported actors are stepped only by the session, in the serialized array order.
public abstract class SceneActor : MonoBehaviour
{
    public string id;
    public abstract string SettingsJson {get;}
    public abstract void Initialize();
    public abstract bool SettingsUnchanged {get;}
    public abstract bool Step(SceneRecording.Input input,Transform player,float seconds);
}
