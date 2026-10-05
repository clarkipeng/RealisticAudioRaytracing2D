using System;
using System.Collections;
using System.IO;
using UnityEngine;
using UnityEngine.EventSystems;

// Scene interaction only. Capture, replay and audio stay owned by RayTraceManager.
[DefaultExecutionOrder(-20)]
public sealed class StationDemo : MonoBehaviour
{
    public RayTraceManager manager;
    public Transform train, carriage, door, player, boardingPlatform;
    public StationLoop route;
    public RaytracedAudioSource horn, roomSignal;
    public Vector3 doorOpenOffset=new Vector3(2.2f,0,0);
    [Min(0)] public float trainSpeed=12, startDistance=22;
    [Min(.1f)] public float trainAcceleration=3, carriageSpacing=7.2f, doorSpeed=1.8f, stationDwell=10;
    public float CurrentTrainSpeed {get;private set;}
    public double TrainDistance => route!=null ? trainDistance%route.Length:trainDistance;
    public bool AtStation {get;private set;}=true;
    public float DwellRemaining {get;private set;}
    [Tooltip("Also save full poses, geometry, source PCM and monitored audio for verification.")]
    public bool retainFullReference;
    public bool doorOpen;
    public bool Busy {get;private set;}
    Vector3 doorStart;
    double trainDistance,nextStopDistance;
    PlayerController walker;
    MouseLook look;
    StationRecording recording;
    StationRecording.Playback playback;
    static string pendingReplay;
    int nextSound;
    StationRecording.Input pending;
    bool ready;
    public long Tick {get;private set;}
    public bool ReadPlayerInput {get;set;}=true;
    public PlayerController Walker => walker;
    public MouseLook Look => look;
    public bool IsInputReplay => playback!=null;
    public bool InputReplayFinished => playback!=null && Tick==playback.Metadata.ticks;

    // Called once on a fresh scene, both for live capture and replay.
    public void InitializeSimulation()
    {
        ConfigureControls(); Tick=0; pending=default;
        trainDistance=nextStopDistance=startDistance; AtStation=true;
        DwellRemaining=stationDwell; CurrentTrainSpeed=0; PlaceTrain(); doorStart=door.localPosition;
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
        manager.RecordingStarted+=BeginInputRecording;
        manager.RecordingObserved+=ObserveInputRecording;
        manager.RecordingStopped+=EndInputRecording;
        Application.runInBackground=true;
        SetWalking(false);
        yield return null; // Let the audio components finish Start before opening the session.
        string replay=pendingReplay; pendingReplay=null;
        ready=Try(()=> {
            if(replay!=null)
            {
                playback=new StationRecording.Playback(replay); playback.ValidateScene(this);
                manager.BeginInputPlayback();
                manager.takePath=replay; ReadPlayerInput=false;
            }
            else
            {
                string root=Application.isEditor ? Path.GetFullPath(Path.Combine(Application.dataPath,"..","Recordings")) : Path.Combine(Application.persistentDataPath,"Recordings");
                string path=Path.Combine(root,"station-"+DateTime.UtcNow.ToString("yyyyMMdd-HHmmss-fff"));
                manager.BeginTakeRecording(path,retainFullReference); StartSceneSounds();
            }
        });
    }

    void Update()
    {
        if(Input.GetKeyDown(KeyCode.Escape)) SetWalking(false);
        if(Input.GetMouseButtonDown(1) && !manager.IsReplayingTake && !IsInputReplay) SetWalking(true);
        if(!ready || manager.IsReplayingTake || IsInputReplay) return;
        if(ReadPlayerInput)
        {
            bool accept=Cursor.lockState==CursorLockMode.Locked;
            SubmitInput(accept ? new Vector2(Input.GetAxis("Horizontal"),Input.GetAxis("Vertical")):Vector2.zero,
                accept ? look.ReadDelta():Vector2.zero,accept && Input.GetButtonDown("Jump"),accept && Input.GetKeyDown(KeyCode.E));
        }
        if(Input.GetKeyDown(KeyCode.R)) SaveRecording();
    }
    public void SubmitInput(Vector2 movement,Vector2 lookDegrees,bool jump=false,bool interact=false)
    {
        pending.movement=movement; pending.look+=lookDegrees; pending.jump|=jump; pending.interact|=interact;
    }
    void FixedUpdate()
    {
        if(!ready || manager.IsReplayingTake) return;
        if(playback!=null) return; // Recorded observations drive replay on the audio clock.
        var input=pending; pending.look=Vector2.zero; pending.jump=pending.interact=false;
        Step(input,Time.fixedDeltaTime);
    }
    public void Step(StationRecording.Input input,float seconds)
    {
        if(walker==null) ConfigureControls();
        if(recording!=null) RecordSafely(()=>recording.BeforeTick(Tick,input,seconds));
        if(input.interact && NearDoor) doorOpen=!doorOpen;
        double previousDistance=trainDistance;
        AdvanceTrain(seconds);
        var previousDoor=door.localPosition;
        door.localPosition=Vector3.MoveTowards(previousDoor,doorStart+(doorOpen?doorOpenOffset:Vector3.zero),doorSpeed*seconds);
        if(trainDistance!=previousDistance || door.localPosition!=previousDoor) Physics.SyncTransforms();
        look.Simulate(input.look); walker.Simulate(input.movement,input.jump,seconds); Tick++;
    }

    void BeginInputRecording(string directory) => RecordSafely(()=>recording=new StationRecording(this,directory));
    void ObserveInputRecording(long sample) => RecordSafely(()=>recording?.Observe(sample,Tick));
    void EndInputRecording(string directory,bool complete)
    {
        RecordSafely(()=>recording?.Complete(complete)); recording?.Dispose(); recording=null;
    }
    void RecordSafely(Action action)
    {
        try { action(); }
        catch(Exception e) { recording?.Dispose(); recording=null; Debug.LogException(e,this); }
    }
    void OnDestroy()
    {
        recording?.Dispose(); playback?.Dispose();
        if(manager!=null) { manager.RecordingStarted-=BeginInputRecording; manager.RecordingObserved-=ObserveInputRecording; manager.RecordingStopped-=EndInputRecording; }
        if(walker!=null) walker.externalClock=false;
        if(look!=null) look.externalClock=false;
    }
    void OnApplicationQuit()
    {
        if(manager!=null && manager.IsRecordingTake) Try(manager.EndTakeRecording);
    }

    void AdvanceTrain(float seconds)
    {
        if(AtStation)
        {
            float wait=Mathf.Min(seconds,DwellRemaining); DwellRemaining-=wait; seconds-=wait;
            if(DwellRemaining>0 || seconds<=0) return;
            AtStation=false; nextStopDistance=trainDistance+route.Length;
        }
        float speed=CurrentTrainSpeed;
        double remaining=nextStopDistance-trainDistance;
        double travel=StationLoop.AdvanceToStop(ref speed,remaining,trainSpeed,trainAcceleration,seconds);
        trainDistance+=travel; CurrentTrainSpeed=speed;
        if(remaining-travel<=1e-6) {trainDistance=nextStopDistance; AtStation=true; DwellRemaining=stationDwell;}
        if(travel>0) PlaceTrain();
    }

    public void PlaceTrain()
    {
        if(route==null) return;
        route.Place(train,trainDistance); if(carriage!=null) route.Place(carriage,trainDistance-carriageSpacing);
    }
    public void SetWalking(bool value)
    {
        Cursor.lockState=value?CursorLockMode.Locked:CursorLockMode.None;
        Cursor.visible=!value;
        if(value && EventSystem.current!=null) EventSystem.current.SetSelectedGameObject(null);
    }
    bool NearDoor => Vector3.Distance(player.position+Vector3.up,
        door.parent!=null ? door.parent.TransformPoint(doorStart) : doorStart)<=2.2f;
    void StartSceneSounds()
    {
        horn?.Play();
        roomSignal?.Play();
    }
    void StartLiveAudio()
    {
        manager.BeginLiveSimulation();
        StartSceneSounds();
    }
    public void SaveRecording()
    {
        if(Busy || !manager.IsRecordingTake || manager.IsReplayingTake || IsInputReplay) return;
        StartCoroutine(AfterStop(StartLiveAudio));
    }
    // Reloading the authored scene gives replay the same initialization as recording.
    public void ReplayInputs(string path)
    {
        if(Busy || manager.IsRecordingTake) throw new InvalidOperationException("Save the run before replaying.");
        using(var input=new StationRecording.Playback(path)) input.ValidateScene(this);
        StartCoroutine(AfterStop(()=> {
            pendingReplay=Path.GetFullPath(path);
#if UNITY_EDITOR
            UnityEditor.SceneManagement.EditorSceneManager.LoadSceneInPlayMode(gameObject.scene.path,
                new UnityEngine.SceneManagement.LoadSceneParameters(UnityEngine.SceneManagement.LoadSceneMode.Single));
#else
            UnityEngine.SceneManagement.SceneManager.LoadScene(gameObject.scene.path);
#endif
        }));
    }
    void LateUpdate()
    {
        if(!ready || playback==null || !manager.IsLiveSimulation) return;
        if(!Try(()=> {
            long now=manager.LiveSample;
            while(nextSound<playback.Events.Length && playback.Events[nextSound].sample<=now)
                manager.AppendReplaySound(playback.Events[nextSound++]);
            while(playback.NextObservation(now,out long sample,out long tick))
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
