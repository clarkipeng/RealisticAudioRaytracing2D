using System;
using System.IO;
using System.Linq;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;

public static partial class ChurchPresentationCapture
{
    public static void VerifySequence()
    {
        if(!Application.isBatchMode||Application.isPlaying)throw new InvalidOperationException("Offscreen verification only.");
        string directory=Argument("-sequenceOutput");
        string reconstructed=SceneReplay.Reconstruct(Path.Combine(directory,"take"),Path.Combine(directory,"reconstructed"));
        var report=JsonUtility.FromJson<SceneReplay.Report>(File.ReadAllText(Path.Combine(reconstructed,"verification.json")));
        if(!report.byteIdentical)throw new InvalidDataException("Sequence replay differs from the recorded poses/events; inspect verification.json.");
        Debug.Log("CHURCH_SEQUENCE_REPLAY byte-identical: "+reconstructed);
    }
    public static void CaptureSequence()
    {
        if(!Application.isBatchMode||Application.isPlaying)throw new InvalidOperationException("Offscreen batch capture only.");
        string directory=Argument("-sequenceOutput");if(string.IsNullOrEmpty(directory))throw new ArgumentException("Pass -sequenceOutput.");
        if(Directory.Exists(directory))throw new IOException("Capture output already exists.");Directory.CreateDirectory(directory);
        bool storyboard=Argument("-sequenceStoryboard")=="1";
        bool takeOnly=Argument("-sequenceTakeOnly")=="1";
        bool firstPerson=Argument("-sequenceFirstPerson")=="1";
        int firstFrame=int.TryParse(Argument("-sequenceFirstFrame"),out var first)?first:0;
        int endFrame=int.TryParse(Argument("-sequenceEndFrame"),out var end)?end:2130;
        double workFraction=double.TryParse(Argument("-sequenceWorkFraction"),System.Globalization.NumberStyles.Float,
            System.Globalization.CultureInfo.InvariantCulture,out var fraction)?fraction:.2;
        if(firstFrame<0||endFrame<=firstFrame||endFrame>2130||!double.IsFinite(workFraction)||workFraction<=0||workFraction>1)
            throw new ArgumentException("Invalid frame range or work fraction; use [0,2130) and (0,1].");
        bool weatherRoute=Argument("-sequenceRoute")=="weather";
        double tail=double.TryParse(Argument("-sequenceTail"),out var tailValue)?tailValue:11;
        if(!double.IsFinite(tail)||tail<0)throw new ArgumentException("Invalid sequence tail.");
        string scene=Argument("-sequenceScene");EditorSceneManager.OpenScene(string.IsNullOrEmpty(scene)?ScenePath:scene);
        var oldMode=Physics.simulationMode;float oldStep=Time.fixedDeltaTime;SceneRecording recording=null;
        int oldAa=QualitySettings.antiAliasing;var oldShadows=QualitySettings.shadows;
        var oldResolution=QualitySettings.shadowResolution;float oldDistance=QualitySettings.shadowDistance;
        Camera camera=null;RenderTexture target=null;Texture2D texture=null;
        try
        {
            Physics.simulationMode=SimulationMode.Script;Time.fixedDeltaTime=.02f;
            float step=Time.fixedDeltaTime; // Unity may quantize the requested value; record/use its actual clock.
            QualitySettings.antiAliasing=4;QualitySettings.shadows=ShadowQuality.All;QualitySettings.shadowResolution=ShadowResolution.High;QualitySettings.shadowDistance=100;
            var session=FindObject<SceneSession>();var manager=session.manager;
            session.GetComponent<ChurchVisualPreset>().Apply(ChurchVisualPreset.Preset.PolishedLowPoly);
            session.InitializeSimulation();var actor=session.actors.OfType<SceneSequenceActor>().Single();
            var environment=FindObject<EnvironmentPresentation>();var bell=FindObject<BellPresentation>();var bird=FindObject<BirdPresentation>();
            var avatar=session.player.GetComponent<ChurchAvatarPresenter>();avatar.Initialize();avatar.animator.Rebind();avatar.animator.Update(0);avatar.SetVisible(!firstPerson);
            var head=session.player.GetComponentInChildren<Camera>(true).transform;
            camera=CreateThirdPersonCamera();camera.farClipPlane=260;
            if(firstPerson)camera.fieldOfView=70;
            int height=int.TryParse(Argument("-sequenceHeight"),out var pixels)?pixels:storyboard?720:1080;
            if(height<180||height>2160||height%18!=0)throw new ArgumentException("Use a 16:9 height divisible by 18.");
            int width=height*16/9;
            target=new RenderTexture(width,height,24,RenderTextureFormat.ARGB32){antiAliasing=4};texture=new Texture2D(width,height,TextureFormat.RGB24,false);camera.targetTexture=target;
            environment.Evaluate(0,camera);
            if(firstPerson)CaptureSequenceMap(directory,session);
            if(!storyboard)
            {
                manager.monitorLive=false;
                manager.RecordingStarted+=path=>recording=new SceneRecording(session,path);
                manager.RecordingObserved+=sample=>recording.Observe(sample,session.Tick);
                manager.RecordingFinalizing+=_=>recording.Complete(true);
                manager.BeginTakeRecording(Path.Combine(directory,"take"));actor.BeginAudio(manager.sampleRate,manager.RecordSourceEventAt);
            }
            double duration=actor.Definition.duration+tail;int frames=(int)(duration*30);float min=1,max=0;
            endFrame=Math.Min(endFrame,frames);
            float[] checkpoints={0,3,6,8.03f,12,15.5f,19.5f,25,30,36,41,45.03f,45.17f,49,55,60,68};
            int checkpoint=0;string frameDirectory=Path.Combine(directory,"frames");Directory.CreateDirectory(frameDirectory);
            double workSeconds=0,idleSeconds=0;int renderedFrames=0;
            using(var route=new StreamWriter(Path.Combine(directory,"route.csv")))
            using(var view=new StreamWriter(Path.Combine(directory,"view.csv")))
            {
                route.WriteLine("tick,seconds,x,y,z,grounded");
                view.WriteLine("frame,seconds,x,y,z,forwardX,forwardY,forwardZ,headDistance");
                for(int frame=0;frame<=frames;frame++)
                {
                    var work=System.Diagnostics.Stopwatch.StartNew();
                    double time=frame/30.0;
                    while((session.Tick+1)*(double)step<=time+1e-7)
                    {
                        var input=weatherRoute?WeatherInput(session,session.Tick*(double)step):WalkthroughInput(session,session.Tick*(double)step);
                        recording?.BeforeTick(session.Tick,input,step);session.Step(input,step);Physics.Simulate(step);avatar.ManualUpdate(step);
                        if(manager.IsRecordingTake)
                        {
                            bool final=actor.Seconds>=actor.Definition.duration;manager.CaptureSimulationObservation(final);
                            if(final)recording=null;
                        }
                        var p=session.player.position;route.WriteLine(FormattableString.Invariant($"{session.Tick},{session.Tick*(double)step:R},{p.x:R},{p.y:R},{p.z:R},{(session.Walker.Ground!=null?1:0)}"));
                    }
                    if(frame==frames)break;
                    if(takeOnly)continue;
                    if(storyboard&&(checkpoint>=checkpoints.Length||time<checkpoints[checkpoint]))continue;
                    if(firstPerson)camera.transform.SetPositionAndRotation(head.position,head.rotation);
                    else PositionSequenceCamera(camera,session.player,actor,time);
                    var eye=camera.transform.position;var forward=camera.transform.forward;
                    view.WriteLine(FormattableString.Invariant($"{frame},{time:R},{eye.x:R},{eye.y:R},{eye.z:R},{forward.x:R},{forward.y:R},{forward.z:R},{Vector3.Distance(eye,head.position):R}"));
                    if(frame<firstFrame||frame>=endFrame)continue;
                    environment.Evaluate(time,camera);bell.Evaluate(time);bird.Evaluate(time);
                    CaptureFrame(camera,target,texture,Path.Combine(frameDirectory,$"frame_{frame:0000}.png"),ref min,ref max);
                    renderedFrames++;workSeconds+=work.Elapsed.TotalSeconds;
                    // Output remains 30 fps on the simulation clock. Yield wall time
                    // between renders so an offline export cannot monopolize the GPU.
                    int idle=(int)Math.Min(int.MaxValue,Math.Ceiling(work.Elapsed.TotalMilliseconds*(1/workFraction-1)));
                    var rest=System.Diagnostics.Stopwatch.StartNew();System.Threading.Thread.Sleep(idle);idleSeconds+=rest.Elapsed.TotalSeconds;
                    if(storyboard)checkpoint++;
                    if(frame%300==0)Debug.Log("Church sequence frame "+frame+" / "+frames);
                }
            }
            if(manager.IsRecordingTake)manager.CaptureSimulationObservation(true);
            var settings=JsonUtility.FromJson<TakeRender.Settings>(JsonUtility.ToJson(manager.directSettings));
            settings.maximumResponsesPerBatch=1;settings.tailSamples=(int)Math.Ceiling(tail*manager.sampleRate);
            File.WriteAllText(Path.Combine(directory,"render-job.json"),"{\"take\":\"take\",\"output\":\"audio\",\"shader\":\""+AssetDatabase.GetAssetPath(manager.directShader)+"\",\"settings\":"+JsonUtility.ToJson(settings)+"}");
            File.WriteAllText(Path.Combine(directory,"summary.json"),JsonUtility.ToJson(new SequenceSummary{frames=takeOnly?0:storyboard?checkpoint:frames,
                width=width,height=height,seconds=duration,finalPosition=session.player.position,maxBounces=settings.maxBounces,sourceCount=manager.takeSources.Count,
                renderedFrames=renderedFrames,firstFrame=firstFrame,endFrame=endFrame,workFraction=workFraction,renderWorkSeconds=workSeconds,renderIdleSeconds=idleSeconds,
                route=weatherRoute?"weather":"walkthrough",listener=firstPerson?"Player head; picture camera uses the same head pose":"Player head; independent of picture camera",minLuma=min,maxLuma=max},true));
            Debug.Log("CHURCH_SEQUENCE_CAPTURE "+directory);
        }
        finally
        {
            recording?.Dispose();Physics.simulationMode=oldMode;Time.fixedDeltaTime=oldStep;
            QualitySettings.antiAliasing=oldAa;QualitySettings.shadows=oldShadows;
            QualitySettings.shadowResolution=oldResolution;QualitySettings.shadowDistance=oldDistance;
            RenderTexture.active=null;if(camera!=null)UnityEngine.Object.DestroyImmediate(camera.gameObject);
            if(target!=null)UnityEngine.Object.DestroyImmediate(target);if(texture!=null)UnityEngine.Object.DestroyImmediate(texture);
        }
    }
    // A static north-up scene image; the movie compositor adds recorded source
    // activity, moving poses and masked emission areas without another scheduler.
    static void CaptureSequenceMap(string directory,SceneSession session)
    {
        var bounds=new Bounds(session.player.position,Vector3.zero);
        foreach(var source in session.manager.takeSources)
        {
            if(source.streamed)continue;
            bounds.Encapsulate(source.transform.position);
            var plane=source.GetComponent<AudioSourcePlane>();
            if(plane==null)continue;
            foreach(float x in new[]{-.5f,.5f})foreach(float z in new[]{-.5f,.5f})
                bounds.Encapsulate(plane.transform.TransformPoint(new Vector3(x*plane.size.x,0,z*plane.size.y)));
        }
        float span=Mathf.Max(bounds.size.x,bounds.size.z)+20;
        var map=CreateThirdPersonCamera();var target=new RenderTexture(768,768,24);var texture=new Texture2D(768,768,TextureFormat.RGB24,false);
        bool fog=RenderSettings.fog;
        try
        {
            RenderSettings.fog=false;map.orthographic=true;map.orthographicSize=span*.5f;map.aspect=1;
            map.transform.SetPositionAndRotation(new Vector3(bounds.center.x,150,bounds.center.z),Quaternion.Euler(90,0,0));
            map.farClipPlane=300;map.targetTexture=target;float min=1,max=0;
            CaptureFrame(map,target,texture,Path.Combine(directory,"map.png"),ref min,ref max);
            File.WriteAllText(Path.Combine(directory,"map.json"),JsonUtility.ToJson(new SequenceMap{centerX=bounds.center.x,centerZ=bounds.center.z,span=span},true));
        }
        finally
        {
            RenderSettings.fog=fog;RenderTexture.active=null;UnityEngine.Object.DestroyImmediate(map.gameObject);
            UnityEngine.Object.DestroyImmediate(target);UnityEngine.Object.DestroyImmediate(texture);
        }
    }
    [Serializable] sealed class SequenceMap {public float centerX,centerZ,span;}
    [Serializable] sealed class SequenceSummary
    {
        public int frames,width,height,maxBounces,sourceCount,renderedFrames,firstFrame,endFrame;
        public double seconds,workFraction,renderWorkSeconds,renderIdleSeconds;
        public Vector3 finalPosition;public string listener,route;public float minLuma,maxLuma;
    }
    static SceneRecording.Input WalkthroughInput(SceneSession session,double time)
    {
        // Waypoint steering supplies ordinary controller input; the character is never teleported.
        Vector3 target;
        if(time<14)target=new Vector3(26,0,-34);
        else if(time<24)target=new Vector3(23,0,-29);
        else if(time<34)target=new Vector3(20,0,-42);
        else if(time<44)target=new Vector3(2,0,-42);
        else if(time<52)target=new Vector3(0,0,-37.4f);
        else target=new Vector3(0,0,-29);
        var delta=target-session.player.position;delta.y=0;
        float speed=time<14?.75f:time<24?.65f:time<44?1.8f:time<52?.8f:1.6f;
        var world=delta.normalized*Mathf.Min(speed,delta.magnitude/Time.fixedDeltaTime);
        var local=session.player.InverseTransformDirection(world)/session.Walker.speed;
        float yaw=delta.sqrMagnitude>.04f?Mathf.Atan2(delta.x,delta.z)*Mathf.Rad2Deg:session.player.eulerAngles.y;
        float turn=Mathf.Clamp(Mathf.DeltaAngle(session.player.eulerAngles.y,yaw),-1.5f,1.5f);
        return new SceneRecording.Input{movement=time>=60?Vector2.zero:new Vector2(local.x,local.z),look=new Vector2(time>=60?0:turn,0),interact=Math.Abs(time-52)<.005};
    }
    static SceneRecording.Input WeatherInput(SceneSession session,double time)
    {
        Vector3 target=time<6?new Vector3(2,0,-40):time<13?new Vector3(0,0,-32.5f):
            time<22?new Vector3(0,0,-29):time<26?new Vector3(0,0,-32.5f):new Vector3(2,0,-40);
        var delta=target-session.player.position;delta.y=0;
        var velocity=delta.normalized*Mathf.Min(1.6f,delta.magnitude/Time.fixedDeltaTime);
        var local=session.player.InverseTransformDirection(velocity)/session.Walker.speed;
        float yaw=time<6||time>=31?60:delta.sqrMagnitude>.04f?Mathf.Atan2(delta.x,delta.z)*Mathf.Rad2Deg:session.player.eulerAngles.y;
        float turn=Mathf.Clamp(Mathf.DeltaAngle(session.player.eulerAngles.y,yaw),-2,2);
        return new SceneRecording.Input {movement=time>=36?Vector2.zero:new Vector2(local.x,local.z),
            look=new Vector2(time>=36?0:turn,0),interact=Math.Abs(time-9)<.005||Math.Abs(time-13)<.005||Math.Abs(time-25)<.005};
    }
    static void PositionSequenceCamera(Camera camera,Transform player,SceneSequenceActor actor,double seconds)
    {
        float t=(float)seconds;Vector3 position,look;float fov=55;
        if(t<5)
        {position=Vector3.Lerp(new Vector3(47,14,-64),new Vector3(38,8,-53),t/5);look=Vector3.Lerp(new Vector3(6,20,-8),player.position+Vector3.up*5,t/5);fov=63;}
        else if(t<13)
        {position=new Vector3(3.2f-(t-5)*.08f,52.6f,-.5f);look=new Vector3(0,52.5f,3.92f);fov=55;}
        else if(t<15)
        {position=player.position+new Vector3(4,2,-5);look=player.position+Vector3.up*1.8f;}
        else if(t<22.2f)
        {var bird=actor.Target("bird").position;position=bird+new Vector3(-2,.6f,-3);look=bird;fov=38;}
        else if(t<28)
        {position=new Vector3(31,9,-39);look=Vector3.Lerp(actor.Target("bird").position,player.position+Vector3.up*3,Mathf.Clamp01((t-24)/4));fov=54;}
        else if(t<34)
        {position=player.position+new Vector3(8,3,-5);look=player.position+new Vector3(-4,9,3);fov=64;}
        else if(t<44)
        {position=player.position+new Vector3(4,2.5f,-5);look=player.position+new Vector3(-3,2.4f,1);}
        else if(t<48)
        {position=player.position+new Vector3(-4,3,-7);look=new Vector3(39,35,-56);fov=64;}
        else
        {position=player.position+new Vector3(.35f,2.3f,-4);look=player.position+new Vector3(0,1.8f,4);fov=60;}
        camera.fieldOfView=fov;camera.transform.SetPositionAndRotation(position,Quaternion.LookRotation(look-position));
    }
}
