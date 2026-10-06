using System;
using System.Collections.Generic;
using System.IO;
using UnityEditor;
using UnityEngine;

// Uses the actual Unity types, scene extraction and file reader. Also callable in batch mode.
public static partial class TakeChecks
{
    static int checks;
    static string directory;

    [MenuItem("Tools/Acoustics/Run take checks")]
    public static void Run()
    {
        checks=0; directory=Path.Combine(Path.GetTempPath(),"acoustic-checks-"+Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(directory);
        try
        {
            RoundTripAndCuts(); InvalidFiles(); Motion(); Materials(); Geometry(); DirectPaths(); AudioEventsAndTail(); ManagerCapture(); ReplayRestoresScene(); CaptureMutation(); GuardSnapshots(); ColliderProxyGuard(); StreamingMatchesOffline(); StreamingMatchesOffline(null,TakeRender.DelayFilter.WindowedSinc); QueueDeadlines(); LongSampleClock(); BoundedHistory(); DelayFilters(); PlayedOutput(); ConcurrentOutputCapture(); ReferenceQuality(); BinauralChecks(); BinauralStreaming(); ComparisonInspection(); ComparisonViews(); StationLoopMotion();
            StationInputs(); BvhChecks();
            Debug.Log("ACOUSTIC_CHECKS_DIR: "+directory);
            Debug.Log("ACOUSTIC_CHECKS_PASS: "+checks+" checks");
            if (Application.isBatchMode) EditorApplication.Exit(0);
        }
        catch (Exception e)
        {
            Debug.LogException(e);
            if (Application.isBatchMode) EditorApplication.Exit(1);
            else throw;
        }
        finally { if (Array.IndexOf(Environment.GetCommandLineArgs(),"-keepTakeChecks")<0) Directory.Delete(directory,true); }
    }

    static AcousticTake.Scene Scene(int listeners=1)
    {
        var objects=new AcousticTake.ObjectDefinition[listeners+1];
        var indices=new int[listeners];
        for (int i=0; i<objects.Length; i++) objects[i]=new AcousticTake.ObjectDefinition { id="object-"+i,name="Object "+i,scale=Vector3.one };
        for (int i=0; i<listeners; i++) indices[i]=i+1;
        return new AcousticTake.Scene { dimension=AcousticTake.Dimension.Spatial3D,objects=objects,
            sources=new[] { new AcousticTake.SourceDefinition {objectIndex=0,clipIndex=0} },listeners=indices,
            clips=new[] {new AcousticTake.ClipDefinition {name="test"}},materials=new AcousticTake.MaterialDefinition[0] };
    }
    static AcousticTake.State[] States(int listeners=1)
    {
        var states=new AcousticTake.State[listeners+1];
        for (int i=0; i<states.Length; i++) states[i]=new AcousticTake.State { position=new Vector3(i,0,0),rotation=Quaternion.identity,
            flags=AcousticTake.StateFlags.Active | (i == 0 ? AcousticTake.StateFlags.Source : AcousticTake.StateFlags.Listener) };
        return states;
    }
    static AcousticTake.Writer Writer(string name, AcousticTake.Scene scene, float[] pcm=null)
    {
        return new AcousticTake.Writer(Path.Combine(directory,name),1000,scene,new AcousticTake.Primitive[0],new[] {pcm ?? new[] {1f,2f,3f}});
    }
    static void Check(bool condition,string message) { checks++; if (!condition) throw new Exception(message); }
    static void Near(float actual,float expected,string message,float tolerance=1e-5f) { Check(Mathf.Abs(actual-expected)<=tolerance,message+": "+actual+" != "+expected); }
    static void Reject(Action action,string message)
    {
        bool rejected=false; try { action(); } catch (InvalidDataException) { rejected=true; } catch (ArgumentException) { rejected=true; } catch (InvalidOperationException) { rejected=true; }
        Check(rejected,message);
    }

    static void RoundTripAndCuts()
    {
        var states=States();
        states[0].position=new Vector3(1.2345678f,-3.456789f,.00000123f);
        var exact=states[0].position;
        using (var writer=Writer("roundtrip",Scene()))
        {
            writer.Append(0,states);
            states[0].position.x+=10; writer.Append(1000,states);
            states[0].position.x+=1; writer.Append(1000,states); // coalesce
            states[0].position.x=100; states[0].flags|=AcousticTake.StateFlags.Cut; writer.Append(2000,states);
            states[0].position.x=102; states[0].flags&=~AcousticTake.StateFlags.Cut; writer.Append(3000,states);
            writer.Complete();
        }
        using (var take=AcousticTake.Open(Path.Combine(directory,"roundtrip")))
        {
            Check(take.Metadata.frameCount==4,"Same-sample observations must coalesce.");
            take.Evaluate(0,states);
            Check(states[0].position.Equals(exact),"Stored pose bits must round trip exactly.");
            take.Evaluate(500,states); Near(states[0].position.x,exact.x+5.5f,"Linear replay");
            take.Evaluate(1500,states); Near(states[0].position.x,exact.x+11,"Hold before teleport");
            Near(states[0].velocity.magnitude,0,"Held pre-teleport path has zero velocity.");
            take.Evaluate(2000,states); Near(states[0].position.x,100,"Teleport exact sample");
            take.Evaluate(2500,states); Check((states[0].flags&AcousticTake.StateFlags.Cut)==0,"Cut does not persist between knots.");
            Reject(()=>take.Evaluate(-1,states),"Reject extrapolation before take.");
        }
    }

    static void InvalidFiles()
    {
        using (var writer=Writer("incomplete",Scene())) writer.Append(0,States());
        Reject(()=>AcousticTake.Open(Path.Combine(directory,"incomplete")),"Reject incomplete capture.");
        var scene=Scene(); scene.objects[1].id=scene.objects[0].id;
        Reject(()=>Writer("duplicate",scene),"Reject duplicate IDs.");
        using (var writer=Writer("corrupt",Scene())) { writer.Append(0,States()); writer.Complete(); }
        using (var f=File.OpenWrite(Path.Combine(directory,"corrupt","states.bin"))) { f.Position=10; f.WriteByte(255); }
        Reject(()=>AcousticTake.Open(Path.Combine(directory,"corrupt")),"Reject corrupted numeric payload.");
        using (var writer=Writer("order",Scene()))
        {
            writer.Append(0,States()); writer.Append(10,States());
            Reject(()=>writer.Append(9,States()),"Reject backwards frame times.");
        }
        using (var writer=Writer("failed-overwrite",Scene()))
        {
            var state=States(); writer.Append(0,state); state[0].rotation=new Quaternion(0,0,0,0);
            Reject(()=>writer.Append(0,state),"Reject invalid same-sample overwrite.");
            Reject(writer.Complete,"Failed overwrite cannot be finalized at the old file length.");
        }
    }

    static void Motion()
    {
        var a=States()[0]; var b=a; b.position.x=2; b.rotation=Quaternion.AngleAxis(90,Vector3.up); b.MeasureMotion(a,.5);
        Near(b.velocity.x,4,"Velocity uses timestamp interval"); Near(b.angularVelocity.y,Mathf.PI,"Angular velocity in rad/s");
        var c=b; c.position.x=5; c.MeasureMotion(b,.5);
        Near(c.acceleration.x,4,"Acceleration in m/s²");
        var cut=c; cut.flags|=AcousticTake.StateFlags.Cut; cut.position.x=100; cut.MeasureMotion(c,.5);
        Check((cut.flags&AcousticTake.StateFlags.VelocityValid)==0,"Teleport must not acquire physical velocity.");
        Near(cut.velocity.magnitude,0,"Teleport clears previous motion estimate.");
    }

    static void Materials()
    {
        var material=ScriptableObject.CreateInstance<AudioMaterial>();
        try
        {
            material.energyCurve=new[] {
                new AudioMaterial.EnergyKnot {frequencyHz=0,absorption=0,scattering=.2f},
                new AudioMaterial.EnergyKnot {frequencyHz=1000,absorption=.8f,scattering=.6f},
                new AudioMaterial.EnergyKnot {frequencyHz=12000,absorption=1,scattering=1} };
            var curve=material.ExportEnergyCurve();
            Check(curve.frequencies.Length==3,"Material export preserves arbitrary authored frequency count.");
            Near(curve.frequencies[0],0,"Material curve supports DC.");
            Near(material.GetShaderData().absorptionLow.z,.4f,"Legacy shader samples the authored curve.");
            material.energyCurve[1].frequencyHz=0;
            Reject(()=>material.ExportEnergyCurve(),"Reject unordered material knots.");
        }
        finally { UnityEngine.Object.DestroyImmediate(material); }
    }

    static void Geometry()
    {
        var material=ScriptableObject.CreateInstance<AudioMaterial>();
        var objects=new List<GameObject>();
        try
        {
            foreach (var kind in new[] {PrimitiveType.Cube,PrimitiveType.Sphere,PrimitiveType.Capsule,PrimitiveType.Cylinder})
            {
                var obj=GameObject.CreatePrimitive(kind); objects.Add(obj);
                obj.transform.position=new Vector3(11,7,-3); obj.transform.localScale=new Vector3(2,3,4);
                var bindings=new List<RayTraceManager.TakeObjectBinding> { new RayTraceManager.TakeObjectBinding {id="native",transform=obj.transform,geometry=true} };
                var scene=SceneToTake.Compile(bindings,AcousticTake.Dimension.Spatial3D,.01f,material,new Transform[0],new AudioClip[0],new Transform[0],out var geometry);
                Check(geometry.Length>0,kind+" exports triangles.");
                if (kind==PrimitiveType.Cube) Check(geometry.Length==12,"Box triangle count.");
                Check(scene.materials.Length==1,"Material table is shared.");
                Check(geometry[0].a.magnitude<10,"Exported geometry is local, not world position.");
            }
            var polygon=new GameObject("Offset polygon"); objects.Add(polygon);
            var collider=polygon.AddComponent<PolygonCollider2D>(); collider.offset=new Vector2(3,4);
            collider.points=new[] {Vector2.zero,Vector2.right,Vector2.up};
            var planarBindings=new List<RayTraceManager.TakeObjectBinding> { new RayTraceManager.TakeObjectBinding {id="polygon",transform=polygon.transform,geometry=true} };
            SceneToTake.Compile(planarBindings,AcousticTake.Dimension.Planar2D,.01f,material,new Transform[0],new AudioClip[0],new Transform[0],out var segments);
            Near(segments[0].a.x,3,"Polygon collider offset x"); Near(segments[0].a.y,4,"Polygon collider offset y");
            var capsule=new GameObject("Collider-only capsule"); objects.Add(capsule);
            var cap=capsule.AddComponent<CapsuleCollider>(); cap.radius=.5f; cap.height=4; cap.direction=2;
            var capBindings=new List<RayTraceManager.TakeObjectBinding> {new RayTraceManager.TakeObjectBinding {id="capsule",transform=capsule.transform,geometry=true}};
            SceneToTake.Compile(capBindings,AcousticTake.Dimension.Spatial3D,.01f,material,new Transform[0],new AudioClip[0],new Transform[0],out var capTriangles);
            float min=float.PositiveInfinity,max=float.NegativeInfinity; bool cylinder=false;
            foreach (var p in capTriangles) foreach (var v in new[] {p.a,p.b,p.c}) { min=Mathf.Min(min,v.z); max=Mathf.Max(max,v.z); }
            foreach (var p in capTriangles) if (Mathf.Abs(p.a.z-p.b.z)>2 || Mathf.Abs(p.a.z-p.c.z)>2) cylinder=true;
            Near(min,-2,"Capsule lower tip"); Near(max,2,"Capsule upper tip"); Check(cylinder,"Capsule has explicit cylinder between hemispheres.");
            var capsule2D=new GameObject("Planar capsule"); objects.Add(capsule2D);
            var cap2=capsule2D.AddComponent<CapsuleCollider2D>(); cap2.size=new Vector2(1,4);
            capBindings[0].transform=capsule2D.transform;
            SceneToTake.Compile(capBindings,AcousticTake.Dimension.Planar2D,.01f,material,new Transform[0],new AudioClip[0],new Transform[0],out var capSegments);
            min=float.PositiveInfinity; max=float.NegativeInfinity;
            foreach (var p in capSegments) { min=Mathf.Min(min,p.a.y); max=Mathf.Max(max,p.a.y); }
            Near(min,-2,"Planar capsule lower tip",.011f); Near(max,2,"Planar capsule upper tip",.011f);
        }
        finally { foreach (var obj in objects) UnityEngine.Object.DestroyImmediate(obj); UnityEngine.Object.DestroyImmediate(material); }
    }

    static void DirectPaths()
    {
        var scene=Scene(); var state=States();
        var settings=new TakeRender.Settings {speedOfSound=1000}; var response=new TakeRender.DirectPath[1];
        TakeRender.SolveDirect(scene,new AcousticTake.Primitive[0],state,settings,response);
        Near(response[0].delaySeconds,.001f,"Analytic direct delay"); Near(response[0].pressureGain,1,"Analytic direct gain");
        // Attach a wall to the first object's transform for this geometry fixture.
        state[0].flags|=AcousticTake.StateFlags.Geometry;
        var wall=new[] {new AcousticTake.Primitive {a=new Vector3(.5f,-1,-1),b=new Vector3(.5f,1,-1),c=new Vector3(.5f,0,1),objectIndex=0}};
        TakeRender.SolveDirect(scene,wall,state,settings,response); Near(response[0].pressureGain,0,"Triangle occlusion");
        scene.dimension=AcousticTake.Dimension.Planar2D;
        TakeRender.SolveDirect(scene,wall,state,settings,response); Near(response[0].pressureGain,0,"Segment occlusion");
        state[0].flags&=~AcousticTake.StateFlags.Geometry;
        TakeRender.SolveDirect(scene,wall,state,settings,response); Near(response[0].pressureGain,1,"Disabled geometry does not block.");
    }

    static void AudioEventsAndTail()
    {
        var scene=Scene(2); var state=States(2);
        using (var writer=Writer("audio",scene))
        {
            writer.Append(0,state);
            writer.Append(new AcousticTake.VoiceEvent {sample=7,sourceIndex=0,action=AcousticTake.VoiceAction.Play,gain=1});
            state[0].flags&=~AcousticTake.StateFlags.Source; writer.Append(8,state); // emitted sample 7 still arrives later
            writer.Append(15,state); writer.Complete();
        }
        using (var take=AcousticTake.Open(Path.Combine(directory,"audio")))
        {
            var report=TakeRender.RenderDirect(take,new TakeRender.Settings {responseStepSamples=11,speedOfSound=1000,tailSamples=3},Path.Combine(directory,"render"));
            Check(report.complete,"Completed render manifest.");
            var a=ReadWave(Path.Combine(directory,"render","listener-0.wav"));
            var b=ReadWave(Path.Combine(directory,"render","listener-1.wav"));
            Near(a[7],0,"No early sound"); Near(a[8],1,"Play event inside block, propagation delay preserved");
            Near(a[9],0,"Disabled source mutes new emission"); Near(b[9],.5f,"Already emitted sound survives source disable");
        }
    }
    static float[] ReadWave(string path)
    {
        using (var reader=new BinaryReader(File.OpenRead(path)))
        {
            reader.BaseStream.Position=44;
            var samples=new float[(reader.BaseStream.Length-44)/4];
            for (int i=0; i<samples.Length; i++) samples[i]=reader.ReadSingle();
            return samples;
        }
    }

    static void ReplayRestoresScene()
    {
        var host=new GameObject("Take host"); var a=new GameObject("Emitter"); var b=new GameObject("Receiver");
        var manager=host.AddComponent<RayTraceManager>();
        try
        {
            manager.takeObjects=new List<RayTraceManager.TakeObjectBinding> {
                new RayTraceManager.TakeObjectBinding {id="object-0",transform=a.transform},
                new RayTraceManager.TakeObjectBinding {id="object-1",transform=b.transform} };
            a.transform.position=new Vector3(99,1,2);
            var controller=a.AddComponent<Oscillator>(); controller.enabled=true;
            using (var writer=Writer("replay",Scene())) { writer.Append(0,States()); writer.Complete(); }
            manager.BeginTakeReplay(Path.Combine(directory,"replay"),false);
            Near(a.transform.position.x,0,"Visual replay applies stored pose"); Check(!controller.enabled,"Suspend existing motion writer.");
            manager.SetTakeReplayPlaying(true); Check(manager.IsTakeReplayPlaying,"Resume pose replay.");
            manager.SetTakeReplayPlaying(false); Check(!manager.IsTakeReplayPlaying,"Pause for scrubbing.");
            manager.StopTakeReplay();
            Near(a.transform.position.x,99,"Restore live pose after replay."); Check(controller.enabled,"Restore motion writer state.");
        }
        finally { UnityEngine.Object.DestroyImmediate(host); UnityEngine.Object.DestroyImmediate(a); UnityEngine.Object.DestroyImmediate(b); }
    }

    static void ManagerCapture()
    {
        var host=new GameObject("Capture host"); var a=new GameObject("Source A"); var b=new GameObject("Source B"); var listener=new GameObject("Listener");
        var clip=AudioClip.Create("Shared PCM",32,1,1000,false); clip.SetData(new float[32],0);
        var manager=host.AddComponent<RayTraceManager>(); manager.source=a.transform; manager.listener=listener.transform; manager.inputClip=clip; manager.sampleRate=1000;
        try
        {
            var first=a.AddComponent<RaytracedAudioSource>(); first.rayTraceManager=manager; first.clip=clip;
            var second=b.AddComponent<RaytracedAudioSource>(); second.rayTraceManager=manager; second.clip=clip;
            second.Apply(); Check(manager.source==a.transform,"Additional emitter must not replace primary emitter.");
            manager.takeSources.Add(first); manager.takeSources.Add(second); manager.RegisterTakeObjects();
            string path=Path.Combine(directory,"manager"); manager.BeginTakeRecording(path);
            manager.RecordSourceEvent(a.transform,AcousticTake.VoiceAction.Play);
            manager.RecordSourceEvent(b.transform,AcousticTake.VoiceAction.Play);
            first.SetGain(.25f);
            Reject(()=>first.SetGain(float.NaN),"Source control rejects nonfinite gain.");
            manager.EndTakeRecording();
            using (var take=AcousticTake.Open(path))
            {
                Check(take.Metadata.scene.sources.Length==2,"Manager captures both emitter bindings.");
                Check(take.Metadata.scene.clips.Length==1,"Shared PCM is stored once.");
                Check(take.Events.Length==3,"Independent source events and source-control gain are recorded.");
                Check(take.Events[2].action==AcousticTake.VoiceAction.Gain && take.Events[2].gain==.25f && first.gain==.25f,"Gain control preserves the exact value in the source and take.");
                Check(take.Metadata.sampleRate==1000,"Silent capture preserves its configured sample rate.");
            }
            // A companion-input finalization failure must revoke acoustic success.
            string failedPath=Path.Combine(directory,"manager-compact-failure");
            Action<string> failure=_=>throw new InvalidOperationException("Injected input finalization failure");
            manager.RecordingFinalizing+=failure;manager.BeginTakeRecording(failedPath,false);
            Reject(manager.EndTakeRecording,"Companion failure aborts compact save");manager.RecordingFinalizing-=failure;
            Check(!manager.IsRecordingTake && manager.TakeStatus.StartsWith("Recording failed:"),"Failed compact session releases ownership and reports failure");
            Reject(()=>AcousticTake.ReadManifest(failedPath),"Acoustic manifest cannot hide a failed required input recorder");
            string aborted=Path.Combine(directory,"manager-input-aborted");manager.BeginTakeRecording(aborted,false);
            manager.AbortTakeRecording("Injected tick failure");
            Reject(()=>AcousticTake.ReadManifest(aborted),"Tick failure cannot be saved as a completed take");
            manager.audioManager=host.AddComponent<AudioManager>();
            manager.useTakeRenderer=true;
            manager.sampleRate=AudioSettings.outputSampleRate==48000 ? 44100:48000;
            string livePath=Path.Combine(directory,"manager-live-rate");
            manager.BeginTakeRecording(livePath);
            Check(manager.sampleRate==AudioSettings.outputSampleRate,"Live capture selects the actual device rate.");
            Check(manager.liveDiagnostics.sampleRate==AudioSettings.outputSampleRate,"Live diagnostics use the device rate.");
            manager.EndTakeRecording();
            using (var take=AcousticTake.Open(livePath))
                Check(take.Metadata.sampleRate==AudioSettings.outputSampleRate,"Live take stores the device rate for offline replay.");
        }
        finally
        {
            UnityEngine.Object.DestroyImmediate(host); UnityEngine.Object.DestroyImmediate(a); UnityEngine.Object.DestroyImmediate(b);
            UnityEngine.Object.DestroyImmediate(listener); UnityEngine.Object.DestroyImmediate(clip);
        }
    }

    static void CaptureMutation()
    {
        var host=new GameObject("Mutation capture"); var box=GameObject.CreatePrimitive(PrimitiveType.Cube);
        var manager=host.AddComponent<RayTraceManager>(); var material=ScriptableObject.CreateInstance<AudioMaterial>();
        var mesh=UnityEngine.Object.Instantiate(box.GetComponent<MeshFilter>().sharedMesh); box.GetComponent<MeshFilter>().sharedMesh=mesh;
        try
        {
            manager.takeDefaultMaterial=material; manager.obstacleObjects=new List<GameObject> {box}; manager.RegisterTakeObjects();
            manager.BeginTakeRecording(Path.Combine(directory,"rigid")); box.transform.rotation=Quaternion.Euler(21,37,54); box.transform.position=Vector3.one;
            manager.EndTakeRecording(); Check(File.Exists(Path.Combine(directory,"rigid","take.json")),"Rigid movement passes strict validation.");
            manager.BeginTakeRecording(Path.Combine(directory,"mutated-mesh"));
            var vertices=mesh.vertices; vertices[0]+=Vector3.right*.1f; mesh.vertices=vertices;
            Reject(manager.EndTakeRecording,"In-place mesh edits invalidate capture.");
            Reject(()=>AcousticTake.Open(Path.Combine(directory,"mutated-mesh")),"Changed geometry cannot finalize a take.");
            manager.BeginTakeRecording(Path.Combine(directory,"mutated-material")); material.porosity+=.1f;
            Reject(manager.EndTakeRecording,"Material edits invalidate capture.");
            var surface=box.AddComponent<AcousticSurface>(); surface.geometrySource=AcousticSurface.GeometrySource.Collider;
            manager.BeginTakeRecording(Path.Combine(directory,"mutated-collider")); box.GetComponent<BoxCollider>().size+=Vector3.one;
            Reject(manager.EndTakeRecording,"Collider shape edits invalidate capture.");
        }
        finally { UnityEngine.Object.DestroyImmediate(host); UnityEngine.Object.DestroyImmediate(box); UnityEngine.Object.DestroyImmediate(mesh); UnityEngine.Object.DestroyImmediate(material); }
    }

    static void StreamingMatchesOffline(ComputeShader shader=null,TakeRender.DelayFilter filter=TakeRender.DelayFilter.Linear)
    {
        string suffix=(shader==null ? "cpu":"gpu")+"-"+filter;
        var scene=Scene(3); scene.sources=new[] {new AcousticTake.SourceDefinition {objectIndex=0,clipIndex=0},new AcousticTake.SourceDefinition {objectIndex=1,clipIndex=0}};
        scene.listeners=new[] {2,3};
        var states=States(3); states[1].flags=AcousticTake.StateFlags.Active|AcousticTake.StateFlags.Source;
        var pcm=new float[512]; for (int i=0; i<pcm.Length; i++) pcm[i]=Mathf.Sin(2*Mathf.PI*53*i/1000);
        long[] observations={0,17,53,110,219,400,600,900,1000};
        var events=new[] {
            new AcousticTake.VoiceEvent {sample=9,sourceIndex=0,action=AcousticTake.VoiceAction.Play,gain=.4f,loop=true},
            new AcousticTake.VoiceEvent {sample=37,sourceIndex=1,action=AcousticTake.VoiceAction.Play,gain=.2f,loop=true},
            new AcousticTake.VoiceEvent {sample=451,sourceIndex=0,action=AcousticTake.VoiceAction.Gain,gain=.1f},
            new AcousticTake.VoiceEvent {sample=701,sourceIndex=1,action=AcousticTake.VoiceAction.Stop} };
        using (var writer=Writer("streaming-"+suffix,scene,pcm))
        {
            foreach (long t in observations)
            {
                states[2].position=new Vector3(2+t*.001f,0,0); states[3].position=new Vector3(4,1+t*.002f,0);
                states[2].flags&=~AcousticTake.StateFlags.Cut;
                if (t==17 || t==53 || t==900) states[2].flags|=AcousticTake.StateFlags.Cut;
                if (t>=600) states[0].flags&=~AcousticTake.StateFlags.Source;
                writer.Append(t,states);
            }
            foreach (var e in events) writer.Append(e); writer.Complete();
        }
        var settings=new TakeRender.Settings {responseStepSamples=31,transitionSamples=11,tailSamples=30,speedOfSound=1000,delayFilter=filter,maximumDelaySeconds=.1f,maximumResponsesPerBatch=4};
        using (var take=AcousticTake.Open(Path.Combine(directory,"streaming-"+suffix)))
        {
            string output=Path.Combine(directory,"streaming-offline-"+suffix); TakeRender.RenderDirect(take,settings,output,null,shader);
            var rendered=new List<float>[] {new List<float>(),new List<float>()};
            Action<long,float[],int> collect=(sample,block,count)=> { for (int n=0; n<count; n++) for (int l=0; l<2; l++) rendered[l].Add(block[n*2+l]); };
            var left=new AcousticTake.State[4]; var right=new AcousticTake.State[4]; take.ReadStoredFrame(0,left);
            using (var live=new TakeRender.Live(scene,take.Geometry,new[] {pcm},1000,settings,null,0,0,shader))
            {
                live.Observe(0,0,left,left,collect); int e=0;
                for (int f=1; f<observations.Length; f++)
                {
                    while (e<events.Length && events[e].sample<=observations[f]) live.Append(events[e++]);
                    take.ReadStoredFrame(f,right); live.Observe(observations[f-1],observations[f],left,right,collect);
                    Array.Copy(right,left,4);
                }
                live.Finish(1000,left,collect);
                Check(live.PendingObservations==0,"Finalization drains owned observations.");
            }
            for (int l=0; l<2; l++)
            {
                var expected=ReadWave(Path.Combine(output,"listener-"+l+".wav")); Check(rendered[l].Count==expected.Length,"Streaming output covers the offline timeline.");
                float error=0; for (int n=0; n<expected.Length; n++) error=Mathf.Max(error,Mathf.Abs(expected[n]-rendered[l][n]));
                Near(error,0,"Incremental live producer matches offline samples",shader==null ? 1e-7f:1e-4f);
            }
            var high=new TakeRender.Settings {responseStepSamples=13,transitionSamples=11,tailSamples=30,speedOfSound=1000,delayFilter=filter,maximumDelaySeconds=.1f,maximumResponsesPerBatch=4};
            var comparison=TakeComparison.Run(take,settings,high,Path.Combine(directory,"comparison-"+suffix));
            Check(comparison.complete && comparison.low.outputSamples==comparison.high.outputSamples,"Comparison timelines align.");
            Check(comparison.high.responseCount>comparison.low.responseCount,"Fine budget performs more solves.");
            Check(comparison.listeners[0].rmsError>0,"Moving scene exposes temporal approximation error.");
        }
    }

    static void GuardSnapshots()
    {
        var root=new GameObject("Snapshot guard"); var box=GameObject.CreatePrimitive(PrimitiveType.Cube);
        box.transform.SetParent(root.transform); var surface=box.AddComponent<AcousticSurface>();
        var manager=root.AddComponent<RayTraceManager>(); manager.takeRoot=root.transform;
        var material=ScriptableObject.CreateInstance<AudioMaterial>(); manager.takeDefaultMaterial=material;
        material.energyCurve=new[] {new AudioMaterial.EnergyKnot {frequencyHz=100,absorption=.2f,scattering=.1f}};
        var mesh=UnityEngine.Object.Instantiate(box.GetComponent<MeshFilter>().sharedMesh); box.GetComponent<MeshFilter>().sharedMesh=mesh;
        try
        {
            manager.RegisterTakeObjects();
            var guard=new SceneToTake.Guard(manager,new Transform[0],new AudioClip[0],new Transform[0]);
            for (int i=0;i<10;i++) guard.Validate();
            long memory=GC.GetAllocatedBytesForCurrentThread(),start=System.Diagnostics.Stopwatch.GetTimestamp();
            for (int i=0;i<1000;i++) guard.Validate();
            long bytes=GC.GetAllocatedBytesForCurrentThread()-memory;
            Debug.Log("ACOUSTIC_GUARD: 1000 observations seconds="+TakeRender.Elapsed(start)+" bytes="+bytes);
            Check(bytes==0,"Stable mesh/material validation allocates no managed memory after warmup.");
            var indices=mesh.triangles; int first=indices[0]; indices[0]=indices[1]; mesh.triangles=indices;
            Reject(guard.Validate,"In-place mesh index edits are detected."); indices[0]=first; mesh.triangles=indices;
            material.energyCurve[0].absorption=.3f; Reject(guard.Validate,"In-place authored material knots are detected."); material.energyCurve[0].absorption=.2f;
            surface.geometrySource=AcousticSurface.GeometrySource.None; Reject(guard.Validate,"Geometry selection changes are detected."); surface.geometrySource=AcousticSurface.GeometrySource.Auto;
            guard.Validate();
            manager.BeginTakeRecording(Path.Combine(directory,"owned-session"));
            Check(manager.Mode==RayTraceManager.SessionMode.Recording,"Recording owns the session.");
            var observe=typeof(RayTraceManager).GetMethod("CaptureTakeState",System.Reflection.BindingFlags.NonPublic|System.Reflection.BindingFlags.Instance);
            box.transform.position=Vector3.right; observe.Invoke(manager,new object[] {0L}); box.transform.position=Vector3.zero;
            Reject(manager.StartPrecomputedPlayback,"Legacy precomputation cannot share a capture session.");
            Reject(manager.RegisterTakeObjects,"Active registration is immutable.");
            manager.EndTakeRecording(); Check(manager.Mode==RayTraceManager.SessionMode.Idle,"Closing a silent take releases session ownership.");
            using (var take=AcousticTake.Open(Path.Combine(directory,"owned-session")))
            {
                var state=new AcousticTake.State[take.Metadata.scene.objects.Length]; take.ReadStoredFrame(0,state);
                Check(state[0].position==Vector3.zero && take.Metadata.observationPolicy=="first-observation-per-audio-sample","Duplicate observations cannot rewrite a pose already used by the live renderer.");
            }
        }
        finally { UnityEngine.Object.DestroyImmediate(root); UnityEngine.Object.DestroyImmediate(mesh); UnityEngine.Object.DestroyImmediate(material); }
    }

    static void ColliderProxyGuard()
    {
        var root=new GameObject("Collider-only proxy");
        var manager=root.AddComponent<RayTraceManager>(); manager.takeRoot=root.transform;
        var material=ScriptableObject.CreateInstance<AudioMaterial>(); manager.takeDefaultMaterial=material;
        root.AddComponent<BoxCollider>(); var surface=root.AddComponent<AcousticSurface>();
        surface.geometrySource=AcousticSurface.GeometrySource.Collider;
        try
        {
            manager.RegisterTakeObjects();
            var guard=new SceneToTake.Guard(manager,new Transform[0],new AudioClip[0],new Transform[0]);
            for(int i=0;i<10;i++) {root.transform.position+=Vector3.right; guard.Validate();}
            Check(true,"Collider-only proxies remain valid while moving without a MeshFilter.");
            UnityEngine.Object.DestroyImmediate(surface);
            Reject(guard.Validate,"Removing a captured acoustic component is detected.");
            root.AddComponent<AcousticSurface>().geometrySource=AcousticSurface.GeometrySource.Collider;
            Reject(guard.Validate,"Replacing a captured acoustic component is detected.");
        }
        finally {UnityEngine.Object.DestroyImmediate(root); UnityEngine.Object.DestroyImmediate(material);}
    }

    static void BoundedHistory()
    {
        var pcm=new float[127]; for (int i=0;i<pcm.Length;i++) pcm[i]=Mathf.Sin(i*.13f);
        var reference=new TakeRender.Signal(pcm); var bounded=new TakeRender.Signal(pcm,64);
        var play=new AcousticTake.VoiceEvent {action=AcousticTake.VoiceAction.Play,gain=1,loop=true};
        reference.Append(play); bounded.Append(play); reference.SetActive(0,true); bounded.SetActive(0,true);
        float error=0;
        for (int sample=0;sample<10000;sample+=10)
        {
            bounded.Prune(sample-100);
            var e=new AcousticTake.VoiceEvent {sample=sample,action=AcousticTake.VoiceAction.Gain,gain=(sample%70)*.01f};
            reference.Append(e); bounded.Append(e); bool enabled=sample%30!=0; reference.SetActive(sample,enabled); bounded.SetActive(sample,enabled);
            for (int at=sample-100;at<=sample;at++) error=Mathf.Max(error,Mathf.Abs(reference.Read(at+.25)-bounded.Read(at+.25)));
        }
        Near(error,0,"Pruning preserves delayed reads and looping voice phase");
        Check(bounded.RetainedEvents<32,"Live history stays bounded across many updates.");
        var tiny=new TakeRender.Signal(pcm,2); tiny.SetActive(0,true); tiny.Append(play);
        Reject(()=>tiny.Append(new AcousticTake.VoiceEvent {sample=1,action=AcousticTake.VoiceAction.Stop}),"History overflow fails explicitly.");
    }

    static void DelayFilters()
    {
        TakeRender.Validate(JsonUtility.FromJson<TakeRender.Settings>("{}"));
        Check(true,"Existing JSON jobs retain defaults for newly introduced settings.");
        var migrated=JsonUtility.FromJson<TakeRender.Settings>("{\"fftSize\":1024,\"raysPerPair\":1024}");
        Check(migrated.raysPerPair==0 && migrated.raysPerBin==2 && migrated.RaysPerPair==1026,"Legacy total-ray budgets migrate to complete frequency bins.");
        migrated.raysPerBin=3;Check(migrated.RaysPerPair==1539,"Migrated settings remain editable through the new field.");
        const int rate=1000,count=256,start=100;
        var pcm=new float[1000]; for (int i=0;i<pcm.Length;i++) pcm[i]=(float)Math.Sin(2*Math.PI*.35*i);
        var signal=new TakeRender.Signal(pcm); signal.SetActive(0,true); signal.Append(new AcousticTake.VoiceEvent {action=AcousticTake.VoiceAction.Play,gain=1,loop=true});
        double linearError=0,sincError=0;
        foreach (var filter in new[] {TakeRender.DelayFilter.Linear,TakeRender.DelayFilter.WindowedSinc})
        {
            var settings=new TakeRender.Settings {delayFilter=filter,sincRadius=16};
            var mixer=new TakeRender.Mixer(Scene(),rate,settings,new[] {signal});
            var paths=new[] {new TakeRender.DirectPath {delaySeconds=.0005f,pressureGain=1}}; mixer.SetResponse(0,paths,States());
            var output=new float[count]; mixer.Render(start,count,output); double error=0;
            for (int i=0;i<count;i++)
            { double expected=Math.Sin(2*Math.PI*.35*(start+i-settings.DspLatencySamples-(double)paths[0].delaySeconds*rate)); error+=Math.Pow(output[i]-expected,2); }
            error=Math.Sqrt(error/count); if (filter==TakeRender.DelayFilter.Linear) linearError=error; else sincError=error;
            paths[0].delaySeconds=11; Reject(()=>mixer.SetResponse(256,paths,States()),"Responses cannot exceed their declared history horizon.");
        }
        Check(sincError<.01 && sincError<linearError*.1,"Configurable sinc FIR improves high-frequency fractional-delay accuracy against an analytic sine.");
    }

    static void PlayedOutput()
    {
        string path=Path.Combine(directory,"played-output");
        var capture=new AudioManager.PlaybackCapture(path,1000,64,4,0,0,new TakeRender.Settings());
        var queue=new AudioManager.PcmQueue(64,1000,capture); var data=new float[8]; var expected=new List<float>();
        queue.Publish(0,new[] {1f,2f,3f,4f},4);
        long memory=GC.GetAllocatedBytesForCurrentThread(); queue.Consume(data,2);
        Check(GC.GetAllocatedBytesForCurrentThread()==memory,"Output capture allocates no managed memory in the callback.");
        for (int i=0;i<4;i++) expected.Add(data[i*2]); capture.Drain();
        queue.Consume(data,2); for (int i=0;i<4;i++) expected.Add(data[i*2]); capture.Drain(); // underrun silence is part of actual output
        queue.Publish(8,new[] {5f,6f,7f,8f},4); queue.Consume(data,2,true);
        for (int i=0;i<4;i++) expected.Add(data[i*2]); capture.Drain(); // mute still advances the sample clock
        for (int block=0;block<32;block++)
        {
            queue.Publish(queue.ReadSample,new[] {(float)block,2f,3f,4f},4); queue.Consume(data,2);
            for (int i=0;i<4;i++) expected.Add(data[i*2]); capture.Drain();
        }
        capture.RecordResponse(queue.ReadSample,queue.ReadSample,new[] {new TakeRender.DirectPath {delaySeconds=.1f,pressureGain=.5f}});
        queue.Complete(); queue.Consume(data,2); for (int i=0;i<4;i++) expected.Add(data[i*2]); capture.Drain();
        var actual=ReadWave(Path.Combine(path,"output.wav"));
        Check(capture.Closed && capture.Metadata.complete && actual.Length==expected.Count,"Output capture completes at a callback boundary.");
        for (int i=0;i<actual.Length;i++) Near(actual[i],expected[i],"Captured PCM equals component output",0);
        Check(new FileInfo(Path.Combine(path,"callbacks.bin")).Length==capture.Metadata.callbacks*40,"Callback sample/timing records align.");
        Check(capture.Metadata.audioHash==AcousticTake.HashFile(Path.Combine(path,"output.wav")),"Output capture has a content hash.");
        using (var overflow=new AudioManager.PlaybackCapture(Path.Combine(directory,"capture-overflow"),1000,4,4,0,0,new TakeRender.Settings()))
        {
            overflow.Publish(0,data,2,0); overflow.Publish(4,data,2,0); overflow.Drain();
            Check(!overflow.Metadata.complete && overflow.Metadata.overflowSamples==4,"Capture overflow cannot be promoted to exact playback.");
        }
    }

    static void ConcurrentOutputCapture()
    {
        string path=Path.Combine(directory,"concurrent-output");
        using (var capture=new AudioManager.PlaybackCapture(path,1000,1024,4,0,0,new TakeRender.Settings()))
        {
            capture.CompleteProducer(1024);
            var worker=new System.Threading.Thread(()=>
            {
                var data=new float[4];
                for (int block=0;block<256;block++)
                {
                    for (int i=0;i<4;i++) data[i]=block+1;
                    capture.Publish(block*4,data,1,System.Diagnostics.Stopwatch.GetTimestamp()); System.Threading.Thread.Yield();
                }
            });
            worker.Start();
            while (worker.IsAlive) { capture.Drain(); System.Threading.Thread.Yield(); }
            worker.Join(); capture.Drain();
            var pcm=ReadWave(Path.Combine(path,"output.wav"));
            Check(capture.Metadata.complete && pcm.Length==1024,"Concurrent finalization drains the last published callback before marking output complete.");
            bool equal=true; for (int i=0;i<pcm.Length;i++) equal &= pcm[i]==i/4+1;
            Check(equal,"Concurrent callback publication preserves PCM order and values.");
        }
    }

    static void ReferenceQuality()
    {
        var scene=Scene(); var state=States(); var pcm=new float[100]; for (int i=0;i<pcm.Length;i++) pcm[i]=Mathf.Sin(i*.7f);
        using (var writer=Writer("reference-static",scene,pcm))
        {
            writer.Append(0,state); writer.Append(100,state);
            writer.Append(new AcousticTake.VoiceEvent {action=AcousticTake.VoiceAction.Play,gain=1,loop=true}); writer.Complete();
        }
        using (var take=AcousticTake.Open(Path.Combine(directory,"reference-static")))
        {
            var result=TakeComparison.Run(take,new TakeRender.Settings {responseStepSamples=16},new TakeRender.Settings {responseStepSamples=4},Path.Combine(directory,"checked-reference"),null,null,new TakeComparison.ReferenceCheck());
            Check(result.referenceAccepted && result.eligibleForTemporalTraining,"A static direct-model reference passes an exact temporal check.");
            Check(result.low.provenance.csharpModule==typeof(AcousticTake).Assembly.ManifestModule.ModuleVersionId.ToString() && result.low.settingsHash!=result.high.settingsHash,"Reports identify compiled code and exact settings.");
            Check(take.Metadata.captureProvenance!=null,"New takes retain capture code provenance.");
        }
        using (var take=AcousticTake.Open(Path.Combine(directory,"streaming-cpu-Linear")))
        {
            var result=TakeComparison.Run(take,new TakeRender.Settings {responseStepSamples=100},new TakeRender.Settings {responseStepSamples=31},Path.Combine(directory,"rejected-reference"),null,null,new TakeComparison.ReferenceCheck());
            Check(result.complete && !result.referenceAccepted && !result.eligibleForTemporalTraining,"A completed but unconverged temporal comparison is not eligible for training.");
        }
    }

    static void QueueDeadlines()
    {
        var queue=new AudioManager.PcmQueue(8); var data=new float[4];
        queue.Publish(0,new[] {1f,2f,3f,4f},4); queue.Consume(data,1);
        Check(data[0]==1 && data[3]==4,"PCM queue preserves sequence.");
        queue.Publish(4,new[] {5f,6f,7f,8f,9f,10f},6); queue.Consume(data,1); queue.Consume(data,1);
        Check(data[0]==9 && data[1]==10 && data[2]==0 && queue.UnderrunSamples==2,"Queue wraps and emits counted silence on missed deadlines.");
        queue.Publish(10,new[] {11f,12f,13f,14f},4); queue.Consume(data,2);
        Check(data[0]==13 && data[1]==13 && data[2]==14 && queue.LateSamples==2,"Late PCM is dropped at its timestamp, not played late.");
        Check(!queue.Publish(14,new float[9],9),"Queue refuses to overwrite unread audio.");
        queue.Complete(); long misses=queue.UnderrunSamples; queue.Consume(data,1);
        Check(queue.UnderrunSamples==misses,"Completed streams do not report idle underruns.");
        long before=GC.GetAllocatedBytesForCurrentThread();
        for (int i=0; i<100; i++) queue.Consume(data,1);
        Check(GC.GetAllocatedBytesForCurrentThread()==before,"Audio consumption allocates no managed memory.");
    }

    static void LongSampleClock()
    {
        long first=1L<<30;
        var signal=new TakeRender.Signal(new[] {1f,-1f}); signal.SetActive(0,true);
        signal.Append(new AcousticTake.VoiceEvent {sample=first,sourceIndex=0,action=AcousticTake.VoiceAction.Play,gain=1,loop=true});
        var mixer=new TakeRender.Mixer(Scene(),1024,new TakeRender.Settings(),new[] {signal});
        mixer.SetResponse(first,new[] {new TakeRender.DirectPath {delaySeconds=1f/1024,pressureGain=1}},States());
        var output=new float[4]; mixer.Render(first+2,4,output);
        Check(output[0]==-1 && output[1]==1 && output[2]==-1 && output[3]==1,"Long takes preserve single-sample phase beyond float cursor precision.");
        var loud=new TakeRender.Signal(new[] {float.MaxValue}); loud.SetActive(0,true);
        loud.Append(new AcousticTake.VoiceEvent {action=AcousticTake.VoiceAction.Play,gain=2,loop=true});
        var overflow=new TakeRender.Mixer(Scene(),1024,new TakeRender.Settings(),new[] {loud});
        overflow.SetResponse(0,new[] {new TakeRender.DirectPath {pressureGain=1}},States());
        Reject(()=>overflow.Render(0,1,output),"Reject overflowed audio instead of completing a non-finite dataset.");
    }

    static void GpuDirectChecks()
    {
        var shader=AssetDatabase.LoadAssetAtPath<ComputeShader>("Assets/Shaders/Direct.compute");
        Check(shader!=null,"Direct shader is imported.");
        var scene=Scene(2); var state=States(2); state[0].flags|=AcousticTake.StateFlags.Geometry;
        var geometry=new[] {new AcousticTake.Primitive {a=new Vector3(.5f,-1,-1),b=new Vector3(.5f,1,-1),c=new Vector3(.5f,0,1),objectIndex=0}};
        var cpu=new TakeRender.DirectPath[2]; var gpu=new TakeRender.DirectPath[2]; var settings=new TakeRender.Settings();
        foreach (var dimension in new[] {AcousticTake.Dimension.Planar2D,AcousticTake.Dimension.Spatial3D})
        {
            scene.dimension=dimension;
            using (var solver=new DirectSolver(scene,geometry,settings,shader))
                for (int i=0; i<25; i++)
                {
                    state[0].rotation=Quaternion.AngleAxis(i*7,Vector3.forward);
                    state[0].position=new Vector3(i*.017f,i*.031f,0);
                    TakeRender.SolveDirect(scene,geometry,state,settings,cpu); solver.Solve(state,gpu);
                    for (int pair=0; pair<2; pair++)
                    { Near(gpu[pair].delaySeconds,cpu[pair].delaySeconds,"GPU direct delay",1e-6f); Near(gpu[pair].pressureGain,cpu[pair].pressureGain,"GPU direct gain/occlusion",1e-5f); Near(Vector4.Distance(gpu[pair].direction,cpu[pair].direction),0,"GPU arrival direction",1e-6f); }
                }
        }
    }

    static void GpuBatches()
    {
        var shader=AssetDatabase.LoadAssetAtPath<ComputeShader>("Assets/Shaders/Direct.compute");
        var scene=Scene(); var states=new AcousticTake.State[10]; var result=new TakeRender.DirectPath[5];
        for (int f=0;f<5;f++) { var state=States(); state[1].position.x=f+1; Array.Copy(state,0,states,f*2,2); }
        using (var solver=new DirectSolver(scene,new AcousticTake.Primitive[0],new TakeRender.Settings(),shader,5))
        {
            solver.BeginSolve(states,5); Reject(()=>solver.BeginSolve(states,5),"Pending GPU response owns its buffers.");
            Check(solver.TryComplete(result,true),"Explicit shutdown wait completes async readback.");
            for (int f=0;f<5;f++) Near(result[f].pressureGain,1f/(f+1),"Batch responses use their own pose frame",1e-6f);
            solver.BeginSolve(states,5); // Disposal must wait before releasing request-owned buffers.
        }
        Debug.Log("ACOUSTIC_GPU_BATCH_PASS");
    }

    [MenuItem("Tools/Acoustics/Check legacy GPU FFT")]
    public static void RunGpu()
    {
        directory=Path.Combine(Path.GetTempPath(),"acoustic-gpu-checks-"+Guid.NewGuid().ToString("N")); Directory.CreateDirectory(directory);
        try
        {
            if (!SystemInfo.supportsComputeShaders) throw new InvalidOperationException("GPU FFT check requires compute support.");
            var shader=AssetDatabase.LoadAssetAtPath<ComputeShader>("Assets/Shaders/Raytrace2D.compute");
            const int count=1024;
            var input=new Vector2[count]; input[0]=Vector2.right;
            using (var buffer=new ComputeBuffer(count,8))
            {
                buffer.SetData(input); int forward=shader.FindKernel("FFT"), inverse=shader.FindKernel("IFFT");
                shader.SetBuffer(forward,"Data",buffer); shader.Dispatch(forward,1,1,1);
                var actual=new Vector2[count]; buffer.GetData(actual);
                for (int i=0; i<count; i++) { Near(actual[i].x,1,"Impulse spectrum real",1e-4f); Near(actual[i].y,0,"Impulse spectrum imaginary",1e-4f); }
                for (int i=0; i<count; i++) input[i]=new Vector2(Mathf.Sin(2*Mathf.PI*5*i/count),0);
                buffer.SetData(input); shader.Dispatch(forward,1,1,1); buffer.GetData(actual);
                Near(actual[5].y,-count*.5f,"Known sine FFT bin",.01f);
                Near(actual[count-5].y,count*.5f,"Conjugate FFT bin",.01f);
                shader.SetBuffer(inverse,"Data",buffer); shader.Dispatch(inverse,1,1,1); buffer.GetData(actual);
                for (int i=0; i<count; i++) { Near(actual[i].x,input[i].x,"FFT round trip",1e-4f); Near(actual[i].y,0,"Round-trip imaginary",1e-4f); }
            }
            var host=new GameObject("Prepare before manager Start");
            var clip=AudioClip.Create("Early preparation",count,1,48000,false);
            try
            {
                var manager=host.AddComponent<RayTraceManager>();
                manager.raytraceShader=shader; manager.inputClip=clip; manager.source=manager.listener=host.transform;
                manager.chunkSamples=128; manager.PrepareCurrentClip();
                Check(manager.chunkSamples==count,"Preparing before manager Start must use the actual kernel size.");
            }
            finally { UnityEngine.Object.DestroyImmediate(host); UnityEngine.Object.DestroyImmediate(clip); }
            GpuDirectChecks(); GpuBatches(); BinauralStreaming(AssetDatabase.LoadAssetAtPath<ComputeShader>("Assets/Shaders/Direct.compute"));
            var direct=AssetDatabase.LoadAssetAtPath<ComputeShader>("Assets/Shaders/Direct.compute");
            BvhChecks(direct);
            OfflineBatchChecks(direct);
            StreamingMatchesOffline(direct); StreamingMatchesOffline(direct,TakeRender.DelayFilter.WindowedSinc);
            Debug.Log("ACOUSTIC_GPU_FFT_PASS: "+SystemInfo.graphicsDeviceType);
            Debug.Log("ACOUSTIC_GPU_DIRECT_PASS: 2D/3D transformed geometry and multiple pairs");
            if (Application.isBatchMode) EditorApplication.Exit(0);
        }
        catch (Exception e)
        {
            Debug.LogException(e);
            if (Application.isBatchMode) EditorApplication.Exit(1); else throw;
        }
    }
}
