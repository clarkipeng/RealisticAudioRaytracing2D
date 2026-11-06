using NUnit.Framework;
using System;
using System.Collections.Generic;
using System.IO;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;

// Uses the actual Unity types, scene extraction and file reader. Also callable in batch mode.
[TestFixture]
public partial class TakeChecks
{
    string directory;

    [SetUp]
    public void SetUp()
    {
        directory=Path.Combine(Path.GetTempPath(),"acoustics-test-"+Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(directory);
        // Test Runner owns and restores the user's scene setup. Each case gets
        // a fresh empty scene, including when the runner's scene is untitled.
        EditorSceneManager.NewScene(NewSceneSetup.EmptyScene,NewSceneMode.Single);
        pacedTestCpuStarted=System.Diagnostics.Process.GetCurrentProcess().TotalProcessorTime.TotalSeconds;
        pacedTestWallStarted=System.Diagnostics.Stopwatch.GetTimestamp();
    }

    [TearDown]
    public void TearDown()
    {
        try
        {
            EditorSceneManager.NewScene(NewSceneSetup.EmptyScene,NewSceneMode.Single);
        }
        finally
        {
            if(Directory.Exists(directory))Directory.Delete(directory,true);
            string budget=Environment.GetEnvironmentVariable("ACOUSTIC_TEST_WORK_FRACTION");
            if(!string.IsNullOrEmpty(budget))
            {
                double fraction=double.Parse(budget,System.Globalization.CultureInfo.InvariantCulture);
                if(!(fraction>0 && fraction<=1))throw new ArgumentException("Invalid acoustic test work fraction.");
                double work=System.Diagnostics.Process.GetCurrentProcess().TotalProcessorTime.TotalSeconds-pacedTestCpuStarted;
                UnityEngine.Debug.LogFormat(LogType.Log,LogOption.NoStacktrace,null,"Acoustic test CPU work: {0:F3}s; pacing fraction {1:F2}; {2}",work,fraction,TestContext.CurrentContext.Test.Name);
                // Credit waits and pacing already performed inside long measurements.
                double elapsed=TakeRender.Elapsed(pacedTestWallStarted);
                double idle=Math.Max(0,work/fraction-elapsed);
                while(idle>0){int ms=(int)Math.Min(1000,Math.Ceiling(idle*1000));System.Threading.Thread.Sleep(ms);idle-=ms*.001;}
            }
        }
    }
    double pacedTestCpuStarted;
    long pacedTestWallStarted;

    static ComputeShader RequireGpu()
    {
        if(!SystemInfo.supportsComputeShaders || !SystemInfo.supportsAsyncGPUReadback)
            Assert.Ignore("Requires compute shaders and asynchronous GPU readback; CPU fallback would not test this contract.");
        var shader=AssetDatabase.LoadAssetAtPath<ComputeShader>("Assets/Shaders/Transport.compute");
        Assert.That(shader,Is.Not.Null,"Transport.compute must import successfully.");
        return shader;
    }

    string WriteMovingTake()
    {
        string path=Path.Combine(directory,"moving-input");
        var scene=Scene(3);scene.sources=new[]{new AcousticTake.SourceDefinition {objectIndex=0,clipIndex=0},
            new AcousticTake.SourceDefinition {objectIndex=1,clipIndex=0}};scene.listeners=new[]{2,3};
        var state=States(3);state[1].flags=AcousticTake.StateFlags.Active|AcousticTake.StateFlags.Source;
        var pcm=new float[113];for(int n=0;n<pcm.Length;n++)pcm[n]=Mathf.Sin(n*.31f)*.1f;
        using(var writer=new AcousticTake.Writer(path,8000,scene,Array.Empty<AcousticTake.Primitive>(),new[]{pcm}))
        {
            for(int sample=0;sample<=1024;sample+=128)
            {
                state[2].position=new Vector3(1+sample*.004f,0,0);
                state[3].position=new Vector3(-1,1+sample*.002f,0);
                writer.Append(sample,state);
            }
            for(int source=0;source<2;source++)writer.Append(new AcousticTake.VoiceEvent
                {sourceIndex=source,action=AcousticTake.VoiceAction.Play,gain=1,loop=true});
            writer.Complete();
        }
        return path;
    }


    AcousticTake.Scene Scene(int listeners=1)
    {
        var objects=new AcousticTake.ObjectDefinition[listeners+1];
        var indices=new int[listeners];
        for (int i=0; i<objects.Length; i++) objects[i]=new AcousticTake.ObjectDefinition { id="object-"+i,name="Object "+i,scale=Vector3.one };
        for (int i=0; i<listeners; i++) indices[i]=i+1;
        return new AcousticTake.Scene { dimension=AcousticTake.Dimension.Spatial3D,objects=objects,
            sources=new[] { new AcousticTake.SourceDefinition {objectIndex=0,clipIndex=0} },listeners=indices,
            clips=new[] {new AcousticTake.ClipDefinition {name="test"}},materials=new AcousticTake.MaterialDefinition[0] };
    }
    AcousticTake.State[] States(int listeners=1)
    {
        var states=new AcousticTake.State[listeners+1];
        for (int i=0; i<states.Length; i++) states[i]=new AcousticTake.State { position=new Vector3(i,0,0),rotation=Quaternion.identity,
            flags=AcousticTake.StateFlags.Active | (i == 0 ? AcousticTake.StateFlags.Source : AcousticTake.StateFlags.Listener) };
        return states;
    }
    AcousticTake.Writer Writer(string name, AcousticTake.Scene scene, float[] pcm=null)
    {
        return new AcousticTake.Writer(Path.Combine(directory,name),1000,scene,new AcousticTake.Primitive[0],new[] {pcm ?? new[] {1f,2f,3f}});
    }
    void Check(bool condition,string message) { Assert.That(condition, Is.True, message); }
    void Near(float actual,float expected,string message,float tolerance=1e-5f) { Check(Mathf.Abs(actual-expected)<=tolerance,message+": "+actual+" != "+expected); }
    void Reject(Action action,string message)
    {
        bool rejected=false; try { action(); } catch (InvalidDataException) { rejected=true; } catch (ArgumentException) { rejected=true; } catch (InvalidOperationException) { rejected=true; }
        Check(rejected,message);
    }

    [Test, Category("Core")]
    public void RoundTripAndCuts()
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

    [Test, Category("Core")]
    public void InvalidFiles()
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

    [Test, Category("Core")]
    public void Motion()
    {
        var a=States()[0]; var b=a; b.position.x=2; b.rotation=Quaternion.AngleAxis(90,Vector3.up); b.MeasureMotion(a,.5);
        Near(b.velocity.x,4,"Velocity uses timestamp interval"); Near(b.angularVelocity.y,Mathf.PI,"Angular velocity in rad/s");
        var c=b; c.position.x=5; c.MeasureMotion(b,.5);
        Near(c.acceleration.x,4,"Acceleration in m/s²");
        var cut=c; cut.flags|=AcousticTake.StateFlags.Cut; cut.position.x=100; cut.MeasureMotion(c,.5);
        Check((cut.flags&AcousticTake.StateFlags.VelocityValid)==0,"Teleport must not acquire physical velocity.");
        Near(cut.velocity.magnitude,0,"Teleport clears previous motion estimate.");
    }

    [Test, Category("Core")]
    public void Materials()
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
            material.admittanceCurve=new[]{new AudioMaterial.AdmittanceKnot {frequencyHz=0,real=.4f},
                new AudioMaterial.AdmittanceKnot {frequencyHz=125,real=.4f,imaginary=-.3f}};
            curve=material.ExportEnergyCurve();Near(curve.admittanceImag[1],-.3f,"Boundary export preserves signed reactive admittance.");
            var scene=Scene();scene.materials=new[]{curve};string path=Path.Combine(directory,"impedance");
            using(var writer=new AcousticTake.Writer(path,8000,scene,Array.Empty<AcousticTake.Primitive>(),new[]{new[]{1f}}))
            {writer.Append(0,States());writer.Complete();}
            using(var take=AcousticTake.Open(path))Near(take.Metadata.scene.materials[0].admittanceImag[1],-.3f,"Version 4 authenticates complex materials.");
            Reject(()=>new AcousticTake.Writer(Path.Combine(directory,"legacy-impedance"),8000,scene,Array.Empty<AcousticTake.Primitive>(),new[]{new[]{1f}},version:3),"Legacy schemas cannot store impedance.");
            material.admittanceCurve[0].real=-1;Reject(()=>material.ExportEnergyCurve(),"Reject an active boundary.");
        }
        finally { UnityEngine.Object.DestroyImmediate(material); }
    }

    [Test, Category("Core")]
    public void Geometry()
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

    [Test, Category("Core")]
    public void DirectPaths()
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

    [Test, Category("Core")]
    public void AudioEventsAndTail()
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
            var report=TakeRender.RenderDirect(take,new TakeRender.Settings {responseStepSamples=11,speedOfSound=1000,tailSamples=3,delayFilter=TakeRender.DelayFilter.Linear},Path.Combine(directory,"render"));
            Check(report.complete,"Completed render manifest.");
            var a=ReadWave(Path.Combine(directory,"render","listener-0.wav"));
            var b=ReadWave(Path.Combine(directory,"render","listener-1.wav"));
            Near(a[7],0,"No early sound"); Near(a[8],1,"Play event inside block, propagation delay preserved");
            Near(a[9],0,"Disabled source mutes new emission"); Near(b[9],.5f,"Already emitted sound survives source disable");
        }
    }
    float[] ReadWave(string path)
    {
        using (var reader=new BinaryReader(File.OpenRead(path)))
        {
            reader.BaseStream.Position=44;
            var samples=new float[(reader.BaseStream.Length-44)/4];
            for (int i=0; i<samples.Length; i++) samples[i]=reader.ReadSingle();
            return samples;
        }
    }

    [Test, Category("Core")]
    public void ReplayRestoresScene()
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

    [Test, Category("Core")]
    public void ManagerCapture()
    {
        var host=new GameObject("Capture host"); var a=new GameObject("Source A"); var b=new GameObject("Source B"); var listener=new GameObject("Listener");
        var clip=AudioClip.Create("Shared PCM",32,1,1000,false); clip.SetData(new float[32],0);
        var manager=host.AddComponent<RayTraceManager>(); manager.takeListeners.Add(listener.transform); manager.sampleRate=1000;
        try
        {
            var first=a.AddComponent<RaytracedAudioSource>(); first.rayTraceManager=manager; first.clip=clip;
            var second=b.AddComponent<RaytracedAudioSource>(); second.rayTraceManager=manager; second.clip=clip;
            manager.takeSources.Add(first); manager.takeSources.Add(second); manager.RegisterTakeObjects();
            string path=Path.Combine(directory,"manager"); manager.BeginTakeRecording(path);
            first.Play();
            second.Play();
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
            manager.monitorLive=true;
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

    [Test, Category("Core")]
    public void CaptureMutation()
    {
        var host=new GameObject("Mutation capture"); var box=GameObject.CreatePrimitive(PrimitiveType.Cube);
        var manager=host.AddComponent<RayTraceManager>(); var material=ScriptableObject.CreateInstance<AudioMaterial>();
        var mesh=UnityEngine.Object.Instantiate(box.GetComponent<MeshFilter>().sharedMesh); box.GetComponent<MeshFilter>().sharedMesh=mesh;
        try
        {
            manager.takeDefaultMaterial=material; manager.takeObjects.Add(new RayTraceManager.TakeObjectBinding {transform=box.transform,geometry=true}); manager.RegisterTakeObjects();
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

    [TestCase(false, TakeRender.DelayFilter.Linear, Category="Integration")]
    [TestCase(false, TakeRender.DelayFilter.WindowedSinc, Category="Integration")]
    [TestCase(true, TakeRender.DelayFilter.Linear, Category="GPU")]
    [TestCase(true, TakeRender.DelayFilter.WindowedSinc, Category="GPU")]
    public void StreamingMatchesOffline(bool gpu,TakeRender.DelayFilter filter)
    {
        var shader=gpu?RequireGpu():null;
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
        }
    }

    [Test, Category("Core")]
    public void GuardSnapshots()
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
            material.admittanceCurve=new[]{new AudioMaterial.AdmittanceKnot {real=.5f}};
            Reject(guard.Validate,"Changing boundary impedance invalidates capture.");material.admittanceCurve=Array.Empty<AudioMaterial.AdmittanceKnot>();
            surface.geometrySource=AcousticSurface.GeometrySource.None; Reject(guard.Validate,"Geometry selection changes are detected."); surface.geometrySource=AcousticSurface.GeometrySource.Auto;
            guard.Validate();
            manager.BeginTakeRecording(Path.Combine(directory,"owned-session"));
            Check(manager.Mode==RayTraceManager.SessionMode.Recording,"Recording owns the session.");
            var observe=typeof(RayTraceManager).GetMethod("CaptureTakeState",System.Reflection.BindingFlags.NonPublic|System.Reflection.BindingFlags.Instance);
            box.transform.position=Vector3.right; observe.Invoke(manager,new object[] {0L}); box.transform.position=Vector3.zero;
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

    [Test, Category("Core")]
    public void ColliderProxyGuard()
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
            UnityEngine.Object.DestroyImmediate(surface);
            Reject(guard.Validate,"Removing a captured acoustic component is detected.");
            root.AddComponent<AcousticSurface>().geometrySource=AcousticSurface.GeometrySource.Collider;
            Reject(guard.Validate,"Replacing a captured acoustic component is detected.");
        }
        finally {UnityEngine.Object.DestroyImmediate(root); UnityEngine.Object.DestroyImmediate(material);}
    }

    [Test, Category("Core")]
    public void BoundedHistory()
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

    [Test, Category("Core")]
    public void DelayFilters()
    {
        TakeRender.Validate(JsonUtility.FromJson<TakeRender.Settings>("{}"));
        var migrated=JsonUtility.FromJson<TakeRender.Settings>("{\"fftSize\":1024,\"raysPerPair\":1024}");
        migrated.reflectionSampling=TakeRender.ReflectionSampling.PrimitiveBaseline; // The only estimator that still traces rays per bin.
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

    [TestCase(1),TestCase(2),Category("Core")]
    public void OutputCapturePreservesPcmBeforeMute(int channels)
    {
        string path=Path.Combine(directory,"output");
        var settings=new TakeRender.Settings {listenerOutput=channels==1?TakeRender.ListenerOutput.Mono:TakeRender.ListenerOutput.Binaural};
        using var capture=new AudioManager.PlaybackCapture(path,44100,16,4,0,0,settings);
        var queue=new AudioManager.PcmQueue(16,44100,capture,channels);var data=new float[8];
        var pcm=new float[12*channels];for(int i=0;i<pcm.Length;i++)pcm[i]=(i+1)*.01f;
        queue.Publish(0,pcm,12,channels);
        capture.RecordResponse(0,0,new[]{new TakeRender.DirectPath {delaySeconds=.1f,pressureGain=.5f}});
        queue.Complete();queue.Consume(data,2);
        for(int i=0;i<data.Length;i++)Near(data[i],pcm[(i/2)*channels+i%channels],"Device channel mapping",0);
        long memory=GC.GetAllocatedBytesForCurrentThread();queue.Consume(data,2,muteOutput:true);
        long allocated=GC.GetAllocatedBytesForCurrentThread()-memory;
        Assert.That(allocated,Is.Zero,"Capture and speaker mute allocate no callback memory.");
        Assert.That(data,Is.All.Zero,"Speaker mute clears the device buffer after recording.");
        queue.Consume(data,2,silence:true);capture.Drain(); // Pausing suppresses the recorded signal too.
        Array.Clear(pcm,8*channels,4*channels);
        Assert.That(ReadWave(Path.Combine(path,"output.wav")),Is.EqualTo(pcm),"Mute preserves samples; pause records silence.");
        Assert.That(queue.ReadSample,Is.EqualTo(12));Assert.That(queue.UnderrunSamples,Is.Zero);
        Check(capture.Closed && capture.Metadata.complete && capture.Metadata.channelsPerListener==channels,"Capture drains with its channel layout intact.");
        Check(new FileInfo(Path.Combine(path,"callbacks.bin")).Length==3*40,"Callback timing records align.");
        Check(capture.Metadata.audioHash==AcousticTake.HashFile(Path.Combine(path,"output.wav")),"Output capture has a content hash.");
        var metrics=TakeComparison.CompareWave(Path.Combine(path,"output.wav"),Path.Combine(path,"output.wav"));
        Check(metrics.channels==channels && metrics.samples==12 && metrics.exactMatch,"Comparison counts frames rather than interleaved values.");
    }

    [Test,Category("Core")]
    public void OutputCaptureOverflowInvalidatesRecording()
    {
        using var capture=new AudioManager.PlaybackCapture(Path.Combine(directory,"overflow"),1000,4,4,0,0,new TakeRender.Settings());
        var data=new float[8];capture.Publish(0,data,2,0);capture.Publish(4,data,2,0);capture.Drain();
        Check(!capture.Metadata.complete && capture.Metadata.overflowSamples==4,"Capture overflow cannot be promoted to exact playback.");
    }

    [Test, Category("Core")]
    public void ConcurrentOutputCapture()
    {
        string path=Path.Combine(directory,"concurrent-output");
        using (var capture=new AudioManager.PlaybackCapture(path,1000,1024,4,0,0,new TakeRender.Settings()))
        {
            capture.CompleteProducer(1024);
            var worker=System.Threading.Tasks.Task.Run(()=>
            {
                var data=new float[4];
                for (int block=0;block<256;block++)
                {
                    for (int i=0;i<4;i++) data[i]=block+1;
                    capture.Publish(block*4,data,1,System.Diagnostics.Stopwatch.GetTimestamp()); System.Threading.Thread.Yield();
                }
            });
            var deadline=System.Diagnostics.Stopwatch.StartNew();
            while(!worker.IsCompleted && deadline.Elapsed.TotalSeconds<10) {capture.Drain();System.Threading.Thread.Yield();}
            Assert.That(worker.IsCompleted,Is.True,"Concurrent PCM producer timed out.");
            worker.GetAwaiter().GetResult();capture.Drain();
            var pcm=ReadWave(Path.Combine(path,"output.wav"));
            Check(capture.Metadata.complete && pcm.Length==1024,"Concurrent finalization drains the last published callback before marking output complete.");
            bool equal=true; for (int i=0;i<pcm.Length;i++) equal &= pcm[i]==i/4+1;
            Check(equal,"Concurrent callback publication preserves PCM order and values.");
        }
    }

    [Test, Category("Core")]
    public void StaticReferenceConverges()
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
    }

    [Test, Category("Integration")]
    public void UnconvergedReferenceIsIneligibleForTraining()
    {
        using (var take=AcousticTake.Open(WriteMovingTake()))
        {
            var result=TakeComparison.Run(take,new TakeRender.Settings {responseStepSamples=100},new TakeRender.Settings {responseStepSamples=31},Path.Combine(directory,"rejected-reference"),null,null,new TakeComparison.ReferenceCheck());
            Check(result.complete && !result.referenceAccepted && !result.eligibleForTemporalTraining,"A completed but unconverged temporal comparison is not eligible for training.");
        }
    }

    [Test, Category("Core")]
    public void QueueWrapPreservesSampleOrder()
    {
        var queue=new AudioManager.PcmQueue(8);var data=new float[4];
        queue.Publish(0,new[]{1f,2f,3f,4f,5f,6f},6);queue.Consume(data,1);
        queue.Publish(6,new[]{7f,8f,9f,10f},4);queue.Consume(data,1);
        Assert.That(data,Is.EqualTo(new[]{5f,6f,7f,8f}));
        var final=new float[2];queue.Consume(final,1);Assert.That(final,Is.EqualTo(new[]{9f,10f}));
        Assert.That(queue.UnderrunSamples,Is.Zero);
    }

    [Test, Category("Core")]
    public void QueueUnderrunEmitsCountedSilence()
    {
        var queue=new AudioManager.PcmQueue(8);var data=new float[4];
        queue.Publish(0,new[]{1f,2f},2);queue.Consume(data,1);
        Assert.That(data,Is.EqualTo(new[]{1f,2f,0f,0f}));Assert.That(queue.UnderrunSamples,Is.EqualTo(2));
    }

    [Test, Category("Core")]
    public void QueueLateSamplesKeepTheirOriginalTime()
    {
        var queue=new AudioManager.PcmQueue(8);queue.Consume(new float[2],1);
        queue.Publish(0,new[]{1f,2f,3f,4f},4);var data=new float[4];queue.Consume(data,2);
        Assert.That(data,Is.EqualTo(new[]{3f,3f,4f,4f}));Assert.That(queue.LateSamples,Is.EqualTo(2));
    }

    [Test, Category("Core")]
    public void QueueOverflowPreservesUnreadAudio()
    {
        var queue=new AudioManager.PcmQueue(4);queue.Publish(0,new[]{1f,2f,3f,4f},4);
        Assert.That(queue.Publish(4,new[]{5f},1),Is.False);Assert.That(queue.OverflowSamples,Is.EqualTo(1));
        var data=new float[4];queue.Consume(data,1);Assert.That(data,Is.EqualTo(new[]{1f,2f,3f,4f}));
    }

    [Test, Category("Core")]
    public void CompletedQueueDoesNotCountIdleUnderruns()
    {
        var queue=new AudioManager.PcmQueue(4);queue.Complete();queue.Consume(new float[8],1);
        Assert.That(queue.UnderrunSamples,Is.Zero);
    }

    [Test, Category("Core")]
    public void LongSampleClock()
    {
        long first=1L<<30;
        var signal=new TakeRender.Signal(new[] {1f,-1f}); signal.SetActive(0,true);
        signal.Append(new AcousticTake.VoiceEvent {sample=first,sourceIndex=0,action=AcousticTake.VoiceAction.Play,gain=1,loop=true});
        var mixer=new TakeRender.Mixer(Scene(),1024,new TakeRender.Settings {delayFilter=TakeRender.DelayFilter.Linear},new[] {signal});
        mixer.SetResponse(first,new[] {new TakeRender.DirectPath {delaySeconds=1f/1024,pressureGain=1}},States());
        var output=new float[4]; mixer.Render(first+2,4,output);
        Check(output[0]==-1 && output[1]==1 && output[2]==-1 && output[3]==1,"Long takes preserve single-sample phase beyond float cursor precision.");
    }

    [Test, Category("Core")]
    public void MixerRejectsNonFiniteOutput()
    {
        var output=new float[4];
        var loud=new TakeRender.Signal(new[] {float.MaxValue}); loud.SetActive(0,true);
        loud.Append(new AcousticTake.VoiceEvent {action=AcousticTake.VoiceAction.Play,gain=2,loop=true});
        var overflow=new TakeRender.Mixer(Scene(),1024,new TakeRender.Settings {delayFilter=TakeRender.DelayFilter.Linear},new[] {loud});
        overflow.SetResponse(0,new[] {new TakeRender.DirectPath {pressureGain=1}},States());
        Reject(()=>overflow.Render(0,1,output),"Reject overflowed audio instead of completing a non-finite dataset.");
    }

    [Test, Category("GPU")]
    public void GpuDirectChecks()
    {
        var shader=RequireGpu();
        var scene=Scene(2); var state=States(2); state[0].flags|=AcousticTake.StateFlags.Geometry;
        var geometry=new[] {new AcousticTake.Primitive {a=new Vector3(.5f,-1,-1),b=new Vector3(.5f,1,-1),c=new Vector3(.5f,0,1),objectIndex=0}};
        var cpu=new TakeRender.DirectPath[2]; var gpu=new TakeRender.DirectPath[2]; var settings=new TakeRender.Settings();
        foreach (var dimension in new[] {AcousticTake.Dimension.Planar2D,AcousticTake.Dimension.Spatial3D})
        {
            scene.dimension=dimension;
            using (var solver=new TransportSolver(scene,geometry,settings,shader))
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

    [Test, Category("GPU")]
    public void GpuBatches()
    {
        var shader=RequireGpu();
        var scene=Scene(); var states=new AcousticTake.State[10]; var result=new TakeRender.DirectPath[5];
        for (int f=0;f<5;f++) { var state=States(); state[1].position.x=f+1; Array.Copy(state,0,states,f*2,2); }
        using (var solver=new TransportSolver(scene,new AcousticTake.Primitive[0],new TakeRender.Settings(),shader,5))
        {
            solver.BeginSolve(states,5); Reject(()=>solver.BeginSolve(states,5),"Pending GPU response owns its buffers.");
            Check(solver.TryComplete(result,true),"Explicit shutdown wait completes async readback.");
            for (int f=0;f<5;f++) Near(result[f].pressureGain,1f/(f+1),"Batch responses use their own pose frame",1e-6f);
            solver.BeginSolve(states,5); // Disposal must wait before releasing request-owned buffers.
        }
    }

}

// One short device-clock smoke test; numerical/capture contracts stay in TakeChecks.
public sealed class AudioOutputTests
{
    [UnityEngine.TestTools.UnityTest,Category("Realtime"),Category("Integration")]
    public System.Collections.IEnumerator SilentOutputRecordsRealCallbacks()
    {
        EditorSceneManager.NewScene(NewSceneSetup.EmptyScene,NewSceneMode.Single);
        yield return new UnityEngine.TestTools.EnterPlayMode();
        string path=Path.Combine(Path.GetTempPath(),"acoustics-device-"+Guid.NewGuid().ToString("N"));
        var host=new GameObject("Silent output fixture");bool background=Application.runInBackground;
        try
        {
            Application.runInBackground=true;
            var listener=new GameObject("Listener");listener.transform.SetParent(host.transform);listener.AddComponent<AudioListener>();
            var manager=host.AddComponent<AudioManager>();
            if(Application.isBatchMode)Assert.That(manager.muteOutput,Is.True,"Batch workers start with speaker output suppressed.");
            manager.muteOutput=true;
            int rate=AudioSettings.outputSampleRate;AudioSettings.GetDSPBufferSize(out int frames,out _);
            using var capture=new AudioManager.PlaybackCapture(path,rate,frames*8,frames,0,0,new TakeRender.Settings());
            var pcm=new float[frames*2];for(int i=0;i<pcm.Length;i++)pcm[i]=.1f*Mathf.Sin(i*.1f);
            var queue=new AudioManager.PcmQueue(frames*8,rate,capture);queue.Publish(0,pcm,pcm.Length);queue.Complete();
            manager.StartPcmStream(queue);
            double deadline=Time.realtimeSinceStartupAsDouble+5;
            while(!capture.Closed)
            {
                Assert.That(Time.realtimeSinceStartupAsDouble,Is.LessThan(deadline),"Real audio callbacks did not drain the fixture.");
                yield return null;
            }
            Assert.That(capture.Metadata.complete,Is.True);Assert.That(queue.CallbackCount,Is.GreaterThanOrEqualTo(2));
            Assert.That(queue.UnderrunSamples,Is.Zero);
            using var wave=new BinaryReader(File.OpenRead(Path.Combine(path,"output.wav")));wave.BaseStream.Position=44;
            for(int i=0;i<pcm.Length;i++)Assert.That(wave.ReadSingle(),Is.EqualTo(pcm[i]),"Real callback recording preserves nonzero PCM.");
            using var clock=new BinaryReader(File.OpenRead(Path.Combine(path,"callbacks.bin")));
            clock.BaseStream.Position=32;double first=clock.ReadDouble();clock.BaseStream.Position=72;
            Assert.That(clock.ReadDouble(),Is.GreaterThan(first),"Unity's DSP clock advances between callbacks.");
        }
        finally
        {
            UnityEngine.Object.DestroyImmediate(host);Application.runInBackground=background;
            if(Directory.Exists(path))Directory.Delete(path,true);
        }
    }
    [UnityEngine.TestTools.UnityTearDown]
    public System.Collections.IEnumerator ExitPlayMode()
    {
        if(Application.isPlaying)yield return new UnityEngine.TestTools.ExitPlayMode();
    }
}
