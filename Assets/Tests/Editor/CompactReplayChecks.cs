using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Threading.Tasks;
using UnityEditor;
using UnityEngine;
using UnityEngine.SceneManagement;

// Opt-in experiment. Reconstructs a fixed-tick station run from a small command log.
// Full takes are validation artifacts; this does not change normal recording.
public static class CompactReplayChecks
{
    [MenuItem("Tools/Acoustics/Test compact input replay")]
    static async void RunMenu()
    {
        try { var report = (Report)await Run(); RenderAudio(); Debug.Log("Compact replay report: " + Path.Combine(report.directory,"report.json")); }
        catch (Exception e) { Debug.LogException(e); }
    }
    [MenuItem("Tools/Acoustics/Test compact input replay",true)]
    static bool CanRun() => Application.isPlaying;
    const BindingFlags Private = BindingFlags.Instance | BindingFlags.NonPublic;
    static readonly MethodInfo Advance = typeof(StationDemo).GetMethod("AdvanceTrain", Private);
    static readonly MethodInfo Move = typeof(PlayerController).GetMethod("Move", Private);
    [Serializable] public sealed class Command
    {
        public int tick;
        public string action;
        public Vector2 movement;
        public float value;
        public Command() {}
        public Command(int tick, string action, float x = 0, float y = 0)
        { this.tick = tick; this.action = action; movement = new Vector2(x, y); value = x; }
    }
    [Serializable] public sealed class Recipe
    {
        public string schema = "station-input-replay-experiment-v1";
        public string scene, sceneGuid, sceneRevision, codeModule, driverHash, unityVersion, platform;
        public int rate = 44100, stepSamples = 882, ticks = 2300;
        public Vector3 playerPosition, gravity;
        public Quaternion playerRotation, headRotation;
        public float startDistance, cruise, acceleration, dwell, spacing;
        public Command[] commands;
    }
    [Serializable] public sealed class Comparison
    {
        public int frames, changedStates, groundedTicks, aboardTicks, movingTicks;
        public double maximumPositionMetres, maximumRotationDegrees, maximumVelocityError;
        public double maximumDelaySeconds, maximumGainError;
        public int occlusionMismatches;
        public string stateHash, eventHash;
        public bool byteIdentical;
    }
    [Serializable] public sealed class Report
    {
        public string scope = "Controlled fixed-tick station resimulation, not arbitrary existing recordings or cross-platform determinism";
        public string directory;
        public Recipe recipe;
        public long fullStateBytes, fullStateGzipBytes, commandBytes, compressedCommandBytes, sharedInputBytes;
        public double movementStorageReduction;
        public bool motionPassed, audioPassed;
        public Comparison reference, replay, secondReplay, changedStep;
        public TakeComparison.Metrics audio;
    }
    [Serializable] public sealed class LiveComparison
    {
        public int frames, changedStates;
        public long ticks;
        public double maximumPositionMetres, maximumRotationDegrees, maximumVelocityError;
        public string stateHash,eventHash;
        public bool comparedReference,byteIdentical;
    }
    sealed class Rig : IDisposable
    {
        public readonly Scene scene;
        public readonly StationDemo demo;
        public readonly PlayerController walker;
        public readonly Transform head;
        public readonly List<RayTraceManager.TakeObjectBinding> bindings = new List<RayTraceManager.TakeObjectBinding>();
        public readonly Dictionary<Transform, Transform> copies = new Dictionary<Transform, Transform>();
        readonly Recipe recipe;
        readonly Vector3 doorStart;
        Vector2 movement;
        int nextCommand;
        public int groundedTicks, aboardTicks, movingTicks;
        public Rig(StationDemo source, Recipe recipe,bool includeCrate=true)
        {
            this.recipe = recipe;
            scene = SceneManager.CreateScene("Compact replay " + Guid.NewGuid(), new CreateSceneParameters(LocalPhysicsMode.Physics3D));
            try
            {
                foreach (var root in source.gameObject.scene.GetRootGameObjects()) Copy(root.transform, null);
                foreach (var b in source.manager.takeObjects)
                    bindings.Add(new RayTraceManager.TakeObjectBinding { id = b.id, geometry = b.geometry, transform = copies[b.transform] });
                var route = copies[source.route.transform].gameObject.AddComponent<StationLoop>();
                route.straightLength = source.route.straightLength; route.radius = source.route.radius;
                demo = copies[source.transform].gameObject.AddComponent<StationDemo>(); demo.enabled = false;
                demo.train = copies[source.train]; demo.carriage = copies[source.carriage]; demo.door = copies[source.door];
                demo.player = copies[source.player]; demo.route = route;
                demo.startDistance = recipe.startDistance; demo.trainSpeed = recipe.cruise; demo.trainAcceleration = recipe.acceleration;
                demo.carriageSpacing = recipe.spacing; demo.stationDwell = recipe.dwell;
                demo.doorOpenOffset = source.doorOpenOffset; demo.doorSpeed = source.doorSpeed;
                Set(demo, "trainDistance", (double)recipe.startDistance); Set(demo, "nextStopDistance", (double)recipe.startDistance);
                Set(demo, "<DwellRemaining>k__BackingField", recipe.dwell); demo.PlaceTrain();
                doorStart = source.door.localPosition;
                demo.door.localPosition = doorStart;
                demo.player.SetPositionAndRotation(recipe.playerPosition, recipe.playerRotation);
                head = copies[source.manager.takeListeners[0]]; head.localRotation = recipe.headRotation;
                var look=head.gameObject.AddComponent<MouseLook>(); look.enabled=false; look.playerBody=demo.player; look.lockOnStart=false;
                var controller = demo.player.gameObject.AddComponent<CharacterController>();
                EditorUtility.CopySerialized(source.player.GetComponent<CharacterController>(), controller);
                controller.enabled = true;
                walker = demo.player.gameObject.AddComponent<PlayerController>(); walker.enabled = false;
                var original = source.player.GetComponent<PlayerController>();
                walker.speed = original.speed; walker.gravity = original.gravity; walker.jumpHeight = original.jumpHeight;
                // A real rigid body exercises PhysX integration/contact solving as well as the walking controller.
                if(includeCrate)
                {
                    var crate = New("Replay test crate", null); crate.position = demo.carriage.TransformPoint(new Vector3(0, 2.8f, 1));
                    crate.gameObject.AddComponent<BoxCollider>().size = Vector3.one * .3f;
                    var body = crate.gameObject.AddComponent<Rigidbody>(); body.mass = 2;
                    body.interpolation = RigidbodyInterpolation.None;
                    bindings.Add(new RayTraceManager.TakeObjectBinding { id = "compact-replay-test-crate", geometry = true, transform = crate });
                }
                Physics.SyncTransforms();
            }
            catch { Dispose(); throw; }
        }
        static void Set(object target, string field, object value) => target.GetType().GetField(field, Private).SetValue(target, value);
        Transform New(string name, Transform parent)
        {
            var go = new GameObject(name); SceneManager.MoveGameObjectToScene(go, scene);
            go.transform.SetParent(parent, false); return go.transform;
        }
        void Copy(Transform source, Transform parent)
        {
            var t = New(source.name, parent); copies.Add(source, t);
            t.gameObject.layer=source.gameObject.layer;
            t.localPosition = source.localPosition; t.localRotation = source.localRotation; t.localScale = source.localScale;
            var mesh = source.GetComponent<MeshFilter>(); if (mesh != null) t.gameObject.AddComponent<MeshFilter>().sharedMesh = mesh.sharedMesh;
            var surface = source.GetComponent<AcousticSurface>();
            if (surface != null) EditorUtility.CopySerialized(surface, t.gameObject.AddComponent<AcousticSurface>());
            foreach (var c in source.GetComponents<Collider>())
                if (!(c is CharacterController)) EditorUtility.CopySerialized(c, t.gameObject.AddComponent(c.GetType()));
            if (source.GetComponent<Rigidbody>() != null) throw new InvalidOperationException("Fixture now contains another rigid body; extend the explicit initial-state recipe before testing it.");
            for (int i = 0; i < source.childCount; i++) Copy(source.GetChild(i), t);
            t.gameObject.SetActive(source.gameObject.activeSelf);
        }
        public void Tick(int tick, float dt)
        {
            bool jump = false;
            while (nextCommand < recipe.commands.Length && recipe.commands[nextCommand].tick == tick)
            {
                var c = recipe.commands[nextCommand++];
                switch (c.action)
                {
                    case "move": movement = c.movement; break;
                    case "jump": jump = true; break;
                    case "look": head.localRotation = Quaternion.Euler(0, c.value, 0); break;
                    case "door": demo.doorOpen = c.value != 0; break;
                    case "gain": break; // The same tick-stamped event is written into the acoustic take.
                    default: throw new InvalidDataException("Unknown command " + c.action);
                }
            }
            Advance.Invoke(demo, new object[] { dt });
            demo.door.localPosition = Vector3.MoveTowards(demo.door.localPosition, doorStart + (demo.doorOpen ? demo.doorOpenOffset : Vector3.zero), demo.doorSpeed * dt);
            Physics.SyncTransforms();
            Move.Invoke(walker, new object[] { movement, jump, dt });
            scene.GetPhysicsScene().Simulate(dt);
            if (walker.Ground != null) { groundedTicks++; if (walker.Ground.IsChildOf(demo.carriage)) aboardTicks++; }
            if (demo.CurrentTrainSpeed > 0) movingTicks++;
        }
        public void Dispose()
        {
            if (!scene.IsValid()) return;
            foreach (var root in scene.GetRootGameObjects()) UnityEngine.Object.DestroyImmediate(root);
            SceneManager.UnloadSceneAsync(scene);
        }
    }
    static AcousticTake.State[] Observe(Rig rig, AcousticTake.Scene description, AcousticTake.State[] previous, float dt)
    {
        var result = new AcousticTake.State[rig.bindings.Count];
        for (int i = 0; i < result.Length; i++)
        {
            var flags = rig.bindings[i].geometry ? AcousticTake.StateFlags.Geometry : 0;
            if (description.sources.Any(s => s.objectIndex == i)) flags |= AcousticTake.StateFlags.Source;
            if (description.listeners.Contains(i)) flags |= AcousticTake.StateFlags.Listener;
            result[i] = AcousticTake.State.Observe(rig.bindings[i].transform, flags);
            if (previous != null) result[i].MeasureMotion(previous[i], dt);
        }
        return result;
    }
    static Comparison Simulate(StationDemo source, Recipe recipe, string output, AcousticTake.Scene description,
        AcousticTake.Primitive[] geometry, float[][] pcm, string reference, float stepScale = 1)
    {
        var result = new Comparison(); float dt = (float)((double)recipe.stepSamples / recipe.rate);
        using (var rig = new Rig(source, recipe))
        using (var writer = new AcousticTake.Writer(output, recipe.rate, JsonUtility.FromJson<AcousticTake.Scene>(JsonUtility.ToJson(description)), geometry, pcm))
        using (var original = reference == null ? null : AcousticTake.Open(reference))
        {
            writer.Metadata.observationPolicy = "fixed-tick-experiment";
            writer.Metadata.captureProvenance = AcousticTake.Provenance.Capture();
            var a = new AcousticTake.State[description.objects.Length];
            var low = new TakeRender.DirectPath[description.sources.Length * description.listeners.Length];
            var high = new TakeRender.DirectPath[low.Length];
            var settings = new TakeRender.Settings();
            for (int s = 0; s < description.sources.Length; s++) writer.Append(new AcousticTake.VoiceEvent { sourceIndex = s, action = AcousticTake.VoiceAction.Play, gain = .2f, loop = true });
            foreach (var c in recipe.commands.Where(c => c.action == "gain"))
                writer.Append(new AcousticTake.VoiceEvent { sample = (long)c.tick * recipe.stepSamples, sourceIndex = 0, action = AcousticTake.VoiceAction.Gain, gain = c.value });
            AcousticTake.State[] states = null;
            for (int tick = 0; tick <= recipe.ticks; tick++)
            {
                if (tick > 0) rig.Tick(tick - 1, dt * stepScale);
                states = Observe(rig, description, states, dt);
                long sample = (long)tick * recipe.stepSamples; writer.Append(sample, states); result.frames++;
                if (original == null) continue;
                original.ReadStoredFrame(tick, a);
                for (int i = 0; i < states.Length; i++)
                {
                    double p = Vector3.Distance(states[i].position, a[i].position);
                    // q and -q encode the same rotation; compare the relative quaternion without near-one acos noise.
                    var q = Quaternion.Inverse(a[i].rotation) * states[i].rotation;
                    double angle = 2 * Math.Atan2(Math.Sqrt(q.x*q.x + q.y*q.y + q.z*q.z), Math.Abs(q.w)) * 180 / Math.PI;
                    result.maximumPositionMetres = Math.Max(result.maximumPositionMetres, p);
                    result.maximumRotationDegrees = Math.Max(result.maximumRotationDegrees, angle);
                    result.maximumVelocityError = Math.Max(result.maximumVelocityError, Vector3.Distance(a[i].velocity, states[i].velocity));
                    if (p != 0 || angle > 1e-7 || a[i].flags != states[i].flags) result.changedStates++;
                }
                TakeRender.SolveDirect(description, geometry, states, settings, low);
                TakeRender.SolveDirect(description, geometry, a, settings, high);
                for (int p = 0; p < low.Length; p++)
                {
                    result.maximumDelaySeconds = Math.Max(result.maximumDelaySeconds, Math.Abs(low[p].delaySeconds - high[p].delaySeconds));
                    result.maximumGainError = Math.Max(result.maximumGainError, Math.Abs(low[p].pressureGain - high[p].pressureGain));
                    if ((low[p].pressureGain == 0) != (high[p].pressureGain == 0)) result.occlusionMismatches++;
                }
            }
            writer.Complete();
            result.stateHash = writer.Metadata.stateHash; result.eventHash = writer.Metadata.eventHash;
            result.byteIdentical = original != null && original.Metadata.stateHash == result.stateHash && original.Metadata.eventHash == result.eventHash;
            result.groundedTicks = rig.groundedTicks; result.aboardTicks = rig.aboardTicks; result.movingTicks = rig.movingTicks;
        }
        return result;
    }
    public static async Task<object> Run()
    {
        if (!Application.isPlaying) throw new InvalidOperationException("Enter Play mode in the station to run the isolated compact replay experiment.");
        var source = UnityEngine.Object.FindAnyObjectByType<StationDemo>(); var manager = source.manager;
        if (manager.IsRecordingTake) throw new InvalidOperationException("Save the current take before running the experiment.");
        if (manager.IsReplayingTake) throw new InvalidOperationException("Stop pose replay before running the experiment.");
        bool wasLive = manager.IsLiveSimulation;
        manager.StopLiveSimulation(); while (manager.Mode == RayTraceManager.SessionMode.Draining) await Task.Delay(20);
        var walker = source.player.GetComponent<PlayerController>(); var look = source.player.GetComponentInChildren<MouseLook>();
        bool sourceEnabled = source.enabled, walkerEnabled = walker.enabled, lookEnabled = look.enabled;
        source.enabled = walker.enabled = look.enabled = false;
        try
        {
            string output = Path.GetFullPath(Path.Combine("Recordings", "compact-experiment-" + DateTime.UtcNow.ToString("yyyyMMdd-HHmmss-fff")));
            Directory.CreateDirectory(output);
            var recipe = new Recipe { scene = source.gameObject.scene.path, sceneGuid = AssetDatabase.AssetPathToGUID(source.gameObject.scene.path),
                sceneRevision = AssetDatabase.GetAssetDependencyHash(source.gameObject.scene.path).ToString(),
                codeModule = typeof(StationDemo).Assembly.ManifestModule.ModuleVersionId.ToString(), unityVersion = Application.unityVersion,
                driverHash = AcousticTake.HashFile("Assets/Tests/Editor/CompactReplayChecks.cs"),
                platform = Application.platform + " / " + SystemInfo.processorType, gravity = Physics.gravity,
                playerPosition = new Vector3(4, 0, -20), playerRotation = Quaternion.identity, headRotation = Quaternion.identity,
                startDistance = source.startDistance, cruise = source.trainSpeed, acceleration = source.trainAcceleration,
                spacing = source.carriageSpacing, dwell = source.stationDwell,
                commands = new[] { new Command(0,"move",0,1), new Command(57,"move",1,0), new Command(152,"move"),
                    new Command(300,"look",65),new Command(400,"look",-45),new Command(450,"look"),
                    new Command(560,"jump"),new Command(650,"gain",.1f),new Command(800,"door",1),new Command(950,"door"),
                    new Command(1800,"move",-1,0),new Command(1857,"move") } };
            string input = Path.Combine(output,"input.json"); File.WriteAllText(input, JsonUtility.ToJson(recipe,true));
            var report = new Report { directory = output, recipe = recipe, commandBytes = new FileInfo(input).Length };
            // Replay reads the persisted input, not a retained command sequence in memory.
            recipe = JsonUtility.FromJson<Recipe>(File.ReadAllText(input));
            AcousticTake.Scene description; AcousticTake.Primitive[] geometry; float[][] pcm;
            using (var rig = new Rig(source, recipe))
            {
                var clips = manager.takeSources.Select(s => s.clip).ToArray();
                description = SceneToTake.Compile(rig.bindings, manager.takeDimension, manager.takeCurveTolerance, manager.takeDefaultMaterial,
                    manager.takeSources.Select(s => rig.copies[s.transform]).ToArray(), clips,
                    manager.takeListeners.Select(l => rig.copies[l]).ToArray(), out geometry);
                pcm = new float[description.clips.Length][];
                for (int s = 0; s < clips.Length; s++) pcm[description.sources[s].clipIndex] = (float[])typeof(RayTraceManager).GetMethod("LoadSample",Private).Invoke(manager,new object[]{clips[s]});
                recipe.rate = manager.sampleRate; recipe.stepSamples = recipe.rate / 50;
                if (recipe.rate % 50 != 0) throw new InvalidOperationException("Fixture requires an integral 50Hz step at the take rate.");
                File.WriteAllText(input,JsonUtility.ToJson(recipe,true));
                report.commandBytes = new FileInfo(input).Length;
            }
            string reference = Path.Combine(output,"reference");
            report.reference = Simulate(source,recipe,reference,description,geometry,pcm,null);
            await Task.Yield();
            recipe = JsonUtility.FromJson<Recipe>(File.ReadAllText(input));
            report.replay = Simulate(source,recipe,Path.Combine(output,"replayed"),description,geometry,pcm,reference);
            await Task.Yield();
            report.secondReplay = Simulate(source,recipe,Path.Combine(output,"replayed-again"),description,geometry,pcm,reference);
            // Negative control: same command ticks with a 1% wrong physics step must be detectable.
            report.changedStep = Simulate(source,recipe,Path.Combine(output,"wrong-step"),description,geometry,pcm,reference,1.01f);
            report.fullStateBytes = new FileInfo(Path.Combine(reference,"states.bin")).Length;
            report.sharedInputBytes = new FileInfo(Path.Combine(reference,"geometry.bin")).Length + pcm.Sum(p => (long)p.Length * 4);
            Compress(input,Path.Combine(output,"input.json.gz"));
            Compress(Path.Combine(reference,"states.bin"),Path.Combine(output,"reference-states.bin.gz"));
            report.fullStateGzipBytes = new FileInfo(Path.Combine(output,"reference-states.bin.gz")).Length;
            report.compressedCommandBytes = new FileInfo(Path.Combine(output,"input.json.gz")).Length;
            report.movementStorageReduction = (double)report.fullStateBytes / report.commandBytes;
            report.motionPassed = report.replay.byteIdentical && report.secondReplay.byteIdentical &&
                report.reference.aboardTicks > 0 && report.reference.movingTicks > 0 && !report.changedStep.byteIdentical;
            File.WriteAllText(Path.Combine(output,"report.json"),JsonUtility.ToJson(report,true));
            Directory.CreateDirectory(".context/compact-replay"); File.WriteAllText(".context/compact-replay/latest.txt",output);
            return report;
        }
        finally
        {
            source.enabled = sourceEnabled; walker.enabled = walkerEnabled; look.enabled = lookEnabled; source.SetWalking(false);
            if (wasLive && Application.isPlaying && manager.Mode == RayTraceManager.SessionMode.Idle)
            { manager.BeginLiveSimulation(); source.horn?.Play(); source.roomSignal?.Play(); }
        }
    }
    static void Compress(string input,string output)
    {
        using (var destination = File.Create(output))
        using (var gzip = new System.IO.Compression.GZipStream(destination,System.IO.Compression.CompressionLevel.Optimal))
        using (var data = File.OpenRead(input)) data.CopyTo(gzip);
    }
    [MenuItem("Tools/Acoustics/Verify recorded station inputs")]
    static async void VerifyMenu()
    {
        string path=EditorUtility.OpenFolderPanel("Select a station take with input.json","Recordings","");
        if(string.IsNullOrEmpty(path))return;
        try { Debug.Log(await VerifyLive(path)); } catch(Exception e) { Debug.LogException(e); }
    }
    [MenuItem("Tools/Acoustics/Verify recorded station inputs",true)]
    static bool CanVerify() => Application.isPlaying;

    // Full observations are used only for validation and shared scene/audio data, never to correct the simulation.
    public static async Task<string> VerifyLive(string path)
    {
        if(!Application.isPlaying)throw new InvalidOperationException("Enter Play mode in the recorded scene first.");
        var source=UnityEngine.Object.FindAnyObjectByType<StationDemo>(); var manager=source.manager;
        if(manager.IsRecordingTake || manager.IsReplayingTake || source.IsInputReplay)throw new InvalidOperationException("Finish recording/replay first.");
        using var playback=new StationRecording.Playback(path); playback.ValidateScene(source);
        var meta=playback.Metadata;
        bool wasLive=manager.IsLiveSimulation;
        manager.StopLiveSimulation(); while(manager.Mode==RayTraceManager.SessionMode.Draining)await Task.Delay(20);
        try
        {
            var authoredScene=UnityEditor.SceneManagement.EditorSceneManager.OpenPreviewScene(meta.scene);
            try
            {
                var authored=authoredScene.GetRootGameObjects().SelectMany(r=>r.GetComponentsInChildren<StationDemo>(true)).Single();
                using var take=File.Exists(Path.Combine(path,"states.bin"))?AcousticTake.Open(path):null;
                var description=playback.TakeMetadata.scene;
                var recipe=new Recipe { playerPosition=authored.player.position,playerRotation=authored.player.rotation,
                    headRotation=authored.manager.takeListeners[0].localRotation,startDistance=meta.settings.startDistance,cruise=meta.settings.cruise,
                    acceleration=meta.settings.acceleration,dwell=meta.settings.dwell,spacing=meta.settings.spacing,commands=new Command[0] };
                var result=new LiveComparison {ticks=meta.ticks,comparedReference=take!=null}; string output=Path.Combine(path,"input-replay-"+DateTime.UtcNow.ToString("yyyyMMdd-HHmmss-fff"));
                using(var rig=new Rig(authored,recipe,false))
                {
                    var pcm=new float[description.clips.Length][];
                    AcousticTake.Primitive[] geometry;
                    if(take!=null) { geometry=take.Geometry; for(int i=0;i<pcm.Length;i++)pcm[i]=take.ReadClip(i); }
                    else
                    {
                        var m=authored.manager;var clips=m.takeSources.Select(s=>s.clip).ToArray();m.sampleRate=playback.TakeMetadata.sampleRate;
                        description=SceneToTake.Compile(rig.bindings,m.takeDimension,m.takeCurveTolerance,m.takeDefaultMaterial,
                            m.takeSources.Select(s=>rig.copies[s.transform]).ToArray(),clips,m.takeListeners.Select(l=>rig.copies[l]).ToArray(),out geometry);
                        for(int s=0;s<clips.Length;s++)pcm[description.sources[s].clipIndex]=(float[])typeof(RayTraceManager).GetMethod("LoadSample",Private).Invoke(m,new object[]{clips[s]});
                    }
                    using var writer=new AcousticTake.Writer(output,playback.TakeMetadata.sampleRate,
                        JsonUtility.FromJson<AcousticTake.Scene>(JsonUtility.ToJson(description)),geometry,pcm);
                    if(writer.Metadata.sceneHash!=playback.TakeMetadata.sceneHash)throw new InvalidDataException("Authored geometry/source audio differs from the recording.");
                    writer.Metadata.observationPolicy="recorded-input-resimulation";
                    meta.settings.Apply(rig.demo); rig.demo.InitializeSimulation();
                    foreach(var e in playback.Events)writer.Append(e);
                    var expected=new AcousticTake.State[description.objects.Length];
                    AcousticTake.State[] previous=null; long simulated=0,previousSample=-1;
                    for(int f=0;playback.NextObservation(long.MaxValue,out long sample,out long tick);f++)
                    {
                        while(simulated<tick)
                        {
                            rig.demo.Step(playback.Read(simulated),meta.stepSeconds); rig.scene.GetPhysicsScene().Simulate(meta.stepSeconds); simulated++;
                        }
                        var state=Observe(rig,description,null,0);
                        if(previous!=null)for(int i=0;i<state.Length;i++)state[i].MeasureMotion(previous[i],(double)(sample-previousSample)/playback.TakeMetadata.sampleRate);
                        writer.Append(sample,state); result.frames++;
                        if(take!=null)
                        {
                            if(sample!=take.FrameSample(f))throw new InvalidDataException("Reference observation time mismatch.");
                            take.ReadStoredFrame(f,expected);
                            for(int i=0;i<state.Length;i++)
                            {
                                double position=Vector3.Distance(state[i].position,expected[i].position);
                                var q=Quaternion.Inverse(expected[i].rotation)*state[i].rotation;
                                double angle=2*Math.Atan2(Math.Sqrt(q.x*q.x+q.y*q.y+q.z*q.z),Math.Abs(q.w))*180/Math.PI;
                                result.maximumPositionMetres=Math.Max(result.maximumPositionMetres,position);
                                result.maximumRotationDegrees=Math.Max(result.maximumRotationDegrees,angle);
                                result.maximumVelocityError=Math.Max(result.maximumVelocityError,Vector3.Distance(state[i].velocity,expected[i].velocity));
                                if(position!=0 || angle>1e-7 || state[i].flags!=expected[i].flags)result.changedStates++;
                            }
                        }
                        previous=state; previousSample=sample;
                    }
                    if(simulated!=meta.ticks)throw new InvalidDataException("Unconsumed input ticks.");
                    writer.Complete(); result.stateHash=writer.Metadata.stateHash; result.eventHash=writer.Metadata.eventHash;
                    result.byteIdentical=take!=null && result.stateHash==take.Metadata.stateHash && result.eventHash==take.Metadata.eventHash;
                }
                File.WriteAllText(Path.Combine(output,"verification.json"),JsonUtility.ToJson(result,true)); return output;
            }
            finally { UnityEditor.SceneManagement.EditorSceneManager.ClosePreviewScene(authoredScene); }
        }
        finally { if(wasLive && manager.Mode==RayTraceManager.SessionMode.Idle) { manager.BeginLiveSimulation(); source.horn?.Play(); source.roomSignal?.Play(); } }
    }
    public static object RenderAudio()
    {
        string output = File.ReadAllText(".context/compact-replay/latest.txt").Trim();
        var report = JsonUtility.FromJson<Report>(File.ReadAllText(Path.Combine(output,"report.json")));
        var settings = new TakeRender.Settings { responseStepSamples = report.recipe.stepSamples, listenerOutput = TakeRender.ListenerOutput.Binaural };
        foreach (string name in new[]{"reference","replayed"})
            if (!Directory.Exists(Path.Combine(output,name,"audio")))
                using (var take = AcousticTake.Open(Path.Combine(output,name))) TakeRender.RenderDirect(take,settings,Path.Combine(output,name,"audio"));
        report.audio = TakeComparison.CompareWave(Path.Combine(output,"reference","audio","listener-0.wav"),Path.Combine(output,"replayed","audio","listener-0.wav"));
        report.audioPassed = report.audio.exactMatch && report.audio.referenceRms > 0;
        File.WriteAllText(Path.Combine(output,"report.json"),JsonUtility.ToJson(report,true));
        return report.audio;
    }
}
