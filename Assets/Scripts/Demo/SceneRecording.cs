using System;
using System.IO;
using UnityEngine;

// Compact scene inputs and the original simulation/audio clock mapping.
public sealed class SceneRecording : IDisposable
{
    public static void ExportInputs(string directory,string output)
    {
        using(var input=new Playback(directory))
        {
            long consumed=0;
            while(input.NextObservation(long.MaxValue,out _,out long tick))while(consumed<tick)input.Read(consumed++);
        }
        if(Directory.Exists(output))throw new IOException("Choose a new output directory.");
        Directory.CreateDirectory(output);
        foreach(string name in new[]{"take.json","events.bin","input.bin","input-observations.bin","input.json"})
            File.Copy(Path.Combine(directory,name),Path.Combine(output,name));
        AcousticTake.CopyStreams(directory,output,AcousticTake.ReadManifest(directory));
    }
    [Serializable] public struct Input
    {
        public Vector2 movement,look;
        public bool jump,interact;
    }
    [Serializable] public struct Settings
    {
        public float cruise,acceleration,dwell,spacing,doorSpeed,walkSpeed,gravity,jumpHeight,startDistance;
        public Vector3 doorOffset;
        // Legacy station-v2 headers remain readable; replay still requires the original project revision.
        public static Settings From(SceneSession session)
        {
            var train=session.GetComponent<StationDemo>();
            var door=Array.Find(session.actors,a=>a is SlidingDoor) as SlidingDoor;
            var w=session.player.GetComponent<PlayerController>();
            return new Settings {cruise=train.trainSpeed,acceleration=train.trainAcceleration,dwell=train.stationDwell,
                spacing=train.carriageSpacing,startDistance=train.startDistance,doorSpeed=door.speed,doorOffset=door.openOffset,
                walkSpeed=w.speed,gravity=w.gravity,jumpHeight=w.jumpHeight};
        }
    }
    [Serializable] public struct PlayerSettings
    {
        public float speed,gravity,jumpHeight;
        public static PlayerSettings From(SceneSession session)
        {
            var w=session.player.GetComponent<PlayerController>();
            return new PlayerSettings {speed=w.speed,gravity=w.gravity,jumpHeight=w.jumpHeight};
        }
    }
    [Serializable] public struct ActorSettings
    {
        public string id,kind,settings;
        public int version;
        public static ActorSettings From(SceneActor actor) => new ActorSettings {id=actor.id,kind=actor.GetType().Name,version=1,settings=actor.SettingsJson};
    }
    [Serializable] public sealed class Manifest
    {
        public string schema="scene-input-from-init-v1",scene,sceneGuid,sceneRevision,codeModule,unityVersion,platform;
        public string takeHash,inputHash,observationHash;
        public string gitCommit,projectRevision;
        public bool gitModified;
        public float stepSeconds;
        public long ticks;
        public int inputCount,observationCount;
        public bool complete;
        public Settings settings; // Read-only station-v2 compatibility.
        public PlayerSettings player;
        public ActorSettings[] actors=Array.Empty<ActorSettings>();
    }
    readonly SceneSession session;
    readonly SceneActor[] actors;
    readonly string directory;
    readonly BinaryWriter input,observations;
    readonly Manifest manifest;
    readonly (Transform transform,bool active,AcousticSurface surface,bool surfaceEnabled,RaytracedAudioSource source,bool sourceEnabled)[] activation;
    Vector2 heldMovement;
    bool closed;
    public SceneRecording(SceneSession session,string directory)
    {
        this.session=session; actors=(SceneActor[])session.actors.Clone(); this.directory=directory;
        activation=new (Transform,bool,AcousticSurface,bool,RaytracedAudioSource,bool)[session.manager.takeObjects.Count];
        for(int i=0;i<activation.Length;i++)
        {
            var transform=session.manager.takeObjects[i].transform;
            var surface=transform.GetComponent<AcousticSurface>();var source=transform.GetComponent<RaytracedAudioSource>();
            activation[i]=(transform,transform.gameObject.activeInHierarchy,surface,surface!=null && surface.enabled,source,source!=null && source.enabled);
        }
        if(session.Tick!=0) throw new InvalidOperationException("Input capture must start at scene initialization; re-enter Play for a new run.");
        foreach(var root in session.gameObject.scene.GetRootGameObjects())
            if(root.GetComponentsInChildren<Rigidbody>(true).Length>0)
                throw new NotSupportedException("Compact scene input supports explicitly stepped actors and the walking controller; arbitrary rigid bodies require full capture.");
        manifest=new Manifest { scene=session.gameObject.scene.path,codeModule=typeof(SceneSession).Assembly.ManifestModule.ModuleVersionId.ToString(),
            unityVersion=Application.unityVersion,platform=Application.platform+" / "+SystemInfo.processorType,
            stepSeconds=Time.fixedDeltaTime,player=PlayerSettings.From(session),actors=Array.ConvertAll(actors,ActorSettings.From) };
#if UNITY_EDITOR
        manifest.sceneGuid=UnityEditor.AssetDatabase.AssetPathToGUID(manifest.scene);
        manifest.sceneRevision=UnityEditor.AssetDatabase.GetAssetDependencyHash(manifest.scene).ToString();
        manifest.projectRevision=HashProject(Path.GetFullPath(Path.Combine(Application.dataPath,"..")));
        manifest.gitCommit=Git("rev-parse HEAD");
        manifest.gitModified=manifest.gitCommit==null || Git("status --porcelain --untracked-files=normal -- Assets Packages ProjectSettings")!="";
#endif
        File.WriteAllText(Path.Combine(directory,"input.json"),JsonUtility.ToJson(manifest,true));
        try
        {
            input=new BinaryWriter(File.Open(Path.Combine(directory,"input.bin"),FileMode.CreateNew));
            observations=new BinaryWriter(File.Open(Path.Combine(directory,"input-observations.bin"),FileMode.CreateNew));
        }
        catch { input?.Dispose(); observations?.Dispose(); throw; }
    }
    public void BeforeTick(long tick,Input value,float seconds)
    {
        if(closed) throw new ObjectDisposedException(nameof(SceneRecording));
        if(seconds!=manifest.stepSeconds || !manifest.player.Equals(PlayerSettings.From(session)))
            throw new InvalidOperationException("Player motion settings changed during input recording; start a new run.");
        if(session.actors.Length!=actors.Length) throw new InvalidOperationException("Scene actor roster changed during input recording.");
        for(int i=0;i<actors.Length;i++)
            if(actors[i]==null || actors[i]!=session.actors[i] || actors[i].id!=manifest.actors[i].id ||
                !actors[i].isActiveAndEnabled || !actors[i].SettingsUnchanged)
                throw new InvalidOperationException("Scene actor order, activity or settings changed during input recording.");
        if(tick!=manifest.ticks) throw new InvalidOperationException("Scene input ticks must be contiguous from zero.");
        if(manifest.inputCount==0 || !value.movement.Equals(heldMovement) || !value.look.Equals(Vector2.zero) || value.jump || value.interact)
        {
            input.Write(manifest.ticks); input.Write(value.movement.x); input.Write(value.movement.y);
            input.Write(value.look.x); input.Write(value.look.y); input.Write((byte)((value.jump?1:0)|(value.interact?2:0)));
            heldMovement=value.movement; manifest.inputCount++;
        }
        manifest.ticks++;
    }
    public void Observe(long sample,long tick)
    {
        foreach(var value in activation)
            if(value.transform==null || value.transform.gameObject.activeInHierarchy!=value.active ||
                (value.surface!=null && value.surface.enabled!=value.surfaceEnabled) || (value.source!=null && value.source.enabled!=value.sourceEnabled))
                throw new InvalidOperationException("Compact scene inputs do not record activation changes; start a new scene run with the desired initial activity.");
        observations.Write(sample); observations.Write(tick); manifest.observationCount++;
    }
    public void Complete(bool valid)
    {
        Dispose(); manifest.complete=valid;
        manifest.takeHash=AcousticTake.HashFile(Path.Combine(directory,"take.json"));
        manifest.inputHash=AcousticTake.HashFile(Path.Combine(directory,"input.bin"));
        manifest.observationHash=AcousticTake.HashFile(Path.Combine(directory,"input-observations.bin"));
        string path=Path.Combine(directory,"input.json"),temp=path+".tmp";
        File.WriteAllText(temp,JsonUtility.ToJson(manifest,true)); File.Replace(temp,path,null);
    }
    public void Dispose() { if(closed)return; closed=true; input?.Dispose(); observations?.Dispose(); }
#if UNITY_EDITOR
    // Compiler module IDs can change when identical sources are rebuilt in another checkout.
    // Hash project inputs with relative paths; Library and machine-local state are excluded.
    public static string HashProject(string root)
    {
        var files=new System.Collections.Generic.List<string>();
        foreach(string folder in new[]{"Assets","Packages","ProjectSettings"})
            foreach(string file in Directory.GetFiles(Path.Combine(root,folder),"*",SearchOption.AllDirectories))
            {
                string relative=Path.GetRelativePath(root,file).Replace('\\','/');
                if(relative.StartsWith("Assets/Resources/NetworkBuildStamp.txt",StringComparison.Ordinal) || relative.StartsWith("Assets/Resources/PerformanceTestRun",StringComparison.Ordinal))continue;
                files.Add(relative);
            }
        files.Sort(StringComparer.Ordinal);
        using var bytes=new MemoryStream();
        using(var writer=new BinaryWriter(bytes,System.Text.Encoding.UTF8,true))
            foreach(string file in files) { writer.Write(file);writer.Write(AcousticTake.HashFile(Path.Combine(root,file))); }
        using var hash=System.Security.Cryptography.SHA256.Create();
        return Convert.ToBase64String(hash.ComputeHash(bytes.ToArray()));
    }
    static string Git(string arguments)
    {
        try
        {
            var info=new System.Diagnostics.ProcessStartInfo("git",arguments) {WorkingDirectory=Path.GetFullPath(Path.Combine(Application.dataPath,"..")),
                UseShellExecute=false,CreateNoWindow=true,RedirectStandardOutput=true,RedirectStandardError=true};
            using(var process=System.Diagnostics.Process.Start(info))
            {
                var output=process.StandardOutput.ReadToEndAsync();var error=process.StandardError.ReadToEndAsync();
                if(!process.WaitForExit(2000)){process.Kill();return null;}
                return process.ExitCode==0?output.GetAwaiter().GetResult().Trim():null;
            }
        }
        catch { return null; } // Recording still works outside a Git checkout.
    }
#endif
    // Sequential input playback. Held axes persist; deltas and buttons last exactly one tick.
    public sealed class Playback : IDisposable
    {
        public readonly Manifest Metadata;
        public readonly AcousticTake.Manifest TakeMetadata;
        public readonly AcousticTake.VoiceEvent[] Events;
        readonly BinaryReader reader,observations;
        long nextTick,readTick;
        long nextSample=-1,observedTick,previousSample=-1,previousTick;
        int observationIndex;
        Vector2 movement;
        public Playback(string directory)
        {
            Metadata=JsonUtility.FromJson<Manifest>(File.ReadAllText(Path.Combine(directory,"input.json")));
            if(Metadata==null || (Metadata.schema!="station-input-from-init-v2" && Metadata.schema!="scene-input-from-init-v1") || !Metadata.complete ||
                !float.IsFinite(Metadata.stepSeconds) || Metadata.stepSeconds<=0 || Metadata.ticks<0 || Metadata.inputCount<0 || Metadata.observationCount<=0)
                throw new InvalidDataException("Select a completed recording made from scene initialization.");
            foreach(var pair in new[]{("take.json",Metadata.takeHash),("input.bin",Metadata.inputHash),("input-observations.bin",Metadata.observationHash)})
                if(AcousticTake.HashFile(Path.Combine(directory,pair.Item1))!=pair.Item2)throw new InvalidDataException("Input checksum mismatch: "+pair.Item1);
            if(new FileInfo(Path.Combine(directory,"input.bin")).Length!=25L*Metadata.inputCount ||
                new FileInfo(Path.Combine(directory,"input-observations.bin")).Length!=16L*Metadata.observationCount)
                throw new InvalidDataException("Input counts do not match payload lengths.");
            try
            {
                TakeMetadata=AcousticTake.ReadManifest(directory);AcousticTake.ValidateStreams(directory,TakeMetadata);Events=AcousticTake.ReadEvents(directory,TakeMetadata);
                if(Metadata.observationCount!=TakeMetadata.frameCount) throw new InvalidDataException("Observation count mismatch.");
                reader=new BinaryReader(File.OpenRead(Path.Combine(directory,"input.bin")));
                observations=new BinaryReader(File.OpenRead(Path.Combine(directory,"input-observations.bin")));
                nextTick=Metadata.inputCount>0?reader.ReadInt64():long.MaxValue;
                if((Metadata.ticks>0 && nextTick!=0) || (Metadata.ticks==0 && Metadata.inputCount!=0))
                    throw new InvalidDataException("Input must begin at tick zero.");
            }
            catch { Dispose(); throw; }
        }
        public void ValidateScene(SceneSession session,bool liveOutput=true)
        {
            bool matchesProject=Metadata.codeModule==typeof(SceneSession).Assembly.ManifestModule.ModuleVersionId.ToString();
#if UNITY_EDITOR
            matchesProject=string.IsNullOrEmpty(Metadata.projectRevision)
                ? matchesProject && Metadata.sceneRevision==UnityEditor.AssetDatabase.GetAssetDependencyHash(Metadata.scene).ToString()
                : Metadata.projectRevision==HashProject(Path.GetFullPath(Path.Combine(Application.dataPath,"..")));
#endif
            if(Metadata.scene!=session.gameObject.scene.path || !matchesProject ||
                Metadata.unityVersion!=Application.unityVersion || Metadata.platform!=Application.platform+" / "+SystemInfo.processorType)
                throw new InvalidDataException("Recorded scene/code/engine/platform differs from the current project.");
            bool settingsMatch=Metadata.schema=="station-input-from-init-v2"
                ? session.GetComponent<StationDemo>()!=null && Metadata.settings.Equals(Settings.From(session))
                : Metadata.player.Equals(PlayerSettings.From(session)) && Metadata.actors!=null && Metadata.actors.Length==session.actors.Length;
            if(settingsMatch && Metadata.schema=="scene-input-from-init-v1")
                for(int i=0;i<Metadata.actors.Length;i++) settingsMatch&=Metadata.actors[i].Equals(ActorSettings.From(session.actors[i]));
            if(!settingsMatch || Time.fixedDeltaTime!=Metadata.stepSeconds ||
                (liveOutput && AudioSettings.outputSampleRate!=TakeMetadata.sampleRate))
                throw new InvalidDataException("Restore the recorded motion settings, fixed step and output sample rate before replaying.");
        }
        public Input Read(long tick)
        {
            if(tick!=readTick || tick>=Metadata.ticks)throw new InvalidOperationException("Input replay is sequential from tick zero.");
            var value=new Input { movement=movement };
            if(nextTick==tick)
            {
                value.movement=movement=new Vector2(reader.ReadSingle(),reader.ReadSingle());
                value.look=new Vector2(reader.ReadSingle(),reader.ReadSingle()); byte flags=reader.ReadByte();
                if(flags>3 || !float.IsFinite(movement.x) || !float.IsFinite(movement.y) || !float.IsFinite(value.look.x) || !float.IsFinite(value.look.y))
                    throw new InvalidDataException("Invalid input state.");
                value.jump=(flags&1)!=0; value.interact=(flags&2)!=0;
                nextTick=reader.BaseStream.Position<reader.BaseStream.Length?reader.ReadInt64():long.MaxValue;
                if(nextTick!=long.MaxValue && (nextTick<=tick || nextTick>=Metadata.ticks))throw new InvalidDataException("Unordered input ticks.");
            }
            readTick++; return value;
        }
        public bool NextObservation(long untilSample,out long sample,out long tick)
        {
            sample=tick=0;
            if(nextSample<0)
            {
                if(observationIndex==Metadata.observationCount)return false;
                nextSample=observations.ReadInt64();observedTick=observations.ReadInt64();
                if(nextSample<=previousSample || nextSample>TakeMetadata.durationSamples || observedTick<previousTick || observedTick>Metadata.ticks ||
                    (observationIndex==0 && (nextSample!=0 || observedTick!=0)) ||
                    (observationIndex==Metadata.observationCount-1 && (observedTick!=Metadata.ticks || nextSample!=TakeMetadata.durationSamples)))
                    throw new InvalidDataException("Invalid observation clock mapping.");
            }
            if(nextSample>untilSample)return false;
            sample=previousSample=nextSample;tick=previousTick=observedTick;nextSample=-1;observationIndex++;return true;
        }
        public void Dispose() { reader?.Dispose(); observations?.Dispose(); }
    }
}
