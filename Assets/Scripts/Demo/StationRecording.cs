using System;
using System.IO;
using UnityEngine;

// Station input sidecar. The normal take remains the reference until a fresh simulation verifies it.
public sealed class StationRecording : IDisposable
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
        public static Settings From(StationDemo d) => new Settings { cruise=d.trainSpeed,acceleration=d.trainAcceleration,
            dwell=d.stationDwell,spacing=d.carriageSpacing,doorSpeed=d.doorSpeed,doorOffset=d.doorOpenOffset,
            walkSpeed=d.player.GetComponent<PlayerController>().speed,gravity=d.player.GetComponent<PlayerController>().gravity,jumpHeight=d.player.GetComponent<PlayerController>().jumpHeight,startDistance=d.startDistance };
        public void Apply(StationDemo d)
        {
            d.trainSpeed=cruise; d.trainAcceleration=acceleration; d.stationDwell=dwell; d.carriageSpacing=spacing;
            d.doorSpeed=doorSpeed; d.doorOpenOffset=doorOffset;
            d.startDistance=startDistance;
            var w=d.player.GetComponent<PlayerController>(); w.speed=walkSpeed; w.gravity=gravity; w.jumpHeight=jumpHeight;
        }
    }
    [Serializable] public sealed class Manifest
    {
        public string schema="station-input-from-init-v2",scene,sceneGuid,sceneRevision,codeModule,unityVersion,platform;
        public string takeHash,inputHash,observationHash;
        public string gitCommit;
        public bool gitModified;
        public float stepSeconds;
        public long ticks;
        public int inputCount,observationCount;
        public bool complete;
        public Settings settings;
    }
    readonly StationDemo station;
    readonly string directory;
    readonly BinaryWriter input,observations;
    readonly Manifest manifest;
    Vector2 heldMovement;
    bool closed;
    public StationRecording(StationDemo station,string directory)
    {
        this.station=station; this.directory=directory;
        if(station.Tick!=0) throw new InvalidOperationException("Input capture must start at scene initialization; re-enter Play for a new run.");
        foreach(var root in station.gameObject.scene.GetRootGameObjects())
            if(root.GetComponentsInChildren<Rigidbody>(true).Length>0)
                throw new NotSupportedException("The station input adapter supports its train and walking controller; arbitrary rigid bodies still require full capture.");
        manifest=new Manifest { scene=station.gameObject.scene.path,codeModule=typeof(StationDemo).Assembly.ManifestModule.ModuleVersionId.ToString(),
            unityVersion=Application.unityVersion,platform=Application.platform+" / "+SystemInfo.processorType,
            stepSeconds=Time.fixedDeltaTime,settings=Settings.From(station) };
#if UNITY_EDITOR
        manifest.sceneGuid=UnityEditor.AssetDatabase.AssetPathToGUID(manifest.scene);
        manifest.sceneRevision=UnityEditor.AssetDatabase.GetAssetDependencyHash(manifest.scene).ToString();
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
        if(closed) throw new ObjectDisposedException(nameof(StationRecording));
        if(seconds!=manifest.stepSeconds || !manifest.settings.Equals(Settings.From(station)))
            throw new InvalidOperationException("Station motion settings changed during input recording; use the full take for this run.");
        if(tick!=manifest.ticks) throw new InvalidOperationException("Station input ticks must be contiguous from zero.");
        if(manifest.inputCount==0 || !value.movement.Equals(heldMovement) || value.look!=Vector2.zero || value.jump || value.interact)
        {
            input.Write(manifest.ticks); input.Write(value.movement.x); input.Write(value.movement.y);
            input.Write(value.look.x); input.Write(value.look.y); input.Write((byte)((value.jump?1:0)|(value.interact?2:0)));
            heldMovement=value.movement; manifest.inputCount++;
        }
        manifest.ticks++;
    }
    public void Observe(long sample,long tick)
    {
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
            if(Metadata==null || Metadata.schema!="station-input-from-init-v2" || !Metadata.complete ||
                !float.IsFinite(Metadata.stepSeconds) || Metadata.stepSeconds<=0 || Metadata.ticks<0 || Metadata.inputCount<0 || Metadata.observationCount<=0)
                throw new InvalidDataException("Select a completed recording made from scene initialization.");
            foreach(var pair in new[]{("take.json",Metadata.takeHash),("input.bin",Metadata.inputHash),("input-observations.bin",Metadata.observationHash)})
                if(AcousticTake.HashFile(Path.Combine(directory,pair.Item1))!=pair.Item2)throw new InvalidDataException("Input checksum mismatch: "+pair.Item1);
            if(new FileInfo(Path.Combine(directory,"input.bin")).Length!=25L*Metadata.inputCount ||
                new FileInfo(Path.Combine(directory,"input-observations.bin")).Length!=16L*Metadata.observationCount)
                throw new InvalidDataException("Input counts do not match payload lengths.");
            try
            {
                TakeMetadata=AcousticTake.ReadManifest(directory);Events=AcousticTake.ReadEvents(directory,TakeMetadata);
                if(Metadata.observationCount!=TakeMetadata.frameCount) throw new InvalidDataException("Observation count mismatch.");
                reader=new BinaryReader(File.OpenRead(Path.Combine(directory,"input.bin")));
                observations=new BinaryReader(File.OpenRead(Path.Combine(directory,"input-observations.bin")));
                nextTick=Metadata.inputCount>0?reader.ReadInt64():long.MaxValue;
                if((Metadata.ticks>0 && nextTick!=0) || (Metadata.ticks==0 && Metadata.inputCount!=0))
                    throw new InvalidDataException("Input must begin at tick zero.");
            }
            catch { Dispose(); throw; }
        }
        public void ValidateScene(StationDemo station,bool liveOutput=true)
        {
            if(Metadata.scene!=station.gameObject.scene.path || Metadata.codeModule!=typeof(StationDemo).Assembly.ManifestModule.ModuleVersionId.ToString() ||
                Metadata.unityVersion!=Application.unityVersion || Metadata.platform!=Application.platform+" / "+SystemInfo.processorType)
                throw new InvalidDataException("Recorded scene/code/engine/platform differs from the current project.");
#if UNITY_EDITOR
            if(Metadata.sceneRevision!=UnityEditor.AssetDatabase.GetAssetDependencyHash(Metadata.scene).ToString())
                throw new InvalidDataException("The saved scene/assets have changed since this recording.");
#endif
            if(!Metadata.settings.Equals(Settings.From(station)) || Time.fixedDeltaTime!=Metadata.stepSeconds ||
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
