using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using UnityEditor;
using UnityEngine;
using UnityEngine.SceneManagement;

// Reconstruct station inputs in an isolated physics scene, without live audio or Play mode.
public static class StationReplay
{
    [MenuItem("Tools/Acoustics/Reconstruct recorded station inputs")]
    static void ReconstructMenu()
    {
        string path=EditorUtility.OpenFolderPanel("Choose a station recording","Recordings","");
        if(string.IsNullOrEmpty(path))return;
        try {Debug.Log(Reconstruct(path,Path.Combine(path,"reconstructed-"+DateTime.UtcNow.ToString("yyyyMMdd-HHmmss-fff"))));}
        catch(Exception e){Debug.LogException(e);}
    }
    public static AcousticTake Open(string path,string output)
    {
        output=Path.GetFullPath(output).TrimEnd(Path.DirectorySeparatorChar,Path.AltDirectorySeparatorChar);
        if(Directory.Exists(output))throw new IOException("Render output directory already exists.");
        if(File.Exists(Path.Combine(path,"states.bin")))return AcousticTake.Open(path);
        return AcousticTake.Open(Reconstruct(path,output+"-inputs"));
    }
    public static string Reconstruct(string path,string output,Action<float> progress=null)
    {
        using var playback=new StationRecording.Playback(path);
        var meta=playback.Metadata;
        var authoredScene=UnityEditor.SceneManagement.EditorSceneManager.OpenPreviewScene(meta.scene);
        try
        {
            var authored=authoredScene.GetRootGameObjects().SelectMany(r=>r.GetComponentsInChildren<StationDemo>(true)).Single();
            using var take=File.Exists(Path.Combine(path,"states.bin"))?AcousticTake.Open(path):null;
            playback.ValidateScene(authored,false);
            var description=playback.TakeMetadata.scene;
            var result=new Report {ticks=meta.ticks,comparedReference=take!=null};
            using(var rig=new Rig(authored))
            {
                var pcm=new float[description.clips.Length][];
                AcousticTake.Primitive[] geometry;
                if(take!=null) { geometry=take.Geometry; for(int i=0;i<pcm.Length;i++)pcm[i]=take.ReadClip(i); }
                else
                {
                    var m=authored.manager;var clips=m.takeSources.Select(s=>s.clip).ToArray();m.sampleRate=playback.TakeMetadata.sampleRate;
                    description=SceneToTake.Compile(rig.bindings,m.takeDimension,m.takeCurveTolerance,m.takeDefaultMaterial,
                        m.takeSources.Select(s=>rig.copies[s.transform]).ToArray(),clips,m.takeListeners.Select(l=>rig.copies[l]).ToArray(),out geometry);
                    for(int s=0;s<clips.Length;s++)pcm[description.sources[s].clipIndex]=RayTraceManager.LoadSample(clips[s],playback.TakeMetadata.sampleRate);
                }
                using var writer=new AcousticTake.Writer(output,playback.TakeMetadata.sampleRate,
                    JsonUtility.FromJson<AcousticTake.Scene>(JsonUtility.ToJson(description)),geometry,pcm);
                if(writer.Metadata.sceneHash!=playback.TakeMetadata.sceneHash)throw new InvalidDataException("Authored geometry/source audio differs from the recording.");
                writer.Metadata.observationPolicy="recorded-input-resimulation";
                // The isolated scene runs the same initialization and step as live playback.
                foreach(var e in playback.Events)writer.Append(e);
                var expected=new AcousticTake.State[description.objects.Length];
                AcousticTake.State[] previous=null; long simulated=0,previousSample=-1;
                for(int f=0;playback.NextObservation(long.MaxValue,out long sample,out long tick);f++)
                {
                    while(simulated<tick)
                    {
                        rig.demo.Step(playback.Read(simulated),meta.stepSeconds); rig.scene.GetPhysicsScene().Simulate(meta.stepSeconds); simulated++;
                    }
                    var state=Observe(rig,description);
                    if(previous!=null)for(int i=0;i<state.Length;i++)state[i].MeasureMotion(previous[i],(double)(sample-previousSample)/playback.TakeMetadata.sampleRate);
                    writer.Append(sample,state); result.frames++;
                    progress?.Invoke((float)(f+1)/meta.observationCount);
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

    [Serializable] public sealed class Report
    {
        public int frames, changedStates;
        public long ticks;
        public double maximumPositionMetres, maximumRotationDegrees, maximumVelocityError;
        public string stateHash,eventHash;
        public bool comparedReference,byteIdentical;
    }
    internal sealed class Rig : IDisposable
    {
        public readonly Scene scene;
        public readonly StationDemo demo;
        public readonly PlayerController walker;
        public readonly Transform head;
        public readonly List<RayTraceManager.TakeObjectBinding> bindings = new List<RayTraceManager.TakeObjectBinding>();
        public readonly Dictionary<Transform, Transform> copies = new Dictionary<Transform, Transform>();
        public Rig(StationDemo source)
        {
            scene = UnityEditor.SceneManagement.EditorSceneManager.NewPreviewScene();
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
                head = copies[source.manager.takeListeners[0]];
                var look=head.gameObject.AddComponent<MouseLook>(); look.enabled=false; look.playerBody=demo.player; look.lockOnStart=false;
                var controller = demo.player.gameObject.AddComponent<CharacterController>();
                EditorUtility.CopySerialized(source.player.GetComponent<CharacterController>(), controller);
                controller.enabled = true;
                walker = demo.player.gameObject.AddComponent<PlayerController>(); walker.enabled = false;
                StationRecording.Settings.From(source).Apply(demo);
                demo.InitializeSimulation();
            }
            catch { Dispose(); throw; }
        }
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
            if (source.GetComponent<Rigidbody>() != null) throw new InvalidOperationException("Fixture now contains another rigid body; compact station replay supports its train and walking controller only.");
            for (int i = 0; i < source.childCount; i++) Copy(source.GetChild(i), t);
            t.gameObject.SetActive(source.gameObject.activeSelf);
        }
        public void Dispose()
        {
            if (!scene.IsValid()) return;
            foreach (var root in scene.GetRootGameObjects()) UnityEngine.Object.DestroyImmediate(root);
            UnityEditor.SceneManagement.EditorSceneManager.ClosePreviewScene(scene);
        }
    }
    static AcousticTake.State[] Observe(Rig rig, AcousticTake.Scene description)
    {
        var result = new AcousticTake.State[rig.bindings.Count];
        for (int i = 0; i < result.Length; i++)
        {
            var flags = rig.bindings[i].geometry ? AcousticTake.StateFlags.Geometry : 0;
            if (description.sources.Any(s => s.objectIndex == i)) flags |= AcousticTake.StateFlags.Source;
            if (description.listeners.Contains(i)) flags |= AcousticTake.StateFlags.Listener;
            result[i] = AcousticTake.State.Observe(rig.bindings[i].transform, flags);
        }
        return result;
    }
}
