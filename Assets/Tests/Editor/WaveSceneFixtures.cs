using System;
using System.Collections.Generic;
using System.IO;
using NUnit.Framework;
using UnityEngine;

// The wave-scene JSON of the reference tools (tools/fdtd_wave.py, the method comparison scenes) and its
// compilation into takes and solid bakes, shared by the path measurements.
public partial class TakeChecks
{
    [Serializable] sealed class WaveWall { public float[] center, half; }
    [Serializable] sealed class WaveMotion { public float[] velocity; }
    [Serializable] sealed class WaveScene
    {
        public string physicalModel;
        public float[] source, points, frequencies;
        public WaveWall[] walls;
        public int[] rays, seeds;
        public int sampleRate, fftSize, maxBounces, completeRouteSamples;
        public float speedOfSound, maximumDelaySeconds;
        public bool decompose, captureFamilies, solidGeometry, bakeSolids;
        public float[] triangles;
        public int[] triangleMaterials;
        public AcousticTake.MaterialDefinition[] materials;
        public WaveMotion receiverMotion;
    }
    static Vector3 WaveVector(float[] values, int offset=0) =>
        new Vector3(values[offset], values[offset+1], values[offset+2]);


    // Exercise the same automatic boundary generation and scene compiler used by live
    // and recorded scenes, rather than substituting a hand-authored union for the renderer.
    sealed class BakedWaveSolid : IDisposable
    {
        readonly GameObject root,source,listener; readonly AudioMaterial material;
        public readonly AcousticTake.Scene scene;
        public readonly AcousticTake.Primitive[] geometry;
        public readonly AcousticTake.State[] states;
        public BakedWaveSolid(WaveScene config,string output)
        {
            Assert.That(config.triangles,Is.Null);Assert.That(config.materials,Is.Null,"This bake/reference fixture uses rigid walls.");
            Assert.That(config.physicalModel,Is.Not.EqualTo("point-extruded"));
            SolidPython();
            root=new GameObject("physical boundary");source=new GameObject("source");listener=new GameObject("listener");
            material=RigidSolidMaterial("rigid");
            try
            {
                var solid=root.AddComponent<AcousticSolid>();solid.members=new AcousticSurface[config.walls.Length];
                var bindings=new List<RayTraceManager.TakeObjectBinding>{new(){id="source",transform=source.transform},new(){id="listener",transform=listener.transform},new(){id="boundary",transform=root.transform,geometry=true}};
                for(int i=0;i<config.walls.Length;i++)
                {
                    var wall=config.walls[i];solid.members[i]=SolidBox(root.transform,"wall-"+i,WaveVector(wall.center),WaveVector(wall.half)*2,material);
                    bindings.Add(new(){id="wall-"+i,transform=solid.members[i].transform,geometry=true});
                }
                File.WriteAllText(Path.Combine(output,"boundary-input.json"),solid.ExportInput(material,out _),new System.Text.UTF8Encoding(false));
                File.WriteAllText(Path.Combine(output,"boundary.json"),JsonUtility.ToJson(solid.Resolve(material),true));
                source.transform.position=WaveVector(config.source);
                scene=SceneToTake.Compile(bindings,AcousticTake.Dimension.Spatial3D,.01f,material,Array.Empty<Transform>(),Array.Empty<AudioClip>(),new[]{listener.transform},out geometry);
                scene.sources=new[]{new AcousticTake.SourceDefinition{objectIndex=0,clipIndex=0}};scene.clips=new[]{new AcousticTake.ClipDefinition()};
                states=new AcousticTake.State[bindings.Count];for(int i=0;i<states.Length;i++)states[i]=SceneToTake.Observe(scene,i,bindings[i].transform);
            }
            catch {Dispose();throw;}
        }
        public void Dispose()
        {
            UnityEngine.Object.DestroyImmediate(root);
            UnityEngine.Object.DestroyImmediate(source);UnityEngine.Object.DestroyImmediate(listener);UnityEngine.Object.DestroyImmediate(material);
        }
    }
    // The same automatic exterior generation and scene compiler as BakedWaveSolid,
    // with one existing material preset (or a rigid wall) per member box. Presets
    // carry different interior media, so each preset forms its own solid group.
    sealed class DatasetSolid : IDisposable
    {
        readonly List<GameObject> objects=new();readonly List<AudioMaterial> created=new();
        public readonly AcousticTake.Scene scene;
        public readonly AcousticTake.Primitive[] geometry;
        public readonly AcousticTake.State[] states;
        // resolve: optional material per name (owned by the caller); default the Church presets.
        public DatasetSolid(WaveScene config,string[] materials,string output,Func<string,AudioMaterial> resolve=null)
        {
            SolidPython();
            try
            {
                var source=new GameObject("source");var listener=new GameObject("listener");objects.Add(source);objects.Add(listener);
                var assets=new Dictionary<string,AudioMaterial>();
                AudioMaterial Material(string name)
                {
                    if(assets.TryGetValue(name,out var asset))return asset;
                    if(resolve!=null)asset=resolve(name);
                    else if(name=="rigid"){asset=RigidSolidMaterial("rigid");created.Add(asset);}
                    else asset=UnityEditor.AssetDatabase.LoadAssetAtPath<AudioMaterial>($"Assets/Materials/Church {name}.asset");
                    Assert.That(asset,Is.Not.Null,$"Unknown material preset {name}.");
                    return assets[name]=asset;
                }
                var bindings=new List<RayTraceManager.TakeObjectBinding>{new(){id="source",transform=source.transform},new(){id="listener",transform=listener.transform}};
                var inputs=new List<string>();var boundaries=new List<string>();
                var groups=new List<string>();foreach(var name in materials)if(!groups.Contains(name))groups.Add(name);
                foreach(var name in groups)
                {
                    var material=Material(name);var root=new GameObject("boundary-"+name);objects.Add(root);
                    var solid=root.AddComponent<AcousticSolid>();var members=new List<AcousticSurface>();
                    bindings.Add(new(){id="boundary-"+name,transform=root.transform,geometry=true});
                    for(int i=0;i<config.walls.Length;i++)if(materials[i]==name)
                    {
                        var wall=config.walls[i];var member=SolidBox(root.transform,"wall-"+i,WaveVector(wall.center),WaveVector(wall.half)*2,material);
                        members.Add(member);bindings.Add(new(){id="wall-"+i,transform=member.transform,geometry=true});
                    }
                    solid.members=members.ToArray();
                    inputs.Add(solid.ExportInput(material,out _));boundaries.Add(JsonUtility.ToJson(solid.Resolve(material)));
                }
                File.WriteAllText(Path.Combine(output,"boundary-input.json"),"["+string.Join(",",inputs)+"]",new System.Text.UTF8Encoding(false));
                File.WriteAllText(Path.Combine(output,"boundary.json"),"{\"groups\":"+JsonUtility.ToJson(new StringList{names=groups.ToArray()})+",\"boundaries\":["+string.Join(",",boundaries)+"]}",new System.Text.UTF8Encoding(false));
                source.transform.position=WaveVector(config.source);
                scene=SceneToTake.Compile(bindings,AcousticTake.Dimension.Spatial3D,.01f,Material(materials[0]),Array.Empty<Transform>(),Array.Empty<AudioClip>(),new[]{listener.transform},out geometry);
                scene.sources=new[]{new AcousticTake.SourceDefinition{objectIndex=0,clipIndex=0}};scene.clips=new[]{new AcousticTake.ClipDefinition()};
                states=new AcousticTake.State[bindings.Count];for(int i=0;i<states.Length;i++)states[i]=SceneToTake.Observe(scene,i,bindings[i].transform);
            }
            catch {Dispose();throw;}
        }
        public void Dispose()
        {
            foreach(var o in objects)UnityEngine.Object.DestroyImmediate(o);
            foreach(var material in created)UnityEngine.Object.DestroyImmediate(material);
        }
    }
    [Serializable] sealed class StringList { public string[] names; }
    static void WriteFloats(string path,float[] values)
    {
        using var writer=new BinaryWriter(File.Create(path));
        foreach(float v in values)writer.Write(v); // BinaryWriter is little-endian on every platform.
    }
}
