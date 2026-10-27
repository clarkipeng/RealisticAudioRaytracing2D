using System;
using System.Collections.Generic;
using System.IO;
using UnityEditor;
using UnityEngine;

// Offline numeric input adapter. Copy only to the isolated runner's Editor folder.
public static class RainTransportReview
{
    [Serializable] sealed class Source { public Vector3 position; public float gain,area; public string file,profile; }
    [Serializable] sealed class Pose { public long sample; public Vector3 position; }
    [Serializable] sealed class Box { public Vector3 center,half; }
    [Serializable] sealed class Fixture { public int rate,seconds; public Source[] sources; public Pose[] positions; public Box[] walls; }

    public static void Prepare()
    {
        try
        {
            if(!Application.isBatchMode || Application.isPlaying)throw new InvalidOperationException("Offline only.");
            var args=Environment.GetCommandLineArgs();int arg=Array.IndexOf(args,"-rainFixture");
            string path=Path.GetFullPath(args[arg+1]),root=Path.GetDirectoryName(path);
            var f=JsonUtility.FromJson<Fixture>(File.ReadAllText(path));int count=f.sources.Length,head=count;
            var objects=new AcousticTake.ObjectDefinition[count+1+f.walls.Length];
            var state=new AcousticTake.State[objects.Length];
            for(int i=0;i<objects.Length;i++)
            {
                objects[i]=new AcousticTake.ObjectDefinition{id="rain-review-"+i,name="Rain review "+i,scale=Vector3.one};
                state[i]=new AcousticTake.State{rotation=Quaternion.identity,flags=AcousticTake.StateFlags.Active};
            }
            var sources=new AcousticTake.SourceDefinition[count];var clips=new AcousticTake.ClipDefinition[count];var pcm=new float[count][];
            for(int i=0;i<count;i++)
            {
                sources[i]=new AcousticTake.SourceDefinition{objectIndex=i,clipIndex=i};
                clips[i]=new AcousticTake.ClipDefinition{name=f.sources[i].profile+" patch "+i};
                var bytes=File.ReadAllBytes(Path.Combine(root,f.sources[i].file));
                if(bytes.Length!=f.seconds*f.rate*4)throw new InvalidDataException("Unexpected source length.");
                pcm[i]=new float[bytes.Length/4];Buffer.BlockCopy(bytes,0,pcm[i],0,bytes.Length);
                state[i].position=f.sources[i].position;state[i].flags|=AcousticTake.StateFlags.Source;
            }
            state[head].flags|=AcousticTake.StateFlags.Listener;
            state[head].rotation=Quaternion.LookRotation(Vector3.left,Vector3.up);
            var geometry=new List<AcousticTake.Primitive>();
            // Outward winding, copied from the existing closed-box transport fixture.
            int[] triangles={0,2,3,0,3,1,4,5,7,4,7,6,0,1,5,0,5,4,2,6,7,2,7,3,0,4,6,0,6,2,1,3,7,1,7,5};
            for(int b=0;b<f.walls.Length;b++)
            {
                int owner=count+1+b;var box=f.walls[b];var vertices=new Vector3[8];
                for(int v=0;v<8;v++)vertices[v]=Vector3.Scale(box.half,new Vector3((v&1)==0?-1:1,(v&2)==0?-1:1,(v&4)==0?-1:1));
                state[owner].position=box.center;state[owner].flags|=AcousticTake.StateFlags.Geometry;
                objects[owner].primitiveStart=geometry.Count;objects[owner].primitiveCount=triangles.Length/3;
                for(int v=0;v<triangles.Length;v+=3)geometry.Add(new AcousticTake.Primitive{objectIndex=owner,
                    a=vertices[triangles[v]],b=vertices[triangles[v+1]],c=vertices[triangles[v+2]],materialIndex=0});
            }
            var scene=new AcousticTake.Scene{dimension=AcousticTake.Dimension.Spatial3D,objects=objects,sources=sources,
                listeners=new[]{head},clips=clips,materials=new[]{new AcousticTake.MaterialDefinition{
                    name="Illustrative opaque surface",frequencies=new[]{0f},absorption=new[]{.35f},scattering=new[]{.15f},transmission=new[]{0f}}}};
            using(var writer=new AcousticTake.Writer(Path.Combine(root,"take"),f.rate,scene,geometry.ToArray(),pcm))
            {
                foreach(var pose in f.positions)
                {
                    state[head].position=pose.position;
                    // Store the same route. The standard take reader derives
                    // interval velocity when interpolating these observations.
                    writer.Append(pose.sample,state);
                }
                for(int s=0;s<count;s++)writer.Append(new AcousticTake.VoiceEvent{sourceIndex=s,gain=f.sources[s].gain});
                for(int s=0;s<count;s++)writer.Append(new AcousticTake.VoiceEvent{sample=(long)f.seconds*f.rate,sourceIndex=s,action=AcousticTake.VoiceAction.Stop});
                writer.Complete();
            }
            using(var take=AcousticTake.Open(Path.Combine(root,"take")))
            {
                if(take.Metadata.scene.sources.Length!=count || take.Metadata.eventCount!=2*count)throw new InvalidDataException("Input round trip failed.");
                Debug.Log("RAIN_TRANSPORT_PREPARED "+count+" sources, "+geometry.Count+" triangles, "+take.Metadata.sceneHash);
            }
            EditorApplication.Exit(0);
        }
        catch(Exception e){Debug.LogException(e);EditorApplication.Exit(1);}
    }
}
