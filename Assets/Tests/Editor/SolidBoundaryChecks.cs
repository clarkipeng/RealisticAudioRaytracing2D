using System;
using System.Collections.Generic;
using System.IO;
using NUnit.Framework;
using UnityEditor;
using UnityEngine;

public partial class TakeChecks
{
    static string SolidPython()
    {
        string python=Environment.GetEnvironmentVariable("ACOUSTIC_SOLID_PYTHON");
        if(string.IsNullOrEmpty(python))Assert.Ignore("Set ACOUSTIC_SOLID_PYTHON to a venv with tools/solid-boundary-requirements.txt.");
        return python;
    }
    static AcousticSurface SolidBox(Transform parent,string name,Vector3 center,Vector3 size,AudioMaterial material)
    {
        var go=new GameObject(name);go.transform.SetParent(parent,false);go.transform.localPosition=center;
        go.AddComponent<BoxCollider>().size=size;
        var surface=go.AddComponent<AcousticSurface>();surface.material=material;surface.geometrySource=AcousticSurface.GeometrySource.Collider;
        return surface;
    }
    static AudioMaterial RigidSolidMaterial(string name,float absorption=0)
    {
        var material=ScriptableObject.CreateInstance<AudioMaterial>();material.name=name;material.overrideBands=true;
        material.absorptionBands=new float[8];material.scatteringBands=new float[8];material.transmissionBands=new float[8];material.dampingBands=new float[8];
        material.iorBands=new[]{1f,1f,1f,1f,1f,1f,1f,1f};
        for(int i=0;i<8;i++)material.absorptionBands[i]=absorption;
        return material;
    }
    [Test,Category("SolidBake")]
    public void SolidBakePreservesMaterialsAndRejectsStaleGeometry()
    {
        string python=SolidPython(),asset="Assets/solid-test-"+Guid.NewGuid().ToString("N")+".json";
        var root=new GameObject("baked room");var manager=root.AddComponent<RayTraceManager>();manager.takeRoot=root.transform;
        var solid=root.AddComponent<AcousticSolid>();var a=RigidSolidMaterial("left",.1f);var b=RigidSolidMaterial("right",.7f);
        try
        {
            solid.members=new[]{SolidBox(root.transform,"left",Vector3.left*.5f,new Vector3(1,2,2),a),
                SolidBox(root.transform,"right",Vector3.right*.5f,new Vector3(1,2,2),b)};
            var door=SolidBox(root.transform,"moving door",Vector3.forward*4,Vector3.one,a);
            AcousticSolidEditor.Bake(solid,python,asset,a);manager.takeDefaultMaterial=a;manager.RegisterTakeObjects();
            var scene=SceneToTake.Compile(manager.takeObjects,AcousticTake.Dimension.Spatial3D,.01f,a,Array.Empty<Transform>(),Array.Empty<AudioClip>(),Array.Empty<Transform>(),out var geometry);
            int owner=manager.takeObjects.FindIndex(x=>x.transform==root.transform),doorOwner=manager.takeObjects.FindIndex(x=>x.transform==door.transform);
            Assert.That(scene.objects[owner].primitiveCount,Is.GreaterThan(0));Assert.That(scene.objects[doorOwner].primitiveCount,Is.EqualTo(12));
            foreach(var member in solid.members)Assert.That(scene.objects[manager.takeObjects.FindIndex(x=>x.transform==member.transform)].primitiveCount,Is.Zero);
            var matCounts=new int[2];
            foreach(var p in geometry)if(p.objectIndex==owner)
            {
                float x=(p.a.x+p.b.x+p.c.x)/3;var material=scene.materials[p.materialIndex];
                Assert.That(material.name,Is.EqualTo(x<0?"left":"right"));matCounts[x<0?0:1]++;
                Assert.That(Mathf.Max(Mathf.Abs(p.a.x),Mathf.Abs(p.b.x),Mathf.Abs(p.c.x)),Is.GreaterThan(0),"No internal joining face.");
            }
            Assert.That(matCounts[0]*matCounts[1],Is.GreaterThan(0));
            var guard=new SceneToTake.Guard(manager,Array.Empty<Transform>(),Array.Empty<AudioClip>(),Array.Empty<Transform>());
            root.transform.SetPositionAndRotation(new Vector3(3,2,-1),Quaternion.Euler(23,41,17));
            door.transform.localPosition+=Vector3.right;guard.Validate();solid.ReadValidated(a);
            for(int i=0;i<10;i++)guard.Validate();long before=GC.GetAllocatedBytesForCurrentThread();
            for(int i=0;i<100;i++)guard.Validate();Assert.That(GC.GetAllocatedBytesForCurrentThread()-before,Is.Zero);
            var groupSurface=root.GetComponent<AcousticSurface>();groupSurface.geometrySource=AcousticSurface.GeometrySource.None;
            var excluded=SceneToTake.Compile(manager.takeObjects,AcousticTake.Dimension.Spatial3D,.01f,a,Array.Empty<Transform>(),Array.Empty<AudioClip>(),Array.Empty<Transform>(),out _);
            Assert.That(excluded.objects[owner].primitiveCount,Is.Zero);
            foreach(var member in solid.members)Assert.That(excluded.objects[manager.takeObjects.FindIndex(x=>x.transform==member.transform)].primitiveCount,Is.EqualTo(12));
            var excludedGuard=new SceneToTake.Guard(manager,Array.Empty<Transform>(),Array.Empty<AudioClip>(),Array.Empty<Transform>());
            solid.members[0].transform.localPosition+=Vector3.right*.1f;excludedGuard.Validate();
            solid.members[0].transform.localPosition-=Vector3.right*.1f;
            groupSurface.geometrySource=AcousticSurface.GeometrySource.Auto;
            solid.members[0].transform.localPosition+=Vector3.right*.1f;
            Assert.Throws<InvalidOperationException>(guard.Validate);Assert.Throws<InvalidOperationException>(()=>solid.ReadValidated(a));
            solid.members[0].transform.localPosition-=Vector3.right*.1f;
            var collider=solid.members[0].GetComponent<BoxCollider>();collider.size+=Vector3.up;
            Assert.Throws<InvalidOperationException>(()=>solid.ReadValidated(a));
            solid.enabled=false;
            var unbaked=SceneToTake.Compile(manager.takeObjects,AcousticTake.Dimension.Spatial3D,.01f,a,Array.Empty<Transform>(),Array.Empty<AudioClip>(),Array.Empty<Transform>(),out _);
            Assert.That(unbaked.objects[owner].primitiveCount,Is.Zero);
            foreach(var member in solid.members)Assert.That(unbaked.objects[manager.takeObjects.FindIndex(x=>x.transform==member.transform)].primitiveCount,Is.EqualTo(12));
        }
        finally {AssetDatabase.DeleteAsset(asset);UnityEngine.Object.DestroyImmediate(root);UnityEngine.Object.DestroyImmediate(a);UnityEngine.Object.DestroyImmediate(b);}
    }
    [Test,Category("SolidBake")]
    public void SolidBakeRejectsDifferentInteriorMedia()
    {
        var root=new GameObject("media");var solid=root.AddComponent<AcousticSolid>();var a=RigidSolidMaterial("a");var b=RigidSolidMaterial("b");
        try
        {
            b.iorBands[0]=.5f;
            solid.members=new[]{SolidBox(root.transform,"a",Vector3.left*.5f,Vector3.one,a),SolidBox(root.transform,"b",Vector3.right*.5f,Vector3.one,b)};
            Assert.Throws<InvalidOperationException>(()=>solid.ExportInput(a,out _));
        }
        finally {UnityEngine.Object.DestroyImmediate(root);UnityEngine.Object.DestroyImmediate(a);UnityEngine.Object.DestroyImmediate(b);}
    }
    [TestCase(0,false),TestCase(17,false),TestCase(17,true),Category("SolidBake"),Category("GPU")]
    public void SolidPartitionsHaveMatchingTransport(int seed,bool moving)
    {
        string python=SolidPython();var material=RigidSolidMaterial("rigid");
        var root=new GameObject("solid");var solid=root.AddComponent<AcousticSolid>();
        var source=new GameObject("source");var listener=new GameObject("listener");source.transform.position=new Vector3(-3,2.5f,.2f);
        if(moving)root.transform.SetPositionAndRotation(new Vector3(.2f,-.3f,.4f),Quaternion.Euler(23,41,17));
        var outputs=new List<System.Numerics.Complex[]>();var audio=new List<float[]>();var assets=new List<string>();
        try
        {
            for(int split=0;split<2;split++)
            {
                foreach(var member in solid.members)UnityEngine.Object.DestroyImmediate(member.gameObject);
                solid.members=split==0?new[]{SolidBox(root.transform,"whole",Vector3.zero,Vector3.one*2,material)}:
                    new[]{SolidBox(root.transform,"left",Vector3.left*.9f,new Vector3(.2f,2,2),material),SolidBox(root.transform,"right",Vector3.right*.1f,new Vector3(1.8f,2,2),material)};
                string asset="Assets/solid-transport-"+Guid.NewGuid().ToString("N")+".json";assets.Add(asset);AcousticSolidEditor.Bake(solid,python,asset,material);
                var bindings=new List<RayTraceManager.TakeObjectBinding>{new(){id="source",transform=source.transform},new(){id="listener",transform=listener.transform},new(){id="solid",transform=root.transform,geometry=true}};
                foreach(var member in solid.members)bindings.Add(new(){id=member.name,transform=member.transform,geometry=true});
                var scene=SceneToTake.Compile(bindings,AcousticTake.Dimension.Spatial3D,.01f,material,Array.Empty<Transform>(),Array.Empty<AudioClip>(),new[]{listener.transform},out var geometry);
                scene.sources=new[]{new AcousticTake.SourceDefinition{objectIndex=0,clipIndex=0}};scene.clips=new[]{new AcousticTake.ClipDefinition()};
                var states=new AcousticTake.State[bindings.Count];for(int i=0;i<states.Length;i++)states[i]=SceneToTake.Observe(scene,i,bindings[i].transform);
                if(moving){states[2].velocity=new Vector3(.1f,0,0);states[2].angularVelocity=new Vector3(0,.2f,0);states[2].flags|=AcousticTake.StateFlags.VelocityValid;}
                var settings=EdgeSettings();settings.fftSize=8;settings.spectralHop=2;settings.raysPerBin=2;settings.maxBounces=2;settings.completeRouteSamples=262144;settings.seed=seed;settings.useBvh=true;
                var values=new List<System.Numerics.Complex>();
                using(var solver=new TransportSolver(scene,geometry,settings,RequireGpu(),sampleRate:1000))
                {
                    var direct=new TakeRender.DirectPath[1];
                    foreach(var point in new[]{new Vector3(3,2.5f,-.3f),new Vector3(0,3,0),new Vector3(3,0,0)})
                    {states[1].position=point;solver.Solve(states,direct);foreach(int bin in new[]{0,1,2,4})values.Add(EdgeResponse(solver.RayPaths,bin,bin*125));}
                    settings.listenerOutput=TakeRender.ListenerOutput.Binaural;
                    var pcm=new float[65];for(int i=0;i<pcm.Length;i++)pcm[i]=.1f*Mathf.Sin(i*.17f);
                    var signal=new TakeRender.Signal(pcm);signal.SetActive(0,true);signal.Append(new AcousticTake.VoiceEvent{gain=1,loop=true});
                    var signals=new[]{signal};var cpu=new TakeRender.Mixer(scene,1000,settings,signals);
                    using var gpu=new GpuAudio(scene,1000,settings,signals,transported:true);
                    cpu.SetResponse(0,direct,states,true,solver.RayPaths,0);gpu.SetResponse(0,true,solver.RayBuffer,0,scene,states);
                    var expected=new float[256];var actual=new float[256];cpu.Render(0,128,expected);gpu.Render(0,128,actual);
                    double audioEnergy=0;for(int i=0;i<actual.Length;i++){Assert.That(actual[i],Is.EqualTo(expected[i]).Within(2e-4));audioEnergy+=expected[i]*expected[i];}
                    Assert.That(audioEnergy,Is.GreaterThan(1e-6));audio.Add(actual);
                }
                outputs.Add(values.ToArray());
            }
            double error=0,energy=0;for(int i=0;i<outputs[0].Length;i++){error+=Math.Pow((outputs[1][i]-outputs[0][i]).Magnitude,2);energy+=Math.Pow(outputs[0][i].Magnitude,2);}
            for(int i=0;i<audio[0].Length;i++)Assert.That(audio[1][i],Is.EqualTo(audio[0][i]).Within(2e-4),"Equivalent solids must also preserve rendered audio.");
            double relative=Math.Sqrt(error/energy);Debug.Log("SOLID_PARTITION: seed="+seed+" moving="+moving+" relative="+relative.ToString("R"));
            Assert.That(relative,Is.LessThan(.01),"Equivalent solids should agree within the sampling control tolerance, versus the previous 11.4% discrepancy.");
        }
        finally {foreach(var asset in assets)AssetDatabase.DeleteAsset(asset);UnityEngine.Object.DestroyImmediate(root);UnityEngine.Object.DestroyImmediate(source);UnityEngine.Object.DestroyImmediate(listener);UnityEngine.Object.DestroyImmediate(material);}
    }
}
