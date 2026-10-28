using System;
using System.Collections;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using NUnit.Framework;
using UnityEditor;
using UnityEngine;
using UnityEngine.TestTools;
using Debug = UnityEngine.Debug;

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
        SolidPython();
        var root=new GameObject("baked room");var manager=root.AddComponent<RayTraceManager>();manager.takeRoot=root.transform;
        var solid=root.AddComponent<AcousticSolid>();var a=RigidSolidMaterial("left",.1f);var b=RigidSolidMaterial("right",.7f);
        try
        {
            solid.members=new[]{SolidBox(root.transform,"left",Vector3.left*.5f,new Vector3(1,2,2),a),
                SolidBox(root.transform,"right",Vector3.right*.5f,new Vector3(1,2,2),b)};
            var door=SolidBox(root.transform,"moving door",Vector3.forward*4,Vector3.one,a);
            manager.takeDefaultMaterial=a;manager.RegisterTakeObjects();
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
            var compiled=solid.Compiled;door.transform.localPosition+=Vector3.right;guard.Validate();Assert.That(solid.Resolve(a),Is.SameAs(compiled));
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
            // A running capture still rejects relative motion; the next compile regenerates.
            solid.members[0].transform.localPosition+=Vector3.right*.1f;
            Assert.Throws<InvalidOperationException>(guard.Validate);
            Assert.That(solid.Resolve(a).inputSha256,Is.Not.EqualTo(compiled.inputSha256));
            solid.members[0].transform.localPosition-=Vector3.right*.1f;
            var collider=solid.members[0].GetComponent<BoxCollider>();collider.size+=Vector3.up;
            Assert.That(solid.Resolve(a).inputSha256,Is.Not.EqualTo(compiled.inputSha256));
            solid.enabled=false;
            var unbaked=SceneToTake.Compile(manager.takeObjects,AcousticTake.Dimension.Spatial3D,.01f,a,Array.Empty<Transform>(),Array.Empty<AudioClip>(),Array.Empty<Transform>(),out _);
            Assert.That(unbaked.objects[owner].primitiveCount,Is.Zero);
            foreach(var member in solid.members)Assert.That(unbaked.objects[manager.takeObjects.FindIndex(x=>x.transform==member.transform)].primitiveCount,Is.EqualTo(12));
        }
        finally {UnityEngine.Object.DestroyImmediate(root);UnityEngine.Object.DestroyImmediate(a);UnityEngine.Object.DestroyImmediate(b);}
    }
    static void WithSolidPython(string python,Action action)
    {
        string previous=Environment.GetEnvironmentVariable(AcousticSolidBuilder.PythonVariable);
        Environment.SetEnvironmentVariable(AcousticSolidBuilder.PythonVariable,python);
        try {action();} finally {Environment.SetEnvironmentVariable(AcousticSolidBuilder.PythonVariable,previous);}
    }
    // A run-unique depth keeps this test's Library cache entries cold.
    static float UniqueSolidDepth()=>1+(Guid.NewGuid().GetHashCode()&0xffff)*2.5e-7f;
    [Test,Category("SolidBake")]
    public void SolidBoundaryRegeneratesOnlyWhenStale()
    {
        string python=SolidPython(),missing=Path.Combine(Path.GetTempPath(),"missing-"+Guid.NewGuid().ToString("N"),"python");
        string fake=Path.Combine(Path.GetTempPath(),"no-backend-"+Guid.NewGuid().ToString("N")+".sh");
        var root=new GameObject("auto solid");var solid=root.AddComponent<AcousticSolid>();var twin=new GameObject("auto solid twin");
        var a=RigidSolidMaterial("a",.1f);var b=RigidSolidMaterial("b",.7f);var c=RigidSolidMaterial("c",.4f);
        var hashes=new List<string>();float depth=UniqueSolidDepth();
        try
        {
            solid.members=new[]{SolidBox(root.transform,"left",Vector3.left*.5f,new Vector3(1,2,depth),a),SolidBox(root.transform,"right",Vector3.right*.5f,new Vector3(1,2,depth),b)};
            int launches=AcousticSolidBuilder.WorkerLaunches;
            var first=solid.Resolve(a);hashes.Add(first.inputSha256);
            Assert.That(AcousticSolidBuilder.WorkerLaunches,Is.EqualTo(launches+1),"A cold input runs the backend once.");
            Assert.That(File.Exists(AcousticSolidBuilder.CachePath(first.inputSha256)));
            Assert.That(solid.SurfaceMaterials,Is.EqualTo(new[]{a,b}));

            // Rigid motion of the whole group is a cache hit: same boundary, no Python at all.
            root.transform.SetPositionAndRotation(new Vector3(3,-2,5),Quaternion.Euler(10,70,-30));
            WithSolidPython(missing,()=>Assert.That(solid.Resolve(a),Is.SameAs(first)));

            // Relative member motion regenerates; reverting it reuses the Library cache.
            solid.members[1].transform.localPosition+=Vector3.right*.25f;
            var moved=solid.Resolve(a);hashes.Add(moved.inputSha256);
            Assert.That(moved.inputSha256,Is.Not.EqualTo(first.inputSha256));Assert.That(moved.components,Is.EqualTo(2),"A real gap survives.");
            Assert.That(AcousticSolidBuilder.WorkerLaunches,Is.EqualTo(launches+2));
            solid.members[1].transform.localPosition-=Vector3.right*.25f;
            WithSolidPython(missing,()=>Assert.That(solid.Resolve(a).inputSha256,Is.EqualTo(first.inputSha256)));

            // A fresh component with the same local arrangement (e.g. a new checkout of an
            // unsaved edit) also hits the Library cache.
            var copy=twin.AddComponent<AcousticSolid>();
            copy.members=new[]{SolidBox(twin.transform,"left",Vector3.left*.5f,new Vector3(1,2,depth),a),SolidBox(twin.transform,"right",Vector3.right*.5f,new Vector3(1,2,depth),b)};
            Assert.That(copy.Compiled,Is.Null);
            WithSolidPython(missing,()=>Assert.That(copy.Resolve(a).inputSha256,Is.EqualTo(first.inputSha256)));

            // Replacing a material while keeping distinct face ownership keeps the same
            // exterior and only remaps the slot asset.
            solid.members[1].material=c;
            WithSolidPython(missing,()=>Assert.That(solid.Resolve(a).inputSha256,Is.EqualTo(first.inputSha256)));
            Assert.That(solid.SurfaceMaterials,Is.EqualTo(new[]{a,c}));
            var bindings=new List<RayTraceManager.TakeObjectBinding>{new(){id="solid",transform=root.transform,geometry=true}};
            foreach(var member in solid.members)bindings.Add(new(){id=member.name,transform=member.transform,geometry=true});
            var scene=SceneToTake.Compile(bindings,AcousticTake.Dimension.Spatial3D,.01f,a,Array.Empty<Transform>(),Array.Empty<AudioClip>(),Array.Empty<Transform>(),out _);
            Assert.That(Array.ConvertAll(scene.materials,m=>m.name),Is.EquivalentTo(new[]{"a","c"}));

            // Sharing one material merges face ownership, so the exterior must regenerate.
            // Missing Python or backend packages fail with actionable messages, never stale geometry.
            solid.members[1].material=a;
            var notFound=Assert.Throws<InvalidOperationException>(()=>WithSolidPython(missing,()=>solid.Resolve(a)));
            StringAssert.Contains("was not found",notFound.Message);StringAssert.Contains(AcousticSolidBuilder.PythonVariable,notFound.Message);
            if(Application.platform!=RuntimePlatform.WindowsEditor)
            {
                File.WriteAllText(fake,"#!/bin/sh\necho \"Solid-boundary backend unavailable: missing Python module 'manifold3d'. Install it with: python -m pip install -r tools/solid-boundary-requirements.txt\" >&2\nexit 3\n");
                using(var chmod=Process.Start("/bin/chmod","+x \""+fake+"\""))chmod.WaitForExit();
                var noBackend=Assert.Throws<InvalidOperationException>(()=>WithSolidPython(fake,()=>solid.Resolve(a)));
                StringAssert.Contains("pip install -r tools/solid-boundary-requirements.txt",noBackend.Message);StringAssert.Contains("Retry",noBackend.Message);
            }
            Assert.That(solid.Compiled.inputSha256,Is.EqualTo(first.inputSha256),"Failures keep the old boundary marked stale; it is never returned.");
            var merged=solid.Resolve(a);hashes.Add(merged.inputSha256);
            Assert.That(merged.inputSha256,Is.Not.EqualTo(first.inputSha256));
            Assert.That(merged.outputTriangles,Is.EqualTo(12));Assert.That(new HashSet<int>(merged.materials),Is.EquivalentTo(new[]{0}));
            Assert.That(solid.SurfaceMaterials,Is.EqualTo(new[]{a}));
            Debug.Log("SOLID_AUTOMATIC: launches="+(AcousticSolidBuilder.WorkerLaunches-launches));
        }
        finally
        {
            foreach(string hash in hashes)File.Delete(AcousticSolidBuilder.CachePath(hash));
            File.Delete(fake);
            UnityEngine.Object.DestroyImmediate(root);UnityEngine.Object.DestroyImmediate(twin);
            UnityEngine.Object.DestroyImmediate(a);UnityEngine.Object.DestroyImmediate(b);UnityEngine.Object.DestroyImmediate(c);
        }
    }
    [UnityTest,Category("SolidBake")]
    public IEnumerator SolidBoundaryRegeneratesAfterEditorEdits()
    {
        SolidPython();
        var root=new GameObject("edited solid");var solid=root.AddComponent<AcousticSolid>();var material=RigidSolidMaterial("edited");
        float depth=UniqueSolidDepth();var hashes=new List<string>();
        try
        {
            solid.members=new[]{SolidBox(root.transform,"low",Vector3.zero,new Vector3(2,1,depth),material),SolidBox(root.transform,"high",Vector3.up,new Vector3(2,1,depth),material)};
            var first=solid.Resolve(material);hashes.Add(first.inputSha256);
            // An ordinary undoable editor edit: change events, settling and the background
            // worker regenerate with no explicit call.
            Undo.RecordObject(solid.members[1].transform,"Move solid member");
            solid.members[1].transform.localPosition+=Vector3.right*.5f;
            var timer=Stopwatch.StartNew();
            while(solid.Compiled==first && timer.Elapsed.TotalSeconds<60)yield return null;
            string expected=solid.InputHash(material,out _,out _);hashes.Add(expected);
            Assert.That(solid.Compiled.inputSha256,Is.EqualTo(expected),"Background regeneration after an editor edit.");
            Assert.That(AcousticSolidBuilder.Describe(solid).state,Is.EqualTo(AcousticSolidBuilder.State.Current));
            // Moving only the root publishes change events but starts no worker.
            var current=solid.Compiled;int launches=AcousticSolidBuilder.WorkerLaunches;
            Undo.RecordObject(root.transform,"Move solid");root.transform.position+=new Vector3(4,0,1);
            timer.Restart();while(timer.Elapsed.TotalSeconds<2)yield return null;
            Assert.That(solid.Compiled,Is.SameAs(current));Assert.That(AcousticSolidBuilder.WorkerLaunches,Is.EqualTo(launches));
        }
        finally
        {
            foreach(string hash in hashes)File.Delete(AcousticSolidBuilder.CachePath(hash));
            UnityEngine.Object.DestroyImmediate(root);UnityEngine.Object.DestroyImmediate(material);
        }
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
        SolidPython();var material=RigidSolidMaterial("rigid");
        var root=new GameObject("solid");var solid=root.AddComponent<AcousticSolid>();
        var source=new GameObject("source");var listener=new GameObject("listener");source.transform.position=new Vector3(-3,2.5f,.2f);
        if(moving)root.transform.SetPositionAndRotation(new Vector3(.2f,-.3f,.4f),Quaternion.Euler(23,41,17));
        var outputs=new List<System.Numerics.Complex[]>();var audio=new List<float[]>();
        try
        {
            for(int split=0;split<2;split++)
            {
                foreach(var member in solid.members)UnityEngine.Object.DestroyImmediate(member.gameObject);
                solid.members=split==0?new[]{SolidBox(root.transform,"whole",Vector3.zero,Vector3.one*2,material)}:
                    new[]{SolidBox(root.transform,"left",Vector3.left*.9f,new Vector3(.2f,2,2),material),SolidBox(root.transform,"right",Vector3.right*.1f,new Vector3(1.8f,2,2),material)};
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
        finally {UnityEngine.Object.DestroyImmediate(root);UnityEngine.Object.DestroyImmediate(source);UnityEngine.Object.DestroyImmediate(listener);UnityEngine.Object.DestroyImmediate(material);}
    }
}
