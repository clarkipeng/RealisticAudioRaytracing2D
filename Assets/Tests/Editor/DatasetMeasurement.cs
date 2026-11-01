using System;
using System.Collections.Generic;
using System.IO;
using NUnit.Framework;
using UnityEngine;

// Offline audio-DLSS dataset renders: complex receiver pressure at several ray
// budgets and seeds, split by path family. Driven by tools/dlss_dataset.py.
public partial class TakeChecks
{
    [Serializable] sealed class DatasetBudget { public string name; public int raysPerBin, completeRouteSamples; public int[] seeds; }
    [Serializable] sealed class DatasetRender { public DatasetBudget[] budgets; }
    [Serializable] sealed class DatasetScene
    {
        public string schema, sceneId;
        public float[] source, points, frequencies;
        public int sampleRate, fftSize, maxBounces;
        public float speedOfSound, maximumDelaySeconds;
        public bool bakeSolids;
        public string[] surfaceMaterials;
        public DatasetRender render;
    }
    [Serializable] sealed class DatasetBudgetRun
    {
        public string name;
        public int raysPerBin, completeRouteSamples, pathsPerSolve;
        public int[] seeds;
        public double busySeconds, wallSeconds;
    }
    [Serializable] sealed class DatasetRun
    {
        public string schema="dlss-tracer-run-v1", sceneId, sceneSha256, variant, backend, unityVersion;
        public string[] materials;
        public string[] families;
        public float[] frequencies;
        public int[] bins;
        public int points, primitives, sampleRate, fftSize, maxBounces;
        public float speedOfSound, maximumDelaySeconds, workFraction;
        public DatasetBudgetRun[] budgets;
        public double wallSeconds;
    }

    // Family order of every stored family axis. R is the deterministic first-order
    // image set; D..RR are complete early routes; walk is the residual random walk.
    static readonly string[] DatasetFamilies={"direct","R","D","DD","RD","DR","RR","walk"};

    [Test, Category("QualityMeasurement"), Timeout(int.MaxValue)]
    public void MeasureDatasetScenes()
    {
        string jobs=Environment.GetEnvironmentVariable("ACOUSTIC_DATASET_JOBS");
        if(string.IsNullOrEmpty(jobs))Assert.Ignore("Set ACOUSTIC_DATASET_JOBS to a file listing prepared scene directories.");
        Assert.That(Application.isBatchMode,Is.True,"Use an isolated batch project.");
        string budget=Environment.GetEnvironmentVariable("ACOUSTIC_DATASET_WORK_FRACTION");
        double workFraction=string.IsNullOrEmpty(budget)?1:double.Parse(budget,System.Globalization.CultureInfo.InvariantCulture);
        Assert.That(workFraction,Is.GreaterThan(0).And.LessThanOrEqualTo(1));
        AudioListener.volume=0;
        // Long renders can outlive an audio-device change (sleep, battery suspend).
        // Device and service logs say nothing about the acoustics; every other error
        // log still fails the render, and tools/dlss_jobs.py verifies each scene.
        var unrelated=new System.Text.RegularExpressions.Regex("FMOD|audio device|UnityConnect");
        var errors=datasetErrors=new System.Collections.Concurrent.ConcurrentQueue<string>();
        Application.LogCallback collect=(message,stack,type)=>
        {if((type==LogType.Error||type==LogType.Exception||type==LogType.Assert)&&!unrelated.IsMatch(message))errors.Enqueue(message);};
        bool ignored=UnityEngine.TestTools.LogAssert.ignoreFailingMessages;
        UnityEngine.TestTools.LogAssert.ignoreFailingMessages=true;
        Application.logMessageReceivedThreaded+=collect;
        try
        {
            var shader=RequireGpu();
            foreach(string line in File.ReadAllLines(jobs))
            {
                // "<scene directory>\t<variant>": tracer uses the scene's material presets,
                // tracer-rigid replaces every surface with a rigid wall (FDTD comparison).
                var fields=line.Split('\t');if(fields[0].Trim().Length==0)continue;
                string variant=fields.Length>1?fields[1].Trim():"tracer";
                Assert.That(variant,Is.EqualTo("tracer").Or.EqualTo("tracer-rigid"));
                RenderDatasetScene(fields[0].Trim(),variant,shader,workFraction);
            }
        }
        finally
        {
            Application.logMessageReceivedThreaded-=collect;
            UnityEngine.TestTools.LogAssert.ignoreFailingMessages=ignored;
        }
    }

    static System.Collections.Concurrent.ConcurrentQueue<string> datasetErrors=new();
    void RenderDatasetScene(string directory,string variant,ComputeShader shader,double workFraction)
    {
        string final=Path.Combine(directory,variant),partial=Path.Combine(directory,variant+".partial");
        Assert.That(Directory.Exists(final),Is.False,"Completed renders are never overwritten.");
        if(Directory.Exists(partial))Directory.Delete(partial,true);
        Directory.CreateDirectory(partial);
        long sceneStarted=System.Diagnostics.Stopwatch.GetTimestamp();
        byte[] bytes=File.ReadAllBytes(Path.Combine(directory,"scene.json"));
        string text=new System.Text.UTF8Encoding(false,true).GetString(bytes);
        string sceneSha256;using(var sha=System.Security.Cryptography.SHA256.Create())sceneSha256=BitConverter.ToString(sha.ComputeHash(bytes)).Replace("-","").ToLowerInvariant();
        var config=JsonUtility.FromJson<WaveScene>(text);var job=JsonUtility.FromJson<DatasetScene>(text);
        Assert.That(job.schema,Is.EqualTo("dlss-scene-v1"));
        Assert.That(job.bakeSolids && config.materials==null && config.triangles==null,Is.True,"Dataset scenes use the automatic solid bake.");
        Assert.That(job.surfaceMaterials.Length,Is.EqualTo(config.walls.Length));
        var names=variant=="tracer"?job.surfaceMaterials:Array.ConvertAll(job.surfaceMaterials,_=>"rigid");
        using var baked=new DatasetSolid(config,names,partial);
        var scene=baked.scene;var geometry=baked.geometry;var state=baked.states;
        int points=job.points.Length/3,frequencies=job.frequencies.Length,families=DatasetFamilies.Length;
        var bins=new int[frequencies];var binIndex=new int[job.fftSize/2+1];Array.Fill(binIndex,-1);
        for(int q=0;q<frequencies;q++)
        {
            float bin=job.frequencies[q]*job.fftSize/job.sampleRate;
            Assert.That(bin,Is.EqualTo(Mathf.Round(bin)),"Frequencies must be exact configured FFT bins.");
            bins[q]=(int)bin;Assert.That(binIndex[bins[q]],Is.EqualTo(-1));binIndex[bins[q]]=q;
        }
        var run=new DatasetRun{sceneId=job.sceneId,sceneSha256=sceneSha256,variant=variant,materials=names,backend=SystemInfo.graphicsDeviceName+" "+SystemInfo.graphicsDeviceType,
            unityVersion=Application.unityVersion,families=DatasetFamilies,frequencies=job.frequencies,bins=bins,points=points,
            primitives=geometry.Length,sampleRate=job.sampleRate,fftSize=job.fftSize,maxBounces=job.maxBounces,
            speedOfSound=job.speedOfSound,maximumDelaySeconds=job.maximumDelaySeconds,workFraction=(float)workFraction,
            budgets=new DatasetBudgetRun[job.render.budgets.Length]};
        var visible=new byte[points];bool visibleKnown=false;var direct=new TakeRender.DirectPath[1];
        for(int b=0;b<job.render.budgets.Length;b++)
        {
            var spec=job.render.budgets[b];int seeds=spec.seeds.Length;
            var values=new double[seeds*points*frequencies*families*2];var counts=new long[seeds*points*frequencies*families];
            var peaks=new float[counts.Length];var seconds=new float[seeds*points];
            var record=new DatasetBudgetRun{name=spec.name,raysPerBin=spec.raysPerBin,completeRouteSamples=spec.completeRouteSamples,seeds=spec.seeds};
            long budgetStarted=System.Diagnostics.Stopwatch.GetTimestamp();
            for(int s=0;s<seeds;s++)
            {
                var settings=EdgeSettings();settings.useBvh=true;settings.airAbsorption=false;
                settings.fftSize=job.fftSize;settings.spectralHop=job.fftSize/2;
                settings.completeRouteSamples=spec.completeRouteSamples;settings.raysPerBin=spec.raysPerBin;
                settings.seed=spec.seeds[s];settings.maxBounces=job.maxBounces;settings.diffractionFactor=1;
                settings.speedOfSound=job.speedOfSound;settings.maximumDelaySeconds=job.maximumDelaySeconds;settings.minimumDistance=.01f;
                using var solver=new TransportSolver(scene,geometry,settings,shader,sampleRate:job.sampleRate);
                int bins0=settings.FrequencyBins,walkStart=bins0*(1+settings.MirrorCandidates(scene));
                int completeStart=walkStart+settings.RaysPerPair*settings.PathSlotsPerRay;
                state[1].position=WaveVector(job.points);solver.Solve(state,direct); // Exclude cold dispatch.
                long paced=System.Diagnostics.Stopwatch.GetTimestamp();
                for(int point=0;point<points;point++)
                {
                    state[1].position=WaveVector(job.points,point*3);
                    long started=System.Diagnostics.Stopwatch.GetTimestamp();solver.Solve(state,direct);
                    seconds[s*points+point]=(float)TakeRender.Elapsed(started);
                    if(!visibleKnown)visible[point]=(byte)(direct[0].pressureGain!=0?1:0);
                    else Assert.That(visible[point],Is.EqualTo(direct[0].pressureGain!=0?1:0));
                    var paths=solver.RayPaths;record.pathsPerSolve=paths.Length;
                    Assert.That(completeStart+settings.CompleteSamples*bins0,Is.EqualTo(paths.Length),"Unexpected single-pair path layout.");
                    var sampleFamilies=solver.CompleteSampleFamilies();
                    int cell=(s*points+point)*frequencies;
                    for(int i=0;i<paths.Length;i++)
                    {
                        var path=paths[i];
                        if(path.pressureGain==0 && path.pressureImag==0)continue;
                        int q=binIndex[path.frequencyBin];if(q<0)continue;
                        int family=i<bins0?0:i<walkStart?1:i<completeStart?7:-1;
                        if(family<0)
                        {
                            int route=sampleFamilies[(i-completeStart)/bins0];
                            if(route<0 || route>4)Assert.Fail("A nonzero complete-route arrival has no selected route.");
                            family=2+route;
                        }
                        var gain=new System.Numerics.Complex(path.pressureGain,path.pressureImag);
                        var value=gain*System.Numerics.Complex.Exp(new System.Numerics.Complex(0,-2*Math.PI*job.frequencies[q]*path.delaySeconds));
                        int at=(cell+q)*families+family;
                        values[2*at]+=value.Real;values[2*at+1]+=value.Imaginary;counts[at]++;
                        peaks[at]=Math.Max(peaks[at],(float)gain.Magnitude);
                    }
                    if(point==0)for(int q=0;q<frequencies;q++)
                    {
                        // Families must reconstruct the ordinary summed response exactly.
                        var total=EdgeResponse(paths,bins[q],job.frequencies[q]);var sum=System.Numerics.Complex.Zero;
                        for(int f=0;f<families;f++){int at=(cell+q)*families+f;sum+=new System.Numerics.Complex(values[2*at],values[2*at+1]);}
                        Assert.That((sum-total).Magnitude,Is.LessThanOrEqualTo(1e-9+1e-9*total.Magnitude),"Family split must reconstruct the summed field.");
                    }
                    double work=TakeRender.Elapsed(paced);
                    if(workFraction<1 && (work>=.05 || point+1==points))
                    {
                        record.busySeconds+=work;
                        System.Threading.Thread.Sleep((int)Math.Min(int.MaxValue,Math.Ceiling(work*1000*(1/workFraction-1))));
                        paced=System.Diagnostics.Stopwatch.GetTimestamp();
                    }
                }
                if(workFraction>=1)record.busySeconds+=TakeRender.Elapsed(paced);
                visibleKnown=true;
            }
            record.wallSeconds=TakeRender.Elapsed(budgetStarted);run.budgets[b]=record;
            foreach(double v in values)Assert.That(double.IsFinite(v),Is.True,"Non-finite family response.");
            WriteFloats(Path.Combine(partial,$"responses-{spec.name}.f32"),Array.ConvertAll(values,v=>(float)v));
            using(var writer=new BinaryWriter(File.Create(Path.Combine(partial,$"counts-{spec.name}.i64"))))foreach(long c in counts)writer.Write(c);
            WriteFloats(Path.Combine(partial,$"peaks-{spec.name}.f32"),peaks);
            WriteFloats(Path.Combine(partial,$"seconds-{spec.name}.f32"),seconds);
            Debug.Log($"Dataset {job.sceneId}: budget {spec.name} done in {record.wallSeconds:F1} s");
        }
        File.WriteAllBytes(Path.Combine(partial,"direct-visible.u8"),visible);
        run.wallSeconds=TakeRender.Elapsed(sceneStarted);
        File.WriteAllText(Path.Combine(partial,"run.json"),JsonUtility.ToJson(run,true),new System.Text.UTF8Encoding(false));
        // An acoustics-relevant error log leaves the render partial, so it is retried.
        Assert.That(datasetErrors,Is.Empty,"Error logged during a dataset render.");
        Directory.Move(partial,final);
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
        public DatasetSolid(WaveScene config,string[] materials,string output)
        {
            SolidPython();
            try
            {
                var source=new GameObject("source");var listener=new GameObject("listener");objects.Add(source);objects.Add(listener);
                var assets=new Dictionary<string,AudioMaterial>();
                AudioMaterial Material(string name)
                {
                    if(assets.TryGetValue(name,out var asset))return asset;
                    if(name=="rigid"){asset=RigidSolidMaterial("rigid");created.Add(asset);}
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

    // The dataset export classifies complete-route samples from the route tables of
    // the ordinary dispatch. It must agree with the separate debug capture.
    [Test, Category("GPU"), Category("CompleteRoutes")]
    public void CompleteSampleFamiliesMatchDebugCapture()
    {
        var shader=RequireGpu();
        var walls=new[]{(new Vector3(0,0,0),new Vector3(.2f,1.5f,1f)),(new Vector3(0,0,-2.5f),new Vector3(3,2,.2f))};
        var scene=Scene(1+walls.Length);scene.listeners=new[]{1};
        scene.materials=new[]{new AcousticTake.MaterialDefinition {name="rigid",frequencies=new[]{0f},absorption=new[]{0f},scattering=new[]{0f},transmission=new[]{0f}}};
        var state=States(1+walls.Length);state[0].position=new Vector3(-2,1,.3f);state[1].position=new Vector3(2,.8f,-.2f);
        var geometry=new List<AcousticTake.Primitive>();
        for(int w=0;w<walls.Length;w++)
        {
            int owner=w+2;state[owner].position=walls[w].Item1;state[owner].flags=AcousticTake.StateFlags.Active|AcousticTake.StateFlags.Geometry;
            scene.objects[owner].primitiveStart=geometry.Count;var mesh=TransportBox(scene.dimension,owner,0,1);scene.objects[owner].primitiveCount=mesh.Length;
            foreach(var triangle in mesh)
            {var p=triangle;p.a=Vector3.Scale(p.a,walls[w].Item2);p.b=Vector3.Scale(p.b,walls[w].Item2);p.c=Vector3.Scale(p.c,walls[w].Item2);geometry.Add(p);}
        }
        var settings=EdgeSettings();settings.useBvh=true;settings.airAbsorption=false;settings.fftSize=8;settings.spectralHop=4;
        settings.completeRouteSamples=4096;settings.raysPerBin=2;settings.maxBounces=4;settings.seed=29;
        settings.maximumDelaySeconds=.1f;settings.minimumDistance=.01f;
        using var solver=new TransportSolver(scene,geometry.ToArray(),settings,shader,sampleRate:1000);
        solver.Solve(state,new TakeRender.DirectPath[1]);
        var hook=solver.CompleteSampleFamilies();const int bin=1;
        var segments=solver.CaptureRays(state,0,bin);
        int start=geometry.Count+1+settings.RaysPerBin*(2+2*settings.maxBounces)+2*settings.MirrorCandidates(scene);
        var seen=new HashSet<int>();int captured=0;
        for(int sample=0;sample<settings.CompleteSamples;sample++)
        {
            var first=segments[start+3*sample];var second=segments[start+3*sample+1];
            if(first.kind==TransportSolver.RayEvent.None)continue;
            int family=first.kind==TransportSolver.RayEvent.Reflection?(second.kind==TransportSolver.RayEvent.Reflection?4:2):
                second.kind==TransportSolver.RayEvent.Listener?0:second.kind==TransportSolver.RayEvent.Diffraction?1:3;
            Assert.That(hook[sample],Is.EqualTo(family),$"Complete sample {sample} family differs from the debug capture.");
            seen.Add(family);captured++;
        }
        Assert.That(captured,Is.GreaterThan(100));
        Assert.That(seen.Count,Is.GreaterThanOrEqualTo(3),"The fixture must exercise several complete-route families.");
    }
}
