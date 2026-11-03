using System;
using System.IO;
using NUnit.Framework;
using UnityEngine;

// Opt-in residual-walk diagnostic: every nonzero walk arrival of selected dataset
// receivers with its ray, bounce slot and complex value, plus the walk's captured
// event segments for one bin and the scene's edge records. Driven by
// docs/benchmarks/residual-walk-2026-11-02/walk.py; measures, never asserts quality.
public partial class TakeChecks
{
    [Serializable] sealed class WalkJob
    {
        public string scene, variant, output;
        public int[] receivers, seeds;
        public int raysPerBin, completeRouteSamples, captureBin=-1, maxBounces=-1, topRays=64, fftSize=-1, sampleRate=-1;
        public float diffractionFactor=1;
    }

    [Test, Category("QualityMeasurement"), Timeout(int.MaxValue)]
    public void MeasureResidualWalk()
    {
        string jobs=Environment.GetEnvironmentVariable("ACOUSTIC_WALK_JOBS");
        if(string.IsNullOrEmpty(jobs))Assert.Ignore("Set ACOUSTIC_WALK_JOBS to a file listing walk job JSON files.");
        Assert.That(Application.isBatchMode,Is.True,"Use an isolated batch project.");
        AudioListener.volume=0;
        string budget=Environment.GetEnvironmentVariable("ACOUSTIC_DATASET_WORK_FRACTION");
        double workFraction=string.IsNullOrEmpty(budget)?1:double.Parse(budget,System.Globalization.CultureInfo.InvariantCulture);
        Assert.That(workFraction,Is.GreaterThan(0).And.LessThanOrEqualTo(1));
        var shader=RequireGpu();
        foreach(string line in File.ReadAllLines(jobs))if(line.Trim().Length>0)RunWalkJob(JsonUtility.FromJson<WalkJob>(File.ReadAllText(line.Trim())),shader,workFraction);
    }

    void RunWalkJob(WalkJob job,ComputeShader shader,double workFraction)
    {
        string partial=job.output+".partial";
        Assert.That(Directory.Exists(job.output),Is.False,"Completed diagnostics are never overwritten.");
        if(Directory.Exists(partial))Directory.Delete(partial,true);
        Directory.CreateDirectory(partial);
        string text=File.ReadAllText(Path.Combine(job.scene,"scene.json"));
        var config=JsonUtility.FromJson<WaveScene>(text);var scene=JsonUtility.FromJson<DatasetScene>(text);
        var names=job.variant=="tracer"?scene.surfaceMaterials:Array.ConvertAll(scene.surfaceMaterials,_=>"rigid");
        using var baked=new DatasetSolid(config,names,partial);
        var state=baked.states;var direct=new TakeRender.DirectPath[1];
        foreach(int seed in job.seeds)
        {
            var settings=EdgeSettings();settings.useBvh=true;settings.airAbsorption=false;
            int fft=job.fftSize>0?job.fftSize:scene.fftSize,rate=job.sampleRate>0?job.sampleRate:scene.sampleRate;
            settings.fftSize=fft;settings.spectralHop=fft/2;
            settings.completeRouteSamples=job.completeRouteSamples;settings.raysPerBin=job.raysPerBin;
            settings.seed=seed;settings.maxBounces=job.maxBounces>0?job.maxBounces:scene.maxBounces;settings.diffractionFactor=job.diffractionFactor;
            settings.speedOfSound=scene.speedOfSound;settings.maximumDelaySeconds=scene.maximumDelaySeconds;settings.minimumDistance=.01f;
            using var solver=new TransportSolver(baked.scene,baked.geometry,settings,shader,sampleRate:rate);
            int bins=settings.FrequencyBins,walkStart=bins*(1+settings.MirrorCandidates(baked.scene));
            int walkEnd=walkStart+settings.RaysPerPair*settings.PathSlotsPerRay;
            float step=(float)rate/fft;
            foreach(int receiver in job.receivers)
            {
                long paced=System.Diagnostics.Stopwatch.GetTimestamp();
                state[1].position=WaveVector(scene.points,3*receiver);
                long started=System.Diagnostics.Stopwatch.GetTimestamp();solver.Solve(state,direct);double seconds=TakeRender.Elapsed(started);
                var paths=solver.RayPaths;
                // Early field (every slot outside the walk) and walk arrivals, one row each:
                // ray in bin, bounce slot, bin, Re, Im of gain*exp(-i w delay), delay.
                var early=new double[2*bins];var rows=new System.Collections.Generic.List<float>();
                for(int i=0;i<paths.Length;i++)
                {
                    var p=paths[i];if(p.pressureGain==0 && p.pressureImag==0)continue;
                    double w=-2*Math.PI*p.frequencyBin*step*p.delaySeconds,re=p.pressureGain*Math.Cos(w)-p.pressureImag*Math.Sin(w),im=p.pressureGain*Math.Sin(w)+p.pressureImag*Math.Cos(w);
                    if(i<walkStart||i>=walkEnd){early[2*p.frequencyBin]+=re;early[2*p.frequencyBin+1]+=im;continue;}
                    int ray=(i-walkStart)/settings.PathSlotsPerRay;
                    rows.AddRange(new[]{ray%settings.RaysPerBin,(i-walkStart)%settings.PathSlotsPerRay,ray/settings.RaysPerBin,(float)re,(float)im,p.delaySeconds});
                }
                string stem=Path.Combine(partial,$"r{receiver}-s{seed}");
                WriteFloats(stem+"-walk.f32",rows.ToArray());
                WriteFloats(stem+"-early.f32",Array.ConvertAll(early,v=>(float)v));
                File.WriteAllText(stem+"-seconds.txt",seconds.ToString("R",System.Globalization.CultureInfo.InvariantCulture));
                // Captured event segments (start xyzw, end xyzw, kind, material, path, weight) of
                // every walk ray in captureBin, and of the rays holding the largest arrivals.
                int perRay=2+2*settings.maxBounces,first=baked.geometry.Length+1;
                var order=new System.Collections.Generic.List<int>();for(int r=0;r<rows.Count/6;r++)order.Add(r);
                order.Sort((a,b)=>Magnitude(rows,b).CompareTo(Magnitude(rows,a)));
                var wanted=new System.Collections.Generic.SortedDictionary<int,System.Collections.Generic.SortedSet<int>>();
                if(job.captureBin>=0){wanted[job.captureBin]=new();for(int r=0;r<settings.RaysPerBin;r++)wanted[job.captureBin].Add(r);}
                for(int k=0;k<Math.Min(job.topRays,order.Count);k++)
                {
                    int bin=(int)rows[6*order[k]+2];
                    if(!wanted.TryGetValue(bin,out var set))wanted[bin]=set=new();
                    set.Add((int)rows[6*order[k]]);
                }
                var flat=new System.Collections.Generic.List<float>();
                foreach(var item in wanted)
                {
                    var segments=solver.CaptureRays(state,0,item.Key);
                    foreach(int ray in item.Value)
                    {
                        flat.Add(item.Key);flat.Add(ray);
                        for(int s=0;s<perRay;s++)
                        {
                            var g=segments[first+ray*perRay+s];
                            flat.AddRange(new[]{g.start.x,g.start.y,g.start.z,g.start.w,g.end.x,g.end.y,g.end.z,g.end.w,(float)g.kind,g.material,g.path,g.pressure});
                        }
                    }
                }
                WriteFloats(stem+"-capture.f32",flat.ToArray());
                if(workFraction<1)System.Threading.Thread.Sleep((int)Math.Ceiling(TakeRender.Elapsed(paced)*1000*(1/workFraction-1)));
            }
            if(seed==job.seeds[0])
            {
                WriteEdges(solver,Path.Combine(partial,"edges.f32"));
                var triangles=new System.Collections.Generic.List<float>();
                foreach(var t in baked.geometry)
                {
                    var pose=state[t.objectIndex];
                    foreach(var v in new[]{t.a,t.b,t.c}){var w=pose.position+pose.rotation*v;triangles.AddRange(new[]{w.x,w.y,w.z});}
                    triangles.Add(t.objectIndex);triangles.Add(t.materialIndex);
                }
                WriteFloats(Path.Combine(partial,"triangles.f32"),triangles.ToArray());
            }
        }
        File.WriteAllText(Path.Combine(partial,"job.json"),JsonUtility.ToJson(job,true));
        Directory.Move(partial,job.output);
    }

    static float Magnitude(System.Collections.Generic.List<float> rows,int r)=>rows[6*r+3]*rows[6*r+3]+rows[6*r+4]*rows[6*r+4];

    // Edge records exactly as the transport shader reads them (a, b, face, normal, ids).
    static void WriteEdges(TransportSolver solver,string path)
    {
        var flags=System.Reflection.BindingFlags.NonPublic|System.Reflection.BindingFlags.Instance;
        var transport=typeof(TransportSolver).GetField("transport",flags).GetValue(solver);
        var buffer=(ComputeBuffer)transport.GetType().GetField("edgeBuffer",flags).GetValue(transport);
        var raw=new float[buffer.count*20];buffer.GetData(raw);
        WriteFloats(path,raw);
    }
}
