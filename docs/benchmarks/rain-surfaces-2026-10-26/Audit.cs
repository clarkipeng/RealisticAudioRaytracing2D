// Copy into the isolated runner's Assets/Tests/Editor for this numeric audit.
// Invoke RoofImpactAudit.Run with -impactTake, -impactOutput and -impactLegacyTake.
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using UnityEngine;

public static class RoofImpactAudit
{
    [Serializable] sealed class Report
    {
        public int candidates,hits,misses,maximumRetained,geometryTriangles;
        public double preparationSeconds;
        public long numericBytes;
        public string sceneHash,stateHash,eventHash,geometryHash,legacySceneHash;
        public string[] objects,materials;
        public int[] objectHits,materialHits;
    }
    static string Argument(string name) {var a=Environment.GetCommandLineArgs();int i=Array.IndexOf(a,name);return i<0?null:a[i+1];}
    public static void CaptureAndRun() { ChurchPresentationCapture.CaptureSequence(); Run(); }
    public static void Run()
    {
        if(!Application.isBatchMode||Application.isPlaying)throw new InvalidOperationException("Silent numeric audit only.");
        using var take=AcousticTake.Open(Argument("-impactTake"));var m=take.Metadata;var scene=m.scene;
        var p=scene.planes.Single(x=>x.generatorVersion==3);int source=p.source;
        var pcm=Enumerable.Range(0,scene.clips.Length).Select(take.ReadClip).ToArray();
        var signals=TakeRender.Signal.FromTake(take,pcm);
        var settings=new TakeRender.Settings{transport=TakeRender.Transport.Raytraced,fftSize=128,spectralHop=32,responseStepSamples=2048,maximumDelaySeconds=1.25f,maximumResponsesPerBatch=1,maxBounces=32,raysPerBin=2};
        var batch=new Emission.Batch(scene,settings,m.sampleRate,1,take.Geometry);var points=new SortedDictionary<long,Vector4>();
        var report=new Report{sceneHash=m.sceneHash,stateHash=m.stateHash,eventHash=m.eventHash,geometryHash=scene.geometryHash,geometryTriangles=take.Geometry.Length};
        void Pose(long sample,AcousticTake.State[] state)=>take.Evaluate(Math.Min(sample,m.durationSamples),state);
        var watch=System.Diagnostics.Stopwatch.StartNew();
        for(long output=0;output<m.durationSamples;output+=settings.responseStepSamples)
        {
            batch.Prepare(signals,output,1,Pose);var range=batch.Ranges[source];report.maximumRetained=Math.Max(report.maximumRetained,range.count);
            for(int n=0;n<range.count;n++)
            {
                int at=range.first+n;var clock=batch.Clocks[at];long birth=(long)(((ulong)clock.high<<32)|clock.low);
                if(points.TryGetValue(birth,out var old)&&old!=batch.Points[at])throw new Exception("Impact moved between responses.");
                points[birth]=batch.Points[at];
            }
        }
        report.preparationSeconds=watch.Elapsed.TotalSeconds;report.numericBytes=batch.NumericBytes;
        var bvh=new DirectBvh(scene,take.Geometry);var sampler=new Emission.PlaneSampler(p);var pose=new AcousticTake.State[scene.objects.Length];
        var objects=new int[scene.objects.Length];var materials=new int[scene.materials.Length];uint key=Emission.SourceKey(scene.objects[scene.sources[source].objectIndex].id,p.seed);
        string outputPath=Argument("-impactOutput");Directory.CreateDirectory(outputPath);
        using(var csv=new StreamWriter(Path.Combine(outputPath,"impacts.csv")))
        {
            csv.WriteLine("birth,x,y,z,amplitude,object,material");
            foreach(var epoch in signals[source].Epochs(0,m.durationSamples))for(long i=0;epoch.start+i*p.burstPeriodSamples<Math.Min(epoch.end,m.durationSamples);i++)
            {
                long birth=Emission.Birth(key,epoch.start,i,p.burstPeriodSamples);if(birth>=Math.Min(epoch.end,m.durationSamples))continue;
                report.candidates++;var local=sampler.Sample(Emission.Random(key,epoch.start,i,1),Emission.Random(key,epoch.start,i,2),Emission.Random(key,epoch.start,i,3),out _);
                Pose(birth,pose);var state=pose[scene.sources[source].objectIndex];var origin=state.position+state.rotation*(Vector3)local;bvh.Update(pose);
                if(!bvh.Raycast(origin,Vector3.down,pose,out var hit)){report.misses++;if(points.ContainsKey(birth))throw new Exception("Miss emitted audio.");continue;}
                report.hits++;if(!points.TryGetValue(birth,out var point)||Vector3.Distance(point,hit.position+hit.normal*Emission.SurfaceOffset)>1e-5)throw new Exception("Emission is not on first surface.");
                var primitive=take.Geometry[hit.primitive];objects[primitive.objectIndex]++;materials[primitive.materialIndex]++;
                csv.WriteLine(FormattableString.Invariant($"{birth},{point.x:R},{point.y:R},{point.z:R},{point.w:R},{primitive.objectIndex},{primitive.materialIndex}"));
            }
        }
        if(points.Count!=report.hits)throw new Exception("Unexpected or duplicate impacts.");
        report.objects=scene.objects.Select(x=>x.name).ToArray();report.objectHits=objects;
        report.materials=scene.materials.Select(x=>x.name).ToArray();report.materialHits=materials;
        using(var legacy=AcousticTake.Open(Argument("-impactLegacyTake")))report.legacySceneHash=legacy.Metadata.sceneHash;
        File.WriteAllText(Path.Combine(outputPath,"audit.json"),JsonUtility.ToJson(report,true));Debug.Log("ROOF_IMPACT_AUDIT "+report.hits+" hits / "+report.candidates+" candidates in "+report.preparationSeconds+" seconds");
    }
}
