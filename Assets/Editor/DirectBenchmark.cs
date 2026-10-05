using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using UnityEditor;
using UnityEngine;

// Repeat whole offline jobs against one immutable take. No live scene or audio device is involved.
public static class DirectBenchmark
{
    [Serializable] public sealed class Sample
    {
        public string name,output;
        public int repetition;
        public double totalSeconds,openSeconds;
        public TakeRender.Report render;
    }
    [Serializable] public sealed class Summary
    {
        public string name;
        public double medianSeconds,minSeconds,maxSeconds,realtimeMultiple;
        public double prepareSeconds,solveSeconds,evaluateSeconds,audioSeconds,writeSeconds,openSeconds,finalizeSeconds;
        public long numericBytes;
    }
    [Serializable] public sealed class Report
    {
        public string revision,take,unity,processor,gpu,error;
        public string timing="Warm file caches; one discarded warm-up per case; alternating case order. Total includes take validation, render, output checksums and final report.";
        public string quality="Direct/opaque model, identical binaural DSP; Low/High differ only in response step. Not reflection quality or an approved training reference.";
        public double durationSeconds;
        public int lowStep,highStep,repetitions;
        public bool complete;
        public List<Sample> samples=new List<Sample>();
        public List<Summary> summaries=new List<Summary>();
        public TakeComparison.Metrics[] cpuGpuAgreement;
    }

    public static async void Run(string takePath,string output,string revision,TakeRender.Settings settings,
        int lowStep=2048,int highStep=256,int repetitions=3)
    {
        var report=new Report {revision=revision,take=Path.GetFullPath(takePath),unity=Application.unityVersion,
            processor=SystemInfo.processorType,gpu=SystemInfo.graphicsDeviceName,lowStep=lowStep,highStep=highStep,repetitions=repetitions};
        string reportPath=Path.Combine(output,"benchmark.json");
        bool created=false;
        try
        {
            if(Application.isPlaying)throw new InvalidOperationException("Stop Play before benchmarking offline renders.");
            if(Directory.Exists(output) || lowStep<=highStep || highStep<=0 || repetitions<1)throw new ArgumentException("Use a new output directory and valid response steps/repetitions.");
            Directory.CreateDirectory(output); created=true;
            var shader=AssetDatabase.LoadAssetAtPath<ComputeShader>("Assets/Shaders/Direct.compute");
            if(shader==null || !SystemInfo.supportsComputeShaders)throw new NotSupportedException("This comparison requires both CPU and GPU backends.");
            var manifest=AcousticTake.ReadManifest(takePath);
            report.durationSeconds=(double)(manifest.durationSamples+1)/manifest.sampleRate;
            var names=new[]{"cpu-low","cpu-high","gpu-low","gpu-high"};
            for(int iteration=-1;iteration<repetitions;iteration++)
            {
                for(int order=0;order<names.Length;order++)
                {
                    await Task.Delay(1); // Keep the Editor/bridge responsive between complete jobs.
                    int which=iteration%2==0 ? names.Length-1-order:order;
                    string name=names[which],path=Path.Combine(output,name,iteration<0 ? "warmup":"run-"+iteration);
                    var configuration=JsonUtility.FromJson<TakeRender.Settings>(JsonUtility.ToJson(settings));
                    configuration.responseStepSamples=which%2==0 ? lowStep:highStep;
                    var timer=Stopwatch.StartNew();
                    double opened; TakeRender.Report rendered;
                    using(var take=AcousticTake.Open(takePath))
                    {
                        opened=timer.Elapsed.TotalSeconds;
                        rendered=TakeRender.RenderDirect(take,configuration,path,null,which<2 ? null:shader);
                    }
                    var sample=new Sample {name=name,output=iteration==0?path:null,repetition=iteration,totalSeconds=timer.Elapsed.TotalSeconds,openSeconds=opened,render=rendered};
                    if(iteration>=0)
                    {
                        var previous=report.samples.Find(s=>s.name==name);
                        if(previous!=null && (!previous.render.audioHashes.SequenceEqual(rendered.audioHashes) || previous.render.responseHash!=rendered.responseHash))
                            throw new InvalidDataException("Repeated identical renders disagree: "+name);
                        report.samples.Add(sample);
                    }
                    File.WriteAllText(reportPath,JsonUtility.ToJson(report,true));
                    UnityEngine.Debug.Log("ACOUSTIC_BENCHMARK "+name+" "+iteration+": "+sample.totalSeconds.ToString("F3")+" s");
                    if(iteration!=0)Directory.Delete(path,true); // Keep one inspectable output, not duplicate benchmark WAVs.
                }
            }
            foreach(string name in names)
            {
                var samples=report.samples.Where(s=>s.name==name).OrderBy(s=>s.totalSeconds).ToArray();
                var middle=samples[samples.Length/2]; var r=middle.render;
                report.summaries.Add(new Summary {name=name,medianSeconds=middle.totalSeconds,minSeconds=samples[0].totalSeconds,
                    maxSeconds=samples[samples.Length-1].totalSeconds,realtimeMultiple=report.durationSeconds/middle.totalSeconds,
                    prepareSeconds=r.preparationSeconds,solveSeconds=r.solveSeconds,evaluateSeconds=r.evaluationSeconds,
                    audioSeconds=r.audioSeconds,writeSeconds=r.writeSeconds,openSeconds=middle.openSeconds,
                    finalizeSeconds=Math.Max(0,middle.totalSeconds-middle.openSeconds-r.wallSeconds),numericBytes=r.workingNumericBytes});
            }
            report.cpuGpuAgreement=new TakeComparison.Metrics[2*manifest.scene.listeners.Length];
            for(int quality=0;quality<2;quality++)for(int listener=0;listener<manifest.scene.listeners.Length;listener++)
            {
                string wave="run-0/listener-"+listener+".wav";
                report.cpuGpuAgreement[quality*manifest.scene.listeners.Length+listener]=TakeComparison.CompareWave(Path.Combine(output,names[quality],wave),Path.Combine(output,names[quality+2],wave));
            }
            report.complete=true;
        }
        catch(Exception e) {report.error=e.ToString();UnityEngine.Debug.LogException(e);}
        finally {if(created)File.WriteAllText(reportPath,JsonUtility.ToJson(report,true));}
    }
}
