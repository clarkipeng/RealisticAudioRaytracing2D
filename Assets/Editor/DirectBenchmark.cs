using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using UnityEditor;
using UnityEngine;
using Complex=System.Numerics.Complex;

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
            var shader=AssetDatabase.LoadAssetAtPath<ComputeShader>("Assets/Shaders/Transport.compute");
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
                    configuration.transport=TakeRender.Transport.Direct; // This benchmark compares the CPU/GPU opaque baseline.
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

    // Swap source and receiver without changing the wall. Widely spaced offsets
    // keep the ray-index ranges separate at the benchmark budgets.
    public static object MeasureReciprocity(bool planar,int rays,float scattering,int seeds=8)
    {
        if(rays<1 || seeds<2 || !float.IsFinite(scattering) || scattering<0 || scattering>1)
            throw new ArgumentException("Use positive rays, at least two seeds and scattering in [0,1].");
        var shader=AssetDatabase.LoadAssetAtPath<ComputeShader>("Assets/Shaders/Transport.compute");
        var scene=new AcousticTake.Scene
        {
            dimension=planar?AcousticTake.Dimension.Planar2D:AcousticTake.Dimension.Spatial3D,
            objects=new[]
            {
                new AcousticTake.ObjectDefinition{id="source",scale=Vector3.one},
                new AcousticTake.ObjectDefinition{id="listener",scale=Vector3.one},
                new AcousticTake.ObjectDefinition{id="wall",scale=Vector3.one,primitiveCount=planar?1:2}
            },
            sources=new[]{new AcousticTake.SourceDefinition{objectIndex=0}},listeners=new[]{1},
            materials=new[]{new AcousticTake.MaterialDefinition
            {
                frequencies=new[]{0f},absorption=new[]{0f},transmission=new[]{0f},
                scattering=new[]{scattering},ior=new[]{1f},damping=new[]{0f}
            }}
        };
        var a=new Vector3(0,-10,planar?0:-10);var b=new Vector3(0,10,planar?0:-10);var c=new Vector3(0,10,10);var d=new Vector3(0,-10,10);
        var geometry=planar?new[]{new AcousticTake.Primitive{objectIndex=2,a=a,b=b}}:new[]
        {
            new AcousticTake.Primitive{objectIndex=2,a=a,b=b,c=c},
            new AcousticTake.Primitive{objectIndex=2,a=a,b=c,c=d}
        };
        var roles=new[]{AcousticTake.StateFlags.Source,AcousticTake.StateFlags.Listener,AcousticTake.StateFlags.Geometry};
        var states=new AcousticTake.State[roles.Length];
        for(int i=0;i<states.Length;i++)
            states[i]=new AcousticTake.State{rotation=Quaternion.identity,flags=AcousticTake.StateFlags.Active|roles[i]};
        var settings=new TakeRender.Settings
        {
            transport=TakeRender.Transport.Raytraced,fftSize=4,spectralHop=1,maxBounces=1,
            raysPerBin=rays,diffractionFactor=0,airAbsorption=false,maximumDelaySeconds=1
        };
        var samples=new double[seeds][];Complex forward=0,reverse=0;double differenceSquares=0;
        for(int seed=0;seed<samples.Length;seed++)
        {
            settings.seed=unchecked((int)((uint)seed*0x9e3779b9u));
            using var solver=new TransportSolver(scene,geometry,settings,shader,sampleRate:1000,readbackPaths:false);
            var values=new Complex[2];
            for(int direction=0;direction<2;direction++)
            {
                states[direction].position=new Vector3(-1,-1,0);states[1-direction].position=new Vector3(-4,2,0);
                foreach(var segment in solver.CaptureRays(states,0,1))
                    if(segment.kind==TransportSolver.RayEvent.Listener && segment.path!=0)
                        values[direction]+=segment.pressure*Complex.Exp(new Complex(0,-2*Math.PI*250*segment.end.w));
            }
            forward+=values[0];reverse+=values[1];double difference=(values[0]-values[1]).Magnitude;
            differenceSquares+=difference*difference;
            samples[seed]=new[]{values[0].Real,values[0].Imaginary,values[1].Real,values[1].Imaginary};
        }
        forward/=samples.Length;reverse/=samples.Length;double gap=(forward-reverse).Magnitude;
        double standardError=Math.Sqrt(Math.Max(0,(differenceSquares-samples.Length*gap*gap)/(samples.Length-1)/samples.Length));
        return new
        {
            scope="First-order reflected complex pressure at 250 Hz; stationary reciprocal wall, no diffraction or air loss. Necessary consistency check, not teacher approval.",
            planar,rays,scattering,samples,meanForward=new[]{forward.Real,forward.Imaginary},meanReverse=new[]{reverse.Real,reverse.Imaginary},
            reciprocityGap=gap,standardError,relativeGap=gap/Math.Max(forward.Magnitude,reverse.Magnitude)
        };
    }
}
