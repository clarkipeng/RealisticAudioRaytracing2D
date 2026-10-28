using System;
using System.Collections.Generic;
using System.IO;
using NUnit.Framework;
using UnityEditor;
using UnityEngine;
public partial class TakeChecks
{
    public static IEnumerable<string> ScalableCases()
    {
        var file=Environment.GetEnvironmentVariable("ACOUSTIC_SCALABLE_CASES");
        return string.IsNullOrEmpty(file)?new[]{""}:File.ReadAllLines(file);
    }
    [TestCaseSource(nameof(ScalableCases)),Category("QualityMeasurement"),Timeout(1800000)]
    public void MeasureScalableRoutes(string directory)
    {
        if(string.IsNullOrEmpty(directory))Assert.Ignore("Set ACOUSTIC_SCALABLE_CASES to archived route inputs.");
        Assert.That(Application.isBatchMode,Is.True);RequireGpu();long setup=System.Diagnostics.Stopwatch.GetTimestamp();
        using var input=new BinaryReader(File.OpenRead(Path.Combine(directory,"generated.bin")));
        Assert.That(input.ReadUInt32(),Is.EqualTo(0x52474731));int samples=input.ReadInt32(),depth=input.ReadInt32(),probes=input.ReadInt32(),boxes=input.ReadInt32(),capture=input.ReadInt32();
        Assert.That(depth,Is.EqualTo(2));Assert.That(samples>0&&(samples&(samples-1))==0,Is.True);
        int edges=checked(boxes*4),groups=(samples+63)/64;float wave=input.ReadSingle(),horizon=input.ReadSingle(),scale=input.ReadSingle();
        var source=new Vector4(input.ReadSingle(),input.ReadSingle(),0,0);
        var shader=UnityEngine.Object.Instantiate(Resources.Load<ComputeShader>("Acoustics/CompleteRoutes"));
        try
        {
            int preparePaths=shader.FindKernel("PreparePaths"),compactPaths=shader.FindKernel("CompactPaths");
            int emit=shader.FindKernel("EmitPaths"),emitMirrors=shader.FindKernel("EmitMirrors");
            int flagKernel=shader.FindKernel("BuildFlags"),compact=shader.FindKernel("Compact"),integrate=shader.FindKernel("IntegratePaths"),mirrors=shader.FindKernel("Mirrors"),reduce=shader.FindKernel("Reduce");
            using var geometry=new ComputeBuffer(Math.Max(1,boxes),16);using var flags=new ComputeBuffer(Math.Max(1,edges),4);using var lists=new ComputeBuffer(Math.Max(1,5*edges),4);using var counts=new ComputeBuffer(5,4);
            using var draw=new ComputeBuffer(samples*5,4);int routeCount=checked((int)(new FileInfo(Path.Combine(directory,"routes.bin")).Length/12));using var routeDraw=new ComputeBuffer(routeCount,12);
            string[] configurations=File.ReadAllLines(Path.Combine(directory,"config.txt"));int capacity=0;foreach(string config in configurations)capacity=Math.Max(capacity,int.Parse(config.Split(',')[1]));Assert.That(capacity,Is.InRange(1,samples));capacity=Math.Max(1,(int)Math.Min(capacity,2L*edges*edges+edges));
            using var dispatch=new ComputeBuffer(6,4,ComputeBufferType.IndirectArguments);using var prepared=new ComputeBuffer(capacity,112);using var selected=new ComputeBuffer(capacity,4);using var valid=new ComputeBuffer(1,4);
            using var arrivals=new ComputeBuffer(checked(3*samples+1+edges),RayTransport.PathStride);
            using var partial=new ComputeBuffer(groups,8);using var result=new ComputeBuffer(2,8);using var invalid=new ComputeBuffer(1,4);
            using var contributions=new ComputeBuffer(capture!=0?samples:1,8);using var mirrorContributions=new ComputeBuffer(capture!=0?samples:1,8);
            using var treeInput=new BinaryReader(File.OpenRead(Path.Combine(directory,"bvh.bin")));int nodeCount=checked((int)(treeInput.BaseStream.Length/32));
            using var nodes=new ComputeBuffer(Math.Max(1,nodeCount),32);
            if(boxes>0)geometry.SetData(RouteRead<Vector4>(input,boxes));draw.SetData(RouteRead<float>(input,samples*5));
            var receivers=RouteRead<Vector2>(input,probes);Assert.That(input.BaseStream.Position,Is.EqualTo(input.BaseStream.Length));
            if(nodeCount>0)nodes.SetData(RouteRead<uint>(treeInput,nodeCount*8));
            using(var random=new BinaryReader(File.OpenRead(Path.Combine(directory,"routes.bin"))))routeDraw.SetData(RouteRead<Vector3>(random,routeCount));
            shader.SetInt("BoxCount",boxes);shader.SetInt("EdgeCount",edges);shader.SetInt("NodeCount",nodeCount);shader.SetInt("SampleCount",samples);shader.SetInt("Dimensions",5);shader.SetInt("GroupCount",groups);shader.SetInt("Capture",capture);
            shader.SetFloat("WaveNumber",wave);shader.SetFloat("Horizon",horizon);shader.SetFloat("ProposalScale",scale);shader.SetVector("Source",source);
            foreach(int kernel in new[]{flagKernel,preparePaths,mirrors}){shader.SetBuffer(kernel,"Boxes",geometry);shader.SetBuffer(kernel,"Nodes",nodes);}
            foreach(int kernel in new[]{flagKernel,compact})shader.SetBuffer(kernel,"Flags",flags);
            shader.SetBuffer(compact,"DispatchArgs",dispatch);shader.SetBuffer(compact,"ListsOut",lists);shader.SetBuffer(compact,"CountsOut",counts);
            foreach(int kernel in new[]{preparePaths,mirrors}){shader.SetBuffer(kernel,"Lists",lists);shader.SetBuffer(kernel,"Counts",counts);shader.SetBuffer(kernel,"RouteDraws",routeDraw);shader.SetBuffer(kernel,"Partials",partial);}
            shader.SetBuffer(preparePaths,"PreparedOut",prepared);shader.SetBuffer(compactPaths,"PreparedIn",prepared);shader.SetBuffer(compactPaths,"Counts",counts);shader.SetBuffer(compactPaths,"SelectedOut",selected);shader.SetBuffer(compactPaths,"ValidOut",valid);
            shader.SetBuffer(integrate,"PreparedIn",prepared);shader.SetBuffer(integrate,"Selected",selected);shader.SetBuffer(integrate,"Valid",valid);shader.SetBuffer(integrate,"Partials",partial);
            shader.SetBuffer(integrate,"Draws",draw);shader.SetBuffer(integrate,"Invalid",invalid);shader.SetBuffer(integrate,"Contributions",contributions);shader.SetBuffer(integrate,"MirrorContributions",mirrorContributions);
            shader.SetBuffer(mirrors,"MirrorContributions",mirrorContributions);shader.SetBuffer(reduce,"Partials",partial);shader.SetBuffer(reduce,"Result",result);shader.SetBuffer(reduce,"Counts",counts);shader.SetBuffer(reduce,"Valid",valid);
            shader.SetBuffer(emit,"PreparedIn",prepared);shader.SetBuffer(emit,"Selected",selected);
            shader.SetBuffer(emit,"Valid",valid);shader.SetBuffer(emit,"Draws",draw);shader.SetBuffer(emit,"Invalid",invalid);
            foreach(int k in new[]{emit,emitMirrors})
            {shader.SetBuffer(k,"AudioArrivals",arrivals);shader.SetBuffer(k,"Boxes",geometry);}
            shader.SetBuffer(emitMirrors,"Nodes",nodes);shader.SetBuffer(emitMirrors,"Lists",lists);
            shader.SetBuffer(emitMirrors,"Counts",counts);shader.SetBuffer(emitMirrors,"RouteDraws",routeDraw);
            shader.SetFloat("AudioSpeed",343);shader.SetFloat("AudioReferenceDistance",1);
            shader.SetInt("AudioOffset",0);shader.SetInt("AudioBins",1);shader.SetInt("AudioIdentity",1);
            double setupSeconds=TakeRender.Elapsed(setup);long bytes=(long)geometry.count*16+(long)(flags.count+lists.count+counts.count+draw.count+invalid.count)*4+(long)routeDraw.count*12+(long)nodes.count*32+24+(long)prepared.count*112+(long)(selected.count+valid.count)*4+(long)(partial.count+result.count+contributions.count+mirrorContributions.count)*8;
            var status=new uint[1];var value=new Vector2[2];
            void Solve(int pathBudget)
            {
                status[0]=0;invalid.SetData(status);if(edges>0)shader.Dispatch(flagKernel,(edges+63)/64,1,1);shader.Dispatch(compact,1,1,1);
                shader.DispatchIndirect(preparePaths,dispatch,0);shader.Dispatch(compactPaths,1,1,1);
                shader.Dispatch(integrate,groups,1,1);shader.DispatchIndirect(mirrors,dispatch,12);shader.Dispatch(reduce,1,1,1);result.GetData(value);invalid.GetData(status);
                Assert.That(status[0],Is.Zero);Assert.That(float.IsFinite(value[0].x)&&float.IsFinite(value[0].y),Is.True);
            }
            using var csv=new StreamWriter(Path.Combine(directory,"scalable.csv"));csv.WriteLine("point,bvh,paths,real,imag,solve_seconds,input_seconds,setup_seconds,gpu_bytes,cold_seconds,candidates,accepted");
            long workStarted=System.Diagnostics.Stopwatch.GetTimestamp();
            for(int p=0;p<probes;p++)for(int rotation=0;rotation<configurations.Length;rotation++)
            {
                string[] config=configurations[(p+rotation)%configurations.Length].Split(',');int bvh=int.Parse(config[0]),pathBudget=int.Parse(config[1]);
                Assert.That(pathBudget,Is.InRange(1,samples));
                long clock=System.Diagnostics.Stopwatch.GetTimestamp();shader.SetVector("Receiver",new Vector4(receivers[p].x,receivers[p].y,0,0));shader.SetInt("UseBvh",bvh);shader.SetInt("PathBudget",pathBudget);double inputSeconds=TakeRender.Elapsed(clock),cold=0;
                if(p==0){clock=System.Diagnostics.Stopwatch.GetTimestamp();Solve(pathBudget);cold=TakeRender.Elapsed(clock);}clock=System.Diagnostics.Stopwatch.GetTimestamp();Solve(pathBudget);double seconds=TakeRender.Elapsed(clock);

                if(p==0)
                {
                    shader.Dispatch(emit,groups,1,1);shader.Dispatch(emitMirrors,(Math.Max(samples,edges)+63)/64,1,1);
                    var paths=new RayTransport.Path[arrivals.count];arrivals.GetData(paths);
                    double re=0,im=0;
                    foreach(var path in paths)
                    {
                        Assert.That(float.IsFinite(path.pressureGain),Is.True);
                        if(path.pressureGain==0)continue;
                        Assert.That(path.delaySeconds,Is.GreaterThan(0).And.LessThan(horizon/343));
                        Assert.That(new Vector3(path.direction.x,path.direction.y,path.direction.z).magnitude,Is.EqualTo(1).Within(2e-5));
                        double phase=-wave*343*path.delaySeconds;
                        re+=path.pressureGain*Math.Cos(phase);im+=path.pressureGain*Math.Sin(phase);
                    }
                    double delta=Math.Sqrt((re-value[0].x)*(re-value[0].x)+(im-value[0].y)*(im-value[0].y));
                    Assert.That(delta,Is.LessThan(2e-5+value[0].magnitude*2e-4),"Audio arrival gains/delays must reproduce the tuned shader's complex field.");
                }
                csv.WriteLine(FormattableString.Invariant($"{p},{bvh},{pathBudget},{value[0].x:R},{value[0].y:R},{seconds:R},{inputSeconds:R},{setupSeconds:R},{bytes},{cold:R},{value[1].x:R},{value[1].y:R}"));
                double workSeconds=TakeRender.Elapsed(workStarted);
                if(workSeconds>=.05 || (p+1==probes && rotation+1==configurations.Length))
                {System.Threading.Thread.Sleep((int)Math.Ceiling(workSeconds*4000));workStarted=System.Diagnostics.Stopwatch.GetTimestamp();}
                if(capture!=0)
                {
                    void Dump(ComputeBuffer buffer,string name){using var file=new BinaryWriter(File.Create(Path.Combine(directory,$"{name}-{p}-{bvh}-{pathBudget}.bin")));var v=new Vector2[samples];buffer.GetData(v);foreach(var z in v){file.Write(z.x);file.Write(z.y);}}
                    Dump(contributions,"edge");Dump(mirrorContributions,"mirror");
                    var c=new uint[5];counts.GetData(c);File.WriteAllText(Path.Combine(directory,$"counts-{p}-{bvh}-{pathBudget}.txt"),string.Join(",",c));var accepted=new uint[1];valid.GetData(accepted);File.WriteAllText(Path.Combine(directory,$"valid-{p}-{bvh}-{pathBudget}.txt"),accepted[0].ToString());
                }
            }
        }
        finally{UnityEngine.Object.DestroyImmediate(shader);}
    }
}
