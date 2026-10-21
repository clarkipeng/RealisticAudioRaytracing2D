using System;
using System.Collections.Generic;
using System.IO;
using System.Runtime.InteropServices;
using NUnit.Framework;
using UnityEditor;
using UnityEngine;

public partial class TakeChecks
{
    [StructLayout(LayoutKind.Sequential)] struct RouteJob { public uint first,depth,end; public float weight; }
    [StructLayout(LayoutKind.Sequential)] struct RouteStep { public Vector4 shape,sine,halfSine; }
    static T[] RouteRead<T>(BinaryReader reader,int count) where T:struct
    {
        int bytes=checked(count*Marshal.SizeOf<T>());var raw=reader.ReadBytes(bytes);
        if(raw.Length!=bytes)throw new EndOfStreamException();
        var result=new T[count];if(count==0)return result;
        var pinned=GCHandle.Alloc(result,GCHandleType.Pinned);
        try {Marshal.Copy(raw,0,pinned.AddrOfPinnedObject(),bytes);} finally {pinned.Free();}
        return result;
    }
    public static IEnumerable<string> RouteCases()
    {
        string file=Environment.GetEnvironmentVariable("ACOUSTIC_ROUTE_CASES");
        return string.IsNullOrEmpty(file)?new[]{""}:File.ReadAllLines(file);
    }
    [Test,Category("GPU")]
    public void RouteAnglesMatchReference()
    {
        RequireGpu();var values=new List<Vector2>();
        for(int sign=-1;sign<=1;sign++)for(int power=-8;power<=8;power++)
            values.Add(new Vector2(Mathf.Pow(10,power),sign));
        for(int i=1;i<=64;i++)values.Add(new Vector2(i/32f,1));
        using var input=new ComputeBuffer(values.Count,8);using var output=new ComputeBuffer(values.Count,8);
        var shader=AssetDatabase.LoadAssetAtPath<ComputeShader>("Assets/Tests/Editor/RouteIntegration.compute");
        int kernel=shader.FindKernel("CheckAngles");input.SetData(values);
        shader.SetBuffer(kernel,"AngleCases",input);shader.SetBuffer(kernel,"Contributions",output);
        shader.Dispatch(kernel,values.Count,1,1);var result=new Vector2[values.Count];output.GetData(result);
        double nativeError=0,stableError=0;
        for(int i=0;i<values.Count;i++)
        {
            double expected=Math.Atan2(values[i].x,values[i].y);
            nativeError=Math.Max(nativeError,Math.Abs(result[i].y-expected));
            stableError=Math.Max(stableError,Math.Abs(result[i].x-expected));
            Assert.That(Math.Abs(result[i].x-expected),Is.LessThan(expected*3e-7),$"CDF span {values[i]}");
        }
        Debug.Log(FormattableString.Invariant($"Route CDF angle max error: native={nativeError:R}, stable={stableError:R} radians; {values.Count} cases."));
    }
    // Opt-in numerical experiment; Python checks parity against the CPU estimator.
    [TestCaseSource(nameof(RouteCases)),Category("QualityMeasurement")]
    public void MeasureCompleteRoutes(string directory)
    {
        if(string.IsNullOrEmpty(directory))Assert.Ignore("Set ACOUSTIC_ROUTE_CASES to prepared case directories.");
        Assert.That(Application.isBatchMode,Is.True);RequireGpu();
        Assert.That(Marshal.SizeOf<RouteJob>(),Is.EqualTo(16));Assert.That(Marshal.SizeOf<RouteStep>(),Is.EqualTo(48));
        using var input=new BinaryReader(File.OpenRead(Path.Combine(directory,"requests.bin")));
        Assert.That(input.ReadUInt32(),Is.EqualTo(0x524d4331));
        int dimensions=input.ReadInt32(),draws=input.ReadInt32(),probes=input.ReadInt32(),
            maxRoutes=input.ReadInt32(),maxSteps=input.ReadInt32(),maxSamples=input.ReadInt32(),capture=input.ReadInt32();
        float waveNumber=input.ReadSingle(),horizon=input.ReadSingle();
        var shader=UnityEngine.Object.Instantiate(AssetDatabase.LoadAssetAtPath<ComputeShader>("Assets/Tests/Editor/RouteIntegration.compute"));
        try
        {
            int integrate=shader.FindKernel("Integrate"),reduce=shader.FindKernel("Reduce");
            Assert.That(shader.IsSupported(integrate)&&shader.IsSupported(reduce),Is.True);
            using var random=new ComputeBuffer(checked(draws*dimensions),4);
            using var routes=new ComputeBuffer(Math.Max(1,maxRoutes),16);
            using var steps=new ComputeBuffer(Math.Max(1,maxSteps),48);
            using var partial=new ComputeBuffer(Math.Max(1,(maxSamples+63)/64),8);
            using var result=new ComputeBuffer(1,8);
            using var invalid=new ComputeBuffer(1,4);
            using var contributions=new ComputeBuffer(capture!=0?Math.Max(1,maxSamples):1,8);
            random.SetData(RouteRead<float>(input,checked(draws*dimensions)));
            shader.SetBuffer(integrate,"Draws",random);shader.SetBuffer(integrate,"Routes",routes);
            shader.SetBuffer(integrate,"Steps",steps);shader.SetBuffer(integrate,"Invalid",invalid);
            shader.SetBuffer(integrate,"Contributions",contributions);
            shader.SetBuffer(integrate,"Partials",partial);shader.SetBuffer(reduce,"Partials",partial);
            shader.SetBuffer(reduce,"Result",result);
            shader.SetInt("Dimensions",dimensions);shader.SetInt("Capture",capture);
            shader.SetFloat("WaveNumber",waveNumber);shader.SetFloat("Horizon",horizon);
            using var csv=new StreamWriter(Path.Combine(directory,"gpu.csv"));
            using var debug=capture!=0?new BinaryWriter(File.Create(Path.Combine(directory,"contributions.bin"))):null;
            csv.WriteLine("point,real,imag,input_seconds,upload_seconds,solve_seconds,gpu_bytes");
            var value=new Vector2[1];var status=new uint[1];
            long gpuBytes=(long)random.count*4+(long)routes.count*16+(long)steps.count*48+(long)partial.count*8+12+(long)contributions.count*8;
            void Solve(int samples)
            {
                status[0]=0;invalid.SetData(status);int groups=(samples+63)/64;
                shader.SetInt("SampleCount",samples);shader.SetInt("GroupCount",groups);
                if(groups>0)shader.Dispatch(integrate,groups,1,1);
                shader.Dispatch(reduce,1,1,1);result.GetData(value);invalid.GetData(status);
                Assert.That(status[0],Is.Zero,"Non-finite sample contributions.");
                Assert.That(float.IsFinite(value[0].x)&&float.IsFinite(value[0].y),Is.True);
            }
            for(int point=0;point<probes;point++)
            {
                long start=System.Diagnostics.Stopwatch.GetTimestamp();
                int routeCount=input.ReadInt32(),stepCount=input.ReadInt32(),samples=input.ReadInt32();
                var jobs=RouteRead<RouteJob>(input,routeCount);var data=RouteRead<RouteStep>(input,stepCount);
                double readSeconds=TakeRender.Elapsed(start);start=System.Diagnostics.Stopwatch.GetTimestamp();
                if(routeCount>0)routes.SetData(jobs,0,0,routeCount);
                if(stepCount>0)steps.SetData(data,0,0,stepCount);
                shader.SetInt("RouteCount",routeCount);double uploadSeconds=TakeRender.Elapsed(start);
                if(point==0)Solve(samples); // Exclude first dispatch/compilation from warm timing.
                start=System.Diagnostics.Stopwatch.GetTimestamp();Solve(samples);double seconds=TakeRender.Elapsed(start);
                csv.WriteLine(FormattableString.Invariant($"{point},{value[0].x:R},{value[0].y:R},{readSeconds:R},{uploadSeconds:R},{seconds:R},{gpuBytes}"));
                if(debug!=null)
                {
                    var values=new Vector2[samples];if(samples>0)contributions.GetData(values,0,0,samples);
                    debug.Write(samples);foreach(var v in values){debug.Write(v.x);debug.Write(v.y);}
                }
            }
            Assert.That(input.BaseStream.Position,Is.EqualTo(input.BaseStream.Length));
        }
        finally {UnityEngine.Object.DestroyImmediate(shader);}
    }
}
