using System;
using System.Diagnostics;
using System.Runtime.InteropServices;
using UnityEngine;
using UnityEngine.Rendering;

// One owner and at most one in-flight response batch. Live work never waits for readback;
// offline rendering and shutdown may explicitly wait. Buffers outlive their request.
public sealed class DirectSolver : IDisposable
{
    [StructLayout(LayoutKind.Sequential)] struct GpuPrimitive
    {
        public Vector4 a,b,c;
        public uint objectIndex,materialIndex,pad0,pad1;
    }
    [StructLayout(LayoutKind.Sequential)] struct GpuState
    {
        public Vector4 position,rotation;
        public uint flags,pad0,pad1,pad2;
    }
    readonly AcousticTake.Scene scene;
    readonly AcousticTake.Primitive[] geometry;
    readonly TakeRender.Settings settings;
    readonly ComputeShader shader;
    readonly int kernel,pairs,maximumFrames;
    int activeFrames=1;
    readonly AcousticTake.State[] cpuState;
    readonly TakeRender.DirectPath[] cpuPaths;
    readonly GpuState[] packed;
    ComputeBuffer primitives,states,sources,listeners,responses;
    readonly TakeRender.DirectPath[] completedPaths;
    readonly Action<AsyncGPUReadbackRequest> completion;
    AsyncGPUReadbackRequest request;
    bool pending,disposed;
    volatile bool completed;
    bool failed;
    long submitted;
    public bool IsPending => pending;
    public double CompletionSeconds {get;private set;}
    public double MaximumCompletionSeconds {get;private set;}
    public long PairPrimitiveUpperBound {get;private set;}
    public int SolveCount {get;private set;}
    public string Backend => shader==null ? "CPU":"GPU-"+SystemInfo.graphicsDeviceType;
    public long CpuBufferBytes => (long)cpuState.Length*80+(long)(cpuPaths.Length+completedPaths.Length)*TakeRender.DirectPathStride+(packed==null ? 0:(long)packed.Length*48);
    public long GpuBufferBytes => shader==null ? 0:(long)Math.Max(1,geometry.Length)*64+(long)scene.objects.Length*48*maximumFrames+(long)(scene.sources.Length+scene.listeners.Length)*4+(long)pairs*TakeRender.DirectPathStride*maximumFrames;
    public double UploadSeconds {get;private set;}
    public double DispatchReadbackSeconds {get;private set;}

    public DirectSolver(AcousticTake.Scene scene,AcousticTake.Primitive[] geometry,TakeRender.Settings settings,ComputeShader shader=null,int maximumFrames=1)
    {
        TakeRender.Validate(settings); this.scene=scene; this.geometry=geometry; this.settings=settings;
        pairs=checked(scene.sources.Length*scene.listeners.Length);
        if (maximumFrames<1) throw new ArgumentOutOfRangeException(nameof(maximumFrames));
        this.maximumFrames=maximumFrames;
        completedPaths=new TakeRender.DirectPath[checked(pairs*maximumFrames)]; completion=CompleteReadback;
        cpuState=new AcousticTake.State[scene.objects.Length]; cpuPaths=new TakeRender.DirectPath[pairs];
        if (shader==null) return;
        if (!SystemInfo.supportsComputeShaders || pairs==0) throw new InvalidOperationException("Direct GPU solve requires compute support and source/listener pairs.");
        if (Marshal.SizeOf<GpuPrimitive>()!=64 || Marshal.SizeOf<GpuState>()!=48 || Marshal.SizeOf<TakeRender.DirectPath>()!=TakeRender.DirectPathStride)
            throw new InvalidOperationException("Direct solver GPU ABI mismatch.");
        this.shader=UnityEngine.Object.Instantiate(shader);
        try
        {
            kernel=this.shader.FindKernel(scene.dimension==AcousticTake.Dimension.Planar2D ? "Direct2D":"Direct3D");
            var data=new GpuPrimitive[Math.Max(1,geometry.Length)];
            for (int i=0; i<geometry.Length; i++) data[i]=new GpuPrimitive {a=geometry[i].a,b=geometry[i].b,c=geometry[i].c,
                objectIndex=(uint)geometry[i].objectIndex,materialIndex=(uint)geometry[i].materialIndex};
            primitives=new ComputeBuffer(data.Length,64); primitives.SetData(data);
            packed=new GpuState[checked(scene.objects.Length*maximumFrames)]; states=new ComputeBuffer(packed.Length,48);
            var indices=new int[scene.sources.Length];
            for (int i=0; i<indices.Length; i++) indices[i]=scene.sources[i].objectIndex;
            sources=new ComputeBuffer(indices.Length,4); sources.SetData(indices);
            listeners=new ComputeBuffer(scene.listeners.Length,4); listeners.SetData(scene.listeners);
            responses=new ComputeBuffer(checked(pairs*maximumFrames),TakeRender.DirectPathStride);
            this.shader.SetBuffer(kernel,"Primitives",primitives); this.shader.SetBuffer(kernel,"States",states);
            this.shader.SetBuffer(kernel,"Sources",sources); this.shader.SetBuffer(kernel,"Listeners",listeners);
            this.shader.SetBuffer(kernel,"Responses",responses);
            this.shader.SetInt("PrimitiveCount",geometry.Length); this.shader.SetInt("PairCount",pairs);
            this.shader.SetInt("ListenerCount",scene.listeners.Length);
            this.shader.SetInt("ObjectCount",scene.objects.Length);
            this.shader.SetFloat("SpeedOfSound",settings.speedOfSound);
            this.shader.SetFloat("ReferenceDistance",settings.referenceDistance);
            this.shader.SetFloat("MinimumDistance",settings.minimumDistance);
        }
        catch { Dispose(); throw; }
    }
    static readonly Unity.Profiling.ProfilerMarker solveMarker=new Unity.Profiling.ProfilerMarker("Acoustics.SolveDirect");
    public void Solve(AcousticTake.State[] state,TakeRender.DirectPath[] output)
    {
        using (solveMarker.Auto())
        {
            if (disposed || pending) throw new InvalidOperationException("Solver is disposed or has pending work.");
            if (state.Length!=scene.objects.Length || output.Length!=pairs) throw new ArgumentException("Direct solve shape mismatch.");
            activeFrames=1;
            if (shader==null) { TakeRender.SolveDirect(scene,geometry,state,settings,output); CountSolve(); return; }
            Upload(state,1);
            long start=Stopwatch.GetTimestamp(); shader.Dispatch(kernel,(pairs+63)/64,1,1); responses.GetData(output);
            DispatchReadbackSeconds+=TakeRender.Elapsed(start);
            TakeRender.ValidatePaths(output); CountSolve();
        }
    }
    void CountSolve() { SolveCount+=activeFrames; PairPrimitiveUpperBound+=checked((long)pairs*geometry.Length*activeFrames); }
    void Upload(AcousticTake.State[] state,int frames)
    {
        long start=Stopwatch.GetTimestamp();
        for (int i=0;i<frames*scene.objects.Length;i++) packed[i]=new GpuState {position=state[i].position,
            rotation=new Vector4(state[i].rotation.x,state[i].rotation.y,state[i].rotation.z,state[i].rotation.w),flags=(uint)state[i].flags};
        states.SetData(packed,0,0,frames*scene.objects.Length); shader.SetInt("FrameCount",frames); UploadSeconds+=TakeRender.Elapsed(start);
    }
    public void BeginSolve(AcousticTake.State[] state,int frames=1)
    {
        using (solveMarker.Auto())
        {
        if (disposed || pending) throw new InvalidOperationException("Solver is disposed or already owns an in-flight response.");
        if (frames<1 || frames>maximumFrames || state.Length<frames*scene.objects.Length) throw new ArgumentException("Direct solve shape mismatch.");
        if (shader!=null && !SystemInfo.supportsAsyncGPUReadback) throw new NotSupportedException("Live GPU tracing requires asynchronous readback; select CPU on this device.");
        activeFrames=frames; submitted=Stopwatch.GetTimestamp(); completed=false; failed=false; pending=true;
        try
        {
            if (shader==null)
            {
                for (int f=0;f<frames;f++)
                {
                    Array.Copy(state,f*cpuState.Length,cpuState,0,cpuState.Length);
                    TakeRender.SolveDirect(scene,geometry,cpuState,settings,cpuPaths);
                    Array.Copy(cpuPaths,0,completedPaths,f*pairs,pairs);
                }
                completed=true;
            }
            else
            {
                Upload(state,frames);
                long start=Stopwatch.GetTimestamp(); shader.Dispatch(kernel,(frames*pairs+63)/64,1,1);
                request=AsyncGPUReadback.Request(responses,frames*pairs*TakeRender.DirectPathStride,0,completion);
                DispatchReadbackSeconds+=TakeRender.Elapsed(start);
            }
            CountSolve();
        }
        catch { pending=false; throw; }
        }
    }
    void CompleteReadback(AsyncGPUReadbackRequest value)
    {
        failed=value.hasError;
        if (!failed)
        {
            var data=value.GetData<TakeRender.DirectPath>();
            for (int i=0;i<data.Length;i++) completedPaths[i]=data[i];
        }
        completed=true; // Publish only after copying; request-owned memory expires after completion.
    }
    public bool TryComplete(TakeRender.DirectPath[] output,bool wait=false)
    {
        if (!pending || disposed) throw new InvalidOperationException("No owned response to complete.");
        if (output.Length<activeFrames*pairs) throw new ArgumentException("Direct response shape mismatch.");
        if (wait && !completed && shader!=null) request.WaitForCompletion();
        if (!completed) return false;
        pending=false;
        double elapsed=TakeRender.Elapsed(submitted); CompletionSeconds+=elapsed; MaximumCompletionSeconds=Math.Max(MaximumCompletionSeconds,elapsed);
        if (failed) throw new InvalidOperationException("GPU response readback failed.");
        TakeRender.ValidatePaths(completedPaths); Array.Copy(completedPaths,output,activeFrames*pairs); return true;
    }
    public void Dispose()
    {
        if (disposed) return;
        if (pending && shader!=null && !completed) request.WaitForCompletion();
        disposed=true; pending=false;
        primitives?.Dispose(); states?.Dispose(); sources?.Dispose(); listeners?.Dispose(); responses?.Dispose();
        primitives=states=sources=listeners=responses=null;
        if (shader!=null) { if (Application.isPlaying) UnityEngine.Object.Destroy(shader); else UnityEngine.Object.DestroyImmediate(shader); }
    }
}
