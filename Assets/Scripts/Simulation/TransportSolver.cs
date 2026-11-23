using System;
using System.Diagnostics;
using System.Runtime.InteropServices;
using UnityEngine;

// One owner and at most one in-flight response batch. Steady live work does not wait;
// startup, shutdown and offline work may wait. Buffers outlive their request.
//
// Two transports: Raytraced runs the path method on the GPU (MethodTransport); it needs `shader`, the path
// method's compute shader (Resources/Acoustics/PathMethod.compute), which also stands for "GPU available" in
// the callers (GpuAudio.Selected, provenance). Direct is the CPU direct-sound reference (TakeRender.SolveDirect):
// retarded direct arrivals with opaque occlusion, no reflections.
public sealed class TransportSolver : IDisposable
{
    // The path method's compute shader: the `shader` that selects GPU transport (RayTraceManager.directShader,
    // offline render jobs) and the one MethodTransport dispatches.
    public const string PathShaderResource = "Acoustics/PathMethod";
    public static ComputeShader LoadPathShader() => Resources.Load<ComputeShader>(PathShaderResource);
    [StructLayout(LayoutKind.Sequential)]
    struct GpuPrimitive
    {
        public Vector4 a, b, c;
        public uint objectIndex, materialIndex, pad0, pad1;
    }
    [StructLayout(LayoutKind.Sequential)]
    struct GpuState
    {
        public Vector4 position, rotation, velocity, angularVelocity;
        public uint flags, pad0, pad1, pad2;
    }
    readonly AcousticTake.Scene scene;
    readonly AcousticTake.Primitive[] geometry;
    readonly TakeRender.Settings settings;
    readonly int pairs, maximumFrames;
    int activeFrames = 1;
    readonly AcousticTake.State[] cpuState;
    readonly TakeRender.DirectPath[] cpuPaths;
    readonly GpuState[] packed;
    readonly DirectBvh bvh;
    readonly MethodTransport method;
    // The path method of a Raytraced solver (null for Direct).
    internal MethodTransport Method => method;
    // Persistent rays: empty the rays in flight before frame `frame` of the next solve (a scene cut).
    public void FlushRays(int frame) => method?.FlushBefore(frame);
    // The exact arrivals (direct sound) as audio paths, and the late histogram of a solved frame.
    internal RayTransport.Path[] RayPaths => method != null && method.Audio ? method.Paths : null;
    internal ComputeBuffer RayBuffer => method?.PathBuffer;
    internal int RayPathsPerFrame => method != null && method.Audio ? method.PathsPerFrame : 0;
    internal RayTransport.LateFrame Late(int frame) => method != null && method.Audio ? new RayTransport.LateFrame(method, frame) : default;
    ComputeBuffer bvhNodes, bvhIndices;
    ComputeBuffer primitives, states;
    readonly TakeRender.DirectPath[] completedPaths;
    bool pending, disposed, readDirect = true;
    bool completed;
    long submitted;
    public bool IsPending => pending;
    public double CompletionSeconds { get; private set; }
    public double MaximumCompletionSeconds { get; private set; }
    public long PairPrimitiveUpperBound { get; private set; }
    public int SolveCount { get; private set; }
    public string Backend => method == null ? "CPU" : "GPU-" + SystemInfo.graphicsDeviceType;
    public long CpuBufferBytes =>
        (long)cpuState.Length * 80 +
        (long)(cpuPaths.Length + completedPaths.Length) * TakeRender.DirectPathStride +
        (packed == null ? 0 : (long)packed.Length * 80) + (bvh?.NumericBytes ?? 0) + (method?.CpuBytes ?? 0);
    public long GpuBufferBytes =>
        method == null
            ? 0
            : (long)Math.Max(1, geometry.Length) * 64 + (long)scene.objects.Length * 80 * maximumFrames +
                  (bvh == null
                       ? 0
                       : (long)Math.Max(1, bvh.Nodes.Length) * 48 + (long)Math.Max(1, geometry.Length) * 4) +
                  method.GpuBytes;
    public double BvhBuildSeconds => bvh?.BuildSeconds ?? 0;
    public double BvhUpdateSeconds => bvh?.UpdateSeconds ?? 0;
    public int BvhUpdates => bvh?.UpdateCount ?? 0;
    public int BvhPrimitiveCount => bvh?.PrimitiveCount ?? 0;
    public double UploadSeconds { get; private set; }
    public double DispatchReadbackSeconds { get; private set; }

    // methodAudio: the path method renders audio (paths and histograms for GpuAudio or the CPU mixer)
    // instead of measurement histograms read back synchronously.
    public TransportSolver(AcousticTake.Scene scene, AcousticTake.Primitive[] geometry,
                           TakeRender.Settings settings, ComputeShader shader = null, int maximumFrames = 1,
                           int sampleRate = 48000, bool readbackPaths = true, bool methodAudio = false)
    {
        TakeRender.Validate(settings);
        AudioSourcePlane.ValidateRender(scene,settings);
        this.scene = scene;
        this.geometry = geometry;
        this.settings = settings;
        pairs = checked(scene.sources.Length * scene.listeners.Length);
        if (maximumFrames < 1)
            throw new ArgumentOutOfRangeException(nameof(maximumFrames));
        this.maximumFrames = maximumFrames;
        completedPaths = new TakeRender.DirectPath[checked(pairs * maximumFrames)];
        cpuState = new AcousticTake.State[scene.objects.Length];
        cpuPaths = new TakeRender.DirectPath[pairs];
        if (settings.useBvh)
            bvh = new DirectBvh(scene, geometry);
        if (settings.transport == TakeRender.Transport.Direct)
            return;
        if (shader == null)
            throw new InvalidOperationException("Raytraced transport (the path method) requires its compute shader.");
        if (!SystemInfo.supportsComputeShaders || pairs == 0)
            throw new InvalidOperationException(
                "GPU transport requires compute support and source/listener pairs.");
        if (Marshal.SizeOf<GpuPrimitive>() != 64 || Marshal.SizeOf<GpuState>() != 80 ||
            Marshal.SizeOf<DirectBvh.Node>() != 48 ||
            Marshal.SizeOf<TakeRender.DirectPath>() != TakeRender.DirectPathStride)
            throw new InvalidOperationException("Transport solver GPU ABI mismatch.");
        try
        {
            var data = new GpuPrimitive[Math.Max(1, geometry.Length)];
            for (int i = 0; i < geometry.Length; i++)
                data[i] = new GpuPrimitive { a = geometry[i].a, b = geometry[i].b, c = geometry[i].c,
                                             objectIndex = (uint)geometry[i].objectIndex,
                                             materialIndex = (uint)geometry[i].materialIndex };
            primitives = new ComputeBuffer(data.Length, 64);
            primitives.SetData(data);
            packed = new GpuState[checked(scene.objects.Length * maximumFrames)];
            states = new ComputeBuffer(packed.Length, 80);
            if (bvh != null)
            {
                bvhNodes = new ComputeBuffer(Math.Max(1, bvh.Nodes.Length), 48);
                bvhIndices = new ComputeBuffer(Math.Max(1, geometry.Length), 4);
            }
            method = new MethodTransport(scene, geometry, settings, primitives, states, bvhNodes, bvhIndices, methodAudio, sampleRate, maximumFrames, readbackPaths);
        }
        catch
        {
            Dispose();
            throw;
        }
    }
    static readonly Unity.Profiling.ProfilerMarker solveMarker =
        new Unity.Profiling.ProfilerMarker("Acoustics.SolveTransport");
    public void Solve(AcousticTake.State[] state, TakeRender.DirectPath[] output)
    {
        using (solveMarker.Auto())
        {
            if (disposed || pending)
                throw new InvalidOperationException("Solver is disposed or has pending work.");
            if (state.Length != scene.objects.Length || output.Length != pairs)
                throw new ArgumentException("Direct solve shape mismatch.");
            activeFrames = 1;
            readDirect = true;
            if (method == null)
            {
                SolveCpu(state, output);
                CountSolve();
                return;
            }
            Upload(state, 1);
            long start = Stopwatch.GetTimestamp();
            // Diagnostic direct path on the CPU; the method traces its own direct sound.
            TakeRender.SolveDirect(scene, geometry, state, settings, output, bvh);
            method.PrepareFrames(state, 1);
            if (method.Audio) method.DispatchAudio(1, bvh?.NodeCount ?? 0, false);
            else method.Dispatch(bvh?.NodeCount ?? 0);
            DispatchReadbackSeconds += TakeRender.Elapsed(start);
            CountSolve();
        }
    }
    void CountSolve()
    {
        SolveCount += activeFrames;
        if (readDirect)
            PairPrimitiveUpperBound += checked((long)pairs * geometry.Length * activeFrames);
    }
    void SolveCpu(AcousticTake.State[] state, TakeRender.DirectPath[] output)
    {
        bvh?.Update(state);
        TakeRender.SolveDirect(scene, geometry, state, settings, output, bvh);
    }
    void Upload(AcousticTake.State[] state, int frames)
    {
        long start = Stopwatch.GetTimestamp();
        if (bvh != null)
        {
            if (bvh.Update(state, frames))
            {
                if (bvh.UpdateCount == 1 && geometry.Length > 0)
                    bvhIndices.SetData(bvh.Indices);
                if (bvh.NodeCount > 0)
                    bvhNodes.SetData(bvh.Nodes, 0, 0, bvh.UpdateCount == 1 ? bvh.NodeCount : bvh.TopNodeCount);
            }
        }
        for (int i = 0; i < frames * scene.objects.Length; i++)
            packed[i] =
                new GpuState { position = state[i].position,
                               rotation = new Vector4(state[i].rotation.x, state[i].rotation.y,
                                                      state[i].rotation.z, state[i].rotation.w),
                               velocity = state[i].velocity, angularVelocity = state[i].angularVelocity,
                               flags = (uint)state[i].flags };
        states.SetData(packed, 0, 0, frames * scene.objects.Length);
        UploadSeconds += TakeRender.Elapsed(start);
    }
    public void BeginSolve(AcousticTake.State[] state, int frames = 1, bool diagnostics = true)
    {
        using (solveMarker.Auto())
        {
            if (disposed || pending)
                throw new InvalidOperationException(
                    "Solver is disposed or already owns an in-flight response.");
            if (frames < 1 || frames > maximumFrames || state.Length < frames * scene.objects.Length)
                throw new ArgumentException("Direct solve shape mismatch.");
            if (method != null && !SystemInfo.supportsAsyncGPUReadback)
                throw new NotSupportedException(
                    "Live GPU tracing requires asynchronous readback; select CPU on this device.");
            if (method != null && !method.Audio)
                throw new InvalidOperationException("Method measurement solvers do not render audio.");
            if (!diagnostics && (method == null || method.Paths.Length != 0))
                throw new InvalidOperationException("GPU-only submission requires GPU transport and audio.");
            readDirect = diagnostics;
            activeFrames = frames;
            submitted = Stopwatch.GetTimestamp();
            completed = false;
            pending = true;
            try
            {
                if (method != null)
                {
                    // Diagnostic direct paths on the CPU (brute force, so the GPU frames' BVH
                    // upload below is untouched); the method traces its own arrivals.
                    if (readDirect)
                        for (int f = 0; f < frames; f++)
                        {
                            Array.Copy(state, f * cpuState.Length, cpuState, 0, cpuState.Length);
                            TakeRender.SolveDirect(scene, geometry, cpuState, settings, cpuPaths, null);
                            Array.Copy(cpuPaths, 0, completedPaths, f * pairs, pairs);
                        }
                    Upload(state, frames);
                    long start = Stopwatch.GetTimestamp();
                    method.PrepareFrames(state, frames);
                    method.DispatchAudio(frames, bvh?.NodeCount ?? 0, true);
                    DispatchReadbackSeconds += TakeRender.Elapsed(start);
                }
                else
                {
                    for (int f = 0; f < frames; f++)
                    {
                        Array.Copy(state, f * cpuState.Length, cpuState, 0, cpuState.Length);
                        SolveCpu(cpuState, cpuPaths);
                        Array.Copy(cpuPaths, 0, completedPaths, f * pairs, pairs);
                    }
                }
                completed = true;
                CountSolve();
            }
            catch
            {
                pending = false;
                throw;
            }
        }
    }
    public bool TryComplete(TakeRender.DirectPath[] output, bool wait = false)
    {
        if (!pending || disposed || !readDirect)
            throw new InvalidOperationException("No diagnostic response to complete.");
        if (output.Length < activeFrames * pairs)
            throw new ArgumentException("Direct response shape mismatch.");
        if (!completed || (method != null && !method.TryComplete(wait)))
            return false;
        pending = false;
        double elapsed = TakeRender.Elapsed(submitted);
        CompletionSeconds += elapsed;
        MaximumCompletionSeconds = Math.Max(MaximumCompletionSeconds, elapsed);
        TakeRender.ValidatePaths(completedPaths);
        Array.Copy(completedPaths, output, activeFrames * pairs);
        return true;
    }
    // The caller must first complete a GPU readback that follows this transport
    // batch on the same GPU queue. That readback also fences these scene buffers.
    internal void CompleteAfterGpuAudio()
    {
        if (!pending || disposed || readDirect)
            throw new InvalidOperationException("No GPU-only batch to complete.");
        pending = false;
        double elapsed = TakeRender.Elapsed(submitted);
        CompletionSeconds += elapsed;
        MaximumCompletionSeconds = Math.Max(MaximumCompletionSeconds, elapsed);
    }
    public void Dispose()
    {
        if (disposed)
            return;
        method?.Dispose();
        disposed = true;
        pending = false;
        primitives?.Dispose();
        states?.Dispose();
        bvhNodes?.Dispose();
        bvhIndices?.Dispose();
        bvhNodes = bvhIndices = null;
        primitives = states = null;
    }
}
