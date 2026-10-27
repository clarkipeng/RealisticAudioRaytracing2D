using System;
using System.Diagnostics;
using System.Runtime.InteropServices;
using UnityEngine;
using UnityEngine.Rendering;

// One owner and at most one in-flight response batch. Steady live work does not wait;
// startup, shutdown and offline work may wait. Buffers outlive their request.
public sealed class TransportSolver : IDisposable
{
    // Optional capture ABI. Times are cumulative propagation seconds, not scene time.
    public enum RayEvent : uint
    {
        None,
        Transmission,
        Reflection,
        Scatter,
        Diffraction,
        Listener,
        Escape,
        Rejected
    }
    [StructLayout(LayoutKind.Sequential, Pack = 4)]
    public struct RaySegment
    {
        public Vector4 start, end; // xyz position, w travel time
        public RayEvent kind;
        public int material;
        public uint path;      // 0 direct; 1 + 2*ray mirror; 2 + 2*ray sampled walk
        public float pressure; // Signed estimator weight; final gain on listener connections.
    }
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
    readonly ComputeShader shader;
    readonly int kernel, pairs, maximumFrames;
    int activeFrames = 1;
    readonly AcousticTake.State[] cpuState;
    readonly TakeRender.DirectPath[] cpuPaths;
    readonly GpuState[] packed;
    readonly DirectBvh bvh;
    readonly RayTransport transport;
    internal RayTransport.Path[] RayPaths => transport?.Paths;
    internal ComputeBuffer RayBuffer => transport?.Buffer;
    internal void PrepareEmissions(TakeRender.Signal[] signals,long first,int frames,Action<long,AcousticTake.State[]> evaluate)
    {
        if(Emission.HasBursts(scene))transport.PrepareEmissions(signals,first,frames,evaluate);
    }
    internal int RayPathsPerFrame => transport?.PathsPerFrame ?? 0;
    ComputeBuffer bvhNodes, bvhIndices;
    ComputeBuffer primitives, states, sources, listeners, responses;
    readonly TakeRender.DirectPath[] completedPaths;
    readonly Action<AsyncGPUReadbackRequest> completion;
    AsyncGPUReadbackRequest request;
    bool pending, disposed, readDirect = true;
    volatile bool completed;
    bool failed;
    long submitted;
    public bool IsPending => pending;
    public double CompletionSeconds { get; private set; }
    public double MaximumCompletionSeconds { get; private set; }
    public long PairPrimitiveUpperBound { get; private set; }
    public int SolveCount { get; private set; }
    public string Backend => shader == null ? "CPU" : "GPU-" + SystemInfo.graphicsDeviceType;
    public long CpuBufferBytes =>
        (long)cpuState.Length * 80 +
        (long)(cpuPaths.Length + completedPaths.Length) * TakeRender.DirectPathStride +
        (packed == null ? 0 : (long)packed.Length * 80) + (bvh?.NumericBytes ?? 0) +
        (transport?.CpuBytes ?? 0);
    public long GpuBufferBytes =>
        shader == null
            ? 0
            : (long)Math.Max(1, geometry.Length) * 64 + (long)scene.objects.Length * 80 * maximumFrames +
                  (long)(scene.sources.Length + scene.listeners.Length) * 4 +
                  (long)pairs * TakeRender.DirectPathStride * maximumFrames +
                  (bvh == null
                       ? 0
                       : (long)Math.Max(1, bvh.Nodes.Length) * 48 + (long)Math.Max(1, geometry.Length) * 4) +
                  (transport?.GpuBytes ?? 0);
    public double BvhBuildSeconds => bvh?.BuildSeconds ?? 0;
    public double BvhUpdateSeconds => bvh?.UpdateSeconds ?? 0;
    public int BvhUpdates => bvh?.UpdateCount ?? 0;
    public int BvhPrimitiveCount => bvh?.PrimitiveCount ?? 0;
    public double UploadSeconds { get; private set; }
    public double DispatchReadbackSeconds { get; private set; }

    public TransportSolver(AcousticTake.Scene scene, AcousticTake.Primitive[] geometry,
                           TakeRender.Settings settings, ComputeShader shader = null, int maximumFrames = 1,
                           int sampleRate = 48000, bool readbackPaths = true)
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
        completion = CompleteReadback;
        cpuState = new AcousticTake.State[scene.objects.Length];
        cpuPaths = new TakeRender.DirectPath[pairs];
        if (settings.useBvh)
            bvh = new DirectBvh(scene, geometry);
        if (shader == null)
        {
            if (settings.transport == TakeRender.Transport.Raytraced)
                throw new InvalidOperationException("Raytraced transport requires Transport.compute.");
            return;
        }
        if (!SystemInfo.supportsComputeShaders || pairs == 0)
            throw new InvalidOperationException(
                "GPU transport requires compute support and source/listener pairs.");
        if (Marshal.SizeOf<GpuPrimitive>() != 64 || Marshal.SizeOf<GpuState>() != 80 ||
            Marshal.SizeOf<DirectBvh.Node>() != 48 ||
            Marshal.SizeOf<TakeRender.DirectPath>() != TakeRender.DirectPathStride)
            throw new InvalidOperationException("Transport solver GPU ABI mismatch.");
        this.shader = UnityEngine.Object.Instantiate(shader);
        try
        {
            kernel = this.shader.FindKernel(
                (scene.dimension == AcousticTake.Dimension.Planar2D ? "Direct2D" : "Direct3D") +
                (bvh == null ? "" : "Bvh"));
            if (!this.shader.IsSupported(kernel))
                throw new NotSupportedException(
                    "The selected direct compute kernel cannot run on this device.");
            var data = new GpuPrimitive[Math.Max(1, geometry.Length)];
            for (int i = 0; i < geometry.Length; i++)
                data[i] = new GpuPrimitive { a = geometry[i].a, b = geometry[i].b, c = geometry[i].c,
                                             objectIndex = (uint)geometry[i].objectIndex,
                                             materialIndex = (uint)geometry[i].materialIndex };
            primitives = new ComputeBuffer(data.Length, 64);
            primitives.SetData(data);
            packed = new GpuState[checked(scene.objects.Length * maximumFrames)];
            states = new ComputeBuffer(packed.Length, 80);
            var indices = new int[scene.sources.Length];
            for (int i = 0; i < indices.Length; i++)
                indices[i] = scene.sources[i].objectIndex;
            sources = new ComputeBuffer(indices.Length, 4);
            sources.SetData(indices);
            listeners = new ComputeBuffer(scene.listeners.Length, 4);
            listeners.SetData(scene.listeners);
            responses = new ComputeBuffer(checked(pairs * maximumFrames), TakeRender.DirectPathStride);
            this.shader.SetBuffer(kernel, "Primitives", primitives);
            this.shader.SetBuffer(kernel, "States", states);
            this.shader.SetBuffer(kernel, "Sources", sources);
            this.shader.SetBuffer(kernel, "Listeners", listeners);
            this.shader.SetBuffer(kernel, "Responses", responses);
            if (bvh != null)
            {
                bvhNodes = new ComputeBuffer(Math.Max(1, bvh.Nodes.Length), 48);
                bvhIndices = new ComputeBuffer(Math.Max(1, geometry.Length), 4);
                this.shader.SetBuffer(kernel, "BvhNodes", bvhNodes);
                this.shader.SetBuffer(kernel, "BvhIndices", bvhIndices);
            }
            this.shader.SetInt("PrimitiveCount", geometry.Length);
            this.shader.SetInt("PairCount", pairs);
            this.shader.SetInt("ListenerCount", scene.listeners.Length);
            this.shader.SetInt("ObjectCount", scene.objects.Length);
            this.shader.SetFloat("SpeedOfSound", settings.speedOfSound);
            this.shader.SetFloat("ReferenceDistance", settings.referenceDistance);
            this.shader.SetFloat("MinimumDistance", settings.minimumDistance);
            if (settings.transport == TakeRender.Transport.Raytraced)
                transport = new RayTransport(scene, geometry, settings, this.shader, primitives, states,
                                             sources, listeners, bvhNodes, bvhIndices, sampleRate,
                                             maximumFrames, readbackPaths);
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
            if (shader == null)
            {
                SolveCpu(state, output);
                CountSolve();
                return;
            }
            Upload(state, 1);
            long start = Stopwatch.GetTimestamp();
            shader.Dispatch(kernel, (pairs + 63) / 64, 1, 1);
            responses.GetData(output);
            transport?.Dispatch(1, false);
            DispatchReadbackSeconds += TakeRender.Elapsed(start);
            TakeRender.ValidatePaths(output);
            CountSolve();
        }
    }
    // Explicit synchronous inspection of one pair/bin. The normal shader variant has
    // no debug storage or readback. The caller owns the returned frozen CPU snapshot.
    public RaySegment[] CaptureRays(AcousticTake.State[] state, int pair, int frequencyBin)
    {
        if(AudioSourcePlane.MaximumSamples(scene)>1)
            throw new NotSupportedException("The single-origin ray debugger does not yet capture multi-sample planes.");
        if (disposed || pending || transport == null)
            throw new InvalidOperationException("Ray capture requires an idle GPU raytraced solver.");
        if (state.Length != scene.objects.Length || pair < 0 || pair >= pairs || frequencyBin < 0 ||
            frequencyBin >= settings.FrequencyBins)
            throw new ArgumentException("Ray capture shape or selection is invalid.");
        if (Marshal.SizeOf<RaySegment>() != 48)
            throw new InvalidOperationException("Ray capture GPU ABI mismatch.");
        int slots = checked(geometry.Length + 1 + settings.RaysPerBin * checked(2 + 2 * settings.maxBounces));
        slots=checked(slots+2*settings.MirrorCandidates(scene));
        var segments = new RaySegment[slots];
        using var buffer = new ComputeBuffer(slots, 48);
        buffer.SetData(segments);
        shader.EnableKeyword("ACOUSTIC_DEBUG");
        try
        {
            shader.SetInt("DebugPair", pair);
            shader.SetInt("DebugBin", frequencyBin);
            transport.BindDebug(buffer);
            Upload(state, 1);
            transport.Dispatch(1, false);
            buffer.GetData(segments);
        }
        finally
        {
            shader.DisableKeyword("ACOUSTIC_DEBUG");
        }
        return segments;
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
                shader.SetInt("BvhNodeCount", bvh.NodeCount);
            }
        }
        for (int i = 0; i < frames * scene.objects.Length; i++)
            packed[i] =
                new GpuState { position = state[i].position,
                               rotation = new Vector4(state[i].rotation.x, state[i].rotation.y,
                                                      state[i].rotation.z, state[i].rotation.w),
                               velocity = state[i].velocity, angularVelocity = state[i].angularVelocity,
                               flags = (uint)state[i].flags };
        transport?.SetActivity(scene, state, frames);
        states.SetData(packed, 0, 0, frames * scene.objects.Length);
        shader.SetInt("FrameCount", frames);
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
            if (shader != null && !SystemInfo.supportsAsyncGPUReadback)
                throw new NotSupportedException(
                    "Live GPU tracing requires asynchronous readback; select CPU on this device.");
            if (!diagnostics && (transport == null || transport.Paths.Length != 0))
                throw new InvalidOperationException("GPU-only submission requires GPU transport and audio.");
            readDirect = diagnostics;
            activeFrames = frames;
            submitted = Stopwatch.GetTimestamp();
            completed = false;
            failed = false;
            pending = true;
            try
            {
                if (shader == null)
                {
                    for (int f = 0; f < frames; f++)
                    {
                        Array.Copy(state, f * cpuState.Length, cpuState, 0, cpuState.Length);
                        SolveCpu(cpuState, cpuPaths);
                        Array.Copy(cpuPaths, 0, completedPaths, f * pairs, pairs);
                    }
                    completed = true;
                }
                else
                {
                    Upload(state, frames);
                    long start = Stopwatch.GetTimestamp();
                    if (readDirect)
                        shader.Dispatch(kernel, (frames * pairs + 63) / 64, 1, 1);
                    transport?.Dispatch(frames, true);
                    if (readDirect)
                        request = AsyncGPUReadback.Request(
                            responses, frames * pairs * TakeRender.DirectPathStride, 0, completion);
                    DispatchReadbackSeconds += TakeRender.Elapsed(start);
                }
                CountSolve();
            }
            catch
            {
                pending = false;
                throw;
            }
        }
    }
    void CompleteReadback(AsyncGPUReadbackRequest value)
    {
        failed = value.hasError;
        if (!failed)
        {
            var data = value.GetData<TakeRender.DirectPath>();
            Unity.Collections.NativeArray<TakeRender.DirectPath>.Copy(data, completedPaths, data.Length);
        }
        completed = true; // Publish only after copying; request-owned memory expires after completion.
    }
    public bool TryComplete(TakeRender.DirectPath[] output, bool wait = false)
    {
        if (!pending || disposed || !readDirect)
            throw new InvalidOperationException("No diagnostic response to complete.");
        if (output.Length < activeFrames * pairs)
            throw new ArgumentException("Direct response shape mismatch.");
        if (shader != null && !completed)
        {
            if (wait)
                request.WaitForCompletion();
            else
                request.Update();
        }
        if (!completed || (transport != null && !transport.TryComplete(wait)))
            return false;
        pending = false;
        double elapsed = TakeRender.Elapsed(submitted);
        CompletionSeconds += elapsed;
        MaximumCompletionSeconds = Math.Max(MaximumCompletionSeconds, elapsed);
        if (failed)
            throw new InvalidOperationException("GPU response readback failed.");
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
        if (pending && shader != null && !completed)
        {
            // Cancellation may dispose the solver before its audio consumer.
            // Fence only at shutdown; normal GPU playback reads back PCM alone.
            if (!readDirect)
                request = AsyncGPUReadback.Request(transport.Buffer, 4, 0);
            request.WaitForCompletion();
        }
        transport?.Dispose();
        disposed = true;
        pending = false;
        primitives?.Dispose();
        states?.Dispose();
        sources?.Dispose();
        listeners?.Dispose();
        responses?.Dispose();
        bvhNodes?.Dispose();
        bvhIndices?.Dispose();
        bvhNodes = bvhIndices = null;
        primitives = states = sources = listeners = responses = null;
        if (shader != null)
        {
            if (Application.isPlaying)
                UnityEngine.Object.Destroy(shader);
            else
                UnityEngine.Object.DestroyImmediate(shader);
        }
    }
}
