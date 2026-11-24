using System;
using System.Runtime.InteropServices;
using UnityEngine;
using UnityEngine.Rendering;

// The path walk's scene queries on Metal's ray-tracing hardware (README part 52): a scene in Native/AcousticRT (one acceleration
// structure per object in its own frame, an instance structure over them posed from the State buffer on the GPU), and the
// render-thread events that rebuild the instance structure and answer a wave's query records, on Unity's own command buffer.
// The walk's kernels write the records and read the answers (MethodPath.hlsl "Waves"); only the intersection runs in Metal.
internal sealed class HardwareQueries : IDisposable
{
    const string Plugin = "AcousticRT";
    [DllImport(Plugin)] static extern int ARTSupported();
    [DllImport(Plugin)] static extern int ARTLastError(byte[] buffer, int capacity);
    [DllImport(Plugin)] static extern int ARTCreate(float[] vertices, int[] triangleObject, int triangleCount, int objectCount);
    [DllImport(Plugin)] static extern int ARTSetBuffers(int handle, IntPtr queries, int queryOffset, IntPtr answers, IntPtr counts, int countIndex,
                                                        IntPtr args, int argsOffset, IntPtr states, int statesOffset);
    [DllImport(Plugin)] static extern IntPtr ARTEventFunction();

    const int EventBuild = 1, EventIntersect = 2, EventDestroy = 3;
    const int MaximumHandles = 0xfff;

    static int availability = -1;
    // Unity on Metal with a device that has ray tracing, and the plugin loaded.
    public static bool Available
    {
        get
        {
            if (availability >= 0) return availability == 1;
            try { availability = SystemInfo.graphicsDeviceType == GraphicsDeviceType.Metal && ARTSupported() == 1 ? 1 : 0; }
            catch (Exception e) when (e is DllNotFoundException || e is EntryPointNotFoundException) { availability = 0; }
            return availability == 1;
        }
    }
    // Throws if the plugin reported an error since the last check (its events run on the render thread and cannot throw).
    public static void ThrowIfFailed() { string e = LastError(); if (e.Length > 0) throw new InvalidOperationException("AcousticRT: " + e); }
    static string LastError() { var b = new byte[512]; int n = ARTLastError(b, b.Length); return System.Text.Encoding.UTF8.GetString(b, 0, n); }

    readonly int handle;
    readonly CommandBuffer commands = new() {name = "Acoustics.HardwareQueries"};
    bool disposed;

    // The scene's triangles in their objects' own frames (the State buffer poses them).
    public HardwareQueries(AcousticTake.Primitive[] geometry, int objectCount)
    {
        if (!Available) throw new NotSupportedException("Hardware ray queries need Unity on Metal with a ray-tracing device.");
        var vertices = new float[9 * geometry.Length]; var owner = new int[geometry.Length];
        for (int i = 0; i < geometry.Length; i++)
        {
            var g = geometry[i]; int at = 9 * i;
            vertices[at] = g.a.x; vertices[at + 1] = g.a.y; vertices[at + 2] = g.a.z;
            vertices[at + 3] = g.b.x; vertices[at + 4] = g.b.y; vertices[at + 5] = g.b.z;
            vertices[at + 6] = g.c.x; vertices[at + 7] = g.c.y; vertices[at + 8] = g.c.z;
            owner[i] = g.objectIndex;
        }
        handle = ARTCreate(vertices, owner, geometry.Length, objectCount);
        if (handle <= 0 || handle > MaximumHandles) throw new InvalidOperationException("AcousticRT: " + LastError());
    }

    // The walk's buffers: query records in `wave` from uint `queryBase`, the query count at uint `countIndex` and the frame's poses
    // (State records) at uint `posesBase` of it, the intersection's indirect arguments (the first three uints of `arguments`), and
    // the answers.
    // Native pointers are taken when the buffers hold data (Unity may replace a buffer behind its pointer when it uploads to it).
    public void Bind(ComputeBuffer wave, int queryBase, int countIndex, ComputeBuffer arguments, int posesBase, ComputeBuffer answers)
    {
        IntPtr w = wave.GetNativeBufferPtr();
        if (ARTSetBuffers(handle, w, 4 * queryBase, answers.GetNativeBufferPtr(), w, countIndex, arguments.GetNativeBufferPtr(), 0, w, 4 * posesBase) != 1)
            throw new InvalidOperationException("AcousticRT: " + LastError());
    }

    IntPtr Data(int stateOffset) => new IntPtr(handle | (long)stateOffset << 12);
    // Before a frame's walk: the instance structure posed from the State buffer's frame starting at `stateOffset`.
    public void Build(int stateOffset) => Issue(EventBuild, stateOffset);
    // A wave's queries answered (indirect: the arguments the walk's bookkeeping wrote).
    public void Intersect(int stateOffset) => Issue(EventIntersect, stateOffset);
    void Issue(int kind, int stateOffset)
    {
        commands.Clear();
        commands.IssuePluginEventAndData(ARTEventFunction(), kind, Data(stateOffset));
        Graphics.ExecuteCommandBuffer(commands);
    }
    // Recording into a caller's command buffer (the update's waves, recorded once).
    public void Intersect(CommandBuffer into, int stateOffset) => into.IssuePluginEventAndData(ARTEventFunction(), EventIntersect, Data(stateOffset));

    public void Dispose()
    {
        if (disposed) return;
        disposed = true;
        Issue(EventDestroy, 0); // (released on the render thread after the GPU's last use)
        commands.Release();
    }
}
