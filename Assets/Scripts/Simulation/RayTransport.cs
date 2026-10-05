using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using UnityEngine;
using UnityEngine.Rendering;

// Direction-bearing arrivals from the legacy geometric heuristics, now using the
// shared 2D/3D scene, stable sampling and one owned output per ray/bounce.
internal sealed class RayTransport : IDisposable
{
    [StructLayout(LayoutKind.Sequential,Pack=4)] public struct Path
    {
        public Vector4 direction;
        public float delaySeconds,pressureGain;
        public uint frequencyBin,hrtfIndex;
    }
    [StructLayout(LayoutKind.Sequential,Pack=4)] struct Material
    {
        public Vector4 surface; // absorption, scattering, transmission, IOR
        public Vector4 medium;  // damping; reserved
    }
    [StructLayout(LayoutKind.Sequential,Pack=4)] struct Edge
    {
        public Vector4 a,b;
        public uint owner,pad0,pad1,pad2;
    }
    readonly ComputeShader shader;
    readonly int directKernel,traceKernel,pairs,rayCount,frequencyBins;
    readonly ComputeBuffer materialBuffer,edgeBuffer,directionsBuffer,pathBuffer;
    readonly Action<AsyncGPUReadbackRequest> completion;
    AsyncGPUReadbackRequest request;
    bool pending,failed;
    volatile bool completed;
    public readonly Path[] Paths;
    public int PathsPerFrame {get;}
    public int SlotsPerPair {get;}
    public long CpuBytes => Paths.Length*32L;
    public long GpuBytes {get;}
    public RayTransport(AcousticTake.Scene scene,AcousticTake.Primitive[] geometry,TakeRender.Settings settings,ComputeShader shader,
        ComputeBuffer primitives,ComputeBuffer states,ComputeBuffer sources,ComputeBuffer listeners,ComputeBuffer bvhNodes,ComputeBuffer bvhIndices,int rate,int maximumFrames)
    {
        if(Marshal.SizeOf<Path>()!=32 || Marshal.SizeOf<Material>()!=32 || Marshal.SizeOf<Edge>()!=48)throw new InvalidOperationException("Spectral GPU ABI mismatch.");
        this.shader=shader;rayCount=settings.raysPerPair;frequencyBins=settings.fftSize/2+1;pairs=checked(scene.sources.Length*scene.listeners.Length);
        int bins=settings.fftSize/2+1;
        SlotsPerPair=checked(bins+settings.raysPerPair*settings.maxBounces);PathsPerFrame=checked(pairs*SlotsPerPair);
        Paths=new Path[checked(PathsPerFrame*maximumFrames)];completion=OnComplete;
        bool planar=scene.dimension==AcousticTake.Dimension.Planar2D;
        try
        {
            directKernel=shader.FindKernel((planar?"SpectralDirect2D":"SpectralDirect3D")+(settings.useBvh?"Bvh":""));
            traceKernel=shader.FindKernel((planar?"Trace2D":"Trace3D")+(settings.useBvh?"Bvh":""));
            if(!shader.IsSupported(directKernel) || !shader.IsSupported(traceKernel))throw new NotSupportedException("Raytraced kernels are unavailable on this GPU.");
            var materials=new Material[Math.Max(1,scene.materials.Length*bins)];
            for(int m=0;m<scene.materials.Length;m++)for(int b=0;b<bins;b++)
            {
                var material=scene.materials[m];float hz=(float)b*rate/settings.fftSize;
                materials[m*bins+b]=new Material {surface=new Vector4(Sample(material,material.absorption,hz,0),Sample(material,material.scattering,hz,0),
                    Sample(material,material.transmission,hz,0),Sample(material,material.ior,hz,1)),medium=new Vector4(Sample(material,material.damping,hz,0),0,0,0)};
            }
            materialBuffer=new ComputeBuffer(materials.Length,32);materialBuffer.SetData(materials);
            var edges=Edges(geometry,planar);edgeBuffer=new ComputeBuffer(Math.Max(1,edges.Length),48);if(edges.Length>0)edgeBuffer.SetData(edges);
            var hrtf=settings.listenerOutput==TakeRender.ListenerOutput.Binaural?Hrtf.Load(rate):null;
            var directions=new Vector4[hrtf==null?1:hrtf.DirectionCount];
            for(int i=0;i<directions.Length;i++)directions[i]=hrtf==null?Vector3.forward:hrtf.Direction(i);
            directionsBuffer=new ComputeBuffer(directions.Length,16);directionsBuffer.SetData(directions);
            pathBuffer=new ComputeBuffer(Paths.Length,32);
            foreach(int kernel in new[]{directKernel,traceKernel})
            {
                shader.SetBuffer(kernel,"Primitives",primitives);shader.SetBuffer(kernel,"States",states);
                shader.SetBuffer(kernel,"Sources",sources);shader.SetBuffer(kernel,"Listeners",listeners);
                shader.SetBuffer(kernel,"RayMaterials",materialBuffer);shader.SetBuffer(kernel,"Edges",edgeBuffer);
                shader.SetBuffer(kernel,"HrtfDirections",directionsBuffer);shader.SetBuffer(kernel,"RayPaths",pathBuffer);
                if(bvhNodes!=null){shader.SetBuffer(kernel,"BvhNodes",bvhNodes);shader.SetBuffer(kernel,"BvhIndices",bvhIndices);}
            }
            shader.SetInt("FrequencyBins",bins);shader.SetInt("RaysPerPair",settings.raysPerPair);shader.SetInt("MaxBounces",settings.maxBounces);
            shader.SetInt("SlotsPerPair",SlotsPerPair);shader.SetInt("EdgeCount",edges.Length);shader.SetInt("HrtfDirectionCount",directions.Length);
            shader.SetFloat("FrequencyStep",(float)rate/settings.fftSize);shader.SetFloat("DiffractionFactor",settings.diffractionFactor);
            shader.SetFloat("MaximumPathDelay",settings.maximumDelaySeconds);shader.SetInt("UseAirAbsorption",settings.airAbsorption?1:0);
            GpuBytes=(long)materials.Length*32+(long)Math.Max(1,edges.Length)*48+directions.Length*16L+Paths.Length*32L;
        }
        catch {Dispose();throw;}
    }
    static float Sample(AcousticTake.MaterialDefinition material,float[] values,float frequency,float fallback)
    {
        if(values==null || values.Length==0)return fallback;
        var knots=material.frequencies;
        if(knots==null || knots.Length!=values.Length)throw new ArgumentException("Material coefficient grids do not match.");
        int upper=0;
        for(int i=0;i<values.Length;i++)
        {
            if(!float.IsFinite(values[i]) || values[i]<0 || !float.IsFinite(knots[i]) || knots[i]<0 || (i>0 && knots[i]<=knots[i-1]))
                throw new ArgumentException("Invalid material coefficient curve.");
        }
        while(upper<knots.Length-1 && frequency>knots[upper])upper++;
        int lower=Math.Max(0,upper-1);return Mathf.Lerp(values[lower],values[upper],lower==upper?0:Mathf.InverseLerp(knots[lower],knots[upper],frequency));
    }
    static Edge[] Edges(AcousticTake.Primitive[] geometry,bool planar)
    {
        var candidates=new Dictionary<(int,Vector3,Vector3),(Vector3 normal,bool keep)>();
        int Compare(Vector3 a,Vector3 b) {int r=a.x.CompareTo(b.x);if(r==0)r=a.y.CompareTo(b.y);return r==0?a.z.CompareTo(b.z):r;}
        void Add(int owner,Vector3 a,Vector3 b,Vector3 normal)
        {
            if(Compare(a,b)>0){var swap=a;a=b;b=swap;}
            var key=(owner,a,b);
            if(candidates.TryGetValue(key,out var previous))candidates[key]=(previous.normal,Mathf.Abs(Vector3.Dot(previous.normal,normal))<.99999f);
            else candidates.Add(key,(normal,true));
        }
        foreach(var p in geometry)
            if(planar){Add(p.objectIndex,p.a,p.a,Vector3.forward);Add(p.objectIndex,p.b,p.b,Vector3.forward);}
            else {var n=Vector3.Cross(p.b-p.a,p.c-p.a).normalized;Add(p.objectIndex,p.a,p.b,n);Add(p.objectIndex,p.b,p.c,n);Add(p.objectIndex,p.c,p.a,n);}
        var result=new List<Edge>();
        foreach(var item in candidates)if(planar || item.Value.keep)result.Add(new Edge {owner=(uint)item.Key.Item1,a=item.Key.Item2,b=item.Key.Item3});
        return result.ToArray();
    }
    public void Dispatch(int frames,bool asynchronous)
    {
        if(pending)throw new InvalidOperationException("Ray responses are still in flight.");
        shader.Dispatch(directKernel,(frames*pairs*frequencyBins+63)/64,1,1);
        shader.Dispatch(traceKernel,(frames*pairs*rayCount+63)/64,1,1);
        if(asynchronous){pending=true;completed=false;failed=false;request=AsyncGPUReadback.Request(pathBuffer,checked(frames*PathsPerFrame*32),0,completion);}
        else pathBuffer.GetData(Paths,0,0,frames*PathsPerFrame);
    }
    void OnComplete(AsyncGPUReadbackRequest value)
    {
        failed=value.hasError;
        if(!failed){var data=value.GetData<Path>();for(int i=0;i<data.Length;i++)Paths[i]=data[i];}
        completed=true;
    }
    public bool TryComplete(bool wait)
    {
        if(!pending)return true;
        if(!completed){if(wait)request.WaitForCompletion();else request.Update();}
        if(!completed)return false;
        pending=false;if(failed)throw new InvalidOperationException("Ray response readback failed.");return true;
    }
    public void Dispose()
    {
        if(pending && !completed)request.WaitForCompletion();pending=false;
        materialBuffer?.Dispose();edgeBuffer?.Dispose();directionsBuffer?.Dispose();pathBuffer?.Dispose();
    }
}
