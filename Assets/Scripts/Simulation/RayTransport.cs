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
        public Vector4 direction; // xyz arrival direction; w = d(propagation seconds)/d(receiver second)
        public float delaySeconds,pressureGain;
        public uint frequencyBin,identity;
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
    [StructLayout(LayoutKind.Sequential,Pack=4)] struct Volume
    {
        public Vector4 minimum,maximum; // w: winding correction, enclosed volume/area (relative units)
        public uint start,count,owner,material;
    }
    readonly ComputeBuffer materialBuffer,edgeBuffer,volumeBuffer,pathBuffer;
    readonly Action<AsyncGPUReadbackRequest> completion;
    AsyncGPUReadbackRequest request;
    bool pending,failed;
    volatile bool completed;
    public readonly Path[] Paths;
    public int PathsPerFrame {get;}
    public int SlotsPerPair {get;}
    internal ComputeBuffer Buffer => pathBuffer;
    public long CpuBytes => Paths.Length*32L;
    public long GpuBytes {get;}
    public RayTransport(AcousticTake.Scene scene,AcousticTake.Primitive[] geometry,TakeRender.Settings settings,ComputeShader shader,
        ComputeBuffer primitives,ComputeBuffer states,ComputeBuffer sources,ComputeBuffer listeners,ComputeBuffer bvhNodes,ComputeBuffer bvhIndices,int rate,int maximumFrames,bool readback=true)
    {
        if(Marshal.SizeOf<Path>()!=32 || Marshal.SizeOf<Material>()!=32 || Marshal.SizeOf<Edge>()!=48 || Marshal.SizeOf<Volume>()!=48)throw new InvalidOperationException("Spectral GPU ABI mismatch.");
        this.shader=shader;rayCount=settings.RaysPerPair;frequencyBins=settings.fftSize/2+1;pairs=checked(scene.sources.Length*scene.listeners.Length);
        int bins=settings.fftSize/2+1;
        SlotsPerPair=settings.SpectralSlotsPerPair;PathsPerFrame=checked(pairs*SlotsPerPair);
        Paths=new Path[readback?checked(PathsPerFrame*maximumFrames):0];completion=OnComplete;
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
            var volumes=Volumes(scene,geometry,materials,bins,planar);
            volumeBuffer=new ComputeBuffer(Math.Max(1,volumes.Length),48);if(volumes.Length>0)volumeBuffer.SetData(volumes);
            pathBuffer=new ComputeBuffer(checked(PathsPerFrame*maximumFrames),32);
            foreach(int kernel in new[]{directKernel,traceKernel})
            {
                shader.SetBuffer(kernel,"Primitives",primitives);shader.SetBuffer(kernel,"States",states);
                shader.SetBuffer(kernel,"Sources",sources);shader.SetBuffer(kernel,"Listeners",listeners);
                shader.SetBuffer(kernel,"RayMaterials",materialBuffer);shader.SetBuffer(kernel,"Edges",edgeBuffer);
                shader.SetBuffer(kernel,"Volumes",volumeBuffer);shader.SetBuffer(kernel,"RayPaths",pathBuffer);
                if(bvhNodes!=null){shader.SetBuffer(kernel,"BvhNodes",bvhNodes);shader.SetBuffer(kernel,"BvhIndices",bvhIndices);}
            }
            shader.SetInt("RaysPerBin",settings.RaysPerBin);shader.SetInt("RaySeed",settings.seed);shader.SetInt("FrequencyBins",bins);shader.SetInt("RaysPerPair",settings.RaysPerPair);shader.SetInt("MaxBounces",settings.maxBounces);
            shader.SetInt("SlotsPerPair",SlotsPerPair);shader.SetInt("EdgeCount",edges.Length);shader.SetInt("VolumeCount",volumes.Length);
            shader.SetFloat("FrequencyStep",(float)rate/settings.fftSize);shader.SetFloat("DiffractionFactor",settings.diffractionFactor);
            shader.SetFloat("MaximumPathDelay",settings.maximumDelaySeconds);shader.SetInt("UseAirAbsorption",settings.airAbsorption?1:0);
            GpuBytes=(long)materials.Length*32+(long)Math.Max(1,edges.Length)*48+Math.Max(1,volumes.Length)*48L+(long)PathsPerFrame*maximumFrames*32;
        }
        catch {Dispose();throw;}
    }
    // Closed, consistently wound meshes/loops have thickness. Open sheets only
    // apply their surface coefficients. This is derived once, not saved in takes.
    static Volume[] Volumes(AcousticTake.Scene scene,AcousticTake.Primitive[] geometry,Material[] materials,int bins,bool planar)
    {
        var result=new List<Volume>();
        for(int owner=0;owner<scene.objects.Length;owner++)
        {
            var obj=scene.objects[owner];if(obj.primitiveCount==0)continue;
            int start=obj.primitiveStart,end=start+obj.primitiveCount;
            var min=geometry[start].a;var max=min;
            for(int i=start;i<end;i++)
            {
                var p=geometry[i];min=Vector3.Min(min,Vector3.Min(p.a,p.b));max=Vector3.Max(max,Vector3.Max(p.a,p.b));
                if(!planar){min=Vector3.Min(min,p.c);max=Vector3.Max(max,p.c);}
            }
            // Weld floating-point seams from curved colliders for topology only.
            float tolerance=Mathf.Max(1e-7f,(max-min).magnitude*1e-6f);
            var vertices=new Dictionary<Vector3Int,int>();
            int Vertex(Vector3 p)
            {
                p=(p-min)/tolerance;var key=new Vector3Int(Mathf.RoundToInt(p.x),Mathf.RoundToInt(p.y),planar?0:Mathf.RoundToInt(p.z));
                if(!vertices.TryGetValue(key,out int id)){id=vertices.Count;vertices.Add(key,id);}return id;
            }
            var edges=new Dictionary<(int,int),(int count,int winding)>();
            void Edge(int a,int b)
            {
                var key=a<b?(a,b):(b,a);edges.TryGetValue(key,out var value);
                edges[key]=(value.count+1,value.winding+(a<b?1:-1));
            }
            double signed=0;var degree=new Dictionary<int,(int incoming,int outgoing)>();
            for(int i=start;i<end;i++)
            {
                var p=geometry[i];int a=Vertex(p.a),b=Vertex(p.b);
                if(planar)
                {
                    degree.TryGetValue(a,out var va);degree[a]=(va.incoming,va.outgoing+1);
                    degree.TryGetValue(b,out var vb);degree[b]=(vb.incoming+1,vb.outgoing);
                    signed+=(double)p.a.x*p.b.y-(double)p.a.y*p.b.x;
                }
                else {int c=Vertex(p.c);Edge(a,b);Edge(b,c);Edge(c,a);signed+=Vector3.Dot(p.a-min,Vector3.Cross(p.b-min,p.c-min));}
            }
            bool closed=signed!=0;
            foreach(var edge in edges.Values)closed&=edge.count==2 && edge.winding==0;
            foreach(var vertex in degree.Values)closed&=vertex.incoming==1 && vertex.outgoing==1;
            if(!closed)continue;
            int material=geometry[start].materialIndex;
            for(int i=start+1;i<end;i++)for(int b=0;b<bins;b++)
            {
                var a=materials[material*bins+b];var other=materials[geometry[i].materialIndex*bins+b];
                if(a.surface.w!=other.surface.w || a.medium.x!=other.medium.x)
                    throw new ArgumentException(obj.name+": a closed acoustic object must share interior IOR/damping across its surface materials.");
            }
            result.Add(new Volume {minimum=new Vector4(min.x,min.y,min.z,(signed>0?1:-1)*(planar?-1:1)),maximum=new Vector4(max.x,max.y,max.z,(float)Math.Abs(signed)),
                start=(uint)start,count=(uint)obj.primitiveCount,owner=(uint)owner,material=(uint)material});
        }
        return result.ToArray();
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
        if(Paths.Length==0)return;
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
        materialBuffer?.Dispose();edgeBuffer?.Dispose();volumeBuffer?.Dispose();pathBuffer?.Dispose();
    }
}
