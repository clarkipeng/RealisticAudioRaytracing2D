using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Runtime.InteropServices;
using UnityEngine;

// Immutable local mesh trees (BLAS), beneath refitted instance bounds (TLAS).
// One batch shares conservative world bounds; leaf tests use each frame's exact pose.
internal sealed class DirectBvh
{
    [StructLayout(LayoutKind.Sequential,Pack=4)] public struct Node
    {
        public Vector4 min,max;
        public uint left,right,first,count; // count: 0 branch, uint.MaxValue instance, otherwise primitives.
    }
    public Node[] Nodes {get;private set;}
    public readonly int[] Indices;
    readonly AcousticTake.Primitive[] local;
    readonly int[] owners,order,roots,stack=new int[32],meshStack=new int[32];
    readonly Node[] bounds;
    readonly Vector3[] centers;
    readonly SahBuilder.CenterComparer comparer;
    readonly bool planar;
    readonly int objectCount,topCount;
    int built;
    bool initialized;
    struct Pose { public Vector3 position; public Quaternion rotation; }
    Pose[] previous=Array.Empty<Pose>();
    int previousFrames;
    public int NodeCount => Nodes.Length;
    // The GPU queries' tree (3D scenes): the unmoved objects merged, after these nodes and indices in the GPU buffers.
    public readonly StaticTraceTree Trace;
    internal bool HasMesh(int owner) => Array.IndexOf(owners,owner)>=0;
    internal Node Instance(int owner){var node=bounds[owner];node.first=(uint)owner;node.left=(uint)roots[owner];node.count=uint.MaxValue;return node;}
    public int TopNodeCount => topCount;
    public int PrimitiveCount => local.Length;
    public double BuildSeconds {get;private set;}
    public double UpdateSeconds {get;private set;}
    public int UpdateCount {get;private set;}
    public long NumericBytes => (long)Nodes.Length*48+(long)Indices.Length*4+(long)owners.Length*4+
        (long)roots.Length*(4+48)+(long)centers.Length*12+(stack.Length+meshStack.Length)*4+(long)previous.Length*28;

    public DirectBvh(AcousticTake.Scene scene,AcousticTake.Primitive[] geometry)
    {
        long start=Stopwatch.GetTimestamp();local=geometry;objectCount=scene.objects.Length;
        planar=scene.dimension==AcousticTake.Dimension.Planar2D;
        roots=new int[objectCount];bounds=new Node[objectCount];
        var counts=new int[objectCount];foreach(var p in geometry)counts[p.objectIndex]++;
        int instances=0,nodeCount=0;foreach(int count in counts)if(count>0){instances++;nodeCount+=2*count-1;}   // an upper bound; trimmed below
        owners=new int[instances];order=new int[instances];topCount=CountNodes(instances,1);Nodes=new Node[checked(topCount+nodeCount)];
        Indices=new int[geometry.Length];centers=new Vector3[objectCount];comparer=new SahBuilder.CenterComparer(centers);
        var starts=new int[objectCount];int first=0,ownerIndex=0;
        for(int owner=0;owner<objectCount;owner++){starts[owner]=first;first+=counts[owner];if(counts[owner]>0)owners[ownerIndex++]=owner;}
        var low=new Vector3[geometry.Length];var high=new Vector3[geometry.Length];var middle=new Vector3[geometry.Length];
        for(int i=0;i<geometry.Length;i++)
        {
            Indices[starts[geometry[i].objectIndex]++]=i;var p=geometry[i];middle[i]=planar?(p.a+p.b)*.5f:(p.a+p.b+p.c)/3;
            low[i]=Vector3.Min(p.a,p.b);high[i]=Vector3.Max(p.a,p.b);if(!planar){low[i]=Vector3.Min(low[i],p.c);high[i]=Vector3.Max(high[i],p.c);}
        }
        var meshes=new SahBuilder(low,high,middle,planar,Nodes,topCount);first=0;
        foreach(int owner in owners){roots[owner]=meshes.Build(Indices,first,counts[owner]);first+=counts[owner];}
        built=meshes.Built;
        var trimmed=Nodes;Array.Resize(ref trimmed,built);Nodes=trimmed;
        if(!planar)Trace=new StaticTraceTree(geometry,objectCount,Nodes.Length,geometry.Length);
        BuildSeconds=TakeRender.Elapsed(start);
    }
    static int CountNodes(int count,int leaf) => count==0 ? 0:count<=leaf ? 1:checked(1+CountNodes(count/2,leaf)+CountNodes(count-count/2,leaf));
    static float AbsMax(Vector3 v) => Mathf.Max(Mathf.Abs(v.x),Mathf.Max(Mathf.Abs(v.y),Mathf.Abs(v.z)));
    static Node Union(Node a,Node b){a.min=Vector3.Min(a.min,b.min);a.max=Vector3.Max(a.max,b.max);return a;}
    // World-space boxes also cover arbitrary rotations projected into the 2D plane.
    internal static Node WorldBounds(Node node,AcousticTake.State pose)
    {
        Vector3 low=node.min,high=node.max,center=(low+high)*.5f,half=(high-low)*.5f;
        var x=pose.rotation*Vector3.right;var y=pose.rotation*Vector3.up;var z=pose.rotation*Vector3.forward;
        var extent=new Vector3(Mathf.Abs(x.x)*half.x+Mathf.Abs(y.x)*half.y+Mathf.Abs(z.x)*half.z,
            Mathf.Abs(x.y)*half.x+Mathf.Abs(y.y)*half.y+Mathf.Abs(z.y)*half.z,
            Mathf.Abs(x.z)*half.x+Mathf.Abs(y.z)*half.y+Mathf.Abs(z.z)*half.z);
        center=pose.position+pose.rotation*center;
        extent+=Vector3.one*(8*1.1920929e-7f*(1+AbsMax(pose.position)+4*Mathf.Max(AbsMax(low),AbsMax(high))));
        node.min=center-extent;node.max=center+extent;return node;
    }
    public bool Update(AcousticTake.State[] state,int frames=1)
    {
        long start=Stopwatch.GetTimestamp();bool changed=!initialized;
        int required=checked(owners.Length*frames);if(previous.Length<required)previous=new Pose[required];
        for(int instance=0;instance<owners.Length;instance++)
        {
            int owner=owners[instance];bool moved=!initialized||previousFrames!=frames;
            for(int f=0;f<frames;f++)
            {
                int index=f*owners.Length+instance;var pose=state[f*objectCount+owner];var old=previous[index];
                moved|=!old.position.Equals(pose.position)||!old.rotation.Equals(pose.rotation);
                previous[index]=new Pose {position=pose.position,rotation=pose.rotation};
            }
            if(!moved)continue;
            var box=WorldBounds(Nodes[roots[owner]],state[owner]);
            for(int f=1;f<frames;f++)box=Union(box,WorldBounds(Nodes[roots[owner]],state[f*objectCount+owner]));
            changed|=!box.min.Equals(bounds[owner].min)||!box.max.Equals(bounds[owner].max);bounds[owner]=box;
            centers[owner]=((Vector3)box.min+(Vector3)box.max)*.5f;
        }
        // The instance tree is rebuilt whenever an instance moved, not refitted: a refit keeps the first update's
        // grouping, so an object that travels far (the station's train on its loop) stretches every box above it across
        // its path, and rays then descend into subtrees they cannot hit (README part 41). The rebuild is the same median
        // split over the instances' current centres as the first build: O(n log n) over the instances, n in the hundreds.
        if(changed){built=0;Array.Copy(owners,order,owners.Length);if(order.Length>0)Build(order,0,order.Length);}
        previousFrames=frames;
        Trace?.Update(state,frames,this);
        if(changed)UpdateCount++;
        if(!initialized){initialized=true;BuildSeconds+=TakeRender.Elapsed(start);}else UpdateSeconds+=TakeRender.Elapsed(start);
        return changed;
    }
    // The instance tree: a median split over the instances' current centres, one instance per leaf.
    int Build(int[] indices,int first,int count)
    {
        int index=built++;
        if(count<=1)
        {
            int owner=indices[first];var node=bounds[owner];node.first=(uint)owner;node.left=(uint)roots[owner];node.count=uint.MaxValue;
            Nodes[index]=node;return index;
        }
        var lowCenter=centers[indices[first]];var highCenter=lowCenter;
        for(int i=first+1;i<first+count;i++){lowCenter=Vector3.Min(lowCenter,centers[indices[i]]);highCenter=Vector3.Max(highCenter,centers[indices[i]]);}
        var size=highCenter-lowCenter;comparer.axis=size.x>=size.y?0:1;if(size.z>size[comparer.axis])comparer.axis=2;
        Array.Sort(indices,first,count,comparer);
        int left=Build(indices,first,count/2),right=Build(indices,first+count/2,count-count/2);
        var branch=Union(Nodes[left],Nodes[right]);branch.left=(uint)left;branch.right=(uint)right;branch.first=branch.count=0;Nodes[index]=branch;return index;
    }
    bool MeshBlocked(int root,AcousticTake.State pose,Vector3 origin,Vector3 end)
    {
        int depth=0,index=root;var direction=end-origin;
        while(true)
        {
            var node=Nodes[index];
            if(Intersects(WorldBounds(node,pose),origin,direction))
            {
                if(node.count==0){meshStack[depth++]=(int)node.right;index=(int)node.left;continue;}
                for(uint j=0;j<node.count;j++)
                {
                    var p=local[Indices[node.first+j]];var a=pose.position+pose.rotation*p.a;var b=pose.position+pose.rotation*p.b;
                    if(planar?TakeRender.CrossesSegment(origin,end,a,b):TakeRender.CrossesTriangle(origin,direction,a,b,pose.position+pose.rotation*p.c))return true;
                }
            }
            if(depth==0)return false;index=meshStack[--depth];
        }
    }
    public bool Blocked(Vector3 origin,Vector3 end,AcousticTake.State[] state)
    {
        int depth=0,index=0;var direction=end-origin;
        while(index<topCount)
        {
            var node=Nodes[index];
            if(Intersects(node,origin,direction))
            {
                if(node.count==0){stack[depth++]=(int)node.right;index=(int)node.left;continue;}
                var pose=state[node.first];
                if((pose.flags&(AcousticTake.StateFlags.Active|AcousticTake.StateFlags.Geometry))==(AcousticTake.StateFlags.Active|AcousticTake.StateFlags.Geometry)&&MeshBlocked((int)node.left,pose,origin,end))return true;
            }
            if(depth==0)break;index=stack[--depth];
        }
        return false;
    }
    internal struct Hit { internal int primitive; internal Vector3 position,normal; internal float distance; }
    // Birth-time queries reuse immutable local trees. They are independent of
    // acoustic ray budgets and never consult the foreground Physics scene.
    internal bool Raycast(Vector3 origin,Vector3 direction,AcousticTake.State[] state,out Hit hit)
    {
        if(planar)throw new InvalidOperationException("Surface impact queries require triangles.");
        hit=new Hit{primitive=-1,distance=float.PositiveInfinity};
        int depth=0,index=0;
        while(index<topCount)
        {
            var node=Nodes[index];
            if(Intersects(node,origin,direction,hit.distance))
            {
                if(node.count==0){stack[depth++]=(int)node.right;index=(int)node.left;continue;}
                var pose=state[node.first];
                if((pose.flags&(AcousticTake.StateFlags.Active|AcousticTake.StateFlags.Geometry))==(AcousticTake.StateFlags.Active|AcousticTake.StateFlags.Geometry))
                    RaycastMesh((int)node.left,pose,origin,direction,ref hit);
            }
            if(depth==0)break;index=stack[--depth];
        }
        if(hit.primitive<0)return false;
        hit.position=origin+direction*hit.distance;return true;
    }
    void RaycastMesh(int root,AcousticTake.State pose,Vector3 origin,Vector3 direction,ref Hit hit)
    {
        var inverse=Quaternion.Inverse(pose.rotation);var o=inverse*(origin-pose.position);var d=inverse*direction;
        int depth=0,index=root;
        while(true)
        {
            var node=Nodes[index];
            if(Intersects(node,o,d,hit.distance))
            {
                if(node.count==0){meshStack[depth++]=(int)node.right;index=(int)node.left;continue;}
                for(uint j=0;j<node.count;j++)
                {
                    int id=Indices[node.first+j];var p=local[id];var e1=p.b-p.a;var e2=p.c-p.a;
                    var h=Vector3.Cross(d,e2);double det=Vector3.Dot(e1,h);
                    if(Math.Abs(det)<1e-12)continue;
                    var s=o-p.a;double u=Vector3.Dot(s,h)/det;if(u<0||u>1)continue;
                    var q=Vector3.Cross(s,e1);double v=Vector3.Dot(d,q)/det;if(v<0||u+v>1)continue;
                    float t=(float)(Vector3.Dot(e2,q)/det);
                    if(t<0||t>hit.distance||!float.IsFinite(t)||(t==hit.distance&&id>=hit.primitive))continue;
                    var normal=pose.rotation*Vector3.Cross(e1,e2).normalized;if(Vector3.Dot(normal,direction)>0)normal=-normal;
                    hit=new Hit{primitive=id,distance=t,normal=normal};
                }
            }
            if(depth==0)break;index=meshStack[--depth];
        }
    }
    bool Intersects(Node node,Vector3 origin,Vector3 direction,double limit=1)
    {
        double near=0,far=limit;
        for(int axis=0;axis<(planar?2:3);axis++)
        {
            if(direction[axis]==0){if(origin[axis]<node.min[axis]||origin[axis]>node.max[axis])return false;continue;}
            double a=((double)node.min[axis]-origin[axis])/direction[axis],b=((double)node.max[axis]-origin[axis])/direction[axis];
            near=Math.Max(near,Math.Min(a,b));far=Math.Min(far,Math.Max(a,b));if(near>far)return false;
        }
        return true;
    }
}
