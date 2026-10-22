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
    public readonly Node[] Nodes;
    public readonly int[] Indices;
    readonly AcousticTake.Primitive[] local;
    readonly int[] owners,roots,stack=new int[32],meshStack=new int[32];
    readonly Node[] bounds;
    readonly Vector3[] centers;
    readonly CenterComparer comparer;
    readonly bool planar;
    readonly int objectCount,topCount;
    int built;
    bool initialized;
    struct Pose { public Vector3 position; public Quaternion rotation; }
    Pose[] previous=Array.Empty<Pose>();
    int previousFrames;
    public int NodeCount => Nodes.Length;
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
        int instances=0,nodeCount=0;foreach(int count in counts)if(count>0){instances++;nodeCount+=CountNodes(count,4);}
        owners=new int[instances];topCount=CountNodes(instances,1);Nodes=new Node[checked(topCount+nodeCount)];
        Indices=new int[geometry.Length];centers=new Vector3[Math.Max(objectCount,geometry.Length)];comparer=new CenterComparer(centers);
        var starts=new int[objectCount];int first=0,ownerIndex=0;
        for(int owner=0;owner<objectCount;owner++){starts[owner]=first;first+=counts[owner];if(counts[owner]>0)owners[ownerIndex++]=owner;}
        for(int i=0;i<geometry.Length;i++){Indices[starts[geometry[i].objectIndex]++]=i;var p=geometry[i];centers[i]=planar?(p.a+p.b)*.5f:(p.a+p.b+p.c)/3;}
        built=topCount;first=0;
        foreach(int owner in owners){roots[owner]=Build(Indices,first,counts[owner],false);first+=counts[owner];}
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
        if(!initialized){built=0;if(owners.Length>0)Build(owners,0,owners.Length,true);}
        else if(changed)for(int i=topCount-1;i>=0;i--)
        {
            var node=Nodes[i];var box=node.count==uint.MaxValue?bounds[node.first]:Union(Nodes[node.left],Nodes[node.right]);
            node.min=box.min;node.max=box.max;Nodes[i]=node;
        }
        previousFrames=frames;
        if(changed)UpdateCount++;
        if(!initialized){initialized=true;BuildSeconds+=TakeRender.Elapsed(start);}else UpdateSeconds+=TakeRender.Elapsed(start);
        return changed;
    }
    sealed class CenterComparer : IComparer<int>
    {
        readonly Vector3[] values;public int axis;
        public CenterComparer(Vector3[] values){this.values=values;}
        public int Compare(int a,int b){int order=values[a][axis].CompareTo(values[b][axis]);return order==0?a.CompareTo(b):order;}
    }
    int Build(int[] indices,int first,int count,bool top)
    {
        int index=built++;
        if(count<=(top?1:4))
        {
            var node=new Node{min=Vector3.positiveInfinity,max=Vector3.negativeInfinity,first=(uint)first,count=(uint)count};
            if(top){int owner=indices[first];node=bounds[owner];node.first=(uint)owner;node.left=(uint)roots[owner];node.count=uint.MaxValue;}
            else for(int i=first;i<first+count;i++)
            {
                var p=local[indices[i]];var low=Vector3.Min(p.a,p.b);var high=Vector3.Max(p.a,p.b);
                if(!planar){low=Vector3.Min(low,p.c);high=Vector3.Max(high,p.c);}
                node.min=Vector3.Min(node.min,low);node.max=Vector3.Max(node.max,high);
            }
            Nodes[index]=node;return index;
        }
        var lowCenter=centers[indices[first]];var highCenter=lowCenter;
        for(int i=first+1;i<first+count;i++){lowCenter=Vector3.Min(lowCenter,centers[indices[i]]);highCenter=Vector3.Max(highCenter,centers[indices[i]]);}
        var size=highCenter-lowCenter;comparer.axis=size.x>=size.y?0:1;if(size.z>size[comparer.axis])comparer.axis=2;
        Array.Sort(indices,first,count,comparer);
        int left=Build(indices,first,count/2,top),right=Build(indices,first+count/2,count-count/2,top);
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
    bool Intersects(Node node,Vector3 origin,Vector3 direction)
    {
        double near=0,far=1;
        for(int axis=0;axis<(planar?2:3);axis++)
        {
            if(direction[axis]==0){if(origin[axis]<node.min[axis]||origin[axis]>node.max[axis])return false;continue;}
            double a=((double)node.min[axis]-origin[axis])/direction[axis],b=((double)node.max[axis]-origin[axis])/direction[axis];
            near=Math.Max(near,Math.Min(a,b));far=Math.Min(far,Math.Max(a,b));if(near>far)return false;
        }
        return true;
    }
}
