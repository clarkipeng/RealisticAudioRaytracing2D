using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Runtime.InteropServices;
using UnityEngine;

// One balanced world-space hierarchy. Rigid poses update cached vertices and refit bounds;
// primitive IDs and tree topology remain stable for the lifetime of the solver.
internal sealed class DirectBvh
{
    [StructLayout(LayoutKind.Sequential,Pack=4)] public struct Node
    {
        public Vector4 min,max;
        public uint left,right,first,count;
    }
    public readonly Node[] Nodes;
    public readonly int[] Indices;
    readonly AcousticTake.Primitive[] local,world;
    readonly AcousticTake.State[] previous;
    readonly int[] heads,next,stack=new int[32];
    readonly float[] padding;
    readonly Vector3[] centers;
    readonly bool planar;
    bool initialized;
    int built;
    public double BuildSeconds {get;private set;}
    public double UpdateSeconds {get;private set;}
    public int UpdateCount {get;private set;}
    public long NumericBytes => (long)world.Length*(44+4+4+12)+(long)Nodes.Length*48+(long)previous.Length*(80+4+4)+stack.Length*4;

    public DirectBvh(AcousticTake.Scene scene,AcousticTake.Primitive[] geometry)
    {
        local=geometry; world=new AcousticTake.Primitive[geometry.Length]; Indices=new int[geometry.Length];
        previous=new AcousticTake.State[scene.objects.Length]; heads=new int[previous.Length]; next=new int[geometry.Length];
        padding=new float[previous.Length]; centers=new Vector3[geometry.Length];
        planar=scene.dimension==AcousticTake.Dimension.Planar2D;
        Array.Fill(heads,-1);
        for(int i=0;i<geometry.Length;i++) {int owner=geometry[i].objectIndex;next[i]=heads[owner];heads[owner]=i;Indices[i]=i;}
        Nodes=new Node[CountNodes(geometry.Length)];
    }
    static int CountNodes(int count) => count==0 ? 0:count<=4 ? 1:checked(1+CountNodes(count/2)+CountNodes(count-count/2));
    static float AbsMax(Vector3 value) => Mathf.Max(Mathf.Abs(value.x),Mathf.Max(Mathf.Abs(value.y),Mathf.Abs(value.z)));
    public bool Update(AcousticTake.State[] state,int offset=0)
    {
        long start=Stopwatch.GetTimestamp(); bool changed=!initialized;
        for(int owner=0;owner<heads.Length;owner++)
        {
            if(heads[owner]<0)continue;
            var pose=state[offset+owner];
            if(initialized && pose.position.Equals(previous[owner].position) && pose.rotation.Equals(previous[owner].rotation))continue;
            previous[owner]=pose; changed=true; float magnitude=1+AbsMax(pose.position);
            for(int i=heads[owner];i>=0;i=next[i])
            {
                var p=local[i];
                world[i]=new AcousticTake.Primitive {a=pose.position+pose.rotation*p.a,b=pose.position+pose.rotation*p.b,
                    c=planar?Vector3.zero:pose.position+pose.rotation*p.c,objectIndex=owner,materialIndex=p.materialIndex};
                magnitude=Mathf.Max(magnitude,1+AbsMax(pose.position)+4*Mathf.Max(AbsMax(p.a),Mathf.Max(AbsMax(p.b),planar?0:AbsMax(p.c))));
                if(!initialized)centers[i]=planar ? (world[i].a+world[i].b)*.5f:(world[i].a+world[i].b+world[i].c)/3;
            }
            // Conservative broad phase: CPU/GPU quaternion arithmetic may round differently.
            padding[owner]=8*1.1920929e-7f*magnitude;
        }
        if(!initialized)
        {
            if(Indices.Length>0)Build(0,Indices.Length,new CenterComparer(centers));
            Refit(); initialized=true; BuildSeconds+=TakeRender.Elapsed(start);
        }
        else
        {
            if(changed)Refit();
            UpdateSeconds+=TakeRender.Elapsed(start);
        }
        if(changed)UpdateCount++;
        return changed;
    }
    sealed class CenterComparer : IComparer<int>
    {
        readonly Vector3[] values;
        public int axis;
        public CenterComparer(Vector3[] values) {this.values=values;}
        public int Compare(int a,int b) {int order=values[a][axis].CompareTo(values[b][axis]);return order==0?a.CompareTo(b):order;}
    }
    int Build(int first,int count,CenterComparer comparer)
    {
        int index=built++;
        if(count<=4) {Nodes[index]=new Node {left=uint.MaxValue,right=uint.MaxValue,first=(uint)first,count=(uint)count};return index;}
        Vector3 min=centers[Indices[first]],max=min;
        for(int i=first+1;i<first+count;i++) {min=Vector3.Min(min,centers[Indices[i]]);max=Vector3.Max(max,centers[Indices[i]]);}
        var size=max-min;comparer.axis=size.x>=size.y ? 0:1;
        if(!planar && size.z>size[comparer.axis])comparer.axis=2;
        Array.Sort(Indices,first,count,comparer);
        int left=Build(first,count/2,comparer),right=Build(first+count/2,count-count/2,comparer);
        Nodes[index]=new Node {left=(uint)left,right=(uint)right};return index;
    }
    void Refit()
    {
        for(int n=Nodes.Length-1;n>=0;n--)
        {
            var node=Nodes[n];
            if(node.count==0)
            {
                node.min=Vector3.Min(Nodes[node.left].min,Nodes[node.right].min);
                node.max=Vector3.Max(Nodes[node.left].max,Nodes[node.right].max);
            }
            else
            {
                Vector3 min=Vector3.positiveInfinity,max=Vector3.negativeInfinity;
                for(uint j=0;j<node.count;j++)
                {
                    var p=world[Indices[node.first+j]];
                    var low=Vector3.Min(p.a,p.b);var high=Vector3.Max(p.a,p.b);
                    if(!planar) {low=Vector3.Min(low,p.c);high=Vector3.Max(high,p.c);}
                    var error=Vector3.one*padding[p.objectIndex];min=Vector3.Min(min,low-error);max=Vector3.Max(max,high+error);
                }
                node.min=min;node.max=max;
            }
            Nodes[n]=node;
        }
    }
    public bool Blocked(Vector3 origin,Vector3 end,AcousticTake.State[] state)
    {
        if(Nodes.Length==0)return false;
        int depth=0,nodeIndex=0;var direction=end-origin;
        // Median splits of an int-sized primitive array require at most 31 pending nodes.
        while(true)
        {
            var node=Nodes[nodeIndex];
            if(Intersects(node,origin,direction))
            {
                if(node.count==0) {stack[depth++]=(int)node.right;nodeIndex=(int)node.left;continue;}
                for(uint j=0;j<node.count;j++)
                {
                    var p=world[Indices[node.first+j]];
                    if((state[p.objectIndex].flags&(AcousticTake.StateFlags.Active|AcousticTake.StateFlags.Geometry))!=(AcousticTake.StateFlags.Active|AcousticTake.StateFlags.Geometry))continue;
                    if(planar ? TakeRender.CrossesSegment(origin,end,p.a,p.b):TakeRender.CrossesTriangle(origin,direction,p.a,p.b,p.c))return true;
                }
            }
            if(depth==0)return false;
            nodeIndex=stack[--depth];
        }
    }
    bool Intersects(Node node,Vector3 origin,Vector3 direction)
    {
        double near=0,far=1;
        for(int axis=0;axis<(planar?2:3);axis++)
        {
            if(direction[axis]==0) {if(origin[axis]<node.min[axis] || origin[axis]>node.max[axis])return false;continue;}
            double a=((double)node.min[axis]-origin[axis])/direction[axis],b=((double)node.max[axis]-origin[axis])/direction[axis];
            near=Math.Max(near,Math.Min(a,b));far=Math.Min(far,Math.Max(a,b));if(near>far)return false;
        }
        return true;
    }
}
