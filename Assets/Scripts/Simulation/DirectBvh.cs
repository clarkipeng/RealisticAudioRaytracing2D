using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Runtime.InteropServices;
using UnityEngine;

// Static geometry uses one world-space hierarchy. Once an object's pose changes it
// stays in the direct-test list for this solver's lifetime, including when it stops.
internal sealed class DirectBvh
{
    [StructLayout(LayoutKind.Sequential,Pack=4)] public struct Node
    {
        public Vector4 min,max;
        public uint left,right,first,count;
    }
    public readonly Node[] Nodes;
    // Static primitives occupy the sorted prefix; moving primitives occupy the suffix.
    // Every entry retains its original primitive ID.
    public readonly int[] Indices;
    readonly AcousticTake.Primitive[] local,world;
    readonly AcousticTake.State[] initial;
    readonly bool[] hasGeometry,moving;
    readonly int[] stack=new int[32];
    readonly Vector3[] centers;
    readonly CenterComparer comparer;
    readonly bool planar;
    bool initialized;
    int built;
    public int NodeCount => built;
    public int StaticPrimitiveCount {get;private set;}
    public double BuildSeconds {get;private set;}
    public double UpdateSeconds {get;private set;}
    public int UpdateCount {get;private set;}
    public long NumericBytes => (long)world.Length*(44+4+12)+(long)Nodes.Length*48+(long)initial.Length*(80+1+1)+stack.Length*4;

    public DirectBvh(AcousticTake.Scene scene,AcousticTake.Primitive[] geometry)
    {
        local=geometry; world=new AcousticTake.Primitive[geometry.Length]; Indices=new int[geometry.Length];
        initial=new AcousticTake.State[scene.objects.Length];hasGeometry=new bool[initial.Length];moving=new bool[initial.Length];
        centers=new Vector3[geometry.Length];comparer=new CenterComparer(centers);
        planar=scene.dimension==AcousticTake.Dimension.Planar2D;
        foreach(var primitive in geometry)hasGeometry[primitive.objectIndex]=true;
        Nodes=new Node[CountNodes(geometry.Length)];
    }
    static int CountNodes(int count) => count==0 ? 0:count<=4 ? 1:checked(1+CountNodes(count/2)+CountNodes(count-count/2));
    static float AbsMax(Vector3 value) => Mathf.Max(Mathf.Abs(value.x),Mathf.Max(Mathf.Abs(value.y),Mathf.Abs(value.z)));
    AcousticTake.Primitive Transform(AcousticTake.Primitive p,AcousticTake.State pose)
    {
        p.a=pose.position+pose.rotation*p.a;p.b=pose.position+pose.rotation*p.b;
        p.c=planar?Vector3.zero:pose.position+pose.rotation*p.c;return p;
    }
    public bool Update(AcousticTake.State[] state,int offset=0)
    {
        long start=Stopwatch.GetTimestamp();bool changed=!initialized;
        if(!initialized)
        {
            Array.Copy(state,offset,initial,0,initial.Length);
            for(int i=0;i<local.Length;i++)
            {
                var p=world[i]=Transform(local[i],initial[local[i].objectIndex]);
                centers[i]=planar?(p.a+p.b)*.5f:(p.a+p.b+p.c)/3;
            }
        }
        else for(int owner=0;owner<initial.Length;owner++)
        {
            if(!hasGeometry[owner] || moving[owner])continue;
            var pose=state[offset+owner];
            if(pose.position.Equals(initial[owner].position) && pose.rotation.Equals(initial[owner].rotation))continue;
            moving[owner]=true;changed=true;
        }
        if(changed)
        {
            int fixedCount=0,dynamicStart=local.Length;
            for(int i=0;i<local.Length;i++)
                if(moving[local[i].objectIndex])Indices[--dynamicStart]=i;
                else Indices[fixedCount++]=i;
            StaticPrimitiveCount=fixedCount;built=0;
            if(fixedCount>0)Build(0,fixedCount);
            UpdateCount++;
        }
        if(!initialized) {initialized=true;BuildSeconds+=TakeRender.Elapsed(start);}
        else UpdateSeconds+=TakeRender.Elapsed(start);
        return changed;
    }
    sealed class CenterComparer : IComparer<int>
    {
        readonly Vector3[] values;
        public int axis;
        public CenterComparer(Vector3[] values) {this.values=values;}
        public int Compare(int a,int b) {int order=values[a][axis].CompareTo(values[b][axis]);return order==0?a.CompareTo(b):order;}
    }
    int Build(int first,int count)
    {
        int index=built++;
        if(count<=4)
        {
            Vector3 min=Vector3.positiveInfinity,max=Vector3.negativeInfinity;
            for(int i=first;i<first+count;i++)
            {
                int id=Indices[i];var p=world[id];var original=local[id];
                var low=Vector3.Min(p.a,p.b);var high=Vector3.Max(p.a,p.b);
                if(!planar) {low=Vector3.Min(low,p.c);high=Vector3.Max(high,p.c);}
                // Conservative broad phase: CPU/GPU quaternion arithmetic may round differently.
                float magnitude=1+AbsMax(initial[p.objectIndex].position)+4*Mathf.Max(AbsMax(original.a),Mathf.Max(AbsMax(original.b),planar?0:AbsMax(original.c)));
                var error=Vector3.one*(8*1.1920929e-7f*magnitude);
                min=Vector3.Min(min,low-error);max=Vector3.Max(max,high+error);
            }
            Nodes[index]=new Node {min=min,max=max,left=uint.MaxValue,right=uint.MaxValue,first=(uint)first,count=(uint)count};
            return index;
        }
        Vector3 centerMin=centers[Indices[first]],centerMax=centerMin;
        for(int i=first+1;i<first+count;i++) {centerMin=Vector3.Min(centerMin,centers[Indices[i]]);centerMax=Vector3.Max(centerMax,centers[Indices[i]]);}
        var size=centerMax-centerMin;comparer.axis=size.x>=size.y ? 0:1;
        if(!planar && size.z>size[comparer.axis])comparer.axis=2;
        Array.Sort(Indices,first,count,comparer);
        int left=Build(first,count/2),right=Build(first+count/2,count-count/2);
        Nodes[index]=new Node {left=(uint)left,right=(uint)right,min=Vector3.Min(Nodes[left].min,Nodes[right].min),max=Vector3.Max(Nodes[left].max,Nodes[right].max)};
        return index;
    }
    bool Hit(AcousticTake.Primitive p,Vector3 origin,Vector3 end,AcousticTake.State[] state)
    {
        if((state[p.objectIndex].flags&(AcousticTake.StateFlags.Active|AcousticTake.StateFlags.Geometry))!=(AcousticTake.StateFlags.Active|AcousticTake.StateFlags.Geometry))return false;
        return planar ? TakeRender.CrossesSegment(origin,end,p.a,p.b):TakeRender.CrossesTriangle(origin,end-origin,p.a,p.b,p.c);
    }
    public bool Blocked(Vector3 origin,Vector3 end,AcousticTake.State[] state)
    {
        if(built>0)
        {
            int depth=0,nodeIndex=0;var direction=end-origin;
            // Median splits of an int-sized primitive array require at most 31 pending nodes.
            while(true)
            {
                var node=Nodes[nodeIndex];
                if(Intersects(node,origin,direction))
                {
                    if(node.count==0) {stack[depth++]=(int)node.right;nodeIndex=(int)node.left;continue;}
                    for(uint j=0;j<node.count;j++)if(Hit(world[Indices[node.first+j]],origin,end,state))return true;
                }
                if(depth==0)break;
                nodeIndex=stack[--depth];
            }
        }
        for(int i=StaticPrimitiveCount;i<Indices.Length;i++)
        {
            var p=local[Indices[i]];
            if(Hit(Transform(p,state[p.objectIndex]),origin,end,state))return true;
        }
        return false;
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
