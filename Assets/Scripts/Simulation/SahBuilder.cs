using System;
using System.Collections.Generic;
using UnityEngine;

// Binned surface-area-heuristic tree over boxes (README part 27): 12 bins on each axis, leaves of up to 8 boxes where the
// heuristic prefers them, a median split beyond depth 18 (with the median levels below it the tree stays within the GPU
// traversal stack of 32 entries). It writes DirectBvh nodes into `nodes` from `built` on. The object meshes (DirectBvh) and
// the static objects merged in world space (StaticTraceTree) share it. A tree only orders and culls the exact triangle tests,
// so its shape changes the time, not the hits.
internal sealed class SahBuilder
{
    const int Bins = 12, MaximumLeaf = 8, SahDepth = 18;
    readonly Vector3[] low, high, centers;
    readonly bool planar;
    readonly DirectBvh.Node[] nodes;
    readonly Vector3[] binLow = new Vector3[Bins], binHigh = new Vector3[Bins];
    readonly int[] binCount = new int[Bins];
    readonly float[] costLeft = new float[Bins];
    readonly CenterComparer comparer;
    public int Built { get; private set; }

    // low/high: each primitive's box; centers: the point that orders it (a triangle's centroid).
    public SahBuilder(Vector3[] low, Vector3[] high, Vector3[] centers, bool planar, DirectBvh.Node[] nodes, int built)
    {
        this.low = low; this.high = high; this.centers = centers; this.planar = planar; this.nodes = nodes; Built = built;
        comparer = new CenterComparer(centers);
    }
    public int Build(int[] indices, int first, int count) => Sah(indices, first, count, 0);

    static float Area(Vector3 low, Vector3 high) { var e = Vector3.Max(high - low, Vector3.zero); return e.x * e.y + e.y * e.z + e.z * e.x; }
    static DirectBvh.Node Union(DirectBvh.Node a, DirectBvh.Node b) { a.min = Vector3.Min(a.min, b.min); a.max = Vector3.Max(a.max, b.max); return a; }
    int Sah(int[] indices, int first, int count, int depth)
    {
        if (count <= 4 || depth >= SahDepth) return depth >= SahDepth ? Median(indices, first, count) : Leaf(indices, first, count);
        var lowCenter = centers[indices[first]]; var highCenter = lowCenter; var lowBox = Vector3.positiveInfinity; var highBox = Vector3.negativeInfinity;
        for (int i = first; i < first + count; i++)
        {
            lowCenter = Vector3.Min(lowCenter, centers[indices[i]]); highCenter = Vector3.Max(highCenter, centers[indices[i]]);
            lowBox = Vector3.Min(lowBox, low[indices[i]]); highBox = Vector3.Max(highBox, high[indices[i]]);
        }
        float parent = Area(lowBox, highBox), best = float.PositiveInfinity; int bestAxis = -1, bestSplit = 0;
        for (int axis = 0; axis < (planar ? 2 : 3); axis++)
        {
            float extent = highCenter[axis] - lowCenter[axis]; if (!(extent > 0)) continue;
            for (int b = 0; b < Bins; b++) { binLow[b] = Vector3.positiveInfinity; binHigh[b] = Vector3.negativeInfinity; binCount[b] = 0; }
            for (int i = first; i < first + count; i++)
            {
                int b = Math.Min(Bins - 1, (int)((centers[indices[i]][axis] - lowCenter[axis]) / extent * Bins));
                binLow[b] = Vector3.Min(binLow[b], low[indices[i]]); binHigh[b] = Vector3.Max(binHigh[b], high[indices[i]]); binCount[b]++;
            }
            var accLow = Vector3.positiveInfinity; var accHigh = Vector3.negativeInfinity; int n = 0;
            for (int b = 0; b < Bins - 1; b++) { accLow = Vector3.Min(accLow, binLow[b]); accHigh = Vector3.Max(accHigh, binHigh[b]); n += binCount[b]; costLeft[b] = n == 0 ? 0 : Area(accLow, accHigh) * n; }
            accLow = Vector3.positiveInfinity; accHigh = Vector3.negativeInfinity; n = 0;
            for (int b = Bins - 1; b > 0; b--)
            {
                accLow = Vector3.Min(accLow, binLow[b]); accHigh = Vector3.Max(accHigh, binHigh[b]); n += binCount[b];
                if (n == 0 || n == count) continue;
                float cost = costLeft[b - 1] + Area(accLow, accHigh) * n;
                if (cost < best) { best = cost; bestAxis = axis; bestSplit = b; }
            }
        }
        // Leaf if splitting does not pay (traversal cost 1 against one triangle test each).
        if (bestAxis < 0 || (count <= MaximumLeaf && 1 + best / Math.Max(parent, 1e-20f) >= count))
            return bestAxis < 0 ? (count <= MaximumLeaf ? Leaf(indices, first, count) : Median(indices, first, count)) : Leaf(indices, first, count);
        float lo = lowCenter[bestAxis], span = highCenter[bestAxis] - lo;
        int left = first, right = first + count - 1;
        while (left <= right)
        {
            int b = Math.Min(Bins - 1, (int)((centers[indices[left]][bestAxis] - lo) / span * Bins));
            if (b < bestSplit) left++; else { int t = indices[left]; indices[left] = indices[right]; indices[right] = t; right--; }
        }
        int split = left - first;
        if (split == 0 || split == count) return Median(indices, first, count);
        int index = Built++;
        int l2 = Sah(indices, first, split, depth + 1), r2 = Sah(indices, first + split, count - split, depth + 1);
        var branch = Union(nodes[l2], nodes[r2]); branch.left = (uint)l2; branch.right = (uint)r2; branch.first = branch.count = 0; nodes[index] = branch; return index;
    }
    // Median split on the centres' longest axis, leaves of up to 4.
    int Median(int[] indices, int first, int count)
    {
        if (count <= 4) return Leaf(indices, first, count);
        int index = Built++;
        var lowCenter = centers[indices[first]]; var highCenter = lowCenter;
        for (int i = first + 1; i < first + count; i++) { lowCenter = Vector3.Min(lowCenter, centers[indices[i]]); highCenter = Vector3.Max(highCenter, centers[indices[i]]); }
        var size = highCenter - lowCenter; comparer.axis = size.x >= size.y ? 0 : 1; if (size.z > size[comparer.axis]) comparer.axis = 2;
        Array.Sort(indices, first, count, comparer);
        int left = Median(indices, first, count / 2), right = Median(indices, first + count / 2, count - count / 2);
        var branch = Union(nodes[left], nodes[right]); branch.left = (uint)left; branch.right = (uint)right; branch.first = branch.count = 0; nodes[index] = branch; return index;
    }
    int Leaf(int[] indices, int first, int count)
    {
        int index = Built++;
        var node = new DirectBvh.Node {min = Vector3.positiveInfinity, max = Vector3.negativeInfinity, first = (uint)first, count = (uint)count};
        for (int i = first; i < first + count; i++) { node.min = Vector3.Min(node.min, low[indices[i]]); node.max = Vector3.Max(node.max, high[indices[i]]); }
        nodes[index] = node; return index;
    }
    internal sealed class CenterComparer : IComparer<int>
    {
        readonly Vector3[] values; public int axis;
        public CenterComparer(Vector3[] values) { this.values = values; }
        public int Compare(int a, int b) { int order = values[a][axis].CompareTo(values[b][axis]); return order == 0 ? a.CompareTo(b) : order; }
    }
}
