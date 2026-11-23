using System;
using System.Threading.Tasks;
using UnityEngine;

// The tree the GPU's scene queries descend (README part 48; Geometry.hlsl TraceScene, BvhTraceRoot): every object that has
// not moved since the first update merged into one world-space mesh tree, under a small instance tree with the objects that
// have moved. Each object of the instance tree (DirectBvh) has its own mesh tree, and the boxes of the Church's walls, roofs,
// arcades and meadow all span the nave, so a ray there descends a dozen mesh trees from their roots (77 node visits per walk
// segment, part 46); the merged tree is one SAH tree over the triangles themselves.
// Exact: the triangles are still tested by their objects' poses in world space and ties go to the lower index, so the hits
// do not depend on the tree; the boxes are padded beyond the rounding of the pose. An object that moves (or changes its active
// or geometry flags) leaves the merged set for good; until the merged tree without it is rebuilt (on a worker thread) the
// queries descend the instance tree, as before. Media (MediumAt) and walk diffraction keep the instance tree.
internal sealed class StaticTraceTree
{
    public const uint MergedInstance = 0xfffffffe; // a top node's count: the merged tree (identity pose), its root in `left`
    readonly AcousticTake.Primitive[] geometry;
    readonly int objectCount, nodeBase, indexBase, topCapacity;
    readonly bool[] dynamicObject;
    readonly Vector3[] referencePosition;
    readonly Quaternion[] referenceRotation;
    bool initialized;
    // Merged tree (relative to its own arrays) and the top tree: nodes from nodeBase in the GPU buffer, indices from indexBase.
    DirectBvh.Node[] merged = Array.Empty<DirectBvh.Node>();
    int[] mergedIndices = Array.Empty<int>();
    int mergedRoot = -1;
    Task<(DirectBvh.Node[] nodes, int[] indices, int root, int tree)> rebuilding;
    public readonly DirectBvh.Node[] Nodes;    // [top tree topCapacity][merged tree], absolute links
    public readonly int[] Indices;             // the merged tree's triangle order, absolute (from indexBase)
    public int TopCount { get; private set; }
    public int MergedNodeCount { get; private set; }
    public bool MergedChanged { get; private set; }
    public int Root { get; private set; }      // 0: no merged tree (the instance tree's root)
    public int Rebuilds { get; private set; }
    public bool Enabled = true;                // profiling: false descends the instance tree (MeasurePathLiveBudget merged=0)
    // The merged tree's leaves hold their triangles in world space in the node array (Geometry.hlsl ConsiderTriangle, README
    // part 49) instead of primitive indices; profiling: false keeps the indices (MeasurePathLiveBudget inline=0).
    public bool InlineTriangles = true;
    public const uint TriangleLeaf = 0x40000000;
    public int Capacity => Nodes.Length;
    public int MergedStart => topCapacity;

    public StaticTraceTree(AcousticTake.Primitive[] geometry, int objectCount, int nodeBase, int indexBase)
    {
        this.geometry = geometry; this.objectCount = objectCount; this.nodeBase = nodeBase; this.indexBase = indexBase;
        dynamicObject = new bool[objectCount]; referencePosition = new Vector3[objectCount]; referenceRotation = new Quaternion[objectCount];
        topCapacity = 2 * (objectCount + 1);
        Nodes = new DirectBvh.Node[topCapacity + Math.Max(1, 3 * geometry.Length)]; // [top][merged tree: at most 2n - 1][its triangles: n]
        Indices = new int[Math.Max(1, geometry.Length)];
    }

    // After DirectBvh's update of the instance tree (its nodes and instance boxes): whether this region of the GPU buffer changed.
    public bool Update(AcousticTake.State[] state, int frames, DirectBvh instances)
    {
        MergedChanged = false;
        bool demoted = false;
        const AcousticTake.StateFlags present = AcousticTake.StateFlags.Active | AcousticTake.StateFlags.Geometry;
        for (int owner = 0; owner < objectCount; owner++)
        {
            if (dynamicObject[owner]) continue;
            if (!initialized) { referencePosition[owner] = state[owner].position; referenceRotation[owner] = state[owner].rotation; }
            for (int f = 0; f < frames && !dynamicObject[owner]; f++)
            {
                var pose = state[f * objectCount + owner];
                if (!pose.position.Equals(referencePosition[owner]) || !pose.rotation.Equals(referenceRotation[owner]) || (pose.flags & present) != present)
                    dynamicObject[owner] = demoted = true;
            }
        }
        if (!initialized || demoted)
        {
            // The first merged tree is built at once; later ones on a worker thread, the instance tree serving meanwhile.
            mergedRoot = -1; rebuilding = null;
            var poses = new (Vector3 position, Quaternion rotation)[objectCount];
            for (int owner = 0; owner < objectCount; owner++) poses[owner] = (referencePosition[owner], referenceRotation[owner]);
            var excluded = (bool[])dynamicObject.Clone();
            bool inline = InlineTriangles;
            if (!initialized) Install(BuildMerged(geometry, poses, excluded, inline));
            else rebuilding = Task.Run(() => BuildMerged(geometry, poses, excluded, inline));
            initialized = true;
        }
        else if (rebuilding != null && rebuilding.IsCompleted)
        {
            var result = rebuilding.Result; rebuilding = null;
            Install(result); Rebuilds++;
        }
        BuildTop(instances);
        return true;
    }
    void Install((DirectBvh.Node[] nodes, int[] indices, int root, int tree) tree)
    {
        merged = tree.nodes; mergedIndices = tree.indices; mergedRoot = tree.root; MergedNodeCount = merged.Length;
        for (int i = 0; i < merged.Length; i++)
        {
            var node = merged[i];
            if (i >= tree.tree) { Nodes[topCapacity + i] = node; continue; } // (a stored triangle)
            if (node.count == 0) { node.left += (uint)(nodeBase + topCapacity); node.right += (uint)(nodeBase + topCapacity); }
            else node.first += (node.count & TriangleLeaf) != 0 ? (uint)(nodeBase + topCapacity) : (uint)indexBase;
            Nodes[topCapacity + i] = node;
        }
        Array.Copy(mergedIndices, Indices, mergedIndices.Length);
        MergedChanged = true;
    }
    // One SAH tree over the merged objects' triangles at their reference poses, the boxes padded beyond the pose's rounding
    // (as DirectBvh.WorldBounds). Thread-safe: it reads only its arguments.
    static (DirectBvh.Node[] nodes, int[] indices, int root, int tree) BuildMerged(AcousticTake.Primitive[] geometry, (Vector3 position, Quaternion rotation)[] poses, bool[] excluded, bool inline)
    {
        int count = 0;
        foreach (var p in geometry) if (!excluded[p.objectIndex]) count++;
        if (count == 0) return (Array.Empty<DirectBvh.Node>(), Array.Empty<int>(), -1, 0);
        var low = new Vector3[geometry.Length]; var high = new Vector3[geometry.Length]; var middle = new Vector3[geometry.Length];
        var world = new Vector3[3 * geometry.Length];
        var order = new int[count]; int n = 0;
        for (int i = 0; i < geometry.Length; i++)
        {
            var p = geometry[i];
            if (excluded[p.objectIndex]) continue;
            var pose = poses[p.objectIndex];
            Vector3 a = pose.position + pose.rotation * p.a, b = pose.position + pose.rotation * p.b, c = pose.position + pose.rotation * p.c;
            world[3 * i] = a; world[3 * i + 1] = b; world[3 * i + 2] = c;
            float local = Mathf.Max(AbsMax(p.a), Mathf.Max(AbsMax(p.b), AbsMax(p.c)));
            var pad = Vector3.one * (8 * 1.1920929e-7f * (1 + AbsMax(pose.position) + 4 * local));
            low[i] = Vector3.Min(a, Vector3.Min(b, c)) - pad; high[i] = Vector3.Max(a, Vector3.Max(b, c)) + pad; middle[i] = (a + b + c) / 3;
            order[n++] = i;
        }
        var nodes = new DirectBvh.Node[2 * count - 1];
        var builder = new SahBuilder(low, high, middle, false, nodes, 0);
        int root = builder.Build(order, 0, count);
        int built = builder.Built;
        if (!inline) { Array.Resize(ref nodes, built); return (nodes, order, root, built); }
        // Each leaf's triangles, in its order, after the tree: node `built + k` holds the k-th triangle of `order`.
        Array.Resize(ref nodes, built + count);
        for (int i = 0; i < built; i++)
            if (nodes[i].count != 0) { nodes[i].first += (uint)built; nodes[i].count |= TriangleLeaf; }
        for (int k = 0; k < count; k++)
        {
            int id = order[k];
            Vector3 a = world[3 * id], e1 = world[3 * id + 1] - a, e2 = world[3 * id + 2] - a;
            nodes[built + k] = new DirectBvh.Node {min = new Vector4(a.x, a.y, a.z, BitConverter.Int32BitsToSingle(id)),
                max = new Vector4(e1.x, e1.y, e1.z, 8 * 1.1920929e-7f * e1.magnitude * e2.magnitude),
                left = (uint)BitConverter.SingleToInt32Bits(e2.x), right = (uint)BitConverter.SingleToInt32Bits(e2.y), first = (uint)BitConverter.SingleToInt32Bits(e2.z)};
        }
        return (nodes, order, root, built);
    }
    static float AbsMax(Vector3 v) => Mathf.Max(Mathf.Abs(v.x), Mathf.Max(Mathf.Abs(v.y), Mathf.Abs(v.z)));
    // The top tree: the merged tree as one instance (identity pose) and every moved object as its own (DirectBvh's instance
    // node: its world box, its mesh root), by a median split of their centres.
    void BuildTop(DirectBvh instances)
    {
        var leaves = new System.Collections.Generic.List<DirectBvh.Node>();
        if (mergedRoot >= 0)
        {
            var root = merged[mergedRoot];
            leaves.Add(new DirectBvh.Node {min = root.min, max = root.max, left = (uint)(nodeBase + topCapacity + mergedRoot), count = MergedInstance});
        }
        bool complete = mergedRoot >= 0 || !Enabled;
        for (int owner = 0; owner < objectCount && complete; owner++)
            if (dynamicObject[owner] && instances.HasMesh(owner)) leaves.Add(instances.Instance(owner));
        TopCount = 0;
        if (!Enabled || mergedRoot < 0 || leaves.Count == 0) { Root = 0; return; }
        var order = new int[leaves.Count];
        for (int i = 0; i < order.Length; i++) order[i] = i;
        Root = nodeBase + Top(leaves, order, 0, order.Length);
    }
    int Top(System.Collections.Generic.List<DirectBvh.Node> leaves, int[] order, int first, int count)
    {
        int index = TopCount++;
        if (count == 1) { Nodes[index] = leaves[order[first]]; return index; }
        Vector3 lo = Vector3.positiveInfinity, hi = Vector3.negativeInfinity;
        for (int i = first; i < first + count; i++) { var c = ((Vector3)leaves[order[i]].min + (Vector3)leaves[order[i]].max) * .5f; lo = Vector3.Min(lo, c); hi = Vector3.Max(hi, c); }
        var size = hi - lo; int axis = size.x >= size.y ? 0 : 1; if (size.z > size[axis]) axis = 2;
        Array.Sort(order, first, count, Comparer(leaves, axis));
        int left = Top(leaves, order, first, count / 2), right = Top(leaves, order, first + count / 2, count - count / 2);
        var a = Nodes[left]; var b = Nodes[right];
        Nodes[index] = new DirectBvh.Node {min = Vector3.Min(a.min, b.min), max = Vector3.Max(a.max, b.max), left = (uint)(nodeBase + left), right = (uint)(nodeBase + right)};
        return index;
    }
    static System.Collections.Generic.IComparer<int> Comparer(System.Collections.Generic.List<DirectBvh.Node> leaves, int axis) =>
        System.Collections.Generic.Comparer<int>.Create((x, y) =>
        {
            float cx = leaves[x].min[axis] + leaves[x].max[axis], cy = leaves[y].min[axis] + leaves[y].max[axis];
            int o = cx.CompareTo(cy); return o == 0 ? x.CompareTo(y) : o;
        });
}
