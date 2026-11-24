// Scene ABI, rigid motion and shared segment/triangle traversal.
struct Primitive
{
    float4 a, b, c;
    uint4 indices;
};
struct State
{
    float4 position, rotation, velocity, angularVelocity;
    uint4 flags;
};
StructuredBuffer<Primitive> Primitives;
StructuredBuffer<State> States;
uint PrimitiveCount, PairCount, ListenerCount, ObjectCount, FrameCount;
float SpeedOfSound, ReferenceDistance, MinimumDistance;

float3 Rotate(float4 q, float3 v)
{
    return v + 2 * cross(q.xyz, cross(q.xyz, v) + q.w * v);
}
// Motion is valid only within a continuous pose segment. Cuts never imply velocity.
float3 Velocity(uint index)
{
    State s = States[index];
    return (s.flags.x & 48) == 32 ? s.velocity.xyz : float3(0, 0, 0);
}
float3 PointVelocity(uint index, float3 location, float lag = 0)
{
    State s = States[index];
    return Velocity(index) +
           ((s.flags.x & 48) == 32
                ? cross(s.angularVelocity.xyz, location - (s.position.xyz - Velocity(index) * lag))
                : float3(0, 0, 0));
}
static const float RayEpsilon = 0.0001, RayPi = 3.141592653589793;
// Work profile (PATH_PROFILE kernels only, MethodTransport.WorkProfile; README part 46): per thread, the scene traversals by
// the kind of query that made them (ProfileKind, set before each query: PROFILE_KIND), and their node visits and triangle tests.
#ifdef PATH_PROFILE
static uint ProfileKind = 0;
static uint4 ProfileQueriesLow = 0, ProfileQueriesHigh = 0, ProfileNodesLow = 0, ProfileNodesHigh = 0, ProfileTrianglesLow = 0, ProfileTrianglesHigh = 0;
void ProfileAdd(inout uint4 low, inout uint4 high, uint amount)
{
    low += amount * (uint4)(ProfileKind == uint4(0, 1, 2, 3)); high += amount * (uint4)(ProfileKind == uint4(4, 5, 6, 7));
}
uint ProfileSum(uint4 v) { return v.x + v.y + v.z + v.w; }
#define PROFILE_KIND(k) ProfileKind = (k)
#define PROFILE_QUERY() ProfileAdd(ProfileQueriesLow, ProfileQueriesHigh, 1)
#define PROFILE_NODE() ProfileAdd(ProfileNodesLow, ProfileNodesHigh, 1)
#define PROFILE_TRIANGLE() ProfileAdd(ProfileTrianglesLow, ProfileTrianglesHigh, 1)
#else
#define PROFILE_KIND(k)
#define PROFILE_QUERY()
#define PROFILE_NODE()
#define PROFILE_TRIANGLE()
#endif
float3 PastPoint(State state, float3 local, float lag)
{
    float3 worldPoint = Rotate(state.rotation, local);
    if ((state.flags.x & 48) == 32)
    {
        float angle = length(state.angularVelocity.xyz) * lag;
        if (angle > 0)
        {
            float3 axis = normalize(state.angularVelocity.xyz);
            worldPoint = worldPoint * cos(angle) - cross(axis, worldPoint) * sin(angle) +
                         axis * dot(axis, worldPoint) * (1 - cos(angle));
        }
        return state.position.xyz - state.velocity.xyz * lag + worldPoint;
    }
    return state.position.xyz + worldPoint;
}
float IntersectPrimitive(uint id, uint stateOffset, float3 o, float3 d, bool is2D, float lag, float minimum,
                         out float3 normal)
{
    Primitive p = Primitives[id];
    State state = States[stateOffset + p.indices.x];
    normal = 0;
    if ((state.flags.x & 3) != 3)
        return 1e30;
    float3 a = PastPoint(state, p.a.xyz, lag), b = PastPoint(state, p.b.xyz, lag);
    if (is2D)
    {
        float2 e = b.xy - a.xy, q = a.xy - o.xy;
        float det = d.x * e.y - d.y * e.x;
        if (abs(det) <= 8 * 1.1920929e-7 * length(d.xy) * length(e))
            return 1e30;
        float t = (q.x * e.y - q.y * e.x) / det, u = (q.x * d.y - q.y * d.x) / det;
        normal = normalize(float3(-e.y, e.x, 0));
        return t > minimum && u >= 0 && u <= 1 ? t : 1e30;
    }
    float3 c = PastPoint(state, p.c.xyz, lag), e1 = b - a, e2 = c - a, h = cross(d, e2);
    float det = dot(e1, h);
    // A rotated coplanar ray has a float-roundoff determinant, not an
    // intersection. Scale the parallel test with the operands, as in 2D.
    if (abs(det) <= 8 * 1.1920929e-7 * length(d) * length(e1) * length(e2))
        return 1e30;
    float3 s = o - a, q = cross(s, e1);
    float u = dot(s, h) / det, v = dot(d, q) / det, t = dot(e2, q) / det;
    normal = normalize(cross(e1, e2));
    return t > minimum && u >= 0 && v >= 0 && u + v <= 1 ? t : 1e30;
}
float PrimitiveDistanceAt(uint id, uint stateOffset, float3 origin, float3 direction, bool is2D, float lag,
                          out float3 normal)
{
    return IntersectPrimitive(id, stateOffset, origin, direction, is2D, lag, RayEpsilon, normal);
}
float PrimitiveDistance(uint id, uint stateOffset, float3 origin, float3 direction, bool is2D,
                        out float3 normal)
{
    return PrimitiveDistanceAt(id, stateOffset, origin, direction, is2D, 0, normal);
}
#ifdef USE_BVH
struct BvhNode
{
    float4 boundsMin, boundsMax;
    uint4 links;
};
StructuredBuffer<BvhNode> BvhNodes;
StructuredBuffer<uint> BvhIndices;
uint BvhNodeCount;
// The scene queries' tree (StaticTraceTree, README part 48): the objects that have not moved merged into one world-space mesh
// tree (a top node with links.w = 0xfffffffe, its root in links.x, descended with the identity pose), under the moved
// objects' instances. 0: the instance tree (its root), which media and walk diffraction always use.
uint BvhTraceRoot;
bool BoundsHit(BvhNode node, float3 o, float3 d, bool is2D)
{
    float near = 0, far = 1;
    for (uint axis = 0; axis < (is2D ? 2 : 3); axis++)
    {
        if (d[axis] == 0)
        {
            if (o[axis] < node.boundsMin[axis] || o[axis] > node.boundsMax[axis])
                return false;
            continue;
        }
        float a = (node.boundsMin[axis] - o[axis]) / d[axis], b = (node.boundsMax[axis] - o[axis]) / d[axis];
        near = max(near, min(a, b));
        far = min(far, max(a, b));
        if (near > far)
            return false;
    }
    return true;
}
#endif
// Resolve exact ties by primitive ID, independently of hierarchy visitation order.
bool ConsiderPrimitive(uint id, uint stateOffset, float3 origin, float3 direction, float minimum,
                       bool is2D, inout uint hit, inout float distance, inout float3 normal)
{
    float3 candidateNormal;
    PROFILE_TRIANGLE();
    float candidate = IntersectPrimitive(id, stateOffset, origin, direction, is2D, 0, minimum, candidateNormal);
    if (candidate < distance || (hit != 0xffffffff && candidate == distance && id < hit))
    {
        hit = id; distance = candidate; normal = candidateNormal; return true;
    }
    return false;
}
#ifdef USE_BVH
// A triangle stored in the node array (StaticTraceTree's merged tree, README part 49): its first vertex and the primitive's index
// (boundsMin), the first edge and 8 epsilon |e1| |e2| (boundsMax), the second edge (links.xyz), all in world space, so the test
// loads no primitive or pose and rotates nothing. The arithmetic is IntersectPrimitive's.
static const uint TriangleLeaf = 0x40000000u; // a leaf's links.w: its triangles are stored from links.z in the node array
bool ConsiderTriangle(BvhNode t, float3 origin, float3 direction, float minimum, inout uint hit, inout float distance, inout float3 normal)
{
    PROFILE_TRIANGLE();
    uint id = asuint(t.boundsMin.w);
    float3 a = t.boundsMin.xyz, e1 = t.boundsMax.xyz, e2 = asfloat(t.links.xyz), h = cross(direction, e2);
    float det = dot(e1, h);
    if (abs(det) <= t.boundsMax.w * length(direction)) return false;
    float3 s = origin - a, q = cross(s, e1);
    float u = dot(s, h) / det, v = dot(direction, q) / det, along = dot(e2, q) / det;
    float candidate = along > minimum && u >= 0 && v >= 0 && u + v <= 1 ? along : 1e30;
    if (candidate < distance || (hit != 0xffffffff && candidate == distance && id < hit))
    {
        hit = id; distance = candidate; normal = normalize(cross(e1, e2)); return true;
    }
    return false;
}
BvhNode WorldBounds(BvhNode node, State pose)
{
    float3 low = node.boundsMin.xyz, high = node.boundsMax.xyz;
    float3 center = (low + high) * .5, halfSize = (high - low) * .5;
    float3 extent = abs(Rotate(pose.rotation, float3(1,0,0))) * halfSize.x +
                    abs(Rotate(pose.rotation, float3(0,1,0))) * halfSize.y +
                    abs(Rotate(pose.rotation, float3(0,0,1))) * halfSize.z;
    float3 magnitude = abs(pose.position.xyz) + 4 * max(abs(low), abs(high));
    extent += 8 * 1.1920929e-7 * (1 + max(magnitude.x, max(magnitude.y, magnitude.z)));
    center = pose.position.xyz + Rotate(pose.rotation, center);
    node.boundsMin.xyz = center - extent; node.boundsMax.xyz = center + extent; return node;
}
// Planar scenes: geometry is projected onto the plane (extruded along world z), which a mesh-frame box cannot
// express under an out-of-plane rotation, so they keep the rotated world bounds per node.
bool TraceMeshPlanar(uint root, State pose, uint stateOffset, float3 origin, float3 direction, float minimum,
               bool is2D, bool anyHit, inout uint hit, inout float distance, inout float3 normal)
{
    uint stack[32], depth = 0, index = root;
    while (true)
    {
        BvhNode node = BvhNodes[index];
        PROFILE_NODE();
        if (BoundsHit(WorldBounds(node, pose), origin, direction * distance, is2D))
        {
            if (node.links.w == 0) { stack[depth++] = node.links.y; index = node.links.x; continue; }
            for (uint leaf = 0; leaf < node.links.w; leaf++)
                if (ConsiderPrimitive(BvhIndices[node.links.z + leaf], stateOffset, origin, direction,
                                      minimum, is2D, hit, distance, normal) && anyHit) return true;
        }
        if (depth == 0) break;
        index = stack[--depth];
    }
    return false;
}
// A mesh tree's box in the mesh's own frame against the ray brought into that frame (exact boxes, no rotated
// world bounds per node), over [0, limit]; `pad` widens the box by more than the transform's rounding, so a
// box is never missed that the world-space triangle test would hit. `entry` orders the children.
bool LocalBox(BvhNode node, float3 o, float3 inv, float limit, float pad, bool is2D, out float entry)
{
    float3 a = (node.boundsMin.xyz - pad - o) * inv, b = (node.boundsMax.xyz + pad - o) * inv;
    float3 low = min(a, b), high = max(a, b);
    entry = max(max(low.x, low.y), max(low.z, 0));
    return entry <= min(min(high.x, high.y), min(high.z, limit));
}
float SafeInverse(float v) { return 1 / (abs(v) < 1e-30 ? (v < 0 ? -1e-30 : 1e-30) : v); }
// Ordered traversal (the nearer child first, the farther one kept on the stack and culled when popped if a
// closer hit has been found since); the triangles are tested in world space exactly as before, and exact ties
// are resolved by primitive ID, so the hit does not depend on the order.
bool TraceMesh(uint root, State pose, uint stateOffset, float3 origin, float3 direction, float minimum,
               bool is2D, bool anyHit, inout uint hit, inout float distance, inout float3 normal)
{
    if (is2D) return TraceMeshPlanar(root, pose, stateOffset, origin, direction, minimum, is2D, anyHit, hit, distance, normal);
    float4 inverse = pose.rotation; inverse.xyz = -inverse.xyz;
    float3 o = Rotate(inverse, origin - pose.position.xyz), d = Rotate(inverse, direction);
    float3 inv = float3(SafeInverse(d.x), SafeInverse(d.y), SafeInverse(d.z));
    BvhNode top = BvhNodes[root];
    float3 m = max(abs(top.boundsMin.xyz), abs(top.boundsMax.xyz)), p = abs(pose.position.xyz), q = abs(origin);
    float pad = 16 * 1.1920929e-7 * (1 + max(p.x, max(p.y, p.z)) + 4 * max(m.x, max(m.y, m.z)) + max(q.x, max(q.y, q.z)));
    uint stack[32], depth = 0, index = root;
    float entry;
    bool tested = false;
    while (true)
    {
        BvhNode node = BvhNodes[index];
        PROFILE_NODE();
        if (tested || LocalBox(node, o, inv, distance, pad, is2D, entry))
        {
            if (node.links.w == 0)
            {
                float nearLeft, nearRight;
                bool left = LocalBox(BvhNodes[node.links.x], o, inv, distance, pad, is2D, nearLeft);
                bool right = LocalBox(BvhNodes[node.links.y], o, inv, distance, pad, is2D, nearRight);
                if (left && right)
                {
                    bool leftFirst = nearLeft <= nearRight;
                    stack[depth++] = leftFirst ? node.links.y : node.links.x;
                    index = leftFirst ? node.links.x : node.links.y; tested = true; continue;
                }
                if (left || right) { index = left ? node.links.x : node.links.y; tested = true; continue; }
            }
            else if ((node.links.w & TriangleLeaf) != 0)
            {
                for (uint stored = 0; stored < (node.links.w & 0xffffu); stored++)
                    if (ConsiderTriangle(BvhNodes[node.links.z + stored], origin, direction, minimum, hit, distance, normal) && anyHit) return true;
            }
            else
                for (uint leaf = 0; leaf < node.links.w; leaf++)
                    if (ConsiderPrimitive(BvhIndices[node.links.z + leaf], stateOffset, origin, direction,
                                          minimum, is2D, hit, distance, normal) && anyHit) return true;
        }
        if (depth == 0) break;
        index = stack[--depth]; tested = false;
    }
    return false;
}
#endif
// Shared any-hit / closest-hit traversal. Each median-split level needs at most 32 stack entries.
bool TraceScene(uint stateOffset, float3 origin, float3 direction, float minimum, float maximum, bool is2D,
                bool anyHit, uint ignoredOwner, out uint hit, out float distance, out float3 normal)
{
    hit = 0xffffffff; distance = maximum; normal = 0;
    PROFILE_QUERY();
#ifdef USE_BVH
    uint stack[32], depth = 0, index = ignoredOwner == 0xffffffff ? BvhTraceRoot : 0;
    while (BvhNodeCount > 0) // (the trace tree lies after the BvhNodeCount nodes of the instance tree)
    {
        BvhNode node = BvhNodes[index];
        PROFILE_NODE();
        if (BoundsHit(node, origin, direction * distance, is2D))
        {
            if (node.links.w == 0) { stack[depth++] = node.links.y; index = node.links.x; continue; }
            bool merged = node.links.w == 0xfffffffe;
            State pose = States[stateOffset + (merged ? 0 : node.links.z)];
            if (merged) { pose.position = 0; pose.rotation = float4(0, 0, 0, 1); pose.flags.x = 3; }
            if ((merged || node.links.z != ignoredOwner) && (pose.flags.x & 3) == 3)
            {
                if (TraceMesh(node.links.x, pose, stateOffset, origin, direction, minimum, is2D, anyHit,
                              hit, distance, normal)) return true;
            }
        }
        if (depth == 0) break;
        index = stack[--depth];
    }
#else
    for (uint id = 0; id < PrimitiveCount; id++)
    {
        if (Primitives[id].indices.x == ignoredOwner) continue;
        if (ConsiderPrimitive(id, stateOffset, origin, direction, minimum, is2D, hit, distance, normal) && anyHit)
            return true;
    }
#endif
    return hit != 0xffffffff;
}
bool NearestHit(uint stateOffset, float3 origin, float3 direction, float limit, bool is2D, out uint hit,
                out float distance, out float3 normal)
{
    return TraceScene(stateOffset, origin, direction, RayEpsilon, limit, is2D, false, 0xffffffff, hit,
                      distance, normal);
}
bool Blocked(uint stateOffset, float3 origin, float3 direction, float minimum, float maximum, bool is2D,
             uint ignoredOwner = 0xffffffff)
{
    uint hit;
    float distance;
    float3 normal;
    return TraceScene(stateOffset, origin, direction, minimum, maximum, is2D, true, ignoredOwner, hit,
                      distance, normal);
}
bool Visible(uint stateOffset, float3 start, float3 end, bool is2D, uint ignoredOwner = 0xffffffff)
{
    float distance = length(end - start);
    return distance <= 2 * RayEpsilon || !Blocked(stateOffset, start, (end - start) / distance, RayEpsilon,
                                                  distance - RayEpsilon, is2D, ignoredOwner);
}
