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
StructuredBuffer<uint> Sources, Listeners;
struct DirectPath
{
    float4 direction;
    float2 response;
    float2 padding;
};
RWStructuredBuffer<DirectPath> Responses;
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
float RetardedDelay(float3 displacement, float3 velocity, float speed)
{
    float a = speed * speed - dot(velocity, velocity), b = dot(displacement, velocity);
    return a > 0 ? (b + sqrt(max(0, b * b + a * dot(displacement, displacement)))) / a : asfloat(0x7fc00000);
}
static const float RayEpsilon = 0.0001, RayPi = 3.141592653589793;
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
        if (abs(det) < 1e-12)
            return 1e30;
        float t = (q.x * e.y - q.y * e.x) / det, u = (q.x * d.y - q.y * d.x) / det;
        normal = normalize(float3(-e.y, e.x, 0));
        return t > minimum && u >= 0 && u <= 1 ? t : 1e30;
    }
    float3 c = PastPoint(state, p.c.xyz, lag), e1 = b - a, e2 = c - a, h = cross(d, e2);
    float det = dot(e1, h);
    if (abs(det) < 1e-12)
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
    float candidate = IntersectPrimitive(id, stateOffset, origin, direction, is2D, 0, minimum, candidateNormal);
    if (candidate < distance || (hit != 0xffffffff && candidate == distance && id < hit))
    {
        hit = id; distance = candidate; normal = candidateNormal; return true;
    }
    return false;
}
#ifdef USE_BVH
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
bool TraceMesh(uint root, State pose, uint stateOffset, float3 origin, float3 direction, float minimum,
               bool is2D, bool anyHit, inout uint hit, inout float distance, inout float3 normal)
{
    uint stack[32], depth = 0, index = root;
    while (true)
    {
        BvhNode node = BvhNodes[index];
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
#endif
// Shared any-hit / closest-hit traversal. Each median-split level needs at most 32 stack entries.
bool TraceScene(uint stateOffset, float3 origin, float3 direction, float minimum, float maximum, bool is2D,
                bool anyHit, uint ignoredOwner, out uint hit, out float distance, out float3 normal)
{
    hit = 0xffffffff; distance = maximum; normal = 0;
#ifdef USE_BVH
    uint stack[32], depth = 0, index = 0;
    while (index < BvhNodeCount)
    {
        BvhNode node = BvhNodes[index];
        if (BoundsHit(node, origin, direction * distance, is2D))
        {
            if (node.links.w == 0) { stack[depth++] = node.links.y; index = node.links.x; continue; }
            State pose = States[stateOffset + node.links.z];
            if (node.links.z != ignoredOwner && (pose.flags.x & 3) == 3)
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
