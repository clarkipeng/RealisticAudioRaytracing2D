// Shared acoustic transport helpers of the path method: materials and media, edges and wedge angles, random
// numbers, air absorption and the exact-arrival row (RayPath). The hybrid renderer they came from was removed
// (Git history). This is not a validated wave/diffraction reference.
// Angle of (x, y) in [0, pi] for y >= 0, without the backend's atan2 (from the wave-validated complete-route shader).
static const float Pi = 3.14159265358979323846;
float Span(float y, float x)
{
    // Two half-angle reductions put atan's argument in [0,tan(pi/8)].
    // The 19th-order series has <2e-9 rad truncation error for the full angle;
    // it avoids backend atan2 approximations amplified by long sampled paths.
    float t = y/(length(float2(x,y))+abs(x));
    t /= sqrt(1+t*t)+1;
    float squared=t*t, polynomial=-1.0/19;
    [unroll] for (int n=8;n>=1;n--)
    polynomial=(n%2==0?1.0:-1.0)/(2*n+1)+squared*polynomial;
    float angle=4*t*(1+squared*polynomial);
    return x<0?Pi-angle:angle;
}
struct RayMaterial
{
    // surface: absorption, scattering, transmission, relative slowness (air speed / material speed).
    // medium: pressure loss/metre, normalized admittance real/imag, boundary mode.
    float4 surface, medium;
};
struct Edge
{
    float4 a, b, face, normal;
    uint4 ids;
};
struct RayPath
{
    // direction.xyz: listener-to-arrival; direction.w: 1 - received/emitted frequency ratio.
    float4 direction;
    float delaySeconds, pressureGain;
    uint frequencyBin, identity;
    float pressureImag;
    uint emissionPhase, birthLow, birthHigh;
};
float2 ComplexProduct(float2 a, float2 b) { return float2(a.x*b.x-a.y*b.y,a.x*b.y+a.y*b.x); }
float2 ReflectionPressure(RayMaterial material, float cosine)
{
    if(material.medium.w==0)
        return float2(sqrt(saturate((1-material.surface.x)*(1-material.surface.z)*(1-material.surface.y))),0);
    float2 y=material.medium.yz, denominator=float2(cosine+y.x,y.y);
    float norm=dot(denominator,denominator);
    return norm>0?ComplexProduct(float2(cosine-y.x,-y.y),float2(denominator.x,-denominator.y))/norm:float2(1,0);
}
StructuredBuffer<Edge> Edges;
StructuredBuffer<uint2> EdgeRanges;
StructuredBuffer<uint> EdgeIndices;
struct Volume
{
    float4 minimum, maximum;
    uint4 ids;
};
StructuredBuffer<Volume> Volumes;
RWStructuredBuffer<RayPath> RayPaths;
uint RaysPerBin, RaySeed, RayTracePair;
uint FrequencyBins, RaysPerPair, MaxBounces, EdgeCount, VolumeCount, UseAirAbsorption;
float FrequencyStep, DiffractionFactor, MaximumPathDelay;
uint HashRay(uint x)
{
    x ^= x >> 16;
    x *= 0x7feb352d;
    x ^= x >> 15;
    x *= 0x846ca68b;
    x ^= x >> 16;
    return x;
}
float RandomRay(inout uint x)
{
    x = HashRay(x + 0x9e3779b9);
    return (x >> 8) * (1.0 / 16777216.0);
}
float3 UnitDirection(inout uint rng, bool is2D)
{
    float angle = 2 * RayPi * RandomRay(rng);
    if (is2D)
        return float3(cos(angle), sin(angle), 0);
    float z = 2 * RandomRay(rng) - 1, r = sqrt(max(0, 1 - z * z));
    return float3(r * cos(angle), r * sin(angle), z);
}
// Air absorption, ISO 9613-1 (20 C, 50% relative humidity, 101.325 kPa) at the octave band
// centres 125 Hz ... 16 kHz, in dB per metre (sound pressure level; the energy factor over d is
// 10^(-losses d / 10)). The previous table held the same values per foot (3.28x too small), so
// the high bands of the late field decayed at the air-limited rate of the wrong table.
float AirCoefficient(float frequencyHz)
{
    if (UseAirAbsorption == 0)
        return 0;
    const float losses[8] = {0.00044, 0.00131, 0.00273, 0.00466, 0.00989, 0.02967, 0.10529, 0.36454};
    float at = clamp(log2(max(125, frequencyHz) / 125), 0, 7);
    uint a = (uint)at;
    return lerp(losses[a], losses[min(a + 1, 7)], frac(at)) / 8.685889638;
}
static const uint AirMedium = 0xffffffff;
// Partitions. A surface material's transmission is the whole assembly's (a wall, a door leaf, a
// window pane): the scene models such an object with two faces (two sheets, or a thin closed
// solid without refraction), and the assembly's sound reduction applies once per crossing. A
// transmitted ray or straight connection therefore continues undeviated and leaves the object
// through its far face when that lies within PartitionDepth (3 m: the Church's buttressed walls
// are 2.4 m). Media are not partitions: a closed object whose material refracts (acoustic index
// not 1, e.g. water or a material slab whose travel time matters) is entered with refraction and
// crossed through its own interior, as before; so is anything thicker than PartitionDepth.
static const float PartitionDepth = 3.0;
// The medium of a closed object (Volumes are sorted by owner), or AirMedium for an open one.
uint VolumeMedium(uint owner)
{
    uint lo = 0, hi = VolumeCount;
    while (lo < hi) { uint mid = lo + (hi - lo) / 2; if (Volumes[mid].ids.z < owner) lo = mid + 1; else hi = mid; }
    return lo < VolumeCount && Volumes[lo].ids.z == owner ? Volumes[lo].ids.w : AirMedium;
}
// medium: the caller's test that the hit object is a refracting medium (see above).
bool PartitionExit(uint stateOffset, uint hit, float3 position, float3 direction, bool is2D, bool medium, out float depth)
{
    uint owner = Primitives[hit].indices.x;
    float travelled = 2 * RayEpsilon;
    depth = 0;
    if (medium) return false;
    [loop] for (uint i = 0; i < 4; i++)
    {
        uint next; float distance; float3 normal;
        if (!NearestHit(stateOffset, position + direction * travelled, direction, PartitionDepth - travelled, is2D, next, distance, normal))
            return false;
        if (Primitives[next].indices.x == owner) { depth = travelled + distance; return true; }
        travelled += distance + 2 * RayEpsilon;
        if (travelled >= PartitionDepth) return false;
    }
    return false;
}
// Containment uses the nearest oriented boundary of each closed object. Choosing
// the smallest containing volume restores the parent medium in nested solids, and
// also initializes sources/listeners that start inside one. No fixed medium stack.
// Volumes are sorted by owner. Reuse the scene's TLAS for candidate owners,
// then its BLAS for the exact oriented-boundary test; no second spatial index.
void ConsiderMedium(uint v, uint root, uint stateOffset, float3 location, bool is2D,
                    inout uint selected, inout float smallest)
{
    Volume volume = Volumes[v];
    State pose = States[stateOffset + volume.ids.z];
    if ((pose.flags.x & 3) != 3 || volume.maximum.w > smallest ||
        (volume.maximum.w == smallest && v >= selected)) return;
    float4 inverse = pose.rotation; inverse.xyz = -inverse.xyz;
    float3 local = Rotate(inverse, location - pose.position.xyz);
    if (any(local.xy < volume.minimum.xy) || any(local.xy > volume.maximum.xy) ||
        (!is2D && (local.z < volume.minimum.z || local.z > volume.maximum.z))) return;
    float3 probe = is2D ? normalize(float3(.819, .573, 0)) : normalize(float3(.819, .573, .371));
    float nearest = 1e30; float3 normal = 0;
#ifdef USE_BVH
    uint hit = 0xffffffff;
    PROFILE_QUERY();
    TraceMesh(root, pose, stateOffset, location, probe, RayEpsilon, is2D, false, hit, nearest, normal);
#else
    for (uint i = 0; i < volume.ids.y; i++)
    {
        float3 n; float distance = PrimitiveDistance(volume.ids.x + i, stateOffset, location, probe, is2D, n);
        if (distance < nearest) { nearest = distance; normal = n; }
    }
#endif
    if (nearest < 1e30 && dot(normal, probe) * volume.minimum.w > 0)
    { smallest = volume.maximum.w; selected = v; }
}
uint MediumAt(uint stateOffset, float3 location, bool is2D)
{
    if (VolumeCount == 0) return AirMedium;
    uint selected = AirMedium; float smallest = 1e30;
#ifdef USE_BVH
    uint stack[32], depth = 0, index = 0;
    while (index < BvhNodeCount)
    {
        BvhNode node = BvhNodes[index];
        PROFILE_NODE();
        if (all(location.xy >= node.boundsMin.xy) && all(location.xy <= node.boundsMax.xy) &&
            (is2D || (location.z >= node.boundsMin.z && location.z <= node.boundsMax.z)))
        {
            if (node.links.w == 0) { stack[depth++] = node.links.y; index = node.links.x; continue; }
            uint lo = 0, hi = VolumeCount;
            while (lo < hi) { uint mid = lo + (hi-lo)/2; if (Volumes[mid].ids.z < node.links.z) lo = mid+1; else hi = mid; }
            if (lo < VolumeCount) if (Volumes[lo].ids.z == node.links.z)
                ConsiderMedium(lo, node.links.x, stateOffset, location, is2D, selected, smallest);
        }
        if (depth == 0) break;
        index = stack[--depth];
    }
#else
    for (uint v = 0; v < VolumeCount; v++) ConsiderMedium(v, 0, stateOffset, location, is2D, selected, smallest);
#endif
    return selected == AirMedium ? AirMedium : Volumes[selected].ids.w;
}
// Rigid-wedge secondary edge sources (Svensson/Calamia/Nakanishi 2009,
// equations 1-6). Each signed sample and its travel delay represents one
// quadrature contribution; the existing DSP retains its phase without a new ABI.
void EdgeFrame(uint id, uint stateOffset, bool is2D, out float3 a, out float3 axis, out float3 face,
               out float3 normal, out float extent, out float angle)
{
    Edge edge = Edges[id];
    State pose = States[stateOffset + edge.ids.x];
    a = pose.position.xyz + Rotate(pose.rotation, edge.a.xyz);
    float3 span = Rotate(pose.rotation, edge.b.xyz - edge.a.xyz);
    extent = length(span);
    axis = is2D ? float3(0, 0, 1) : span / extent;
    face = Rotate(pose.rotation, edge.face.xyz);
    normal = Rotate(pose.rotation, edge.normal.xyz);
    angle = edge.face.w;
}
// EDtoolbox EDwedge2nd/EDwedgeN: an edge-to-edge leg on a shared
// solid face has multiplicity 1/2. Thin-sheet edge pairs retain both sides.
float EdgeConnectionWeight(uint first, uint second)
{
    Edge a = Edges[first], b = Edges[second];
    float weight = 1;
    if (a.ids.x == b.ids.x && !(a.ids.y == a.ids.z && b.ids.y == b.ids.z))
    {
        float3 an[2] = {a.normal.xyz, cos(a.face.w) * a.normal.xyz - sin(a.face.w) * a.face.xyz};
        float3 bn[2] = {b.normal.xyz, cos(b.face.w) * b.normal.xyz - sin(b.face.w) * b.face.xyz};
        [unroll] for (uint i = 0; i < 2; i++)
            [unroll] for (uint j = 0; j < 2; j++)
                if (abs(dot(an[i], bn[j])) > .99999 &&
                    abs(dot(an[i], b.a.xyz - a.a.xyz)) <= RayEpsilon)
                    weight = .5;
    }
    return weight;
}

float TunedEdgeAngle(float3 direction, float3 face, float3 normal, float exterior)
{
    float2 radial = float2(dot(direction, face), dot(direction, normal));
    float angle = Span(abs(radial.y),radial.x);
    if(radial.y<0)angle=-angle;
    angle = angle < 0 ? angle + 2 * RayPi : angle;
    // Face-grazing connections use the visibility tolerance. Rigid rotation
    // must not wrap roundoff at zero to 2*pi and classify the ray as solid.
    if (min(angle, 2 * RayPi - angle) * length(radial) <= RayEpsilon)
        angle = 0;
    if (abs(angle - exterior) * length(radial) <= RayEpsilon)
        angle = exterior;
    return angle;
}
float3 CompleteImage(uint primitive,uint state,float3 location,bool planar,out float3 normal)
{
    Primitive p=Primitives[primitive];State pose=States[state+p.indices.x];
    float3 a=pose.position.xyz+Rotate(pose.rotation,p.a.xyz),b=Rotate(pose.rotation,p.b.xyz-p.a.xyz);
    normal=planar?normalize(float3(-b.y,b.x,0)):normalize(cross(b,Rotate(pose.rotation,p.c.xyz-p.a.xyz)));
    return location-2*dot(location-a,normal)*normal;
}
float3 LateLambert(float3 n, inout uint rng, bool is2D)
{
    if (is2D)
    {
        float s = 2 * RandomRay(rng) - 1; // sin(theta) for the planar cos/2 law
        return normalize(n * sqrt(max(0, 1 - s * s)) + float3(-n.y, n.x, 0) * s);
    }
    float u = RandomRay(rng), angle = 2 * RayPi * RandomRay(rng), r = sqrt(u);
    float3 t = abs(n.x) < .9 ? normalize(cross(n, float3(1, 0, 0))) : normalize(cross(n, float3(0, 1, 0)));
    return normalize(t * (r * cos(angle)) + cross(n, t) * (r * sin(angle)) + n * sqrt(max(0, 1 - u)));
}
