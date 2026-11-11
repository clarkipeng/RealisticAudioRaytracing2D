#include "DiffractionIntegral.hlsl"
// Shared transport: phase-exact direct sound, first-order images and complete
// early routes (CompleteDiffraction.hlsl); the late field is incoherent energy
// (LateEnergy.hlsl). This is not a validated wave/diffraction reference.
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
#include "RayPath.hlsl"
float2 ComplexProduct(float2 a, float2 b) { return float2(a.x*b.x-a.y*b.y,a.x*b.y+a.y*b.x); }
float2 ReflectionPressure(RayMaterial material, float cosine)
{
    if(material.medium.w==0)
        return float2(sqrt(saturate((1-material.surface.x)*(1-material.surface.z)*(1-material.surface.y))),0);
    float2 y=material.medium.yz, denominator=float2(cosine+y.x,y.y);
    float norm=dot(denominator,denominator);
    return norm>0?ComplexProduct(float2(cosine-y.x,-y.y),float2(denominator.x,-denominator.y))/norm:float2(1,0);
}
RayPath ApplyMaterial(RayPath path, float2 coefficient)
{
    float2 value=ComplexProduct(float2(path.pressureGain,path.pressureImag),coefficient);
    path.pressureGain=value.x;path.pressureImag=value.y;return path;
}
StructuredBuffer<RayMaterial> RayMaterials;
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
StructuredBuffer<uint> RayPairEnabled;
StructuredBuffer<uint2> RayPairRanges,RayDirectSites;
uint RayPathsPerFrame,RayDirectSiteCount,PathSlotsPerRay;
StructuredBuffer<uint4> EmissionRanges; // sample offset/count, stable seed, plane flag
StructuredBuffer<float4> EmissionPoints; // local legacy site or frozen world burst position, pressure weight
StructuredBuffer<uint4> EmissionClocks; // phase/clip offset, birth low/high, reserved
uint EmissionFrameStride;
uint RaysPerBin, RaySeed, RayTracePair;
uint FrequencyBins, RaysPerPair, MaxBounces, EdgeCount, VolumeCount, UseAirAbsorption;
uint MirrorCandidateCount, MirrorSiteCapacity, UseVisibleMirrors;
float FrequencyStep, DiffractionFactor, MaximumPathDelay;
// Fixed slots per captured ray avoid atomics and never truncate a path. All stores
// and selection checks compile out of the normal transport variant.
static const uint DebugTransmission = 1, DebugReflection = 2, DebugScatter = 3, DebugDiffraction = 4,
                  DebugListener = 5, DebugEscape = 6, DebugRejected = 7;
#if defined(ACOUSTIC_DEBUG)
struct RaySegment
{
    float4 start, end;
    uint kind;
    int material;
    uint path;
    float pressure;
};
RWStructuredBuffer<RaySegment> DebugSegments;
uint DebugPair, DebugBin;
#endif
bool DebugSelected(uint query, uint bin)
{
#if defined(ACOUSTIC_DEBUG)
    return query == DebugPair && bin == DebugBin;
#else
    return true;
#endif
}
// Capture slots: direct crossings, then baseline (2 per ray) or visible (2 per
// candidate) mirrors, then complete routes (3 per sample).
uint DebugMirrorBase() { return PrimitiveCount + 1; }
uint DebugCompleteBase() { return DebugMirrorBase() + 2 * (UseVisibleMirrors != 0 ? MirrorCandidateCount : RaysPerBin); }
void DebugClearDirect()
{
#if defined(ACOUSTIC_DEBUG)
    // Retarded-time iteration may cross different boundaries. Keep only the final proposal.
    for (uint i = 0; i <= PrimitiveCount; i++)
        DebugSegments[i] = (RaySegment)0;
#endif
}
void DebugWrite(uint slot, uint path, uint kind, int material, float3 a, float3 b, float t0, float t1,
                float pressure)
{
#if defined(ACOUSTIC_DEBUG)
    RaySegment segment;
    segment.start = float4(a, t0);
    segment.end = float4(b, t1);
    segment.kind = kind;
    segment.material = material;
    segment.path = path;
    segment.pressure = pressure;
    DebugSegments[slot] = segment;
#endif
}
void DebugArrival(uint slot, uint path, float3 a, float3 b, float time, RayPath arrival)
{
    float pressure=arrival.pressureImag==0?arrival.pressureGain:length(float2(arrival.pressureGain,arrival.pressureImag));
    if (pressure != 0)
        DebugWrite(slot, path, DebugListener, -1, a, b, time, arrival.delaySeconds, pressure);
}
uint HashRay(uint x)
{
    x ^= x >> 16;
    x *= 0x7feb352d;
    x ^= x >> 15;
    x *= 0x846ca68b;
    x ^= x >> 16;
    return x;
}
float3 EmissionPosition(uint stateOffset,uint sourceIndex,uint sampleIndex)
{
    uint source=Sources[sourceIndex];uint4 plane=EmissionRanges[(stateOffset/ObjectCount)*EmissionFrameStride+sourceIndex];
    float3 sitePosition=EmissionPoints[plane.x+sampleIndex].xyz;
    return plane.w==2?sitePosition:States[stateOffset+source].position.xyz+Rotate(States[stateOffset+source].rotation,sitePosition);
}
RayPath ApplyEmission(RayPath path,uint emissionRow,uint sampleIndex,uint raysForEmission)
{
    uint4 plane=EmissionRanges[emissionRow];
    if(plane.w==0)return path;
    float weight=EmissionPoints[plane.x+sampleIndex].w;
    if(raysForEmission!=0)weight*=plane.w==2?(float)RaysPerBin/raysForEmission:plane.y;
    path.pressureGain*=weight;path.pressureImag*=weight;
    if(plane.w==2)
    {
        uint4 clock=EmissionClocks[plane.x+sampleIndex];path.emissionPhase=clock.x;path.birthLow=clock.y;path.birthHigh=clock.z;return path;
    }
    path.emissionPhase=HashRay(plane.z ^ (sampleIndex*0x85ebca6bu+0x9e3779b9u))>>16;
    path.identity=HashRay(path.identity ^ HashRay(sampleIndex+plane.z+0x51ed270bu));
    return path;
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
float MediumIor(uint medium, uint frequencyBin)
{
    return medium == AirMedium ? 1 : max(.001, RayMaterials[medium * FrequencyBins + frequencyBin].surface.w);
}
float MediumLoss(uint medium, uint frequencyBin, float frequencyHz)
{
    return medium == AirMedium ? AirCoefficient(frequencyHz)
                               : RayMaterials[medium * FrequencyBins + frequencyBin].medium.x;
}
void Travel(float distance, uint medium, uint frequencyBin, float frequencyHz, inout float time,
            inout float gain)
{
    time += distance * MediumIor(medium, frequencyBin) / SpeedOfSound;
    gain *= exp(-MediumLoss(medium, frequencyBin, frequencyHz) * distance);
}
RayPath Arrival(float3 origin, float3 end, float delay, float gain, uint frequencyBin,
                float frequencyRatio = 1, uint identity = 0)
{
    RayPath value = (RayPath)0;
    if (delay < 0 || delay > MaximumPathDelay || gain == 0)
        return value;
    float3 direction = origin - end;
    float distance = length(direction);
    direction = distance > 0 ? direction / distance : float3(0, 0, 0);
    value.direction = float4(direction, 1 - frequencyRatio);
    value.delaySeconds = delay;
    value.pressureGain = gain;
    value.frequencyBin = frequencyBin;
    value.identity = identity;
    return value;
}
// Ratio f_received / f_emitted for two legs meeting at a moving interaction.
float InteractionFrequencyRatio(float speed, float3 incoming, float3 outgoing, float3 sourceVelocity,
                                float3 listenerVelocity, float3 interactionVelocity)
{
    return (speed - dot(listenerVelocity, outgoing)) / (speed - dot(sourceVelocity, incoming)) *
           (speed - dot(interactionVelocity, incoming)) / (speed - dot(interactionVelocity, outgoing));
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
float4 EdgeAngles(float source, float receiver)
{
    return RayPi + float4(source + receiver, source - receiver, -source + receiver, -source - receiver);
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
float EdgeOpacity(uint id, uint frequencyBin)
{
    Edge edge = Edges[id];
    float4 m = RayMaterials[Primitives[edge.ids.y].indices.y * FrequencyBins + frequencyBin].surface;
    return 1 - sqrt(saturate((1 - m.x) * m.z));
}
float BoundaryWeight(uint hit, uint stateOffset, float3 position, bool is2D)
{
    if (DiffractionFactor <= 0)
        return 1;
    uint2 range = EdgeRanges[hit];
    for (uint i = 0; i < range.y; i++)
    {
        float3 a, axis, face, normal;
        float extent, angle;
        EdgeFrame(EdgeIndices[range.x + i], stateOffset, is2D, a, axis, face, normal, extent, angle);
        float z = dot(position - a, axis);
        if (is2D || (z >= 0 && z <= extent))
            if (length(position - a - z * axis) <= RayEpsilon)
                return .5;
    }
    return 1;
}
// Stable truncated Cauchy CDF: atan(hi)-atan(lo) loses the finite
// interval when a narrow peak lies far outside an edge.
float4 EdgeHeightSpan(float center, float4 scale, float extent, bool is2D)
{
    if (is2D) return RayPi;
    float4 y = extent * scale, x = scale * scale + center * (center - extent);
    float4 ratio = y / abs(x), angle = atan2(y, abs(x));
    // Preserve relative accuracy in narrow tail intervals.
    angle = ratio < .01 ? ratio * (1 - ratio * ratio / 3) : angle;
    return x < 0 ? RayPi - angle : angle;
}
float EdgeHeight(float center, float scale, float span, float extent, float u, bool is2D)
{
    if (is2D) return center + scale * tan(lerp(-RayPi * .5, RayPi * .5, u));
    // Interpolate endpoint directions using positive weights. This avoids
    // subtracting the peak center and keeps every draw on the finite edge.
    float2 lengths = float2(length(float2(center, scale)), length(float2(extent - center, scale)));
    float2 weights = lengths * sin(float2(u, 1 - u) * span);
    return extent * (weights.x / (weights.x + weights.y));
}
// Retarded source position; interfaces retain the scene pose of this solve.
// Iteration follows piecewise material travel time, not distance / air speed.
void EvaluateDirect(uint index, bool is2D)
{
    if (index >= FrameCount * RayDirectSiteCount * FrequencyBins)return;
    uint frequencyBin=index%FrequencyBins,site=(index/FrequencyBins)%RayDirectSiteCount,frame=index/(FrequencyBins*RayDirectSiteCount);
    uint2 job=RayDirectSites[site];uint pair=job.x,sampleIndex=job.y,query=frame*PairCount+pair;
    uint slot=frame*RayPathsPerFrame+RayPairRanges[pair].x+sampleIndex*FrequencyBins+frequencyBin,sourceIndex=pair/ListenerCount;
    uint emissionRow=frame*EmissionFrameStride+sourceIndex;
    RayPaths[slot] = (RayPath)0;
    if (sampleIndex>=EmissionRanges[emissionRow].y || EmissionPoints[EmissionRanges[emissionRow].x+sampleIndex].w==0 ||
        RayPairEnabled[query] == 0 || !DebugSelected(query, frequencyBin))
        return;
    uint stateOffset = frame * ObjectCount, source = Sources[pair / ListenerCount],
         listener = Listeners[pair % ListenerCount];
    float3 sourceNow = EmissionPosition(stateOffset,sourceIndex,sampleIndex),
           b = States[stateOffset + listener].position.xyz;
    float3 sourceVelocity = EmissionRanges[emissionRow].w==2?float3(0,0,0):PointVelocity(stateOffset + source,sourceNow),
           listenerVelocity = Velocity(stateOffset + listener);
    if (is2D)
    {
        sourceNow.z = 0;
        b.z = 0;
        sourceVelocity.z = 0;
        listenerVelocity.z = 0;
    }
    float frequencyHz = max(1, frequencyBin * FrequencyStep),
          estimate =
              RetardedDelay(b - sourceNow, sourceVelocity,
                            SpeedOfSound / MediumIor(MediumAt(stateOffset, sourceNow, is2D), frequencyBin));
    float time = 0, gain = 0, frequencyRatio = 1, lower = 0, upper = MaximumPathDelay;
    float3 a = sourceNow;
    uint identity = 1;
    bool converged = false;
    estimate = min(estimate, upper);
    // Alternate fixed-edgePoint proposals with bisection. A bracket prevents slow
    // material transitions or float-level cycles from defeating convergence.
    for (uint iteration = 0; iteration < 64; iteration++)
    {
        DebugClearDirect();
        uint debugSlot = 0;
        a = sourceNow - sourceVelocity * estimate;
        float remaining = length(b - a);
        float3 direction = remaining > 0 ? (b - a) / remaining : float3(0, 0, 0), position = a;
        gain = ReferenceDistance / max(MinimumDistance, remaining);
        time = 0;
        identity = 1;
        uint medium = MediumAt(stateOffset, a, is2D);
        float speed = SpeedOfSound / MediumIor(medium, frequencyBin);
        float totalDistance = remaining, boundaryRate = 0;
        float3 sourceGradient = -direction / speed, listenerGradient = direction / speed;
        for (uint crossing = 0; crossing < PrimitiveCount && remaining > 2 * RayEpsilon; crossing++)
        {
            uint hit;
            float distance;
            float3 normal;
            if (!NearestHit(stateOffset, position, direction, remaining - RayEpsilon, is2D, hit, distance,
                            normal))
                break;
            float segmentTime = time, segmentGain = gain;
            Travel(distance, medium, frequencyBin, frequencyHz, time, gain);
            RayMaterial material = RayMaterials[Primitives[hit].indices.y * FrequencyBins + frequencyBin];
            float transmitted = sqrt(saturate((1 - material.surface.x) * material.surface.z));
            gain *=
                lerp(1, transmitted, BoundaryWeight(hit, stateOffset, position + direction * distance, is2D));
            if (segmentGain != 0)
                DebugWrite(debugSlot++, 0, gain == 0 ? DebugRejected : DebugTransmission,
                           Primitives[hit].indices.y, position, position + direction * distance, segmentTime,
                           time, gain);
            position += direction * distance;
            remaining -= distance;
            float depth;
            uint volume = VolumeMedium(Primitives[hit].indices.x);
            if (gain != 0 && PartitionExit(stateOffset, hit, position, direction, is2D, volume != AirMedium && MediumIor(volume, frequencyBin) != 1, depth) && depth < remaining)
            {
                // The far face of the same partition: its transmission was the whole assembly's.
                Travel(depth, medium, frequencyBin, frequencyHz, time, gain);
                position += direction * depth;
                remaining -= depth;
            }
            uint next = MediumAt(stateOffset, position + direction * (2 * RayEpsilon), is2D);
            float nextSpeed = SpeedOfSound / MediumIor(next, frequencyBin);
            float3 velocity = PointVelocity(stateOffset + Primitives[hit].indices.x, position);
            // Differentiate this straight, piecewise-speed route. Its crossings
            // slide along even stationary interfaces as source/listener move.
            if (nextSpeed != speed)
            {
                float fraction = (totalDistance - remaining) / totalDistance;
                float slowness = 1 / nextSpeed - 1 / speed;
                float3 crossing = normal / dot(normal, direction), bend = crossing - direction;
                sourceGradient += (1 - fraction) * bend * slowness;
                listenerGradient += (direction + fraction * bend) * slowness;
                boundaryRate -= dot(velocity, crossing) * slowness;
            }
            identity = HashRay(identity ^ (hit + 1));
            medium = next;
            speed = nextSpeed;
        }
        float segmentTime = time;
        Travel(remaining, medium, frequencyBin, frequencyHz, time, gain);
        if (gain != 0)
            DebugWrite(debugSlot, 0, time <= MaximumPathDelay ? DebugListener : DebugRejected, -1, position,
                       b, segmentTime, time, gain);
        frequencyRatio = (1 - dot(listenerGradient, listenerVelocity) - boundaryRate) /
                         (1 + dot(sourceGradient, sourceVelocity));
        float tolerance = 4e-7 * max(1, time);
        if (abs(time - estimate) <= tolerance)
        {
            converged = true;
            break;
        }
        if (time > MaximumPathDelay && estimate >= MaximumPathDelay)
            break;
        if (time > estimate)
            lower = estimate;
        else
            upper = estimate;
        if (upper - lower <= tolerance)
        {
            time = estimate;
            converged = true;
            break;
        }
        estimate = (iteration % 2 == 0 && time > lower && time < upper) ? time : (lower + upper) * .5;
    }
    RayPath direct =
        Arrival(a, b, time, gain, frequencyBin, converged ? frequencyRatio : asfloat(0x7fc00000), identity);
    RayPaths[slot] = ApplyEmission(direct,emissionRow,sampleIndex,0);
}
// Complete first-order image route. The caller owns candidate selection and its
// probability correction; all visibility, material and retarded-time work is shared.
RayPath FirstReflection(uint stateOffset, uint source, uint listener, uint frequencyBin, uint sampleIndex,
                        bool is2D, uint debugBase, float3 sourceNow, float3 sourceVelocity,
                        uint primitive, float samplingWeight, bool capture)
{
    RayPath result = (RayPath)0;
    if (PrimitiveCount == 0)
        return result;
    uint id = primitive;
    Primitive p = Primitives[id];
    State pose = States[stateOffset + p.indices.x];
    if ((pose.flags.x & 3) != 3)
        return result;
    float3 end = States[stateOffset + listener].position.xyz;
    float3 listenerVelocity = Velocity(stateOffset + listener);
    if (is2D)
    {
        sourceNow.z = 0;
        end.z = 0;
        sourceVelocity.z = 0;
        listenerVelocity.z = 0;
    }
    uint medium = MediumAt(stateOffset, sourceNow, is2D);
    float speed = SpeedOfSound / MediumIor(medium, frequencyBin);
    float total = length(end - sourceNow) / speed, last = total * .5, distance = 0;
    float3 start = sourceNow, reflectionPoint = 0, image = 0, direction = 0, unused;
    bool converged = false;
    for (uint iteration = 0; iteration < 32; iteration++)
    {
        float3 a = PastPoint(pose, p.a.xyz, last), b = PastPoint(pose, p.b.xyz, last);
        float3 normal = is2D ? normalize(float3(-(b - a).y, (b - a).x, 0))
                             : normalize(cross(b - a, PastPoint(pose, p.c.xyz, last) - a));
        start = sourceNow - sourceVelocity * total;
        if (dot(start - a, normal) * dot(end - a, normal) <= 0)
            return result;
        image = start - 2 * normal * dot(start - a, normal);
        direction = normalize(end - image);
        distance = PrimitiveDistanceAt(id, stateOffset, image, direction, is2D, last, unused);
        if (distance >= length(end - image) - RayEpsilon)
            return result;
        reflectionPoint = image + direction * distance;
        float nextLast = length(end - reflectionPoint) / speed,
              nextTotal = length(start - reflectionPoint) / speed + nextLast;
        if (abs(nextTotal - total) + abs(nextLast - last) < 4e-7 * max(1, nextTotal))
        {
            total = nextTotal;
            last = nextLast;
            converged = true;
            break;
        }
        total = nextTotal;
        last = nextLast;
    }
    for (uint other = 0; other < id; other++)
    {
        if (Primitives[other].indices.x != p.indices.x)
            continue;
        float t = PrimitiveDistanceAt(other, stateOffset, image, direction, is2D, last, unused);
        if (abs(t - distance) < RayEpsilon)
            return result;
    }
    if (!Visible(stateOffset, start, reflectionPoint, is2D, p.indices.x) ||
        !Visible(stateOffset, reflectionPoint, end, is2D, p.indices.x))
        return result;
    float firstLength = length(reflectionPoint - start), lastLength = length(end - reflectionPoint);
    for (uint sibling = 0; sibling < PrimitiveCount; sibling++)
    {
        if (sibling == id || Primitives[sibling].indices.x != p.indices.x)
            continue;
        if (PrimitiveDistanceAt(sibling, stateOffset, start, normalize(reflectionPoint - start), is2D, last,
                                unused) < firstLength - RayEpsilon ||
            PrimitiveDistanceAt(sibling, stateOffset, reflectionPoint, normalize(end - reflectionPoint), is2D,
                                last, unused) < lastLength - RayEpsilon)
            return result;
    }
    float3 incoming = normalize(reflectionPoint - start), outgoing = normalize(end - reflectionPoint),
           velocity = PointVelocity(stateOffset + p.indices.x, reflectionPoint, last);
    if (is2D)
        velocity.z = 0;
    float frequencyRatio =
        InteractionFrequencyRatio(speed, incoming, outgoing, sourceVelocity, listenerVelocity, velocity);
    RayMaterial material = RayMaterials[p.indices.y * FrequencyBins + frequencyBin];
    float2 reflected=ReflectionPressure(material,abs(dot(incoming,normalize(image-start))));
    float gain = 1;
    gain *= BoundaryWeight(id, stateOffset, reflectionPoint, is2D);
    float time = 0, lengthTo = total * speed;
    Travel(lengthTo, medium, frequencyBin, max(1, frequencyBin * FrequencyStep), time, gain);
    result = ApplyMaterial(Arrival(
        reflectionPoint, end, time,
        gain * samplingWeight * ReferenceDistance / max(MinimumDistance, lengthTo),
        frequencyBin, converged ? frequencyRatio : asfloat(0x7fc00000), HashRay(0x40000000u ^ id)),reflected);
    if (capture && (result.pressureGain != 0 || result.pressureImag != 0))
    {
        DebugWrite(debugBase, 1 + 2 * sampleIndex, DebugReflection, p.indices.y, start, reflectionPoint, 0,
                   firstLength / speed, result.pressureGain);
        DebugArrival(debugBase + 1, 1 + 2 * sampleIndex, reflectionPoint, end, firstLength / speed, result);
    }
    return result;
}

// Complete mirror routes retain delay, direction, material phase and Doppler.
// Mirror storage is bounded by the candidate budget and preserves each delay.
void PrepareMirrors(uint index, bool is2D)
{
    uint capacity=MirrorCandidateCount*FrequencyBins*MirrorSiteCapacity;
    if(index>=FrameCount*capacity)return;
    uint candidate=index%MirrorCandidateCount, row=index/MirrorCandidateCount;
    uint bin=row%FrequencyBins, site=(row/FrequencyBins)%MirrorSiteCapacity;
    uint frame=row/(FrequencyBins*MirrorSiteCapacity), pair=RayTracePair;
    uint slot=frame*RayPathsPerFrame+RayPairRanges[pair].x+MirrorSiteCapacity*FrequencyBins+index%capacity;
    RayPaths[slot]=(RayPath)0;
    uint emissionRow=frame*EmissionFrameStride+pair/ListenerCount;
    uint4 plane=EmissionRanges[emissionRow];
    if(site>=plane.y || RayPairEnabled[frame*PairCount+pair]==0 || EmissionPoints[plane.x+site].w==0 || !DebugSelected(frame*PairCount+pair,bin))return;
    uint stateOffset=frame*ObjectCount, source=Sources[pair/ListenerCount], listener=Listeners[pair%ListenerCount];
    uint birthSeed=plane.w==2?HashRay(EmissionClocks[plane.x+site].y)^HashRay(EmissionClocks[plane.x+site].z):0;
    uint primitive=(candidate+HashRay(source*0x85ebca6b+listener+RaySeed+birthSeed)%PrimitiveCount)%PrimitiveCount;
    float3 emission=EmissionPosition(stateOffset,pair/ListenerCount,site);
    float3 velocity=plane.w==2?float3(0,0,0):PointVelocity(stateOffset+source,emission);
    uint debug=DebugMirrorBase()+2*candidate;
    RayPath arrival=FirstReflection(stateOffset,source,listener,bin,candidate,is2D,debug,emission,velocity,
        primitive,(float)PrimitiveCount/MirrorCandidateCount,true);
    RayPaths[slot]=ApplyEmission(arrival,emissionRow,site,0);
}
#include "CompleteDiffraction.hlsl"
// PrimitiveBaseline comparison estimator: each ray samples one primitive's first
// image. First Images (the default) enumerate candidates in PrepareMirrors instead.
void BaselineMirrors(uint index, bool is2D)
{
    if (index >= FrameCount * RaysPerPair)
        return;
    uint ray = index % RaysPerPair, frame = index / RaysPerPair, pair = RayTracePair,
         query = frame * PairCount + pair;
    uint stateOffset = frame * ObjectCount, source = Sources[pair / ListenerCount],
         listener = Listeners[pair % ListenerCount];
    uint slot = frame*RayPathsPerFrame + RayPairRanges[pair].x + MirrorSiteCapacity*FrequencyBins + ray*PathSlotsPerRay;
    RayPaths[slot] = (RayPath)0;
    if (RayPairEnabled[query] == 0 || !DebugSelected(query, ray / RaysPerBin))
        return;
    uint frequencyBin = ray / RaysPerBin;
    uint sourceIndex=pair/ListenerCount,emissionRow=frame*EmissionFrameStride+sourceIndex;
    uint4 plane=EmissionRanges[emissionRow];
    if(plane.y==0)return;
    uint sampleIndex=plane.w==0?0:plane.w==2?(ray%RaysPerBin)%plane.y:((ray%RaysPerBin)+HashRay(RaySeed ^ source))%plane.y;
    uint localRay=plane.w==2?(ray%RaysPerBin)/plane.y:ray%RaysPerBin;
    uint raysForEmission=plane.w==2?1+(RaysPerBin-1-sampleIndex)/plane.y:RaysPerBin;
    uint birthSeed=0;
    if(plane.w==2){uint4 clock=EmissionClocks[plane.x+sampleIndex];birthSeed=HashRay(clock.y)^HashRay(clock.z);}
    if(EmissionPoints[plane.x+sampleIndex].w==0)return;
    float3 emission=EmissionPosition(stateOffset,sourceIndex,sampleIndex);
    float3 emissionVelocity=plane.w==2?float3(0,0,0):PointVelocity(stateOffset+source,emission);
    uint primitive=PrimitiveCount==0?0:(localRay+birthSeed+HashRay(source*0x85ebca6b+listener+RaySeed))%PrimitiveCount;
    RayPath mirror=FirstReflection(stateOffset,source,listener,frequencyBin,localRay+birthSeed,is2D,DebugMirrorBase()+2*(ray%RaysPerBin),
        emission,emissionVelocity,primitive,(float)PrimitiveCount/RaysPerBin,true);
    RayPaths[slot]=ApplyEmission(mirror,emissionRow,sampleIndex,raysForEmission);
}
#include "LateEnergy.hlsl"
