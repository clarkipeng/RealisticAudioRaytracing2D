#include "DiffractionIntegral.hlsl"
// Shared geometric preview: surface and signed edge events share a path loop.
// This is not a validated wave/diffraction reference.
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
uint MirrorCandidateCount, MirrorSiteCapacity, UseVisibleMirrors,UseCompleteRoutes;
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
float AirCoefficient(float frequencyHz)
{
    if (UseAirAbsorption == 0)
        return 0;
    const float losses[8] = {0.0001, 0.0003, 0.0007, 0.0015, 0.0035, 0.0100, 0.0320, 0.1100};
    float at = clamp(log2(max(125, frequencyHz) / 125), 0, 7);
    uint a = (uint)at;
    return lerp(losses[a], losses[min(a + 1, 7)], frac(at)) / 8.685889638;
}
static const uint AirMedium = 0xffffffff;
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
float EdgeAngle(float3 direction, float3 face, float3 normal, float exterior)
{
    float2 radial = float2(dot(direction, face), dot(direction, normal));
    float angle = atan2(radial.y, radial.x);
    angle = angle < 0 ? angle + 2 * RayPi : angle;
    // Face-grazing connections use the visibility tolerance. Rigid rotation
    // must not wrap roundoff at zero to 2*pi and classify the ray as solid.
    if (min(angle, 2 * RayPi - angle) * length(radial) <= RayEpsilon)
        angle = 0;
    if (abs(angle - exterior) * length(radial) <= RayEpsilon)
        angle = exterior;
    return angle;
}
float4 EdgeAngles(float source, float receiver)
{
    return RayPi + float4(source + receiver, source - receiver, -source + receiver, -source - receiver);
}
float EdgeKernel(uint id, uint stateOffset, float3 source, float3 edgePoint, float3 receiver, bool is2D)
{
    float3 a, axis, face, normal;
    float extent, angle;
    EdgeFrame(id, stateOffset, is2D, a, axis, face, normal, extent, angle);
    float3 incoming = source - edgePoint, outgoing = receiver - edgePoint;
    float m = length(incoming), l = length(outgoing), zs = dot(incoming, axis), zr = dot(outgoing, axis);
    float rs = length(incoming - zs * axis), rr = length(outgoing - zr * axis);
    if (rs <= RayEpsilon || rr <= RayEpsilon)
        return 0;
    float ts = EdgeAngle(incoming, face, normal, angle), tr = EdgeAngle(outgoing, face, normal, angle);
    if (ts > angle || tr > angle)
        return 0; // Inside the solid wedge.
    float nu = RayPi / angle;
    // Rationalize cosh(eta)-1; subtracting nearly equal products loses the
    // narrow peak next to a shadow boundary in single precision.
    float axial = rs * zr + rr * zs;
    float x = abs(axial) / sqrt(2 * rs * rr * (m * l + rs * rr - zs * zr));
    float sinhHalf = x < .01 ? nu * x * (1 + (nu * nu - 1) * x * x / 6) : sinh(nu * log(x + sqrt(1 + x * x)));
    float4 phi = nu * EdgeAngles(ts, tr), halfSine = sin(phi * .5);
    float beta = 0;
    for (uint i = 0; i < 4; i++)
    {
        float denominator = 2 * (sinhHalf * sinhHalf + halfSine[i] * halfSine[i]);
        // Principal value at the exact zone boundary; geometrical components
        // use half weight there. Avoid subtracting nearly equal cosh/cos values.
        if (abs(halfSine[i]) > 1e-6)
            beta += sin(phi[i]) / denominator;
    }
    return -nu * beta / (4 * RayPi);
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
// A mixture of the current listener-blocking face's edges and all scene edges.
// The global component keeps every edge reachable, including multi-edge routes.
// Sampling probability includes both components even when they select the same edge.
bool SampleEdge(uint stateOffset, float3 source, float3 receiver, bool is2D, inout uint rng, out uint id,
                out float3 edgePoint, out float inversePdf)
{
    id = 0;
    edgePoint = 0;
    inversePdf = 0;
    float distance = length(receiver - source);
    uint hit;
    float t;
    float3 n;
    uint2 local = 0;
    if (distance > RayEpsilon && NearestHit(stateOffset, source, (receiver - source) / distance,
                                            distance - RayEpsilon, is2D, hit, t, n))
        local = EdgeRanges[hit];
    bool localSample = local.y > 0 && RandomRay(rng) < .5;
    id = localSample ? EdgeIndices[local.x + min(local.y - 1, (uint)(RandomRay(rng) * local.y))]
                     : min(EdgeCount - 1, (uint)(RandomRay(rng) * EdgeCount));
    float probability = (local.y > 0 ? .5 : 1) / EdgeCount;
    for (uint i = 0; i < local.y; i++)
        if (EdgeIndices[local.x + i] == id)
            probability += .5 / local.y;
    if ((States[stateOffset + Edges[id].ids.x].flags.x & 3) != 3)
        return false;
    float3 a, axis, face, normal;
    float extent, angle;
    EdgeFrame(id, stateOffset, is2D, a, axis, face, normal, extent, angle);
    float3 s = source - a, r = receiver - a;
    float zs = dot(s, axis), zr = dot(r, axis);
    float rs = length(s - zs * axis), rr = length(r - zr * axis);
    if (rs <= RayEpsilon || rr <= RayEpsilon)
        return false;
    float apex = (zs * rr + zr * rs) / (rs + rr);
    float m = length(s - apex * axis), l = length(r - apex * axis), nu = RayPi / angle;
    float4 phi = nu * EdgeAngles(EdgeAngle(s, face, normal, angle), EdgeAngle(r, face, normal, angle));
    // Cauchy importance sampling resolves narrow zone-boundary contributions.
    // Planar scenes model point sources around infinitely extruded XY geometry.
    float4 scale = max(RayEpsilon, 2 * abs(sin(phi * .5)) * (m * l / (m + l)) / nu);
    float4 span = EdgeHeightSpan(apex, scale, extent, is2D);
    uint component = min(3u, (uint)(RandomRay(rng) * 4));
    float u = clamp(RandomRay(rng), 1e-7, 1 - 1e-7);
    float z = EdgeHeight(apex, scale[component], span[component], extent, u, is2D);
    float4 delta = z - apex;
    float4 density = scale / ((scale * scale + delta * delta) * span);
    float pdf = probability * dot(density, float4(.25, .25, .25, .25));
    edgePoint = a + axis * z;
    inversePdf = 1 / pdf;
    return isfinite(inversePdf) && inversePdf > 0;
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
    uint debug=PrimitiveCount+1+RaysPerBin*(2+2*MaxBounces)+2*candidate;
    RayPath arrival=FirstReflection(stateOffset,source,listener,bin,candidate,is2D,debug,emission,velocity,
        primitive,(float)PrimitiveCount/MirrorCandidateCount,true);
    RayPaths[slot]=ApplyEmission(arrival,emissionRow,site,0);
}
#include "CompleteDiffraction.hlsl"
void TracePaths(uint index, bool is2D)
{
    if (index >= FrameCount * RaysPerPair)
        return;
    uint ray = index % RaysPerPair, frame = index / RaysPerPair, pair = RayTracePair,
         query = frame * PairCount + pair;
    uint stateOffset = frame * ObjectCount, source = Sources[pair / ListenerCount],
         listener = Listeners[pair % ListenerCount];
    // Walks follow the direct/mirror prefix. Complete routes append after the
    // walk, so the end of the pair is no longer its starting-offset invariant.
    uint first = frame*RayPathsPerFrame + RayPairRanges[pair].x +
        MirrorSiteCapacity*FrequencyBins*(1+(UseVisibleMirrors!=0?MirrorCandidateCount:0)) + ray*PathSlotsPerRay;
    for (uint k = 0; k < PathSlotsPerRay; k++)
        RayPaths[first + k] = (RayPath)0;
    if (RayPairEnabled[query] == 0 || !DebugSelected(query, ray / RaysPerBin))
        return;
    uint frequencyBin = ray / RaysPerBin, rng = HashRay((ray % RaysPerBin) + pair * 0x85ebca6b + RaySeed + 1);
    uint debugBase = PrimitiveCount + 1 + (ray % RaysPerBin) * (2 + 2 * MaxBounces),
         debugPath = 2 + 2 * (ray % RaysPerBin);
    uint sourceIndex=pair/ListenerCount,emissionRow=frame*EmissionFrameStride+sourceIndex;
    uint4 plane=EmissionRanges[emissionRow];
    if(plane.y==0)return;
    uint sampleIndex=plane.w==0?0:plane.w==2?(ray%RaysPerBin)%plane.y:((ray%RaysPerBin)+HashRay(RaySeed ^ source))%plane.y;
    uint localRay=plane.w==2?(ray%RaysPerBin)/plane.y:ray%RaysPerBin;
    uint raysForEmission=plane.w==2?1+(RaysPerBin-1-sampleIndex)/plane.y:RaysPerBin;
    uint birthSeed=0;
    if(plane.w==2){uint4 clock=EmissionClocks[plane.x+sampleIndex];birthSeed=HashRay(clock.y)^HashRay(clock.z);rng=HashRay(localRay^birthSeed^HashRay(pair+RaySeed));}
    if(EmissionPoints[plane.x+sampleIndex].w==0)return;
    float3 emission=EmissionPosition(stateOffset,sourceIndex,sampleIndex);
    float3 emissionVelocity=plane.w==2?float3(0,0,0):PointVelocity(stateOffset+source,emission);
    if(UseVisibleMirrors==0)
    {
        uint primitive=PrimitiveCount==0?0:(localRay+birthSeed+HashRay(source*0x85ebca6b+listener+RaySeed))%PrimitiveCount;
        RayPath mirror=FirstReflection(stateOffset,source,listener,frequencyBin,localRay+birthSeed,is2D,debugBase,emission,emissionVelocity,
            primitive,(float)PrimitiveCount/RaysPerBin,true);
        RayPaths[first++]=ApplyEmission(mirror,emissionRow,sampleIndex,raysForEmission);
    }
    debugBase+=2;
    float frequencyHz = max(1, frequencyBin * FrequencyStep);
    float3 position = emission,
           end = States[stateOffset + listener].position.xyz;
    if (is2D)
    {
        position.z = 0;
        end.z = 0;
    }
    float3 direction = UnitDirection(rng, is2D);
    float gain = 1, time = 0;
    float2 materialWeight=float2(1,0);
    uint medium = MediumAt(stateOffset, position, is2D);
    float3 sourceVelocity = emissionVelocity,
           listenerVelocity = Velocity(stateOffset + listener);
    if (is2D)
    {
        sourceVelocity.z = 0;
        listenerVelocity.z = 0;
    }
    float frequencyRatio = 1;
    uint identity = HashRay(0x80000000u ^ localRay);
    float weight = 1.0 / RaysPerBin;
    uint previousEdge = 0xffffffff;
    float3 edgeSource = position, vertexVelocity = sourceVelocity;
    float edgeDistance = 0, spreadingDistance = 0;
    bool canSampleEdge = true,hadSurface=false;
    float3 surfaceNormal = 0;
    float surfaceScatter = 1;
    for (uint bounce = 0; bounce < MaxBounces; bounce++)
    {
        float3 segmentStart = position;
        float segmentTime = time;
        uint debugSlot = debugBase + 2 * bounce;
        float currentIor = MediumIor(medium, frequencyBin), speed = SpeedOfSound / currentIor,
              limit = (MaximumPathDelay - time) * speed;
        if (limit <= 0 || gain == 0)
            break;
        // Edge and surface choices share the same path state, bounce budget,
        // BVH and arrival slots. No route graph or separate diffraction dispatch.
        float edgeProbability =
            DiffractionFactor > 0 && EdgeCount > 0 && medium == AirMedium && canSampleEdge ? .5 : 0;
        if (edgeProbability > 0 && RandomRay(rng) < edgeProbability)
        {
            uint edge;
            float3 edgePoint;
            float inversePdf;
            if (!SampleEdge(stateOffset, position, end, is2D, rng, edge, edgePoint, inversePdf) ||
                edge == previousEdge)
                break;
            float distance = length(edgePoint - position);
            if (distance <= RayEpsilon || distance > limit ||
                !Visible(stateOffset, position, edgePoint, is2D))
            {
                DebugWrite(debugSlot, debugPath, DebugRejected, -1, position, edgePoint, time,
                           time + distance / speed, gain * weight);
                break;
            }
            float3 incoming = (edgePoint - position) / distance;
            frequencyRatio /= 1 - dot(vertexVelocity, incoming) / speed;
            if (previousEdge != 0xffffffff)
            {
                gain *= EdgeConnectionWeight(previousEdge, edge) *
                        EdgeKernel(previousEdge, stateOffset, edgeSource, position, edgePoint, is2D) / edgeDistance;
                spreadingDistance = 0;
            }
            else if (bounce > 0)
                gain *= surfaceScatter * max(0, dot(surfaceNormal, incoming));
            Travel(distance, medium, frequencyBin, frequencyHz, time, gain);
            spreadingDistance += distance;
            gain *= inversePdf * DiffractionFactor * EdgeOpacity(edge, frequencyBin) / edgeProbability;
            // Preserve the previous vertex exactly when this is an edge-to-edge leg.
            edgeSource = position - incoming * (spreadingDistance - distance);
            edgeDistance = spreadingDistance;
            vertexVelocity = PointVelocity(stateOffset + Edges[edge].ids.x, edgePoint);
            frequencyRatio *= 1 - dot(vertexVelocity, incoming) / speed;
            position = edgePoint;
            DebugWrite(debugSlot, debugPath, DebugDiffraction, Primitives[Edges[edge].ids.y].indices.y,
                       segmentStart, position, segmentTime, time, gain * weight);
            identity = HashRay(identity ^ (edge + 1) ^ 0x20000000u);
            float last = length(end - position);
            if ((UseCompleteRoutes==0 || hadSurface || bounce>=2) && last > RayEpsilon && Visible(stateOffset, position, end, is2D))
            {
                float3 outgoing = (end - position) / last;
                float ratio = frequencyRatio * (1 - dot(listenerVelocity, outgoing) / speed) /
                              (1 - dot(vertexVelocity, outgoing) / speed);
                float pressure = gain * EdgeKernel(edge, stateOffset, edgeSource, position, end, is2D) /
                                 (edgeDistance * last) * ReferenceDistance * weight *
                                 exp(-MediumLoss(medium, frequencyBin, frequencyHz) * last);
                RayPath candidate =
                    ApplyEmission(ApplyMaterial(Arrival(position, end, time + last / speed, pressure, frequencyBin, ratio, identity),materialWeight),emissionRow,sampleIndex,raysForEmission);
                RayPaths[first+bounce]=candidate;
                DebugArrival(debugSlot+1,debugPath,position,end,time,candidate);
            }
            previousEdge = edge;
            canSampleEdge = true;
            direction = UnitDirection(rng, is2D);
            continue;
        }
        gain /= 1 - edgeProbability;
        uint hit;
        float distance;
        float3 normal;
        bool intersects = NearestHit(stateOffset, position, direction, limit, is2D, hit, distance, normal);
        if (!intersects)
        {
            DebugWrite(debugSlot, debugPath, DebugEscape, -1, position, position + direction * limit, time,
                       MaximumPathDelay, gain * weight);
            break;
        }
        frequencyRatio /= 1 - dot(vertexVelocity, direction) / speed;
        if (previousEdge != 0xffffffff)
        {
            gain *= EdgeKernel(previousEdge, stateOffset, edgeSource, position,
                               position + direction * distance, is2D) /
                    edgeDistance;
            spreadingDistance = 0;
            previousEdge = 0xffffffff;
        }
        hadSurface=true;
        position += direction * distance;
        Travel(distance, medium, frequencyBin, frequencyHz, time, gain);
        spreadingDistance += distance;
        RayMaterial material = RayMaterials[Primitives[hit].indices.y * FrequencyBins + frequencyBin];
        gain *= sqrt(saturate(1 - material.surface.x));
        bool entering = dot(direction, normal) < 0;
        float3 n = entering ? normal : -normal;
        float3 velocity = PointVelocity(stateOffset + Primitives[hit].indices.x, position);
        if (is2D)
            velocity.z = 0;
        identity = HashRay(identity ^ (hit + 1));
        float3 toListener = normalize(end - position);
        vertexVelocity = velocity;
        frequencyRatio *= 1 - dot(velocity, direction) / speed;
        float arrivalRatio = frequencyRatio / (1 - dot(velocity, toListener) / speed) *
                             (1 - dot(listenerVelocity, toListener) / speed);
        float last = length(end - position), cosine = last > 0 ? max(0, dot(n, (end - position) / last)) : 0;
        if (cosine > 0 && Visible(stateOffset, position + n * RayEpsilon, end, is2D))
        {
            RayPath candidate =
                ApplyEmission(ApplyMaterial(Arrival(position, end, time + last / speed,
                        gain * sqrt(saturate((1 - material.surface.z) * material.surface.y)) * weight *
                            cosine * .5 * exp(-MediumLoss(medium, frequencyBin, frequencyHz) * last) *
                            ReferenceDistance / max(MinimumDistance, spreadingDistance + last),
                        frequencyBin, arrivalRatio, identity),materialWeight),emissionRow,sampleIndex,raysForEmission);
            RayPaths[first+bounce]=candidate;
            DebugArrival(debugSlot+1,debugPath,position,end,time,candidate);
        }
        surfaceNormal = n;
        surfaceScatter = sqrt(saturate((1 - material.surface.z) * material.surface.y));
        canSampleEdge = surfaceScatter > 0;
        uint nextMedium = MediumAt(stateOffset, position + direction * (2 * RayEpsilon), is2D);
        float nextIor = MediumIor(nextMedium, frequencyBin);
        float transmission = saturate(material.surface.z);
        float3 transmitted = refract(direction, n, currentIor / nextIor);
        if (dot(transmitted, transmitted) == 0)
            transmission = 0; // Total internal reflection leaves no transmitted branch.
        uint eventKind;
        if (RandomRay(rng) < transmission)
        {
            // The legacy contract gives energy fractions. Pressure coefficient / PDF
            // is sqrt(t)/t, not 1. Complex opaque boundaries take the other branch.
            gain /= sqrt(transmission);
            identity = HashRay(identity ^ 0x10000000u);
            direction = normalize(transmitted);
            medium = nextMedium;
            canSampleEdge = false;
            eventKind = DebugTransmission;
        }
        else
        {
            gain /= sqrt(1 - transmission);
            float3 diffuse = UnitDirection(rng, is2D);
            if (dot(diffuse, n) < 0)
                diffuse = -diffuse;
            float scattering = saturate(material.surface.y);
            bool scattered = RandomRay(rng) < scattering;
            gain /= sqrt(scattered ? scattering : 1 - scattering);
            if(material.medium.w!=0)
                materialWeight=ComplexProduct(materialWeight,ReflectionPressure(material,abs(dot(direction,n))));
            direction = scattered ? diffuse : reflect(direction, n);
            eventKind = scattered ? DebugScatter : DebugReflection;
        }
        DebugWrite(debugSlot, debugPath, eventKind,
                   Primitives[hit].indices.y, segmentStart, position, segmentTime, time, gain * weight);
    }
}
