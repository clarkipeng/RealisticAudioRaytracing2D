// Shared support of the path method (MethodPath.hlsl, docs/method.md): 3D scenes, one source and one listener
// per solve; all eight octave bands (125 Hz * 2^b) are carried by every ray as a vector, so the bands share one
// geometric path and its random numbers. Uses Geometry.hlsl and Transport.hlsl (media, edges, wedge kernels).
#ifndef ACOUSTIC_METHOD_COMMON
#define ACOUSTIC_METHOD_COMMON
static const uint MethodBands = 8, MethodDepths = 64, MethodCounters = 16, MethodMagnitudes = 64, MethodStatStride = 66;
StructuredBuffer<RayMaterial> MethodMaterials; // [material * 8 + band] at band centres
RWStructuredBuffer<uint> MethodOutput;         // (low, high) 32.32 fixed-point sums: energy cells or signed pressure train samples
RWStructuredBuffer<uint> MethodStats;          // [band][depth][count, max|c| bits, 64 log2|c| buckets], then counters
uint MethodTotalRays, MethodRayOffset, MethodRayCount, MethodBins, MethodTrainBase, MethodMaxEvents;
uint MethodCaptureRays, MethodCaptureEvents, MethodAnyTransmission, MethodEdgeCount, MethodOutputCount, MethodStatsCount, MethodClearOffset;
float MethodBinSeconds, MethodHorizon, MethodDetectRadius, MethodTrainRate0, MethodLeadSeconds, MethodEdgeProbability;
// One solve: state frame offset (frame * ObjectCount); histogram cell offset of its
// (frame, pair) block; MethodHeadFrame bins arrival directions in the listener's head
// frame (audio) instead of world axes (measurement). Zero by default.
uint MethodState, MethodCellBase, MethodHeadFrame, MethodListenerObject;
// Counters after the depth statistics.
static const uint CounterRays = 0, CounterEvents = 1, CounterEscapes = 2, CounterRoulette = 3, CounterHorizon = 4,
                  CounterEventCap = 5, CounterNonFinite = 6, CounterDetections = 7, CounterChains = 8,
                  CounterOverflow = 9, CounterEdgeConnections = 10, CounterDiffuse = 11, CounterEdgeEvents = 12;

// The latest vertex that is not a delta event: the start of the current straight
// (unfolded) chain and of every listener connection.
struct MethodAnchor
{
    float3 position, normal, previous;  // previous: an edge anchor's wedge-kernel source
    uint kind, edge, primitive, medium; // kind 0 source, 1 diffuse point, 2 edge point (pressure)
    float time;
};
void ClearMethodBuffers(uint index)
{
    index += MethodClearOffset; // dispatched in chunks below Metal's 65535 thread groups
    if (index < MethodOutputCount) MethodOutput[index] = 0;
    if (index < MethodStatsCount) MethodStats[index] = 0;
}

float MethodBandHz(uint band) { return 125.0 * (float)(1u << band); }
float MethodWavenumber(uint band) { return 2 * RayPi * MethodBandHz(band) / SpeedOfSound; }
RayMaterial MethodMaterialOf(uint material, uint band) { return MethodMaterials[material * MethodBands + band]; }
RayMaterial MethodSurface(uint primitive, uint band) { return MethodMaterialOf(Primitives[primitive].indices.y, band); }
float MethodIor(uint medium, uint band) { return medium == AirMedium ? 1 : max(.001, MethodMaterialOf(medium, band).surface.w); }
// Pressure loss per metre (Np); energy uses twice this.
float MethodLoss(uint medium, uint band) { return medium == AirMedium ? AirCoefficient(MethodBandHz(band)) : MethodMaterialOf(medium, band).medium.x; }
// Travel time uses the 1 kHz band's speed: one geometric path carries all bands.
float MethodSlowness(uint medium) { return MethodIor(medium, 3) / SpeedOfSound; }

void MethodCount(uint counter, uint amount = 1)
{
#ifdef PATH_WAVE
    if (!PathFresh()) return; // (made by an earlier wave, or past a query not answered yet: Geometry.hlsl "Waves")
#endif
    InterlockedAdd(MethodStats[MethodBands * MethodDepths * MethodStatStride + counter], amount);
}

// Whether the hit object is a refracting medium (closed, acoustic index not 1): not a partition.
bool MethodRefracts(uint hit)
{
    uint volume = VolumeMedium(Primitives[hit].indices.x);
    return volume != AirMedium && MethodIor(volume, 3) != 1;
}

// ---- straight connections ----
// Visibility with transmission through every crossed surface, as the direct sound
// (straight line, no refraction): factor[b] is multiplied by the pressure (energy)
// transmission coefficient of each crossing (once per partition, see PartitionExit) and the medium loss; time accumulates
// the piecewise travel time. False when an opaque surface blocks the segment.
bool MethodConnect(float3 a, float3 b, uint medium, bool energy, inout float factor[8], out float time)
{
    float total = length(b - a);
    time = 0;
    if (total <= 2 * RayEpsilon) return true;
    float3 d = (b - a) / total;
    if (MethodAnyTransmission == 0 && medium == AirMedium)
    {
        if (!Visible(MethodState, a, b, false)) return false;
        time = total / SpeedOfSound;
        [unroll] for (uint k = 0; k < MethodBands; k++) factor[k] *= exp(-(energy ? 2 : 1) * MethodLoss(AirMedium, k) * total);
        return true;
    }
    float3 p = a;
    float remaining = total;
    [loop] for (uint crossing = 0; crossing < 64 && remaining > 2 * RayEpsilon; crossing++)
    {
        uint hit; float distance; float3 normal;
        if (!NearestHit(MethodState, p, d, remaining - RayEpsilon, false, hit, distance, normal)) break;
        time += distance * MethodSlowness(medium);
        bool passes = false;
        [unroll] for (uint k = 0; k < MethodBands; k++)
        {
            RayMaterial m = MethodSurface(hit, k);
            float t = m.medium.w != 0 ? 0 : saturate((1 - m.surface.x) * m.surface.z);
            factor[k] *= exp(-(energy ? 2 : 1) * MethodLoss(medium, k) * distance) * (energy ? t : sqrt(t));
            passes = passes || factor[k] > 0;
        }
        if (!passes) return false;
        p += d * distance;
        remaining -= distance;
        float depth;
        if (PartitionExit(MethodState, hit, p, d, false, MethodRefracts(hit), depth) && depth < remaining)
        {
            // The far face of the same partition: its transmission was the whole assembly's.
            time += depth * MethodSlowness(medium);
            [unroll] for (uint k = 0; k < MethodBands; k++) factor[k] *= exp(-(energy ? 2 : 1) * MethodLoss(medium, k) * depth);
            p += d * depth;
            remaining -= depth;
        }
        medium = MediumAt(MethodState, p + d * (2 * RayEpsilon), false);
    }
    time += remaining * MethodSlowness(medium);
    [unroll] for (uint k = 0; k < MethodBands; k++) factor[k] *= exp(-(energy ? 2 : 1) * MethodLoss(medium, k) * remaining);
    return true;
}

// ---- edges ----
// Diffracting edges only (the C# side drops edges whose wedge index pi/angle is an
// integer: those wedges are exactly represented by images). A vertex on a surface
// selects half the time among its own primitive's edges, else uniformly.
uint MethodPickEdge(uint primitive, inout uint rng, out float probability)
{
    probability = 0;
    if (MethodEdgeCount == 0) return 0xffffffff;
    uint2 local = primitive == 0xffffffff ? uint2(0, 0) : EdgeRanges[primitive];
    float u = RandomRay(rng);
    uint edge;
    if (local.y > 0 && u < .5) edge = EdgeIndices[local.x + min(local.y - 1, (uint)(2 * u * local.y))];
    else edge = min(MethodEdgeCount - 1, (uint)(RandomRay(rng) * MethodEdgeCount));
    probability = (local.y > 0 ? .5 : 1) / MethodEdgeCount;
    [loop] for (uint i = 0; i < local.y; i++)
        if (EdgeIndices[local.x + i] == edge) probability += .5 / local.y;
    return edge;
}
// The edge after an edge: half the time an edge on one of the previous edge's two
// faces (the other edge of a thick wall's end, the far edge of a slab), else uniform.
float MethodLocalProbability(uint face, uint edge)
{
    uint2 local = EdgeRanges[face];
    if (local.y == 0) return 1.0 / MethodEdgeCount;
    float found = 0;
    [loop] for (uint i = 0; i < local.y; i++) found += EdgeIndices[local.x + i] == edge ? 1 : 0;
    return found / local.y;
}
uint MethodPickNextEdge(uint previous, inout uint rng, out float probability)
{
    uint2 faces = Edges[previous].ids.yz;
    float u = RandomRay(rng);
    uint edge;
    if (u < .5) edge = min(MethodEdgeCount - 1, (uint)(RandomRay(rng) * MethodEdgeCount));
    else
    {
        uint face = faces.x == faces.y || RandomRay(rng) < .5 ? faces.x : faces.y;
        uint2 local = EdgeRanges[face];
        edge = local.y == 0 ? min(MethodEdgeCount - 1, (uint)(RandomRay(rng) * MethodEdgeCount))
                            : EdgeIndices[local.x + min(local.y - 1, (uint)(RandomRay(rng) * local.y))];
    }
    probability = .5 / MethodEdgeCount + (faces.x == faces.y ? .5 * MethodLocalProbability(faces.x, edge)
                  : .25 * (MethodLocalProbability(faces.x, edge) + MethodLocalProbability(faces.y, edge)));
    return edge;
}
// Edge-to-edge leg: the hybrid's shared-face multiplicity (pressure 1/2 along a
// common solid face, EdgeConnectionWeight). A leg grazing a face of the first wedge
// is tested slightly lifted off that face, as the complete routes do.
// first == 0xffffffff: a plain visible leg from a non-edge vertex (weight 1).
float MethodEdgeLeg(uint first, uint second, float3 a, float3 b)
{
    // One visibility test, no loop and no early return inside one (the HLSL compiler
    // flattened such returns and rejected every leg), and no edge lookup for a
    // non-edge vertex (both sides of ?: are evaluated in this HLSL dialect).
    float3 offset = 0;
    float weight = 1;
    if (first != 0xffffffff)
    {
        float3 o, u, f, n; float extent, angle;
        EdgeFrame(first, MethodState, false, o, u, f, n, extent, angle);
        float3 normalA = n, normalB = sin(angle) * f - cos(angle) * n; // air-side face normals
        float span = length(b - a);
        if (abs(dot(normalA, b - a)) <= RayEpsilon * span) offset = 2 * RayEpsilon * normalA;
        else if (abs(dot(normalB, b - a)) <= RayEpsilon * span) offset = 2 * RayEpsilon * normalB;
        weight = EdgeConnectionWeight(first, second);
    }
    return Visible(MethodState, a + offset, b + offset, false) ? weight : 0;
}
float MethodEdgeOpacity(uint edge, uint band)
{
    RayMaterial m = MethodSurface(Edges[edge].ids.y, band);
    return 1 - sqrt(saturate((1 - m.surface.x) * m.surface.z));
}
float3 MethodCosineDirection(float3 n, inout uint rng) { return LateLambert(n, rng, false); }
// Energy kernels of the path method's walk (MethodPath.hlsl), from the energy method that the path method
// replaced (removed; Git history): energies use the direct-sound convention (ReferenceDistance / r)^2, every
// listener contribution is (integrand) / (density), and all eight bands ride on one ray.
//   - edge: the uniform theory of diffraction (Kouyoumjian & Pathak 1974, rigid
//     faces) used as an energy kernel, |D|^2 (Tsingos et al. 2001): the diffracted
//     ray leaves on Keller's cone with energy flux per unit edge length and azimuth
//     |D(phi, phi', beta; kL)|^2 sin^2(beta) I_in, whose ray-tube spreading gives
//     |D|^2 I_in m / (l (m + l)) at distance l. |D|^2 is non-negative and finite at
//     the shadow and reflection boundaries (Fresnel transition functions), which a
//     signed pressure kernel (BTM, BEDRF arXiv 2306.01974) is not, so it can be
//     transported as energy.
//   - a small transmitted share (under half of the continuing energy: a wall, a door, a window) is split off
//     instead of drawn and kept as a one-slot weighted reservoir candidate (MethodPath.hlsl), with Russian
//     roulette relative to a source ray's energy (EnergyBranchFloor).
// Russian roulette: survival min(1, max over bands of energy factor / (probability * floor)):
// paths whose throughput is above MethodRouletteFloor always continue, so the number of
// paths stays constant over the decay instead of falling with the energy.
float MethodRouletteFloor;
// The exact direct arrival of a solve (MethodPath.hlsl PathDirectArrival) and the solve's endpoints.
struct EnergyArrival
{
    float4 direction;   // xyz: listener-to-arrival (world); w: delay rate, 1 - received/emitted frequency
    float4 low, high;   // band energies 125 Hz ... 16 kHz, direct-sound convention
    float delay;        // seconds, retarded for a moving source
    uint identity;      // stable across solves: plane multiset or edge/plane indices
    uint family;        // EnergyFamily* | interaction depth << 8
    float length;       // geometric path length (m)
};
static const uint EnergyFamilyDirect = 0, EnergyFamilySpecular = 1, EnergyFamilyD = 2, EnergyFamilyDD = 3, EnergyFamilyRD = 4,
                  EnergyFamilyDR = 5;
static const uint CounterArrivalOverflow = 13, CounterBranches = 14;
static const float EnergyBranchFloor = .05;
RWStructuredBuffer<EnergyArrival> MethodArrivals;
RWStructuredBuffer<uint> MethodArrivalCounts, MethodSelected;
uint MethodSourceObject, MethodArrivalFamilies, MethodArrivalOrder, MethodPlaneCount;
uint MethodArrivalBase, MethodArrivalCapacity, MethodArrivalSlots, MethodArrivalDeposit, MethodCountIndex, MethodSelectedBase;
uint MethodPathBase, MethodFrequencyBins, MethodFftSize, MethodCountCells;
float MethodSampleRate;
// Edge rays leave the source only, importance-sampled per solve (EnergyEdgeWeights/EnergyEdgeScan):
// edge e is chosen with probability proportional to angle_e * integral of 1/m^2 along it,
// angle_e * (atan((L - z0)/r) - atan(-z0/r)) / r for a source at distance r from the edge line
// (foot of the perpendicular at z0), times 1 if a quick visibility test sees it, else
// MethodEdgeHidden; the point on the edge is drawn with density proportional to 1/m^2 (a
// Cauchy law about z0). The ray's flux is then |D|^2 sin^2(beta) I times the total weight over
// the visibility factor: bounded, where uniform edge and point choices multiplied it by the
// edge count and 1/m^2 (the Church's reverb-tail bursts, docs/benchmarks/energy-audio-2026-11-09).
RWStructuredBuffer<float> MethodEdgeCdf;   // per edge, then its running sum (EnergyEdgeScan)
// The readers' view of it: the path shader's Doppler walks read it through a read-only buffer bound to the same data
// (PathMethod.compute), so they do not count it among their UAVs (README part 44).
#ifndef METHOD_EDGE_CDF_READ
#define METHOD_EDGE_CDF_READ MethodEdgeCdf
#endif
static const float MethodEdgeHidden = .1;
float EnergyEdgeSpan(float3 a, float3 u, float extent, float3 x, out float z0, out float r)
{
    float3 s = x - a;
    z0 = dot(s, u);
    r = max(1e-4, length(s - z0 * u));
    return atan((extent - z0) / r) - atan(-z0 / r);
}
groupshared float EnergyScanParts[256];
void EnergyEdgeScan(uint thread)
{
    uint n = MethodEdgeCount, per = (n + 255) / 256, first = thread * per, last = min(n, first + per);
    float sum = 0;
    [loop] for (uint i = first; i < last; i++) sum += MethodEdgeCdf[i];
    EnergyScanParts[thread] = sum;
    GroupMemoryBarrierWithGroupSync();
    if (thread == 0)
    {
        float run = 0;
        [loop] for (uint j = 0; j < 256; j++) { float v = EnergyScanParts[j]; EnergyScanParts[j] = run; run += v; }
    }
    GroupMemoryBarrierWithGroupSync();
    float run = EnergyScanParts[thread];
    [loop] for (uint k = first; k < last; k++) { run += MethodEdgeCdf[k]; MethodEdgeCdf[k] = run; }
}
uint EnergySampleEdge(float u, out float probability)
{
    float total = METHOD_EDGE_CDF_READ[MethodEdgeCount - 1], target = u * total;
    uint lo = 0, hi = MethodEdgeCount - 1;
    [loop] while (lo < hi) { uint mid = (lo + hi) / 2; if (METHOD_EDGE_CDF_READ[mid] > target) hi = mid; else lo = mid + 1; }
    probability = (METHOD_EDGE_CDF_READ[lo] - (lo > 0 ? METHOD_EDGE_CDF_READ[lo - 1] : 0)) / total;
    return lo;
}
float EnergyDirectivity(MethodAnchor v, float3 w) { return v.kind == 1 ? max(0, dot(w, v.normal)) : 1; }
float EnergySpecular(RayMaterial m, float cosine)
{
    if (m.medium.w != 0) { float2 r = ReflectionPressure(m, cosine); return dot(r, r); }
    return saturate((1 - m.surface.x) * (1 - m.surface.z) * (1 - m.surface.y));
}
// Kouyoumjian-Pathak transition function F(X) = 2i sqrt(X) e^{iX} int_sqrt(X)^inf e^{-i t^2} dt
// = sqrt(2 pi X) (f(z) + i g(z)), z = sqrt(2X/pi), with the Fresnel auxiliary
// functions f, g of Abramowitz & Stegun 7.3.32-33 (|error| <= 2e-3).
float2 UtdTransition(float X)
{
    X = max(X, 0);
    float z = sqrt(2 * X / RayPi);
    float f = (1 + .926 * z) / (2 + 1.792 * z + 3.104 * z * z), g = 1 / (2 + 4.142 * z + 3.492 * z * z + 6.670 * z * z * z);
    return sqrt(2 * RayPi * X) * float2(f, g);
}
// |D|^2 of a rigid wedge with exterior angle n*pi (Kouyoumjian & Pathak 1974, eq. 25-29).
float UtdEnergy(float n, float phiIn, float phiOut, float sinBeta, float k, float distance)
{
    float2 sum = 0;
    [unroll] for (uint i = 0; i < 4; i++)
    {
        float sign = (i & 1) == 0 ? 1 : -1, x = i < 2 ? phiOut - phiIn : phiOut + phiIn;
        float argument = (RayPi + sign * x) / (2 * n);
        if (abs(sin(argument)) < 1e-5) { x += sign * 4e-5 * n; argument = (RayPi + sign * x) / (2 * n); }
        float N = round((x + sign * RayPi) / (2 * RayPi * n)), c = cos(n * RayPi * N - .5 * x);
        sum += (cos(argument) / sin(argument)) * UtdTransition(k * distance * 2 * c * c);
    }
    return dot(sum, sum) / (8 * RayPi * k * n * n * sinBeta * sinBeta);
}
// Keller's stationary point between two points, on the finite edge only.
bool UtdApex(uint edge, float3 from, float3 to, out float3 e)
{
    float3 a, u, f, n; float extent, angle;
    EdgeFrame(edge, MethodState, false, a, u, f, n, extent, angle);
    float3 s = from - a, r = to - a;
    float zs = dot(s, u), zr = dot(r, u), rs = length(s - zs * u), rr = length(r - zr * u);
    e = a;
    if (rs + rr <= 1e-6) return false;
    float z = (zs * rr + zr * rs) / (rs + rr);
    e = a + z * u;
    return z >= 0 && z <= extent;
}
bool UtdAngles(uint edge, float3 from, float3 e, float3 to, out float n, out float phiIn, out float phiOut, out float sinBeta)
{
    float3 a, u, f, nn; float extent, angle;
    EdgeFrame(edge, MethodState, false, a, u, f, nn, extent, angle);
    float3 s = from - e, r = to - e;
    phiIn = TunedEdgeAngle(s, f, nn, angle); phiOut = TunedEdgeAngle(r, f, nn, angle);
    n = angle / RayPi;
    sinBeta = length(s - dot(s, u) * u) / max(1e-12, length(s));
    return phiIn <= angle && phiOut <= angle && sinBeta > 1e-4 && length(r) > RayEpsilon;
}
// Keller's stationary path over two parallel edges (every wall end and slab of the
// box scenes): one common slope along the edges in the unfolded transverse plane.
// Skew edge pairs are not connected (no closed form; not present in box scenes).
bool UtdApexPair(uint first, uint second, float3 from, float3 to, out float3 p1, out float3 p2)
{
    float3 a1, u1, f1, n1, a2, u2, f2, n2; float e1, w1, e2, w2;
    EdgeFrame(first, MethodState, false, a1, u1, f1, n1, e1, w1);
    EdgeFrame(second, MethodState, false, a2, u2, f2, n2, e2, w2);
    p1 = a1; p2 = a2;
    if (first == second || abs(dot(u1, u2)) < .9999) return false;
    float3 s = from - a1, r = to - a1, g = a2 - a1;
    float zs = dot(s, u1), zr = dot(r, u1), zg = dot(g, u1);
    float rs = length(s - zs * u1), d = length(g - zg * u1), rr = length((to - a2) - dot(to - a2, u1) * u1);
    float total = rs + d + rr;
    if (d <= RayEpsilon || total <= 1e-6) return false;
    float z1 = zs + (zr - zs) * rs / total, z2 = zs + (zr - zs) * (rs + d) / total;
    float t2 = (z2 - zg) * sign(dot(u1, u2));
    p1 = a1 + z1 * u1; p2 = a2 + t2 * u2;
    return z1 >= 0 && z1 <= e1 && t2 >= 0 && t2 <= e2;
}

// Deposit one arrival into the histogram (measurement output, or audio arrivals that do
// not fit the path slots). Shares are multiplied by the ray count the readout divides by.
float EnergyBand(EnergyArrival a, uint band) { float4 v = band < 4 ? a.low : a.high; return v[band & 3]; }
// Band energies at frequency f: the comparison's power-complementary raised-cosine
// bands (cos^2(pi/2 log2(f/fc)) between neighbouring centres), flat outside 125 Hz-16 kHz.
float EnergyAt(EnergyArrival a, float f)
{
    float x = clamp(log2(max(f, 1) / 125.0), 0, 7);
    uint b = min(6u, (uint)floor(x));
    float t = x - b, w = cos(.5 * RayPi * t);
    float lower = EnergyBand(a, b), upper = EnergyBand(a, b + 1);
    return w * w * lower + (1 - w * w) * upper;
}
#endif
