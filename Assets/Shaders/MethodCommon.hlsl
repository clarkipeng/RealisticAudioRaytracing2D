// Shared support for the two single-loop measurement methods, MethodPressure.hlsl
// and MethodEnergy.hlsl (docs/benchmarks/methods-2026-11-06). Static 3D scenes, one
// point source (Sources[0]) and one listener (Listeners[0]) per solve; all eight
// octave bands (125 Hz * 2^b) are carried by every ray as a vector, so the bands
// share one geometric path and its random numbers. Uses Geometry.hlsl and
// Transport.hlsl (media, edges, wedge kernels); no state is shared with the hybrid.
#ifndef ACOUSTIC_METHOD_COMMON
#define ACOUSTIC_METHOD_COMMON
static const uint MethodBands = 8, MethodDepths = 64, MethodCounters = 16, MethodMagnitudes = 64, MethodStatStride = 66;
StructuredBuffer<RayMaterial> MethodMaterials; // [material * 8 + band] at band centres
RWStructuredBuffer<uint> MethodOutput;         // (low, high) 32.32 fixed-point sums: energy cells or signed pressure train samples
RWStructuredBuffer<uint> MethodStats;          // [band][depth][count, max|c| bits, 64 log2|c| buckets], then counters
RWStructuredBuffer<float4> MethodCapture;      // two float4 per captured event
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
// Captured event kinds (Assets/Tests/Editor/MethodMeasurement.cs mirrors them).
static const uint CaptureSource = 1, CaptureSpecular = 2, CaptureDiffuse = 3, CaptureTransmission = 4, CaptureEdge = 5,
                  CaptureConnection = 6, CaptureChain = 7, CaptureEscape = 8, CaptureEnd = 9, CaptureEdgeConnection = 10;

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
    InterlockedAdd(MethodStats[MethodBands * MethodDepths * MethodStatStride + counter], amount);
}
// Only integer atomics: float compare-exchange loops (a loop whose exit depends on a
// buffer read) make the HLSL compiler expand each call site, and the pressure kernel
// has dozens of them. Sums are 32.32 fixed point (low word, high word) with a dither
// whose expectation makes the rounding unbiased; they are exact and order independent.
void MethodAddFixed(uint index, float value, uint key)
{
    if (value == 0) return;
    if (!isfinite(value) || abs(value) >= 2147483000.0) { MethodCount(CounterOverflow); return; }
    float whole = floor(value), part = (value - whole) * 4294967296.0 + (HashRay(key ^ HashRay(index)) >> 8) * (1.0 / 16777216.0);
    if (part >= 4294967296.0) { whole += 1; part -= 4294967296.0; }
    uint low = (uint)part, high = (uint)(int)whole, old;
    InterlockedAdd(MethodOutput[2 * index], low, old);
    if (old + low < old) high++;
    if (high != 0) InterlockedAdd(MethodOutput[2 * index + 1], high);
}
// Per band and interaction depth: count, max |c| and a histogram of floor(log2 |c|)
// (2^-40 ... 2^23) of every listener contribution before the 1/N average. The growth
// of E|c| with depth, and the tail of |c|, are the divergence measurement.
void MethodStatistic(uint band, uint depth, float magnitude)
{
    if (!(magnitude > 0)) return;
    magnitude = isfinite(magnitude) ? magnitude : 3.0e38;
    uint cell = (band * MethodDepths + min(depth, MethodDepths - 1)) * MethodStatStride;
    InterlockedAdd(MethodStats[cell], 1u);
    InterlockedMax(MethodStats[cell + 1], asuint(magnitude));
    int bucket = clamp((int)floor(log2(magnitude)) + 40, 0, (int)MethodMagnitudes - 1);
    InterlockedAdd(MethodStats[cell + 2 + bucket], 1u);
}

// ---- debug capture: rays below MethodCaptureRays record their events ----
void MethodCaptureEvent(uint ray, inout uint slot, uint kind, float3 position, float weight, float contribution,
                        float time, float extra = 0)
{
    if (ray >= MethodCaptureRays || slot >= MethodCaptureEvents) return;
    uint at = 2 * (ray * MethodCaptureEvents + slot++);
    MethodCapture[at] = float4(position, kind);
    MethodCapture[at + 1] = float4(weight, contribution, time, extra);
}

// ---- outputs ----
// Energy histogram [direction(6)][band][bin] at MethodCellBase, 32.32 fixed point; shares
// are summed over rays and divided by MethodTotalRays (as LateEnergy.hlsl, whose layout
// [frame][pair][direction][band][bin] the audio solves share).
void MethodDepositEnergy(float3 arrivalFrom, uint band, float time, float share, uint key)
{
    if (!(share > 0) || !isfinite(share) || time < 0) return;
    uint bin = (uint)floor(time / MethodBinSeconds);
    if (bin >= MethodBins) return;
    if (MethodHeadFrame != 0)
    {
        float4 head = States[MethodState + MethodListenerObject].rotation;
        arrivalFrom = Rotate(float4(-head.xyz, head.w), arrivalFrom);
    }
    uint cell = MethodCellBase + (LateDirection(arrivalFrom) * MethodBands + band) * MethodBins + bin;
    float whole = floor(share), part = (share - whole) * 4294967296.0 + (HashRay(key ^ HashRay(cell)) >> 8) * (1.0 / 16777216.0);
    if (part >= 4294967296.0) { whole += 1; part -= 4294967296.0; }
    uint low = (uint)part, high = (uint)min(whole, 4294967295.0), old;
    InterlockedAdd(MethodOutput[2 * cell], low, old);
    if (old + low < old) high++;
    if (high != 0) InterlockedAdd(MethodOutput[2 * cell + 1], high);
}
// Pressure band trains: band b is sampled at MethodTrainRate0 * 2^b (16 samples per
// centre-frequency period), MethodTrainBase << b samples, real then imaginary train,
// with MethodLeadSeconds of lead. Deposits use 4-tap cubic Lagrange interpolation
// (MethodBands.Deposit on the CPU is identical): its response error is below 0.1% up
// to 2 fc, where a linear split loses up to 10% of the energy depending on where
// the delay falls between samples. The CPU applies the band filters. Values are / N.
void MethodDepositPressure(uint band, float time, float2 value, uint key)
{
    if (!all(isfinite(value)) || (value.x == 0 && value.y == 0) || time < 0) return;
    uint length = MethodTrainBase << band, offset = 2 * MethodTrainBase * ((1u << band) - 1);
    float u = (time + MethodLeadSeconds) * (MethodTrainRate0 * (float)(1u << band));
    uint i0 = (uint)floor(u);
    if (i0 < 1 || i0 + 2 >= length) return;
    float w = u - i0;
    float4 c = float4(-w * (w - 1) * (w - 2) / 6, (w + 1) * (w - 1) * (w - 2) / 2, -(w + 1) * w * (w - 2) / 2, (w + 1) * w * (w - 1) / 6);
    [loop] for (uint t = 0; t < 4; t++)
    {
        uint at = offset + i0 - 1 + t;
        MethodAddFixed(at, value.x * c[t], key + 2 * t);
        MethodAddFixed(at + length, value.y * c[t], key + 2 * t + 1);
    }
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

// ---- unfolding of specular chains ----
// U maps real space after the latest reflection into the straight (unfolded) space
// of the ray that left the anchor: U(y) = A y + c. Reflection at the plane n.y = d
// composes U with the mirror: A' = A - 2 (A n) n^T, c' = c + 2 d (A n).
void MethodUnfoldReflect(inout float3x3 A, inout float3 c, float3 n, float d)
{
    float3 an = mul(A, n);
    A -= 2 * float3x3(an.x * n, an.y * n, an.z * n);
    c += 2 * d * an;
}
float3x3 MethodIdentity() { return float3x3(1, 0, 0, 0, 1, 0, 0, 0, 1); }

// Exact specular chain from the anchor towards the listener image `target` (in the
// anchor's unfolded space): retrace `hits` interactions, reflecting except where the
// walk transmitted (bit i of `transmitted`, first 64 interactions), and require that
// the last leg reaches the real listener unobstructed. Any plane sequence with the
// same image is the same arrival; it is evaluated with its own exact coefficients.
// pressure[b] and energy[b] receive the products of the exact coefficients.
bool MethodRetraceChain(float3 anchor, float3 target, float3 listener, uint hits, uint2 transmitted,
                        out float2 pressure[8], out float energy[8], out float lengthTotal, out float3 arrivalFrom)
{
    [unroll] for (uint k = 0; k < MethodBands; k++) { pressure[k] = float2(1, 0); energy[k] = 1; }
    lengthTotal = length(target - anchor);
    arrivalFrom = 0;
    if (lengthTotal <= 2 * RayEpsilon) return false;
    float3 p = anchor, d = (target - anchor) / lengthTotal;
    float travelled = 0;
    [loop] for (uint i = 0; i < hits; i++)
    {
        uint hit; float distance; float3 normal;
        if (!NearestHit(MethodState, p, d, lengthTotal - travelled - RayEpsilon, false, hit, distance, normal)) return false;
        travelled += distance;
        p += d * distance;
        float3 n = dot(d, normal) < 0 ? normal : -normal;
        float cosine = abs(dot(d, n));
        bool through = i < 64 && ((i < 32 ? transmitted.x >> i : transmitted.y >> (i - 32)) & 1) != 0;
        [unroll] for (uint k = 0; k < MethodBands; k++)
        {
            RayMaterial m = MethodSurface(hit, k);
            if (through)
            {
                float t = m.medium.w != 0 ? 0 : saturate((1 - m.surface.x) * m.surface.z);
                pressure[k] *= sqrt(t); energy[k] *= t;
            }
            else if (m.medium.w != 0)
            {
                float2 r = ReflectionPressure(m, cosine);
                pressure[k] = ComplexProduct(pressure[k], r); energy[k] *= dot(r, r);
            }
            else
            {
                float s = saturate((1 - m.surface.x) * (1 - m.surface.z) * (1 - m.surface.y));
                pressure[k] *= sqrt(s); energy[k] *= s;
            }
        }
        if (!through) d = reflect(d, n);
    }
    float remaining = lengthTotal - travelled;
    if (remaining <= RayEpsilon) return false;
    float3 end = p + d * remaining;
    if (length(end - listener) > 1e-3 * max(1, lengthTotal)) return false;
    if (!Visible(MethodState, p, listener, false)) return false;
    arrivalFrom = -d;
    return true;
}
// Probability that a ray drawn at the anchor passes within MethodDetectRadius of a
// point at distance `distance` in direction w: the cone of half-angle asin(r/distance).
// kind 0: uniform sphere; kind 1: cosine hemisphere about normal n (exact while the
// cone lies inside the hemisphere: integral of cos/pi over the cone is (w.n) sin^2).
float MethodConeProbability(uint kind, float3 w, float distance, float3 n)
{
    float s = MethodDetectRadius / distance, s2 = s * s;
    if (kind == 1) return max(0, dot(w, n)) * s2;
    return .5 * s2 / (1 + sqrt(max(0, 1 - s2))); // (1 - cos)/2 without cancellation
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
// Pressure-random phase of a diffusely scattering surface cell (MethodPressure.hlsl):
// square cells of side wavelength/sqrt(pi) on the primitive's plane.
float MethodCellPhase(uint primitive, float3 position, float3 normal, uint band)
{
    float3 t = abs(normal.x) < .9 ? normalize(cross(normal, float3(1, 0, 0))) : normalize(cross(normal, float3(0, 1, 0)));
    float3 b = cross(normal, t);
    float cell = 2 * RayPi / MethodWavenumber(band) / sqrt(RayPi);
    int2 ij = int2(floor(float2(dot(position, t), dot(position, b)) / cell));
    uint plane = HashRay(asuint(round(dot(normal, position) * 1000)) ^ HashRay(asuint(round(normal.x * 1000)) +
                 HashRay(asuint(round(normal.y * 1000)) + HashRay(asuint(round(normal.z * 1000))))));
    return 2 * RayPi * (HashRay(plane ^ HashRay((uint)ij.x * 0x9e3779b9u + HashRay((uint)ij.y + band * 0x85ebca6bu))) >> 8) / 16777216.0;
}
float3 MethodCosineDirection(float3 n, inout uint rng) { return LateLambert(n, rng, false); }
#endif
