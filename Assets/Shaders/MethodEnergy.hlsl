// Method "energy": one energy path-tracing loop solving the room acoustic rendering
// equation (Siltanen et al. 2007, JASA 122) by Monte Carlo (offline measurement
// kernel; docs/benchmarks/methods-2026-11-06). Energies use the direct-sound
// convention (ReferenceDistance / r)^2 and are summed per band, arrival time and
// arrival direction. Every listener contribution is (integrand) / (density).
//
// Vertices and their energy kernels (band centre values, no air absorption in the
// comparison suite but supported):
//   - source: isotropic, intensity ReferenceDistance^2;
//   - surface: absorbed a; transmitted (1-a)t (delta, refracted); scattered
//     (1-a)(1-t)s with Lambert's law (ISO 17497-1 scattering coefficient); specular
//     (1-a)(1-t)(1-s) (delta); impedance surfaces reflect |R(theta)|^2 specularly.
//     The continuation is drawn with the band-mean energy fractions; each band's
//     weight is multiplied by its own fraction / that probability. A small
//     transmitted share into air (under half of the continuing energy, through an
//     open sheet such as a wall face or a door leaf, or out of a closed solid) is
//     split off instead of drawn: the ray continues with the
//     reflected and scattered shares only, and the transmitted branch becomes a
//     candidate of a one-slot weighted reservoir (probability of keeping it = its
//     weight / the summed weight of all candidates so far; the kept branch carries
//     that summed weight). When the ray ends, the kept branch is traced from its
//     surface with Russian roulette on its energy relative to a source ray's
//     (survival min(1, w / EnergyBranchFloor), at its start and after every event:
//     between the two faces of a wall modelled as two sheets a branch would otherwise
//     bounce for hundreds of reflections). Entering a closed solid stays a draw: the
//     energy that gets back out of it (about t^2 / a) is negligible, and splitting
//     there cost the Church's energy solve 47% more for no audible change. Drawing a transmission of t = 1e-4 sent one ray in
//     10^4 through the wall with all its energy, which drew streaks of full-strength
//     interior sound outside; a weak branch that survives carries EnergyBranchFloor
//     of a source ray's energy, and more branches get through: same expectation;
//   - edge: the uniform theory of diffraction (Kouyoumjian & Pathak 1974, rigid
//     faces) used as an energy kernel, |D|^2 (Tsingos et al. 2001): the diffracted
//     ray leaves on Keller's cone with energy flux per unit edge length and azimuth
//     |D(phi, phi', beta; kL)|^2 sin^2(beta) I_in, whose ray-tube spreading gives
//     |D|^2 I_in m / (l (m + l)) at distance l. |D|^2 is non-negative and finite at
//     the shadow and reflection boundaries (Fresnel transition functions), which a
//     signed pressure kernel (BTM, BEDRF arXiv 2306.01974) is not, so it can be
//     transported as energy. Two consecutive parallel edges (DD, every wall end of a
//     thick wall) connect to the listener by the double stationary-phase result;
//     an edge ray never continues into a second edge (no caustic tracking).
// Listener connections at every vertex:
//   - source and every scattering point: straight leg, transmitting as the direct
//     sound; and through one edge (stationary point on Keller's cone, D) or one
//     mirror and one edge (RD), from the same points;
//   - specular chains: point-to-point delta paths cannot be sampled by a walk (a delta
//     vertex next to a point receiver). Pure specular paths from the source up to
//     MethodArrivalOrder reflections are enumerated exactly instead (EnergyArrivals
//     below, with the direct sound and the D/DD/RD/DR edge routes from the source);
//     every other specular segment is scored by a track-length receiver sphere of
//     radius MethodDetectRadius (as LateEnergy.hlsl), which estimates the energy
//     averaged over the sphere. The earlier exact-retrace detection (score the image
//     path when the ray passes within r of the listener image, divided by the cone
//     probability) is biased low wherever the cone straddles a visibility boundary:
//     in the furnished suite room 95% of its detections failed the retrace and the
//     early decay was 40-50% short (docs/benchmarks/energy-audio-2026-11-09).
// Output split (docs/simulation-contracts.md): the enumerated paths are discrete
// arrivals (exact delay, band energies, direction and delay rate); everything else is
// the diffuse histogram. Each path class is counted by exactly one of the two.
// Russian roulette: survival min(1, max over bands of energy factor / (probability * floor)):
// paths whose throughput is above MethodRouletteFloor always continue, so the number of
// paths stays constant over the decay instead of falling with the energy.
float MethodRouletteFloor;
// Arrival enumeration (EnergyArrivals/EnergySelect/EnergyExpand) and the solve's endpoints.
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
static const float EnergyBranchFloor = .05; // transmitted branches below this share of a source ray's energy play roulette
RWStructuredBuffer<EnergyArrival> MethodArrivals;
RWStructuredBuffer<uint> MethodArrivalCounts, MethodSelected;
StructuredBuffer<uint> MethodPrimitivePlane, MethodPlanePrimitive;
uint MethodSourceObject, MethodArrivalFamilies, MethodArrivalOrder, MethodPlaneCount;
uint MethodArrivalBase, MethodArrivalCapacity, MethodArrivalSlots, MethodArrivalDeposit, MethodCountIndex, MethodSelectedBase;
uint MethodPathBase, MethodFrequencyBins, MethodFftSize, MethodCountCells;
uint4 MethodFamilyEnds0, MethodFamilyEnds1; // cumulative candidates: direct, order 1, order 2, D, DD, RD, DR
float MethodSampleRate;
bool EnergyEnumerated(uint family) { return ((MethodArrivalFamilies >> family) & 1) != 0; }
// Edge rays leave the source only, importance-sampled per solve (EnergyEdgeWeights/EnergyEdgeScan):
// edge e is chosen with probability proportional to angle_e * integral of 1/m^2 along it,
// angle_e * (atan((L - z0)/r) - atan(-z0/r)) / r for a source at distance r from the edge line
// (foot of the perpendicular at z0), times 1 if a quick visibility test sees it, else
// MethodEdgeHidden; the point on the edge is drawn with density proportional to 1/m^2 (a
// Cauchy law about z0). The ray's flux is then |D|^2 sin^2(beta) I times the total weight over
// the visibility factor: bounded, where uniform edge and point choices multiplied it by the
// edge count and 1/m^2 (the Church's reverb-tail bursts, docs/benchmarks/energy-audio-2026-11-09).
RWStructuredBuffer<float> MethodEdgeCdf;   // per edge, then its running sum (EnergyEdgeScan)
static const float MethodEdgeHidden = .1;
float EnergyEdgeSpan(float3 a, float3 u, float extent, float3 x, out float z0, out float r)
{
    float3 s = x - a;
    z0 = dot(s, u);
    r = max(1e-4, length(s - z0 * u));
    return atan((extent - z0) / r) - atan(-z0 / r);
}
void EnergyEdgeWeights(uint edge)
{
    if (edge >= MethodEdgeCount) return;
    float3 S = States[MethodState + MethodSourceObject].position.xyz, a, u, f, nn;
    float extent, angle, z0, r;
    EdgeFrame(edge, MethodState, false, a, u, f, nn, extent, angle);
    float weight = 0;
    if (TunedEdgeAngle(S - (a + .5 * extent * u), f, nn, angle) <= angle)
    {
        weight = angle * EnergyEdgeSpan(a, u, extent, S, z0, r) / r;
        float3 near = a + clamp(z0, 0, extent) * u;
        bool seen = false;
        [loop] for (uint i = 0; i < 2 && !seen; i++) seen = Visible(MethodState, S, i == 0 ? near : a + .5 * extent * u, false);
        weight *= seen ? 1 : MethodEdgeHidden;
    }
    MethodEdgeCdf[edge] = weight;
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
    float total = MethodEdgeCdf[MethodEdgeCount - 1], target = u * total;
    uint lo = 0, hi = MethodEdgeCount - 1;
    [loop] while (lo < hi) { uint mid = (lo + hi) / 2; if (MethodEdgeCdf[mid] > target) hi = mid; else lo = mid + 1; }
    probability = (MethodEdgeCdf[lo] - (lo > 0 ? MethodEdgeCdf[lo - 1] : 0)) / total;
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
void EnergyDeposit(float3 from, uint band, float time, float share, uint depth, uint key)
{
    if (time > MethodHorizon) return;
    MethodStatistic(band, depth, share);
    MethodDepositEnergy(from, band, time, share, key);
}
void EnergyListener(MethodAnchor v, float I[8], float3 L, float scale, uint depth, uint ray, inout uint slot, uint key)
{
    float3 w = L - v.position;
    float d = length(w);
    if (d <= RayEpsilon) return;
    w /= d;
    float D = EnergyDirectivity(v, w);
    if (D == 0) return;
    float factor[8]; float time;
    [unroll] for (uint k = 0; k < MethodBands; k++) factor[k] = 1;
    if (!MethodConnect(v.position, L, v.medium, true, factor, time)) return;
    float r = max(MinimumDistance, d), magnitude = 0;
    [unroll] for (uint k = 0; k < MethodBands; k++)
    {
        float share = I[k] * D * factor[k] / (r * r) * scale;
        EnergyDeposit(-w, k, v.time + time, share, depth, key ^ k);
        if (k == 2) magnitude = share;
    }
    MethodCaptureEvent(ray, slot, CaptureConnection, v.position, 0, magnitude, v.time + time);
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
// Listener connection through one edge (D), one mirror and one edge (RD), or two
// parallel edges (DD). Double diffraction: stationary phase of the double BTM
// integral gives |p|^2 = |D1|^2 |D2|^2 I / (m d l (m + d + l)) (|D|^2 = 2 pi K^2 /
// (k sin^2 beta) at Keller's point), with UTD's |D|^2 at each edge (distance
// parameters from the unfolded legs) and the shared-face multiplicity squared.
void EnergyEdgeConnection(MethodAnchor v, float I[8], float3 L, uint depth, uint ray, inout uint slot, inout uint rng)
{
    if (MethodEdgeCount == 0 || DiffractionFactor <= 0) return;
    float pick, pick2 = 1;
    uint edge = MethodPickEdge(v.kind == 1 ? v.primitive : 0xffffffff, rng, pick), edge2 = edge;
    float r = RandomRay(rng);
    bool mirrors = PrimitiveCount > 0;
    bool image = mirrors && r < 1.0 / 3, pair = r >= (mirrors ? 2.0 / 3 : .5);
    float variant = mirrors ? (image ? 1.0 / (3 * PrimitiveCount) : 1.0 / 3) : .5;
    uint mirror = image ? min(PrimitiveCount - 1, (uint)(RandomRay(rng) * PrimitiveCount)) : 0;
    float3 from = v.position, unused, e, e2 = 0, first, hit;
    if (image)
    {
        if ((States[MethodState + Primitives[mirror].indices.x].flags.x & 3) != 3) return;
        from = CompleteImage(mirror, MethodState, v.position, false, unused);
    }
    float n, phiIn, phiOut, sinBeta, n2 = 1, phiIn2 = 0, phiOut2 = 0, sinBeta2 = 1, cosine = 0, leg = 1;
    if (pair)
    {
        edge2 = MethodPickNextEdge(edge, rng, pick2);
        if (!UtdApexPair(edge, edge2, from, L, e, e2) || !UtdAngles(edge, from, e, e2, n, phiIn, phiOut, sinBeta) ||
            !UtdAngles(edge2, e, e2, L, n2, phiIn2, phiOut2, sinBeta2)) return;
        leg = MethodEdgeLeg(edge, edge2, e, e2);
        if (leg == 0) return;
    }
    else if (!UtdApex(edge, from, L, e) || !UtdAngles(edge, from, e, L, n, phiIn, phiOut, sinBeta)) return;
    // Routes from the source are discrete arrivals when their family is enumerated.
    if (v.kind == 0 && v.medium == AirMedium && MediumAt(MethodState, L, false) == AirMedium &&
        EnergyEnumerated(image ? EnergyFamilyRD : pair ? EnergyFamilyDD : EnergyFamilyD)) return;
    float3 last = pair ? e2 : e;
    first = e;
    if (image) { if (!CompleteMirror(mirror, MethodState, v.position, e, false, hit, cosine)) return; first = hit; }
    // Both straight legs (anchor -> first edge unless mirrored, last edge -> listener) at one call site.
    [loop] for (uint segment = image ? 1u : 0u; segment < 2; segment++)
        if (!Visible(MethodState, segment == 0 ? v.position : last, segment == 0 ? e : L, false)) return;
    float D = EnergyDirectivity(v, normalize(first - v.position));
    float m = length(e - from), d = pair ? length(e2 - e) : 0, l = length(L - last), total = m + d + l;
    if (D == 0 || m <= RayEpsilon || l <= RayEpsilon) return;
    MethodCount(CounterEdgeConnections);
    float time = v.time + total * MethodSlowness(AirMedium), magnitude = 0, s2 = sinBeta * sinBeta;
    [unroll] for (uint k = 0; k < MethodBands; k++)
    {
        float opacity = MethodEdgeOpacity(edge, k), factor = DiffractionFactor * DiffractionFactor * opacity * opacity;
        float share = UtdEnergy(n, phiIn, phiOut, sinBeta, MethodWavenumber(k), m * (d + l) / total * s2) * I[k] * D;
        if (pair)
        {
            float opacity2 = MethodEdgeOpacity(edge2, k);
            share *= UtdEnergy(n2, phiIn2, phiOut2, sinBeta2, MethodWavenumber(k), (m + d) * l / total * s2) * leg * leg *
                     DiffractionFactor * DiffractionFactor * opacity2 * opacity2 / (d * pick2);
        }
        share *= factor * exp(-2 * MethodLoss(AirMedium, k) * total) / (m * l * total * pick * variant);
        if (image) share *= EnergySpecular(MethodSurface(mirror, k), cosine);
        EnergyDeposit(normalize(last - L), k, time, share, depth + (image || pair ? 2 : 1), HashRay(rng ^ k));
        if (k == 2) magnitude = share;
    }
    MethodCaptureEvent(ray, slot, CaptureEdgeConnection, e, 0, magnitude, time, image ? 1 : pair ? 2 : 0);
}

// Track-length receiver sphere for one straight segment from p along d (span metres):
// a ray of flux Phi crossing the sphere along a chord c scores Phi c 3 / (4 pi r^3),
// whose expectation is the energy averaged over the sphere (free field: I / d^2;
// LateEnergy.hlsl uses the same estimator). The chord counts only where the listener
// sees it, so a sphere that straddles a wall does not hear the far side.
void EnergySphere(float3 p, float3 d, float span, float flux[8], float tau, uint medium, float3 L, uint depth,
                  uint ray, inout uint slot, uint key)
{
    float r = MethodDetectRadius;
    float3 v = L - p;
    float along = dot(v, d), closest = dot(v, v) - along * along;
    if (closest >= r * r) return;
    float halfChord = sqrt(r * r - closest), a = max(0, along - halfChord), b = min(span, along + halfChord);
    if (b <= a) return;
    float at = clamp(along, a, b);
    if (!Visible(MethodState, L, p + d * at, false)) return;
    MethodCount(CounterChains);
    float time = tau + at * MethodSlowness(medium), magnitude = 0, scale = 3 * (b - a) / (4 * RayPi * r * r * r);
    [unroll] for (uint k = 0; k < MethodBands; k++)
    {
        float share = flux[k] * exp(-2 * MethodLoss(medium, k) * at) * scale;
        EnergyDeposit(-d, k, time, share, depth, key ^ k);
        if (k == 2) magnitude = share;
    }
    MethodCaptureEvent(ray, slot, CaptureChain, p + d * at, 0, magnitude, time);
}

// Late-field accumulation across solves (audio, settings.lateAccumulationSeconds; see
// MethodTransport): each late cell of the batch's frames becomes the exponential moving
// average acc += alpha (e - acc) of the solves so far, in the histogram's 32.32 fixed point
// (value = energy x rays). MethodAccumulator keeps the average between batches.
RWStructuredBuffer<float> MethodAccumulator;
uint MethodAccumulateCells, MethodAccumulateFrames, MethodAccumulateReset;
float MethodAccumulateAlpha;
void EnergyAccumulate(uint cell)
{
    if (cell >= MethodAccumulateCells) return;
    float acc = MethodAccumulator[cell];
    for (uint f = 0; f < MethodAccumulateFrames; f++)
    {
        uint at = 2 * (f * MethodAccumulateCells + cell);
        float e = MethodOutput[at + 1] + MethodOutput[at] / 4294967296.0;
        acc = MethodAccumulateReset != 0 && f == 0 ? e : acc + MethodAccumulateAlpha * (e - acc);
        float whole = floor(acc);
        MethodOutput[at + 1] = (uint)whole;
        MethodOutput[at] = (uint)min(4294967040.0, (acc - whole) * 4294967296.0);
    }
    MethodAccumulator[cell] = acc;
}

// Whether an object is a closed solid (a volume with its own medium; Volumes are sorted by owner).
bool EnergySolid(uint object)
{
    uint lo = 0, hi = VolumeCount;
    while (lo < hi) { uint mid = lo + (hi - lo) / 2; if (Volumes[mid].ids.z < object) lo = mid + 1; else hi = mid; }
    return lo < VolumeCount && Volumes[lo].ids.z == object;
}

// One flat loop: each iteration makes the pending connections of the latest vertex,
// then traces one segment (an edge ray or a direction from the anchor starts a new
// chain), so every heavy helper has a single call site (small kernel and stack).
void TraceEnergy(uint index)
{
    if (index >= MethodRayCount) return;
    uint ray = MethodRayOffset + index, slot = 0;
    MethodCount(CounterRays);
    float3 S = States[MethodState + MethodSourceObject].position.xyz, L = States[MethodState + MethodListenerObject].position.xyz;
    uint rng = HashRay(ray * 0x9e3779b9u ^ HashRay(RaySeed * 0xc2b2ae35u + 0x2545f491u));
    MethodAnchor v;
    v.position = S; v.normal = 0; v.previous = S; v.kind = 0; v.edge = 0; v.primitive = 0xffffffff; v.time = 0;
    v.medium = MediumAt(MethodState, S, false);
    // Discrete arrivals hold these path classes only when both ends are in air (EnergyArrivals).
    bool air = v.medium == AirMedium && MediumAt(MethodState, L, false) == AirMedium;
    float I[8], cI[8], Phi0[8], C[8];
    [unroll] for (uint k = 0; k < MethodBands; k++) { I[k] = ReferenceDistance * ReferenceDistance; cI[k] = I[k]; Phi0[k] = 0; C[k] = 1; }
    MethodCaptureEvent(ray, slot, CaptureSource, S, 1, 0, 0);
    // Pending connections of vertex c; ray 0 carries the deterministic direct sound for
    // every ray unless the direct sound is a discrete arrival.
    MethodAnchor c = v;
    bool connectListener = ray == 0 && !EnergyEnumerated(EnergyFamilyDirect), connectEdge = true;
    float listenerScale = MethodTotalRays;
    uint depth = 0, connectDepth = 0, events = 0;
    bool edges = MethodEdgeCount > 0 && DiffractionFactor > 0, travelling = false, done = false;
    // Chain state. An edge ray's flux needs the distance to its first hit (UTD's
    // distance parameter), so it is completed there (fluxPending).
    float3 p = S, d = 0;
    float Q = 1, tau = 0;
    uint hits = 0, reflections = 0, medium = v.medium, edge = 0;
    bool through = false, viaEdge = false, fluxPending = false, inBranch = false;
    // Transmitted-branch reservoir (one slot): summed candidate weight, and the kept
    // branch's surface point, direction, medium, time, path counters and band flux / weight.
    float branchSum = 0, branchTau = 0; float3 branchP = 0, branchD = 0; float branchF[8];
    uint branchMedium = 0, branchHits = 0, branchReflections = 0, branchDepth = 0;
    [unroll] for (uint k = 0; k < MethodBands; k++) branchF[k] = 0;
    float edgePick = 1, edgeExtent = 1, edgeAngle = 1, edgeM = 1, edgeSinBeta = 1, edgePhiIn = 0, edgePhi = 0, edgeD = 1;
    [loop] for (;;)
    {
        if (connectListener)
        {
            connectListener = false;
            EnergyListener(c, cI, L, listenerScale, connectDepth, ray, slot, HashRay(rng ^ 0x1234567u));
        }
        if (connectEdge)
        {
            connectEdge = false;
            EnergyEdgeConnection(c, cI, L, connectDepth, ray, slot, rng);
            continue;
        }
        if (done)
        {
            if (!(branchSum > 0)) break;
            // Trace the kept transmitted branch from its surface, as a specular chain,
            // after roulette on its share of a source ray's flux (4 pi r_ref^2).
            float keep = min(1, branchSum / (4 * RayPi * ReferenceDistance * ReferenceDistance * EnergyBranchFloor));
            if (RandomRay(rng) >= keep) break;
            [unroll] for (uint k = 0; k < MethodBands; k++) { Phi0[k] = branchF[k] * branchSum / keep; C[k] = 1; }
            p = branchP; d = branchD; medium = branchMedium; tau = branchTau; Q = 1;
            hits = branchHits; reflections = branchReflections; depth = branchDepth;
            through = true; viaEdge = false; fluxPending = false; travelling = true; done = false; branchSum = 0; inBranch = true;
            MethodCount(CounterBranches);
        }
        if (!travelling)
        {
            if (++events >= MethodMaxEvents) { MethodCount(CounterEventCap); break; }
            // Edge rays from the source only (see EnergyEdgeWeights); a scattering point's
            // diffraction is its D/RD/DD connections.
            bool edgeRays = edges && v.kind == 0 && MethodEdgeCdf[max(MethodEdgeCount, 1u) - 1] > 0; // (no short-circuit in HLSL)
            viaEdge = edgeRays && RandomRay(rng) < MethodEdgeProbability;
            [unroll] for (uint k = 0; k < MethodBands; k++) C[k] = 1;
            hits = 0; reflections = 0; through = false; medium = v.medium;
            if (viaEdge)
            {
                // Edge ray: edge and point importance-sampled by 1/m^2 (EnergyEdgeWeights),
                // azimuth uniform on Keller's cone.
                float3 a, u, f, nn; float extent, angle, z0, r;
                edge = EnergySampleEdge(RandomRay(rng), edgePick);
                EdgeFrame(edge, MethodState, false, a, u, f, nn, extent, angle);
                float span = EnergyEdgeSpan(a, u, extent, v.position, z0, r);
                float z = clamp(z0 + r * tan(atan(-z0 / r) + RandomRay(rng) * span), 0, extent);
                float3 e = a + z * u, w = e - v.position;
                float m = length(w);
                edgeExtent = (r * r + (z - z0) * (z - z0)) * span / r; // 1 / (density of the point on the edge)
                if (m <= RayEpsilon || TunedEdgeAngle(-w, f, nn, angle) > angle || !Visible(MethodState, v.position, e, false))
                { MethodCaptureEvent(ray, slot, CaptureEnd, v.position, 0, 0, v.time); done = true; continue; }
                w /= m;
                float cosBeta = dot(w, u);
                edgeSinBeta = sqrt(max(0, 1 - cosBeta * cosBeta));
                edgeD = EnergyDirectivity(v, w);
                if (edgeD == 0 || edgeSinBeta <= 1e-4) { MethodCaptureEvent(ray, slot, CaptureEnd, v.position, 0, 0, v.time); done = true; continue; }
                edgePhiIn = TunedEdgeAngle(v.position - e, f, nn, angle);
                edgePhi = RandomRay(rng) * angle;
                d = normalize(u * cosBeta + edgeSinBeta * (f * cos(edgePhi) + nn * sin(edgePhi)));
                edgeAngle = angle; edgeM = m;
                tau = v.time + m * MethodSlowness(AirMedium);
                p = e; Q = 1; fluxPending = true;
                depth++;
                MethodCount(CounterEdgeEvents);
                MethodCaptureEvent(ray, slot, CaptureEdge, e, 0, 0, tau);
            }
            else
            {
                float3 w0 = v.kind == 1 ? MethodCosineDirection(v.normal, rng) : UnitDirection(rng, false);
                float pdf = v.kind == 1 ? max(1e-12, dot(w0, v.normal)) / RayPi : 1 / (4 * RayPi);
                float D = EnergyDirectivity(v, w0);
                if (D == 0) { done = true; continue; }
                [unroll] for (uint k = 0; k < MethodBands; k++) Phi0[k] = I[k] * D / pdf;
                Q = edgeRays ? 1 - MethodEdgeProbability : 1;
                p = v.position; d = w0; tau = v.time;
            }
            travelling = true;
        }
        if (++events >= MethodMaxEvents) { MethodCount(CounterEventCap); break; }
        float limit = (MethodHorizon - tau) / MethodSlowness(medium);
        uint hit; float distance; float3 normal;
        if (limit <= 0) { MethodCount(CounterHorizon); done = true; continue; }
        bool found = NearestHit(MethodState, p, d, limit, false, hit, distance, normal);
        if (fluxPending && found)
        {
            // UTD energy flux per unit edge length and azimuth, over the sampling densities.
            fluxPending = false;
            float spread = edgeM * distance / (edgeM + distance);
            [unroll] for (uint k = 0; k < MethodBands; k++)
            {
                float opacity = MethodEdgeOpacity(edge, k);
                Phi0[k] = UtdEnergy(edgeAngle / RayPi, edgePhiIn, edgePhi, edgeSinBeta, MethodWavenumber(k), spread * edgeSinBeta * edgeSinBeta) *
                          edgeSinBeta * edgeSinBeta * I[k] * edgeD * opacity * opacity * DiffractionFactor * DiffractionFactor *
                          exp(-2 * MethodLoss(AirMedium, k) * edgeM) / (edgeM * edgeM) * edgeExtent * edgeAngle / (edgePick * MethodEdgeProbability);
            }
        }
        // Specular segments (after a reflection; a transmitted-only leg is the anchor's
        // straight connection) reach the listener through the receiver sphere, except the
        // path classes held by the discrete arrivals: pure reflections from the source up
        // to the enumerated order, and one reflection after an edge ray from the source (DR).
        bool held = air && v.kind == 0 && !through &&
                    (viaEdge ? reflections == 1 && EnergyEnumerated(EnergyFamilyDR)
                             : reflections <= MethodArrivalOrder && EnergyEnumerated(EnergyFamilySpecular));
        if (reflections > 0 && !held)
        {
            float flux[8];
            [unroll] for (uint k = 0; k < MethodBands; k++) flux[k] = Phi0[k] * C[k] / Q;
            EnergySphere(p, d, found ? distance : limit, flux, tau, medium, L, depth, ray, slot, HashRay(rng ^ 0x5bd1e995u ^ events));
        }
        if (!found)
        {
            MethodCount(CounterEscapes);
            MethodCaptureEvent(ray, slot, CaptureEscape, p + d * min(limit, 50), 0, 0, tau);
            done = true; continue;
        }
        p += d * distance; tau += distance * MethodSlowness(medium);
        [unroll] for (uint k = 0; k < MethodBands; k++) C[k] *= exp(-2 * MethodLoss(medium, k) * distance);
        float3 n = dot(d, normal) < 0 ? normal : -normal;
        float cosI = max(0, -dot(d, n));
        depth++;
        MethodCount(CounterEvents);
        float eT[8], eD[8], eS[8], mT = 0, mD = 0, mS = 0;
        [unroll] for (uint k = 0; k < MethodBands; k++)
        {
            RayMaterial m = MethodSurface(hit, k);
            bool impedance = m.medium.w != 0;
            eT[k] = impedance ? 0 : saturate((1 - m.surface.x) * m.surface.z);
            eD[k] = impedance ? 0 : saturate((1 - m.surface.x) * (1 - m.surface.z) * m.surface.y);
            eS[k] = EnergySpecular(m, cosI);
            mT += eT[k]; mD += eD[k]; mS += eS[k];
        }
        MethodAnchor x;
        x.position = p; x.normal = n; x.previous = p; x.kind = 1; x.edge = 0; x.primitive = hit; x.time = tau; x.medium = medium;
        if (mD > 0)
        {
            [unroll] for (uint k = 0; k < MethodBands; k++) cI[k] = Phi0[k] * C[k] / Q * eD[k] / RayPi;
            c = x; connectDepth = depth; listenerScale = 1;
            connectListener = connectEdge = true;
        }
        float total = mT + mD + mS;
        if (!(total > 0)) { MethodCaptureEvent(ray, slot, CaptureEnd, p, 0, 0, tau); done = true; continue; }
        bool split = mT > 0 && mT < .5 * total && (medium != AirMedium || !EnergySolid(Primitives[hit].indices.x));
        if (split)
        {
            // Small transmitted share: a reservoir candidate (see the header), not a draw.
            float branch[8], weight = 0;
            [unroll] for (uint k = 0; k < MethodBands; k++) { branch[k] = Phi0[k] * C[k] * eT[k] / Q; weight = max(weight, branch[k]); }
            if (weight > 0)
            {
                branchSum += weight;
                if (RandomRay(rng) * branchSum < weight)
                {
                    uint next = MediumAt(MethodState, p + d * (2 * RayEpsilon), false);
                    float3 bent = refract(d, n, MethodIor(medium, 3) / MethodIor(next, 3));
                    bool refracted = dot(bent, bent) > 0;
                    branchP = p; branchD = refracted ? normalize(bent) : reflect(d, n); branchMedium = refracted ? next : medium;
                    branchTau = tau; branchHits = hits + 1; branchReflections = reflections + (refracted ? 0 : 1); branchDepth = depth;
                    [unroll] for (uint k = 0; k < MethodBands; k++) branchF[k] = branch[k] / weight;
                }
            }
            total = mD + mS;
        }
        float u = RandomRay(rng) * total;
        if (u < mD)
        {
            [unroll] for (uint k = 0; k < MethodBands; k++) I[k] = cI[k] * (total / mD);
            v = x;
            travelling = false;
            MethodCount(CounterDiffuse);
            MethodCaptureEvent(ray, slot, CaptureDiffuse, p, I[2] * RayPi, 0, tau);
            if (v.time > MethodHorizon) { MethodCount(CounterHorizon); done = true; }
            continue;
        }
        bool transmit = !split && u < mD + mT;
        if (transmit)
        {
            uint next = MediumAt(MethodState, p + d * (2 * RayEpsilon), false);
            float3 bent = refract(d, n, MethodIor(medium, 3) / MethodIor(next, 3));
            [unroll] for (uint k = 0; k < MethodBands; k++) C[k] *= eT[k];
            Q *= mT / total;
            if (dot(bent, bent) > 0) { d = normalize(bent); medium = next; through = true; }
            else { d = reflect(d, n); reflections++; }
        }
        else
        {
            [unroll] for (uint k = 0; k < MethodBands; k++) C[k] *= eS[k];
            Q *= mS / total;
            d = reflect(d, n);
            reflections++;
        }
        hits++;
        float weight = 0;
        [unroll] for (uint k = 0; k < MethodBands; k++) weight = max(weight, C[k]);
        MethodCaptureEvent(ray, slot, transmit ? CaptureTransmission : CaptureSpecular, p, Phi0[2] * C[2] / Q, 0, tau);
        float survive = min(1, weight / (Q * MethodRouletteFloor));
        if (inBranch)
        {
            // A branch plays roulette on its share of a source ray's flux, so energy caught
            // between the two faces of a wall stops being traced once it is weak.
            float flux = 0;
            [unroll] for (uint k = 0; k < MethodBands; k++) flux = max(flux, Phi0[k] * C[k]);
            survive = min(survive, flux / (Q * 4 * RayPi * ReferenceDistance * ReferenceDistance * EnergyBranchFloor));
        }
        if (survive < 1)
        {
            if (RandomRay(rng) >= survive)
            { MethodCount(CounterRoulette); MethodCaptureEvent(ray, slot, CaptureEnd, p, 0, 0, tau); done = true; continue; }
            Q *= survive;
        }
    }
}

// ---- discrete arrivals ----
// Mirror image of x in a plane (its representative primitive, world pose of this solve).
float3 EnergyPlaneImage(uint plane, float3 x)
{
    float3 normal;
    return CompleteImage(MethodPlanePrimitive[plane], MethodState, x, false, normal);
}
// Exact specular leg from `origin` towards `target` (the end point unfolded through the
// planes), reflecting at `count` (0..2) hits that must lie on planes.x then planes.y,
// and reaching `end` unobstructed. Multiplies energy[] by the specular coefficients.
bool EnergyRetrace(float3 origin, float3 target, float3 end, uint count, uint2 planes, inout float energy[8],
                   out float span, out float3 departure, out float3 arrival)
{
    span = length(target - origin);
    departure = 0; arrival = 0;
    if (span <= 2 * RayEpsilon) return false;
    float3 p = origin, d = (target - origin) / span;
    departure = d;
    float travelled = 0;
    [loop] for (uint i = 0; i < count; i++)
    {
        uint hit; float distance; float3 normal;
        if (!NearestHit(MethodState, p, d, span - travelled - RayEpsilon, false, hit, distance, normal)) return false;
        if (MethodPrimitivePlane[hit] != (i == 0 ? planes.x : planes.y) ||
            (States[MethodState + Primitives[hit].indices.x].flags.x & 3) != 3) return false;
        travelled += distance;
        p += d * distance;
        float3 n = dot(d, normal) < 0 ? normal : -normal;
        float cosine = -dot(d, n);
        [unroll] for (uint k = 0; k < MethodBands; k++) energy[k] *= EnergySpecular(MethodSurface(hit, k), cosine);
        d = reflect(d, n);
    }
    float remaining = span - travelled;
    if (remaining <= RayEpsilon || length(p + d * remaining - end) > 1e-3 * max(1, span)) return false;
    if (!Visible(MethodState, p, end, false)) return false;
    arrival = d;
    return true;
}
uint EnergyFamilyDepth(uint family, uint order) { return family == EnergyFamilyDirect ? 0 : family == EnergyFamilyD ? 1 : family == EnergyFamilySpecular ? order : 2; }
// Deposit one arrival into the histogram (measurement output, or audio arrivals that do
// not fit the path slots). Shares are multiplied by the ray count the readout divides by.
float EnergyBand(EnergyArrival a, uint band) { float4 v = band < 4 ? a.low : a.high; return v[band & 3]; }
void EnergyDepositArrival(EnergyArrival a, uint key)
{
    [unroll] for (uint k = 0; k < MethodBands; k++)
        EnergyDeposit(a.direction.xyz, k, a.delay, EnergyBand(a, k) * MethodTotalRays, a.family >> 8, key ^ k);
}
// One thread per candidate route from the source: the direct sound, every plane
// (order 1) and ordered plane pair (order 2), and the D, DD, RD, DR edge routes.
// Candidates are exact paths evaluated with the walk's own kernels; valid ones are
// appended to this solve's arrival list. Specular legs and straight legs share one
// retrace call; the source is isotropic.
void EnergyArrivals(uint index)
{
    uint ends[7] = {MethodFamilyEnds0.x, MethodFamilyEnds0.y, MethodFamilyEnds0.z, MethodFamilyEnds0.w,
                    MethodFamilyEnds1.x, MethodFamilyEnds1.y, MethodFamilyEnds1.z};
    if (index >= ends[6]) return;
    uint route = 0;
    [unroll] for (uint g = 0; g < 6; g++) route += index >= ends[g] ? 1 : 0;
    uint local = index - (route == 0 ? 0 : ends[route - 1]);
    float3 S = States[MethodState + MethodSourceObject].position.xyz, L = States[MethodState + MethodListenerObject].position.xyz;
    uint mediumS = MediumAt(MethodState, S, false);
    float E[8], coefficient[8];
    [unroll] for (uint k = 0; k < MethodBands; k++) { E[k] = ReferenceDistance * ReferenceDistance; coefficient[k] = 1; }
    float time = 0, total = 0;
    float3 departure = 0, arrival = 0;
    uint identity = 0, family = EnergyFamilyDirect, order = 0;
    bool valid = true;
    if (route == 0)
    {
        valid = MethodConnect(S, L, mediumS, true, coefficient, time);
        total = length(L - S);
        departure = arrival = total > 0 ? (L - S) / total : float3(0, 0, 1);
        float r = max(MinimumDistance, total);
        [unroll] for (uint k = 0; k < MethodBands; k++) E[k] *= coefficient[k] / (r * r);
        identity = 0x2545f491u;
    }
    else if (mediumS != AirMedium || MediumAt(MethodState, L, false) != AirMedium) valid = false;
    else
    {
        // Up to two legs (origin, unfolded target, end, reflections, planes).
        float3 fromA = S, targetA = L, endA = L, fromB = 0, targetB = 0, endB = 0;
        uint countA = 0, countB = 0, legs = 1, edge = 0, edge2 = 0;
        uint2 planesA = 0, planesB = 0;
        float n = 1, phiIn = 0, phiOut = 0, sinBeta = 1, n2 = 1, phiIn2 = 0, phiOut2 = 0, sinBeta2 = 1, m = 0, gap = 0, l = 0, leg = 1;
        if (route <= 2)
        {
            family = EnergyFamilySpecular; order = route;
            uint p1 = route == 1 ? local : local / MethodPlaneCount, p2 = route == 1 ? 0 : local % MethodPlaneCount;
            valid = route == 1 || p1 != p2;
            targetA = route == 1 ? EnergyPlaneImage(p1, L) : EnergyPlaneImage(p1, EnergyPlaneImage(p2, L));
            countA = route; planesA = uint2(p1, p2);
            identity = HashRay(0x51ed27u * route + HashRay(p1 + 1) + (route == 2 ? HashRay(p2 + 1) : 0));
        }
        else
        {
            family = route == 3 ? EnergyFamilyD : route == 4 ? EnergyFamilyDD : route == 5 ? EnergyFamilyRD : EnergyFamilyDR;
            uint plane = route >= 5 ? local / MethodEdgeCount : 0;
            edge = route == 4 ? local / MethodEdgeCount : local % MethodEdgeCount;
            edge2 = route == 4 ? local % MethodEdgeCount : edge;
            float3 from = route == 5 ? EnergyPlaneImage(plane, S) : S, to = route == 6 ? EnergyPlaneImage(plane, L) : L, e = 0, e2 = 0;
            if (route == 4)
                valid = edge != edge2 && UtdApexPair(edge, edge2, from, to, e, e2) && UtdAngles(edge, from, e, e2, n, phiIn, phiOut, sinBeta) &&
                        UtdAngles(edge2, e, e2, to, n2, phiIn2, phiOut2, sinBeta2);
            else valid = UtdApex(edge, from, to, e) && UtdAngles(edge, from, e, to, n, phiIn, phiOut, sinBeta);
            if (valid && route == 4) { leg = MethodEdgeLeg(edge, edge2, e, e2); valid = leg > 0; }
            e2 = route == 4 ? e2 : e;
            m = length(e - from); gap = length(e2 - e); l = length(to - e2);
            valid = valid && m > RayEpsilon && l > RayEpsilon && (route != 4 || gap > RayEpsilon);
            // Leg A reaches the first edge (via the plane for RD); leg B leaves the last edge (via the plane for DR).
            targetA = route == 5 ? EnergyPlaneImage(plane, e) : e; endA = e; countA = route == 5 ? 1 : 0; planesA = uint2(plane, 0);
            fromB = e2; targetB = to; endB = L; countB = route == 6 ? 1 : 0; planesB = uint2(plane, 0); legs = 2;
            identity = HashRay(0x7f4a7c15u * route + HashRay(edge + 1) + 0x632be5abu * HashRay(route == 4 ? edge2 + 1 : plane + 1));
        }
        float spans[2] = {0, 0};
        [loop] for (uint i = 0; i < legs && valid; i++)
        {
            float span; float3 out0, out1;
            valid = EnergyRetrace(i == 0 ? fromA : fromB, i == 0 ? targetA : targetB, i == 0 ? endA : endB, i == 0 ? countA : countB,
                                  i == 0 ? planesA : planesB, coefficient, span, out0, out1);
            if (i == 0) departure = out0;
            arrival = out1;
            spans[i] = span;
        }
        if (valid && family == EnergyFamilySpecular)
        {
            total = spans[0];
            float r = max(MinimumDistance, total);
            [unroll] for (uint k = 0; k < MethodBands; k++) E[k] *= coefficient[k] * exp(-2 * MethodLoss(AirMedium, k) * total) / (r * r);
        }
        else if (valid)
        {
            // As EnergyEdgeConnection without the sampling densities.
            total = m + gap + l;
            float s2 = sinBeta * sinBeta;
            [loop] for (uint k = 0; k < MethodBands; k++) // one UtdEnergy body per edge (compile size)
            {
                float opacity = MethodEdgeOpacity(edge, k);
                float share = UtdEnergy(n, phiIn, phiOut, sinBeta, MethodWavenumber(k), m * (gap + l) / total * s2) *
                              DiffractionFactor * DiffractionFactor * opacity * opacity;
                if (family == EnergyFamilyDD)
                {
                    float opacity2 = MethodEdgeOpacity(edge2, k);
                    share *= UtdEnergy(n2, phiIn2, phiOut2, sinBeta2, MethodWavenumber(k), (m + gap) * l / total * s2) * leg * leg *
                             DiffractionFactor * DiffractionFactor * opacity2 * opacity2 / gap;
                }
                E[k] *= share * coefficient[k] * exp(-2 * MethodLoss(AirMedium, k) * total) / (m * l * total);
            }
        }
        time = total * MethodSlowness(AirMedium);
    }
    if (!valid) return;
    // Doppler: retarded emission for a moving source (image sources move with the
    // mirrored velocity, which has the same component along the departure direction)
    // and the frequency ratio (c - vL.u_arrival) / (c - vS.u_departure).
    float c = SpeedOfSound;
    float3 vS = Velocity(MethodState + MethodSourceObject), vL = Velocity(MethodState + MethodListenerObject);
    float a = c * c - dot(vS, vS), b = total * dot(vS, departure);
    float retarded = a > 0 ? (b + sqrt(max(0, b * b + a * total * total))) / a : total / c;
    float3 emitted = normalize(total * departure + vS * retarded);
    if (family == EnergyFamilyDirect) arrival = emitted;
    float ratio = (c - dot(vL, arrival)) / (c - dot(vS, emitted));
    EnergyArrival result;
    result.direction = float4(-arrival, 1 - ratio);
    result.low = float4(E[0], E[1], E[2], E[3]);
    result.high = float4(E[4], E[5], E[6], E[7]);
    result.delay = time + retarded - total / c;
    result.identity = identity;
    result.family = family | (EnergyFamilyDepth(family, order) << 8);
    result.length = total;
    if (!(result.delay >= 0 && result.delay <= MethodHorizon && ratio > 0) || !(dot(result.low + result.high, 1) > 0)) return;
    uint at;
    InterlockedAdd(MethodArrivalCounts[MethodCountIndex], 1u, at);
    if (at < MethodArrivalCapacity) MethodArrivals[MethodArrivalBase + at] = result;
    else { MethodCount(CounterArrivalOverflow); EnergyDepositArrival(result, HashRay(index ^ 0x3c6ef372u)); }
}
float EnergyKey(EnergyArrival a) { return dot(a.low, 1) + dot(a.high, 1); }
// Deterministic order of the solve's arrivals (energy, then identity, then index): the
// strongest MethodArrivalSlots become audio paths, the rest go to the histogram. With
// MethodArrivalDeposit (measurement) every arrival is also deposited.
void EnergySelect(uint index)
{
    uint count = min(MethodArrivalCounts[MethodCountIndex], MethodArrivalCapacity);
    if (index >= count) return;
    EnergyArrival a = MethodArrivals[MethodArrivalBase + index];
    float key = EnergyKey(a);
    uint rank = 0;
    [loop] for (uint j = 0; j < count; j++)
    {
        EnergyArrival b = MethodArrivals[MethodArrivalBase + j];
        float other = EnergyKey(b);
        rank += other > key || (other == key && (b.identity < a.identity || (b.identity == a.identity && j < index))) ? 1 : 0;
    }
    if (rank < MethodArrivalSlots) MethodSelected[MethodSelectedBase + rank] = index;
    if (rank >= MethodArrivalSlots || MethodArrivalDeposit != 0) EnergyDepositArrival(a, HashRay(index ^ 0x9e3779b9u));
}
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
// Audio paths: one RayPath per selected arrival and FFT bin with pressure gain sqrt(E(f))
// (no phase), the arrival's delay, direction, delay rate and identity.
void EnergyExpand(uint index)
{
    if (index >= MethodArrivalSlots * MethodFrequencyBins) return;
    uint slot = index / MethodFrequencyBins, bin = index % MethodFrequencyBins;
    uint count = min(min(MethodArrivalCounts[MethodCountIndex], MethodArrivalCapacity), MethodArrivalSlots);
    RayPath path = (RayPath)0;
    if (slot < count)
    {
        EnergyArrival a = MethodArrivals[MethodArrivalBase + MethodSelected[MethodSelectedBase + slot]];
        float e = EnergyAt(a, bin * MethodSampleRate / MethodFftSize);
        if (e > 0)
        {
            path.direction = a.direction;
            path.delaySeconds = a.delay;
            path.pressureGain = sqrt(e);
            path.frequencyBin = bin;
            path.identity = a.identity;
        }
    }
    RayPaths[MethodPathBase + index] = path;
}
