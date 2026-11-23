// Method "path": one energy path tracer (docs/benchmarks/path-tracer-2026-11-12).
// One loop per ray, one output: the energy histogram [direction 6][band 8][bin] plus the
// number of contributions per bin. Everything is an unbiased Monte Carlo estimate of the
// room acoustic rendering equation with measured material fractions; noise is left to a
// denoiser (no temporal averaging, no tail extrapolation, no separate stages).
//
//   ray starts at the source (all eight octave bands carried as a vector)
//   loop over vertices:
//     connect the latest non-delta vertex (source or scattering point) to the listener:
//       straight (transmitting through walls), and through one edge, one mirror and one edge,
//       or two parallel edges (UTD energy kernel, edge drawn at random)
//     at the source, with probability methodEdgeProbability the next segment is an edge ray
//       (edge and point importance-sampled by 1/m^2, azimuth uniform on Keller's cone)
//     trace the segment with air absorption; at a surface draw absorb / transmit / scatter /
//       specular by the material's energy fractions (small transmissions are split off into
//       one reservoir branch per ray), Russian roulette as an unbiased terminator
//     after a specular reflection the ray carries a beam: every later segment scores at the
//       listener with a normalised Gaussian kernel of its distance from the listener (the
//       unfolded line against the listener image), whose width is that of the narrowest
//       Gaussian beam spanning the unfolded path, per band (see "Beams" below). The first
//       segment after the source, a scattering point or an edge is counted by the connections.
//   persistent (PathPersistent, see "Persistent rays" below): the same loop, stopped after one update
//     period of path time and resumed at the next update, with new rays launched every update.
//
// Uses MethodCommon.hlsl, with its UTD and edge-sampling helpers.
#ifndef ACOUSTIC_METHOD_PATH
#define ACOUSTIC_METHOD_PATH
// Time bins: PathBin0 wide up to PathFineBins * PathBin0, then constant relative resolution
// (each bin PathRatio times its start time).
float PathBin0, PathRatio;
uint PathFineBins, PathBins, PathCountBase, PathCountCells, PathAudio;
uint PathProfileFlags; // profiling only: skips bit 0 histogram writes, 2 straight connections, 3 edge connections; bit 4 traces every blocked connection;
                      // walk diffraction: bit 17 skips the candidates past a hit, bit 18 the whole candidate search (their share of the time);
                      // diagnostics (docs/benchmarks/path-denoise-2026-11-12, part 6): bit 5 stops rays (and zeroes branches) that refract into a medium
                      // outside a partition, counted as nonFinite; bit 10 draws small transmissions instead of keeping reservoir branches; bit 14 keeps
                      // only unobstructed straight connections; bit 15 traces every walk segment twice (the traversal's share)
// Survival in time (Russian roulette per time bin, weight-compensated, unbiased): a path is alive at
// time t with probability S(t) = min(1, PathSurvivalSeconds / t). The histogram's bins widen in
// proportion to t after the fine region, so with S proportional to 1/t each bin keeps about as many
// contributing paths as a fine early bin, while the work over a horizon T falls from T to
// T0 (1 + ln(T / T0)). Zero disables it.
float PathSurvivalSeconds;
static const uint CounterResumed = 15; // persistent rays continued from the previous update
// The path method's use of counters 7 and 8 (the energy method's detections and chains): the most loop
// iterations of one thread in a dispatch, and their total (the walk's serial chain and its work).
static const uint CounterLongestChain = 7, CounterIterations = 8;
// Counter 13 (the energy method's arrival overflow): rays ended because they were inside a closed solid in
// air, or in a slit (below).
static const uint CounterInsideSolid = 13;
// The orientation sign of the closed object `owner` (Volumes are sorted by owner; as ConsiderMedium uses it:
// a ray leaving the object meets a face with dot(normal, direction) * sign > 0), or 0 for an open object.
float PathVolumeSide(uint owner)
{
    uint lo = 0, hi = VolumeCount;
    [loop] while (lo < hi) { uint mid = lo + (hi - lo) / 2; if (Volumes[mid].ids.z < owner) lo = mid + 1; else hi = mid; }
    return lo < VolumeCount && Volumes[lo].ids.z == owner ? Volumes[lo].minimum.w : 0;
}
// Beams only (settings.pathBeamsOnly, README part 20): no next-event connections at scattering points and no
// edge connections (the source's direct sound stays exact); every arrival is a ray segment that passes the
// listener, scored by the beam kernel, now also the first segment after a scattering point or an edge (the
// connections counted those before). After a scattering point the chain starts there, so the beam's width uses
// the unfolded length from it: a diffuse reflection keeps no phase relation with the path before it, so the
// scattering point is the new source of the Gaussian beam argument (w^2 = lambda L / pi, L from the chain's origin).
uint PathBeamsOnly;
// Persistent rays: past this many loop iterations within one update, every further reflection is survived with
// probability PathIterationSurvival (weight / probability: unbiased); bounds the update's longest serial chains.
uint PathIterationCap; float PathIterationSurvival;
// Persistent rays (see "Persistent rays" below): on/off, pool capacity and offsets, the update period.
uint PathPersistent, PathCapacity, PathReadBase, PathWriteBase, PathReadCount, PathWriteCount, PathOverflowCount, PathFlush;
float PathStep, PathLaunchScale;
// Beams (docs/benchmarks/path-denoise-2026-11-12, part 8; they replaced an exact connection of
// specular chains by detection regions, geometric trials and image retrace): every ray segment after
// at least one specular reflection scores at the listener with a normalised Gaussian kernel of its distance rho
// from the listener (the ray's unfolded line against the listener image), K(rho) = 2 / (pi w^2)
// exp(-2 rho^2 / w^2), whose width per band is that of the narrowest Gaussian beam that spans the
// unfolded path length Lu: w^2 = lambda Lu / pi (waist sqrt(lambda Lu / 2 pi) at mid-path). The first
// segment after a source, a scattering point or an edge is not scored: the exact direct sound and the
// next-event connections count it.
// The listener's clearance R (distance to the nearest surface, from 256 rays; PathClearance, once per
// solve). Within R of the listener every ray that passes a listener image has traversed that image's
// cell (the room's copy in the unfolded space), so the kernel is truncated at R and renormalised: wider,
// it would lose the mass of rays that belong to a neighbouring image.
RWStructuredBuffer<uint> PathClearanceOut;
#ifndef PATH_CLEARANCE_READ
#define PATH_CLEARANCE_READ PathClearanceOut   // (the Doppler walks read a read-only view of it: PathMethod.compute)
#endif
StructuredBuffer<float> PathMirrorCdf; // cumulative primitive areas (RD mirrors drawn by area)
groupshared uint PathClearanceMin;
void PathClearance(uint thread)
{
    if (thread == 0) PathClearanceMin = asuint(1e4f);
    GroupMemoryBarrierWithGroupSync();
    float z = 1 - (2 * thread + 1) / 256.0, r = sqrt(max(0, 1 - z * z)), a = thread * 2.399963229728653;
    float3 d = float3(r * cos(a), z, r * sin(a)), L = States[MethodState + MethodListenerObject].position.xyz;
    uint hit; float distance; float3 normal;
    if (NearestHit(MethodState, L, d, 1e4, false, hit, distance, normal)) InterlockedMin(PathClearanceMin, asuint(distance));
    GroupMemoryBarrierWithGroupSync();
    if (thread == 0) PathClearanceOut[0] = PathClearanceMin;
}
float PathSurvival(float time) { return PathSurvivalSeconds > 0 ? min(1, PathSurvivalSeconds / max(time, 1e-6)) : 1; }
RWStructuredBuffer<uint> PathCounts; // contributions per [solve][bin]

uint PathBinOf(float time)
{
    float fine = PathBin0 * PathFineBins;
    if (time < fine) return (uint)floor(time / PathBin0);
    return PathFineBins + (uint)floor(log(time / fine) / log(1 + PathRatio));
}
// Persistent rays: deposits go to a ring of PathRingSlots future histograms per pair, stored after the
// frames' histograms in MethodOutput (cells) and PathCounts; a contribution heard `ahead` updates after the
// current one goes to slot (PathRingSlot + ahead), and after each update its slot becomes the update's
// histogram (PathRing, mode 0). Mode 1 empties a pair's ring (a flush).
uint PathRingSlots, PathRingBase, PathRingCountBase, PathRingSlot, PathRingCells, PathRingMode, PathRingOffset, PathRingFar;
// Delay rates (Doppler, README part 40). Every contribution carries the rate of change of its travel time with the
// time it is heard, r = dT/dt. To first order r = -v_source . s_emit + v_listener . s_arrive + sum over its events of
// v_P . (s_in - s_out), with s the slowness vector of a leg (its direction over its medium's speed) and v_P the
// velocity of the event's point on its moving surface (a scattering point or an edge point moves with its surface;
// at a specular reflection or a refraction the tangential parts cancel by the reflection law and Snell's, so only the
// normal velocity counts: 2 (v . n) cos theta for a mirror). Exactly, each leg from a point moving at v_a to one
// moving at v_b scales the frequency by (1 - v_b . s) / (1 - v_a . s), so the path carries rho = -ln(product), the sum
// of -ln(1 - v . s) at every receiving end and ln(1 - v . s) at every emitting end (PathReceive, PathEmit; each is
// its first-order term above to first order), and r = 1 - exp(-rho). Per solve and bin the rate channel holds, weighted
// by the contribution's energy (summed over the bands), the first two moments of r and the first of its delay within the
// bin (from the bin's start; persistent rays: referred to the update's own time, tau - r * (when it is heard - that time),
// since an update gathers what is heard within half a period of its time, over which a moving echo's delay drifts): [moment 3][bin], 32.32 signed fixed point, after the histograms and the ring in
// MethodOutput, with a ring of its own (persistent rays). The synthesis renders a bin whose contributions share one
// rate as one echo at that mean delay, sliding at that rate (PathReverb).
// All of it is compiled only into the walk's PATH_DOPPLER kernels (TracePath3DDoppler; MethodTransport dispatches them
// while anything in the scene moves, and for a horizon after): a static scene runs the walk without it, so its outputs stay bit-identical. (A
// first version had the rate code in every walk: no rate was ever deposited in a static scene, yet the grown loop's
// histograms changed in their last bits from run to run, as part 24 found Metal miscompiling this loop as it grows.)
uint PathRateSolveBase, PathRateRingBase, PathRateClearBase, PathRateClearCount;
float PathReceive(float3 velocity, float3 slowness) { return -log(1 - dot(velocity, slowness)); }
float PathEmit(float3 velocity, float3 slowness) { return log(1 - dot(velocity, slowness)); }
void PathAddSigned(uint cell, float value, uint key)
{
    if (!isfinite(value)) { MethodCount(CounterNonFinite); return; }
    float whole = clamp(floor(value), -2147483648.0, 2147483520.0), part = (value - whole) * 4294967296.0 + (HashRay(key ^ HashRay(cell)) >> 8) * (1.0 / 16777216.0);
    if (part >= 4294967296.0) { whole += 1; part -= 4294967296.0; }
    uint low = (uint)part, high = (uint)(int)whole, old;
    InterlockedAdd(MethodOutput[2 * cell], low, old);
    if (old + low < old) high++;
    if (high != 0) InterlockedAdd(MethodOutput[2 * cell + 1], high);
}
// Adds band energies arriving from `from` (listener towards the arrival) at `time`, `ahead` updates from now
// (persistent rays; 0 otherwise). Directions split the energy over the six axes by the squared positive
// components (they sum to one), so a moving arrival never jumps between direction bins. 32.32 fixed point
// with an unbiased dither. `rate`: the contribution's delay rate (the rate channel above); `heard`: when it is heard,
// relative to the time of the update it is added to (persistent rays; 0 otherwise).
#if PATH_DOPPLER
void PathDeposit(float3 from, float4 low, float4 high, float time, uint key, uint ahead, float rate, float heard)
#else
void PathDeposit(float3 from, float4 low, float4 high, float time, uint key, uint ahead)
#endif
{
    if ((PathProfileFlags & 1) != 0) return;
    if (!(time >= 0) || time > MethodHorizon) { MethodCount(CounterNonFinite); return; }
    uint bin = PathBinOf(time);
    if (bin >= PathBins) { MethodCount(CounterNonFinite); return; }
    if (MethodHeadFrame != 0)
    {
        float4 head = States[MethodState + MethodListenerObject].rotation;
        from = Rotate(float4(-head.xyz, head.w), from);
    }
    float norm = length(from);
    float3 u = norm > 0 ? from / norm : float3(0, 0, 1);
    uint slot = (PathRingSlot + ahead) % max(PathRingSlots, 1u);
    uint cellBase = PathPersistent != 0 ? PathRingBase + slot * PathRingCells : MethodCellBase;
    InterlockedAdd(PathCounts[(PathPersistent != 0 ? PathRingCountBase + slot * PathBins : PathCountBase) + bin], 1u);
#if PATH_DOPPLER
    {
        // (The delay moment covers every contribution, moving or not, so that a bin's mean delay is over all its energy;
        // the rate moments of a contribution at rest are zero.)
        float energy = dot(low + high, 1);
        uint rates = (PathPersistent != 0 ? PathRateRingBase + slot * 3 * PathBins : PathRateSolveBase) + bin;
        float fine = PathBin0 * PathFineBins, start = bin < PathFineBins ? bin * PathBin0 : fine * pow(1 + PathRatio, (float)(bin - PathFineBins));
        if (rate != 0)
        {
            PathAddSigned(rates, energy * rate, key ^ 0x3c6ef372u);
            PathAddSigned(rates + PathBins, energy * rate * rate, key ^ 0xa54ff53au);
        }
        PathAddSigned(rates + 2 * PathBins, energy * (time - rate * heard - start), key ^ 0x510e527fu);
    }
#endif
    [loop] for (uint axis = 0; axis < 6; axis++)
    {
        float component = (axis & 1) != 0 ? -u[axis >> 1] : u[axis >> 1];
        if (component <= 0) continue;
        [loop] for (uint k = 0; k < MethodBands; k++)
        {
            float4 lane = float4((k & 3) == uint4(0, 1, 2, 3));
            float share = dot(k < 4 ? low : high, lane) * component * component;
            if (!(share > 0) || !isfinite(share)) { if (!isfinite(share)) MethodCount(CounterNonFinite); continue; }
            uint cell = cellBase + (axis * MethodBands + k) * PathBins + bin;
            float whole = floor(share), part = (share - whole) * 4294967296.0 + (HashRay(key ^ HashRay(cell)) >> 8) * (1.0 / 16777216.0);
            if (part >= 4294967296.0) { whole += 1; part -= 4294967296.0; }
            uint low = (uint)part, high = (uint)min(whole, 4294967295.0), old;
            InterlockedAdd(MethodOutput[2 * cell], low, old);
            if (old + low < old) high++;
            if (high != 0) InterlockedAdd(MethodOutput[2 * cell + 1], high);
        }
    }
}

// The state of a chain: the straight (unfolded) run of delta events since the latest
// non-delta vertex (source, scattering point, or the edge point of an edge ray).
struct PathChain
{
    uint kind;                // 0 source, 1 scattering point, 2 edge ray from the source
    uint edge; float edgePhi, edgePick;
    uint hits, reflections;   // events, of which reflections
    float travelled, time;    // unfolded length since origin; time at origin
};
PathChain PathStart(uint kind, float time)
{
    PathChain c;
    c.kind = kind; c.edge = 0; c.edgePhi = 0; c.edgePick = 1;
    c.hits = 0; c.reflections = 0; c.travelled = 0; c.time = time;
    return c;
}
// One event of a chain: transmitted or reflected.
void PathEvent(inout PathChain c, bool transmitted)
{
    if (!transmitted) c.reflections++;
    c.hits++;
}

// The solve's edge distribution (edge rays from the source and the source's edge connections draw
// from it): edge e has weight angle_e x (integral of 1/m^2 along it from the source) x (the same
// from the listener) x 0.1 for each end that a quick test does not see (and for a source outside
// the wedge, which an image source may still reach). Every edge keeps a positive weight, so any
// path is still sampled (unbiased); the weight only steers samples to edges between the two ends.
// EnergyEdgeWeights (energy method) weighted by the source alone, which left the edges of an open
// doorway between the organ and a listener outside almost never drawn.
// Persistent rays keep each pair's weights (PathEdgeWeight) and refresh a slice of them per update
// (PathEdgeOffset); PathEdgeCopy then makes the update's distribution from them. Any positive weights
// give an unbiased walk (a sample's probability is read from the same distribution it was drawn from),
// so weights a few updates old only steer a little less well.
RWStructuredBuffer<float> PathEdgeWeight; // persistent: [pair][edge]
uint PathEdgeOffset, PathEdgeWeightBase;
void PathEdgeCopy(uint edge) { if (edge < MethodEdgeCount) MethodEdgeCdf[edge] = PathEdgeWeight[PathEdgeWeightBase + edge]; }
void PathEdgeWeights(uint index)
{
    uint edge = index + PathEdgeOffset;
    if (edge >= MethodEdgeCount) return;
    float3 S = States[MethodState + MethodSourceObject].position.xyz, L = States[MethodState + MethodListenerObject].position.xyz;
    float3 a, u, f, nn; float extent, angle, z0, r;
    EdgeFrame(edge, MethodState, false, a, u, f, nn, extent, angle);
    float3 mid = a + .5 * extent * u;
    float weight = angle * (TunedEdgeAngle(S - mid, f, nn, angle) <= angle ? 1 : MethodEdgeHidden);
    [loop] for (uint end = 0; end < 2; end++)
    {
        float3 x = end == 0 ? S : L;
        weight *= EnergyEdgeSpan(a, u, extent, x, z0, r) / r;
        float3 nearest = a + clamp(z0, 0, extent) * u;
        bool seen = false;
        [loop] for (uint i = 0; i < 2 && !seen; i++) seen = Visible(MethodState, x, i == 0 ? nearest : mid, false);
        weight *= seen ? 1 : MethodEdgeHidden;
    }
    if (PathPersistent != 0) PathEdgeWeight[PathEdgeWeightBase + edge] = weight; else MethodEdgeCdf[edge] = weight;
}

// Straight connection of a non-delta vertex to the listener (through transmitting walls).
// Unless `exact` (the source's direct sound), a connection blocked by a surface is traced through
// the walls only with probability q = that surface's largest energy transmission over the bands, and
// weighted by 1 / q (Russian roulette, unbiased): the transmitted energy is at most that factor, so
// the weight stays bounded, and in stone rooms (t of 1e-6) the crossing loop is almost never run.
bool PathListener(MethodAnchor v, float I[8], float3 L, float scale, bool exact, inout uint rng, out float4 low, out float4 high, out float time, out float3 from, out float lengthOut)
{
    low = 0; high = 0;
    time = 0; from = 0; lengthOut = 0;
    float3 w = L - v.position;
    float d = length(w);
    if (d <= RayEpsilon) return false;
    w /= d;
    float D = EnergyDirectivity(v, w);
    if (D == 0) return false;
    float factor[8]; float t, keep = 1;
    [unroll] for (uint k = 0; k < MethodBands; k++) factor[k] = 1;
    bool clear = false;
    if (!exact && MethodAnyTransmission != 0 && (PathProfileFlags & 16) == 0) // (bit 4: always trace, a check of this roulette)
    {
        uint hit; float distance; float3 normal;
        if (NearestHit(MethodState, v.position, w, d - RayEpsilon, false, hit, distance, normal))
        {
            keep = 0;
            [unroll] for (uint k2 = 0; k2 < MethodBands; k2++)
            {
                RayMaterial m = MethodSurface(hit, k2);
                keep = max(keep, m.medium.w != 0 ? 0 : saturate((1 - m.surface.x) * m.surface.z));
            }
            if (!(RandomRay(rng) < keep)) return false;
        }
        else clear = true; // nothing in between: one medium, no crossing loop needed
    }
    if ((PathProfileFlags & 16384) != 0 && !exact && !clear) return false; // (bit 14: unobstructed connections only)
    if (clear)
    {
        t = d * MethodSlowness(v.medium);
        [unroll] for (uint k3 = 0; k3 < MethodBands; k3++) factor[k3] = exp(-2 * MethodLoss(v.medium, k3) * d);
    }
    else if (!MethodConnect(v.position, L, v.medium, true, factor, t)) return false;
    float r = max(MinimumDistance, d);
    float e[8];
    [unroll] for (uint k1 = 0; k1 < MethodBands; k1++) e[k1] = I[k1] * D * factor[k1] / (r * r) * scale / keep;
    low = float4(e[0], e[1], e[2], e[3]); high = float4(e[4], e[5], e[6], e[7]);
    time = v.time + t; from = -w; lengthOut = d;
    return true;
}
// Mirror leg start -> primitive -> end (CompleteMirror without its scan of every lower-index
// primitive of the object, which made each RD connection cost a pass over the whole scene): a hit
// on the triangle's boundary, where a neighbour could claim the same contact, is rejected instead
// (a set of measure zero, so no bias).
bool PathMirror(uint primitive, float3 start, float3 end, out float3 hit, out float cosine)
{
    hit = 0; cosine = 0;
    float3 n, unused;
    float3 image = CompleteImage(primitive, MethodState, start, false, n), delta = end - image;
    float lengthTo = length(delta);
    if (lengthTo <= RayEpsilon) return false;
    float3 direction = delta / lengthTo;
    float distance = PrimitiveDistance(primitive, MethodState, image, direction, false, unused);
    if (distance <= RayEpsilon || distance >= lengthTo - RayEpsilon) return false;
    hit = image + distance * direction;
    Primitive t = Primitives[primitive]; State pose = States[MethodState + t.indices.x];
    float3 a = pose.position.xyz + Rotate(pose.rotation, t.a.xyz), b = pose.position.xyz + Rotate(pose.rotation, t.b.xyz),
           c = pose.position.xyz + Rotate(pose.rotation, t.c.xyz);
    float3 m = cross(b - a, c - a);
    float area = dot(m, m);
    float3 w = float3(dot(cross(c - b, hit - b), m), dot(cross(a - c, hit - c), m), dot(cross(b - a, hit - a), m)) / max(area, 1e-20);
    if (min(w.x, min(w.y, w.z)) <= 1e-5) return false;
    [loop] for (uint leg = 0; leg < 2; leg++)
        if (!Visible(MethodState, leg == 0 ? start : hit, leg == 0 ? hit : end, false)) return false;
    cosine = abs(dot(normalize(hit - start), n));
    return true;
}
// UtdEnergy with its reflection-boundary terms (the two with phiOut + phiIn) weighted by
// 1 - exp(-2 X s'/s), X = k L a their transition argument and s, s' the source- and listener-side
// distances. Near a reflection boundary the beams (see "Beams") already smooth the reflected field
// over the beam width w, w^2 = lambda (s + s') / pi; a listener offset x from the boundary has
// 2 x^2 / w^2 = 2 X s'/s, so the weight is one minus the beam kernel at that offset: where the beam
// reaches past the boundary UTD's transition is not added again, far beyond it UTD is unchanged.
// The incident-boundary terms (phiOut - phiIn) stay: the direct sound is exact, not a beam.
float PathUtdEnergy(float n, float phiIn, float phiOut, float sinBeta, float k, float distance, float ratio)
{
    float2 sum = 0;
    [unroll] for (uint i = 0; i < 4; i++)
    {
        float sign = (i & 1) == 0 ? 1 : -1, x = i < 2 ? phiOut - phiIn : phiOut + phiIn;
        float argument = (RayPi + sign * x) / (2 * n);
        if (abs(sin(argument)) < 1e-5) { x += sign * 4e-5 * n; argument = (RayPi + sign * x) / (2 * n); }
        float N = round((x + sign * RayPi) / (2 * RayPi * n)), c = cos(n * RayPi * N - .5 * x), X = k * distance * 2 * c * c;
        float weight = i < 2 || ratio <= 0 ? 1 : 1 - exp(-2 * X * ratio); // ratio 0: no beam to compare with
        sum += weight * (cos(argument) / sin(argument)) * UtdTransition(X);
    }
    return dot(sum, sum) / (8 * RayPi * k * n * n * sinBeta * sinBeta);
}
// Listener connection through one edge (D), one mirror and one edge (RD), or two parallel
// edges (DD): EnergyEdgeConnection's kernels, with one UtdEnergy body per edge (bands in a loop).
// rate: the connection's delay rate, from the anchor's (vRate: up to and including its incoming term; vVelocity: its
// point's velocity) through the mirror and the edges (each point moving with its object) to the listener (vL).
#if PATH_DOPPLER
bool PathEdgeConnection(MethodAnchor v, float4 inLow, float4 inHigh, float3 L, float vRate, float3 vVelocity, float3 vL, inout uint rng, out float4 low, out float4 high, out float time, out float3 fromOut, out float rate)
{
    low = 0; high = 0;
    time = 0; fromOut = 0; rate = 0;
#else
bool PathEdgeConnection(MethodAnchor v, float4 inLow, float4 inHigh, float3 L, inout uint rng, out float4 low, out float4 high, out float time, out float3 fromOut)
{
    low = 0; high = 0;
    time = 0; fromOut = 0;
#endif
    if (MethodEdgeCount == 0 || DiffractionFactor <= 0) return false;
    float pick, pick2 = 1;
    // From the source, draw the edge from the solve's distribution (PathEdgeWeights: edges between
    // source and listener) instead of uniformly over every edge of the scene: in the Church (14,075
    // edges) a uniform pick found the edges of the doorway between the organ and a listener outside
    // in one seed out of three.
    // Scattering points mix that distribution in too (half the time; the other half the local/uniform
    // pick), with the mixture's probability: in the Church a uniform pick of the doorway's edges from
    // points in the nave made the diffracted sound outside heavy-tailed (one solve's median a tenth of
    // the mean). Same estimator, lower variance.
    uint edge, edge2;
    float uSource = RandomRay(rng);
    bool distribution = METHOD_EDGE_CDF_READ[max(MethodEdgeCount, 1u) - 1] > 0;
    if (v.kind == 0 && distribution) edge = EnergySampleEdge(uSource, pick);
    else if (v.kind == 1 && distribution)
    {
        float pickCdf, pickLocal;
        if (uSource < .5) edge = EnergySampleEdge(2 * uSource, pickCdf);
        else edge = MethodPickEdge(v.primitive, rng, pickLocal);
        float total = METHOD_EDGE_CDF_READ[MethodEdgeCount - 1];
        pickCdf = (METHOD_EDGE_CDF_READ[edge] - (edge > 0 ? METHOD_EDGE_CDF_READ[edge - 1] : 0)) / total;
        uint2 local = v.primitive == 0xffffffff ? uint2(0, 0) : EdgeRanges[v.primitive];
        pickLocal = (local.y > 0 ? .5 : 1) / MethodEdgeCount;
        [loop] for (uint i = 0; i < local.y; i++) if (EdgeIndices[local.x + i] == edge) pickLocal += .5 / local.y;
        pick = .5 * pickCdf + .5 * pickLocal;
    }
    else edge = MethodPickEdge(v.kind == 1 ? v.primitive : 0xffffffff, rng, pick);
    edge2 = edge;
    float r = RandomRay(rng);
    bool mirrors = PrimitiveCount > 0;
    bool image = mirrors && r < 1.0 / 3, pair = r >= (mirrors ? 2.0 / 3 : .5);
    float variant = mirrors ? 1.0 / 3 : .5;
    uint mirror = 0;
    if (image)
    {
        // The mirror in proportion to its area (PathMirrorCdf), with that probability.
        float total = PathMirrorCdf[PrimitiveCount - 1], target = RandomRay(rng) * total;
        uint lo = 0, hi = PrimitiveCount - 1;
        [loop] while (lo < hi) { uint mid = (lo + hi) / 2; if (PathMirrorCdf[mid] > target) hi = mid; else lo = mid + 1; }
        mirror = lo;
        float area = PathMirrorCdf[lo] - (lo > 0 ? PathMirrorCdf[lo - 1] : 0);
        if (!(area > 0)) return false;
        variant *= area / total;
    }
    float3 from = v.position, unused, e, e2 = 0, first, hit;
    if (image)
    {
        if ((States[MethodState + Primitives[mirror].indices.x].flags.x & 3) != 3) return false;
        from = CompleteImage(mirror, MethodState, v.position, false, unused);
    }
    float n, phiIn, phiOut, sinBeta, n2 = 1, phiIn2 = 0, phiOut2 = 0, sinBeta2 = 1, cosine = 0, leg = 1;
    if (pair)
    {
        edge2 = MethodPickNextEdge(edge, rng, pick2);
        if (!UtdApexPair(edge, edge2, from, L, e, e2) || !UtdAngles(edge, from, e, e2, n, phiIn, phiOut, sinBeta) ||
            !UtdAngles(edge2, e, e2, L, n2, phiIn2, phiOut2, sinBeta2)) return false;
        leg = MethodEdgeLeg(edge, edge2, e, e2);
        if (leg == 0) return false;
    }
    else if (!UtdApex(edge, from, L, e) || !UtdAngles(edge, from, e, L, n, phiIn, phiOut, sinBeta)) return false;
    float3 last = pair ? e2 : e;
    first = e;
    if (image) { if (!PathMirror(mirror, v.position, e, hit, cosine)) return false; first = hit; }
    [loop] for (uint segment = image ? 1u : 0u; segment < 2; segment++)
        if (!Visible(MethodState, segment == 0 ? v.position : last, segment == 0 ? e : L, false)) return false;
    float D = EnergyDirectivity(v, normalize(first - v.position));
    float m = length(e - from), d = pair ? length(e2 - e) : 0, l = length(L - last), total = m + d + l;
    if (D == 0 || m <= RayEpsilon || l <= RayEpsilon) return false;
    MethodCount(CounterEdgeConnections);
    time = v.time + total * MethodSlowness(AirMedium);
    float s2 = sinBeta * sinBeta;
    float4 sumLow = 0, sumHigh = 0; // see PathDetect
    [loop] for (uint k = 0; k < MethodBands; k++)
    {
        float opacity = MethodEdgeOpacity(edge, k);
        float4 lane = float4((k & 3) == uint4(0, 1, 2, 3)); // one-hot: no dynamic vector writes in a loop
        float share = PathUtdEnergy(n, phiIn, phiOut, sinBeta, MethodWavenumber(k), m * (d + l) / total * s2, pair ? 0 : l / m) * dot(k < 4 ? inLow : inHigh, lane) * D *
                      DiffractionFactor * DiffractionFactor * opacity * opacity;
        if (pair)
        {
            float opacity2 = MethodEdgeOpacity(edge2, k);
            share *= UtdEnergy(n2, phiIn2, phiOut2, sinBeta2, MethodWavenumber(k), (m + d) * l / total * s2) * leg * leg *
                     DiffractionFactor * DiffractionFactor * opacity2 * opacity2 / (d * pick2);
        }
        share *= exp(-2 * MethodLoss(AirMedium, k) * total) / (m * l * total * pick * variant);
        if (image) share *= EnergySpecular(MethodSurface(mirror, k), cosine);
        if (k < 4) sumLow += share * lane; else sumHigh += share * lane;
    }
    low = sumLow; high = sumHigh;
    fromOut = normalize(last - L);
#if PATH_DOPPLER
    // Legs: anchor -> (mirror) -> edge -> (second edge) -> listener.
    float3 w1 = normalize(first - v.position), w2 = image ? normalize(e - hit) : w1, w3 = pair ? normalize(e2 - e) : -fromOut;
    float slowness = MethodSlowness(AirMedium);
    float3 vEdge = PointVelocity(MethodState + Edges[edge].ids.x, e);
    rate = vRate + PathEmit(vVelocity, w1 * slowness) + PathReceive(vEdge, w2 * slowness) + PathEmit(vEdge, w3 * slowness) + PathReceive(vL, -fromOut * slowness);
    if (image) { float3 vMirror = PointVelocity(MethodState + Primitives[mirror].indices.x, hit); rate += PathReceive(vMirror, w1 * slowness) + PathEmit(vMirror, w2 * slowness); }
    if (pair) { float3 vEdge2 = PointVelocity(MethodState + Edges[edge2].ids.x, e2); rate += PathReceive(vEdge2, w3 * slowness) + PathEmit(vEdge2, -fromOut * slowness); }
#endif
    return true;
}

// Audio: ray 0's direct connection is written as the solve's one exact arrival (retarded delay
// and Doppler of the moving ends, as EnergyArrivals) instead of into the histogram.
void PathDirectArrival(float3 S, float3 L, float4 low, float4 high, float scale, float time, float total)
{
    float c = SpeedOfSound;
    float3 vS = Velocity(MethodState + MethodSourceObject), vL = Velocity(MethodState + MethodListenerObject);
    float3 departure = total > 0 ? (L - S) / total : float3(0, 0, 1);
    float a = c * c - dot(vS, vS), b = total * dot(vS, departure);
    float retarded = a > 0 ? (b + sqrt(max(0, b * b + a * total * total))) / a : total / c;
    float3 emitted = normalize(total * departure + vS * retarded);
    float ratio = (c - dot(vL, emitted)) / (c - dot(vS, emitted));
    EnergyArrival result;
    result.direction = float4(-emitted, 1 - ratio);
    result.low = low / scale;
    result.high = high / scale;
    result.delay = time + retarded - total / c;
    result.identity = 0x2545f491u;
    result.family = EnergyFamilyDirect;
    result.length = total;
    if (!(result.delay >= 0 && result.delay <= MethodHorizon && ratio > 0) || !(dot(result.low + result.high, 1) > 0)) return;
    MethodArrivals[MethodArrivalBase] = result;
    MethodArrivalCounts[MethodCountIndex] = 1;
}

// Persistent rays (docs/benchmarks/path-denoise-2026-11-12, part 17): the rays are the sound in flight.
// Their state stays in PathRays across updates; every update advances each live ray by one update
// period of path time (PathStep; it stops mid-segment and resumes there), launches MethodRayCount new
// rays from the source's current position, and scores what the rays meet during the update against the
// listener's current position, at delay = path time (+ the connection's travel time). A contribution is
// added to the response of the update in which it is heard (the ring below): a next-event connection made
// now through a door that just opened reaches the listener after its travel time, as the sound would. A cohort launched
// at t0 is an unbiased estimate from the source at t0, so the update's response at delay tau comes from
// the cohort launched about tau earlier: the retarded-time response of a moving scene, with doors met in
// their state when the sound reaches them. A static scene's response after one horizon of updates equals
// the full solve's in expectation (each contribution of a ray is made once, in the update during which
// the ray gets there). Branch reservoirs (transmitted shares) are kept per update: a ray still holding one
// when it stops queues it as a ray of its own (after the same roulette the full solve applies), so the
// branch's contributions come at most one update late; at the ray's end the branch continues in its place,
// as in the full solve. PathPersistent = 0 is the full solve (every ray traced to its end in one update).
struct PathRay // float4 fields only (160 bytes), so that every backend lays it out alike
{
    float4 position;              // xyz, w path time
    float4 direction;             // xyz, w the cohort's age at the end of the latest update
    float4 phiLow, phiHigh;       // Phi0: the flux of the current chain's origin
    float4 wLow, wHigh;           // C / Q: attenuation since then over the roulette's survival
    float4 plane;                 // the latest event's plane, outgoing side (beams): normal, offset
    float4 scalars;               // survival, chain's unfolded length, chain's origin time, distance since the segment's start
    uint4 meta;                   // reflections | flags << 24 (1 in a branch, 2 the segment's beam is scored, 4 at a surface: trace from zero distance) | chain kind << 27, medium, rng, events
    float4 motion;                // x: the delay rate accumulated so far (the rate channel); yzw walk diffraction's walkPhiIn, walkM, walkMeta
};
RWStructuredBuffer<PathRay> PathRays;  // [pair][pool 2][PathCapacity]
RWStructuredBuffer<uint> PathLive;     // [pair][pool 2]: rays in each pool; then [pair]: rays dropped by overflow, ever
// Before an update: the pool it writes starts empty (and with PathFlush, e.g. at a scene cut, so does the one it reads).
// Population control (settings.pathPopulation, README part 20): the rays in flight are held at N by
// cohort. After the walk (which then applies no survival in time itself), cohort k (age a) is brought to its
// target n S(a), S(a) = min(1, T_f / a), T_f the end of the histogram's fine bins, by splitting each of its A rays
// into floor(rho) or floor(rho) + 1 copies (rho = n S(a) / A, at most PathSplitLimit) or, for rho < 1, by keeping
// it with probability rho; every kept copy's weight is divided by rho, so the expected weight of the cohort is
// unchanged (unbiased), whatever A was. n = PathCohortTarget new rays are launched every update.
RWStructuredBuffer<uint> PathCohortCount; // [pair][cohort]: rays kept by the walk this update, by age
uint PathPopulation, PathCohortBase, PathCohorts, PathCohortStart;
float PathCohortTarget, PathFineEnd;
static const float PathSplitLimit = 16; // a cost guard on one ray's copies (any limit keeps the weights exact)
uint PathCohortOf(float age) { return min(PathCohorts - 1, (uint)max(0, round(age / PathStep + .5))); }
float PathCohortWanted(float age) { return PathCohortTarget * min(1, PathFineEnd / max(age, 1e-6)); }
void PathPersistentBegin()
{
    PathLive[PathWriteCount] = 0;
    if (PathFlush != 0) PathLive[PathReadCount] = 0;
    if (PathPopulation != 0) { [loop] for (uint k = 0; k < PathCohorts; k++) PathCohortCount[PathCohortBase + k] = 0; }
}
// Population control, after the walk: the walk wrote its rays to the scratch pool (PathWriteBase); they go back
// to the pool the next update reads (PathReadBase), split or kept by lottery. A ray's copies and the divisor of its
// weight (rho) depend only on the ray and its cohort's count, so the passes below recompute them alike.
uint PathCopies(PathRay r, out float rho, out uint rng)
{
    float age = r.direction.w;
    rho = min(PathCohortWanted(age) / max(1, PathCohortCount[PathCohortBase + PathCohortOf(age)]), PathSplitLimit);
    rng = HashRay(r.meta.z ^ 0x27d4eb2fu);
    return (uint)floor(rho) + (RandomRay(rng) < frac(rho) ? 1u : 0u);
}
// Coherence (README part 31): the kept copies are written by their direction's octant, then in Morton order of their
// position's cell, by a counting sort fused with population control (count, scan, scatter, in place of an atomic
// append), so that neighbouring threads of the next update's walk trace neighbouring rays in similar directions and
// visit the same tree nodes in the same order (the octant first: it was 0.5 ms faster than the cell first). Only the order of the pool changes, so every estimate is unchanged in distribution.
// Cells: 2^PathSortBits per axis, the quantiles of the rays' positions along that axis at the previous update (equal
// numbers of rays per slab), read from a histogram of PathSortFine bins over the scene's box (PathSortLow, PathSortScale:
// bins per metre). A box of the geometry alone would not do: the Church's meadow makes its box ten times the church's.
// ProfileFlags bit 16 gives every ray key 0 (the atomic order, as before).
static const uint PathSortBits = 4, PathSortKeys = 1u << (3 * PathSortBits + 3), PathScanThreads = 1024, PathSortFine = 1024;
RWStructuredBuffer<uint> PathSortBuckets; // [pair]: copies per key [PathSortKeys] (zero between updates), each key's first slot [PathSortKeys],
                                          // then per axis the copies per fine bin [PathSortFine] (zero between updates) and each fine bin's cell [PathSortFine]
RWStructuredBuffer<uint2> PathSortEntry;  // [pair][PathCapacity]: per ray of the scratch pool, its key and copies (key | copies << 24) and its
                                          // first copy's offset in the key
uint PathSortBase, PathSortEntryBase;
float4 PathSortLow, PathSortScale;
uint PathSpread(uint v) // bits of a PathSortBits-bit value two apart (Morton interleave)
{
    v &= 0x3ff;
    v = (v | (v << 16)) & 0x030000ffu; v = (v | (v << 8)) & 0x0300f00fu;
    v = (v | (v << 4)) & 0x030c30c3u; v = (v | (v << 2)) & 0x09249249u;
    return v;
}
uint PathSortAxis(uint axis) { return PathSortBase + 2 * PathSortKeys + axis * 2 * PathSortFine; }
// Pass 1, per ray of the scratch pool: its copies, counted under its key (the count before them is its offset) and in
// the fine bins of its position (the next update's cells).
void PathSortCount(uint index)
{
    if (index >= min(PathLive[PathWriteCount], PathCapacity)) return;
    PathRay r = PathRays[PathWriteBase + index];
    float rho; uint rng;
    uint copies = PathCopies(r, rho, rng), key = 0, offset = 0;
    if (copies > 0 && (PathProfileFlags & 65536) == 0)
    {
        uint3 fine = (uint3)clamp(floor((r.position.xyz - PathSortLow.xyz) * PathSortScale.xyz), 0.0, (float)(PathSortFine - 1)), cell = 0;
        [unroll] for (uint axis = 0; axis < 3; axis++)
        {
            InterlockedAdd(PathSortBuckets[PathSortAxis(axis) + fine[axis]], copies);
            cell[axis] = PathSortBuckets[PathSortAxis(axis) + PathSortFine + fine[axis]];
        }
        uint octant = (r.direction.x < 0 ? 1u : 0u) | (r.direction.y < 0 ? 2u : 0u) | (r.direction.z < 0 ? 4u : 0u);
        key = octant << (3 * PathSortBits) | PathSpread(cell.x) | PathSpread(cell.y) << 1 | PathSpread(cell.z) << 2;
    }
    if (copies > 0) InterlockedAdd(PathSortBuckets[PathSortBase + key], copies, offset);
    PathSortEntry[PathSortEntryBase + index] = uint2(key | copies << 24, offset);
}
// Pass 2, one group: each key's first slot (an exclusive scan of the counts), the pool's new count, and per axis the
// next update's cells (fine bin -> the quantile its middle ray falls in); every count is emptied for the next update.
groupshared uint PathScanTotal[PathScanThreads];
uint PathScanInclusive(uint thread, uint value)
{
    GroupMemoryBarrierWithGroupSync(); // (the previous scan's reads are done)
    PathScanTotal[thread] = value;
    GroupMemoryBarrierWithGroupSync();
    [loop] for (uint step = 1; step < PathScanThreads; step <<= 1)
    {
        uint before = thread >= step ? PathScanTotal[thread - step] : 0;
        GroupMemoryBarrierWithGroupSync();
        PathScanTotal[thread] += before;
        GroupMemoryBarrierWithGroupSync();
    }
    return PathScanTotal[thread];
}
void PathSortScan(uint thread)
{
    const uint per = PathSortKeys / PathScanThreads;
    uint base = PathSortBase + thread * per, sum = 0;
    [loop] for (uint i = 0; i < per; i++) sum += PathSortBuckets[base + i];
    uint running = PathScanInclusive(thread, sum) - sum;
    if (thread == PathScanThreads - 1) PathLive[PathReadCount] = min(running + sum, PathCapacity);
    [loop] for (uint j = 0; j < per; j++)
    {
        uint count = PathSortBuckets[base + j];
        PathSortBuckets[base + PathSortKeys + j] = running;
        PathSortBuckets[base + j] = 0;
        running += count;
    }
    [loop] for (uint axis = 0; axis < 3; axis++) // (PathSortFine == PathScanThreads: one fine bin per thread)
    {
        uint at = PathSortAxis(axis) + thread, count = PathSortBuckets[at], upTo = PathScanInclusive(thread, count), total = PathScanTotal[PathScanThreads - 1];
        if (total > 0) PathSortBuckets[at + PathSortFine] = min((1u << PathSortBits) - 1, (uint)(((upTo - .5 * count) / total) * (1u << PathSortBits)));
        PathSortBuckets[at] = 0;
    }
}
// Pass 3, per ray of the scratch pool: its copies, each with its own random sequence and the weight divided by rho,
// written from its key's first slot plus its offset. Copies past the pool's capacity are dropped and counted.
void PathSortScatter(uint index)
{
    if (index >= min(PathLive[PathWriteCount], PathCapacity)) return;
    PathRay r = PathRays[PathWriteBase + index];
    float rho; uint rng;
    uint2 entry = PathSortEntry[PathSortEntryBase + index];
    uint copies = entry.x >> 24, key = entry.x & 0xffffff;
    if (copies == 0) return;
    PathCopies(r, rho, rng); // (the weight's divisor and the random sequence)
    uint first = PathSortBuckets[PathSortBase + PathSortKeys + key] + entry.y;
    r.phiLow /= rho; r.phiHigh /= rho;
    [loop] for (uint copy = 0; copy < copies; copy++)
    {
        r.meta.z = HashRay(rng + copy * 0x9e3779b9u);
        if (first + copy < PathCapacity) PathRays[PathReadBase + first + copy] = r;
        else { MethodCount(CounterOverflow); InterlockedAdd(PathLive[PathOverflowCount], 1u); }
    }
}
// Appends a ray to the pool being written (overflow is counted and the ray dropped: a bias, reported).
void PathKeep(PathRay r)
{
    uint slot;
    InterlockedAdd(PathLive[PathWriteCount], 1u, slot);
    if (slot < PathCapacity) PathRays[PathWriteBase + slot] = r;
    else { MethodCount(CounterOverflow); InterlockedAdd(PathLive[PathOverflowCount], 1u); }
    if (PathPopulation != 0 && slot < PathCapacity) InterlockedAdd(PathCohortCount[PathCohortBase + PathCohortOf(r.direction.w)], 1u);
}

// PartitionExit for the walk, where overlapping solids are one: the crossing ends at the partition's far
// face only once every other object entered on the way has been left too (an even number of their faces
// crossed). The Church's door frame runs 8 mm into the floor slab and its stone mullions through the window
// panes; ending at the frame's far face put transmitted rays inside the floor slab, in air, where they
// bounced between faces 8 mm apart for hundreds of events per update (README part 18).
bool PathPartitionExit(uint hit, float3 position, float3 direction, bool medium, out float depth)
{
    uint owner = Primitives[hit].indices.x, foreign = 0;
    float travelled = 2 * RayEpsilon;
    bool left = false;
    depth = 0;
    if (medium) return false;
    [loop] for (uint i = 0; i < 8; i++)
    {
        uint next; float distance; float3 normal;
        if (!NearestHit(MethodState, position + direction * travelled, direction, PartitionDepth - travelled, false, next, distance, normal)) return false;
        travelled += distance;
        if (Primitives[next].indices.x == owner) left = true; else foreign++;
        if (left && (foreign & 1) == 0) { depth = travelled; return true; }
        travelled += 2 * RayEpsilon;
        if (travelled >= PartitionDepth) return false;
    }
    return false;
}

// Mode 0: the current slot of the pair's ring becomes the update's histogram (MethodCellBase, PathCountBase)
// and is emptied for reuse; mode 1: every slot of the pair's ring is emptied.
void PathRing(uint index)
{
    uint i = index + PathRingOffset;
    if (PathRingMode == 0)
    {
        if (i >= PathRingCells) return;
        uint from = PathRingBase + PathRingSlot * PathRingCells + i, to = MethodCellBase + i;
        MethodOutput[2 * to] = MethodOutput[2 * from]; MethodOutput[2 * to + 1] = MethodOutput[2 * from + 1];
        MethodOutput[2 * from] = 0; MethodOutput[2 * from + 1] = 0;
        if (i < PathBins)
        {
            uint count = PathRingCountBase + PathRingSlot * PathBins + i;
            PathCounts[PathCountBase + i] = PathCounts[count]; PathCounts[count] = 0;
        }
        if (i < 3 * PathBins)
        {
            uint rate = PathRateRingBase + PathRingSlot * 3 * PathBins + i, solve = PathRateSolveBase + i;
            MethodOutput[2 * solve] = MethodOutput[2 * rate]; MethodOutput[2 * solve + 1] = MethodOutput[2 * rate + 1];
            MethodOutput[2 * rate] = 0; MethodOutput[2 * rate + 1] = 0;
        }
        return;
    }
    if (i >= PathRingSlots * PathRingCells) return;
    MethodOutput[2 * (PathRingBase + i)] = 0; MethodOutput[2 * (PathRingBase + i) + 1] = 0;
    if (i < PathRingSlots * PathBins) PathCounts[PathRingCountBase + i] = 0;
    if (i < PathRingSlots * 3 * PathBins) { MethodOutput[2 * (PathRateRingBase + i)] = 0; MethodOutput[2 * (PathRateRingBase + i) + 1] = 0; }
}

#if PATH_WALK_EDGE
// Walk diffraction (settings.pathWalkDiffraction; docs/benchmarks/bass-and-diffraction-2026-11-22/README-part-b.md): edge
// diffraction as an event of the walk, in place of the source's edge rays and every edge connection.
//  * Deposit. A ray piece of flux Phi whose line passes an edge at offset rho and angle gamma deposits Phi K1(rho) / sin(gamma)
//    at its closest point on the edge, K1 = sqrt(2 / (pi w^2)) exp(-2 rho^2 / w^2) the beam kernel's 1-D marginal,
//    w^2 = lambda L / pi, L the chain's unfolded length to that point: summed over rays, the incident intensity per unit
//    edge length. The whole kernel counts: a ray that hits a plane containing one of the edge's faces before its closest
//    approach deposits with its line extended past the hit (those edges are listed per plane, PathPlaneEdge*), and a ray
//    whose latest event was on such a plane deposits nothing on that edge (UTD's reflection-boundary terms already
//    diffract the reflected field).
//  * Grazing. With the chain's origin on one of the edge's faces only the rays on its open side exist: the deposit
//    counts twice (the rigid face's image) and |D|^2 a quarter (Kouyoumjian and Pathak's grazing factor 1/2).
//  * The diffracted ray leaves the closest point on Keller's cone of the incidence from the chain's origin (p - d *
//    travelled: the source, a scattering point, an edge, or their image), its azimuth uniform over the exterior angle,
//    with flux deposit x angle and no D yet. Its first segment is scored at the listener by the beam kernel (width
//    from the edge) times |D|^2 sin^2(beta) at the listener's own azimuth and distance, so the kernel only estimates the
//    ray tube's spreading; a deposit of that segment on a second edge takes D of the first at the second's point. If it
//    hits a surface and goes on, it takes D at its own azimuth with the distance to the hit. UTD's transition terms keep
//    part 11's beam weights, term by term where the term's boundary exists (PathWalkD).
//  * Sampling: splitting. The ray goes on undiffracted with its full weight and every deposit is a candidate of a
//    one-slot weighted reservoir per ray (as the transmitted branches); the kept one becomes a diffracted ray of its own
//    when the ray ends (persistent rays: when the update ends), carrying the summed weight. (A choice per piece between
//    diffracting, probability q, and going on with weight 1 / (1 - q) leaves the undiffracted ray a weight (1 - q)^-n
//    with probability (1 - q)^n after n pieces that pass an edge.) The proposal weight is the deposit times the angle
//    and UTD's scale 1 / (8 pi k) at 125 Hz (per D not yet applied), so the slot's sum is an energy for the roulette.
// A diffracted ray's first segment carries its edge's state in walkMeta (edge | grazing << 24 | reflected beam << 25 |
// first segment << 26), walkPhiIn and walkM (incidence azimuth, incident length); persistent rays keep it in motion.yzw.
static const float PathWalkTolerance = 4 * RayEpsilon; // a point lies in a plane
// Per plane (PathPrimitivePlane: the coplanar faces of one object), the edges with a face in it: the candidates past a
// hit on that plane.
StructuredBuffer<uint> PathPrimitivePlane;
StructuredBuffer<uint2> PathPlaneEdgeRanges;
StructuredBuffer<uint> PathPlaneEdgeIndices;
static const uint PathWalkGrazing = 1u << 24, PathWalkBeam = 2u << 24, PathWalkFirst = 4u << 24;
// The plane (normal, offset) is that of the face of normal n through the edge point a.
bool PathWalkOnPlane(float4 plane, float3 a, float3 n) { return abs(dot(plane.xyz, n)) > .99999 && abs(dot(plane.xyz, a) - plane.w) < PathWalkTolerance; }
struct PathWalkRay
{
    float3 p, d, origin;
    float travelled, tau, survival;
    float reach, limit;       // the piece is [0, reach); the horizon's distance
    uint hitPlane;            // past a hit at `reach`: its plane's edges (PathPrimitivePlane), else 0xffffffff
    float4 plane;             // the latest event's plane (normal, offset)
    float4 low, high;         // Phi0 C / Q per band
    float flux;               // their largest
    uint meta; bool beam;     // walkMeta (a diffracted ray's first segment); the chain is a reflected beam
    float phiIn, m;
};
struct PathWalkSlot           // the kept candidate (its flux per band is made once, at launch: PathWalkLaunch)
{
    float sum, tau, survival;
    float3 e, q;              // the edge point; the ray's closest point (visibility; q = e: none needed)
    float4 low, high;         // the passing ray's Phi0 C / Q per band
    float t, rho2, w20, sinGamma, weight; // closest approach, offset^2, the widest band's w^2, sin(gamma), proposal weight
    float phiIn, m, cosBeta;  // incidence azimuth, incident length, cosine of Keller's cone
    uint meta, count;         // edge | grazing | beam; deposits made (counter)
    uint previous; float previousPhiIn, previousM; float3 previousD, previousE; // a diffracted ray's first segment: its edge's D
};
// PathUtdEnergy with a weight ratio per term (incident-boundary terms first; 0: unweighted). (A function of its own: the
// static walks compile PathUtdEnergy as before, so that their outputs stay bit-identical, part 44.)
float PathWalkUtd(float n, float phiIn, float phiOut, float sinBeta, float k, float distance, float4 ratios)
{
    float2 sum = 0;
    [unroll] for (uint i = 0; i < 4; i++)
    {
        float sign = (i & 1) == 0 ? 1 : -1, x = i < 2 ? phiOut - phiIn : phiOut + phiIn;
        float argument = (RayPi + sign * x) / (2 * n);
        if (abs(sin(argument)) < 1e-5) { x += sign * 4e-5 * n; argument = (RayPi + sign * x) / (2 * n); }
        float N = round((x + sign * RayPi) / (2 * RayPi * n)), c = cos(n * RayPi * N - .5 * x), X = k * distance * 2 * c * c;
        float weight = ratios[i] <= 0 ? 1 : 1 - exp(-2 * X * ratios[i]);
        sum += weight * (cos(argument) / sin(argument)) * UtdTransition(X);
    }
    return dot(sum, sum) / (8 * RayPi * k * n * n * sinBeta * sinBeta);
}
// |D|^2 sin^2(beta) of a diffracted ray's edge toward azimuth phiOut at distance s (UTD distance s m sin^2 / (s + m), m the
// incident length), with the edge's opacity, the diffraction factor and the grazing factor, all squared.
// Part 11's beam weights, term by term, only where the term's boundary lies in the wedge's exterior (otherwise no traced
// field has a boundary there for the beams to smooth): terms 0 and 1 (the shadow boundaries at phi' - pi and phi' + pi)
// where the incident field is a reflected beam, terms 2 and 3 (the reflection boundaries of face n at 2 angle - pi - phi'
// and of face 0 at pi - phi') always. At grazing incidence the face's reflected field is the incident field itself (no
// reflected beam is traced): its reflection term takes the coinciding shadow-boundary term's weight.
float PathWalkD(uint meta, float angle, float phiIn, float m, float phiOut, float sinBeta, float s, uint band)
{
    if (!(phiOut <= angle)) return 0;
    float ratio = s / m, s2 = sinBeta * sinBeta, opacity = MethodEdgeOpacity(meta & 0xffffff, band);
    bool beam = (meta & PathWalkBeam) != 0;
    float4 ratios = ratio * float4(beam && phiIn >= RayPi, beam && phiIn <= angle - RayPi, phiIn >= angle - RayPi, phiIn <= RayPi);
    if ((meta & PathWalkGrazing) != 0) { if (phiIn < .5 * angle) ratios.w = ratios.y; else ratios.z = ratios.x; }
    return PathWalkUtd(angle / RayPi, phiIn, phiOut, sinBeta, MethodWavenumber(band), s * m / (s + m) * s2, ratios) * s2 *
           opacity * opacity * DiffractionFactor * DiffractionFactor * ((meta & PathWalkGrazing) != 0 ? .25 : 1);
}
// One edge against the ray piece (beyond: past its hit, the edge in the hit's plane): its deposit, offered to the slot.
void PathWalkConsider(uint edge, PathWalkRay r, bool beyond, inout uint rng, inout PathWalkSlot slot)
{
    if ((r.meta & PathWalkFirst) != 0 && edge == (r.meta & 0xffffff)) return; // (the diffracted ray's own edge)
    float3 a, u, f, nn; float extent, angle;
    EdgeFrame(edge, MethodState, false, a, u, f, nn, extent, angle);
    float b = dot(r.d, u), sin2 = 1 - b * b;
    if (sin2 < 1e-6) return;
    float3 w0 = r.p - a;
    float dd = dot(w0, r.d), ee = dot(w0, u), t = (b * ee - dd) / sin2, z = (ee - b * dd) / sin2;
    if (t <= RayEpsilon || (t >= r.reach) != beyond || t >= r.limit || z < 0 || z > extent) return;
    float3 e = a + u * z, q = r.p + r.d * t, gap = q - e;
    float rho2 = dot(gap, gap), w20 = SpeedOfSound / MethodBandHz(0) * (r.travelled + t) / RayPi;
    if (rho2 > 9 * w20) return; // beyond three widths of the widest band (exp(-18))
    float3 f1 = f * cos(angle) + nn * sin(angle), n1 = nn * cos(angle) - f * sin(angle);
    if ((PathWalkOnPlane(r.plane, a, nn) || PathWalkOnPlane(r.plane, a, n1)) && abs(dot(r.plane.xyz, r.origin) - r.plane.w) >= PathWalkTolerance) return;
    float3 o = r.origin - a, incident = e - r.origin;
    float m = length(incident);
    if (m <= RayEpsilon) return;
    incident /= m;
    bool on0 = abs(dot(o, nn)) < PathWalkTolerance && dot(o, f) > 0, on1 = abs(dot(o, n1)) < PathWalkTolerance && dot(o, f1) > 0, grazing = on0 || on1;
    // (a thin sheet has both faces in one plane: the side the ray passes on)
    float phiIn = grazing ? (on0 && (!on1 || dot(gap, nn) >= 0) ? 0 : angle) : TunedEdgeAngle(-incident, f, nn, angle);
    if (phiIn > angle) return; // an origin behind a face (an image): no incidence from the wedge's exterior
    float sinGamma = sqrt(sin2), scale = 1 / (8 * RayPi * MethodWavenumber(0));
    float weight = r.flux * sqrt(2 / (RayPi * w20)) * exp(-2 * rho2 / w20) / sinGamma * angle * (grazing ? 2 : 1) * scale * ((r.meta & PathWalkFirst) != 0 ? scale : 1);
    if (!(weight > 0) || !isfinite(weight)) return;
    slot.count++;
    slot.sum += weight;
    if (RandomRay(rng) * slot.sum >= weight) return;
    slot.low = r.low; slot.high = r.high; slot.t = t; slot.rho2 = rho2; slot.w20 = w20; slot.sinGamma = sinGamma; slot.weight = weight;
    slot.previous = r.meta; slot.previousPhiIn = r.phiIn; slot.previousM = r.m; slot.previousD = r.d; slot.previousE = r.origin;
    slot.e = e; slot.q = t >= r.reach ? e : q; slot.tau = r.tau + t * MethodSlowness(AirMedium); slot.survival = r.survival;
    slot.phiIn = phiIn; slot.m = m; slot.cosBeta = dot(incident, u);
    slot.meta = edge | (grazing ? PathWalkGrazing : 0) | (r.beam ? PathWalkBeam : 0);
}
#ifdef USE_BVH
// Every edge whose first face is in a mesh-tree leaf whose box, padded by `radius`, the piece [0, length] crosses (a
// superset of the edges within `radius` of it), each once.
void PathWalkMesh(uint root, State pose, PathWalkRay r, float span, float radius, inout uint rng, inout PathWalkSlot slot)
{
    float4 inverse = pose.rotation; inverse.xyz = -inverse.xyz;
    float3 o = Rotate(inverse, r.p - pose.position.xyz), d = Rotate(inverse, r.d);
    float3 inv = float3(SafeInverse(d.x), SafeInverse(d.y), SafeInverse(d.z));
    BvhNode top = BvhNodes[root];
    float3 extent = max(abs(top.boundsMin.xyz), abs(top.boundsMax.xyz)), at = abs(pose.position.xyz), from = abs(r.p);
    float pad = radius + 16 * 1.1920929e-7 * (1 + max(at.x, max(at.y, at.z)) + 4 * max(extent.x, max(extent.y, extent.z)) + max(from.x, max(from.y, from.z)));
    uint stack[32], depth = 0, index = root;
    float entry;
    while (true)
    {
        BvhNode node = BvhNodes[index];
        if (LocalBox(node, o, inv, span, pad, false, entry))
        {
            if (node.links.w == 0) { stack[depth++] = node.links.y; index = node.links.x; continue; }
            for (uint leaf = 0; leaf < node.links.w; leaf++)
            {
                uint primitive = BvhIndices[node.links.z + leaf];
                uint2 range = EdgeRanges[primitive];
                for (uint i = 0; i < range.y; i++)
                {
                    uint edge = EdgeIndices[range.x + i];
                    if (Edges[edge].ids.y == primitive) PathWalkConsider(edge, r, false, rng, slot);
                }
            }
        }
        if (depth == 0) break;
        index = stack[--depth];
    }
}
// The candidate edges of a piece through the scene's trees: the piece [0, reach) with the radius of three widths at its
// far end, clipped to the scene's box padded by that radius (no edge point lies outside the box).
void PathWalkNear(PathWalkRay r, inout uint rng, inout PathWalkSlot slot)
{
    if (BvhNodeCount == 0) return;
    float lambda0 = SpeedOfSound / MethodBandHz(0), span = r.reach, radius = 3 * sqrt(lambda0 * (r.travelled + span) / RayPi);
    BvhNode root = BvhNodes[0];
    float3 inv = float3(SafeInverse(r.d.x), SafeInverse(r.d.y), SafeInverse(r.d.z));
    float3 lo = (root.boundsMin.xyz - radius - r.p) * inv, hi = (root.boundsMax.xyz + radius - r.p) * inv;
    float3 near3 = min(lo, hi), far3 = max(lo, hi);
    float enter = max(max(near3.x, near3.y), max(near3.z, 0)), leave = min(min(far3.x, far3.y), far3.z);
    if (leave < enter) return;
    span = min(span, leave); radius = 3 * sqrt(lambda0 * (r.travelled + span) / RayPi);
    uint stack[32], depth = 0, index = 0;
    while (index < BvhNodeCount)
    {
        BvhNode node = BvhNodes[index];
        node.boundsMin.xyz -= radius; node.boundsMax.xyz += radius;
        if (BoundsHit(node, r.p, r.d * span, false))
        {
            if (node.links.w == 0) { stack[depth++] = node.links.y; index = node.links.x; continue; }
            State pose = States[MethodState + node.links.z];
            if ((pose.flags.x & 3) == 3) PathWalkMesh(node.links.x, pose, r, span, radius, rng, slot);
        }
        if (depth == 0) break;
        index = stack[--depth];
    }
}
#else
void PathWalkNear(PathWalkRay r, inout uint rng, inout PathWalkSlot slot)
{
    [loop] for (uint edge = 0; edge < MethodEdgeCount; edge++) PathWalkConsider(edge, r, false, rng, slot);
}
#endif
// Every candidate of a piece: near it (PathWalkNear), and past its hit those of the hit's plane (each edge passes the
// three-width test against its own closest approach, so no length bound is needed past the hit).
void PathWalkEdges(PathWalkRay r, inout uint rng, inout PathWalkSlot slot)
{
    PathWalkNear(r, rng, slot);
    if (r.hitPlane == 0xffffffff) return;
    uint2 range = PathPlaneEdgeRanges[r.hitPlane];
    [loop] for (uint i = 0; i < range.y; i++) PathWalkConsider(PathPlaneEdgeIndices[range.x + i], r, true, rng, slot);
}
// The slot's candidate as a diffracted ray leaving its edge point: direction on Keller's cone (uniform azimuth), flux.
// Its flux per band: the deposit Phi K1(rho) / sin(gamma) (air absorption to the closest point) times the angle, the
// grazing image's 2, and D of a diffracted ray's first segment at this point, over the proposal weight, times the slot's
// sum over the roulette's keep.
bool PathWalkLaunch(PathWalkSlot slot, float keep, inout uint rng, out float3 d, out float4 low, out float4 high)
{
    float3 a, u, f, nn; float extent, angle;
    EdgeFrame(slot.meta & 0xffffff, MethodState, false, a, u, f, nn, extent, angle);
    float sinBeta = sqrt(max(0, 1 - slot.cosBeta * slot.cosBeta)), phi = RandomRay(rng) * angle;
    d = normalize(u * slot.cosBeta + sinBeta * (f * cos(phi) + nn * sin(phi)));
    bool first = (slot.previous & PathWalkFirst) != 0;
    float3 pa = 0, pu = 0, pf = 0, pn = 0; float pextent = 0, pangle = 1, phiAt = 0, sinPrevious = 1;
    if (first)
    {
        EdgeFrame(slot.previous & 0xffffff, MethodState, false, pa, pu, pf, pn, pextent, pangle);
        float pb = dot(slot.previousD, pu);
        sinPrevious = sqrt(max(0, 1 - pb * pb)); phiAt = TunedEdgeAngle(slot.e - slot.previousE, pf, pn, pangle);
    }
    low = 0; high = 0;
    [loop] for (uint k = 0; k < MethodBands; k++)
    {
        float4 lane = float4((k & 3) == uint4(0, 1, 2, 3));
        float w2 = slot.w20 / (float)(1u << k);
        float value = dot(k < 4 ? slot.low : slot.high, lane) * exp(-2 * MethodLoss(AirMedium, k) * slot.t) * sqrt(2 / (RayPi * w2)) * exp(-2 * slot.rho2 / w2) /
                      slot.sinGamma * angle * ((slot.meta & PathWalkGrazing) != 0 ? 2 : 1) / slot.weight * (slot.sum / keep);
        if (first) value *= PathWalkD(slot.previous, pangle, slot.previousPhiIn, slot.previousM, phiAt, sinPrevious, slot.m, k);
        if (k < 4) low += value * lane; else high += value * lane;
    }
    return sinBeta > 1e-4 && Visible(MethodState, slot.q, slot.e, false);
}
#endif

// One flat loop; every heavy helper has a single call site. Each iteration either deposits the
// pending contribution, makes one connection, or traces one segment (or, persistent, the part of it
// that fits in the update).
void TracePath(uint index)
{
    uint ray = 0, slot = 0;
    bool resumed = false;
    if (PathPersistent != 0)
    {
        uint thread = MethodRayOffset + index, live = min(PathLive[PathReadCount], PathCapacity);
        if (thread < live) { resumed = true; slot = thread; }
        else if (thread - live < MethodRayCount) ray = thread - live;
        else return;
    }
    else
    {
        if (index >= MethodRayCount) return;
        ray = MethodRayOffset + index;
    }
    MethodCount(resumed ? CounterResumed : CounterRays);
    float3 S = States[MethodState + MethodSourceObject].position.xyz, L = States[MethodState + MethodListenerObject].position.xyz;
    uint rng = HashRay(ray * 0x9e3779b9u ^ HashRay(RaySeed * 0xc2b2ae35u + 0x2545f491u));
    MethodAnchor v;
    v.position = S; v.normal = 0; v.previous = S; v.kind = 0; v.edge = 0; v.primitive = 0xffffffff; v.time = 0;
    bool airListener = true;
    v.medium = AirMedium;
    [loop] for (uint end = 0; end < 2; end++)
    {
        uint here = MediumAt(MethodState, end == 0 ? S : L, false);
        if (end == 0) v.medium = here; else airListener = here == AirMedium;
    }
    float I[8], cI[8], Phi0[8], C[8];
    // A new ray's weight: the histogram is divided by MethodTotalRays when read, so a cohort of n rays carries
    // MethodTotalRays / n each (PathLaunchScale, persistent rays; 1 otherwise) and a change of n between updates
    // (MethodTransport.Population, the auto-budget) leaves every cohort's estimate unbiased.
    float launch = PathPersistent != 0 ? PathLaunchScale : 1;
    float4 pendingLow = 0, pendingHigh = 0; // bands of the pending contribution (no arrays across function boundaries)
    [unroll] for (uint k = 0; k < MethodBands; k++)
    { I[k] = ReferenceDistance * ReferenceDistance * launch; cI[k] = I[k]; Phi0[k] = 0; C[k] = 1; }
    MethodAnchor c = v;
    // Ray 0 makes the source's straight connection for every ray (it is deterministic).
    bool connectListener = ray == 0, connectEdge = PathBeamsOnly == 0, hasPending = false;
#if PATH_WALK_EDGE
    connectEdge = false; // (walk diffraction: no edge connections)
    uint walkMeta = 0; float walkPhiIn = 0, walkM = 1; // a diffracted ray's first segment ("Walk diffraction")
    PathWalkSlot walkSlot = (PathWalkSlot)0;
    bool walkQueued = false; float3 walkD = 0; float4 walkLow = 0, walkHigh = 0; // (launched: its direction and flux)
#endif
    float listenerScale = MethodTotalRays / launch, pendingTime = 0; // (the source's exact connection: launch cancels)
    // Delay rates (the rate channel): the ray's rho so far (-ln of its frequency ratio, "Delay rates" above), and for the
    // walk's vertex v and the connections' anchor c theirs up to and including the incoming leg and their point's
    // velocity; vL the listener's.
#if PATH_DOPPLER
    float3 vL = Velocity(MethodState + MethodListenerObject), vVelocity = Velocity(MethodState + MethodSourceObject), cVelocity = vVelocity;
    float rho = 0, vRate = 0, cRate = 0, pendingRate = 0, branchRate = 0;
#endif
    float3 pendingFrom = 0;
    uint events = 0;
    bool edges = MethodEdgeCount > 0 && DiffractionFactor > 0, travelling = false, done = false, fluxPending = false, inBranch = false;
    float3 p = S, d = 0;
    float Q = 1, tau = 0;
    uint medium = v.medium;
    PathChain chain = PathStart(0, 0), branch = chain;
    float branchSum = 0, branchTau = 0, branchSurvival = 1, survival = 1; float3 branchP = 0, branchD = 0; float branchF[8];
    uint branchMedium = 0;
    float4 plane = 0, branchPlane = 0; // the latest event's plane, outgoing side (beams)
    [unroll] for (uint k1 = 0; k1 < MethodBands; k1++) branchF[k1] = 0;
    float edgeExtent = 1, edgeAngle = 1, edgeM = 1, edgeSinBeta = 1, edgePhiIn = 0, edgeD = 1;
    // Persistent: the path time this update reaches, whether the current segment's beam is scored, and
    // how far the ray has come since the segment's start (it may resume in mid-air).
    // New rays reach half a period in their first update, so that update n covers the path times of every cohort
    // that are heard within half a period of its state's time (the response is centred on it).
    float age = 0, target = PathPersistent != 0 ? .5 * PathStep : 1e30, sinceStart = 0;
    bool scored = false, suspending = false, offSurface = false;
    if (resumed)
    {
        PathRay r = PathRays[PathReadBase + slot];
        p = r.position.xyz; tau = r.position.w; d = r.direction.xyz; age = r.direction.w; target = age + PathStep;
        [unroll] for (uint kr = 0; kr < MethodBands; kr++)
        {
            float4 lane = float4((kr & 3) == uint4(0, 1, 2, 3));
            Phi0[kr] = dot(kr < 4 ? r.phiLow : r.phiHigh, lane); C[kr] = dot(kr < 4 ? r.wLow : r.wHigh, lane);
        }
        plane = r.plane; survival = r.scalars.x; sinceStart = r.scalars.w;
        chain.travelled = r.scalars.y; chain.time = r.scalars.z; chain.reflections = r.meta.x & 0xffffff; chain.kind = (r.meta.x >> 27) & 3;
        medium = r.meta.y; rng = r.meta.z; events = r.meta.w;
#if PATH_DOPPLER
        rho = r.motion.x;
#endif
        inBranch = (r.meta.x & (1u << 24)) != 0; scored = (r.meta.x & (2u << 24)) != 0;
#if PATH_WALK_EDGE
        walkPhiIn = r.motion.y; walkM = r.motion.z; walkMeta = asuint(r.motion.w);
#endif
        offSurface = (r.meta.x & (4u << 24)) != 0;
        connectListener = false; connectEdge = false; travelling = true;
    }
    uint iterations = 0;
    [loop] for (;;)
    {
        iterations++;
        if (hasPending)
        {
            hasPending = false;
            // Persistent: heard ceil((time - target) / step) updates from now (this update's rays reach path time `target`).
            // The ring is deep enough for any connection in the scene (MethodTransport.RingSlots); one heard later
            // still is counted (PathLive[PathRingFar]) and asserted absent by the checks.
            float aheadUpdates = PathPersistent != 0 ? max(0, ceil((pendingTime - target) / PathStep)) : 0;
            if (aheadUpdates > (float)PathRingSlots - 1 && pendingTime <= MethodHorizon) InterlockedAdd(PathLive[PathRingFar], 1u); // (later than the horizon: dropped)
            uint ahead = (uint)min(aheadUpdates, (float)PathRingSlots - 1);
            // (This update's rays reach path time `target`; its own time is half a period before, PathStep later per slot ahead.)
#if PATH_DOPPLER
            float heard = PathPersistent != 0 ? pendingTime - (target - .5 * PathStep) - ahead * PathStep : 0;
            PathDeposit(pendingFrom, pendingLow, pendingHigh, pendingTime, HashRay(rng ^ (events * 0x85ebca6bu)), ahead, 1 - exp(-pendingRate), heard);
#else
            PathDeposit(pendingFrom, pendingLow, pendingHigh, pendingTime, HashRay(rng ^ (events * 0x85ebca6bu)), ahead);
#endif
        }
#if PATH_WALK_EDGE
        if ((done || suspending) && walkSlot.sum > 0)
        {
            // The kept diffraction candidate becomes a diffracted ray (after roulette on its summed weight, as a branch): this
            // ray, once it has ended, or a ray of its own queued after the loop when this one stops for the update.
            float keepWalk = min(1, walkSlot.sum / (4 * RayPi * ReferenceDistance * ReferenceDistance * EnergyBranchFloor));
            walkQueued = RandomRay(rng) < keepWalk;
            if (walkQueued) walkQueued = PathWalkLaunch(walkSlot, keepWalk, rng, walkD, walkLow, walkHigh);
            walkSlot.sum = 0;
            if (!suspending)
            {
                if (!walkQueued) continue; // (the next iteration takes the transmitted branch, if any)
                walkQueued = false;
                d = walkD;
                [unroll] for (uint kw0 = 0; kw0 < MethodBands; kw0++) { float4 lane = float4((kw0 & 3) == uint4(0, 1, 2, 3)); Phi0[kw0] = dot(kw0 < 4 ? walkLow : walkHigh, lane); C[kw0] = 1; }
                p = walkSlot.e; tau = walkSlot.tau; medium = AirMedium; Q = 1; survival = walkSlot.survival;
                chain = PathStart(2, tau); chain.edge = walkSlot.meta & 0xffffff;
                walkMeta = walkSlot.meta | PathWalkFirst; walkPhiIn = walkSlot.phiIn; walkM = walkSlot.m;
                plane = float4(d, -1e30); offSurface = false; scored = false; sinceStart = 0;
                fluxPending = false; travelling = true; done = false; inBranch = true;
                MethodCount(CounterEdgeEvents);
            }
        }
#endif
        if (suspending) break;
        if (connectListener)
        {
            connectListener = false;
            float length0;
            if ((PathProfileFlags & 4) != 0 && ray != 0) continue;
            hasPending = PathListener(c, cI, L, listenerScale, c.kind == 0, rng, pendingLow, pendingHigh, pendingTime, pendingFrom, length0);
#if PATH_DOPPLER
            pendingRate = cRate + PathEmit(cVelocity, -pendingFrom * MethodSlowness(AirMedium)) + PathReceive(vL, -pendingFrom * MethodSlowness(AirMedium));
#endif
            if (hasPending && PathAudio != 0 && c.kind == 0)
            {
                PathDirectArrival(S, L, pendingLow, pendingHigh, MethodTotalRays, pendingTime, length0); // exact, rendered as a delayed path
                hasPending = false;
            }
            continue;
        }
        if (connectEdge)
        {
            connectEdge = false;
            if ((PathProfileFlags & 8) != 0) continue;
#if PATH_DOPPLER
            hasPending = PathEdgeConnection(c, float4(cI[0], cI[1], cI[2], cI[3]), float4(cI[4], cI[5], cI[6], cI[7]), L, cRate, cVelocity, vL, rng, pendingLow, pendingHigh, pendingTime, pendingFrom, pendingRate);
#else
            hasPending = PathEdgeConnection(c, float4(cI[0], cI[1], cI[2], cI[3]), float4(cI[4], cI[5], cI[6], cI[7]), L, rng, pendingLow, pendingHigh, pendingTime, pendingFrom);
#endif
            continue;
        }
        if (done)
        {
#if PATH_WALK_EDGE
            walkMeta = 0;
#endif
            if (!(branchSum > 0)) break;
            // Trace the kept transmitted branch from its surface, after roulette on its share of a
            // source ray's flux (4 pi r_ref^2).
            float keep = min(1, branchSum / (4 * RayPi * ReferenceDistance * ReferenceDistance * EnergyBranchFloor));
            if (RandomRay(rng) >= keep) break;
            [unroll] for (uint k2 = 0; k2 < MethodBands; k2++) { Phi0[k2] = branchF[k2] * branchSum / keep; C[k2] = 1; }
            p = branchP + branchPlane.xyz * RayEpsilon; offSurface = true;
            d = branchD; medium = branchMedium; tau = branchTau; Q = 1; survival = branchSurvival; plane = branchPlane;
#if PATH_DOPPLER
            rho = branchRate;
#endif
            chain = branch;
            fluxPending = false; travelling = true; done = false; branchSum = 0; inBranch = true; scored = false; sinceStart = 0;
            MethodCount(CounterBranches);
        }
        if (!travelling)
        {
            if (++events >= MethodMaxEvents) { MethodCount(CounterEventCap); break; }
            bool edgeRays = edges && v.kind == 0 && METHOD_EDGE_CDF_READ[max(MethodEdgeCount, 1u) - 1] > 0;
#if PATH_WALK_EDGE
            edgeRays = false; // (walk diffraction: no edge rays from the source)
#endif
            bool viaEdge = edgeRays && RandomRay(rng) < MethodEdgeProbability;
            [unroll] for (uint k3 = 0; k3 < MethodBands; k3++) C[k3] = 1;
            medium = v.medium;
            scored = false; sinceStart = 0;
            if (viaEdge)
            {
                float3 a, u, f, nn; float extent, angle, z0, r, edgePick;
                uint edge = EnergySampleEdge(RandomRay(rng), edgePick);
                EdgeFrame(edge, MethodState, false, a, u, f, nn, extent, angle);
                float span = EnergyEdgeSpan(a, u, extent, v.position, z0, r);
                float z = clamp(z0 + r * tan(atan(-z0 / r) + RandomRay(rng) * span), 0, extent);
                float3 e = a + z * u, w = e - v.position;
                float m = length(w);
                edgeExtent = (r * r + (z - z0) * (z - z0)) * span / r; // 1 / (density of the point on the edge)
                if (m <= RayEpsilon || TunedEdgeAngle(-w, f, nn, angle) > angle || !Visible(MethodState, v.position, e, false)) { done = true; continue; }
                w /= m;
                float cosBeta = dot(w, u);
                edgeSinBeta = sqrt(max(0, 1 - cosBeta * cosBeta));
                edgeD = EnergyDirectivity(v, w);
                if (edgeD == 0 || edgeSinBeta <= 1e-4) { done = true; continue; }
                edgePhiIn = TunedEdgeAngle(v.position - e, f, nn, angle);
                float phi = RandomRay(rng) * angle;
                d = normalize(u * cosBeta + edgeSinBeta * (f * cos(phi) + nn * sin(phi)));
                edgeAngle = angle; edgeM = m;
#if PATH_DOPPLER
                float3 vEdge = PointVelocity(MethodState + Edges[edge].ids.x, e);
                rho = vRate + PathEmit(vVelocity, w * MethodSlowness(AirMedium)) + PathReceive(vEdge, w * MethodSlowness(AirMedium)) + PathEmit(vEdge, d * MethodSlowness(AirMedium));
#endif
                tau = v.time + m * MethodSlowness(AirMedium);
                p = e; Q = 1; fluxPending = true;
                chain = PathStart(2, tau);
                chain.edge = edge; chain.edgePhi = phi; chain.edgePick = edgePick;
                plane = float4(d, -1e30); // (beams only: no plane to stand in front of; the normal sets the start inset)
                MethodCount(CounterEdgeEvents);
                // Edge rays and source edge connections share the edge's energy kernel; a pending
                // flux is completed at the first hit (UTD's distance parameter).
                [unroll] for (uint k4 = 0; k4 < MethodBands; k4++) Phi0[k4] = I[k4];
            }
            else
            {
                float3 w0 = v.kind == 1 ? MethodCosineDirection(v.normal, rng) : UnitDirection(rng, false);
                float pdf = v.kind == 1 ? max(1e-12, dot(w0, v.normal)) / RayPi : 1 / (4 * RayPi);
                float D = EnergyDirectivity(v, w0);
                if (D == 0) { done = true; continue; }
                [unroll] for (uint k5 = 0; k5 < MethodBands; k5++) Phi0[k5] = I[k5] * D / pdf;
                Q = edgeRays ? 1 - MethodEdgeProbability : 1;
                p = v.position; d = w0; tau = v.time;
#if PATH_DOPPLER
                rho = vRate + PathEmit(vVelocity, w0 * MethodSlowness(v.medium));
#endif
                if (v.kind == 1) { plane = float4(v.normal, dot(v.normal, p)); p += v.normal * RayEpsilon; offSurface = true; }
                chain = PathStart(v.kind, v.time);
            }
            travelling = true;
        }
        if (++events >= MethodMaxEvents) { MethodCount(CounterEventCap); break; }
        float limit = (MethodHorizon - tau) / MethodSlowness(medium);
        uint hit; float distance; float3 normal;
        if (limit <= 0) { MethodCount(CounterHorizon); done = true; continue; }
        // A segment that leaves a surface starts RayEpsilon out along that surface's outgoing normal and is traced
        // from zero distance: a fixed minimum distance would skip another object's surface within RayEpsilon of the
        // start (a mullion's face meeting a window pane let rays into the pane, where they bounced for hundreds of events).
        // (Profiling bit 15: every segment is traced twice, the same query, so the extra time is one traversal per segment.)
        bool found = false;
        [loop] for (uint traces = (PathProfileFlags & 32768) != 0 ? 0u : 1u; traces < 2; traces++)
            found = TraceScene(MethodState, p, d, offSurface ? 0 : RayEpsilon, limit, false, false, 0xffffffff, hit, distance, normal);
        // A ray in air cannot meet a closed solid's face from inside: it got there through a leak (README part 18)
        // and carries energy that is not in the room, so it ends before scoring anything more.
        if (found && medium == AirMedium && dot(normal, d) * PathVolumeSide(Primitives[hit].indices.x) > 0) { MethodCount(CounterInsideSolid); done = true; continue; }
        if (fluxPending) // (an escaping edge ray takes the horizon as its distance: beams only scores its first segment)
        {
            // UTD energy flux per unit edge length and azimuth, over the sampling densities.
            fluxPending = false;
            float spread = edgeM * (found ? distance : limit) / (edgeM + (found ? distance : limit));
            [loop] for (uint k6 = 0; k6 < MethodBands; k6++)
            {
                float opacity = MethodEdgeOpacity(chain.edge, k6);
                Phi0[k6] = UtdEnergy(edgeAngle / RayPi, edgePhiIn, chain.edgePhi, edgeSinBeta, MethodWavenumber(k6), spread * edgeSinBeta * edgeSinBeta) *
                           edgeSinBeta * edgeSinBeta * Phi0[k6] * edgeD * opacity * opacity * DiffractionFactor * DiffractionFactor *
                           exp(-2 * MethodLoss(AirMedium, k6) * edgeM) / (edgeM * edgeM) * edgeExtent * edgeAngle / (chain.edgePick * MethodEdgeProbability);
            }
        }
        // The part of the segment this update covers (all of it in the full solve); an escaping segment ends here.
        float segment = found ? distance : limit;
        float advance = found ? min(distance, max(0, (target - tau) / MethodSlowness(medium))) : segment;
        // (Never stop within a few RayEpsilon of the hit: the resumed trace starts RayEpsilon out and would miss it.)
        bool ends = advance >= segment - 4 * RayEpsilon;
        float along = dot(L - p, d);
        if (!scored && (along < advance || ends) && (chain.reflections >= 1 || (PathBeamsOnly != 0 && chain.kind != 0)
#if PATH_WALK_EDGE
            || (walkMeta & PathWalkFirst) != 0 // (a diffracted ray's first segment, in either mode)
#endif
            ) && medium == AirMedium && airListener && dot(L, plane.xyz) > plane.w)
        {
            // Beam kernel at the listener of the ray's unfolded line (this segment extended): each prefix of the
            // chain (one listener image) is scored once, by its own segment, whether or not the closest approach
            // falls inside the segment (persistent: in the update during which the ray passes the closest point,
            // or at the segment's end); the listener must be in front of the latest plane and visible from the
            // segment. Width and time use the unfolded distance from the chain's origin to the listener image.
            scored = true;
            float3 offset = L - p - d * along;
            float rho2 = dot(offset, offset), lc = chain.travelled + along;
            float lu = sqrt(max(lc * lc + rho2, RayEpsilon * RayEpsilon));
            float lambda0 = SpeedOfSound / MethodBandHz(0), w20 = lambda0 * lu / RayPi, clearance = asfloat(PATH_CLEARANCE_READ[0]);
            // A point of the segment nearest the listener, kept 4 RayEpsilon (perpendicular) off the walls at both
            // ends, so that the visibility test cannot start inside a wall's tolerance and see through it.
            float startInset = max(0, 4 * RayEpsilon / max(abs(dot(d, plane.xyz)), 1e-3) - sinceStart);
            float endInset = found ? 4 * RayEpsilon / max(abs(dot(d, normal)), 1e-3) : 0;
            float3 qb = p + d * clamp(along, min(startInset, .5 * segment), max(segment - endInset, .5 * segment));
            bool seen = lc > 0 && rho2 < clearance * clearance && Visible(MethodState, qb, L, false);
            if (seen)
            {
                float4 sumLow = 0, sumHigh = 0;
                float extra = lu - chain.travelled; // unfolded distance from here to the listener image
#if PATH_WALK_EDGE
                // A diffracted ray's first segment: |D|^2 sin^2(beta) at the listener's own azimuth and distance.
                bool walkScore = (walkMeta & PathWalkFirst) != 0;
                float walkAngle = 1, walkPhiL = 0, walkSinBeta = 1;
                if (walkScore)
                {
                    float3 wa, wu, wf, wn; float wextent;
                    EdgeFrame(walkMeta & 0xffffff, MethodState, false, wa, wu, wf, wn, wextent, walkAngle);
                    float cb = dot(d, wu);
                    walkSinBeta = sqrt(max(0, 1 - cb * cb)); walkPhiL = TunedEdgeAngle(L - (p - d * chain.travelled), wf, wn, walkAngle);
                }
#endif
                [unroll] for (uint kb = 0; kb < MethodBands; kb++)
                {
                    float4 lane = float4((kb & 3) == uint4(0, 1, 2, 3));
                    float w2 = w20 / (float)(1u << kb);
                    float value = Phi0[kb] * C[kb] / Q * 2 / (RayPi * w2) * exp(-2 * rho2 / w2) / (1 - exp(-2 * clearance * clearance / w2)) *
                                  exp(-2 * MethodLoss(AirMedium, kb) * extra);
#if PATH_WALK_EDGE
                    if (walkScore) value *= PathWalkD(walkMeta, walkAngle, walkPhiIn, walkM, walkPhiL, walkSinBeta, lc, kb);
#endif
                    if (kb < 4) sumLow += value * lane; else sumHigh += value * lane;
                }
                if (dot(sumLow + sumHigh, 1) > 0)
                {
                    hasPending = true; pendingLow = sumLow; pendingHigh = sumHigh;
                    pendingTime = chain.time + lu * MethodSlowness(AirMedium); pendingFrom = -d;
#if PATH_DOPPLER
                    pendingRate = rho + PathReceive(vL, d * MethodSlowness(AirMedium));
#endif
                }
            }
        }
#if PATH_WALK_EDGE
        if (edges && medium == AirMedium && (PathProfileFlags & 262144) == 0)
        {
            // Walk diffraction: this piece's deposits on the edges (past a hit, in the hit's plane), offered to the slot.
            PathWalkRay wr;
            wr.p = p; wr.d = d; wr.origin = p - d * chain.travelled; wr.travelled = chain.travelled; wr.tau = tau; wr.survival = survival;
            wr.reach = ends ? segment : advance; wr.limit = limit;
            wr.hitPlane = ends && found && (PathProfileFlags & 131072) == 0 ? PathPrimitivePlane[hit] : 0xffffffff;
            wr.plane = plane; wr.flux = 0;
            [unroll] for (uint kc = 0; kc < MethodBands; kc++) wr.flux = max(wr.flux, Phi0[kc] * C[kc] / Q);
            wr.low = float4(Phi0[0] * C[0], Phi0[1] * C[1], Phi0[2] * C[2], Phi0[3] * C[3]) / Q;
            wr.high = float4(Phi0[4] * C[4], Phi0[5] * C[5], Phi0[6] * C[6], Phi0[7] * C[7]) / Q;
            wr.meta = walkMeta; wr.beam = chain.reflections >= 1; wr.phiIn = walkPhiIn; wr.m = walkM;
            PathWalkEdges(wr, rng, walkSlot);
        }
#endif
        if (!found) { MethodCount(CounterEscapes); done = true; continue; }
        if (!ends)
        {
            // Persistent: stop in mid-air at the update's path time; the next update resumes here.
            p += d * advance; tau += advance * MethodSlowness(medium); chain.travelled += advance;
            [unroll] for (uint ka = 0; ka < MethodBands; ka++) C[ka] *= exp(-2 * MethodLoss(medium, ka) * advance);
            sinceStart += advance; suspending = true;
            offSurface = offSurface && advance <= 0; // still at the surface if it has not moved
            continue;
        }
#if PATH_WALK_EDGE
        if ((walkMeta & PathWalkFirst) != 0)
        {
            // A diffracted ray going on past its first hit: D at its own azimuth, UTD's distance to the hit.
            float3 wa, wu, wf, wn; float wextent, wangle;
            EdgeFrame(walkMeta & 0xffffff, MethodState, false, wa, wu, wf, wn, wextent, wangle);
            float cb = dot(d, wu), sinBeta = sqrt(max(0, 1 - cb * cb)), phiOut = TunedEdgeAngle(d, wf, wn, wangle);
            [loop] for (uint kw1 = 0; kw1 < MethodBands; kw1++) Phi0[kw1] *= PathWalkD(walkMeta, wangle, walkPhiIn, walkM, phiOut, sinBeta, chain.travelled + distance, kw1);
            walkMeta = 0;
        }
#endif
        p += d * distance; tau += distance * MethodSlowness(medium); chain.travelled += distance;
        [unroll] for (uint k7 = 0; k7 < MethodBands; k7++) C[k7] *= exp(-2 * MethodLoss(medium, k7) * distance);
        float3 n = dot(d, normal) < 0 ? normal : -normal;
        float cosI = max(0, -dot(d, n));
        // A slit: this face and the previous event's face (plane) are parallel, face each other, and are closer than
        // half the shortest band's wavelength, so no band has a transverse mode across the gap and a ray bouncing
        // across it models nothing (the field there is a plane wave along the slit). Such gaps are mesh defects in the
        // Church (an open door-frame mesh whose threshold top lies 8 mm above the floor; rays entered from its open
        // ends and bounced every centimetre for hundreds of events per update, README part 18). The ray ends.
        if (dot(n, plane.xyz) < -.99 && dot(plane.xyz, p) - plane.w < .5 * SpeedOfSound / MethodBandHz(MethodBands - 1))
        { MethodCount(CounterInsideSolid); done = true; continue; }
        MethodCount(CounterEvents);
        float eT[8], eD[8], eS[8], mT = 0, mD = 0, mS = 0;
        [unroll] for (uint k8 = 0; k8 < MethodBands; k8++)
        {
            RayMaterial mat = MethodSurface(hit, k8);
            bool impedance = mat.medium.w != 0;
            eT[k8] = impedance ? 0 : saturate((1 - mat.surface.x) * mat.surface.z);
            eD[k8] = impedance ? 0 : saturate((1 - mat.surface.x) * (1 - mat.surface.z) * mat.surface.y);
            eS[k8] = EnergySpecular(mat, cosI);
            mT += eT[k8]; mD += eD[k8]; mS += eS[k8];
        }
        MethodAnchor x;
        x.position = p; x.normal = n; x.previous = p; x.kind = 1; x.edge = 0; x.primitive = hit; x.time = tau; x.medium = medium;
#if PATH_DOPPLER
        float3 hitVelocity = PointVelocity(MethodState + Primitives[hit].indices.x, p);
        float rateIn = rho + PathReceive(hitVelocity, d * MethodSlowness(medium)); // (the outgoing term follows the event's choice)
#endif
        if (mD > 0)
        {
            [unroll] for (uint k9 = 0; k9 < MethodBands; k9++) cI[k9] = Phi0[k9] * C[k9] / Q * eD[k9] / RayPi;
            c = x; listenerScale = 1;
#if PATH_DOPPLER
            cRate = rateIn; cVelocity = hitVelocity;
#endif
            connectListener = connectEdge = PathBeamsOnly == 0;
#if PATH_WALK_EDGE
            connectEdge = false;
#endif
        }
        float total = mT + mD + mS;
        if (!(total > 0)) { done = true; continue; }
        // Choose first (the reservoir candidate and the continuation), then find where a transmission
        // goes (out of a partition's far face, or into the medium behind the surface) only if one was
        // chosen: one call site each, and no traversal for the many hits that do not transmit.
        bool split = mT > 0 && mT < .5 * total && (PathProfileFlags & 1024) == 0, keepBranch = false; // (bit 10: no reservoir branches)
        float part[8], partWeight = 0;
        [unroll] for (uint k10 = 0; k10 < MethodBands; k10++) { part[k10] = Phi0[k10] * C[k10] * eT[k10] / Q; partWeight = max(partWeight, part[k10]); }
        if (split && partWeight > 0)
        {
            // Small transmitted share: a reservoir candidate (MethodCommon.hlsl), not a draw.
            branchSum += partWeight;
            keepBranch = RandomRay(rng) * branchSum < partWeight;
        }
        if (split) total = mD + mS;
        float pick = RandomRay(rng) * total;
        bool transmit = !split && pick >= mD && pick < mD + mT;
        float wall = 0; bool partition = false; uint next = medium;
        if (keepBranch || transmit)
        {
            partition = PathPartitionExit(hit, p, d, MethodRefracts(hit), wall);
            next = MediumAt(MethodState, p + d * ((partition ? wall : 0) + 2 * RayEpsilon), false);
        }
        if (keepBranch)
        {
            branch = chain; branchSurvival = survival;
            if (partition)
            {
                // Through a partition: undeviated, out of its far face.
                branchP = p + d * wall; branchD = d; branchMedium = next;
                branchTau = tau + wall * MethodSlowness(medium);
                PathEvent(branch, true);
                branch.travelled += wall;
            }
            else
            {
                float3 bent = refract(d, n, MethodIor(medium, 3) / MethodIor(next, 3));
                bool refracted = dot(bent, bent) > 0;
                branchP = p; branchD = refracted ? normalize(bent) : reflect(d, n); branchMedium = refracted ? next : medium;
                branchTau = tau;
                PathEvent(branch, refracted);
                if (refracted && !(medium == AirMedium && next == AirMedium) && (PathProfileFlags & 32) != 0) { MethodCount(CounterNonFinite); partWeight = 1e30; }
            }
            [unroll] for (uint k11 = 0; k11 < MethodBands; k11++) branchF[k11] = part[k11] / partWeight;
#if PATH_DOPPLER
            branchRate = rateIn + PathEmit(hitVelocity, branchD * MethodSlowness(branchMedium));
#endif
            float3 bn = dot(branchD, n) > 0 ? n : -n;
            branchPlane = float4(bn, dot(bn, branchP));
        }
        if (pick < mD)
        {
            // Survival in time also at scattering points (the specular events apply it below).
            float alive = PathSurvival(tau), keepAlive = alive / survival;
            survival = alive;
            if (keepAlive < 1 && RandomRay(rng) >= keepAlive) { MethodCount(CounterRoulette); done = true; continue; }
            [unroll] for (uint k12 = 0; k12 < MethodBands; k12++) I[k12] = cI[k12] * (total / mD) / min(1, keepAlive);
            v = x;
#if PATH_DOPPLER
            vRate = rateIn; vVelocity = hitVelocity;
#endif
            travelling = false;
            MethodCount(CounterDiffuse);
            if (v.time > MethodHorizon) { MethodCount(CounterHorizon); done = true; }
            continue;
        }
        if (transmit)
        {
            [unroll] for (uint k13 = 0; k13 < MethodBands; k13++) C[k13] *= eT[k13];
            Q *= mT / total;
            if (partition)
            {
                PathEvent(chain, true);
                tau += wall * MethodSlowness(medium); p += d * wall; chain.travelled += wall;
                medium = next;
            }
            else
            {
                float3 bent = refract(d, n, MethodIor(medium, 3) / MethodIor(next, 3));
                if (dot(bent, bent) > 0)
                {
                    PathEvent(chain, true);
                    if (!(medium == AirMedium && next == AirMedium) && (PathProfileFlags & 32) != 0) { MethodCount(CounterNonFinite); done = true; continue; }
                    d = normalize(bent); medium = next;
                }
                else { PathEvent(chain, false); d = reflect(d, n); }
            }
        }
        else
        {
            [unroll] for (uint k14 = 0; k14 < MethodBands; k14++) C[k14] *= eS[k14];
            Q *= mS / total;
            PathEvent(chain, false);
            d = reflect(d, n);
        }
#if PATH_DOPPLER
        rho = rateIn + PathEmit(hitVelocity, d * MethodSlowness(medium));
#endif
        float3 beamNormal = dot(d, n) > 0 ? n : -n;
        plane = float4(beamNormal, dot(beamNormal, p));
        p += beamNormal * RayEpsilon; offSurface = true;
        scored = false; sinceStart = 0;
        float weight = 0;
        [unroll] for (uint k16 = 0; k16 < MethodBands; k16++) weight = max(weight, C[k16]);
        float survive = min(1, weight / (Q * MethodRouletteFloor));
        if (inBranch)
        {
            float fluxNow = 0;
            [unroll] for (uint k17 = 0; k17 < MethodBands; k17++) fluxNow = max(fluxNow, Phi0[k17] * C[k17]);
            survive = min(survive, fluxNow / (Q * 4 * RayPi * ReferenceDistance * ReferenceDistance * EnergyBranchFloor));
        }
        if (PathIterationCap != 0 && iterations > PathIterationCap) survive *= PathIterationSurvival;
        float alive = PathSurvival(tau);
        survive *= alive / survival;
        survival = alive;
        if (survive < 1)
        {
            if (RandomRay(rng) >= survive) { MethodCount(CounterRoulette); done = true; continue; }
            Q *= survive;
        }
    }
    InterlockedMax(MethodStats[MethodBands * MethodDepths * MethodStatStride + CounterLongestChain], iterations);
    MethodCount(CounterIterations, iterations);
#if PATH_WALK_EDGE
    MethodCount(CounterEdgeConnections, walkSlot.count); // (walk diffraction: deposits offered to the slot)
#endif
    // Persistent: keep the stopped ray, and queue its branch reservoir as a ray of its own (writes after the
    // loop: on Metal, buffer writes inside a [loop] just before a return were dropped, part 14).
    if (suspending)
    {
        PathRay r;
        r.position = float4(p, tau); r.direction = float4(d, target);
        r.phiLow = float4(Phi0[0], Phi0[1], Phi0[2], Phi0[3]); r.phiHigh = float4(Phi0[4], Phi0[5], Phi0[6], Phi0[7]);
        r.wLow = float4(C[0], C[1], C[2], C[3]) / Q; r.wHigh = float4(C[4], C[5], C[6], C[7]) / Q;
        r.plane = plane; r.scalars = float4(survival, chain.travelled, chain.time, sinceStart);
        r.meta = uint4(chain.reflections | (inBranch ? 1u << 24 : 0u) | (scored ? 2u << 24 : 0u) | (offSurface ? 4u << 24 : 0u) | (chain.kind << 27), medium, rng, events);
#if PATH_DOPPLER
        r.motion = float4(rho, 0, 0, 0);
#else
        r.motion = 0; // (traced while nothing moved: no rate)
#endif
#if PATH_WALK_EDGE
        r.motion = float4(0, walkPhiIn, walkM, asfloat(walkMeta));
#endif
        PathKeep(r);
        float keepBranchRay = min(1, branchSum / (4 * RayPi * ReferenceDistance * ReferenceDistance * EnergyBranchFloor));
        if (branchSum > 0 && RandomRay(rng) < keepBranchRay)
        {
            MethodCount(CounterBranches);
            r.position = float4(branchP + branchPlane.xyz * RayEpsilon, branchTau); r.direction = float4(branchD, target);
            r.phiLow = float4(branchF[0], branchF[1], branchF[2], branchF[3]) * (branchSum / keepBranchRay);
            r.phiHigh = float4(branchF[4], branchF[5], branchF[6], branchF[7]) * (branchSum / keepBranchRay);
            r.wLow = 1; r.wHigh = 1;
            r.plane = branchPlane; r.scalars = float4(branchSurvival, branch.travelled, branch.time, 0);
            r.meta = uint4(branch.reflections | 1u << 24 | 4u << 24 | (branch.kind << 27), branchMedium, HashRay(rng ^ 0x68e31da4u), events);
#if PATH_DOPPLER
            r.motion = float4(branchRate, 0, 0, 0);
#else
            r.motion = 0;
#endif
            PathKeep(r);
        }
#if PATH_WALK_EDGE
        // The update's kept diffraction candidate, launched in the loop, queued as a diffracted ray of its own (as the branch above).
        if (walkQueued)
        {
            MethodCount(CounterEdgeEvents);
            r.position = float4(walkSlot.e, walkSlot.tau); r.direction = float4(walkD, target);
            r.phiLow = walkLow; r.phiHigh = walkHigh; r.wLow = 1; r.wHigh = 1;
            r.plane = float4(walkD, -1e30); r.scalars = float4(walkSlot.survival, 0, walkSlot.tau, 0);
            r.meta = uint4(1u << 24 | (2u << 27), AirMedium, HashRay(rng ^ 0x1b873593u), events);
            r.motion = float4(0, walkSlot.phiIn, walkSlot.m, asfloat(walkSlot.meta | PathWalkFirst));
            PathKeep(r);
        }
#endif
    }
}

// Audio: the direct arrival as one path per FFT bin (gain sqrt of the band-interpolated energy).
void PathExpand(uint bin)
{
    if (bin >= MethodFrequencyBins) return;
    RayPath path = (RayPath)0;
    if (MethodArrivalCounts[MethodCountIndex] > 0)
    {
        EnergyArrival a = MethodArrivals[MethodArrivalBase];
        float e = EnergyAt(a, bin * MethodSampleRate / MethodFftSize);
        if (e > 0)
        {
            path.direction = a.direction; path.delaySeconds = a.delay; path.pressureGain = sqrt(e);
            path.frequencyBin = bin; path.identity = a.identity;
        }
    }
    RayPaths[MethodPathBase + bin] = path;
}
#endif
