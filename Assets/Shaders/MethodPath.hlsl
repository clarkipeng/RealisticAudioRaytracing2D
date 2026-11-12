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
//     after every specular reflection or transmission: connect the chain to the listener
//       (PathDetect): if the ray left its vertex inside a small region around the listener's
//       mirror image, retrace the exact image path and score it divided by the probability of
//       that detection. The probability is the region's sampling probability times the share of
//       rays in the region that follow the same plane sequence; the share's inverse is estimated
//       without bias by the number of fresh rays drawn in the region until one follows it
//       (geometric trials), so a region that straddles a visibility boundary is not biased low.
//       The region size (specularDetectionRadius) changes variance and cost, never the mean.
//
// Uses MethodCommon.hlsl and the UTD and edge-sampling helpers of MethodEnergy.hlsl.
#ifndef ACOUSTIC_METHOD_PATH
#define ACOUSTIC_METHOD_PATH
// Time bins: PathBin0 wide up to PathFineBins * PathBin0, then constant relative resolution
// (each bin PathRatio times its start time).
float PathBin0, PathRatio;
uint PathFineBins, PathBins, PathCountBase, PathCountCells, PathAudio, PathMaxTrials;
RWStructuredBuffer<uint> PathCounts; // contributions per [solve][bin]
static const uint CounterTrialCap = 13, CounterTrials = 15;

uint PathBinOf(float time)
{
    float fine = PathBin0 * PathFineBins;
    if (time < fine) return (uint)floor(time / PathBin0);
    return PathFineBins + (uint)floor(log(time / fine) / log(1 + PathRatio));
}
// Adds band energies arriving from `from` (listener towards the arrival) at `time`. Directions
// split the energy over the six axes by the squared positive components (they sum to one), so a
// moving arrival never jumps between direction bins. 32.32 fixed point with an unbiased dither.
void PathDeposit(float3 from, float4 low, float4 high, float time, uint key)
{
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
    InterlockedAdd(PathCounts[PathCountBase + bin], 1u);
    [loop] for (uint axis = 0; axis < 6; axis++)
    {
        float component = (axis & 1) != 0 ? -u[axis >> 1] : u[axis >> 1];
        if (component <= 0) continue;
        [loop] for (uint k = 0; k < MethodBands; k++)
        {
            float4 lane = float4((k & 3) == uint4(0, 1, 2, 3));
            float share = dot(k < 4 ? low : high, lane) * component * component;
            if (!(share > 0) || !isfinite(share)) { if (!isfinite(share)) MethodCount(CounterNonFinite); continue; }
            uint cell = MethodCellBase + (axis * MethodBands + k) * PathBins + bin;
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
// non-delta vertex. Edge chains start at the edge point; their detection anchor is the source.
struct PathChain
{
    float3 origin, normal, omega; // first vertex, its surface normal (scattering point), first direction
    uint kind;                    // 0 source, 1 scattering point, 2 edge ray from the source
    uint edge; float edgeZ, edgePhi, edgePick;
    float3x3 unfold; float3 shift; // real space after the latest reflection -> unfolded space
    uint hits, reflections, hash; uint2 through; // plane sequence hash, transmitted events (bits)
    float travelled, time;        // unfolded length since origin; time at origin
    bool detectable;              // in air, every transmission a partition crossing
    float4 edgeLow, edgeHigh;     // edge chains: the edge ray's band flux (the walk's Phi0 at its first hit)
};
float PathEdgeFlux(PathChain c, uint band) { float4 v = band < 4 ? c.edgeLow : c.edgeHigh; return v[band & 3]; }
PathChain PathStart(float3 origin, float3 normal, float3 omega, uint kind, float time, bool air)
{
    PathChain c;
    c.origin = origin; c.normal = normal; c.omega = omega; c.kind = kind;
    c.edge = 0; c.edgeZ = 0; c.edgePhi = 0; c.edgePick = 1;
    c.unfold = MethodIdentity(); c.shift = 0;
    c.hits = 0; c.reflections = 0; c.hash = 0x2545f491u; c.through = 0;
    c.travelled = 0; c.time = time; c.detectable = air; c.edgeLow = 1; c.edgeHigh = 1;
    return c;
}
// One event of a chain: plane, transmitted or reflected (normal n, point p).
void PathEvent(inout PathChain c, uint plane, bool transmitted, float3 n, float3 p)
{
    c.hash = HashRay(c.hash ^ (plane * 2 + (transmitted ? 1u : 0u)));
    if (transmitted)
    {
        if (c.hits >= 64) c.detectable = false;
        else if (c.hits < 32) c.through.x |= 1u << c.hits; else c.through.y |= 1u << (c.hits - 32);
    }
    else
    {
        MethodUnfoldReflect(c.unfold, c.shift, n, dot(n, p));
        c.reflections++;
    }
    c.hits++;
}
// Follows `count` events of a chain from p along d: reflect, or cross the partition where the
// chain transmitted. Returns the plane sequence hash, end point and direction, unfolded length.
bool PathFollow(float3 p, float3 d, uint count, uint2 through, out uint hash, out float3 q, out float3 e, out float travelled)
{
    hash = 0x2545f491u; travelled = 0; q = p; e = d;
    float limit = MethodHorizon * SpeedOfSound;
    [loop] for (uint i = 0; i < count; i++)
    {
        uint hit; float distance; float3 normal;
        if (!NearestHit(MethodState, p, d, limit - travelled, false, hit, distance, normal)) return false;
        p += d * distance; travelled += distance;
        bool t = i < 64 && ((i < 32 ? through.x >> i : through.y >> (i - 32)) & 1) != 0;
        hash = HashRay(hash ^ (MethodPrimitivePlane[hit] * 2 + (t ? 1u : 0u)));
        if (t)
        {
            float wall;
            if (!PartitionExit(MethodState, hit, p, d, false, MethodRefracts(hit), wall)) return false;
            p += d * wall; travelled += wall;
        }
        else d = reflect(d, dot(d, normal) < 0 ? normal : -normal);
    }
    q = p; e = d;
    return true;
}
float3 PathCone(float3 w, float oneMinusCos, inout uint rng)
{
    float c = 1 - RandomRay(rng) * oneMinusCos, s = sqrt(max(0, 1 - c * c)), a = 2 * RayPi * RandomRay(rng);
    float3 t = abs(w.x) < .9 ? normalize(cross(w, float3(1, 0, 0))) : normalize(cross(w, float3(0, 1, 0)));
    return c * w + s * (cos(a) * t + sin(a) * cross(w, t));
}

// Connection of a specular chain to the listener (see the header). flux: the walk's
// Phi0 C / Q right after the chain's latest event (edge chains: over the edge ray's flux, i.e.
// the chain's C / Q, which also holds for a transmitted branch of the chain). Point origins: the region is a cone about the
// direction to the listener image with half-angle asin(r / distance) (at most 30 degrees);
// trial directions are uniform in it and, for a scattering point, accepted with probability
// cos / (largest cos in the cone), so trials follow the walk's own density. Edge chains: the
// region is a box in (point on the edge, azimuth) about Keller's stationary point, sampled with
// the walk's densities. score = exact contribution x N / (region probability, without p_seq).
// Bands travel as two float4 (125-1000 Hz, 2-16 kHz): array parameters in and out of this
// function, and writes to out parameters inside a [loop], lost the result on Metal.
bool PathDetect(PathChain c, float4 fluxLow, float4 fluxHigh, float3 S, float3 L, inout uint rng, out float4 low, out float4 high, out float time, out float3 from)
{
    low = 0; high = 0;
    time = 0; from = 0;
    float3 image = mul(c.unfold, L) + c.shift;
    bool edgeChain = c.kind == 2;
    MethodCount(CounterDetections);
    // Edge chain: Keller's stationary point between the source and the listener image.
    float3 a = 0, u = 0, f = 0, nn = 0; float extent = 1, angle = 1, n = 1, phiIn = 0, phiStar = 0, sinBeta = 1, zStar = 0, m = 0, l = 0;
    float z0 = 0, re = 1, span = 1, lowZ = 0, highZ = 0, lowPhi = 0, highPhi = 0, oneMinus = 0, cmax = 1, region = 1;
    float3 start = c.origin, toward = 0;
    if (edgeChain)
    {
        float3 e;
        EdgeFrame(c.edge, MethodState, false, a, u, f, nn, extent, angle);
        if (!UtdApex(c.edge, S, image, e) || !UtdAngles(c.edge, S, e, image, n, phiIn, phiStar, sinBeta)) return false;
        zStar = dot(e - a, u); m = length(e - S); l = length(image - e);
        if (m <= RayEpsilon || l <= RayEpsilon) return false;
        float r = MethodDetectRadius, dz = min(extent, r * m / ((m + l) * sinBeta)), dphi = min(.5 * angle, r / (l * sinBeta));
        lowZ = max(0, zStar - dz); highZ = min(extent, zStar + dz); lowPhi = max(0, phiStar - dphi); highPhi = min(angle, phiStar + dphi);
        if (c.edgeZ < lowZ || c.edgeZ > highZ || c.edgePhi < lowPhi || c.edgePhi > highPhi) return false;
        span = EnergyEdgeSpan(a, u, extent, S, z0, re);
        // Region probability: edge ray, this edge, z in the box (Cauchy law), azimuth in the box.
        region = MethodEdgeProbability * c.edgePick * (atan((highZ - z0) / re) - atan((lowZ - z0) / re)) / span * (highPhi - lowPhi) / angle;
        start = e; toward = normalize(image - e);
    }
    else
    {
        float3 w = image - c.origin;
        float distance = length(w);
        if (distance <= RayEpsilon) return false;
        w /= distance;
        float s = min(MethodDetectRadius / distance, .5), cosA = sqrt(1 - s * s);
        if (dot(c.omega, w) < cosA) return false;
        oneMinus = s * s / (1 + cosA);
        if (c.kind == 1)
        {
            float cw = dot(w, c.normal);
            if (cw <= 0) return false;
            cmax = cos(max(0, acos(min(1, cw)) - asin(s)));
        }
        region = 2 * RayPi * oneMinus; // solid angle; the density bound is folded into the score
        toward = w;
    }
    // Attempt 0 is the exact image path (with its last leg to the listener); attempts 1.. are the
    // geometric trials. One visibility call site: the edge chain's first leg, then the last leg.
    float total = edgeChain ? l : length(image - c.origin);
    uint trials = 0;
    float3 qExact = 0, eExact = 0;
    [loop] for (uint attempt = 0; attempt <= PathMaxTrials; attempt++)
    {
        float3 p = start, d = toward;
        bool valid = true;
        if (attempt > 0)
        {
            if (edgeChain)
            {
                // The walk's edge ray restricted to the box.
                float t0 = atan((lowZ - z0) / re), t1 = atan((highZ - z0) / re);
                float z = clamp(z0 + re * tan(t0 + RandomRay(rng) * (t1 - t0)), 0, extent), phi = lowPhi + RandomRay(rng) * (highPhi - lowPhi);
                p = a + z * u;
                float3 w = p - S; float mw = length(w);
                valid = mw > RayEpsilon && TunedEdgeAngle(-w, f, nn, angle) <= angle;
                w /= max(mw, 1e-12);
                float cosBeta = dot(w, u), sb = sqrt(max(0, 1 - cosBeta * cosBeta));
                valid = valid && sb > 1e-4;
                d = normalize(u * cosBeta + sb * (f * cos(phi) + nn * sin(phi)));
            }
            else
            {
                d = PathCone(toward, oneMinus, rng);
                if (c.kind == 1) valid = RandomRay(rng) * cmax <= max(0, dot(d, c.normal));
            }
        }
        uint hash = 0; float3 q = 0, e = 0; float travelled = 0;
        bool follows = false;
        if (valid) follows = PathFollow(p, d, c.hits, c.through, hash, q, e, travelled) && hash == c.hash; // (no short-circuit in HLSL)
        if (attempt == 0 && follows)
        {
            float remaining = total - travelled;
            follows = remaining > RayEpsilon && length(q + e * remaining - L) <= 1e-3 * max(1, total);
        }
        [loop] for (uint leg = 0; leg < 2; leg++)
        {
            // The edge chain's first leg (source -> edge point) as for the walk; the exact path's last leg.
            bool need = follows && (leg == 0 ? edgeChain : attempt == 0);
            if (need) follows = Visible(MethodState, leg == 0 ? S : q, leg == 0 ? p : L, false);
        }
        if (attempt == 0)
        {
            if (!follows) return false;
            qExact = q; eExact = e;
            continue;
        }
        MethodCount(CounterTrials);
        if (follows) { trials = attempt; break; }
    }
    if (trials == 0) { MethodCount(CounterTrialCap); trials = PathMaxTrials; }
    MethodCount(CounterChains);
    from = -eExact;
    float ratio = (float)trials / region, s2 = sinBeta * sinBeta;
    time = edgeChain ? (m + l) * MethodSlowness(AirMedium) : c.time + total * MethodSlowness(AirMedium);
    // Point origin: flux (pdf/D)(omega) D(exact) / total^2 over the density bound M:
    // source pdf/D = M = 1/(4 pi); scattering point pdf/D = 1/pi, M = cmax/pi.
    float shape = c.kind == 1 ? max(0, dot(toward, c.normal)) / cmax : 1;
    float4 sumLow = 0, sumHigh = 0; // written to the out parameters after the loop (writes to them inside a [loop] were lost on Metal)
    [loop] for (uint k = 0; k < MethodBands; k++)
    {
        float4 lane = float4((k & 3) == uint4(0, 1, 2, 3)); // one-hot: no dynamic vector writes in a loop
        float flux = dot(k < 4 ? fluxLow : fluxHigh, lane), value;
        if (edgeChain)
        {
            float opacity = MethodEdgeOpacity(c.edge, k);
            value = ReferenceDistance * ReferenceDistance * UtdEnergy(n, phiIn, phiStar, sinBeta, MethodWavenumber(k), m * l / (m + l) * s2) *
                    opacity * opacity * DiffractionFactor * DiffractionFactor * flux *
                    exp(-2 * MethodLoss(AirMedium, k) * (m + l - c.travelled)) / (m * l * (m + l)) * ratio;
        }
        else value = flux * shape * exp(-2 * MethodLoss(AirMedium, k) * (total - c.travelled)) / (total * total) * ratio;
        if (k < 4) sumLow += value * lane; else sumHigh += value * lane;
    }
    low = sumLow; high = sumHigh;
    return true;
}

// Deferred chain connections. A detection's retrace and trials take tens to hundreds of ray
// casts while the other rays of its SIMD group wait, so the walk only makes the cheap region test
// and queues the candidates; PathConnect then runs one candidate per thread. Same estimator.
struct PathRequest
{
    float4 originKind, normalHits, omegaHash, unfold0, unfold1, unfold2, edge, travel, fluxLow, fluxHigh;
};
RWStructuredBuffer<PathRequest> PathRequests;
RWStructuredBuffer<uint> PathRequestCount; // [0] requests of the current ray chunk
uint PathRequestCapacity;
static const uint CounterRequestOverflow = 9;
// The walk's region test: could this ray's direction lie in its chain's detection region?
// Point chains: the cone about the listener image. Edge chains: always queued (box test there).
bool PathCandidate(PathChain c, float3 L)
{
    if (c.kind == 2) return true;
    float3 w = mul(c.unfold, L) + c.shift - c.origin;
    float distance = length(w);
    if (distance <= RayEpsilon) return false;
    float s = min(MethodDetectRadius / distance, .5);
    return dot(c.omega, w / distance) >= sqrt(1 - s * s);
}
void PathQueue(PathChain c, float flux[8])
{
    uint at;
    InterlockedAdd(PathRequestCount[0], 1u, at);
    if (at >= PathRequestCapacity) { MethodCount(CounterRequestOverflow); return; }
    PathRequest r;
    r.originKind = float4(c.origin, asfloat(c.kind));
    r.normalHits = float4(c.normal, asfloat(c.hits));
    r.omegaHash = float4(c.omega, asfloat(c.hash));
    r.unfold0 = float4(c.unfold[0], c.shift.x); r.unfold1 = float4(c.unfold[1], c.shift.y); r.unfold2 = float4(c.unfold[2], c.shift.z);
    r.edge = float4(asfloat(c.edge), c.edgeZ, c.edgePhi, c.edgePick);
    r.travel = float4(c.travelled, c.time, asfloat(c.through.x), asfloat(c.through.y));
    r.fluxLow = float4(flux[0], flux[1], flux[2], flux[3]); r.fluxHigh = float4(flux[4], flux[5], flux[6], flux[7]);
    PathRequests[at] = r;
}
void PathConnect(uint index)
{
    if (index >= min(PathRequestCount[0], PathRequestCapacity)) return;
    PathRequest r = PathRequests[index];
    PathChain c;
    c.origin = r.originKind.xyz; c.kind = asuint(r.originKind.w); c.normal = r.normalHits.xyz; c.hits = asuint(r.normalHits.w);
    c.omega = r.omegaHash.xyz; c.hash = asuint(r.omegaHash.w);
    c.unfold = float3x3(r.unfold0.xyz, r.unfold1.xyz, r.unfold2.xyz); c.shift = float3(r.unfold0.w, r.unfold1.w, r.unfold2.w);
    c.edge = asuint(r.edge.x); c.edgeZ = r.edge.y; c.edgePhi = r.edge.z; c.edgePick = r.edge.w;
    c.travelled = r.travel.x; c.time = r.travel.y; c.through = uint2(asuint(r.travel.z), asuint(r.travel.w));
    c.reflections = 0; c.detectable = true; c.edgeLow = 1; c.edgeHigh = 1;
    float3 S = States[MethodState + MethodSourceObject].position.xyz, L = States[MethodState + MethodListenerObject].position.xyz;
    uint rng = HashRay(index * 0x9e3779b9u ^ HashRay(RaySeed * 0x27d4eb2fu + MethodRayOffset * 0x165667b1u + 0x61c88647u));
    float4 low, high; float time; float3 from;
    if (PathDetect(c, r.fluxLow, r.fluxHigh, S, L, rng, low, high, time, from))
        PathDeposit(from, low, high, time, HashRay(rng ^ 0x5bd1e995u));
}
void PathResetRequests() { PathRequestCount[0] = 0; }

// Straight connection of a non-delta vertex to the listener (through transmitting walls).
bool PathListener(MethodAnchor v, float I[8], float3 L, float scale, out float4 low, out float4 high, out float time, out float3 from, out float lengthOut)
{
    low = 0; high = 0;
    time = 0; from = 0; lengthOut = 0;
    float3 w = L - v.position;
    float d = length(w);
    if (d <= RayEpsilon) return false;
    w /= d;
    float D = EnergyDirectivity(v, w);
    if (D == 0) return false;
    float factor[8]; float t;
    [unroll] for (uint k = 0; k < MethodBands; k++) factor[k] = 1;
    if (!MethodConnect(v.position, L, v.medium, true, factor, t)) return false;
    float r = max(MinimumDistance, d);
    float e[8];
    [unroll] for (uint k1 = 0; k1 < MethodBands; k1++) e[k1] = I[k1] * D * factor[k1] / (r * r) * scale;
    low = float4(e[0], e[1], e[2], e[3]); high = float4(e[4], e[5], e[6], e[7]);
    time = v.time + t; from = -w; lengthOut = d;
    return true;
}
// Listener connection through one edge (D), one mirror and one edge (RD), or two parallel
// edges (DD): EnergyEdgeConnection's kernels, with one UtdEnergy body per edge (bands in a loop).
bool PathEdgeConnection(MethodAnchor v, float4 inLow, float4 inHigh, float3 L, inout uint rng, out float4 low, out float4 high, out float time, out float3 fromOut)
{
    low = 0; high = 0;
    time = 0; fromOut = 0;
    if (MethodEdgeCount == 0 || DiffractionFactor <= 0) return false;
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
    if (image) { if (!CompleteMirror(mirror, MethodState, v.position, e, false, hit, cosine)) return false; first = hit; }
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
        float share = UtdEnergy(n, phiIn, phiOut, sinBeta, MethodWavenumber(k), m * (d + l) / total * s2) * dot(k < 4 ? inLow : inHigh, lane) * D *
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

// One flat loop; every heavy helper has a single call site. Each iteration either deposits the
// pending contribution, makes one connection, or traces one segment.
void TracePath(uint index)
{
    if (index >= MethodRayCount) return;
    uint ray = MethodRayOffset + index;
    MethodCount(CounterRays);
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
    float I[8], cI[8], Phi0[8], C[8], flux[8];
    float4 pendingLow = 0, pendingHigh = 0; // bands of the pending contribution (no arrays across function boundaries)
    [unroll] for (uint k = 0; k < MethodBands; k++)
    { I[k] = ReferenceDistance * ReferenceDistance; cI[k] = I[k]; Phi0[k] = 0; C[k] = 1; flux[k] = 0; }
    MethodAnchor c = v;
    // Ray 0 makes the source's straight connection for every ray (it is deterministic).
    bool connectListener = ray == 0, connectEdge = true, hasPending = false;
    float listenerScale = MethodTotalRays, pendingTime = 0;
    float3 pendingFrom = 0;
    uint events = 0;
    bool edges = MethodEdgeCount > 0 && DiffractionFactor > 0, travelling = false, done = false, fluxPending = false, inBranch = false;
    float3 p = S, d = 0;
    float Q = 1, tau = 0;
    uint medium = v.medium;
    PathChain chain = PathStart(S, 0, 0, 0, 0, false), branch = chain;
    float branchSum = 0, branchTau = 0; float3 branchP = 0, branchD = 0; float branchF[8];
    uint branchMedium = 0;
    [unroll] for (uint k1 = 0; k1 < MethodBands; k1++) branchF[k1] = 0;
    float edgeExtent = 1, edgeAngle = 1, edgeM = 1, edgeSinBeta = 1, edgePhiIn = 0, edgeD = 1;
    [loop] for (;;)
    {
        if (hasPending)
        {
            hasPending = false;
            PathDeposit(pendingFrom, pendingLow, pendingHigh, pendingTime, HashRay(rng ^ (events * 0x85ebca6bu)));
        }
        if (connectListener)
        {
            connectListener = false;
            float length0;
            hasPending = PathListener(c, cI, L, listenerScale, pendingLow, pendingHigh, pendingTime, pendingFrom, length0);
            if (hasPending && PathAudio != 0 && c.kind == 0)
            {
                PathDirectArrival(S, L, pendingLow, pendingHigh, listenerScale, pendingTime, length0); // exact, rendered as a delayed path
                hasPending = false;
            }
            continue;
        }
        if (connectEdge)
        {
            connectEdge = false;
            hasPending = PathEdgeConnection(c, float4(cI[0], cI[1], cI[2], cI[3]), float4(cI[4], cI[5], cI[6], cI[7]), L, rng, pendingLow, pendingHigh, pendingTime, pendingFrom);
            continue;
        }
        if (done)
        {
            if (!(branchSum > 0)) break;
            // Trace the kept transmitted branch from its surface, after roulette on its share of a
            // source ray's flux (4 pi r_ref^2).
            float keep = min(1, branchSum / (4 * RayPi * ReferenceDistance * ReferenceDistance * EnergyBranchFloor));
            if (RandomRay(rng) >= keep) break;
            [unroll] for (uint k2 = 0; k2 < MethodBands; k2++) { Phi0[k2] = branchF[k2] * branchSum / keep; C[k2] = 1; }
            p = branchP; d = branchD; medium = branchMedium; tau = branchTau; Q = 1;
            chain = branch;
            fluxPending = false; travelling = true; done = false; branchSum = 0; inBranch = true;
            MethodCount(CounterBranches);
        }
        if (!travelling)
        {
            if (++events >= MethodMaxEvents) { MethodCount(CounterEventCap); break; }
            bool edgeRays = edges && v.kind == 0 && MethodEdgeCdf[max(MethodEdgeCount, 1u) - 1] > 0;
            bool viaEdge = edgeRays && RandomRay(rng) < MethodEdgeProbability;
            [unroll] for (uint k3 = 0; k3 < MethodBands; k3++) C[k3] = 1;
            medium = v.medium;
            bool air = v.medium == AirMedium && airListener;
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
                tau = v.time + m * MethodSlowness(AirMedium);
                p = e; Q = 1; fluxPending = true;
                chain = PathStart(e, 0, d, 2, tau, air);
                chain.edge = edge; chain.edgeZ = z; chain.edgePhi = phi; chain.edgePick = edgePick;
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
                chain = PathStart(v.position, v.normal, w0, v.kind, v.time, air);
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
            [loop] for (uint k6 = 0; k6 < MethodBands; k6++)
            {
                float opacity = MethodEdgeOpacity(chain.edge, k6);
                Phi0[k6] = UtdEnergy(edgeAngle / RayPi, edgePhiIn, chain.edgePhi, edgeSinBeta, MethodWavenumber(k6), spread * edgeSinBeta * edgeSinBeta) *
                           edgeSinBeta * edgeSinBeta * Phi0[k6] * edgeD * opacity * opacity * DiffractionFactor * DiffractionFactor *
                           exp(-2 * MethodLoss(AirMedium, k6) * edgeM) / (edgeM * edgeM) * edgeExtent * edgeAngle / (chain.edgePick * MethodEdgeProbability);
            }
            chain.edgeLow = float4(Phi0[0], Phi0[1], Phi0[2], Phi0[3]); chain.edgeHigh = float4(Phi0[4], Phi0[5], Phi0[6], Phi0[7]);
        }
        if (!found) { MethodCount(CounterEscapes); done = true; continue; }
        p += d * distance; tau += distance * MethodSlowness(medium); chain.travelled += distance;
        [unroll] for (uint k7 = 0; k7 < MethodBands; k7++) C[k7] *= exp(-2 * MethodLoss(medium, k7) * distance);
        float3 n = dot(d, normal) < 0 ? normal : -normal;
        float cosI = max(0, -dot(d, n));
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
        if (mD > 0)
        {
            [unroll] for (uint k9 = 0; k9 < MethodBands; k9++) cI[k9] = Phi0[k9] * C[k9] / Q * eD[k9] / RayPi;
            c = x; listenerScale = 1;
            connectListener = connectEdge = true;
        }
        float total = mT + mD + mS;
        if (!(total > 0)) { done = true; continue; }
        uint plane = MethodPrimitivePlane[hit];
        // Where a transmission would go (one call site each): out of a partition's far face,
        // or into the medium behind the surface.
        float wall = 0; bool partition = false; uint next = medium;
        if (mT > 0)
        {
            partition = PartitionExit(MethodState, hit, p, d, false, MethodRefracts(hit), wall);
            next = MediumAt(MethodState, p + d * ((partition ? wall : 0) + 2 * RayEpsilon), false);
        }
        bool split = mT > 0 && mT < .5 * total;
        if (split)
        {
            // Small transmitted share: a reservoir candidate (MethodEnergy.hlsl header), not a draw.
            float part[8], weight = 0;
            [unroll] for (uint k10 = 0; k10 < MethodBands; k10++) { part[k10] = Phi0[k10] * C[k10] * eT[k10] / Q; weight = max(weight, part[k10]); }
            if (weight > 0)
            {
                branchSum += weight;
                if (RandomRay(rng) * branchSum < weight)
                {
                    branch = chain;
                    if (partition)
                    {
                        // Through a partition: undeviated, out of its far face.
                        branchP = p + d * wall; branchD = d; branchMedium = next;
                        branchTau = tau + wall * MethodSlowness(medium);
                        PathEvent(branch, plane, true, n, p);
                        branch.travelled += wall;
                    }
                    else
                    {
                        float3 bent = refract(d, n, MethodIor(medium, 3) / MethodIor(next, 3));
                        bool refracted = dot(bent, bent) > 0;
                        branchP = p; branchD = refracted ? normalize(bent) : reflect(d, n); branchMedium = refracted ? next : medium;
                        branchTau = tau;
                        if (refracted) branch.detectable = false; else PathEvent(branch, plane, false, n, p);
                    }
                    [unroll] for (uint k11 = 0; k11 < MethodBands; k11++) branchF[k11] = part[k11] / weight;
                }
            }
            total = mD + mS;
        }
        float pick = RandomRay(rng) * total;
        if (pick < mD)
        {
            [unroll] for (uint k12 = 0; k12 < MethodBands; k12++) I[k12] = cI[k12] * (total / mD);
            v = x;
            travelling = false;
            MethodCount(CounterDiffuse);
            if (v.time > MethodHorizon) { MethodCount(CounterHorizon); done = true; }
            continue;
        }
        bool transmit = !split && pick < mD + mT;
        if (transmit)
        {
            [unroll] for (uint k13 = 0; k13 < MethodBands; k13++) C[k13] *= eT[k13];
            Q *= mT / total;
            if (partition)
            {
                PathEvent(chain, plane, true, n, p);
                tau += wall * MethodSlowness(medium); p += d * wall; chain.travelled += wall;
                medium = next;
            }
            else
            {
                float3 bent = refract(d, n, MethodIor(medium, 3) / MethodIor(next, 3));
                if (dot(bent, bent) > 0) { d = normalize(bent); medium = next; chain.detectable = false; }
                else { PathEvent(chain, plane, false, n, p); d = reflect(d, n); }
            }
        }
        else
        {
            [unroll] for (uint k14 = 0; k14 < MethodBands; k14++) C[k14] *= eS[k14];
            Q *= mS / total;
            PathEvent(chain, plane, false, n, p);
            d = reflect(d, n);
        }
        // Connect the chain to the listener (PathConnect), with the flux before roulette.
        if (chain.detectable && (chain.kind == 2 ? chain.hits >= 1 : chain.reflections >= 1) && PathCandidate(chain, L))
        {
            bool any = false;
            [unroll] for (uint k15 = 0; k15 < MethodBands; k15++)
            {
                float scale = chain.kind == 2 ? PathEdgeFlux(chain, k15) : 1;
                flux[k15] = scale > 0 ? Phi0[k15] * C[k15] / Q / scale : 0;
                any = any || flux[k15] > 0;
            }
            if (any) PathQueue(chain, flux);
        }
        float weight = 0;
        [unroll] for (uint k16 = 0; k16 < MethodBands; k16++) weight = max(weight, C[k16]);
        float survive = min(1, weight / (Q * MethodRouletteFloor));
        if (inBranch)
        {
            float fluxNow = 0;
            [unroll] for (uint k17 = 0; k17 < MethodBands; k17++) fluxNow = max(fluxNow, Phi0[k17] * C[k17]);
            survive = min(survive, fluxNow / (Q * 4 * RayPi * ReferenceDistance * ReferenceDistance * EnergyBranchFloor));
        }
        if (survive < 1)
        {
            if (RandomRay(rng) >= survive) { MethodCount(CounterRoulette); done = true; continue; }
            Q *= survive;
        }
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
