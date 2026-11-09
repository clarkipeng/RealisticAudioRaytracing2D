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
//     weight is multiplied by its own fraction / that probability;
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
//   - specular chains: point-to-point delta paths cannot be sampled by any
//     unidirectional or bidirectional estimator (a delta vertex next to a point
//     receiver), so the walk detects them: the exact image path is retraced and scored
//     with its exact energy divided by the probability that the walk's ray passed
//     within MethodDetectRadius of the listener image (MethodCommon.hlsl). Unlike a
//     receiver sphere it scores the exact path (delay, direction, energy), not a chord,
//     and it is unbiased wherever the detection cone lies in the chain's visibility
//     beam; a glossy-lobe regularization would instead bias every reflection by a
//     width our material data does not specify.
// Russian roulette: survival min(1, max over bands of energy factor / probability).
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
    EdgeFrame(edge, 0, false, a, u, f, n, extent, angle);
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
    EdgeFrame(edge, 0, false, a, u, f, nn, extent, angle);
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
    EdgeFrame(first, 0, false, a1, u1, f1, n1, e1, w1);
    EdgeFrame(second, 0, false, a2, u2, f2, n2, e2, w2);
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
        if ((States[Primitives[mirror].indices.x].flags.x & 3) != 3) return;
        from = CompleteImage(mirror, 0, v.position, false, unused);
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
    float3 last = pair ? e2 : e;
    first = e;
    if (image) { if (!CompleteMirror(mirror, 0, v.position, e, false, hit, cosine)) return; first = hit; }
    // Both straight legs (anchor -> first edge unless mirrored, last edge -> listener) at one call site.
    [loop] for (uint segment = image ? 1u : 0u; segment < 2; segment++)
        if (!Visible(0, segment == 0 ? v.position : last, segment == 0 ? e : L, false)) return;
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

// One flat loop: each iteration makes the pending connections of the latest vertex,
// then traces one segment (an edge ray or a direction from the anchor starts a new
// chain), so every heavy helper has a single call site (small kernel and stack).
void TraceEnergy(uint index)
{
    if (index >= MethodRayCount) return;
    uint ray = MethodRayOffset + index, slot = 0;
    MethodCount(CounterRays);
    float3 S = States[Sources[0]].position.xyz, L = States[Listeners[0]].position.xyz;
    uint rng = HashRay(ray * 0x9e3779b9u ^ HashRay(RaySeed * 0xc2b2ae35u + 0x2545f491u));
    MethodAnchor v;
    v.position = S; v.normal = 0; v.previous = S; v.kind = 0; v.edge = 0; v.primitive = 0xffffffff; v.time = 0;
    v.medium = MediumAt(0, S, false);
    float I[8], cI[8], Phi0[8], C[8];
    [unroll] for (uint k = 0; k < MethodBands; k++) { I[k] = ReferenceDistance * ReferenceDistance; cI[k] = I[k]; Phi0[k] = 0; C[k] = 1; }
    MethodCaptureEvent(ray, slot, CaptureSource, S, 1, 0, 0);
    // Pending connections of vertex c; ray 0 carries the deterministic direct sound for every ray.
    MethodAnchor c = v;
    bool connectListener = ray == 0, connectEdge = true;
    float listenerScale = MethodTotalRays;
    uint depth = 0, connectDepth = 0, events = 0;
    bool edges = MethodEdgeCount > 0 && DiffractionFactor > 0, travelling = false, done = false;
    // Chain state. An edge ray's flux needs the distance to its first hit (UTD's
    // distance parameter), so it is completed there (fluxPending).
    float3 p = S, d = 0, w0 = 0, Uc = 0, chainStart = S;
    float3x3 U = MethodIdentity();
    float Q = 1, tau = 0;
    uint hits = 0, reflections = 0, medium = v.medium, edge = 0;
    uint2 transmitted = 0;
    bool straight = true, viaEdge = false, fluxPending = false;
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
        if (done) break;
        if (!travelling)
        {
            if (++events >= MethodMaxEvents) { MethodCount(CounterEventCap); break; }
            viaEdge = edges && RandomRay(rng) < MethodEdgeProbability;
            [unroll] for (uint k = 0; k < MethodBands; k++) C[k] = 1;
            Uc = 0; U = MethodIdentity(); hits = 0; reflections = 0; transmitted = 0; straight = true; medium = v.medium;
            if (viaEdge)
            {
                // Edge ray: point uniform on an edge, azimuth uniform on Keller's cone.
                float3 a, u, f, nn; float extent, angle;
                edge = MethodPickEdge(v.kind == 1 ? v.primitive : 0xffffffff, rng, edgePick);
                EdgeFrame(edge, 0, false, a, u, f, nn, extent, angle);
                float3 e = a + RandomRay(rng) * extent * u, w = e - v.position;
                float m = length(w);
                if (m <= RayEpsilon || TunedEdgeAngle(-w, f, nn, angle) > angle || !Visible(0, v.position, e, false))
                { MethodCaptureEvent(ray, slot, CaptureEnd, v.position, 0, 0, v.time); break; }
                w /= m;
                float cosBeta = dot(w, u);
                edgeSinBeta = sqrt(max(0, 1 - cosBeta * cosBeta));
                edgeD = EnergyDirectivity(v, w);
                if (edgeD == 0 || edgeSinBeta <= 1e-4) { MethodCaptureEvent(ray, slot, CaptureEnd, v.position, 0, 0, v.time); break; }
                edgePhiIn = TunedEdgeAngle(v.position - e, f, nn, angle);
                edgePhi = RandomRay(rng) * angle;
                d = normalize(u * cosBeta + edgeSinBeta * (f * cos(edgePhi) + nn * sin(edgePhi)));
                edgeExtent = extent; edgeAngle = angle; edgeM = m;
                tau = v.time + m * MethodSlowness(AirMedium);
                p = e; chainStart = e; w0 = d; Q = 1; fluxPending = true;
                depth++;
                MethodCount(CounterEdgeEvents);
                MethodCaptureEvent(ray, slot, CaptureEdge, e, 0, 0, tau);
            }
            else
            {
                w0 = v.kind == 1 ? MethodCosineDirection(v.normal, rng) : UnitDirection(rng, false);
                float pdf = v.kind == 1 ? max(1e-12, dot(w0, v.normal)) / RayPi : 1 / (4 * RayPi);
                float D = EnergyDirectivity(v, w0);
                if (D == 0) break;
                [unroll] for (uint k = 0; k < MethodBands; k++) Phi0[k] = I[k] * D / pdf;
                Q = edges ? 1 - MethodEdgeProbability : 1;
                p = v.position; chainStart = p; d = w0; tau = v.time;
            }
            travelling = true;
        }
        if (++events >= MethodMaxEvents) { MethodCount(CounterEventCap); break; }
        float limit = (MethodHorizon - tau) / MethodSlowness(medium);
        uint hit; float distance; float3 normal;
        if (limit <= 0) { MethodCount(CounterHorizon); break; }
        if (!NearestHit(0, p, d, limit, false, hit, distance, normal))
        {
            MethodCount(CounterEscapes);
            MethodCaptureEvent(ray, slot, CaptureEscape, p + d * min(limit, 50), 0, 0, tau);
            break;
        }
        if (fluxPending)
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
        bool transmit = u < mD + mT;
        if (transmit)
        {
            uint next = MediumAt(0, p + d * (2 * RayEpsilon), false);
            float3 bent = refract(d, n, MethodIor(medium, 3) / MethodIor(next, 3));
            [unroll] for (uint k = 0; k < MethodBands; k++) C[k] *= eT[k];
            Q *= mT / total;
            if (dot(bent, bent) > 0)
            {
                if (MethodIor(next, 3) != MethodIor(medium, 3)) straight = false;
                d = normalize(bent);
                medium = next;
                if (hits < 64) { if (hits < 32) transmitted.x |= 1u << hits; else transmitted.y |= 1u << (hits - 32); }
                else straight = false;
            }
            else { d = reflect(d, n); straight = false; }
        }
        else
        {
            [unroll] for (uint k = 0; k < MethodBands; k++) C[k] *= eS[k];
            Q *= mS / total;
            d = reflect(d, n);
            MethodUnfoldReflect(U, Uc, n, dot(n, p));
            reflections++;
        }
        hits++;
        float weight = 0;
        [unroll] for (uint k = 0; k < MethodBands; k++) weight = max(weight, C[k]);
        MethodCaptureEvent(ray, slot, transmit ? CaptureTransmission : CaptureSpecular, p, Phi0[2] * C[2] / Q, 0, tau);
        if (reflections > 0 && straight)
        {
            float3 image = mul(U, L) + Uc, w = image - chainStart;
            float span = length(w), along = dot(w, w0);
            if (along > 0 && span > MethodDetectRadius && span * span - along * along < MethodDetectRadius * MethodDetectRadius)
            {
                MethodCount(CounterDetections);
                // Exact path start and the factor that multiplies the chain's exact
                // energy coefficients; after an edge ray the path passes Keller's
                // stationary point and the detection probability is the edge-sample
                // density times the (z, phi) area of a disc of radius r in the ray tube.
                float factor[8]; float3 origin = 0; float m = 0;
                bool valid = false;
                w /= span;
                if (!viaEdge)
                {
                    float Dw = EnergyDirectivity(v, w), P = MethodConeProbability(v.kind == 1 ? 1 : 0, w, span, v.normal);
                    origin = v.kind == 1 ? v.position + v.normal * RayEpsilon : v.position;
                    valid = Dw != 0 && P > 0;
                    [unroll] for (uint k = 0; k < MethodBands; k++) factor[k] = I[k] * Dw / (P * Q);
                }
                else
                {
                    float3 e; float nw, phiIn, phiOut, sinBeta;
                    if (UtdApex(edge, v.position, image, e) && UtdAngles(edge, v.position, e, image, nw, phiIn, phiOut, sinBeta))
                    {
                        m = length(e - v.position);
                        float l = length(image - e), Dw = EnergyDirectivity(v, (e - v.position) / max(m, 1e-12));
                        float P = edgePick * MethodEdgeProbability / (edgeExtent * edgeAngle) * RayPi * MethodDetectRadius *
                                  MethodDetectRadius * m / (l * sinBeta * sinBeta * (l + m));
                        origin = e;
                        valid = Dw != 0 && m > RayEpsilon && Visible(0, v.position, e, false);
                        [unroll] for (uint k = 0; k < MethodBands; k++)
                        {
                            float opacity = MethodEdgeOpacity(edge, k);
                            factor[k] = UtdEnergy(nw, phiIn, phiOut, sinBeta, MethodWavenumber(k), m * l / (m + l) * sinBeta * sinBeta) *
                                        I[k] * Dw * opacity * opacity * DiffractionFactor * DiffractionFactor * exp(-2 * MethodLoss(AirMedium, k) * m) /
                                        (m * l * (m + l) * P * Q);
                        }
                    }
                    else [unroll] for (uint k = 0; k < MethodBands; k++) factor[k] = 0;
                }
                float2 unusedPressure[8]; float exact[8]; float total0; float3 from;
                if (valid && MethodRetraceChain(origin, image, L, hits, transmitted, unusedPressure, exact, total0, from))
                {
                    MethodCount(CounterChains);
                    float r = max(MinimumDistance, total0), magnitude = 0;
                    float time = v.time + (m + total0) * MethodSlowness(AirMedium);
                    [unroll] for (uint k = 0; k < MethodBands; k++)
                    {
                        float share = factor[k] * exact[k] * exp(-2 * MethodLoss(AirMedium, k) * total0) / (viaEdge ? 1 : r * r);
                        EnergyDeposit(from, k, time, share, depth, HashRay(rng ^ (k + 77)));
                        if (k == 2) magnitude = share;
                    }
                    MethodCaptureEvent(ray, slot, CaptureChain, p, 0, magnitude, time);
                }
            }
        }
        float survive = min(1, weight / Q);
        if (survive < 1)
        {
            if (RandomRay(rng) >= survive)
            { MethodCount(CounterRoulette); MethodCaptureEvent(ray, slot, CaptureEnd, p, 0, 0, tau); done = true; continue; }
            Q *= survive;
        }
    }
}
