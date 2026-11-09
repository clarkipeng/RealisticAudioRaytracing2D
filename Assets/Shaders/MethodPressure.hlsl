// Method "pressure": one coherent complex-pressure path-tracing loop (offline
// measurement kernel; docs/benchmarks/methods-2026-11-06). Every interaction is an
// event of the same loop and every listener contribution is (path integrand) /
// (probability density of the sampled path). Nothing is clipped, fitted or capped
// except by the 2 s output horizon, the event cap and float range (all counted).
//
// The estimated quantity is the complex pressure H(f) at the listener, per octave
// band b (material coefficients at the band centre), as delays and complex gains
// deposited into band trains. Paths are sequences of vertices joined by straight
// legs; a vertex is
//   - the source: unit point source, pressure ReferenceDistance / r;
//   - a specular reflection (delta): pressure coefficient sqrt((1-a)(1-t)(1-s)), or
//     the complex plane-wave coefficient of an impedance surface; spreading is the
//     unfolded image distance;
//   - a transmission (delta): sqrt((1-a) t), refracted into the next medium;
//   - a diffuse scattering point x (area measure): Kirchhoff secondary sources
//     (i k / 2 pi) sqrt((1-a)(1-t)s) sqrt(cos_i cos_o) exp(i phi(x)) / (l_in l_out),
//     with a random phase phi fixed per surface cell of area lambda^2/pi. Averaged
//     over phase realizations its energy is exactly Lambert's law with scattered
//     energy fraction (1-a)(1-t)s (the cell area makes (k/2pi)^2 * lambda^2/pi = 1/pi);
//     one realization is a coherent pressure, as a rough surface is;
//   - an edge point z (length measure): the Biot-Tolstoy-Medwin secondary source
//     K(z) / (m l) (the kernel of CompleteDiffraction.hlsl, exact for rigid wedges),
//     times the edge opacity, as the last interaction before the listener (D, RD).
//     Known gap: paths that continue through an edge point (DD, DR, edge chains)
//     are not sampled; that step failed its cube control on Metal and is left out
//     until fixed (docs/benchmarks/methods-2026-11-06).
// Branches at a surface are drawn in proportion to their pressure amplitudes, so
// every outcome carries their sum, which exceeds one when more than one branch is
// open: this is the expected divergence and it is measured, not repaired.
//
// Listener connections (next-event estimation):
//   - from the source, every diffuse point and every edge point: a straight leg to
//     the listener, transmitting through surfaces as the direct sound does;
//   - through one edge (D), or one mirror and one edge (RD), from the source and
//     every diffuse point;
//   - specular chains (delta vertices cannot be connected to a point listener): at
//     every delta vertex the listener image in the anchor's unfolded space is tested;
//     when the ray left the anchor within MethodDetectRadius of it, the exact image
//     path is retraced and scored with its own exact value divided by the probability
//     of that detection cone (MethodCommon.hlsl). Unbiased whenever the cone lies in
//     the chain's visibility beam; biased within MethodDetectRadius of a reflection
//     boundary. Delay and direction are those of the exact path.
// Russian roulette after every delta event keeps survival min(1, |coefficients| /
// probabilities) since the anchor; weights divided by it.
// Angle of a direction about an edge, measured from its first face into the air.
// Angle of a direction about an edge from its first face into the air: the hybrid's
// TunedEdgeAngle (series arctangent, face-grazing snapping). A plain atan2 here
// placed the source of every cube edge inside the solid on Metal.
float PressureWedgeAngle(float3 d, float3 face, float3 normal, float exterior)
{
    return TunedEdgeAngle(d, face, normal, exterior);
}
// The Biot-Tolstoy-Medwin secondary-source kernel K of a rigid wedge (the formula of
// TunedEdgeKernel/Kernel in CompleteDiffraction.hlsl and DiffractionIntegral.hlsl),
// written compactly: per unit edge length the edge point radiates K / (m l).
float PressureWedge(uint edge, float3 source, float3 at, float3 receiver)
{
    float3 a, u, f, n; float extent, angle;
    EdgeFrame(edge, 0, false, a, u, f, n, extent, angle);
    float3 si = source - at, so = receiver - at;
    float zs = dot(si, u), zr = dot(so, u);
    float3 ps = si - zs * u, pr = so - zr * u;
    float rs = length(ps), rr = length(pr);
    if (rs <= RayEpsilon || rr <= RayEpsilon) return 0;
    float ts = PressureWedgeAngle(ps, f, n, angle), tr = PressureWedgeAngle(pr, f, n, angle);
    if (ts > angle || tr > angle) return 0; // inside the solid
    float nu = RayPi / angle, m = length(si), l = length(so);
    float x = abs(rs * zr + rr * zs) / sqrt(2 * rs * rr * max(1e-20, m * l + rs * rr - zs * zr));
    float sh = x < .01 ? nu * x * (1 + (nu * nu - 1) * x * x / 6) : sinh(nu * log(x + sqrt(1 + x * x)));
    float4 phi = nu * (RayPi + float4(ts + tr, ts - tr, tr - ts, -ts - tr));
    float4 h = sin(.5 * phi), terms = sin(phi) / (2 * (sh * sh + h * h));
    terms = abs(h) > 1e-6 ? terms : 0; // a term exactly on its boundary is absent
    return -nu * dot(terms, 1) / (4 * RayPi);
}
float PressureDirectivity(MethodAnchor v, float3 w) { return v.kind == 1 ? sqrt(max(0, dot(w, v.normal))) : 1; }
float2 PressureSpecular(RayMaterial m, float cosine)
{
    if (m.medium.w != 0) return ReflectionPressure(m, cosine);
    return float2(sqrt(saturate((1 - m.surface.x) * (1 - m.surface.z) * (1 - m.surface.y))), 0);
}
void PressureDeposit(uint band, float time, float2 value, uint depth, float scale)
{
    if (time > MethodHorizon) return;
    MethodStatistic(band, depth, length(value));
    MethodDepositPressure(band, time, value * scale, HashRay(asuint(time) ^ HashRay(asuint(value.x) + band * 0x9e3779b9u)));
}
// Straight connection from an anchor to the listener.
void PressureListener(MethodAnchor v, float2 A[8], float3 L, float scale, uint depth, uint ray, inout uint slot)
{
    float3 w = L - v.position;
    float d = length(w);
    if (d <= RayEpsilon) return;
    w /= d;
    float D = PressureDirectivity(v, w);
    if (D == 0) return;
    float factor[8]; float time;
    [loop] for (uint k = 0; k < MethodBands; k++) factor[k] = 1;
    if (!MethodConnect(v.position, L, v.medium, false, factor, time)) return;
    float magnitude = 0;
    [loop] for (uint k = 0; k < MethodBands; k++)
    {
        float2 value = A[k] * (D * factor[k] / max(MinimumDistance, d));
        PressureDeposit(k, v.time + time, value, depth, scale);
        if (k == 2) magnitude = length(value);
    }
    MethodCaptureEvent(ray, slot, CaptureConnection, v.position, 0, magnitude, v.time + time);
}
// Edge-point density on the finite edge: half uniform, half a truncated Cauchy
// centred on the stationary (Keller) point between `from` and the listener, with the
// width of the BTM integrand's peak there. Only the variance depends on this choice;
// the estimator divides by the exact density.
float3 PressureEdgePoint(uint edge, float3 from, float3 L, inout uint rng, out float density)
{
    float3 a, u, f, n; float extent, angle;
    EdgeFrame(edge, 0, false, a, u, f, n, extent, angle);
    float3 s = from - a, r = L - a;
    float zs = dot(s, u), zr = dot(r, u), rs = length(s - zs * u), rr = length(r - zr * u);
    float center = (zs * rr + zr * rs) / max(rs + rr, 1e-6), scale = max(1e-3, rs * rr / max(rs + rr, 1e-6));
    float span = EdgeHeightSpan(center, float4(scale, scale, scale, scale), extent, false).x;
    float pick = RandomRay(rng), draw = clamp(RandomRay(rng), 1e-7, 1 - 1e-7);
    float z = pick < .5 ? draw * extent : EdgeHeight(center, scale, span, extent, draw, false);
    float offset = z - center;
    density = .5 / extent + .5 * scale / ((scale * scale + offset * offset) * span);
    return a + z * u;
}
// The only call site of the edge sampler: a listener connection through one edge
// point (D), or through one mirror and one edge point (RD), from the source or a
// scattering point. Known gap: continuing a path through an edge point (DD, DR, edge
// chains) is not part of this loop; see docs/benchmarks/methods-2026-11-06.
void PressureEdge(MethodAnchor v, float2 A[8], float3 L, float scale, uint depth, uint ray, inout uint slot, inout uint rng)
{
    if (MethodEdgeCount == 0 || DiffractionFactor <= 0) return;
    float pick;
    uint edge = MethodPickEdge(v.kind == 1 ? v.primitive : 0xffffffff, rng, pick);
    bool image = PrimitiveCount > 0 && RandomRay(rng) < .5;
    float variant = PrimitiveCount > 0 ? (image ? .5 / PrimitiveCount : .5) : 1;
    uint mirror = image ? min(PrimitiveCount - 1, (uint)(RandomRay(rng) * PrimitiveCount)) : 0;
    float3 from = v.position, unused;
    if (image)
    {
        if ((States[Primitives[mirror].indices.x].flags.x & 3) != 3) return;
        from = CompleteImage(mirror, 0, v.position, false, unused);
    }
    float density;
    float3 e = PressureEdgePoint(edge, from, L, rng, density);
    if (!(density > 0) || !all(isfinite(e))) return;
    float3 first = e, hit;
    float cosine = 0;
    if (image) { if (!CompleteMirror(mirror, 0, v.position, e, false, hit, cosine)) return; first = hit; }
    float m = length(e - from), l = length(L - e);
    float D = PressureDirectivity(v, normalize(first - v.position)), K = PressureWedge(edge, from, e, L);
    if (D == 0 || K == 0 || m <= RayEpsilon || l <= RayEpsilon) return;
    // Both straight legs (vertex -> edge unless mirrored, edge -> listener) at one call site.
    [loop] for (uint segment = image ? 1u : 0u; segment < 2; segment++)
        if (!Visible(0, segment == 0 ? v.position : e, segment == 0 ? e : L, false)) return;
    MethodCount(CounterEdgeConnections);
    float time = v.time + (m + l) * MethodSlowness(AirMedium), magnitude = 0;
    [loop] for (uint k = 0; k < MethodBands; k++)
    {
        float2 value = A[k] * (D * K * DiffractionFactor * MethodEdgeOpacity(edge, k) *
                               exp(-MethodLoss(AirMedium, k) * (m + l)) / (m * l * pick * density * variant));
        if (image) value = ComplexProduct(value, PressureSpecular(MethodSurface(mirror, k), cosine));
        PressureDeposit(k, time, value, depth + (image ? 2 : 1), scale);
        if (k == 2) magnitude = length(value);
    }
    MethodCaptureEvent(ray, slot, CaptureEdgeConnection, e, 0, magnitude, time, image ? 1 : 0);
}

// One flat loop. Each iteration first makes the pending listener connections of the
// latest vertex, then either one edge sample or one traced segment, so every heavy
// helper (traversal, edge sampler, chain retrace) has a single call site: a small
// kernel with a small per-thread stack.
void TracePressure(uint index)
{
    if (index >= MethodRayCount) return;
    uint ray = MethodRayOffset + index, slot = 0;
    MethodCount(CounterRays);
    float3 S = States[Sources[0]].position.xyz, L = States[Listeners[0]].position.xyz;
    uint rng = HashRay(ray * 0x9e3779b9u ^ HashRay(RaySeed * 0xc2b2ae35u + 0x5bd1e995u));
    float inverseRays = 1.0 / MethodTotalRays;
    MethodAnchor v;
    v.position = S; v.normal = 0; v.previous = S; v.kind = 0; v.edge = 0; v.primitive = 0xffffffff; v.time = 0;
    v.medium = MediumAt(0, S, false);
    float2 A[8], cA[8], C[8];
    [loop] for (uint k = 0; k < MethodBands; k++) { A[k] = float2(ReferenceDistance, 0); cA[k] = A[k]; C[k] = float2(1, 0); }
    MethodCaptureEvent(ray, slot, CaptureSource, S, ReferenceDistance, 0, 0);
    // Pending connections of vertex c. The direct sound is deterministic: ray 0 carries it for every ray.
    MethodAnchor c = v;
    bool connectListener = ray == 0, connectEdge = true;
    float listenerScale = 1;
    uint depth = 0, connectDepth = 0, events = 0;
    bool travelling = false, done = false;
    // Delta chain since the anchor: C = coefficients and losses, Q = probability of every choice.
    float3 p = S, d = 0, w0 = 0, Uc = 0;
    float3x3 U = MethodIdentity();
    float Q = 1, ell = 0, tau = 0, pdf = 1, D = 1;
    uint hits = 0, reflections = 0, medium = v.medium;
    uint2 transmitted = 0;
    bool straight = true;
    [loop] for (;;)
    {
        if (connectListener)
        {
            connectListener = false;
            PressureListener(c, cA, L, listenerScale, connectDepth, ray, slot);
        }
        if (connectEdge)
        {
            connectEdge = false;
            PressureEdge(c, cA, L, inverseRays, connectDepth, ray, slot, rng);
            continue;
        }
        if (done) break;
        if (!travelling)
        {
            if (++events >= MethodMaxEvents) { MethodCount(CounterEventCap); break; }
            w0 = v.kind == 1 ? MethodCosineDirection(v.normal, rng) : UnitDirection(rng, false);
            pdf = v.kind == 1 ? max(1e-12, dot(w0, v.normal)) / RayPi : 1 / (4 * RayPi);
            D = PressureDirectivity(v, w0);
            if (D == 0) { MethodCaptureEvent(ray, slot, CaptureEnd, v.position, 0, 0, v.time); break; }
            [loop] for (uint k = 0; k < MethodBands; k++) C[k] = float2(1, 0);
            Q = 1;
            ell = 0; tau = v.time; p = v.position; d = w0; Uc = 0; U = MethodIdentity();
            hits = 0; reflections = 0; medium = v.medium; transmitted = 0; straight = true;
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
        p += d * distance; ell += distance; tau += distance * MethodSlowness(medium);
        [loop] for (uint k = 0; k < MethodBands; k++) C[k] *= exp(-MethodLoss(medium, k) * distance);
        float3 n = dot(d, normal) < 0 ? normal : -normal;
        float cosI = max(0, -dot(d, n));
        depth++;
        MethodCount(CounterEvents);
        float aT[8], aD[8]; float2 aS[8];
        float mT = 0, mD = 0, mS = 0;
        [loop] for (uint k = 0; k < MethodBands; k++)
        {
            RayMaterial m = MethodSurface(hit, k);
            bool impedance = m.medium.w != 0;
            aT[k] = impedance ? 0 : sqrt(saturate((1 - m.surface.x) * m.surface.z));
            aD[k] = impedance ? 0 : sqrt(saturate((1 - m.surface.x) * (1 - m.surface.z) * m.surface.y));
            aS[k] = PressureSpecular(m, cosI);
            mT += aT[k]; mD += aD[k]; mS += length(aS[k]);
        }
        if (cosI <= 1e-6) mD = 0; // grazing: no scattering lobe
        // Diffuse lobe of this point: its listener and edge connections are pending.
        MethodAnchor x;
        x.position = p; x.normal = n; x.previous = p; x.kind = 1; x.edge = 0; x.primitive = hit; x.time = tau; x.medium = medium;
        if (mD > 0)
        {
            float area = ell / (cosI * pdf * Q);
            [loop] for (uint k = 0; k < MethodBands; k++)
            {
                float phase = MethodCellPhase(hit, p, n, k) + .5 * RayPi; // i exp(i phi)
                cA[k] = ComplexProduct(ComplexProduct(A[k], C[k]) * (D * area * MethodWavenumber(k) / (2 * RayPi) * aD[k] * sqrt(cosI)),
                                       float2(cos(phase), sin(phase)));
            }
            c = x; connectDepth = depth; listenerScale = inverseRays;
            connectListener = connectEdge = true;
        }
        float total = mT + mD + mS;
        if (!(total > 0)) { MethodCaptureEvent(ray, slot, CaptureEnd, p, 0, 0, tau); done = true; continue; }
        float u = RandomRay(rng) * total;
        if (u < mD)
        {
            // Diffuse scattering: this point becomes the anchor.
            bool finite = true;
            [loop] for (uint k = 0; k < MethodBands; k++) { A[k] = cA[k] * (total / mD); finite = finite && all(isfinite(A[k])); }
            v = x;
            travelling = false;
            MethodCount(CounterDiffuse);
            MethodCaptureEvent(ray, slot, CaptureDiffuse, p, length(A[2]), 0, tau);
            if (!finite) { MethodCount(CounterNonFinite); done = true; }
            if (v.time > MethodHorizon) { MethodCount(CounterHorizon); done = true; }
            continue;
        }
        bool transmit = u < mD + mT;
        if (transmit)
        {
            uint next = MediumAt(0, p + d * (2 * RayEpsilon), false);
            float3 bent = refract(d, n, MethodIor(medium, 3) / MethodIor(next, 3));
            [loop] for (uint k = 0; k < MethodBands; k++) C[k] *= aT[k];
            Q *= mT / total;
            if (dot(bent, bent) > 0)
            {
                if (MethodIor(next, 3) != MethodIor(medium, 3)) straight = false;
                d = normalize(bent);
                medium = next;
                if (hits < 64) { if (hits < 32) transmitted.x |= 1u << hits; else transmitted.y |= 1u << (hits - 32); }
                else straight = false;
            }
            else { d = reflect(d, n); straight = false; } // total internal reflection
        }
        else
        {
            [loop] for (uint k = 0; k < MethodBands; k++) C[k] = ComplexProduct(C[k], aS[k]);
            Q *= mS / total;
            d = reflect(d, n);
            MethodUnfoldReflect(U, Uc, n, dot(n, p));
            reflections++;
        }
        hits++;
        float weight = 0;
        [loop] for (uint k = 0; k < MethodBands; k++) weight = max(weight, length(C[k]));
        MethodCaptureEvent(ray, slot, transmit ? CaptureTransmission : CaptureSpecular, p,
                           length(A[2]) * D * length(C[2]) / Q / max(MinimumDistance, ell), 0, tau);
        // Specular chain detection (see header).
        if (reflections > 0 && straight)
        {
            float3 image = mul(U, L) + Uc, w = image - v.position;
            float span = length(w), along = dot(w, w0);
            if (along > 0 && span > MethodDetectRadius && span * span - along * along < MethodDetectRadius * MethodDetectRadius)
            {
                MethodCount(CounterDetections);
                w /= span;
                float2 exact[8]; float energy[8]; float total0; float3 from;
                float3 origin = v.kind == 1 ? v.position + v.normal * RayEpsilon : v.position;
                float Dw = PressureDirectivity(v, w);
                if (Dw != 0 && MethodRetraceChain(origin, image, L, hits, transmitted, exact, energy, total0, from))
                {
                    MethodCount(CounterChains);
                    float P = MethodConeProbability(v.kind == 1 ? 1 : 0, w, span, v.normal), magnitude = 0;
                    float time = v.time + total0 * MethodSlowness(AirMedium);
                    [loop] for (uint k = 0; k < MethodBands; k++)
                    {
                        float2 value = ComplexProduct(A[k], exact[k]) *
                                       (Dw * exp(-MethodLoss(AirMedium, k) * total0) / (max(MinimumDistance, total0) * P * Q));
                        PressureDeposit(k, time, value, depth, inverseRays);
                        if (k == 2) magnitude = length(value);
                    }
                    MethodCaptureEvent(ray, slot, CaptureChain, p, 0, magnitude, time);
                }
            }
        }
        // Russian roulette on the weight gained since the anchor.
        if (!(weight / Q < 3.0e38) || !isfinite(weight)) { MethodCount(CounterNonFinite); done = true; continue; }
        float survive = min(1, weight / Q);
        if (survive < 1)
        {
            if (RandomRay(rng) >= survive)
            { MethodCount(CounterRoulette); MethodCaptureEvent(ray, slot, CaptureEnd, p, 0, 0, tau); done = true; continue; }
            Q *= survive;
        }
    }
}
