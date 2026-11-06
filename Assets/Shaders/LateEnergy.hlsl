// Late field as incoherent energy (Vorlander 1989; Savioja & Svensson 2015).
// Each thread traces one ray in one octave band; all bands share a ray's random
// numbers. A ray starts with energy weight 1 (times its emission site's weight)
// and its weight never grows: absorption, impedance reflection and air/medium
// loss are deterministic factors <= 1, and transmission, diffuse and specular
// continuations are chosen with their energy fractions (ISO 17497-1 defines the
// scattering coefficient as an energy fraction).
//
// Detection at the listener:
// - diffuse rain: at every surface the scattered fraction is connected to the
//   listener (Lambert next-event estimation, visible connections only);
// - receiver sphere: segments whose last event is a specular reflection or a
//   transmission are scored by their chord through a sphere of LateReceiverRadius
//   (track-length estimator). Segments after a diffuse event are not scored,
//   because diffuse rain already holds their direct connection.
// Paths held by the phase-exact early part are never scored: no scattering and
// either no reflection (direct and transmitted direct sound) or only unobstructed
// specular reflections up to LateEarlyOrder (first images; second order when the
// complete early routes are on). Edge diffraction is not part of the late field.
//
// Energies are in the direct-sound convention (ReferenceDistance / distance)^2.
// The histogram stores, per frame, pair, head-frame direction, band and bin, the
// sum over rays of each ray's share in 32.32 fixed point; dividing by LateRays
// gives the energy. Integer atomics keep sums order-independent and exact.
StructuredBuffer<RayMaterial> LateMaterials; // [material * LateBands + band], sampled at band centres
StructuredBuffer<float4> LateSites;          // xyz site (source-local, or world), w energy weight (mean 1)
StructuredBuffer<uint4> LateSiteRanges;      // per source: first, count, world flag
RWStructuredBuffer<uint> LateHistogram;      // (low, high) per cell
uint LateRays, LateBins, LateEarlyOrder, LatePairCells;
float LateBinSeconds, LateReceiverRadius;
static const uint LateBands = 8, LateDirections = 6;
static const float LateEnergyFloor = 1e-8; // -80 dB of a ray's emitted energy

float LateBandHz(uint band) { return 125.0 * (float)(1u << band); }
float LateIor(uint medium, uint band)
{
    return medium == AirMedium ? 1 : max(.001, LateMaterials[medium * LateBands + band].surface.w);
}
float LateLoss(uint medium, uint band)
{
    return medium == AirMedium ? AirCoefficient(LateBandHz(band)) : LateMaterials[medium * LateBands + band].medium.x;
}
bool LateEarly(uint reflections, bool scattered, bool transmitted, uint earlyOrder)
{
    return !scattered && (reflections == 0 || (!transmitted && reflections <= earlyOrder));
}
// Head-frame bins +x, -x, +y, -y, +z, -z (right, left, up, down, front, back).
uint LateDirection(float3 local)
{
    float3 a = abs(local);
    return a.x >= a.y && a.x >= a.z ? (local.x >= 0 ? 0 : 1) : a.y >= a.z ? (local.y >= 0 ? 2 : 3) : (local.z >= 0 ? 4 : 5);
}
// The dither is a hash of the deposit, not the ray's stream, so a moving listener
// never changes a ray's path. Its expectation makes the rounding unbiased.
void LateDeposit(uint cells, float4 inverseHead, float3 arrival, uint band, float time, float share, uint key)
{
    if (!(share > 0) || !isfinite(share) || time < 0)
        return;
    uint bin = (uint)floor(time / LateBinSeconds);
    if (bin >= LateBins)
        return;
    uint cell = cells + (LateDirection(Rotate(inverseHead, arrival)) * LateBands + band) * LateBins + bin;
    float whole = floor(share), part = (share - whole) * 4294967296.0 + (HashRay(key ^ HashRay(cell)) >> 8) * (1.0 / 16777216.0);
    if (part >= 4294967296.0) { whole += 1; part -= 4294967296.0; }
    uint low = (uint)part, high = (uint)whole, old;
    InterlockedAdd(LateHistogram[2 * cell], low, old);
    if (old + low < old)
        high++;
    if (high != 0)
        InterlockedAdd(LateHistogram[2 * cell + 1], high);
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
void ClearLateCells(uint index)
{
    if (index < FrameCount * PairCount * LatePairCells * 2)
        LateHistogram[index] = 0;
}
void TraceLate(uint index, bool is2D)
{
    if (index >= FrameCount * LateRays * LateBands)
        return;
    uint ray = index % LateRays, band = (index / LateRays) % LateBands, frame = index / (LateRays * LateBands);
    uint pair = RayTracePair, query = frame * PairCount + pair, sourceIndex = pair / ListenerCount;
    uint4 range = LateSiteRanges[sourceIndex];
    if (RayPairEnabled[query] == 0 || range.y == 0)
        return;
    uint stateOffset = frame * ObjectCount, source = Sources[sourceIndex], listener = Listeners[pair % ListenerCount];
    // Rays are stratified over the sites; every site gets equal total weight
    // whatever the ray count, so the histogram is the sites' weighted mean.
    uint site = ray % range.y, assigned = LateRays / range.y + (site < LateRays % range.y ? 1 : 0);
    float4 entry = LateSites[range.x + site];
    float weight = entry.w * (float)LateRays / ((float)range.y * assigned) * ReferenceDistance * ReferenceDistance;
    State emitter = States[stateOffset + source];
    float3 position = range.z != 0 ? entry.xyz : emitter.position.xyz + Rotate(emitter.rotation, entry.xyz);
    float3 end = States[stateOffset + listener].position.xyz;
    float4 head = States[stateOffset + listener].rotation;
    head.xyz = -head.xyz;
    if (is2D)
    {
        position.z = 0;
        end.z = 0;
    }
    uint cells = (frame * PairCount + pair) * LatePairCells;
    uint rng = HashRay(ray * 0x9e3779b9u ^ HashRay(pair * 0x85ebca6bu + RaySeed * 0xc2b2ae35u + 0x2545f491u));
    uint key = HashRay(rng ^ 0x68e31da4u ^ band);
    float3 direction = UnitDirection(rng, is2D);
    uint medium = MediumAt(stateOffset, position, is2D), earlyOrder = LateEarlyOrder;
    // Second-order images exist only where the complete routes trace them.
    if (earlyOrder > 1 && (medium != AirMedium || MediumAt(stateOffset, end, is2D) != AirMedium))
        earlyOrder = 1;
    float time = 0, travelled = 0, radius = LateReceiverRadius;
    uint reflections = 0;
    bool scattered = false, transmitted = false, diffuseLast = false;
    for (uint segment = 0; segment <= MaxBounces; segment++)
    {
        float speed = SpeedOfSound / LateIor(medium, band), loss = 2 * LateLoss(medium, band);
        float limit = (MaximumPathDelay - time) * speed;
        if (limit <= 0)
            break;
        uint hit;
        float distance;
        float3 normal;
        bool intersects = NearestHit(stateOffset, position, direction, limit, is2D, hit, distance, normal);
        float span = intersects ? distance : limit;
        if (segment > 0 && !diffuseLast && !LateEarly(reflections, scattered, transmitted, earlyOrder))
        {
            float3 v = end - position;
            float along = dot(v, direction), closest = dot(v, v) - along * along;
            if (closest < radius * radius)
            {
                float halfChord = sqrt(radius * radius - closest), a = max(0, along - halfChord), b = min(span, along + halfChord);
                float at = clamp(along, a, b);
                if (b > a && Visible(stateOffset, end, position + direction * at, is2D))
                {
                    float chord = b - a;
                    float share = weight * exp(-loss * at) *
                                  (is2D ? 2 * chord / (radius * radius * max(MinimumDistance, travelled + at))
                                        : 3 * chord / (radius * radius * radius));
                    LateDeposit(cells, head, -direction, band, time + at / speed, share, key ^ (segment * 2 + 1));
                }
            }
        }
        if (!intersects || segment == MaxBounces)
            break;
        position += direction * distance;
        time += distance / speed;
        travelled += distance;
        weight *= exp(-loss * distance);
        RayMaterial material = LateMaterials[Primitives[hit].indices.y * LateBands + band];
        float3 n = dot(direction, normal) < 0 ? normal : -normal;
        float transmit = 0, diffuse = 0;
        if (material.medium.w != 0)
        {
            float2 r = ReflectionPressure(material, abs(dot(direction, n)));
            weight *= dot(r, r); // Smooth impedance boundary: angle-dependent specular energy.
        }
        else
        {
            weight *= saturate(1 - material.surface.x);
            transmit = saturate(material.surface.z);
            diffuse = (1 - transmit) * saturate(material.surface.y);
        }
        if (!(weight >= LateEnergyFloor))
            break;
        float3 toListener = end - position;
        float gap = length(toListener), cosine = gap > RayEpsilon ? dot(n, toListener) / gap : 0;
        if (diffuse > 0 && cosine > 0 && Visible(stateOffset, position + n * RayEpsilon, end, is2D))
        {
            float d = max(MinimumDistance, gap);
            float share = weight * diffuse * exp(-loss * gap) *
                          (is2D ? RayPi * cosine / (d * max(MinimumDistance, travelled + gap)) : 4 * cosine / (d * d));
            LateDeposit(cells, head, -toListener / gap, band, time + gap / speed, share, key ^ (segment * 2 + 2));
        }
        float u = RandomRay(rng);
        bool refracted = false;
        if (u < transmit)
        {
            uint next = MediumAt(stateOffset, position + direction * (2 * RayEpsilon), is2D);
            float3 bent = refract(direction, n, LateIor(medium, band) / LateIor(next, band));
            if (dot(bent, bent) > 0) // Total internal reflection keeps the specular branch.
            {
                direction = normalize(bent);
                medium = next;
                refracted = transmitted = true;
                diffuseLast = false;
            }
        }
        if (!refracted)
        {
            if (u >= transmit && u < transmit + diffuse)
            {
                direction = LateLambert(n, rng, is2D);
                scattered = diffuseLast = true;
            }
            else
            {
                direction = reflect(direction, n);
                reflections++;
                diffuseLast = false;
            }
        }
    }
}
