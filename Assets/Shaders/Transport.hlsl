// Shared geometric preview model. Reflection/transmission/scattering and a
// dominant knife-edge path extend the original 2D shader to triangles.
// This is not a validated wave/diffraction reference.
struct RayMaterial
{
    // surface: absorption, scattering, transmission, relative slowness (air speed / material speed).
    // medium.x: pressure attenuation per metre.
    float4 surface, medium;
};
struct Edge
{
    float4 a, b;
    uint4 ids;
};
struct RayPath
{
    // direction.xyz: listener-to-arrival; direction.w: 1 - received/emitted frequency ratio.
    float4 direction;
    float delaySeconds, pressureGain;
    uint frequencyBin, identity;
};
StructuredBuffer<RayMaterial> RayMaterials;
StructuredBuffer<Edge> Edges;
// corner.xyz diffraction position, corner.w Doppler frequencyRatio; distances: first/last/direct/excess.
// ids: edge, blocked direct segment, crossing face, reserved.
struct EdgeRoute
{
    float4 corner, distances;
    uint4 ids;
};
RWStructuredBuffer<EdgeRoute> EdgeRoutes;
struct Volume
{
    float4 minimum, maximum;
    uint4 ids;
};
StructuredBuffer<Volume> Volumes;
RWStructuredBuffer<RayPath> RayPaths;
StructuredBuffer<uint> RayPairEnabled;
uint RaysPerBin, RaySeed;
uint FrequencyBins, RaysPerPair, MaxBounces, SlotsPerPair, EdgeCount, VolumeCount, UseAirAbsorption;
float FrequencyStep, DiffractionFactor, MaximumPathDelay;
uint HashRay(uint x)
{
    x ^= x >> 16;
    x *= 0x7feb352d;
    x ^= x >> 15;
    x *= 0x846ca68b;
    x ^= x >> 16;
    return x;
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
uint MediumAt(uint stateOffset, float3 location, bool is2D)
{
    uint medium = AirMedium;
    float smallest = 1e30;
    float3 probe = is2D ? normalize(float3(.819, .573, 0)) : normalize(float3(.819, .573, .371));
    for (uint v = 0; v < VolumeCount; v++)
    {
        Volume volume = Volumes[v];
        State pose = States[stateOffset + volume.ids.z];
        if ((pose.flags.x & 3) != 3)
            continue;
        float4 inverse = pose.rotation;
        inverse.xyz = -inverse.xyz;
        float3 local = Rotate(inverse, location - pose.position.xyz);
        if (any(local.xy < volume.minimum.xy) || any(local.xy > volume.maximum.xy) ||
            (!is2D && (local.z < volume.minimum.z || local.z > volume.maximum.z)))
            continue;
        float nearest = 1e30;
        float3 normal = 0;
        for (uint i = 0; i < volume.ids.y; i++)
        {
            float3 n;
            float distance = PrimitiveDistance(volume.ids.x + i, stateOffset, location, probe, is2D, n);
            if (distance < nearest)
            {
                nearest = distance;
                normal = n;
            }
        }
        if (volume.maximum.w < smallest && nearest < 1e30 && dot(normal, probe) * volume.minimum.w > 0)
        {
            smallest = volume.maximum.w;
            medium = volume.ids.w;
        }
    }
    return medium;
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
// Geometry is independent of the frequency grid and random ray budget.
// Select one visible finite-edge route, at the constrained minimum of |s-e|+|e-r|.
bool AcrossEdgeFace(uint face, State pose, float3 corner, float3 source, float3 listener, bool is2D)
{
    Primitive p = Primitives[face];
    float3 e = p.b.xyz - p.a.xyz;
    float3 normal = Rotate(pose.rotation, is2D ? float3(-e.y, e.x, 0) : cross(e, p.c.xyz - p.a.xyz));
    return dot(normal, source - corner) * dot(normal, listener - corner) <= 0;
}
void FindEdgeRoute(uint query, bool is2D)
{
    if (query >= FrameCount * PairCount)
        return;
    EdgeRoute route = (EdgeRoute)0;
    route.ids.x = 0xffffffff;
    EdgeRoutes[query] = route;
    if (RayPairEnabled[query] == 0)
        return;
    uint stateOffset = (query / PairCount) * ObjectCount, pair = query % PairCount;
    uint source = Sources[pair / ListenerCount], listener = Listeners[pair % ListenerCount];
    float3 start = States[stateOffset + source].position.xyz,
           end = States[stateOffset + listener].position.xyz;
    if (is2D)
    {
        start.z = 0;
        end.z = 0;
    }
    float direct = length(end - start);
    if (direct <= 2 * RayEpsilon)
        return;
    if (MediumAt(stateOffset, start, is2D) != AirMedium || MediumAt(stateOffset, end, is2D) != AirMedium)
        return;
    uint hit;
    float distance;
    float3 normal;
    bool blocked = NearestHit(stateOffset, start, (end - start) / direct, direct - RayEpsilon, is2D, hit,
                              distance, normal);
    float best = MaximumPathDelay * SpeedOfSound;
    for (uint i = 0; i < EdgeCount; i++)
    {
        Edge edge = Edges[i];
        State pose = States[stateOffset + edge.ids.x];
        if ((pose.flags.x & 3) != 3)
            continue;
        float3 a = pose.position.xyz + Rotate(pose.rotation, edge.a.xyz),
               b = pose.position.xyz + Rotate(pose.rotation, edge.b.xyz);
        if (is2D)
        {
            a.z = 0;
            b.z = 0;
        }
        float3 axis = b - a, corner = a;
        float size = length(axis);
        if (size > RayEpsilon)
        {
            axis /= size;
            float sa = dot(start - a, axis), ra = dot(end - a, axis);
            float s = length(start - a - axis * sa), r = length(end - a - axis * ra);
            if (s + r <= RayEpsilon)
                continue;
            corner += axis * clamp((sa * r + ra * s) / (s + r), 0, size);
        }
        uint face = edge.ids.y;
        if (!AcrossEdgeFace(face, pose, corner, start, end, is2D))
        {
            face = edge.ids.z;
            if (!AcrossEdgeFace(face, pose, corner, start, end, is2D))
                continue;
        }
        float first = length(corner - start), last = length(end - corner), total = first + last;
        if (first <= RayEpsilon || last <= RayEpsilon || total >= best)
            continue;
        if (!Visible(stateOffset, start, corner, is2D) || !Visible(stateOffset, corner, end, is2D))
            continue;
        float3 incoming = (corner - start) / first, outgoing = (end - corner) / last;
        float3 sourceVelocity = Velocity(stateOffset + source),
               listenerVelocity = Velocity(stateOffset + listener),
               edgeVelocity = PointVelocity(stateOffset + edge.ids.x, corner);
        if (is2D)
        {
            sourceVelocity.z = 0;
            listenerVelocity.z = 0;
            edgeVelocity.z = 0;
        }
        float frequencyRatio = InteractionFrequencyRatio(SpeedOfSound, incoming, outgoing, sourceVelocity,
                                                         listenerVelocity, edgeVelocity);
        route.corner = float4(corner, frequencyRatio);
        route.distances = float4(first, last, direct, max(0, total - direct));
        route.ids = uint4(i, blocked ? 1 : 0, face, 0);
        best = total;
    }
    EdgeRoutes[query] = route;
}
// ITU-R P.526-16 (31): pressure magnitude approximation to the scalar Fresnel
// knife-edge field. nu^2 = 4 * path excess / wavelength (paraxial equivalent).
float KnifeEdgePressure(float nu)
{
    if (nu <= -.78)
        return 1;
    float x = nu - .1;
    return min(1, pow(10, -6.9 / 20) / (sqrt(1 + x * x) + x));
}
void ApplyEdgeRoute(uint query, uint stateOffset, uint listener, uint frequencyBin, inout RayPath direct)
{
    if (DiffractionFactor <= 0)
        return;
    uint slot = query * SlotsPerPair + SlotsPerPair - FrequencyBins + frequencyBin;
    EdgeRoute route = EdgeRoutes[query];
    if (route.ids.x == 0xffffffff)
        return;
    float frequencyHz = frequencyBin * FrequencyStep,
          nu = sqrt(4 * route.distances.w * frequencyHz / SpeedOfSound);
    RayMaterial material = RayMaterials[Primitives[route.ids.z].indices.y * FrequencyBins + frequencyBin];
    float opaque = 1 - sqrt(saturate((1 - material.surface.x) * material.surface.z));
    if (route.ids.y == 0)
    {
        // Replace the lit-side field near its shadow boundary; do not add a
        // second copy of free-field sound. Unity's existing fades handle motion.
        direct.pressureGain *= 1 - saturate(DiffractionFactor) * opaque * (1 - KnifeEdgePressure(-nu));
        return;
    }
    float total = route.distances.x + route.distances.y;
    float gain = DiffractionFactor * opaque * KnifeEdgePressure(nu) *
                 exp(-AirCoefficient(frequencyHz) * total) * ReferenceDistance / max(MinimumDistance, total);
    RayPaths[slot] =
        Arrival(route.corner.xyz, States[stateOffset + listener].position.xyz, total / SpeedOfSound, gain,
                frequencyBin, route.corner.w, HashRay(0x20000000u ^ route.ids.x));
}
// Retarded source position; interfaces retain the scene pose of this solve.
// Iteration follows piecewise material travel time, not distance / air speed.
void EvaluateDirect(uint index, bool is2D)
{
    if (index >= FrameCount * PairCount * FrequencyBins)
        return;
    uint frequencyBin = index % FrequencyBins, query = index / FrequencyBins, frame = query / PairCount,
         pair = query % PairCount;
    RayPaths[query * SlotsPerPair + frequencyBin] = (RayPath)0;
    if (DiffractionFactor > 0)
        RayPaths[query * SlotsPerPair + SlotsPerPair - FrequencyBins + frequencyBin] = (RayPath)0;
    if (RayPairEnabled[query] == 0)
        return;
    uint stateOffset = frame * ObjectCount, source = Sources[pair / ListenerCount],
         listener = Listeners[pair % ListenerCount];
    float3 sourceNow = States[stateOffset + source].position.xyz,
           b = States[stateOffset + listener].position.xyz;
    float3 sourceVelocity = Velocity(stateOffset + source),
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
    // Alternate fixed-point proposals with bisection. A bracket prevents slow
    // material transitions or float-level cycles from defeating convergence.
    for (uint iteration = 0; iteration < 64; iteration++)
    {
        a = sourceNow - sourceVelocity * estimate;
        float remaining = length(b - a);
        float3 direction = remaining > 0 ? (b - a) / remaining : float3(0, 0, 0), position = a;
        gain = ReferenceDistance / max(MinimumDistance, remaining);
        time = 0;
        identity = 1;
        uint medium = MediumAt(stateOffset, a, is2D);
        float speed = SpeedOfSound / MediumIor(medium, frequencyBin);
        frequencyRatio = 1 / (1 - dot(sourceVelocity, direction) / speed);
        for (uint crossing = 0; crossing < PrimitiveCount && remaining > 2 * RayEpsilon; crossing++)
        {
            uint hit;
            float distance;
            float3 normal;
            if (!NearestHit(stateOffset, position, direction, remaining - RayEpsilon, is2D, hit, distance,
                            normal))
                break;
            Travel(distance, medium, frequencyBin, frequencyHz, time, gain);
            RayMaterial material = RayMaterials[Primitives[hit].indices.y * FrequencyBins + frequencyBin];
            gain *= sqrt(saturate((1 - material.surface.x) * material.surface.z));
            position += direction * distance;
            remaining -= distance;
            uint next = MediumAt(stateOffset, position + direction * (2 * RayEpsilon), is2D);
            float nextSpeed = SpeedOfSound / MediumIor(next, frequencyBin);
            float3 velocity = PointVelocity(stateOffset + Primitives[hit].indices.x, position);
            frequencyRatio *=
                (1 - dot(velocity, direction) / speed) / (1 - dot(velocity, direction) / nextSpeed);
            identity = HashRay(identity ^ (hit + 1));
            medium = next;
            speed = nextSpeed;
        }
        Travel(remaining, medium, frequencyBin, frequencyHz, time, gain);
        frequencyRatio *= 1 - dot(listenerVelocity, direction) / speed;
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
    ApplyEdgeRoute(query, stateOffset, listener, frequencyBin, direct);
    RayPaths[query * SlotsPerPair + frequencyBin] = direct;
}
// First-order mirror path from a sampled primitive's image source. Sampling
// primitive IDs avoids estimating a delta reflection with a diffuse connection.
// A cyclic stratification covers each primitive once per complete sample stratum.
RayPath FirstReflection(uint stateOffset, uint source, uint listener, uint frequencyBin, uint sampleIndex,
                        bool is2D)
{
    RayPath result = (RayPath)0;
    if (PrimitiveCount == 0)
        return result;
    uint id = (sampleIndex + HashRay(source * 0x85ebca6b + listener + RaySeed)) % PrimitiveCount;
    Primitive p = Primitives[id];
    State pose = States[stateOffset + p.indices.x];
    if ((pose.flags.x & 3) != 3)
        return result;
    float3 sourceNow = States[stateOffset + source].position.xyz,
           end = States[stateOffset + listener].position.xyz;
    float3 sourceVelocity = Velocity(stateOffset + source),
           listenerVelocity = Velocity(stateOffset + listener);
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
    float gain =
        sqrt(saturate((1 - material.surface.x) * (1 - material.surface.z) * (1 - material.surface.y)));
    float time = 0, lengthTo = total * speed;
    Travel(lengthTo, medium, frequencyBin, max(1, frequencyBin * FrequencyStep), time, gain);
    return Arrival(reflectionPoint, end, time,
                   gain * (float)PrimitiveCount / RaysPerBin * ReferenceDistance /
                       max(MinimumDistance, lengthTo),
                   frequencyBin, converged ? frequencyRatio : asfloat(0x7fc00000), HashRay(0x40000000u ^ id));
}
void TraceReflections(uint index, bool is2D)
{
    if (index >= FrameCount * PairCount * RaysPerPair)
        return;
    uint ray = index % RaysPerPair, query = index / RaysPerPair, frame = query / PairCount,
         pair = query % PairCount;
    uint stateOffset = frame * ObjectCount, source = Sources[pair / ListenerCount],
         listener = Listeners[pair % ListenerCount];
    uint first = query * SlotsPerPair + FrequencyBins + ray * (MaxBounces + 1);
    for (uint k = 0; k <= MaxBounces; k++)
        RayPaths[first + k] = (RayPath)0;
    if (RayPairEnabled[query] == 0)
        return;
    uint frequencyBin = ray / RaysPerBin, rng = HashRay((ray % RaysPerBin) + pair * 0x85ebca6b + RaySeed + 1);
    RayPaths[first] = FirstReflection(stateOffset, source, listener, frequencyBin, ray % RaysPerBin, is2D);
    first++;
    float frequencyHz = max(1, frequencyBin * FrequencyStep);
    float3 position = States[stateOffset + source].position.xyz,
           end = States[stateOffset + listener].position.xyz;
    if (is2D)
    {
        position.z = 0;
        end.z = 0;
    }
    float3 direction = UnitDirection(rng, is2D);
    float gain = 1, time = 0, pathLength = 0;
    uint medium = MediumAt(stateOffset, position, is2D);
    float3 sourceVelocity = Velocity(stateOffset + source),
           listenerVelocity = Velocity(stateOffset + listener);
    if (is2D)
    {
        sourceVelocity.z = 0;
        listenerVelocity.z = 0;
    }
    float frequencyRatio =
        1 / (1 - dot(sourceVelocity, direction) / (SpeedOfSound / MediumIor(medium, frequencyBin)));
    uint identity = HashRay(0x80000000u ^ (ray % RaysPerBin));
    float weight = 1.0 / RaysPerBin;
    for (uint bounce = 0; bounce < MaxBounces; bounce++)
    {
        float currentIor = MediumIor(medium, frequencyBin), speed = SpeedOfSound / currentIor,
              limit = (MaximumPathDelay - time) * speed;
        if (limit <= 0 || gain == 0)
            break;
        uint hit;
        float distance;
        float3 normal;
        bool intersects = NearestHit(stateOffset, position, direction, limit, is2D, hit, distance, normal);
        if (!intersects)
            break;
        position += direction * distance;
        Travel(distance, medium, frequencyBin, frequencyHz, time, gain);
        pathLength += distance;
        RayMaterial material = RayMaterials[Primitives[hit].indices.y * FrequencyBins + frequencyBin];
        gain *= sqrt(saturate(1 - material.surface.x));
        bool entering = dot(direction, normal) < 0;
        float3 n = entering ? normal : -normal;
        float3 velocity = PointVelocity(stateOffset + Primitives[hit].indices.x, position);
        if (is2D)
            velocity.z = 0;
        identity = HashRay(identity ^ (hit + 1));
        float3 toListener = normalize(end - position);
        float arrivalRatio = frequencyRatio * (1 - dot(velocity, direction) / speed) /
                             (1 - dot(velocity, toListener) / speed) *
                             (1 - dot(listenerVelocity, toListener) / speed);
        float last = length(end - position), cosine = last > 0 ? max(0, dot(n, (end - position) / last)) : 0;
        if (cosine > 0 && Visible(stateOffset, position + n * RayEpsilon, end, is2D))
            RayPaths[first + bounce] =
                Arrival(position, end, time + last / speed,
                        gain * sqrt(saturate((1 - material.surface.z) * material.surface.y)) * weight *
                            cosine * .5 * exp(-MediumLoss(medium, frequencyBin, frequencyHz) * last) *
                            ReferenceDistance / max(MinimumDistance, pathLength + last),
                        frequencyBin, arrivalRatio, identity);
        uint nextMedium = MediumAt(stateOffset, position + direction * (2 * RayEpsilon), is2D);
        float nextIor = MediumIor(nextMedium, frequencyBin);
        float reflectance = 1 - saturate(material.surface.z), cosTheta = abs(dot(direction, n));
        reflectance += (1 - reflectance) * pow(saturate(1 - cosTheta), 5);
        if (RandomRay(rng) > reflectance)
        {
            float3 transmitted = refract(direction, n, currentIor / nextIor);
            if (dot(transmitted, transmitted) > 0)
            {
                transmitted = normalize(transmitted);
                frequencyRatio *= (1 - dot(velocity, direction) / speed) /
                                  (1 - dot(velocity, transmitted) / (SpeedOfSound / nextIor));
                identity = HashRay(identity ^ 0x10000000u);
                direction = transmitted;
                medium = nextMedium;
                continue;
            }
        }
        float3 diffuse = UnitDirection(rng, is2D);
        if (dot(diffuse, n) < 0)
            diffuse = -diffuse;
        float3 nextDirection =
            RandomRay(rng) < saturate(material.surface.y) ? diffuse : reflect(direction, n);
        frequencyRatio *= (1 - dot(velocity, direction) / speed) / (1 - dot(velocity, nextDirection) / speed);
        direction = nextDirection;
    }
}
