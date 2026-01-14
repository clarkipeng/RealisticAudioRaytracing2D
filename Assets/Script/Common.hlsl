#ifndef COMMON_HLSL
#define COMMON_HLSL

static const float eps = 1e-4;
static const float inf = 1e8;
static const float PI = 3.14159265;

float random(inout uint state) {
    state = state * 747796405 + 2891336453;
    uint res = ((state >> ((state >> 28) + 4)) ^ state) * 277803737;
    return ((res >> 22) ^ res) / 4294967295.0;
}

float intersect(float2 o, float2 d, float2 a, float2 b) {
    float2 v1 = o - a, v2 = b - a, v3 = float2(-d.y, d.x);
    float dotP = dot(v2, v3);
    if (abs(dotP) < eps) return inf;
    float t1 = (v2.x * v1.y - v2.y * v1.x) / dotP;
    float t2 = dot(v1, v3) / dotP;
    return (t1 >= eps && t2 >= 0 && t2 <= 1) ? t1 : inf;
}

float intersectCircle(float2 rayPos, float2 rayDir, float2 center, float radius) {
    float2 L = center - rayPos;
    float tca = dot(L, rayDir);
    if (tca < 0) return inf;
    float d2 = dot(L, L) - tca * tca;
    float r2 = radius * radius;
    if (d2 > r2) return inf;
    float thc = sqrt(r2 - d2);
    float t0 = tca - thc;
    float t1 = tca + thc;
    if (t0 > eps) return t0;
    if (t1 > eps) return t1;
    return inf;
}

float3 Refract(float3 i, float3 n, float eta) {
    float cosi = dot(-i, n);
    float cost2 = 1.0 - eta * eta * (1.0 - cosi*cosi);
    float3 t = eta * i + ((eta * cosi - sqrt(abs(cost2))) * n);
    return t * (float3)(cost2 > 0);
}

#endif
