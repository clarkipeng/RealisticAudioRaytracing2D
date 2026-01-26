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


#pragma kernel FFT
#pragma kernel IFFT

#define PI 3.14159265359

// Change this to modify FFT size
#define WINDOW_SIZE 1024

RWStructuredBuffer<float2> Data;

// Create fast shared memory for the thread group
groupshared float2 sharedData[WINDOW_SIZE];

float2 complex_mult(float2 a, float2 b) {
    return float2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

float2 complex_exp(float theta) {
    float c, s;
    sincos(theta, s, c); // Faster intrinsic
    return float2(c, s);
}

// Helper to handle loading/unloading based on GroupID
// This allows you to process arrays larger than 128 by dispatching multiple groups
void LoadShared(uint groupIdx, uint groupThreadIdx) {
    uint globalIdx = groupIdx * WINDOW_SIZE + groupThreadIdx;
    sharedData[groupThreadIdx] = Data[globalIdx];
}

void StoreShared(uint groupIdx, uint groupThreadIdx) {
    uint globalIdx = groupIdx * WINDOW_SIZE + groupThreadIdx;
    Data[globalIdx] = sharedData[groupThreadIdx];
}

[numthreads(WINDOW_SIZE, 1, 1)]
void FFT(uint3 groupThreadID : SV_GroupThreadID, uint3 groupID : SV_GroupID) {
    uint idx = groupThreadID.x;
    int n = WINDOW_SIZE;
    int bits = (int)log2(n);

    // 1. Load Global -> Shared
    LoadShared(groupID.x, idx);
    GroupMemoryBarrierWithGroupSync();

    // 2. Bit-reversal
    uint rev = reversebits(idx) >> (32 - bits);
    if (idx < rev) {
        float2 temp = sharedData[idx];
        sharedData[idx] = sharedData[rev];
        sharedData[rev] = temp;
    }
    GroupMemoryBarrierWithGroupSync();

    // 3. FFT Stages
    for (int s = 1; s <= bits; s++) {
        int m = 1 << s;       // Stage size
        int m2 = m >> 1;      // Half stage

        // Butterfly Mapping
        int k = (idx / m2) * m; // Start of the butterfly group
        int j = idx % m2;       // Index within the butterfly group

        // Only threads that map to a valid butterfly need to run
        if (j < m2) {
            // FIX: Calculate Angle Directly (Precise & Fast)
            // -2 * PI * j / m
            float angle = -2.0 * PI * j / m; 
            float2 w = complex_exp(angle);

            int i1 = k + j;
            int i2 = k + j + m2;

            float2 u = sharedData[i1];
            float2 t = complex_mult(w, sharedData[i2]);

            sharedData[i1] = u + t;
            sharedData[i2] = u - t;
        }
        
        GroupMemoryBarrierWithGroupSync();
    }

    // 4. Store Shared -> Global
    StoreShared(groupID.x, idx);
}



[numthreads(WINDOW_SIZE, 1, 1)]
void IFFT(uint3 groupThreadID : SV_GroupThreadID, uint3 groupID : SV_GroupID) {
    uint idx = groupThreadID.x;
    int n = WINDOW_SIZE;
    int bits = (int)log2(n);

    // 1. Load
    LoadShared(groupID.x, idx);
    GroupMemoryBarrierWithGroupSync();

    // 2. Bit-reversal
    uint rev = reversebits(idx) >> (32 - bits);
    if (idx < rev) {
        float2 temp = sharedData[idx];
        sharedData[idx] = sharedData[rev];
        sharedData[rev] = temp;
    }
    GroupMemoryBarrierWithGroupSync();

    // 3. IFFT Stages
    for (int s = 1; s <= bits; s++) {
        int m = 1 << s;
        int m2 = m >> 1;

        int k = (idx / m2) * m;
        int j = idx % m2;

        if (j < m2) {
            // INVERSE: Positive angle
            float angle = 2.0 * PI * j / m;
            float2 w = complex_exp(angle);

            int i1 = k + j;
            int i2 = k + j + m2;

            float2 u = sharedData[i1];
            float2 t = complex_mult(w, sharedData[i2]);

            sharedData[i1] = u + t;
            sharedData[i2] = u - t;
        }
        GroupMemoryBarrierWithGroupSync();
    }

    // 4. Normalization and Store
    // (Inverse FFT requires dividing by N)
    float invN = 1.0 / n;
    Data[groupID.x * WINDOW_SIZE + idx] = sharedData[idx] * invN;
}


#endif
