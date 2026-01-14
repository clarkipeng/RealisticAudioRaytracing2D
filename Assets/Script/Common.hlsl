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

const int WINDOW_SIZE = 128;
float2 complex_exp(float theta) {
    return float2(cos(theta), sin(theta));
}

float2 complex_mult(float2 a, float2 b) {
    return float2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

float2 nth_root(int n, int k) {
    float angle = -2.0 * PI * k / n;
    return complex_exp(angle);
}

float2 nth_root_inv(int n, int k) {
    float angle = 2.0 * PI * k / n;
    return complex_exp(angle);
}

RWStructuredBuffer<float2> Data;

[numthreads(128, 1, 1)]
void FFT(uint3 id : SV_DispatchThreadID) {
    int n = WINDOW_SIZE;
    int bits = (int)log2(n);
    uint idx = id.x;
    
    if (idx >= (uint)n) return;

    // Bit-reversal permutation (parallel)
    uint rev = reversebits(idx) >> (32 - bits);
    if (idx < rev) {
        float2 temp = Data[idx];
        Data[idx] = Data[rev];
        Data[rev] = temp;
    }
    
    GroupMemoryBarrierWithGroupSync(); // Wait for all bit reversals

    // FFT stages - each stage must complete before next begins
    for (int s = 1; s <= bits; s++) {
        int m = 1 << s;
        int m2 = m >> 1;
        float2 w_m = nth_root(m, 1);

        // Each thread processes its butterfly operations
        int k_base = (idx / m2) * m;
        int j = idx % m2;
        
        if (k_base + j + m2 < n) {
            float2 w = w_m;
            // Calculate w^j
            for (int p = 0; p < j; p++) {
                w = complex_mult(w, w_m);
            }
            
            float2 t = complex_mult(w, Data[k_base + j + m2]);
            float2 u = Data[k_base + j];
            Data[k_base + j] = u + t;
            Data[k_base + j + m2] = u - t;
        }
        
        GroupMemoryBarrierWithGroupSync(); // Wait for stage to complete
    }
}



#pragma kernel IFFT
[numthreads(128, 1, 1)]
void IFFT(uint3 id : SV_DispatchThreadID) {
    int n = WINDOW_SIZE;
    int bits = (int)log2(n);
    uint idx = id.x;
    
    if (idx >= (uint)n) return;

    // Bit-reversal permutation (parallel)
    uint rev = reversebits(idx) >> (32 - bits);
    if (idx < rev) {
        float2 temp = Data[idx];
        Data[idx] = Data[rev];
        Data[rev] = temp;
    }
    
    GroupMemoryBarrierWithGroupSync(); // Wait for all bit reversals

    // IFFT stages - each stage must complete before next begins
    for (int s = 1; s <= bits; s++) {
        int m = 1 << s;
        int m2 = m >> 1;
        float2 w_m = nth_root_inv(m, 1);

        // Each thread processes its butterfly operations
        int k_base = (idx / m2) * m;
        int j = idx % m2;
        
        if (k_base + j + m2 < n) {
            float2 w = w_m;
            // Calculate w^j
            for (int p = 0; p < j; p++) {
                w = complex_mult(w, w_m);
            }
            
            float2 t = complex_mult(w, Data[k_base + j + m2]);
            float2 u = Data[k_base + j];
            Data[k_base + j] = u + t;
            Data[k_base + j + m2] = u - t;
        }
        
        GroupMemoryBarrierWithGroupSync(); // Wait for stage to complete
    }

    // Normalization (parallel)
    if (idx < (uint)n) {
        Data[idx] = Data[idx] * (1.0 / n);
    }
}


#endif
