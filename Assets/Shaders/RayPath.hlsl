#ifndef ACOUSTIC_RAY_PATH
#define ACOUSTIC_RAY_PATH
struct RayPath
{
    // direction.xyz: listener-to-arrival; direction.w: 1 - received/emitted frequency ratio.
    float4 direction;
    float delaySeconds, pressureGain;
    uint frequencyBin, identity;
    float pressureImag;
    uint emissionPhase, birthLow, birthHigh;
};
#endif
