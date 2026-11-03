// Opt-in per-family gains (FamilyGains.cs, TakeRender.Settings.familyGains). A pass
// after the mirror and complete-route kernels scales each sampled early arrival of
// family f (0 first-order image, 1-5 complete D, DD, RD, DR, RR) by
// (1 + k_f rho)^-power, where rho is that family's candidate routes per sample.
// Direct sound and the residual walk are never touched. The transport kernels are
// unchanged, so with gains off (no pass) every path is bit-identical to before.
float4 FamilyGainK0, FamilyGainK1; // k of R, D, DD, RD; then DR, RR
float FamilyGainPower;
void ScaleFamily(uint slot,uint family,float rho)
{
    float k=dot(FamilyGainK0,(float4)(uint4(0,1,2,3)==family))+dot(FamilyGainK1.xy,(float2)(uint2(4,5)==family));
    float x=k*rho;
    if(x==0)return;
    float gain=pow(1+x,-FamilyGainPower);
    RayPath path=RayPaths[slot];path.pressureGain*=gain;path.pressureImag*=gain;RayPaths[slot]=path;
}
// Same slots as PrepareMirrors; rho is its compensated candidate weight.
void GainMirrors(uint index)
{
    uint capacity=MirrorCandidateCount*FrequencyBins*MirrorSiteCapacity;
    if(index>=FrameCount*capacity)return;
    uint slot=(index/capacity)*RayPathsPerFrame+RayPairRanges[RayTracePair].x+MirrorSiteCapacity*FrequencyBins+index%capacity;
    ScaleFamily(slot,0,(float)PrimitiveCount/MirrorCandidateCount);
}
// Every route shares the samples when all candidates are enumerated; otherwise each
// family draws its own share and a route's inclusion weight is its routes per draw.
float CompleteRoutesPerSample(CompleteRoute route)
{
    float total=dot(CompleteSizes(),1)+CompleteMirrorPairs();
    return total<=CompleteCapacity?total/CompleteSamples:route.weight;
}
// Same slots and sample-to-route mapping as IntegrateCompleteRoutes, for the route
// tables of the dispatch that has just run.
void GainComplete(uint index,bool planar)
{
    uint symmetry=planar?2:1;
    if(index>=CompleteSamples*symmetry*FrequencyBins)return;
    uint count=CompleteCount[0];if(count==0)return;
    uint sample=index/(symmetry*FrequencyBins);
    uint per=CompleteSamples/count,extra=CompleteSamples%count,split=(per+1)*extra;
    uint rank=sample<split?sample/(per+1):extra+(sample-split)/per;
    CompleteRoute route=CompleteCandidates[CompleteSelected[rank]];
    ScaleFamily(CompleteOffset+index,1+route.family,CompleteRoutesPerSample(route));
}
