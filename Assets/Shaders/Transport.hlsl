// Shared geometric preview model. Reflection/transmission/scattering and the
// wavelength-scaled edge heuristic extend the original 2D shader to triangles.
// This is not a validated wave/diffraction reference.
struct RayMaterial { float4 surface,medium; };
struct Edge { float4 a,b; uint4 ids; };
struct RayPath { float4 direction; float delaySeconds,pressureGain; uint frequencyBin,hrtfIndex; };
StructuredBuffer<RayMaterial> RayMaterials;
StructuredBuffer<Edge> Edges;
StructuredBuffer<float4> HrtfDirections;
RWStructuredBuffer<RayPath> RayPaths;
uint FrequencyBins,RaysPerPair,MaxBounces,SlotsPerPair,EdgeCount,HrtfDirectionCount,UseAirAbsorption;
float FrequencyStep,DiffractionFactor,MaximumPathDelay;
static const float RayEpsilon=0.0001,RayPi=3.141592653589793;
uint HashRay(uint x) { x^=x>>16;x*=0x7feb352d;x^=x>>15;x*=0x846ca68b;x^=x>>16;return x; }
float RandomRay(inout uint x) {x=HashRay(x+0x9e3779b9);return (x>>8)*(1.0/16777216.0);}
float3 UnitDirection(inout uint rng,bool planar)
{
    float angle=2*RayPi*RandomRay(rng);
    if(planar)return float3(cos(angle),sin(angle),0);
    float z=2*RandomRay(rng)-1,r=sqrt(max(0,1-z*z));return float3(r*cos(angle),r*sin(angle),z);
}
float AirCoefficient(float hz)
{
    if(UseAirAbsorption==0)return 0;
    const float losses[8]={0.0001,0.0003,0.0007,0.0015,0.0035,0.0100,0.0320,0.1100};
    float at=clamp(log2(max(125,hz)/125),0,7);uint a=(uint)at;
    return lerp(losses[a],losses[min(a+1,7)],frac(at))/8.685889638;
}
float PrimitiveDistance(uint id,uint offset,float3 o,float3 d,bool planar,out float3 normal)
{
    Primitive p=Primitives[id];State state=States[offset+p.indices.x];normal=0;
    if((state.flags.x&3)!=3)return 1e30;
    float3 a=state.position.xyz+Rotate(state.rotation,p.a.xyz),b=state.position.xyz+Rotate(state.rotation,p.b.xyz);
    if(planar)
    {
        float2 e=b.xy-a.xy,q=a.xy-o.xy;float det=d.x*e.y-d.y*e.x;
        if(abs(det)<1e-12)return 1e30;
        float t=(q.x*e.y-q.y*e.x)/det,u=(q.x*d.y-q.y*d.x)/det;
        normal=normalize(float3(-e.y,e.x,0));return t>RayEpsilon && u>=0 && u<=1?t:1e30;
    }
    float3 c=state.position.xyz+Rotate(state.rotation,p.c.xyz),e1=b-a,e2=c-a,h=cross(d,e2);
    float det=dot(e1,h);if(abs(det)<1e-12)return 1e30;
    float3 s=o-a,q=cross(s,e1);float u=dot(s,h)/det,v=dot(d,q)/det,t=dot(e2,q)/det;
    normal=normalize(cross(e1,e2));return t>RayEpsilon && u>=0 && v>=0 && u+v<=1?t:1e30;
}
bool NearestHit(uint offset,float3 o,float3 d,float limit,bool planar,out uint hit,out float distance,out float3 normal)
{
    hit=0xffffffff;distance=limit;normal=0;
#ifdef USE_BVH
    if(BvhNodeCount>0)
    {
        uint stack[32],depth=0,index=0;
        while(true)
        {
            BvhNode node=BvhNodes[index];
            if(BoundsHit(node,o,d*distance,planar))
            {
                if(node.links.w==0){stack[depth++]=node.links.y;index=node.links.x;continue;}
                for(uint j=0;j<node.links.w;j++)
                {uint id=BvhIndices[node.links.z+j];float3 n;float t=PrimitiveDistance(id,offset,o,d,planar,n);if(t<distance){hit=id;distance=t;normal=n;}}
            }
            if(depth==0)break;index=stack[--depth];
        }
    }
    for(uint j=StaticPrimitiveCount;j<PrimitiveCount;j++)
    {uint id=BvhIndices[j];float3 n;float t=PrimitiveDistance(id,offset,o,d,planar,n);if(t<distance){hit=id;distance=t;normal=n;}}
#else
    for(uint id=0;id<PrimitiveCount;id++)
    {float3 n;float t=PrimitiveDistance(id,offset,o,d,planar,n);if(t<distance){hit=id;distance=t;normal=n;}}
#endif
    return hit!=0xffffffff;
}
bool Visible(uint offset,float3 a,float3 b,bool planar)
{
    float lengthTo=length(b-a);if(lengthTo<=2*RayEpsilon)return true;
    uint hit;float distance;float3 normal;
    return !NearestHit(offset,a,(b-a)/lengthTo,lengthTo-RayEpsilon,planar,hit,distance,normal);
}
uint HeadDirection(uint offset,uint listener,float3 direction)
{
    float4 q=States[offset+listener].rotation;q.xyz=-q.xyz;float3 local=Rotate(q,direction);
    uint closest=0;float best=-1e30;
    for(uint i=0;i<HrtfDirectionCount;i++){float score=dot(local,HrtfDirections[i].xyz);if(score>best){best=score;closest=i;}}
    return closest;
}
RayPath Arrival(uint offset,uint listener,float3 origin,float3 end,float delay,float gain,uint bin)
{
    RayPath value=(RayPath)0;
    if(delay<0 || delay>MaximumPathDelay || gain==0)return value;
    float3 direction=origin-end;float distance=length(direction);direction=distance>0?direction/distance:float3(0,0,1);
    value.direction=float4(direction,0);value.delaySeconds=delay;value.pressureGain=gain;
    value.frequencyBin=bin;value.hrtfIndex=HeadDirection(offset,listener,direction);return value;
}
void SpectralDirect(uint index,bool planar)
{
    if(index>=FrameCount*PairCount*FrequencyBins)return;
    uint bin=index%FrequencyBins,query=index/FrequencyBins,frame=query/PairCount,pair=query%PairCount;
    uint offset=frame*ObjectCount,source=Sources[pair/ListenerCount],listener=Listeners[pair%ListenerCount];
    float3 a=States[offset+source].position.xyz,b=States[offset+listener].position.xyz;if(planar){a.z=0;b.z=0;}
    float lengthTo=length(b-a),gain=ReferenceDistance/max(MinimumDistance,lengthTo),hz=max(1,bin*FrequencyStep);
    float3 direction=lengthTo>0?(b-a)/lengthTo:float3(0,0,0),position=a;float remaining=lengthTo;
    // Deterministic direct/transmitted path: keep every frequency even at low ray budgets.
    for(uint crossing=0;crossing<PrimitiveCount && remaining>2*RayEpsilon;crossing++)
    {
        uint hit;float distance;float3 normal;
        if(!NearestHit(offset,position,direction,remaining-RayEpsilon,planar,hit,distance,normal))break;
        RayMaterial material=RayMaterials[Primitives[hit].indices.y*FrequencyBins+bin];
        gain*=sqrt(saturate((1-material.surface.x)*material.surface.z));
        if(gain==0)break;
        position+=direction*(distance+RayEpsilon);remaining-=distance+RayEpsilon;
    }
    gain*=exp(-AirCoefficient(hz)*lengthTo);
    RayPaths[query*SlotsPerPair+bin]=Arrival(offset,listener,a,b,lengthTo/SpeedOfSound,gain,bin);
}
bool NearbyEdge(uint offset,float3 origin,float3 direction,float before,float threshold,bool planar,out float3 edgePoint,out float weight)
{
    edgePoint=0;weight=0;float closest=before;
    if(threshold<=0)return false;
    for(uint i=0;i<EdgeCount;i++)
    {
        Edge edge=Edges[i];State pose=States[offset+edge.ids.x];if((pose.flags.x&3)!=3)continue;
        float3 a=pose.position.xyz+Rotate(pose.rotation,edge.a.xyz),b=pose.position.xyz+Rotate(pose.rotation,edge.b.xyz);
        if(planar){a.z=0;b.z=0;}
        float3 e=b-a,w=origin-a;float c=dot(e,e),projection=dot(direction,e),denom=c-projection*projection;
        float along=denom>1e-12?(projection*dot(e,w)-c*dot(direction,w))/denom:dot(a-origin,direction);
        float u=c>0?saturate((dot(e,w)+projection*along)/c):0;
        float3 candidate=a+e*u;along=dot(candidate-origin,direction);
        float distance=length(candidate-(origin+direction*along));
        if(along<=RayEpsilon || along>=closest || length(candidate-origin)<=threshold || distance>=threshold)continue;
        if(!Visible(offset,origin,candidate,planar))continue;
        edgePoint=candidate;closest=along;weight=1-distance/threshold;
    }
    return weight>0;
}
void TracePaths(uint index,bool planar)
{
    if(index>=FrameCount*PairCount*RaysPerPair)return;
    uint ray=index%RaysPerPair,query=index/RaysPerPair,frame=query/PairCount,pair=query%PairCount;
    uint offset=frame*ObjectCount,source=Sources[pair/ListenerCount],listener=Listeners[pair%ListenerCount];
    uint first=query*SlotsPerPair+FrequencyBins+ray*MaxBounces;
    for(uint k=0;k<MaxBounces;k++)RayPaths[first+k]=(RayPath)0;
    uint bin=(ray*FrequencyBins/RaysPerPair)%FrequencyBins,rng=HashRay(ray+pair*0x85ebca6b+1);
    float hz=max(1,bin*FrequencyStep),threshold=SpeedOfSound/hz*DiffractionFactor;
    float3 position=States[offset+source].position.xyz,end=States[offset+listener].position.xyz;
    if(planar){position.z=0;end.z=0;}
    float3 direction=UnitDirection(rng,planar);float gain=1,time=0,pathLength=0,currentIor=1;
    float weight=(float)FrequencyBins/RaysPerPair;
    for(uint bounce=0;bounce<MaxBounces;bounce++)
    {
        float speed=SpeedOfSound/currentIor,limit=(MaximumPathDelay-time)*speed;
        if(limit<=0 || gain==0)break;
        uint hit;float distance;float3 normal;
        bool intersects=NearestHit(offset,position,direction,limit,planar,hit,distance,normal);
        float3 corner;float cornerWeight;
        if(currentIor==1 && NearbyEdge(offset,position,direction,distance,threshold,planar,corner,cornerWeight))
        {
            float travelled=length(corner-position);time+=travelled/speed;pathLength+=travelled;
            gain*=cornerWeight*exp(-AirCoefficient(hz)*travelled);
            float last=length(end-corner);
            if(Visible(offset,corner,end,planar))RayPaths[first+bounce]=Arrival(offset,listener,corner,end,time+last/SpeedOfSound,
                gain*weight*exp(-AirCoefficient(hz)*last)*ReferenceDistance/max(MinimumDistance,pathLength+last),bin);
            direction=normalize(lerp(direction,normalize(corner-position),cornerWeight));position=corner+direction*RayEpsilon;continue;
        }
        if(!intersects)break;
        position+=direction*distance;time+=distance/speed;pathLength+=distance;
        RayMaterial material=RayMaterials[Primitives[hit].indices.y*FrequencyBins+bin];
        gain*=sqrt(saturate(1-material.surface.x))*exp(-(currentIor==1?AirCoefficient(hz):material.medium.x)*distance);
        bool entering=dot(direction,normal)<0;float3 n=entering?normal:-normal;
        float last=length(end-position),cosine=last>0?max(0,dot(n,(end-position)/last)):0;
        if(currentIor==1 && cosine>0 && Visible(offset,position+n*RayEpsilon,end,planar))
            RayPaths[first+bounce]=Arrival(offset,listener,position,end,time+last/SpeedOfSound,
                gain*sqrt(saturate(1-material.surface.z))*weight*cosine*.5*exp(-AirCoefficient(hz)*last)*ReferenceDistance/max(MinimumDistance,pathLength+last),bin);
        float nextIor=entering?max(.001,material.surface.w):1;
        float reflectance=1-saturate(material.surface.z),cosTheta=abs(dot(direction,n));
        reflectance+= (1-reflectance)*pow(saturate(1-cosTheta),5);
        if(RandomRay(rng)>reflectance)
        {
            float3 transmitted=refract(direction,n,currentIor/nextIor);
            if(dot(transmitted,transmitted)>0){direction=normalize(transmitted);currentIor=nextIor;position+=direction*RayEpsilon;continue;}
        }
        float3 diffuse=UnitDirection(rng,planar);if(dot(diffuse,n)<0)diffuse=-diffuse;
        direction=normalize(lerp(reflect(direction,n),diffuse,saturate(material.surface.y)));
        position+=n*RayEpsilon;
    }
}
[numthreads(64,1,1)] void SpectralDirect2D(uint3 id:SV_DispatchThreadID){SpectralDirect(id.x,true);}
[numthreads(64,1,1)] void SpectralDirect3D(uint3 id:SV_DispatchThreadID){SpectralDirect(id.x,false);}
[numthreads(64,1,1)] void SpectralDirect2DBvh(uint3 id:SV_DispatchThreadID){SpectralDirect(id.x,true);}
[numthreads(64,1,1)] void SpectralDirect3DBvh(uint3 id:SV_DispatchThreadID){SpectralDirect(id.x,false);}
[numthreads(64,1,1)] void Trace2D(uint3 id:SV_DispatchThreadID){TracePaths(id.x,true);}
[numthreads(64,1,1)] void Trace3D(uint3 id:SV_DispatchThreadID){TracePaths(id.x,false);}
[numthreads(64,1,1)] void Trace2DBvh(uint3 id:SV_DispatchThreadID){TracePaths(id.x,true);}
[numthreads(64,1,1)] void Trace3DBvh(uint3 id:SV_DispatchThreadID){TracePaths(id.x,false);}
