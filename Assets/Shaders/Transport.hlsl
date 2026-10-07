// Shared geometric preview model. Reflection/transmission/scattering and the
// wavelength-scaled edge heuristic extend the original 2D shader to triangles.
// This is not a validated wave/diffraction reference.
struct RayMaterial { float4 surface,medium; };
struct Edge { float4 a,b; uint4 ids; };
struct RayPath { float4 direction; float delaySeconds,pressureGain; uint frequencyBin,identity; };
StructuredBuffer<RayMaterial> RayMaterials;
StructuredBuffer<Edge> Edges;
struct Volume { float4 minimum,maximum; uint4 ids; };
StructuredBuffer<Volume> Volumes;
RWStructuredBuffer<RayPath> RayPaths;
StructuredBuffer<uint> RayPairEnabled;
uint RaysPerBin,RaySeed;
uint FrequencyBins,RaysPerPair,MaxBounces,SlotsPerPair,EdgeCount,VolumeCount,UseAirAbsorption;
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
float3 PastPoint(State state,float3 local,float lag)
{
    float3 worldPoint=Rotate(state.rotation,local);
    if((state.flags.x&48)==32)
    {
        float angle=length(state.angularVelocity.xyz)*lag;
        if(angle>0){float3 axis=normalize(state.angularVelocity.xyz);worldPoint=worldPoint*cos(angle)-cross(axis,worldPoint)*sin(angle)+axis*dot(axis,worldPoint)*(1-cos(angle));}
        return state.position.xyz-state.velocity.xyz*lag+worldPoint;
    }
    return state.position.xyz+worldPoint;
}
float PrimitiveDistanceAt(uint id,uint offset,float3 o,float3 d,bool planar,float lag,out float3 normal)
{
    Primitive p=Primitives[id];State state=States[offset+p.indices.x];normal=0;
    if((state.flags.x&3)!=3)return 1e30;
    float3 a=PastPoint(state,p.a.xyz,lag),b=PastPoint(state,p.b.xyz,lag);
    if(planar)
    {
        float2 e=b.xy-a.xy,q=a.xy-o.xy;float det=d.x*e.y-d.y*e.x;
        if(abs(det)<1e-12)return 1e30;
        float t=(q.x*e.y-q.y*e.x)/det,u=(q.x*d.y-q.y*d.x)/det;
        normal=normalize(float3(-e.y,e.x,0));return t>RayEpsilon && u>=0 && u<=1?t:1e30;
    }
    float3 c=PastPoint(state,p.c.xyz,lag),e1=b-a,e2=c-a,h=cross(d,e2);
    float det=dot(e1,h);if(abs(det)<1e-12)return 1e30;
    float3 s=o-a,q=cross(s,e1);float u=dot(s,h)/det,v=dot(d,q)/det,t=dot(e2,q)/det;
    normal=normalize(cross(e1,e2));return t>RayEpsilon && u>=0 && v>=0 && u+v<=1?t:1e30;
}
float PrimitiveDistance(uint id,uint offset,float3 o,float3 d,bool planar,out float3 normal)
{return PrimitiveDistanceAt(id,offset,o,d,planar,0,normal);}
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
static const uint AirMedium=0xffffffff;
// Containment uses the nearest oriented boundary of each closed object. Choosing
// the smallest containing volume restores the parent medium in nested solids, and
// also initializes sources/listeners that start inside one. No fixed medium stack.
uint MediumAt(uint offset,float3 location,bool planar)
{
    uint medium=AirMedium;float smallest=1e30;
    float3 probe=planar?normalize(float3(.819,.573,0)):normalize(float3(.819,.573,.371));
    for(uint v=0;v<VolumeCount;v++)
    {
        Volume volume=Volumes[v];State pose=States[offset+volume.ids.z];if((pose.flags.x&3)!=3)continue;
        float4 inverse=pose.rotation;inverse.xyz=-inverse.xyz;float3 local=Rotate(inverse,location-pose.position.xyz);
        if(any(local.xy<volume.minimum.xy) || any(local.xy>volume.maximum.xy) ||
            (!planar && (local.z<volume.minimum.z || local.z>volume.maximum.z)))continue;
        float nearest=1e30;float3 normal=0;
        for(uint i=0;i<volume.ids.y;i++)
        {
            float3 n;float distance=PrimitiveDistance(volume.ids.x+i,offset,location,probe,planar,n);
            if(distance<nearest){nearest=distance;normal=n;}
        }
        if(volume.maximum.w<smallest && nearest<1e30 && dot(normal,probe)*volume.minimum.w>0)
        {smallest=volume.maximum.w;medium=volume.ids.w;}
    }
    return medium;
}
float MediumIor(uint medium,uint bin){return medium==AirMedium?1:max(.001,RayMaterials[medium*FrequencyBins+bin].surface.w);}
float MediumLoss(uint medium,uint bin,float hz){return medium==AirMedium?AirCoefficient(hz):RayMaterials[medium*FrequencyBins+bin].medium.x;}
void Travel(float distance,uint medium,uint bin,float hz,inout float time,inout float gain)
{
    time+=distance*MediumIor(medium,bin)/SpeedOfSound;
    gain*=exp(-MediumLoss(medium,bin,hz)*distance);
}
RayPath Arrival(uint offset,uint listener,float3 origin,float3 end,float delay,float gain,uint bin,float ratio=1,uint identity=0)
{
    RayPath value=(RayPath)0;
    if(delay<0 || delay>MaximumPathDelay || gain==0)return value;
    float3 direction=origin-end;float distance=length(direction);direction=distance>0?direction/distance:float3(0,0,0);
    value.direction=float4(direction,1-ratio);value.delaySeconds=delay;value.pressureGain=gain;
    value.frequencyBin=bin;value.identity=identity;return value;
}
// Retarded source position; interfaces retain the scene pose of this solve.
// Iteration follows piecewise material travel time, not distance / air speed.
void SpectralDirect(uint index,bool planar)
{
    if(index>=FrameCount*PairCount*FrequencyBins)return;
    uint bin=index%FrequencyBins,query=index/FrequencyBins,frame=query/PairCount,pair=query%PairCount;
    RayPaths[query*SlotsPerPair+bin]=(RayPath)0;
    if(RayPairEnabled[query]==0)return;
    uint offset=frame*ObjectCount,source=Sources[pair/ListenerCount],listener=Listeners[pair%ListenerCount];
    float3 sourceNow=States[offset+source].position.xyz,b=States[offset+listener].position.xyz;
    float3 sv=Velocity(offset+source),lv=Velocity(offset+listener);if(planar){sourceNow.z=0;b.z=0;sv.z=0;lv.z=0;}
    float hz=max(1,bin*FrequencyStep),estimate=RetardedDelay(b-sourceNow,sv,SpeedOfSound/MediumIor(MediumAt(offset,sourceNow,planar),bin));
    float time=0,gain=0,ratio=1,lower=0,upper=MaximumPathDelay;float3 a=sourceNow;uint identity=1;bool converged=false;
    estimate=min(estimate,upper);
    // Alternate fixed-point proposals with bisection. A bracket prevents slow
    // material transitions or float-level cycles from defeating convergence.
    for(uint iteration=0;iteration<64;iteration++)
    {
        a=sourceNow-sv*estimate;float remaining=length(b-a);float3 direction=remaining>0?(b-a)/remaining:float3(0,0,0),position=a;
        gain=ReferenceDistance/max(MinimumDistance,remaining);time=0;identity=1;
        uint medium=MediumAt(offset,a,planar);float speed=SpeedOfSound/MediumIor(medium,bin);
        ratio=1/(1-dot(sv,direction)/speed);
        for(uint crossing=0;crossing<PrimitiveCount && remaining>2*RayEpsilon;crossing++)
        {
            uint hit;float distance;float3 normal;
            if(!NearestHit(offset,position,direction,remaining-RayEpsilon,planar,hit,distance,normal))break;
            Travel(distance,medium,bin,hz,time,gain);
            RayMaterial material=RayMaterials[Primitives[hit].indices.y*FrequencyBins+bin];
            gain*=sqrt(saturate((1-material.surface.x)*material.surface.z));
            position+=direction*distance;remaining-=distance;
            uint next=MediumAt(offset,position+direction*(2*RayEpsilon),planar);
            float nextSpeed=SpeedOfSound/MediumIor(next,bin);float3 velocity=PointVelocity(offset+Primitives[hit].indices.x,position);
            ratio*=(1-dot(velocity,direction)/speed)/(1-dot(velocity,direction)/nextSpeed);
            identity=HashRay(identity^(hit+1));medium=next;speed=nextSpeed;
        }
        Travel(remaining,medium,bin,hz,time,gain);ratio*=1-dot(lv,direction)/speed;
        float tolerance=4e-7*max(1,time);
        if(abs(time-estimate)<=tolerance){converged=true;break;}
        if(time>MaximumPathDelay && estimate>=MaximumPathDelay)break;
        if(time>estimate)lower=estimate;else upper=estimate;
        if(upper-lower<=tolerance){time=estimate;converged=true;break;}
        estimate=(iteration%2==0 && time>lower && time<upper)?time:(lower+upper)*.5;
    }
    RayPaths[query*SlotsPerPair+bin]=Arrival(offset,listener,a,b,time,gain,bin,converged?ratio:asfloat(0x7fc00000),identity);
}
bool NearbyEdge(uint offset,float3 origin,float3 direction,float before,float threshold,bool planar,out float3 edgePoint,out float weight,out uint edgeId)
{
    edgePoint=0;weight=0;edgeId=0;float closest=before;
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
        edgePoint=candidate;closest=along;weight=1-distance/threshold;edgeId=i;
    }
    return weight>0;
}
// First-order mirror path from a sampled primitive's image source. Sampling
// primitive IDs avoids estimating a delta reflection with a diffuse connection.
// A cyclic stratification covers each primitive once per complete sample stratum.
RayPath MirrorPath(uint offset,uint source,uint listener,uint bin,uint sampleIndex,bool planar)
{
    RayPath result=(RayPath)0;if(PrimitiveCount==0)return result;
    uint id=(sampleIndex+HashRay(source*0x85ebca6b+listener+RaySeed))%PrimitiveCount;
    Primitive p=Primitives[id];State pose=States[offset+p.indices.x];if((pose.flags.x&3)!=3)return result;
    float3 sourceNow=States[offset+source].position.xyz,end=States[offset+listener].position.xyz;
    float3 sv=Velocity(offset+source),lv=Velocity(offset+listener);if(planar){sourceNow.z=0;end.z=0;sv.z=0;lv.z=0;}
    uint medium=MediumAt(offset,sourceNow,planar);float speed=SpeedOfSound/MediumIor(medium,bin);
    float total=length(end-sourceNow)/speed,last=total*.5,distance=0;float3 start=sourceNow,reflectionPoint=0,image=0,direction=0,unused;
    bool converged=false;
    for(uint iteration=0;iteration<32;iteration++)
    {
        float3 a=PastPoint(pose,p.a.xyz,last),b=PastPoint(pose,p.b.xyz,last);
        float3 normal=planar?normalize(float3(-(b-a).y,(b-a).x,0)):normalize(cross(b-a,PastPoint(pose,p.c.xyz,last)-a));
        start=sourceNow-sv*total;
        if(dot(start-a,normal)*dot(end-a,normal)<=0)return result;
        image=start-2*normal*dot(start-a,normal);direction=normalize(end-image);
        distance=PrimitiveDistanceAt(id,offset,image,direction,planar,last,unused);
        if(distance>=length(end-image)-RayEpsilon)return result;
        reflectionPoint=image+direction*distance;
        float nextLast=length(end-reflectionPoint)/speed,nextTotal=length(start-reflectionPoint)/speed+nextLast;
        if(abs(nextTotal-total)+abs(nextLast-last)<4e-7*max(1,nextTotal)){total=nextTotal;last=nextLast;converged=true;break;}
        total=nextTotal;last=nextLast;
    }
    for(uint other=0;other<id;other++)
    {
        if(Primitives[other].indices.x!=p.indices.x)continue;
        float t=PrimitiveDistanceAt(other,offset,image,direction,planar,last,unused);
        if(abs(t-distance)<RayEpsilon)return result;
    }
    // Test the two legs against other objects at the solve pose. The reflecting
    // primitive is tested at reflection time above, so omit its owner here.
    for(uint other=0;other<PrimitiveCount;other++)
    {
        if(other==id)continue;
        float lag=Primitives[other].indices.x==p.indices.x?last:0;
        float firstLength=length(reflectionPoint-start),lastLength=length(end-reflectionPoint);
        if(PrimitiveDistanceAt(other,offset,start,normalize(reflectionPoint-start),planar,lag,unused)<firstLength-RayEpsilon ||
            PrimitiveDistanceAt(other,offset,reflectionPoint,normalize(end-reflectionPoint),planar,lag,unused)<lastLength-RayEpsilon)return result;
    }
    float3 incoming=normalize(reflectionPoint-start),outgoing=normalize(end-reflectionPoint),velocity=PointVelocity(offset+p.indices.x,reflectionPoint,last);
    if(planar)velocity.z=0;
    float ratio=(speed-dot(lv,outgoing))/(speed-dot(sv,incoming))*(speed-dot(velocity,incoming))/(speed-dot(velocity,outgoing));
    RayMaterial material=RayMaterials[p.indices.y*FrequencyBins+bin];
    float gain=sqrt(saturate((1-material.surface.x)*(1-material.surface.z)*(1-material.surface.y)));
    float time=0,lengthTo=total*speed;Travel(lengthTo,medium,bin,max(1,bin*FrequencyStep),time,gain);
    return Arrival(offset,listener,reflectionPoint,end,time,gain*(float)PrimitiveCount/RaysPerBin*ReferenceDistance/max(MinimumDistance,lengthTo),bin,
        converged?ratio:asfloat(0x7fc00000),HashRay(0x40000000u^id));
}
void TracePaths(uint index,bool planar)
{
    if(index>=FrameCount*PairCount*RaysPerPair)return;
    uint ray=index%RaysPerPair,query=index/RaysPerPair,frame=query/PairCount,pair=query%PairCount;
    uint offset=frame*ObjectCount,source=Sources[pair/ListenerCount],listener=Listeners[pair%ListenerCount];
    uint first=query*SlotsPerPair+FrequencyBins+ray*(MaxBounces+1);
    for(uint k=0;k<=MaxBounces;k++)RayPaths[first+k]=(RayPath)0;
    if(RayPairEnabled[query]==0)return;
    uint bin=ray/RaysPerBin,rng=HashRay((ray%RaysPerBin)+pair*0x85ebca6b+RaySeed+1);
    RayPaths[first]=MirrorPath(offset,source,listener,bin,ray%RaysPerBin,planar);first++;
    float hz=max(1,bin*FrequencyStep),threshold=SpeedOfSound/hz*DiffractionFactor;
    float3 position=States[offset+source].position.xyz,end=States[offset+listener].position.xyz;
    if(planar){position.z=0;end.z=0;}
    float3 direction=UnitDirection(rng,planar);float gain=1,time=0,pathLength=0;
    uint medium=MediumAt(offset,position,planar);
    float3 sv=Velocity(offset+source),lv=Velocity(offset+listener);if(planar){sv.z=0;lv.z=0;}
    float ratio=1/(1-dot(sv,direction)/(SpeedOfSound/MediumIor(medium,bin)));
    uint identity=HashRay(0x80000000u^(ray%RaysPerBin));
    float weight=1.0/RaysPerBin;
    for(uint bounce=0;bounce<MaxBounces;bounce++)
    {
        float currentIor=MediumIor(medium,bin),speed=SpeedOfSound/currentIor,limit=(MaximumPathDelay-time)*speed;
        if(limit<=0 || gain==0)break;
        uint hit;float distance;float3 normal;
        bool intersects=NearestHit(offset,position,direction,limit,planar,hit,distance,normal);
        float3 corner;float cornerWeight;uint edgeId;
        if(medium==AirMedium && NearbyEdge(offset,position,direction,distance,threshold,planar,corner,cornerWeight,edgeId))
        {
            float travelled=length(corner-position);time+=travelled/speed;pathLength+=travelled;
            gain*=cornerWeight*exp(-AirCoefficient(hz)*travelled);
            float last=length(end-corner);float3 outgoing=normalize(end-corner),velocity=PointVelocity(offset+Edges[edgeId].ids.x,corner);
            if(planar)velocity.z=0;identity=HashRay(identity^(edgeId+0x20000000u));
            float arrivalRatio=ratio*(1-dot(velocity,direction)/speed)/(1-dot(velocity,outgoing)/SpeedOfSound)*(1-dot(lv,outgoing)/SpeedOfSound);
            if(Visible(offset,corner,end,planar))RayPaths[first+bounce]=Arrival(offset,listener,corner,end,time+last/SpeedOfSound,
                gain*weight*exp(-AirCoefficient(hz)*last)*ReferenceDistance/max(MinimumDistance,pathLength+last),bin,arrivalRatio,identity);
            float3 nextDirection=normalize(lerp(direction,normalize(corner-position),cornerWeight));
            ratio*=(1-dot(velocity,direction)/speed)/(1-dot(velocity,nextDirection)/speed);direction=nextDirection;position=corner+direction*RayEpsilon;continue;
        }
        if(!intersects)break;
        position+=direction*distance;Travel(distance,medium,bin,hz,time,gain);pathLength+=distance;
        RayMaterial material=RayMaterials[Primitives[hit].indices.y*FrequencyBins+bin];
        gain*=sqrt(saturate(1-material.surface.x));
        bool entering=dot(direction,normal)<0;float3 n=entering?normal:-normal;
        float3 velocity=PointVelocity(offset+Primitives[hit].indices.x,position);if(planar)velocity.z=0;
        identity=HashRay(identity^(hit+1));
        float3 toListener=normalize(end-position);
        float arrivalRatio=ratio*(1-dot(velocity,direction)/speed)/(1-dot(velocity,toListener)/speed)*(1-dot(lv,toListener)/speed);
        float last=length(end-position),cosine=last>0?max(0,dot(n,(end-position)/last)):0;
        if(cosine>0 && Visible(offset,position+n*RayEpsilon,end,planar))
            RayPaths[first+bounce]=Arrival(offset,listener,position,end,time+last/speed,
                gain*sqrt(saturate((1-material.surface.z)*material.surface.y))*weight*cosine*.5*exp(-MediumLoss(medium,bin,hz)*last)*ReferenceDistance/max(MinimumDistance,pathLength+last),bin,arrivalRatio,identity);
        uint nextMedium=MediumAt(offset,position+direction*(2*RayEpsilon),planar);
        float nextIor=MediumIor(nextMedium,bin);
        float reflectance=1-saturate(material.surface.z),cosTheta=abs(dot(direction,n));
        reflectance+= (1-reflectance)*pow(saturate(1-cosTheta),5);
        if(RandomRay(rng)>reflectance)
        {
            float3 transmitted=refract(direction,n,currentIor/nextIor);
            if(dot(transmitted,transmitted)>0)
            {transmitted=normalize(transmitted);ratio*=(1-dot(velocity,direction)/speed)/(1-dot(velocity,transmitted)/(SpeedOfSound/nextIor));
                identity=HashRay(identity^0x10000000u);direction=transmitted;medium=nextMedium;continue;}
        }
        float3 diffuse=UnitDirection(rng,planar);if(dot(diffuse,n)<0)diffuse=-diffuse;
        float3 nextDirection=RandomRay(rng)<saturate(material.surface.y)?diffuse:reflect(direction,n);
        ratio*=(1-dot(velocity,direction)/speed)/(1-dot(velocity,nextDirection)/speed);direction=nextDirection;
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
