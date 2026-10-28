bool TunedBoundaryTerm(float phase)
{
    float turns=round(phase/(2*RayPi));
    return abs(phase-turns*(2*RayPi))<=4*1.192092896e-7*RayPi*max(1,abs(turns));
}
float TunedEdgeAngle(float3 direction, float3 face, float3 normal, float exterior)
{
    float2 radial = float2(dot(direction, face), dot(direction, normal));
    float angle = Span(abs(radial.y),radial.x);
    if(radial.y<0)angle=-angle;
    angle = angle < 0 ? angle + 2 * RayPi : angle;
    // Face-grazing connections use the visibility tolerance. Rigid rotation
    // must not wrap roundoff at zero to 2*pi and classify the ray as solid.
    if (min(angle, 2 * RayPi - angle) * length(radial) <= RayEpsilon)
        angle = 0;
    if (abs(angle - exterior) * length(radial) <= RayEpsilon)
        angle = exterior;
    return angle;
}
float TunedEdgeKernel(uint id, uint stateOffset, float3 source, float3 edgePoint, float3 receiver, bool is2D)
{
    float3 a, axis, face, normal;
    float extent, angle;
    EdgeFrame(id, stateOffset, is2D, a, axis, face, normal, extent, angle);
    float3 incoming = source - edgePoint, outgoing = receiver - edgePoint;
    float m = length(incoming), l = length(outgoing), zs = dot(incoming, axis), zr = dot(outgoing, axis);
    float rs = length(incoming - zs * axis), rr = length(outgoing - zr * axis);
    if (rs <= RayEpsilon || rr <= RayEpsilon)
        return 0;
    float ts = TunedEdgeAngle(incoming, face, normal, angle), tr = TunedEdgeAngle(outgoing, face, normal, angle);
    if (ts > angle || tr > angle)
        return 0; // Inside the solid wedge.
    Step step=(Step)0;step.shape=float4(rs,rr,0,RayPi/angle);
    float4 phi=step.shape.w*EdgeAngles(ts,tr);uint term=0;
    for(uint i=0;i<4;i++)
    {
        if(TunedBoundaryTerm(phi[i]))continue;
        float4 mask=(float4)(uint4(0,1,2,3)==term);
        step.sine+=sin(phi[i])*mask;step.halfSine+=sin(.5*phi[i])*mask;term++;
    }
    return Kernel(step,zs,zr);
}
// Complete one/two-interaction R/D routes. The quadrature is shared with the
// tuned wave benchmark; finite edges retain their actual support and visibility.
struct CompleteRoute { uint first,second,family; float weight; };
RWStructuredBuffer<CompleteRoute> CompleteCandidates;
RWStructuredBuffer<Step> CompleteSteps;
RWStructuredBuffer<uint> CompleteSelected,CompleteCount;
StructuredBuffer<float4> CompleteDraws;
uint CompleteSamples,CompleteCapacity,CompleteFrame,CompleteSite,CompleteOffset;
float4 CompleteSizes()
{
    return DiffractionFactor<=0?float4(0,0,0,0):float4(EdgeCount,MaxBounces>1?(float)EdgeCount*EdgeCount:0,
        MaxBounces>1?(float)PrimitiveCount*EdgeCount:0,MaxBounces>1?(float)PrimitiveCount*EdgeCount:0);
}
float CompleteMirrorPairs(){return MaxBounces>1?(float)PrimitiveCount*PrimitiveCount:0;}
float3 CompleteSource(bool planar)
{
    float3 s=EmissionPosition(CompleteFrame*ObjectCount,RayTracePair/ListenerCount,CompleteSite);
    if(planar)s.z=0;return s;
}
float3 CompleteListener(bool planar)
{
    float3 p=States[CompleteFrame*ObjectCount+Listeners[RayTracePair%ListenerCount]].position.xyz;
    if(planar)p.z=0;return p;
}
bool CompleteEnabled()
{
    uint emission=CompleteFrame*EmissionFrameStride+RayTracePair/ListenerCount;
    uint4 range=EmissionRanges[emission];
    return CompleteSite<range.y && RayPairEnabled[CompleteFrame*PairCount+RayTracePair]!=0 && EmissionPoints[range.x+CompleteSite].w!=0;
}
float3 CompleteAnchor(uint id,uint state,bool planar)
{
    float3 a,u,f,n;float extent,angle;EdgeFrame(id,state,planar,a,u,f,n,extent,angle);
    if(planar)a.z=0;return a+(planar?0:.5*extent)*u;
}
float3 CompleteImage(uint primitive,uint state,float3 location,bool planar,out float3 normal)
{
    Primitive p=Primitives[primitive];State pose=States[state+p.indices.x];
    float3 a=pose.position.xyz+Rotate(pose.rotation,p.a.xyz),b=Rotate(pose.rotation,p.b.xyz-p.a.xyz);
    normal=planar?normalize(float3(-b.y,b.x,0)):normalize(cross(b,Rotate(pose.rotation,p.c.xyz-p.a.xyz)));
    return location-2*dot(location-a,normal)*normal;
}
bool CompleteMirror(uint primitive,uint state,float3 start,float3 end,bool planar,out float3 hit,out float cosine)
{
    hit=0;cosine=0;float3 n;float3 image=CompleteImage(primitive,state,start,planar,n),delta=end-image;
    float lengthTo=length(delta);if(lengthTo<=RayEpsilon)return false;
    float3 direction=delta/lengthTo,unused;float distance=PrimitiveDistance(primitive,state,image,direction,planar,unused);
    if(distance<=RayEpsilon || distance>=lengthTo-RayEpsilon)return false;
    hit=image+distance*direction;
    // Each shared-triangle contact belongs to a single primitive.
    for(uint i=0;i<primitive;i++)if(Primitives[i].indices.x==Primitives[primitive].indices.x)
        if(abs(PrimitiveDistance(i,state,image,direction,planar,unused)-distance)<RayEpsilon)return false;
    if(!Visible(state,start,hit,planar) || !Visible(state,hit,end,planar))return false;
    cosine=abs(dot(normalize(hit-start),n));return true;
}
Step CompleteStep(uint edge,uint state,float3 source,float3 target,bool planar,float remaining)
{
    float3 a,u,f,n;float extent,angle;EdgeFrame(edge,state,planar,a,u,f,n,extent,angle);
    float3 s=source-a,r=target-a;Step result=(Step)0;
    result.shape=float4(length(s-dot(s,u)*u),length(r-dot(r,u)*u),remaining,RayPi/angle);
    float ts=TunedEdgeAngle(s,f,n,angle),tr=TunedEdgeAngle(r,f,n,angle);if(max(ts,tr)>angle)return result;
    float4 theta=EdgeAngles(ts,tr);uint count=0;
    for(uint i=0;i<4;i++)
    {
        // Resolve angular boundaries at operand precision, as in MakeStep.
        float phase=result.shape.w*theta[i];
        if(TunedBoundaryTerm(phase))continue;
        float4 mask=(float4)(uint4(0,1,2,3)==count);
        result.sine+=sin(phase)*mask;result.halfSine+=sin(.5*phase)*mask;count++;
    }
    return result;
}
bool CompleteExterior(uint edge,uint state,float3 location,bool planar)
{
    float3 a,u,f,n;float extent,angle;EdgeFrame(edge,state,planar,a,u,f,n,extent,angle);
    return TunedEdgeAngle(location-a,f,n,angle)<=angle;
}
bool CompleteDoubleMirror(uint first,uint second,uint state,float3 source,float3 end,bool planar,
                          out float3 a,out float3 b,out float cosineA,out float cosineB)
{
    a=0;b=0;cosineA=0;cosineB=0;if(first==second)return false;
    if((States[state+Primitives[first].indices.x].flags.x&3)!=3 || (States[state+Primitives[second].indices.x].flags.x&3)!=3)return false;
    float3 normal,image=CompleteImage(first,state,source,planar,normal);
    image=CompleteImage(second,state,image,planar,normal);float3 delta=end-image;
    float distance=length(delta);if(distance<=RayEpsilon)return false;float3 unused,direction=delta/distance;
    float t=PrimitiveDistance(second,state,image,direction,planar,unused);
    if(t<=RayEpsilon || t>=distance-RayEpsilon)return false;b=image+t*direction;
    for(uint i=0;i<second;i++)if(Primitives[i].indices.x==Primitives[second].indices.x)
        if(abs(PrimitiveDistance(i,state,image,direction,planar,unused)-t)<RayEpsilon)return false;
    if(!CompleteMirror(first,state,source,b,planar,a,cosineA) || !Visible(state,b,end,planar))return false;
    cosineB=abs(dot(normalize(b-a),normal));return true;
}
void PrepareCompleteRoutes(uint index,bool planar)
{
    if(index>=CompleteCapacity)return;
    CompleteRoute route=(CompleteRoute)0;CompleteCandidates[index]=route;
    if(!CompleteEnabled())return;
    float4 sizes=CompleteSizes();float total=dot(sizes,1)+CompleteMirrorPairs();if(total==0)return;
    uint family=0,a=0,b=0;float weight=1;
    if(total<=CompleteCapacity)
    {
        if(index>=(uint)total)return;uint rank=index;
        for(uint j=0;j<4;j++){if(rank<(uint)sizes[j])break;rank-=(uint)sizes[j];family++;}
        uint columns=family==0?1:family>=3?PrimitiveCount:EdgeCount;a=rank/columns;b=rank%columns;
    }
    else
    {
        uint emission=CompleteFrame*EmissionFrameStride+RayTracePair/ListenerCount;
        uint4 range=EmissionRanges[emission],clock=EmissionClocks[range.x+CompleteSite];
        uint birth=range.w==2?HashRay(clock.y)^HashRay(clock.z):0;
        uint rng=HashRay(index^RaySeed^birth^HashRay(RayTracePair)^0x51ed270bu);
        // Give each nonempty interaction family support at small budgets. A
        // quadratic number of RR candidates must not starve single diffraction.
        uint families=(sizes.x>0)+(sizes.y>0)+(sizes.z>0)+(sizes.w>0)+(CompleteMirrorPairs()>0);
        uint rank=min(families-1,(uint)(((index+RandomRay(rng))/CompleteCapacity)*families));
        for(uint j=0;j<4;j++)
        {if(sizes[j]>0){if(rank==0)break;rank--;}family++;}
        uint rows=(family==2 || family==4)?PrimitiveCount:EdgeCount,cols=family==0?1:family>=3?PrimitiveCount:EdgeCount;
        a=min(rows-1,(uint)(RandomRay(rng)*rows));b=min(cols-1,(uint)(RandomRay(rng)*cols));
        weight=(family<4?sizes[family]:CompleteMirrorPairs())*families/CompleteCapacity;
    }
    uint state=CompleteFrame*ObjectCount;float3 source=CompleteSource(planar),end=CompleteListener(planar);
    if(MediumAt(state,source,planar)!=AirMedium || MediumAt(state,end,planar)!=AirMedium)return;
    if(family==4)
    {
        float3 x,y;float c0,c1;
        if(!CompleteDoubleMirror(a,b,state,source,end,planar,x,y,c0,c1))return;
        if(planar)
        {Step cached=(Step)0;cached.shape=float4(x.xy,y.xy);cached.sine=float4(c0,c1,length(source-x)+length(x-y)+length(y-end),0);CompleteSteps[2*index]=cached;CompleteSteps[2*index+1]=(Step)0;}
        route.first=a;route.second=b;route.family=family;route.weight=weight;CompleteCandidates[index]=route;return;
    }
    uint first=family==2?b:a,last=family==1?b:first;
    if((States[state+Edges[first].ids.x].flags.x&3)!=3 || (States[state+Edges[last].ids.x].flags.x&3)!=3)return;
    if(family==1 && a==b)return;
    float3 from=source,to=end,unused;
    if(family==2 || family==3)
    {
        uint mirror=family==2?a:b;if((States[state+Primitives[mirror].indices.x].flags.x&3)!=3)return;
        if(family==2)from=CompleteImage(mirror,state,source,planar,unused);
        else to=CompleteImage(mirror,state,end,planar,unused);
    }
    if(!CompleteExterior(first,state,from,planar) || !CompleteExterior(last,state,to,planar))return;
    if(planar)
    {
        float3 x=CompleteAnchor(first,state,true),y=CompleteAnchor(last,state,true);float3 hit;float cosine;
        if(family==2){if(!CompleteMirror(a,state,source,x,true,hit,cosine))return;}
        else if(!Visible(state,source,x,true))return;
        if(family==3){if(!CompleteMirror(b,state,y,end,true,hit,cosine))return;}
        else if(!Visible(state,y,end,true))return;
        if(family==1 && !Visible(state,x,y,true))return;
        Step firstStep=CompleteStep(first,state,from,family==1?y:to,true,length(y-x)+length(to-y));
        Step secondStep=(Step)0;
        if(family==1)secondStep=CompleteStep(last,state,x,to,true,length(to-y));
        else if(family==2 || family==3)
        {
            // The unused second step holds the planar mirror geometry.
            if(family==2)CompleteMirror(a,state,source,x,true,hit,cosine);
            else CompleteMirror(b,state,y,end,true,hit,cosine);
            float fraction=family==2?length(source-hit)/firstStep.shape.x:length(end-hit)/firstStep.shape.y;
            secondStep.shape=float4(hit.xy,cosine,fraction);
        }
        if(firstStep.shape.x+firstStep.shape.z>=MaximumPathDelay*SpeedOfSound || Terms(firstStep)==0 || (family==1 && Terms(secondStep)==0))return;
        CompleteSteps[2*index]=firstStep;CompleteSteps[2*index+1]=secondStep;
    }
    route.first=a;route.second=b;route.family=family;route.weight=weight;CompleteCandidates[index]=route;
}
void CompactCompleteRoutes()
{
    uint count=0;for(uint i=0;i<CompleteCapacity;i++)if(CompleteCandidates[i].weight!=0)CompleteSelected[count++]=i;
    CompleteCount[0]=count;
}
// A finite-support extension of the same Cauchy mixture. The uniform component
// retains support for nonparallel edges, whose angular terms change with height.
float CompleteFiniteHeight(uint edge,uint previous,uint state,float3 before,float3 source,float3 next,float3 end,
                           float remaining,float2 draw,out float density)
{
    float3 a,u,f,n;float extent,angle;EdgeFrame(edge,state,false,a,u,f,n,extent,angle);
    Step step=CompleteStep(edge,state,source,next,false,remaining);
    float zs=dot(source-a,u),zr=dot(end-a,u),rs=step.shape.x,rr=max(remaining,RayEpsilon),nu=step.shape.w;
    float center=(zs*rr+zr*rs)/(rs+rr),m=length(float2(rs,zs-center)),l=length(float2(rr,zr-center));
    float centers[13],scales[13],spans[13];uint count=1;
    for(uint i=0;i<13;i++){centers[i]=0;scales[i]=0;spans[i]=0;}
    for(uint i=0;i<Terms(step);i++)
    {centers[count]=center;scales[count++]=max(1e-8,2*abs(step.halfSine[i])*(m*l/(m+l))/nu);}
    if(previous!=0xffffffff)
    {
        float3 pa,pu,pf,pn;float pe,pangle;EdgeFrame(previous,state,false,pa,pu,pf,pn,pe,pangle);
        if(abs(dot(u,pu))>.99999)
        {
            // In a common parallel-edge frame this is exactly SampleHeight's
            // incoming proposal, including all four preceding angular terms.
            float radial=length((a-source)-dot(a-source,u)*u);
            float priorRadius=length((before-source)-dot(before-source,u)*u);
            float incoming=zs-dot(before-source,u)*radial/max(priorRadius,RayEpsilon);
            Step prior=CompleteStep(previous,state,before,a+incoming*u,false,radial);
            for(uint i=0;i<Terms(prior);i++)
            {centers[count]=incoming;scales[count++]=max(1e-8,2*abs(prior.halfSine[i])*length(float2(radial,incoming-zs))/prior.shape.w);}
        }
        else
        {
            // On skew edges the preceding eta=0 locus is a cone. Its
            // intersections generalize the parallel incoming Cauchy peak.
            float3 incoming=before-source,d=a-source;
            float axial=dot(incoming,pu),length2=dot(incoming,incoming),c=dot(u,pu),d0=dot(d,pu);
            float qa=length2*c*c-axial*axial,qb=2*(length2*d0*c-axial*axial*dot(d,u));
            float qc=length2*d0*d0-axial*axial*dot(d,d);float2 roots=0;uint rootsCount=0;
            if(abs(qa)<1e-8*max(length2,1e-8))
            {if(abs(qb)>1e-12){roots.x=-qc/qb;rootsCount=1;}}
            else
            {
                float discriminant=qb*qb-4*qa*qc;
                if(discriminant>=0)
                {
                    float q=-.5*(qb+(qb<0?-1:1)*sqrt(discriminant));
                    if(q!=0){roots=float2(q/qa,qc/q);rootsCount=2;}
                    else {roots.x=-qb/(2*qa);rootsCount=1;}
                }
            }
            [unroll] for(uint root=0;root<2;root++)
            {
                if(root>=rootsCount)break;float h=roots[root];float3 outgoing=d+h*u;
                float z=dot(outgoing,pu),radius2=dot(outgoing,outgoing)-z*z;
                if(z*axial>RayEpsilon || radius2<=RayEpsilon*RayEpsilon)continue;
                float derivative=abs((c*dot(outgoing,outgoing)-z*dot(outgoing,u))/(radius2*length(outgoing)));
                if(derivative<=1e-12)continue;
                Step prior=CompleteStep(previous,state,before,a+h*u,false,remaining);
                for(uint i=0;i<Terms(prior);i++)
                {centers[count]=h;scales[count++]=max(1e-8,2*abs(prior.halfSine[i])/(prior.shape.w*derivative));}
            }
        }
    }
    for(uint i=1;i<count;i++)spans[i]=Span(extent*scales[i],scales[i]*scales[i]+centers[i]*(centers[i]-extent));
    uint component=min(count-1,(uint)(draw.x*count));float height=draw.y*extent;
    if(component>0)
    {
        height=EdgeHeight(centers[component],scales[component],spans[component],extent,draw.y,false);
    }
    density=1/extent;
    for(uint i=1;i<count;i++){float d=height-centers[i];density+=scales[i]/((scales[i]*scales[i]+d*d)*spans[i]);}
    density/=count;return height;
}
float3 CompletePoint(uint edge,uint previous,uint state,bool planar,Step step,Step prior,
                     float3 before,float3 source,float3 next,float3 end,float previousHeight,
                     float travelled,float2 draw,out float density)
{
    float3 a,u,f,n;float extent,angle;EdgeFrame(edge,state,planar,a,u,f,n,extent,angle);
    float height;
    if(planar)
    {
        a.z=0;
        height=SampleHeight(step,prior,previous==0xffffffff,previousHeight,0,travelled,
                            MaximumPathDelay*SpeedOfSound,1,draw,density);
    }
    else height=CompleteFiniteHeight(edge,previous,state,before,source,next,end,step.shape.z,draw,density);
    return a+height*u;
}
float CompleteVelocityRatio(uint owner,uint state,float3 before,float3 vertex,float3 after,bool planar)
{
    float3 v=PointVelocity(state+owner,vertex);if(planar)v.z=0;
    return (SpeedOfSound-dot(v,normalize(vertex-before)))/(SpeedOfSound-dot(v,normalize(after-vertex)));
}
void DebugComplete(uint index,uint family,uint first,uint second,float3 source,float3 a,float3 b,float3 end,RayPath path)
{
#if defined(ACOUSTIC_DEBUG)
    if(!DebugSelected(CompleteFrame*PairCount+RayTracePair,path.frequencyBin) || (path.pressureGain==0 && path.pressureImag==0))return;
    uint slot=PrimitiveCount+1+RaysPerBin*(2+2*MaxBounces)+2*MirrorCandidateCount+3*index;
    uint identity=2*max(RaysPerBin,MirrorCandidateCount)+2+index;
    float t=length(a-source)/SpeedOfSound,pressure=path.pressureImag==0?path.pressureGain:length(float2(path.pressureGain,path.pressureImag));
    bool firstMirror=family==2 || family==4;
    int material=firstMirror?Primitives[first].indices.y:Primitives[Edges[first].ids.y].indices.y;
    DebugWrite(slot,identity,firstMirror?DebugReflection:DebugDiffraction,material,source,a,0,t,pressure);
    if(family==0){DebugArrival(slot+1,identity,a,end,t,path);return;}
    float next=t+length(b-a)/SpeedOfSound;bool secondMirror=family>=3;
    material=secondMirror?Primitives[second].indices.y:Primitives[Edges[second].ids.y].indices.y;
    DebugWrite(slot+1,identity,secondMirror?DebugReflection:DebugDiffraction,material,a,b,t,next,pressure);
    DebugArrival(slot+2,identity,b,end,next,path);
#endif
}
uint CompleteIdentity(CompleteRoute route,uint draw)
{return HashRay(0xc0000000u^HashRay(route.first)^HashRay(route.second+0x9e3779b9u)^HashRay(route.family)^draw);}
void CompleteInvalid(uint slot)
{
    for(uint bin=0;bin<FrequencyBins;bin++)
    {RayPath p=(RayPath)0;p.pressureGain=asfloat(0x7fc00000);p.frequencyBin=bin;RayPaths[slot+bin]=p;}
}
void IntegrateCompletePlanar(CompleteRoute route,uint selected,uint index,uint samples,uint draw,uint slot)
{
    uint state=CompleteFrame*ObjectCount,emission=CompleteFrame*EmissionFrameStride+RayTracePair/ListenerCount;
    float3 source=CompleteSource(true),end=CompleteListener(true);uint family=route.family;
    Step firstStep=CompleteSteps[2*selected],secondStep=CompleteSteps[2*selected+1];
    float3 a=0,b=0,arrival=0;float distance,gain,ratio=1;
    uint first=family==2?route.second:route.first,last=family==1?route.second:first,depth=family==1?2:1;
    uint identity=CompleteIdentity(route,family==4?0:draw);
    float3 sv=EmissionRanges[emission].w==2?float3(0,0,0):PointVelocity(state+Sources[RayTracePair/ListenerCount],source);
    float3 lv=Velocity(state+Listeners[RayTracePair%ListenerCount]);sv.z=0;lv.z=0;
    if(family==4)
    {
        a=float3(firstStep.shape.xy,0);b=float3(firstStep.shape.zw,0);arrival=b;
        distance=firstStep.sine.z;gain=route.weight/(samples*distance);
        ratio=CompleteVelocityRatio(Primitives[route.first].indices.x,state,source,a,b,true)*CompleteVelocityRatio(Primitives[route.second].indices.x,state,a,b,end,true);
    }
    else
    {
        bool valid;float4 value=IntegrateEdgePair(firstStep,secondStep,depth,CompleteDraws[draw],MaximumPathDelay*SpeedOfSound,1,route.weight/samples,valid);
        if(!valid){CompleteInvalid(slot);return;}
        distance=value.x;gain=value.y;
        a=CompleteAnchor(first,state,true);a.z=value.w;b=CompleteAnchor(last,state,true);b.z=value.z;
        if(family==2)
        {b=a;a=float3(secondStep.shape.xy,secondStep.shape.w*value.w);ratio=CompleteVelocityRatio(Primitives[route.first].indices.x,state,source,a,b,true)*CompleteVelocityRatio(Edges[last].ids.x,state,a,b,end,true);}
        else if(family==3)
        {b=float3(secondStep.shape.xy,secondStep.shape.w*value.z);ratio=CompleteVelocityRatio(Edges[first].ids.x,state,source,a,b,true)*CompleteVelocityRatio(Primitives[route.second].indices.x,state,a,b,end,true);}
        else
        {ratio=CompleteVelocityRatio(Edges[first].ids.x,state,source,a,depth==2?b:end,true);if(depth==2)ratio*=CompleteVelocityRatio(Edges[last].ids.x,state,a,b,end,true);}
        arrival=family==0?a:b;
        if(depth==2)gain*=EdgeConnectionWeight(first,last);
    }
    ratio*=(SpeedOfSound-dot(lv,normalize(end-arrival)))/(SpeedOfSound-dot(sv,normalize(a-source)));
    for(uint bin=0;bin<FrequencyBins;bin++)
    {
        float weight=exp(-AirCoefficient(max(1,bin*FrequencyStep))*distance);float2 material=float2(1,0);
        if(family==4)material=ComplexProduct(ReflectionPressure(RayMaterials[Primitives[route.first].indices.y*FrequencyBins+bin],firstStep.sine.x),ReflectionPressure(RayMaterials[Primitives[route.second].indices.y*FrequencyBins+bin],firstStep.sine.y));
        else
        {
            weight*=pow(DiffractionFactor,depth)*EdgeOpacity(first,bin)*(depth==2?EdgeOpacity(last,bin):1);
            if(family==2 || family==3)material=ReflectionPressure(RayMaterials[Primitives[family==2?route.first:route.second].indices.y*FrequencyBins+bin],secondStep.shape.z);
        }
        RayPath p=ApplyMaterial(Arrival(arrival,end,distance/SpeedOfSound,gain*weight*ReferenceDistance,bin,ratio,identity),material);
        RayPaths[slot+bin]=ApplyEmission(p,emission,CompleteSite,0);
        DebugComplete(index,family,route.first,route.second,source,a,b,end,RayPaths[slot+bin]);
        if(family!=4){p.direction.z=-p.direction.z;p.identity=HashRay(identity^0x73a91u);RayPaths[slot+FrequencyBins+bin]=ApplyEmission(p,emission,CompleteSite,0);}
    }
}
void IntegrateCompleteRoutes(uint index,bool planar)
{
    if(index>=CompleteSamples)return;
    uint state=CompleteFrame*ObjectCount,emission=CompleteFrame*EmissionFrameStride+RayTracePair/ListenerCount;
    uint symmetry=planar?2:1,slot=CompleteOffset+index*symmetry*FrequencyBins;
    for(uint bin=0;bin<symmetry*FrequencyBins;bin++)RayPaths[slot+bin]=(RayPath)0;
    uint count=CompleteCount[0];if(count==0 || !CompleteEnabled())return;
    uint per=CompleteSamples/count,extra=CompleteSamples%count,split=(per+1)*extra;
    uint rank=index<split?index/(per+1):extra+(index-split)/per;
    uint begin=rank*per+min(rank,extra),samples=per+(rank<extra),draw=index-begin;
    uint selected=CompleteSelected[rank];CompleteRoute route=CompleteCandidates[selected];
    if(planar){IntegrateCompletePlanar(route,selected,index,samples,draw,slot);return;}
    if(route.family==4)
    {
        float3 source=CompleteSource(planar),end=CompleteListener(planar),a,b;float ca,cb;
        if(!CompleteDoubleMirror(route.first,route.second,state,source,end,planar,a,b,ca,cb))return;
        float distance=length(source-a)+length(a-b)+length(b-end),gain=route.weight/(samples*distance);
        uint identity=HashRay(0xd0000000u^HashRay(route.first)^route.second);
        float3 incoming=normalize(a-source),between=normalize(b-a),outgoing=normalize(end-b);
        float3 sv=EmissionRanges[emission].w==2?float3(0,0,0):PointVelocity(state+Sources[RayTracePair/ListenerCount],source);
        float3 va=PointVelocity(state+Primitives[route.first].indices.x,a),vb=PointVelocity(state+Primitives[route.second].indices.x,b);
        float ratio=(SpeedOfSound-dot(Velocity(state+Listeners[RayTracePair%ListenerCount]),outgoing))/(SpeedOfSound-dot(sv,incoming));
        ratio*=(SpeedOfSound-dot(va,incoming))/(SpeedOfSound-dot(va,between));
        ratio*=(SpeedOfSound-dot(vb,between))/(SpeedOfSound-dot(vb,outgoing));
        for(uint bin=0;bin<FrequencyBins;bin++)
        {
            float2 coefficient=ComplexProduct(ReflectionPressure(RayMaterials[Primitives[route.first].indices.y*FrequencyBins+bin],ca),ReflectionPressure(RayMaterials[Primitives[route.second].indices.y*FrequencyBins+bin],cb));
            RayPath p=ApplyMaterial(Arrival(b,end,distance/SpeedOfSound,gain*ReferenceDistance*exp(-AirCoefficient(max(1,bin*FrequencyStep))*distance),bin,ratio,identity),coefficient);
            RayPaths[slot+bin]=ApplyEmission(p,emission,CompleteSite,0);
            DebugComplete(index,4,route.first,route.second,source,a,b,end,RayPaths[slot+bin]);
        }
        return;
    }
    uint family=route.family,first=family==2?route.second:route.first,last=family==1?route.second:first;
    uint depth=family==1?2:1;float3 source=CompleteSource(planar),end=CompleteListener(planar),from=source,to=end,unused;
    if(family==2)from=CompleteImage(route.first,state,source,planar,unused);
    if(family==3)to=CompleteImage(route.second,state,end,planar,unused);
    float3 p0=CompleteAnchor(first,state,planar),p1=CompleteAnchor(last,state,planar);
    Step s0=CompleteStep(first,state,from,depth==2?p1:to,planar,length(p1-p0)+length(to-p1));
    Step s1=CompleteStep(last,state,p0,to,planar,length(to-p1));
    if(planar && (s0.shape.x+s0.shape.z>=MaximumPathDelay*SpeedOfSound || Terms(s0)==0 || (depth==2 && Terms(s1)==0)))return;
    float density0,density1=1;
    float3 pointA=CompletePoint(first,0xffffffff,state,planar,s0,(Step)0,from,from,depth==2?p1:to,to,0,0,CompleteDraws[draw].xy,density0);
    if(!(density0>0) || !all(isfinite(pointA)))return;
    float travelled=length(pointA-from);float3 pointB=pointA;
    if(depth==2)
    {
        pointB=CompletePoint(last,first,state,planar,s1,s0,from,pointA,to,to,pointA.z,travelled,CompleteDraws[draw].zw,density1);
        if(!(density1>0) || !all(isfinite(pointB)))return;
        travelled+=length(pointB-pointA);
    }
    float gain=route.weight/(samples*density0*density1);
    float3 mirror=0;float cosine=0;
    if(family==2){if(!CompleteMirror(route.first,state,source,pointA,planar,mirror,cosine))return;}
    else if(!Visible(state,source,pointA,planar))return;
    if(depth==2 && !Visible(state,pointA,pointB,planar))return;
    if(family==3){if(!CompleteMirror(route.second,state,pointB,end,planar,mirror,cosine))return;}
    else if(!Visible(state,pointB,end,planar))return;
    float distance0=length(pointA-from),distance1=depth==2?length(pointB-pointA):length(to-pointA);
    float finalDistance=length(to-pointB);if(min(min(distance0,distance1),finalDistance)<=RayEpsilon)return;
    if(depth==2)gain*=EdgeConnectionWeight(first,last)*TunedEdgeKernel(first,state,from,pointA,pointB,planar)/distance0;
    gain*=TunedEdgeKernel(last,state,depth==2?pointA:from,pointB,to,planar)/(distance1*(depth==2?finalDistance:distance0));
    float distance=travelled+finalDistance;
    uint identity=CompleteIdentity(route,draw),owner=family==3?Primitives[route.second].indices.x:Edges[last].ids.x;
    float3 arrival=family==3?mirror:pointB;
    float3 sourceVelocity=EmissionRanges[emission].w==2?float3(0,0,0):PointVelocity(state+Sources[RayTracePair/ListenerCount],source);
    float3 firstPoint=family==2?mirror:pointA;float3 firstDirection=normalize(firstPoint-source),lastDirection=normalize(end-arrival);
    float ratio=(SpeedOfSound-dot(Velocity(state+Listeners[RayTracePair%ListenerCount]),lastDirection))/(SpeedOfSound-dot(sourceVelocity,firstDirection));
    // Boundaries use the existing frozen-pose convention. Include their velocity
    // factors at every physical interaction, including a mixed-path mirror.
    ratio*=CompleteVelocityRatio(Edges[first].ids.x,state,family==2?mirror:source,pointA,depth==2?pointB:(family==3?mirror:end),planar);
    if(depth==2)ratio*=CompleteVelocityRatio(Edges[last].ids.x,state,pointA,pointB,family==3?mirror:end,planar);
    if(family==2 || family==3)
    {
        uint p=family==2?route.first:route.second;float3 v=PointVelocity(state+Primitives[p].indices.x,mirror);
        float3 incoming=normalize(mirror-(family==2?source:pointA)),outgoing=normalize((family==2?pointA:end)-mirror);
        ratio*=(SpeedOfSound-dot(v,incoming))/(SpeedOfSound-dot(v,outgoing));
    }
    for(uint bin=0;bin<FrequencyBins;bin++)
    {
        float weight=pow(DiffractionFactor,depth)*EdgeOpacity(first,bin)*(depth==2?EdgeOpacity(last,bin):1);
        weight*=exp(-AirCoefficient(max(1,bin*FrequencyStep))*distance);
        RayPath p=Arrival(arrival,end,distance/SpeedOfSound,gain*weight*ReferenceDistance,bin,ratio,identity);
        if(family==2 || family==3)p=ApplyMaterial(p,ReflectionPressure(RayMaterials[Primitives[family==2?route.first:route.second].indices.y*FrequencyBins+bin],cosine));
        RayPaths[slot+bin]=ApplyEmission(p,emission,CompleteSite,0);
        DebugComplete(index,family,route.first,route.second,source,family==2?mirror:pointA,family==3?mirror:pointB,end,RayPaths[slot+bin]);
        if(planar)
        {p.direction.z=-p.direction.z;p.identity=HashRay(identity^0x73a91u);RayPaths[slot+FrequencyBins+bin]=ApplyEmission(p,emission,CompleteSite,0);}
    }
}
