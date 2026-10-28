// Shared with the wave-validated complete-route shader (21–22 October 2026).
// Keep signed pressure, the incoming/outgoing Cauchy mixture and its exact PDF.
#ifndef ACOUSTIC_DIFFRACTION_INTEGRAL
#define ACOUSTIC_DIFFRACTION_INTEGRAL
static const float Pi = 3.14159265358979323846;
struct Step { float4 shape, sine, halfSine; };
float Span(float y, float x)
{
    // Two half-angle reductions put atan's argument in [0,tan(pi/8)].
    // The 19th-order series has <2e-9 rad truncation error for the full angle;
    // it avoids backend atan2 approximations amplified by long sampled paths.
    float t = y/(length(float2(x,y))+abs(x));
    t /= sqrt(1+t*t)+1;
    float squared=t*t, polynomial=-1.0/19;
    [unroll] for (int n=8;n>=1;n--)
    polynomial=(n%2==0?1.0:-1.0)/(2*n+1)+squared*polynomial;
    float angle=4*t*(1+squared*polynomial);
    return x<0?Pi-angle:angle;
}
uint Terms(Step s)
{
    uint count = 0;
    for (uint i = 0; i < 4; i++) count += s.halfSine[i] != 0;
    return count;
}
float Kernel(Step s, float zs, float zr)
{
    float rs = s.shape.x, rr = s.shape.y, nu = s.shape.w;
    float m = length(float2(rs, zs)), l = length(float2(rr, zr));
    // Stable equivalent of sinh(nu*(asinh(zs/rs)+asinh(zr/rr))/2).
    float x = abs(rs*zr + rr*zs) / sqrt(2*rs*rr*(m*l + rs*rr - zs*zr));
    float sh = x < .01 ? nu*x*(1+(nu*nu-1)*x*x/6) : sinh(nu*log(x+sqrt(1+x*x)));
    float beta = 0;
    for (uint i = 0; i < 4; i++)
    if (s.halfSine[i] != 0)
        beta += s.sine[i] / (2*(sh*sh+s.halfSine[i]*s.halfSine[i]));
    return -nu*beta/(4*Pi);
}
float SampleHeight(Step s,Step prior,bool first,float previous,float before,
           float travelled,float horizon,float proposalScale,float2 draw,out float density)
{
    float rs = s.shape.x, rr = s.shape.z, nu = s.shape.w;
    float budget = horizon-travelled, focal = length(float2(rs+rr,previous));
    float horizontal = (budget-previous)*(budget+previous);
    // Factored ellipse intersection avoids subtracting large quadratic terms
    // when the remaining physical path support becomes narrow.
    float middle = previous*(horizontal-(rs-rr)*(rs+rr))/(2*horizontal);
    float radius = budget*sqrt(max(0,(budget-focal)*(budget+focal)*
                   (horizontal-(rs-rr)*(rs-rr))))/(2*horizontal);
    float lo = first ? 0 : middle-radius, hi = middle+radius;
    float center = previous*rr/(rs+rr);
    float m = length(float2(rs,previous-center)), l = length(float2(rr,center));
    uint own = Terms(s), count = own+(first ? 0 : Terms(prior));
    float centers[8] = {0,0,0,0,0,0,0,0}, scales[8] = {0,0,0,0,0,0,0,0}, spans[8] = {0,0,0,0,0,0,0,0};
    for (uint j = 0; j < own; j++)
    {
        centers[j] = center;
        scales[j] = proposalScale*max(1e-8,2*abs(s.halfSine[j])*(m*l/(m+l))/nu);
    }
    if (!first)
    {
        float incoming = previous+(previous-before)*rs/prior.shape.x;
        for (uint incomingTerm = own; incomingTerm < count; incomingTerm++)
        {
        centers[incomingTerm] = incoming;
        scales[incomingTerm] = proposalScale*max(1e-8,2*abs(prior.halfSine[incomingTerm-own])*
            length(float2(rs,incoming-previous))/prior.shape.w);
        }
    }
    for (uint term = 0; term < count; term++)
        spans[term] = Span((hi-lo)*scales[term],scales[term]*scales[term]+(lo-centers[term])*(hi-centers[term]));
    uint component = min(count-1,(uint)(draw.x*count));
    // Rotate the nearer endpoint direction. Unlike interpolation of heights,
    // this avoids cancelling large opposite endpoints near a narrow peak.
    float u = draw.y;
    float v = (u < .5 ? lo : hi)-centers[component], scale = scales[component];
    float sn, cs; sincos((u < .5 ? u : u-1)*spans[component],sn,cs);
    float z = centers[component]+scale*(v*cs+scale*sn)/(scale*cs-v*sn), pdf = 0;
    for (uint mixture = 0; mixture < count; mixture++)
    {
        float delta = z-centers[mixture];
        pdf += scales[mixture]/((scales[mixture]*scales[mixture]+delta*delta)*spans[mixture]);
    }
    density=pdf/count;
    return z;
}
// The original complete-route integrator, before its final complex-phase sum.
// Returning geometry keeps the same quadrature usable by actual audio.
float4 IntegrateEdgePair(Step first,Step second,uint depth,float4 draw,float horizon,float proposalScale,
                         float weight,out bool valid)
{
    float previous=0,before=0,travelled=0,density=1,gain=1;Step prior=(Step)0;
    for(uint i=0;i<depth;i++)
    {
        Step step=first;if(i>0)step=second;float pdf;
        float z=SampleHeight(step,prior,i==0,previous,before,travelled,horizon,proposalScale,i==0?draw.xy:draw.zw,pdf);
        density*=pdf;travelled+=length(float2(step.shape.x,z-previous));
        if(i>0)gain*=Kernel(prior,before-previous,z-previous)/length(float2(prior.shape.x,before-previous));
        before=previous;previous=z;prior=step;
    }
    float last=length(float2(prior.shape.y,previous));travelled+=last;
    gain*=Kernel(prior,before-previous,-previous)/length(float2(prior.shape.x,before-previous))/last;
    float value=gain/density*weight;valid=isfinite(value)&&isfinite(density)&&density>0;
    return valid?float4(travelled,value,previous,depth==1?previous:before):float4(0,0,0,0);
}
#endif
