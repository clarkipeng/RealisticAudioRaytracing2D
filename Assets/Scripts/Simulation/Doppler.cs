using System;
using UnityEngine;

// Local constant-velocity acoustics in a stationary, nondispersive medium.
// Delay rate is seconds/second; source read speed is 1 - delayRate.
internal static class Doppler
{
    public static Vector3 Velocity(AcousticTake.State state) =>
        (state.flags&(AcousticTake.StateFlags.VelocityValid|AcousticTake.StateFlags.Cut))==AcousticTake.StateFlags.VelocityValid ? state.velocity:Vector3.zero;
    public static void Direct(Vector3 displacement,Vector3 sourceVelocity,Vector3 listenerVelocity,double speed,
        out float delay,out float delayRate,out Vector3 direction)
    {
        double a=speed*speed-sourceVelocity.sqrMagnitude,b=Vector3.Dot(displacement,sourceVelocity),d=displacement.sqrMagnitude;
        if(a<=0)throw new NotSupportedException("The propagation model requires subsonic source motion relative to its medium.");
        double seconds=d==0?0:(b+Math.Sqrt(b*b+a*d))/a;
        direction=seconds==0?Vector3.zero:(displacement+sourceVelocity*(float)seconds).normalized;
        double ratio=(speed-Vector3.Dot(listenerVelocity,direction))/(speed-Vector3.Dot(sourceVelocity,direction));
        if(!(ratio>0) || !double.IsFinite(ratio))throw new NotSupportedException("Noncausal Doppler path; supersonic receiver motion is unsupported.");
        delay=(float)seconds;delayRate=(float)(1-ratio);
    }
    // Interpolate one source clock, rather than mixing two independently pitched reads.
    // Endpoint continuation also supplies the short filter's overlap outside this interval.
    public static bool Continuous(double span,double left,double right,double leftRate,double rightRate) =>
        span>0 && 2-leftRate-rightRate<=3*(1-(right-left)/span);
    public static void Clock(double time,double span,double left,double right,double leftRate,double rightRate,out double delay,out double rate)
    {
        if(time<=0){delay=left+leftRate*time;rate=leftRate;return;}
        if(time>=span){delay=right+rightRate*(time-span);rate=rightRate;return;}
        double u=time/span,delta=right-left,a=3*delta-span*(2*leftRate+rightRate),b=-2*delta+span*(leftRate+rightRate);
        delay=left+time*leftRate+u*u*(a+u*b);rate=leftRate+u*(2*a+3*u*b)/span;
    }
    public static float Read(TakeRender.Signal signal,double sample,double ratio,TakeRender.DelayFilter filter,int radius,uint phase=0)
    {
        if(!(ratio>0) || !double.IsFinite(ratio) || !double.IsFinite(sample))throw new InvalidOperationException("Invalid propagation clock.");
        // Linear is the existing low-cost approximation. Sinc also low-passes
        // when pitch-up downsamples source time; its lookahead is radius receiver samples.
        if(filter==TakeRender.DelayFilter.Linear)return signal.Read(sample,phase);
        double cutoff=Math.Min(1,1/ratio),sum=0,value=0;
        int extent=checked((int)Math.Ceiling(radius/cutoff));long center=checked((long)Math.Floor(sample));
        for(int k=1-extent;k<=extent;k++)
        {
            double x=(center+k-sample)*cutoff;
            if(Math.Abs(x)>=radius)continue;
            double weight=Sinc(x)*Sinc(x/radius)*cutoff;
            value+=signal.Read(center+k,phase)*weight;sum+=weight;
        }
        return (float)(value/sum);
    }
    static double Sinc(double x)=>Math.Abs(x)<1e-12?1:Math.Sin(Math.PI*x)/(Math.PI*x);
}
