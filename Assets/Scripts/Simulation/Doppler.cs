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
    public static float Read(TakeRender.Signal signal,double sample,double ratio,TakeRender.DelayFilter filter,int radius)
    {
        if(!(ratio>0) || !double.IsFinite(ratio) || !double.IsFinite(sample))throw new InvalidOperationException("Invalid propagation clock.");
        // Linear is the existing low-cost approximation. Sinc also low-passes
        // when pitch-up downsamples source time; its lookahead is radius receiver samples.
        if(filter==TakeRender.DelayFilter.Linear)return signal.Read(sample);
        double cutoff=Math.Min(1,1/ratio),sum=0,value=0;
        int extent=checked((int)Math.Ceiling(radius/cutoff));long center=checked((long)Math.Floor(sample));
        for(int k=1-extent;k<=extent;k++)
        {
            double x=(center+k-sample)*cutoff;
            if(Math.Abs(x)>=radius)continue;
            double weight=Sinc(x)*Sinc(x/radius)*cutoff;
            value+=signal.Read(center+k)*weight;sum+=weight;
        }
        return (float)(value/sum);
    }
    static double Sinc(double x)=>Math.Abs(x)<1e-12?1:Math.Sin(Math.PI*x)/(Math.PI*x);
}
