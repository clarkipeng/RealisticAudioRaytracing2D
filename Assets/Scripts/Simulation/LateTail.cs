using System;

// Continuation of a late-field envelope beyond the simulated horizon
// (Settings.lateTailSeconds; GpuAudio.compute LateExtend mirrors it). The late field
// is synthesised as noise under a band envelope, the diffuse-field model of Polack
// (1993), whose envelope decays exponentially. The simulation stops at
// maximumDelaySeconds; a room whose reverberation outlasts it would be cut off
// while still audible. Per band, the decay per bin is fitted over the last two
// windows of the simulated envelope (both channels together, so the ears decay
// alike): r = (E2 / E1)^(1 / W), E1 and E2 the energies of the second-to-last and
// last window of W = max(2, simulated / 4) bins. Each channel continues the
// exponential that has its last window's mean (A r^(k - k0) summed over the window
// k0 .. k0 + W - 1 equals that window's energy, so steep decays are not biased by
// anchoring the mean at the window's centre). The last W / 2
// simulated bins blend linearly into the fit (a smooth join), and the last eighth
// of the continuation fades out with a raised cosine, so the kernel's end is never
// a cut. Envelopes that do not decay over the fit (r >= 1, or an empty window) are
// not continued. With a reference curve (the energy method's room energy: all rays,
// not only those near the listener) the decay is fitted on it instead, which is far
// steadier where few paths reach the listener late in the tail (the bass bands).
internal static class LateTail
{
    internal static int Window(int simulated) => Math.Max(2,simulated/4);
    internal static bool Applies(int simulated,int total) => total>simulated && simulated>=8;

    // energy[channel][bin] for bins < total; bins >= simulated are written (and the
    // join bins blended). Returns the fitted decay per bin, or 0 if not continued.
    internal static double Extend(double[][] energy,int simulated,int total,double[] reference=null)
    {
        if(!Applies(simulated,total))return 0;
        int window=Window(simulated),join=Math.Max(1,window/2),fade=Math.Max(1,(total-simulated)/8);
        double first=0,last=0;
        if(reference!=null)for(int k=0;k<window;k++){first+=reference[simulated-2*window+k];last+=reference[simulated-window+k];}
        else foreach(var e in energy)for(int k=0;k<window;k++){first+=e[simulated-2*window+k];last+=e[simulated-window+k];}
        if(!(first>0) || !(last>0) || last>=first)
        {
            foreach(var e in energy)for(int k=simulated;k<total;k++)e[k]=0;
            return 0;
        }
        double decay=Math.Pow(last/first,1.0/window);int start=simulated-window;
        double shape=decay>1-1e-9?1:(1-Math.Pow(decay,window))/(window*(1-decay)); // window mean of decay^(k - start)
        foreach(var e in energy)
        {
            double mean=0;for(int k=start;k<simulated;k++)mean+=e[k];mean/=window*shape;
            for(int k=simulated-join;k<simulated;k++)
            {
                double w=(k-(simulated-join)+1.0)/(join+1);
                e[k]+=(mean*Math.Pow(decay,k-start)-e[k])*w;
            }
            for(int k=simulated;k<total;k++)
            {
                int left=total-k;double f=left>fade?1:Math.Sin(.5*Math.PI*left/(fade+1.0));
                e[k]=mean*Math.Pow(decay,k-start)*f*f;
            }
        }
        return decay;
    }
}
