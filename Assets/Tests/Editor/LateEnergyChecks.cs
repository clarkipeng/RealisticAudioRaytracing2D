using System;
using System.Collections.Generic;
using System.IO;
using NUnit.Framework;
using UnityEngine;

// Late energy field: estimator controls, synthesis and CPU/GPU parity
// (docs/benchmarks/late-energy-2026-11-04 holds the measured ladders).
public partial class TakeChecks
{
    // A room of open sheets, one object per face, so its interior is air
    // (a closed mesh is a solid whose interior is its own medium).
    static void LateRoom(AcousticTake.Dimension dimension,Vector3 size,AcousticTake.MaterialDefinition material,Vector3 source,Vector3 listener,
        out AcousticTake.Scene scene,out AcousticTake.Primitive[] geometry,out AcousticTake.State[] states)
    {
        bool planar=dimension==AcousticTake.Dimension.Planar2D;int faces=planar?4:6;
        var objects=new List<AcousticTake.ObjectDefinition>{new(){id="source",name="Source",scale=Vector3.one},new(){id="listener",name="Listener",scale=Vector3.one}};
        var list=new List<AcousticTake.Primitive>();var h=size*.5f;
        for(int f=0;f<faces;f++)
        {
            int owner=objects.Count,start=list.Count;
            if(planar)
            {
                var corners=new[]{new Vector3(-h.x,-h.y,0),new Vector3(h.x,-h.y,0),new Vector3(h.x,h.y,0),new Vector3(-h.x,h.y,0)};
                list.Add(new AcousticTake.Primitive{objectIndex=owner,a=corners[f],b=corners[(f+1)%4]});
            }
            else
            {
                int axis=f/2;float sign=f%2==0?-1:1;var u=Vector3.zero;var v=Vector3.zero;var c=Vector3.zero;
                c[axis]=sign*h[axis];u[(axis+1)%3]=h[(axis+1)%3];v[(axis+2)%3]=h[(axis+2)%3];
                list.Add(new AcousticTake.Primitive{objectIndex=owner,a=c-u-v,b=c+u-v,c=c+u+v});
                list.Add(new AcousticTake.Primitive{objectIndex=owner,a=c-u-v,b=c+u+v,c=c-u+v});
            }
            objects.Add(new(){id="face-"+f,name="Face "+f,scale=Vector3.one,primitiveStart=start,primitiveCount=list.Count-start});
        }
        scene=new AcousticTake.Scene{dimension=dimension,objects=objects.ToArray(),sources=new[]{new AcousticTake.SourceDefinition{objectIndex=0,clipIndex=0}},
            listeners=new[]{1},clips=new[]{new AcousticTake.ClipDefinition{name="test"}},materials=new[]{material}};
        geometry=list.ToArray();states=new AcousticTake.State[objects.Count];
        for(int i=0;i<states.Length;i++)states[i]=new AcousticTake.State{rotation=Quaternion.identity,
            flags=AcousticTake.StateFlags.Active|(i==0?AcousticTake.StateFlags.Source:i==1?AcousticTake.StateFlags.Listener:AcousticTake.StateFlags.Geometry)};
        states[0].position=planar?new Vector3(source.x,source.y,0):source;states[1].position=planar?new Vector3(listener.x,listener.y,0):listener;
    }
    static AcousticTake.MaterialDefinition LateMaterial(float absorption,float scattering,float transmission=0) =>
        new AcousticTake.MaterialDefinition{name="late",frequencies=new[]{0f},absorption=new[]{absorption},scattering=new[]{scattering},transmission=new[]{transmission}};
    TakeRender.Settings LateSettings(int rate,float horizon)
    {
        var settings=SpectralSettings();settings.speedOfSound=343;settings.minimumDistance=.1f;settings.fftSize=64;settings.spectralHop=16;
        settings.responseStepSamples=64;settings.maximumDelaySeconds=horizon;settings.maxBounces=512;settings.useBvh=true;settings.airAbsorption=false;
        return settings;
    }
    // Band-summed (over directions) energy per bin of pair 0, frame 0.
    static double[] LateEnergy(TransportSolver solver,int band)
    {
        var late=solver.Late(0);var energy=new double[late.Transport.LateBins];
        for(int k=0;k<energy.Length;k++)for(int d=0;d<TakeRender.Settings.LateDirections;d++)energy[k]+=late.Energy(0,d,band,k);
        return energy;
    }
    // Analytic shoebox image lattice: visible images of order >= order, energy (1-a)^k / L^2.
    static double[] ImageLattice(Vector3 size,Vector3 source,Vector3 listener,double absorption,double binSeconds,int bins,int minimumOrder,int maximumOrder=int.MaxValue)
    {
        var energy=new double[bins];double horizon=bins*binSeconds*343;int reach=(int)Math.Ceiling(horizon/Math.Min(size.x,Math.Min(size.y,size.z)))+1;
        for(int i=-reach;i<=reach;i++)for(int j=-reach;j<=reach;j++)for(int k=-reach;k<=reach;k++)for(int mask=0;mask<8;mask++)
        {
            // Image coordinate per axis: 2*n*L +/- s; reflection count |2n| or |2n-1|.
            int[] n={i,j,k};var image=Vector3.zero;int order=0;
            for(int a=0;a<3;a++)
            {
                bool mirrored=(mask>>a&1)!=0;
                image[a]=2*n[a]*size[a]+(mirrored?-source[a]:source[a]);
                order+=mirrored?Math.Abs(2*n[a]-1):Math.Abs(2*n[a]);
            }
            if(order<minimumOrder || order>maximumOrder)continue;
            double distance=(image-listener).magnitude;int bin=(int)Math.Floor(distance/343/binSeconds);
            if(bin<bins)energy[bin]+=Math.Pow(1-absorption,order)/(distance*distance);
        }
        return energy;
    }

    // The specular late field and the early images partition the image lattice:
    // with no scattering, late energy is exactly the lattice beyond the early order.
    [TestCase(0),TestCase(256),Category("GPU"),Category("LateEnergy")]
    public void LateSpecularEnergyCompletesTheImageLattice(int complete)
    {
        var size=new Vector3(5,4,3);var source=new Vector3(-1.1f,.4f,.3f);var listener=new Vector3(1.3f,-.6f,-.2f);
        LateRoom(AcousticTake.Dimension.Spatial3D,size,LateMaterial(.1f,0),source,listener,out var scene,out var geometry,out var state);
        var settings=LateSettings(8000,.1f);settings.lateRays=65536;settings.lateReceiverRadius=.25f;
        settings.completeRouteSamples=complete;settings.diffractionFactor=0;
        // Room coordinates have the origin at the centre; the lattice uses a corner origin.
        var corner=size*.5f;
        using var solver=new TransportSolver(scene,geometry,settings,RequireGpu(),sampleRate:8000);solver.Solve(state,new TakeRender.DirectPath[1]);
        int early=settings.LateEarlyOrder;Assert.That(early,Is.EqualTo(complete>0?2:1));
        var late=LateEnergy(solver,0);double binSeconds=(double)settings.LatePartition/8000;
        var expected=ImageLattice(size,source+corner,listener+corner,.1,binSeconds,late.Length,early+1);
        var held=ImageLattice(size,source+corner,listener+corner,.1,binSeconds,late.Length,0,early);
        double sum=0,want=0,before=0;int firstLate=Array.FindIndex(expected,e=>e>0);
        for(int k=0;k<late.Length;k++){sum+=late[k];want+=expected[k];}
        // Nothing before the first late image (minus the sphere's timing smear) may be scored.
        double smear=2*settings.lateReceiverRadius/343;
        for(int k=0;k<late.Length && (k+1)*binSeconds<firstLate*binSeconds-smear;k++)before+=late[k];
        TestContext.WriteLine($"late {sum:G6}, lattice beyond order {early} {want:G6}, early-held {Sum(held):G6}, early-window late {before:G6}");
        Assert.That(before,Is.Zero,"Early-order specular paths must not be scored in the late field.");
        Assert.That(sum/want,Is.EqualTo(1).Within(.05),"Late specular energy equals the image lattice beyond the early order.");
    }
    static double Sum(double[] values){double s=0;foreach(var v in values)s+=v;return s;}

    // A fully diffusing shoebox decays at the Eyring rate and holds the diffuse
    // steady-state energy (Barron's revised theory); tolerances are generous here,
    // the benchmark reports the measured errors.
    [Test,Category("GPU"),Category("LateEnergy")]
    public void LateDiffuseRoomDecaysAtTheEyringRate()
    {
        var size=new Vector3(6,5,4);var source=new Vector3(-1.5f,.5f,.4f);var listener=new Vector3(1.4f,-.7f,-.3f);const float absorption=.2f;
        LateRoom(AcousticTake.Dimension.Spatial3D,size,LateMaterial(absorption,1),source,listener,out var scene,out var geometry,out var state);
        var settings=LateSettings(8000,1);settings.lateRays=8192;
        using var solver=new TransportSolver(scene,geometry,settings,RequireGpu(),sampleRate:8000);solver.Solve(state,new TakeRender.DirectPath[1]);
        var late=LateEnergy(solver,0);double binSeconds=(double)settings.LatePartition/8000;
        // Schroeder backward integral, T20 from -5 to -25 dB.
        var curve=new double[late.Length];double tail=0;for(int k=late.Length-1;k>=0;k--){tail+=late[k];curve[k]=tail;}
        double x=0,y=0,xx=0,xy=0;int n=0;
        for(int k=0;k<late.Length;k++)
        {
            double level=10*Math.Log10(curve[k]/curve[0]);if(level>-5 || level<-25)continue;
            double t=(k+.5)*binSeconds;x+=t;y+=level;xx+=t*t;xy+=t*level;n++;
        }
        double slope=(n*xy-x*y)/(n*xx-x*x),t60=-60/slope;
        double volume=size.x*size.y*size.z,surface=2*(size.x*size.y+size.y*size.z+size.z*size.x);
        double eyring=.161*volume/(-surface*Math.Log(1-absorption));
        TestContext.WriteLine($"T20-based T60 {t60:F3} s, Eyring {eyring:F3} s, total {Sum(late):G5}, 16pi/A' {16*Math.PI*(1-absorption)/(surface*absorption):G5}");
        Assert.That(t60/eyring,Is.EqualTo(1).Within(.15));
    }

    // Weights never grow: in a lossless closed room the late energy arriving in any
    // bin is bounded by the emitted energy spread over the room's crossing rate.
    // Here: a lossless diffuse room, whose late energy per second is constant.
    [Test,Category("GPU"),Category("LateEnergy")]
    public void LateLosslessRoomKeepsConstantEnergyFlow()
    {
        var size=new Vector3(4,3,2.5f);
        LateRoom(AcousticTake.Dimension.Spatial3D,size,LateMaterial(0,1),new Vector3(-1,.3f,.2f),new Vector3(1,-.4f,-.3f),out var scene,out var geometry,out var state);
        var settings=LateSettings(8000,.5f);settings.lateRays=4096;
        using var solver=new TransportSolver(scene,geometry,settings,RequireGpu(),sampleRate:8000);solver.Solve(state,new TakeRender.DirectPath[1]);
        var late=LateEnergy(solver,0);double binSeconds=(double)settings.LatePartition/8000;
        // Diffuse steady-state energy density gives 16*pi/A per unit absorption; for a
        // lossless room the energy arriving per second is c*S/(4V) * 4/S... = 4*pi*c/V.
        double perSecond=4*Math.PI*343/(size.x*size.y*size.z);
        double first=0,second=0;int half=late.Length/2,skip=(int)(.05/binSeconds);
        for(int k=skip;k<half;k++)first+=late[k];for(int k=half;k<late.Length;k++)second+=late[k];
        first/=(half-skip)*binSeconds;second/=(late.Length-half)*binSeconds;
        TestContext.WriteLine($"energy per second {first:G5} then {second:G5}, diffuse {perSecond:G5}");
        Assert.That(second/first,Is.EqualTo(1).Within(.1),"A lossless room neither gains nor loses late energy.");
        Assert.That(second/perSecond,Is.EqualTo(1).Within(.15));
    }

    // Synthesis: the late impulse response's energy per partition follows the
    // histogram (mono, all directions weighted one).
    [Test,Category("GPU"),Category("LateEnergy")]
    public void LateSynthesisFollowsTheHistogramEnvelope()
    {
        var size=new Vector3(6,5,4);
        LateRoom(AcousticTake.Dimension.Spatial3D,size,LateMaterial(.15f,.6f),new Vector3(-1.5f,.5f,.4f),new Vector3(1.4f,-.7f,-.3f),out var scene,out var geometry,out var state);
        var settings=LateSettings(8000,.8f);settings.lateRays=4096;settings.audioBackend=TakeRender.AudioBackend.CpuReference;
        using var solver=new TransportSolver(scene,geometry,settings,RequireGpu(),sampleRate:8000);solver.Solve(state,new TakeRender.DirectPath[1]);
        var impulse=new TakeRender.Signal(new[]{1f});impulse.SetActive(0,true);impulse.Append(new AcousticTake.VoiceEvent{gain=1});
        var late=new LateReverb(scene,8000,settings,new[]{impulse});late.SetResponse(0,true,solver.Late(0),state);
        int partition=settings.LatePartition,bins=solver.Late(0).Transport.LateBins,latency=settings.DspLatencySamples;
        var output=new float[(bins+2)*partition+latency];late.Render(0,output.Length,output);
        // Expected energy of a partition: the noise is white with unit variance and its
        // spectrum is shaped by A(f) = sum_b sqrt(E_b) w_b(f), so E = mean over f of A^2.
        var band=new double[TakeRender.Settings.LateBands];double worst=0;
        for(int group=2;group+8<=bins;group+=8)
        {
            double want=0,have=0;
            for(int k=group;k<group+8;k++)
            {
                for(int b=0;b<band.Length;b++){band[b]=0;for(int d=0;d<TakeRender.Settings.LateDirections;d++)band[b]+=solver.Late(0).Energy(0,d,b,k);}
                double mean=0;for(int f=0;f<=4000;f++){double a=0;for(int b=0;b<band.Length;b++)a+=Math.Sqrt(band[b])*LateBasis.EnvelopeWeight(b,f);a=Math.Max(0,a);mean+=a*a;}
                want+=mean/4001;
                for(int n=0;n<partition;n++){double v=output[latency+k*partition+n];have+=v*v;}
            }
            if(want>1e-9)worst=Math.Max(worst,Math.Abs(10*Math.Log10(have/want)));
        }
        TestContext.WriteLine($"largest group level error {worst:F2} dB");
        Assert.That(worst,Is.LessThan(1.5),"Synthesized late energy follows the histogram envelope.");
    }

    [TestCase(TakeRender.Synthesis.Spectral,false,false),TestCase(TakeRender.Synthesis.ImpulseResponse,false,false)]
    [TestCase(TakeRender.Synthesis.Spectral,true,false),TestCase(TakeRender.Synthesis.ImpulseResponse,true,true)]
    [TestCase(TakeRender.Synthesis.Spectral,true,false,true),Category("GPU"),Category("LateEnergy")]
    public void LateAudioMatchesCpuReference(TakeRender.Synthesis method,bool ears,bool planar,bool tail=false)
    {
        var dimension=planar?AcousticTake.Dimension.Planar2D:AcousticTake.Dimension.Spatial3D;
        LateRoom(dimension,new Vector3(5,4,3),LateMaterial(.2f,.5f,.05f),new Vector3(-1.2f,.4f,.2f),new Vector3(1.1f,-.5f,-.3f),out var scene,out var geometry,out var state);
        var settings=LateSettings(8000,.3f);settings.lateRays=1024;settings.synthesis=method;settings.transitionSamples=32;
        settings.listenerOutput=ears?TakeRender.ListenerOutput.Binaural:TakeRender.ListenerOutput.Mono;
        if(tail)settings.lateTailSeconds=1.2f; // the continued envelope (LateTail / LateExtend) is heard after 0.3 s
        var clip=new float[331];for(int i=0;i<clip.Length;i++)clip[i]=.3f*Mathf.Sin(i*.21f)*(i%7==0?1:.4f);
        var signal=new TakeRender.Signal(clip);signal.SetActive(0,true);signal.Append(new AcousticTake.VoiceEvent{gain=1,loop=true});
        var signals=new[]{signal};var cpu=new TakeRender.Mixer(scene,8000,settings,signals);
        using var gpu=new GpuAudio(scene,8000,settings,signals);
        using var solver=new TransportSolver(scene,geometry,settings,RequireGpu(),sampleRate:8000);
        var direct=new TakeRender.DirectPath[1];int channels=settings.ChannelsPerListener,step=settings.responseStepSamples;
        var a=new float[step*channels];var b=new float[a.Length];double error=0,energy=0,peak=0;
        for(int frame=0;frame<(tail?200:40);frame++)
        {
            long sample=frame*step;state[1].position+=new Vector3(.02f,.01f,0);state[1].rotation=Quaternion.Euler(0,frame*7,0);
            solver.Solve(state,direct);
            cpu.SetResponse(sample,direct,state,false,solver.RayPaths,0,late:solver.Late(0));
            gpu.SetResponse(sample,false,solver.RayBuffer,0,scene,state,late:solver.Late(0));
            cpu.Render(sample,step,a);gpu.Render(sample,step,b);
            for(int i=0;i<a.Length;i++){double d=a[i]-b[i];error+=d*d;energy+=a[i]*a[i];peak=Math.Max(peak,Math.Abs(d));}
        }
        TestContext.WriteLine($"relative error {Math.Sqrt(error/energy):G3}, peak {peak:G3}");
        Assert.That(energy,Is.GreaterThan(1e-6));Assert.That(Math.Sqrt(error/energy),Is.LessThan(1e-3));
    }

    // Late spectral envelope: the interpolation weights sum to one, pass through each band
    // centre and are smooth across a centre (no octave kinks).
    [Test,Category("LateEnergy")]
    public void LateEnvelopeInterpolatesTheBandsSmoothly()
    {
        for(double hz=50;hz<20000;hz*=1.05)
        {
            double sum=0;for(int b=0;b<LateBasis.Bands;b++)sum+=LateBasis.EnvelopeWeight(b,hz);
            Assert.That(sum,Is.EqualTo(1).Within(1e-5));
        }
        for(int b=0;b<LateBasis.Bands;b++)Assert.That(LateBasis.EnvelopeWeight(b,LateBasis.BandCentre(b)),Is.EqualTo(1).Within(1e-5));
        // A tilted spectrum: the envelope's slope in log frequency is continuous at the 1 kHz centre.
        double Envelope(double hz){double a=0;for(int b=0;b<LateBasis.Bands;b++)a+=LateBasis.EnvelopeWeight(b,hz)*Math.Pow(.7,b*b*.3);return a;}
        double below=(Envelope(1000)-Envelope(1000/1.01))/Math.Log(1.01),above=(Envelope(1000*1.01)-Envelope(1000))/Math.Log(1.01);
        Assert.That(Math.Abs(above-below),Is.LessThan(.02*Math.Abs(below)+1e-4),"No kink at the band centre.");
    }

    // The envelope continuation beyond the horizon follows the fitted exponential,
    // joins it smoothly, fades out at the kernel's end, and leaves rising envelopes alone.
    [Test,Category("LateEnergy")]
    public void LateTailContinuesTheFittedDecay()
    {
        int simulated=40,total=200;double rate=Math.Pow(10,-6.0/50); // 60 dB per 50 bins: steep, so the window's curvature matters
        var random=new System.Random(3);
        var energy=new double[2][];
        for(int c=0;c<2;c++){energy[c]=new double[total];for(int k=0;k<simulated;k++)energy[c][k]=(c+1)*Math.Pow(rate,k)*(1+.1*(random.NextDouble()-.5));}
        double decay=LateTail.Extend(energy,simulated,total);
        Assert.That(decay,Is.EqualTo(rate).Within(.01*rate),"Fitted decay per bin.");
        for(int c=0;c<2;c++)
        {
            double want=(c+1)*Math.Pow(rate,simulated);
            Assert.That(energy[c][simulated]/want,Is.EqualTo(1).Within(.1),"The continuation starts on the decay line.");
            Assert.That(energy[c][150]/((c+1)*Math.Pow(rate,150)),Is.EqualTo(1).Within(.1),"and follows it.");
            Assert.That(energy[c][total-1],Is.LessThan(.05*(c+1)*Math.Pow(rate,total-1)),"The kernel's end fades out.");
            for(int k=simulated-5;k<simulated;k++)Assert.That(energy[c][k]/((c+1)*Math.Pow(rate,k)),Is.EqualTo(1).Within(.12),"Join bins stay on the line.");
        }
        // A reference curve (the room energy) sets the decay; the channels only set the level.
        var noisy=new double[1][];noisy[0]=new double[total];var reference=new double[total];
        for(int k=0;k<simulated;k++){noisy[0][k]=Math.Pow(rate,k)*(k%2==0?1.6:.4);reference[k]=5*Math.Pow(rate,k);}
        Assert.That(LateTail.Extend(noisy,simulated,total,reference),Is.EqualTo(rate).Within(1e-9*rate),"The reference's decay.");
        var rising=new double[1][];rising[0]=new double[total];for(int k=0;k<simulated;k++)rising[0][k]=1+k;
        Assert.That(LateTail.Extend(rising,simulated,total),Is.EqualTo(0));
        for(int k=simulated;k<total;k++)Assert.That(rising[0][k],Is.EqualTo(0),"A field that does not decay is not continued.");
    }

    // Synthesis with the continuation: a room whose reverberation outlasts the horizon
    // keeps decaying at the room's rate past it (no step, no cut) and fades out.
    [Test,Category("GPU"),Category("LateEnergy")]
    public void LateTailFadesTheReverberationOut()
    {
        var size=new Vector3(6,5,4);
        LateRoom(AcousticTake.Dimension.Spatial3D,size,LateMaterial(.1f,.6f),new Vector3(-1.5f,.5f,.4f),new Vector3(1.4f,-.7f,-.3f),out var scene,out var geometry,out var state);
        var settings=LateSettings(8000,.4f);settings.lateRays=4096;settings.audioBackend=TakeRender.AudioBackend.CpuReference;settings.lateTailSeconds=2.4f;
        using var solver=new TransportSolver(scene,geometry,settings,RequireGpu(),sampleRate:8000);solver.Solve(state,new TakeRender.DirectPath[1]);
        var impulse=new TakeRender.Signal(new[]{1f});impulse.SetActive(0,true);impulse.Append(new AcousticTake.VoiceEvent{gain=1});
        var late=new LateReverb(scene,8000,settings,new[]{impulse});late.SetResponse(0,true,solver.Late(0),state);
        int partition=settings.LatePartition,simulated=settings.LateBins(8000),total=settings.LateKernelBins(8000),latency=settings.DspLatencySamples;
        var output=new float[(total+2)*partition+latency];late.Render(0,output.Length,output);
        double Level(double from,double to){double e=0;int a=latency+(int)(from*8000),b=latency+(int)(to*8000);for(int n=a;n<b;n++)e+=output[n]*output[n];return 10*Math.Log10(e/(b-a)+1e-30);}
        // Eyring decay of the room (dB per second): 60 / T.
        double V=size.x*size.y*size.z,S=2*(size.x*size.y+size.y*size.z+size.x*size.z),eyring=60/(.161*V/(-S*Math.Log(1-.1)));
        double after=(Level(.5,.9)-Level(1.3,1.7))/.8,step=Level(.4,.5)-Level(.3,.4)+after*.1;
        TestContext.WriteLine($"decay after the horizon {after:F1} dB/s, Eyring {eyring:F1} dB/s, step at the horizon {step:F2} dB, end {Level(2.3,2.4)-Level(.3,.4):F0} dB");
        Assert.That(after/eyring,Is.EqualTo(1).Within(.2),"The continuation decays at the room's rate.");
        Assert.That(Math.Abs(step),Is.LessThan(1.5),"No step where the simulation ends.");
        Assert.That(Level(2.3,2.4)-Level(.3,.4),Is.LessThan(-60),"The tail fades out.");
    }

    // Late audio must exist and stay finite when the scene has a late field, and
    // a static scene must give identical late responses (fixed seeds).
    [Test,Category("GPU"),Category("LateEnergy")]
    public void LateHistogramIsStableForAStaticScene()
    {
        LateRoom(AcousticTake.Dimension.Spatial3D,new Vector3(5,4,3),LateMaterial(.2f,.5f),new Vector3(-1.2f,.4f,.2f),new Vector3(1.1f,-.5f,-.3f),out var scene,out var geometry,out var state);
        var settings=LateSettings(8000,.3f);settings.lateRays=512;
        using var solver=new TransportSolver(scene,geometry,settings,RequireGpu(),sampleRate:8000);
        solver.Solve(state,new TakeRender.DirectPath[1]);var first=(uint[])solver.Late(0).Transport.Late.Clone();
        solver.Solve(state,new TakeRender.DirectPath[1]);
        Assert.That(solver.Late(0).Transport.Late,Is.EqualTo(first),"Integer histogram sums are order-independent and seeds are fixed.");
        Assert.That(Sum(LateEnergy(solver,0)),Is.GreaterThan(0));
    }

    // ---- Benchmark measurement (docs/benchmarks/late-energy-2026-11-04) ----
    [Serializable] sealed class LateJob
    {
        public string name,output,kind="room",sceneDirectory,variant="tracer",dimension="3d";
        public float[] size,source,listener,listeners;
        public float absorption=.1f,scattering=1,transmission;
        public float[] frequencies,absorptionCurve,scatteringCurve; // optional frequency-dependent room material
        public int points=int.MaxValue; // pilot scenes: first receivers of the scene's grid
        public int sampleRate=8000,fftSize=64,responseStepSamples=64,maxBounces=1024,completeRouteSamples;
        public float maximumDelaySeconds=1,receiverRadius=.5f;
        public bool airAbsorption;
        public int[] rays,seeds;
    }
    [Serializable] sealed class LateJobList { public LateJob[] jobs; }
    [Serializable] sealed class LateRunRecord { public int rays,seed; public float[] solveSeconds; public string file,earlyFile; }
    [Serializable] sealed class LateJobResult
    {
        public string name,kind,backend,unityVersion;public int bins,bands,directions,points,partition,sampleRate,earlyOrder;public float binSeconds,receiverRadius;
        public float[] listeners;public LateRunRecord[] runs;public double wallSeconds,busySeconds;
    }
    [Test,Category("QualityMeasurement"),Timeout(int.MaxValue)]
    public void MeasureLateEnergy()
    {
        string file=Environment.GetEnvironmentVariable("ACOUSTIC_LATE_JOBS");
        if(string.IsNullOrEmpty(file))Assert.Ignore("Set ACOUSTIC_LATE_JOBS to a job list.");
        string budget=Environment.GetEnvironmentVariable("ACOUSTIC_TEST_WORK_FRACTION");
        double fraction=string.IsNullOrEmpty(budget)?1:double.Parse(budget,System.Globalization.CultureInfo.InvariantCulture);
        var shader=RequireGpu();
        foreach(var job in JsonUtility.FromJson<LateJobList>(File.ReadAllText(file)).jobs)
        {
            if(File.Exists(Path.Combine(job.output,"late.json")))continue;
            Directory.CreateDirectory(job.output);long started=System.Diagnostics.Stopwatch.GetTimestamp();
            AcousticTake.Scene scene;AcousticTake.Primitive[] geometry;AcousticTake.State[] state;DatasetSolid baked=null;
            var dimension=job.dimension=="2d"?AcousticTake.Dimension.Planar2D:AcousticTake.Dimension.Spatial3D;
            if(job.kind=="pilot")
            {
                string text=File.ReadAllText(Path.Combine(job.sceneDirectory,"scene.json"));
                var config=JsonUtility.FromJson<WaveScene>(text);var meta=JsonUtility.FromJson<DatasetScene>(text);
                var names=job.variant=="tracer"?meta.surfaceMaterials:Array.ConvertAll(meta.surfaceMaterials,_=>"rigid");
                baked=new DatasetSolid(config,names,job.output);scene=baked.scene;geometry=baked.geometry;state=baked.states;
            }
            else
            {
                var material=LateMaterial(job.absorption,job.scattering,job.transmission);
                if(job.frequencies!=null && job.frequencies.Length>0)
                {
                    material.frequencies=job.frequencies;material.absorption=job.absorptionCurve;material.scattering=job.scatteringCurve;
                    material.transmission=Array.ConvertAll(job.frequencies,_=>job.transmission);
                }
                LateRoom(dimension,V(job.size),material,V(job.source),V(job.listeners??job.listener),out scene,out geometry,out state);
            }
            try
            {
                var listeners=job.listeners??job.listener;
                if(job.kind=="pilot")
                {
                    var meta=JsonUtility.FromJson<DatasetScene>(File.ReadAllText(Path.Combine(job.sceneDirectory,"scene.json")));
                    listeners=meta.points;Array.Resize(ref listeners,Math.Min(listeners.Length,3*job.points));
                }
                int points=listeners.Length/3;
                var result=new LateJobResult{name=job.name,kind=job.kind,backend=SystemInfo.graphicsDeviceName,unityVersion=Application.unityVersion,listeners=listeners,points=points,
                    bands=TakeRender.Settings.LateBands,directions=TakeRender.Settings.LateDirections,sampleRate=job.sampleRate,receiverRadius=job.receiverRadius};
                var runs=new List<LateRunRecord>();double busy=0;
                foreach(int rays in job.rays)foreach(int seed in job.seeds)
                {
                    var settings=LateSettings(job.sampleRate,job.maximumDelaySeconds);settings.fftSize=job.fftSize;settings.spectralHop=job.fftSize/4;
                    settings.responseStepSamples=job.responseStepSamples;settings.maxBounces=job.maxBounces;settings.lateRays=rays;settings.seed=seed;
                    settings.lateReceiverRadius=job.receiverRadius;settings.airAbsorption=job.airAbsorption;settings.completeRouteSamples=job.completeRouteSamples;
                    settings.diffractionFactor=0;
                    using var solver=new TransportSolver(scene,geometry,settings,shader,sampleRate:job.sampleRate);
                    var late=solver.Late(0);int cells=TakeRender.Settings.LateDirections*TakeRender.Settings.LateBands;
                    var record=new LateRunRecord{rays=rays,seed=seed,solveSeconds=new float[points],file=$"late-r{rays}-s{seed}.f32"};
                    var values=new float[points*cells*(late.Transport?.LateBins??0)];var early=new List<float>();
                    state[1].position=V(listeners,0);solver.Solve(state,new TakeRender.DirectPath[1]); // exclude cold dispatch
                    for(int p=0;p<points;p++)
                    {
                        state[1].position=V(listeners,p);
                        long t0=System.Diagnostics.Stopwatch.GetTimestamp();solver.Solve(state,new TakeRender.DirectPath[1]);
                        double work=TakeRender.Elapsed(t0);record.solveSeconds[p]=(float)work;busy+=work;
                        int bins=late.Transport.LateBins;result.bins=bins;result.partition=settings.LatePartition;result.binSeconds=(float)((double)settings.LatePartition/job.sampleRate);
                        result.earlyOrder=settings.LateEarlyOrder;
                        for(int d=0;d<TakeRender.Settings.LateDirections;d++)for(int b=0;b<TakeRender.Settings.LateBands;b++)for(int k=0;k<bins;k++)
                            values[((p*TakeRender.Settings.LateDirections+d)*TakeRender.Settings.LateBands+b)*bins+k]=(float)late.Energy(0,d,b,k);
                        // Early (image/complete-route) arrivals of frequency bin 1: listener point, delay and complex
                        // pressure. Complete routes split one route over several samples; sum them before squaring.
                        for(int i=settings.FrequencyBins;i<solver.RayPaths.Length;i++)
                        {var path=solver.RayPaths[i];if(path.frequencyBin!=1 || (path.pressureGain==0 && path.pressureImag==0))continue;early.Add(p);early.Add(path.delaySeconds);early.Add(path.pressureGain);early.Add(path.pressureImag);}
                        if(fraction<1)System.Threading.Thread.Sleep((int)Math.Ceiling(work*1000*(1/fraction-1)));
                    }
                    WriteFloats(Path.Combine(job.output,record.file),values);
                    record.earlyFile=$"early-r{rays}-s{seed}.f32";WriteFloats(Path.Combine(job.output,record.earlyFile),early.ToArray());
                    runs.Add(record);
                }
                result.runs=runs.ToArray();result.busySeconds=busy;result.wallSeconds=TakeRender.Elapsed(started);
                File.WriteAllText(Path.Combine(job.output,"late.json"),JsonUtility.ToJson(result,true));
                Debug.Log($"Late energy job {job.name} done in {result.wallSeconds:F1} s");
            }
            finally{baked?.Dispose();}
        }
    }
    static Vector3 V(float[] values,int index=0)=>new Vector3(values[3*index],values[3*index+1],values[3*index+2]);
}
