using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using NUnit.Framework;
using UnityEngine;
using UnityEngine.Rendering;
using Complex = System.Numerics.Complex;

// Offline combination of the room modes with the path method's histogram: tools/room_modes.py combine, in C# (band
// energies [R, 8, T] of the coherent part W_m H_modal + (1 - W_m) H_direct on a 1/16 Hz grid, band-filtered with the
// suite's raised-cosine filters, plus g_b times the path histogram less its direct sound, smoothed by each band's
// energy kernel as method_compare does for histogram methods).
static class RoomModeOffline
{
    public const double Df=1/16.0, BinSeconds=1e-3;
    public const int Bins=2000;
    static readonly double[] Bands=RoomModes.Bands;

    // pathEnergy and the result: [receiver][band][bin]. visible[r]: the path method renders a direct sound there.
    public static double[] Combine(RoomModes modes,Vector3 source,Vector3[] points,double[] visible,double[] pathEnergy,out int[] touched)
    {
        var bank=modes.Bank;int N=bank.Modes,R=points.Length;double fs=modes.SchroederHz,c=modes.Speed;
        var t=new List<int>();for(int b=0;b<Bands.Length;b++)if(Bands[b]/2<fs*Math.Sqrt(2))t.Add(b);
        touched=t.ToArray();
        double top=2*Bands[touched[touched.Length-1]];int F=(int)Math.Round(top/Df)+1;
        var aS=new double[N];var aR=new double[R][];
        if(!bank.Probe(source,aS))throw new ArgumentException("The source is outside the bank's air.");
        for(int r=0;r<R;r++){aR[r]=new double[N];if(!bank.Probe(points[r],aR[r]))throw new ArgumentException($"Receiver {r} is outside the bank's air.");}
        // H_r(f) = sum_n psi_n(r) [4 pi c^2 psi_n(s) / (w_n^2 - w^2 + 2 i delta_n w)]
        var H=new Complex[R][];for(int r=0;r<R;r++)H[r]=new Complex[F];
        var g=new Complex[N];var wn2=new double[N];for(int n=0;n<N;n++){double w=bank.K[n]*c;wn2[n]=w*w;}
        double scale=4*Math.PI*c*c;
        for(int i=1;i<F;i++)
        {
            double f=i*Df,w=2*Math.PI*Math.Max(f,1e-6);
            for(int n=0;n<N;n++)g[n]=scale*aS[n]/new Complex(wn2[n]-w*w,2*modes.Delta[n]*w);
            double W=RoomModes.Crossover(f,fs);
            for(int r=0;r<R;r++)
            {
                Complex sum=Complex.Zero;var a=aR[r];for(int n=0;n<N;n++)sum+=a[n]*g[n];
                double d=Vector3.Distance(source,points[r]);
                var direct=visible[r]*Complex.FromPolarCoordinates(1/d,-2*Math.PI*f*d/c);
                H[r][i]=W*sum+(1-W)*direct;
            }
        }
        var output=new double[R*8*Bins];var rest=new double[Bins];
        for(int r=0;r<R;r++)
        {
            double d=Vector3.Distance(source,points[r]);int at=(int)Math.Floor(d/c/BinSeconds);
            for(int b=0;b<8;b++)
            {
                var path=new double[Bins];Array.Copy(pathEnergy,(r*8+b)*Bins,path,0,Bins);
                int j=Array.IndexOf(touched,b);
                if(j<0){Array.Copy(Smooth(path,Bands[b]),0,output,(r*8+b)*Bins,Bins);continue;}
                Array.Copy(path,rest,Bins);if(at<Bins)rest[at]-=visible[r]/(d*d);
                var smooth=Smooth(rest,Bands[b]);var coherent=EnergyFromSpectrum(H[r],Df,Bands[b]);
                for(int k=0;k<Bins;k++)output[(r*8+b)*Bins+k]=coherent[k]+modes.Share[b]*smooth[k];
            }
        }
        return output;
    }

    // method_compare.energy_from_spectrum for one band: p_b = irfft(G_b H, n) / dt on n = 2 (F - 1) points, the last
    // 0.1 s read as negative times (into bin 0), p_b^2 dt binned per 1 ms, over the filter's energy.
    public static double[] EnergyFromSpectrum(Complex[] H,double df,double fc)
    {
        int F=H.Length,n=2*(F-1);double dt=1/(n*df);int neg=(int)Math.Round(.1/dt);
        var spectrum=new Complex[n];
        for(int k=0;k<F;k++){var x=H[k]*Math.Sqrt(MethodBands.Power(k*df,fc));if(k==0||k==n/2)x=new Complex(x.Real,0);spectrum[k]=x;if(k>0&&k<n/2)spectrum[n-k]=Complex.Conjugate(x);}
        var p=InverseDft(spectrum);
        var energy=new double[Bins];double t0=0.0-neg*dt,norm=MethodBands.FilterEnergy(fc);
        for(int i=0;i<n;i++)
        {
            double v=p[(i+n-neg)%n].Real/dt,t=t0+i*dt;
            int k=Math.Max(0,(int)Math.Floor(t/BinSeconds));if(k<Bins)energy[k]+=v*v*dt;
        }
        for(int k=0;k<Bins;k++)energy[k]/=norm;
        return energy;
    }
    // (1/n) sum_k X_k exp(2 pi i jk / n) for any n, by Bluestein's chirp (radix-2 transforms of 2^m >= 2n - 1).
    static Complex[] InverseDft(Complex[] X)
    {
        int n=X.Length,m=1;while(m<2*n-1)m<<=1;
        var w=new Complex[n];for(long k=0;k<n;k++)w[k]=Complex.FromPolarCoordinates(1,Math.PI*((k*k)%(2L*n))/n);   // exp(i pi k^2 / n)
        double[] ar=new double[m],ai=new double[m],br=new double[m],bi=new double[m];
        for(int k=0;k<n;k++){var a=X[k]*w[k];ar[k]=a.Real;ai[k]=a.Imaginary;}
        for(int k=0;k<n;k++){var b=Complex.Conjugate(w[k]);br[k]=b.Real;bi[k]=b.Imaginary;if(k>0){br[m-k]=b.Real;bi[m-k]=b.Imaginary;}}
        RoomModes.Fft(ar,ai,false);RoomModes.Fft(br,bi,false);
        for(int k=0;k<m;k++){double r=ar[k]*br[k]-ai[k]*bi[k],i=ar[k]*bi[k]+ai[k]*br[k];ar[k]=r;ai[k]=i;}
        RoomModes.Fft(ar,ai,true);
        var x=new Complex[n];for(int j=0;j<n;j++)x[j]=w[j]*new Complex(ar[j],ai[j])/n;
        return x;
    }
    // method_compare.energy_kernel: |g_b(t)|^2 binned per 1 ms around bin 0, over the filter's energy.
    static readonly Dictionary<double,double[]> kernels=new Dictionary<double,double[]>();
    public static double[] EnergyKernel(double fc)
    {
        lock(kernels)if(kernels.TryGetValue(fc,out var known))return known;
        int half=Math.Max(3,(int)Math.Ceiling(8/fc/BinSeconds));double rate=Math.Max(64*fc,20/BinSeconds);
        int n=1<<(int)Math.Ceiling(Math.Log(4*half*BinSeconds*rate+16,2));
        var re=new double[n];var im=new double[n];
        for(int k=0;k<=n/2;k++){re[k]=Math.Sqrt(MethodBands.Power(k*rate/n,fc));if(k>0&&k<n/2)re[n-k]=re[k];}
        RoomModes.Fft(re,im,true);
        var kernel=new double[2*half+1];
        for(int i=0;i<n;i++)
        {
            double g=re[(i+n/2)%n]*rate,t=(double)(i-n/2)/rate;   // rolled by n/2
            int k=(int)Math.Round(t/BinSeconds,MidpointRounding.ToEven);
            if(Math.Abs(k)<=half)kernel[k+half]+=g*g/rate;
        }
        double norm=MethodBands.FilterEnergy(fc);for(int k=0;k<kernel.Length;k++)kernel[k]/=norm;
        lock(kernels)kernels[fc]=kernel;
        return kernel;
    }
    // method_compare.smooth: the band's kernel, full convolution, the spread before emission into bin 0.
    public static double[] Smooth(double[] x,double fc)
    {
        var kernel=EnergyKernel(fc);int h=kernel.Length/2,T=x.Length;var output=new double[T];double lead=0;
        for(int i=0;i<T;i++){if(x[i]==0)continue;for(int j=0;j<kernel.Length;j++){int at=i+j-h;double v=x[i]*kernel[j];if(at<0)lead+=v;else if(at<T)output[at]+=v;}}
        output[0]+=lead;
        return output;
    }
    // Total energy of band b of a sampled kernel (unit sample = free field at 1 m), from its spectrum.
    public static double BandTotal(double[] samples,int rate,double fc)
    {
        int n=1;while(n<2*samples.Length)n<<=1;
        var re=new double[n];var im=new double[n];Array.Copy(samples,re,samples.Length);RoomModes.Fft(re,im,false);
        double sum=0;for(int k=1;k<n/2;k++)sum+=MethodBands.Power((double)k*rate/n,fc)*(re[k]*re[k]+im[k]*im[k]);
        return 2*sum*rate/n/MethodBands.FilterEnergy(fc);
    }
}

public partial class TakeChecks
{
    // The worktree's .context (the runner sits at .context/gpu-audio/runner, the editor at the root).
    static string ContextPath(string relative)
    {
        for(var d=new DirectoryInfo(Application.dataPath);d!=null;d=d.Parent)
        {
            string candidate=Path.Combine(d.FullName,".context",relative);
            if(File.Exists(candidate)||Directory.Exists(candidate))return candidate;
        }
        return null;
    }
    [Serializable] sealed class RoomModeReference
    {
        public string sceneId,bank,pathEnergy,reference,take;
        public float[] source,points,lineOfSight,absorption,transmission,admittance,pathShare;
        public string[] materialNames;
        public float speedOfSound;
        public double schroederHz;
        public int receivers;
    }
    static float[] ReadFloats(string path){var bytes=File.ReadAllBytes(path);var v=new float[bytes.Length/4];Buffer.BlockCopy(bytes,0,v,0,bytes.Length);return v;}
    static Vector3[] Points(float[] xyz){var p=new Vector3[xyz.Length/3];for(int i=0;i<p.Length;i++)p[i]=new Vector3(xyz[3*i],xyz[3*i+1],xyz[3*i+2]);return p;}
    static double[][] Admittance(float[] values,int materials){var a=new double[materials][];for(int m=0;m<materials;m++){a[m]=new double[8];for(int b=0;b<8;b++)a[m][b]=values[m*8+b];}return a;}

    // The C# combination (RoomModes and RoomModeOffline) against tools/room_modes.py combine for shoebox-0020-presets with
    // the path method's own seed-0 histogram: 0.05 dB per receiver and band. Needs the exported bank and the reference
    // (tools/room_modes_unity.py reference shoebox-0020-presets); banks stay out of git, so without them it is skipped.
    [Test, Category("Methods")]
    public void RoomModesMatchThePythonCombination()
    {
        string file=ContextPath("bass-and-diffraction/shoebox-0020-presets.reference.json");
        if(file==null)Assert.Ignore("Run tools/room_modes_unity.py reference shoebox-0020-presets (the bank stays in .context).");
        var spec=JsonUtility.FromJson<RoomModeReference>(File.ReadAllText(file));
        var bank=RoomModeBank.Load(spec.bank);int M=bank.Materials.Length;
        Assert.That(bank.Materials, Is.EqualTo(spec.materialNames));
        // The admittance from absorption + transmission (Paris, hard branch) is the suite's.
        var admittance=new double[M][];double worst=0;
        for(int m=0;m<M;m++){admittance[m]=new double[8];for(int b=0;b<8;b++){admittance[m][b]=RoomModes.AdmittanceFromAbsorption(spec.absorption[m*8+b]+spec.transmission[m*8+b]);worst=Math.Max(worst,Math.Abs(admittance[m][b]/spec.admittance[m*8+b]-1));}}
        Assert.That(worst, Is.LessThan(1e-6), "admittance from absorption");
        var modes=new RoomModes(bank,admittance,spec.speedOfSound,1,8000,16000,16);
        Assert.That(modes.SchroederHz/spec.schroederHz, Is.EqualTo(1).Within(1e-6), "Schroeder frequency");
        for(int b=0;b<8;b++)Assert.That(modes.Share[b], Is.EqualTo(spec.pathShare[b]).Within(1e-6), "path share");
        var path=ReadFloats(spec.pathEnergy);var energy=new double[path.Length];for(int i=0;i<path.Length;i++)energy[i]=path[i];
        var visible=new double[spec.receivers];for(int r=0;r<visible.Length;r++)visible[r]=spec.lineOfSight[r];
        var watch=Stopwatch.StartNew();
        var combined=RoomModeOffline.Combine(modes,new Vector3(spec.source[0],spec.source[1],spec.source[2]),Points(spec.points),visible,energy,out var touched);
        double seconds=watch.Elapsed.TotalSeconds;
        var reference=ReadFloats(spec.reference);double maximum=0,rms=0;int count=0;
        for(int r=0;r<spec.receivers;r++)for(int b=0;b<8;b++)
        {
            double a=0,e=0;for(int k=0;k<RoomModeOffline.Bins;k++){a+=combined[(r*8+b)*RoomModeOffline.Bins+k];e+=reference[(r*8+b)*RoomModeOffline.Bins+k];}
            double db=10*Math.Log10(a/e);maximum=Math.Max(maximum,Math.Abs(db));rms+=db*db;count++;
        }
        rms=Math.Sqrt(rms/count);
        TestContext.WriteLine($"shoebox-0020-presets: {bank.Modes} modes, Schroeder {modes.SchroederHz:F1} Hz, bands {string.Join(",",touched)}: C# against Python max {maximum:F4} dB, RMS {rms:F4} dB over {count} receiver-bands ({seconds:F1} s)");
        Assert.That(maximum, Is.LessThan(.05), "the C# combination matches room_modes.combine per receiver and band");
    }

    // Opt-in (ACOUSTIC_ROOM_MODES_COST: ';'-separated reference files (tools/room_modes_unity.py), each with @horizon
    // seconds): per bank the cost of one update at 48 kHz, two ears, 1,024-sample updates. CPU: one pair's probes and
    // line of sight (RoomModes.Pair), and the CPU reference's synthesis of dh; GPU audio: one update's modal work alone
    // (the CPU part, then the kernels: synthesis, transforms, crossover, interpolation), fenced. Pairs: random air cells.
    // The Church's bank runs on its take (frame 0, so its sidecar's geometry and poses hold), the suite's on a bare scene.
    [Test, Category("QualityMeasurement"), Timeout(int.MaxValue)]
    public void MeasureRoomModeCost()
    {
        string list=Environment.GetEnvironmentVariable("ACOUSTIC_ROOM_MODES_COST");
        if(string.IsNullOrEmpty(list))Assert.Ignore("Set ACOUSTIC_ROOM_MODES_COST.");
        var shader=RequireGpu();const int rate=48000,step=1024,trials=24;
        using var probe=new ComputeBuffer(1,4);
        void Fence()=>AsyncGPUReadback.Request(probe).WaitForCompletion();
        double Median(List<double> v){var x=v.ToArray();Array.Sort(x);return x[x.Length/2];}
        foreach(string entry in list.Split(';'))
        {
            var parts=entry.Split('@');double horizon=parts.Length>1?double.Parse(parts[1],System.Globalization.CultureInfo.InvariantCulture):2;
            var spec=JsonUtility.FromJson<RoomModeReference>(File.ReadAllText(parts[0]));
            long before=GC.GetTotalMemory(true);var watch=Stopwatch.StartNew();
            var bank=RoomModeBank.Load(spec.bank);double loadSeconds=watch.Elapsed.TotalSeconds;
            var settings=SpectralSettings();settings.method=TakeRender.Method.Path;settings.audioBackend=TakeRender.AudioBackend.Gpu;
            settings.listenerOutput=TakeRender.ListenerOutput.Binaural;settings.responseStepSamples=step;settings.fftSize=step;settings.spectralHop=step/4;
            settings.maximumDelaySeconds=(float)horizon;settings.speedOfSound=spec.speedOfSound;settings.minimumDistance=.1f;settings.delayFilter=TakeRender.DelayFilter.WindowedSinc;
            settings.roomModesBank=spec.bank;settings.pathPersistent=true;
            AcousticTake take=null;AcousticTake.Scene scene;AcousticTake.State[] state;TakeRender.Signal[] signals;
            if(!string.IsNullOrEmpty(spec.take))
            {
                take=AcousticTake.Open(spec.take);scene=take.Metadata.scene;state=new AcousticTake.State[scene.objects.Length];take.ReadStoredFrame(0,state);
                var clips=new float[scene.clips.Length][];for(int c=0;c<clips.Length;c++)clips[c]=take.ReadClip(c);signals=TakeRender.Signal.FromTake(take,clips);
            }
            else
            {
                scene=Scene();state=States();
                scene.materials=new AcousticTake.MaterialDefinition[spec.materialNames.Length];
                for(int m=0;m<scene.materials.Length;m++)
                {
                    var real=new float[8];Array.Copy(spec.admittance,m*8,real,0,8);
                    scene.materials[m]=new AcousticTake.MaterialDefinition{name=spec.materialNames[m],frequencies=Array.ConvertAll(RoomModes.Bands,f=>(float)f),admittanceReal=real,admittanceImag=new float[8]};
                }
                signals=new[]{new TakeRender.Signal(new float[64])};
            }
            try
            {
                int P=settings.LatePartition,count=(int)Math.Ceiling(new PathBins(settings).Horizon*rate/P)+1;
                var modes=RoomModes.ForScene(scene,settings,rate,count*P);
                Assert.That(modes.Disabled, Is.Null, "the bank applies to its scene");
                Assert.That(modes.Applies(state), "the scene's objects are at the bank's baked poses");
                var random=new System.Random(7);int sourceObject=scene.sources[0].objectIndex,listenerObject=scene.listeners[0];
                Vector3 Air(){int g=bank.AirCell[random.Next(bank.Cells)];int z=g%bank.Shape[2],y=g/bank.Shape[2]%bank.Shape[1],x=g/(bank.Shape[1]*bank.Shape[2]);return new Vector3((float)bank.Centres[0][x],(float)bank.Centres[1][y],(float)bank.Centres[2][z]);}
                var amplitudes=new double[bank.Modes];var dh=new double[count*P];
                List<double> pair=new List<double>(),synthesis=new List<double>(),gpuCpu=new List<double>(),gpuKernels=new List<double>();
                using var gpu=new GpuAudio(scene,rate,settings,signals);
                for(int t=0;t<trials;t++)
                {
                    state[sourceObject].position=Air();state[listenerObject].position=Air();
                    watch.Restart();bool inside=modes.Pair(state[sourceObject].position,state[listenerObject].position,settings.minimumDistance,amplitudes,out double direct,out double delay);pair.Add(watch.Elapsed.TotalMilliseconds);
                    Assert.That(inside);
                    if(t<6){watch.Restart();modes.Synthesize(amplitudes,direct,delay,dh);synthesis.Add(watch.Elapsed.TotalMilliseconds);}
                    Fence();watch.Restart();gpu.ProfileModes(scene,state);double cpuPart=watch.Elapsed.TotalMilliseconds;Fence();double all=watch.Elapsed.TotalMilliseconds;
                    Assert.That(gpu.ModalPairs, Is.GreaterThanOrEqualTo(1));
                    if(t>=2){gpuCpu.Add(cpuPart);gpuKernels.Add(all-cpuPart);}
                }
                long gpuBytes=(long)bank.Modes*(16+4+4*scene.sources.Length*scene.listeners.Length)+2L*modes.FftSize*8*scene.sources.Length*scene.listeners.Length+(modes.Weights.Length+modes.Interpolation.Length)*4;
                UnityEngine.Debug.Log(FormattableString.Invariant($"ROOM_MODES_COST {spec.sceneId}: {bank.Modes} modes to {modes.TopHz:F0} Hz, Schroeder {modes.SchroederHz:F1} Hz, {bank.Cells} air cells; ") +
                    FormattableString.Invariant($"horizon {horizon} s, decimated {modes.Rate} Hz ({modes.Length} samples, FFT {modes.FftSize}); bank RAM {bank.Bytes/1048576.0:F1} MiB (load {loadSeconds:F1} s, managed heap +{(GC.GetTotalMemory(false)-before)/1048576.0:F0} MiB), GPU {gpuBytes/1048576.0:F2} MiB; ") +
                    FormattableString.Invariant($"CPU per update: pair (2 probes + line of sight) {Median(pair):F2} ms, CPU-reference synthesis {Median(synthesis):F1} ms; GPU audio per update: CPU part {Median(gpuCpu):F2} ms, kernels {Median(gpuKernels):F2} ms (fenced)"));
            }
            finally{take?.Dispose();}
        }
    }

    // A rigid box's analytic modes written as a bank (RMB1): interior [origin, origin + size] in cells of h, one solid
    // layer of material "wall" around it; modes to fmax.
    static string WriteBoxBank(string path,Vector3 origin,Vector3 size,double h,double fmax)
    {
        var L=new double[]{size.x,size.y,size.z};var o=new double[]{origin.x,origin.y,origin.z};var counts=new int[3];
        var edges=new float[3][];
        for(int a=0;a<3;a++){counts[a]=(int)Math.Round(L[a]/h);edges[a]=new float[counts[a]+3];for(int i=0;i<edges[a].Length;i++)edges[a][i]=(float)(o[a]+(i-1)*h);}
        int nx=counts[0]+2,ny=counts[1]+2,nz=counts[2]+2;var solid=new short[nx*ny*nz];var air=new List<Vector3>();
        for(int x=0;x<nx;x++)for(int y=0;y<ny;y++)for(int z=0;z<nz;z++)
        {
            bool inside=x>0&&y>0&&z>0&&x<nx-1&&y<ny-1&&z<nz-1;solid[(x*ny+y)*nz+z]=(short)(inside?-1:0);
            if(inside)air.Add(new Vector3((float)(o[0]+(x-.5)*h-o[0]),(float)(o[1]+(y-.5)*h-o[1]),(float)(o[2]+(z-.5)*h-o[2])));
        }
        double kmax=2*Math.PI*fmax/343;var modes=new List<(int,int,int,double)>();
        for(int i=0;i<=2*fmax*L[0]/343;i++)for(int j=0;j<=2*fmax*L[1]/343;j++)for(int k=0;k<=2*fmax*L[2]/343;k++)
        {double kk=Math.PI*Math.Sqrt(i*i/(L[0]*L[0])+j*j/(L[1]*L[1])+k*k/(L[2]*L[2]));if(kk<=kmax)modes.Add((i,j,k,kk));}
        using var w=new BinaryWriter(File.Create(path));
        w.Write(new[]{'R','M','B','1'});w.Write(1);w.Write(nx);w.Write(ny);w.Write(nz);
        foreach(var e in edges)foreach(float v in e)w.Write(v);
        foreach(short s in solid)w.Write(s);
        w.Write(1);var name=System.Text.Encoding.UTF8.GetBytes("wall");w.Write(name.Length);w.Write(name);
        w.Write(modes.Count);foreach(var m in modes)w.Write((float)m.Item4);
        foreach(var m in modes)
        {
            double eps=(m.Item1==0?1:.5)*(m.Item2==0?1:.5)*(m.Item3==0?1:.5),norm2=L[0]*L[1]*L[2]*eps;var e3=new[]{m.Item1==0?1:.5,m.Item2==0?1:.5,m.Item3==0?1:.5};
            double S=0;for(int a=0;a<3;a++){double face=1;for(int b=0;b<3;b++)if(b!=a)face*=L[b]*e3[b];S+=2*face/norm2;}
            w.Write((float)S);
        }
        w.Write(air.Count);
        foreach(var m in modes)
        {
            double norm=Math.Sqrt(L[0]*L[1]*L[2]*(m.Item1==0?1:.5)*(m.Item2==0?1:.5)*(m.Item3==0?1:.5));
            foreach(var p in air)w.Write(Mathf.FloatToHalf((float)(Math.Cos(Math.PI*m.Item1*p.x/L[0])*Math.Cos(Math.PI*m.Item2*p.y/L[1])*Math.Cos(Math.PI*m.Item3*p.z/L[2])/norm)));
        }
        return path;
    }
    static AcousticTake.MaterialDefinition Wall(float absorption) =>
        new AcousticTake.MaterialDefinition {name="wall",frequencies=new[]{0f},absorption=new[]{absorption},scattering=new[]{0f},transmission=new[]{0f}};

    // The CPU reference renders the modal difference dh itself: a static pair in a box bank, an empty histogram (so the
    // reverb is dh alone) and a unit impulse; the rendered response's energy in the bands the modes reach matches
    // W_m (H_modal - H_direct) computed in frequency (the transfer function exactly, on a fine grid).
    [Test, Category("Methods")]
    public void RoomModesRenderTheModalResponse()
    {
        const int rate=8000,size=256;
        string bank=WriteBoxBank(Path.Combine(directory,"box.roommodes.bytes"),new Vector3(-1,.5f,-.5f),new Vector3(6,2.5f,2.5f),.25,360);
        var scene=Scene();scene.materials=new[]{Wall(.3f)};var state=States();
        state[0].position=new Vector3(0,2,0);state[1].position=new Vector3(3.3f,1.4f,1.1f);
        var settings=ControlSettings(TakeRender.Method.Path,64,2f);settings.roomModesBank=bank;settings.listenerOutput=TakeRender.ListenerOutput.Mono;
        settings.responseStepSamples=size;settings.fftSize=size;settings.spectralHop=size/4;
        var clip=new float[1];clip[0]=1;
        var signal=new TakeRender.Signal(clip);signal.SetActive(0,true);signal.Append(new AcousticTake.VoiceEvent {action=AcousticTake.VoiceAction.Play,gain=1});
        var reverb=new PathReverb(scene,rate,settings,new[]{signal});
        reverb.SetResponse(0,true,(p,d,b,k)=>0,state);
        Assert.That(reverb.ModalPairs, Is.EqualTo(1), "the pair is inside the bank's room");
        var output=new float[(int)reverb.SupportSamples+size];reverb.Render(0,output.Length,output);
        var rendered=new double[output.Length];for(int n=0;n<output.Length;n++)rendered[n]=output[n];
        // Expected: the same amplitudes and damping, in frequency.
        var modes=reverb.Modes;var amplitudes=new double[modes.Bank.Modes];
        Assert.That(modes.Pair(state[0].position,state[1].position,settings.minimumDistance,amplitudes,out double direct,out double delay));
        double c=modes.Speed,fs=modes.SchroederHz;
        foreach(int b in new[]{0,1})
        {
            double fc=RoomModes.Bands[b],expected=0,df=1/64.0;
            for(double f=fc/2+df/2;f<2*fc;f+=df)
            {
                double w=2*Math.PI*f;Complex h=Complex.Zero;
                for(int n=0;n<amplitudes.Length;n++){double wn=modes.Bank.K[n]*c;h+=amplitudes[n]/new Complex(wn*wn-w*w,2*modes.Delta[n]*w);}
                var x=RoomModes.Crossover(f,fs)*(h-Complex.FromPolarCoordinates(direct,-w*delay));
                expected+=MethodBands.Power(f,fc)*x.Magnitude*x.Magnitude*df;
            }
            expected=2*expected/MethodBands.FilterEnergy(fc);
            double got=RoomModeOffline.BandTotal(rendered,rate,fc),db=10*Math.Log10(got/expected);
            TestContext.WriteLine($"band {fc} Hz: rendered {got:G4}, expected {expected:G4} ({db:+0.000;-0.000} dB); Schroeder {fs:F0} Hz, {modes.Bank.Modes} modes, decimated {modes.Rate} Hz");
            Assert.That(Math.Abs(db), Is.LessThan(.1), "the rendered modal difference has the modal transfer function's band energy");
        }
    }

    // The GPU's dh at the decimated rate against the CPU reference's (RoomModes.Synthesize at the decimated samples).
    [Test, Category("GPU"), Category("Methods")]
    public void GpuRoomModeSynthesisMatchesTheCpu()
    {
        RequireGpu();const int rate=8000,size=256;
        string bank=WriteBoxBank(Path.Combine(directory,"box.roommodes.bytes"),new Vector3(-1,.5f,-.5f),new Vector3(6,2.5f,2.5f),.25,360);
        var scene=Scene();scene.materials=new[]{Wall(.3f)};var state=States();
        state[0].position=new Vector3(0,2,0);state[1].position=new Vector3(3.3f,1.4f,1.1f);
        var settings=ControlSettings(TakeRender.Method.Path,64,.25f);settings.roomModesBank=bank;settings.transport=TakeRender.Transport.Raytraced;
        settings.responseStepSamples=size;settings.fftSize=size;settings.spectralHop=size/4;
        using var gpu=new GpuAudio(scene,rate,settings,new[]{new TakeRender.Signal(new float[8])});
        gpu.ProfileModes(scene,state);
        var modes=gpu.Modes;var got=gpu.ReadModes(0);
        var amplitudes=new double[modes.Bank.Modes];modes.Pair(state[0].position,state[1].position,settings.minimumDistance,amplitudes,out double direct,out double delay);
        var dh=new double[modes.KernelSamples];modes.Synthesize(amplitudes,direct,delay,dh);
        double error=0,energy=0,peak=0;
        for(int j=0;j*modes.Ratio<dh.Length;j++){double e=got[j]-dh[j*modes.Ratio];error+=e*e;energy+=dh[j*modes.Ratio]*dh[j*modes.Ratio];peak=Math.Max(peak,Math.Abs(dh[j*modes.Ratio]));}
        gpu.ModeStages=1;gpu.ProfileModes(scene,state);var gh=gpu.ReadModes(0);var ch=modes.ModalImpulse(amplitudes);
        double he=0,hn=0;for(int j=0;j<ch.Length;j++){double g2=gh[j]*modes.FftSize*modes.OutputRate;he+=(g2-ch[j])*(g2-ch[j]);hn+=ch[j]*ch[j];}
        TestContext.WriteLine($"GPU h_modal against CPU: relative {Math.Sqrt(he/hn):G3}; h[1] {gh[1]*modes.FftSize*modes.OutputRate:G5}/{ch[1]:G5}, h[50] {gh[50]*modes.FftSize*modes.OutputRate:G5}/{ch[50]:G5}; length {modes.Length}, FFT {modes.FftSize}, ratio {modes.Ratio}");
        TestContext.WriteLine($"GPU dh against CPU: relative {Math.Sqrt(error/energy):G3}, peak {peak:G3}, first {got[0]:G4}/{dh[0]:G4}, at 40: {got[40]:G4}/{dh[40*modes.Ratio]:G4}");
        Assert.That(Math.Sqrt(error/energy), Is.LessThan(1e-3));
    }

    // GPU against CPU with the room modes on (GpuPathReverbMatchesTheCpuReference's fixture, inside a box bank): the
    // histogram scaled by g_b and dh added, on both.
    [TestCase(TakeRender.ListenerOutput.Binaural)]
    [TestCase(TakeRender.ListenerOutput.Mono)]
    [Category("GPU"), Category("Methods")]
    public void GpuRoomModesMatchTheCpuReference(TakeRender.ListenerOutput ears)
    {
        var shader=RequireGpu();
        const int rate=8000,size=256;
        string bank=WriteBoxBank(Path.Combine(directory,"box.roommodes.bytes"),new Vector3(-1,.5f,-.5f),new Vector3(6,2.5f,2.5f),.25,360);
        var scene=Scene(2);scene.listeners=new[]{1};scene.materials=new[]{RigidControl(),Wall(.3f)};
        var a=new Vector3(-20,0,-20);var b=new Vector3(20,0,-20);var c=new Vector3(20,0,20);var d=new Vector3(-20,0,20);
        var geometry=new[]{new AcousticTake.Primitive{objectIndex=2,a=a,b=c,c=b},new AcousticTake.Primitive{objectIndex=2,a=a,b=d,c=c}};
        scene.objects[2].primitiveCount=geometry.Length;
        var state=States(2);
        state[0].position=new Vector3(0,2,0);state[1].position=new Vector3(3,1.5f,1);
        state[2].position=Vector3.zero;state[2].flags=AcousticTake.StateFlags.Active|AcousticTake.StateFlags.Geometry;
        var settings=ControlSettings(TakeRender.Method.Path,4096,.25f);settings.useBvh=false;settings.roomModesBank=bank;
        settings.transport=TakeRender.Transport.Raytraced;settings.synthesis=TakeRender.Synthesis.Spectral;settings.listenerOutput=ears;
        settings.pathPersistent=true;settings.responseStepSamples=size;settings.fftSize=size;settings.spectralHop=size/4;settings.transitionSamples=size/4;
        var clip=new float[331];for(int n=0;n<clip.Length;n++)clip[n]=.2f*Mathf.Sin(n*.37f)+.1f*Mathf.Cos(n*1.93f);
        TakeRender.Signal Signal()
        {
            var s=new TakeRender.Signal(clip);s.SetActive(0,true);
            s.Append(new AcousticTake.VoiceEvent{action=AcousticTake.VoiceAction.Play,gain=1,loop=true});
            return s;
        }
        using var gpu=new GpuAudio(scene,rate,settings,new[]{Signal()});
        var cpu=new TakeRender.Mixer(scene,rate,settings,new[]{Signal()});
        using var solver=new TransportSolver(scene,geometry,settings,shader,sampleRate:rate,methodAudio:true);
        var direct=new TakeRender.DirectPath[1];int channels=settings.ChannelsPerListener;
        var x=new float[size*channels];var y=new float[x.Length];double peak=0,error=0,energy=0;int modal=0;
        for(int frame=0;frame<14;frame++)
        {
            long sample=frame*(long)size;bool cut=frame==9;
            state[1].position.x=3+frame*.05f;state[1].rotation=Quaternion.Euler(0,frame*11,0);
            solver.Solve(state,direct);
            cpu.SetResponse(sample,direct,state,cut,solver.RayPaths,0,late:solver.Late(0));
            gpu.SetResponse(sample,cut,solver.RayBuffer,0,scene,state,late:solver.Late(0));
            Assert.That(gpu.ModalPairs, Is.EqualTo(cpu.PathLate.ModalPairs), "the same pairs get the modes");
            modal+=cpu.PathLate.ModalPairs;
            int at=0;
            foreach(int count in new[]{37,size-37})
            {
                Array.Clear(x,0,x.Length);Array.Clear(y,0,y.Length);
                cpu.Render(sample+at,count,x);gpu.Render(sample+at,count,y);
                for(int n=0;n<count*channels;n++){double e=x[n]-y[n];peak=Math.Max(peak,Math.Abs(e));error+=e*e;energy+=x[n]*x[n];}
                at+=count;
            }
        }
        double relative=Math.Sqrt(error/energy);
        TestContext.WriteLine($"{ears} with room modes ({modal} pair updates, Schroeder {cpu.PathLate.Modes.SchroederHz:F0} Hz): GPU against CPU peak {peak:G3}, relative {relative:G3}, output energy {energy:G3}");
        Assert.That(modal, Is.EqualTo(14), "every update has the pair inside the bank's room");
        Assert.That(relative, Is.LessThan(2e-3), "the GPU room modes render what the CPU reference does");
    }
}
