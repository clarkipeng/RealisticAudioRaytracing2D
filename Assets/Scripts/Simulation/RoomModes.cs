using System;
using System.Collections.Generic;
using System.IO;
using UnityEngine;

// Room modes below the Schroeder frequency, crossed over with the path method (tools/room_modes.py is the reference;
// docs/benchmarks/bass-and-diffraction-2026-11-22). A per-scene bank holds the rigid room's eigenmodes psi_n on a
// rectilinear grid (finite volumes), their fourth-order corrected wavenumbers k_n and per material the wall integrals
// S_nm = integral of psi_n^2 over that material's faces (export format: room_modes.export, 'RMB1').
//
// Damping: delta_n = (c / 2) sum_m beta_m(f_n) S_nm, beta the real normalised admittance of each material at the mode's
// own frequency (linear in log2 f between the eight band centres, held outside), from the energy that leaves through
// the wall (absorption + transmission) by Paris's random-incidence formula, hard branch. Transfer (free field 1/r, the
// path method's units): H(w) = sum_n 4 pi c^2 psi_n(s) psi_n(r) / (w_n^2 - w^2 + 2 i delta_n w).
// Crossover at the Schroeder frequency f_S measured from the bank (modal overlap three): the modal weight
// W_m(f) = cos(pi/2 clip(log2(f / f_S) + 1/2, 0, 1)) in amplitude. The exact direct sound stays as the path method renders
// it; the path method's band energies (in audio, the histogram: everything but the direct sound) are scaled by
// g_b = the band filter's energy share of W_g^2 = 1 - W_m^2 (PathShare), and the coherent modal difference
// dh = W_m * (h_modal - h_direct) is added: W_m h_modal + (1 - W_m) h_direct + W_g^2 (rest), nothing counted twice.
// The bank describes one closed room in one configuration: dh is used only while both ends are in its air cells and
// every object it was baked with is at its baked pose (the sidecar, RoomModes.Binding); otherwise g_b = 1 and dh = 0.
//
// The per-update response of a pair (Synthesize; GpuAudio.Modes mirrors it): h_modal is synthesised at a decimated
// rate, the smallest divisor of the output rate that is at least four times the bank's top mode, over the reverb's
// kernel; W_m is applied by FFT on that IR (zero-phase, on a period of at least twice the IR, so its wrap-around lies
// a whole IR away) together with the direct pulse r^-1 at r/c (exact in frequency, so band-limited by W_m itself);
// then the result is interpolated to the output rate by the renderer's windowed sinc. Approximations: the IR ends at
// the reverb's horizon, as the path method's own does; the zero-phase crossover's precursor before emission (t < 0)
// is dropped (the kernel starts at the emission); the sampled h_modal aliases its modes' 1/f^2 tails from above the
// decimated Nyquist frequency (two octaves above the top mode, so about delta/w of the in-band level).
internal sealed class RoomModeBank
{
    public readonly double[][] Edges,Centres,Widths;
    public readonly int[] Shape;
    public readonly short[] Solid;       // [x][y][z] (C order): -1 air, else the material
    public readonly int[] AirIndex;      // grid cell -> air cell (C order of the air cells), -1 solid
    public readonly int[] AirCell;       // air cell -> grid cell
    public readonly string[] Materials;
    public readonly double[] K;          // rad/m
    public readonly double[] Surface;    // [mode][material]
    public readonly int Modes,Cells;
    readonly ushort[] psi;               // [air cell][mode], float16: a probe reads every mode of a few cells
    static readonly float[] halves=BuildHalves();
    public long Bytes => psi.LongLength*2+Solid.LongLength*2+(AirIndex.LongLength+AirCell.LongLength)*4+(K.LongLength+Surface.LongLength)*8;
    public double TopHz(double speed){double k=0;foreach(double v in K)k=Math.Max(k,v);return k*speed/(2*Math.PI);}

    static float[] BuildHalves(){var t=new float[65536];for(int i=0;i<t.Length;i++)t[i]=Mathf.HalfToFloat((ushort)i);return t;}

    RoomModeBank(double[][] edges,short[] solid,string[] materials,double[] k,double[] surface,ushort[] cellMajor,int cells)
    {
        Edges=edges;Shape=new[]{edges[0].Length-1,edges[1].Length-1,edges[2].Length-1};Solid=solid;Materials=materials;K=k;Surface=surface;Modes=k.Length;
        Centres=new double[3][];Widths=new double[3][];
        for(int a=0;a<3;a++){int n=Shape[a];Centres[a]=new double[n];Widths[a]=new double[n];for(int i=0;i<n;i++){Centres[a][i]=.5*(edges[a][i]+edges[a][i+1]);Widths[a][i]=edges[a][i+1]-edges[a][i];}}
        AirIndex=new int[solid.Length];var air=new List<int>();
        for(int g=0;g<solid.Length;g++){AirIndex[g]=solid[g]<0?air.Count:-1;if(solid[g]<0)air.Add(g);}
        if(air.Count!=cells)throw new InvalidDataException("Room mode bank: air cells do not match the grid.");
        AirCell=air.ToArray();
        Cells=cells;psi=cellMajor;
    }

    // One loaded copy per file (banks are immutable): the CPU reverb, the GPU reverb and measurements share it.
    static readonly Dictionary<string,(DateTime,WeakReference<RoomModeBank>)> cache=new Dictionary<string,(DateTime,WeakReference<RoomModeBank>)>();
    public static RoomModeBank Load(string path)
    {
        path=Path.GetFullPath(path);var stamp=File.GetLastWriteTimeUtc(path);
        lock(cache)
        {
            if(cache.TryGetValue(path,out var entry) && entry.Item1==stamp && entry.Item2.TryGetTarget(out var known))return known;
            var bank=Read(path);cache[path]=(stamp,new WeakReference<RoomModeBank>(bank));return bank;
        }
    }
    static RoomModeBank Read(string path)
    {
        using var reader=new BinaryReader(File.OpenRead(path));
        if(new string(reader.ReadChars(4))!="RMB1" || reader.ReadInt32()!=1)throw new InvalidDataException(path+" is not a room mode bank (RMB1, version 1).");
        int nx=reader.ReadInt32(),ny=reader.ReadInt32(),nz=reader.ReadInt32();
        var edges=new double[3][];int[] shape={nx,ny,nz};
        for(int a=0;a<3;a++){edges[a]=new double[shape[a]+1];for(int i=0;i<=shape[a];i++)edges[a][i]=reader.ReadSingle();}
        var solid=new short[checked(nx*ny*nz)];for(int i=0;i<solid.Length;i++)solid[i]=reader.ReadInt16();
        var names=new string[reader.ReadInt32()];
        for(int m=0;m<names.Length;m++)names[m]=System.Text.Encoding.UTF8.GetString(reader.ReadBytes(reader.ReadInt32()));
        int modes=reader.ReadInt32();var k=new double[modes];for(int n=0;n<modes;n++)k[n]=reader.ReadSingle();
        var surface=new double[modes*names.Length];for(int i=0;i<surface.Length;i++)surface[i]=reader.ReadSingle();
        int cells=reader.ReadInt32();
        var cellMajor=new ushort[checked((long)cells*modes)];var row=new ushort[cells];
        for(int n=0;n<modes;n++)
        {
            var bytes=reader.ReadBytes(2*cells);if(bytes.Length!=2*cells)throw new EndOfStreamException(path+": truncated mode shapes.");
            Buffer.BlockCopy(bytes,0,row,0,bytes.Length);
            for(int c=0;c<cells;c++)cellMajor[(long)c*modes+n]=row[c];
        }
        return new RoomModeBank(edges,solid,names,k,surface,cellMajor,cells);
    }

    int Grid(int x,int y,int z)=>(x*Shape[1]+y)*Shape[2]+z;
    int Air(int[] c)=>c[0]<0||c[1]<0||c[2]<0||c[0]>=Shape[0]||c[1]>=Shape[1]||c[2]>=Shape[2]?-1:AirIndex[Grid(c[0],c[1],c[2])];
    // First index i with edges[i] >= x (numpy searchsorted, side left).
    static int SearchLeft(double[] e,double x){int lo=0,hi=e.Length;while(lo<hi){int m=(lo+hi)>>1;if(e[m]<x)lo=m+1;else hi=m;}return lo;}
    // The grid cell holding a point (room_modes.probe's c0), or false outside the grid.
    bool CellOf(Vector3 p,int[] c)
    {
        for(int a=0;a<3;a++){double x=p[a];if(!(x>=Edges[a][0] && x<=Edges[a][Shape[a]]))return false;c[a]=Math.Clamp(SearchLeft(Edges[a],x)-1,0,Shape[a]-1);}
        return true;
    }
    public bool InAir(Vector3 p){var c=new int[3];return CellOf(p,c) && Air(c)>=0;}

    // psi_n at a point, fourth-order accurate (room_modes.probe): trilinear interpolation over the eight cell centres
    // around it minus its leading error, sum_a t_a (1 - t_a) h_a^2 / 2 times the interpolated second difference along a
    // (the stencil's own, walls included). Across a wall the neighbour is the cell's mirror image (the zero normal
    // gradient makes psi even about the wall). False if the point is outside the grid or in a solid cell.
    public bool Probe(Vector3 point,double[] output,double[] scratch=null)
    {
        var c0=new int[3];if(!CellOf(point,c0) || Air(c0)<0)return false;
        var t=new double[3];var span=new double[3];var step=new int[3];
        for(int a=0;a<3;a++)
        {
            double d=point[a]-Centres[a][c0[a]];int st=d>=0?1:-1;var nb=(int[])c0.Clone();nb[a]+=st;
            if(Air(nb)>=0){span[a]=Math.Abs(Centres[a][nb[a]]-Centres[a][c0[a]]);step[a]=st;}
            else{span[a]=Widths[a][c0[a]];step[a]=0;}
            t[a]=Math.Abs(d)/span[a];
        }
        Array.Clear(output,0,Modes);
        scratch??=new double[Modes];var target=new int[3];
        for(int dx=0;dx<2;dx++)for(int dy=0;dy<2;dy++)for(int dz=0;dz<2;dz++)
        {
            int[] delta={dx,dy,dz};double w=1;
            for(int a=0;a<3;a++)w*=delta[a]!=0?t[a]:1-t[a];
            if(w==0)continue;
            for(int a=0;a<3;a++)target[a]=c0[a]+delta[a]*step[a];
            for(int a=0;a<3;a++){if(Air(target)>=0)break;target[a]=c0[a];}   // a solid diagonal corner: step back along an axis
            int cell=Air(target);
            // psi_j - sum_a t_a (1 - t_a) span_a^2 / 2 (second difference along a at j)
            long row=(long)cell*Modes;
            for(int n=0;n<Modes;n++)scratch[n]=halves[psi[row+n]];
            for(int a=0;a<3;a++)
            {
                double factor=t[a]*(1-t[a])*span[a]*span[a]/2;if(factor==0)continue;
                for(int side=-1;side<=1;side+=2)
                {
                    var nb=(int[])target.Clone();nb[a]+=side;int other=Air(nb);if(other<0)continue;
                    double g=factor/(Widths[a][target[a]]*.5*(Widths[a][target[a]]+Widths[a][nb[a]]));
                    long o=(long)other*Modes;
                    for(int n=0;n<Modes;n++)scratch[n]-=g*(halves[psi[o+n]]-halves[psi[row+n]]);
                }
            }
            for(int n=0;n<Modes;n++)output[n]+=w*scratch[n];
        }
        return true;
    }

    // Line of sight through the bank's air cells (a voxel walk along the segment): the direct sound the path method
    // renders, in the bank's own geometry. It differs from the renderer's opaque test only where the segment grazes a
    // surface inside one cell.
    public bool LineOfSight(Vector3 from,Vector3 to)
    {
        var c=new int[3];if(!CellOf(from,c) || Air(c)<0)return false;
        var end=new int[3];if(!CellOf(to,end) || Air(end)<0)return false;
        var dir=new double[3];var next=new double[3];var stepT=new int[3];
        for(int a=0;a<3;a++)
        {
            dir[a]=to[a]-from[a];stepT[a]=dir[a]>0?1:dir[a]<0?-1:0;
            next[a]=stepT[a]==0?double.PositiveInfinity:((stepT[a]>0?Edges[a][c[a]+1]:Edges[a][c[a]])-from[a])/dir[a];
        }
        while(c[0]!=end[0] || c[1]!=end[1] || c[2]!=end[2])
        {
            int a=next[0]<=next[1]?(next[0]<=next[2]?0:2):(next[1]<=next[2]?1:2);
            if(next[a]>1)break;   // rounding at the end cell's face
            c[a]+=stepT[a];if(Air(c)<0)return false;
            next[a]=((stepT[a]>0?Edges[a][c[a]+1]:Edges[a][c[a]])-from[a])/dir[a];
        }
        return true;
    }
}

internal sealed class RoomModes
{
    public const double SchroederOverlap=3;            // Schroeder's criterion: three modes per modal bandwidth
    public const double StatisticalAbsorptionMax=.95122, ZetaAtMax=1.56692;   // tools/fdtd_wave.py
    public static readonly double[] Bands={125,250,500,1000,2000,4000,8000,16000};
    public readonly RoomModeBank Bank;
    public readonly double Speed,Reference,SchroederHz,TopHz;
    public readonly double[] Delta,Share;             // per mode (1/s); per band, g_b
    public readonly int OutputRate,Rate,Ratio,Length,FftSize,Radius,KernelSamples;
    public readonly double[] Weights;                 // W_m at the decimated FFT's bins 0..N/2
    public readonly double[] Interpolation;           // [n % Ratio][2 Radius]: the windowed sinc's normalised taps j0 - Radius + 1 .. j0 + Radius
    readonly double[] decay,spread,omega;             // per mode: decay rate (overdamped: the slower pole), overdamped: the poles' difference; damped w
    readonly Binding binding;
    readonly bool[] overdamped;
    double[] probeS,probeR,scratch,h,re,im;
    public string Disabled {get;private set;}          // why the bank never applies here (a mismatched scene), or null

    // The configuration a bank was baked in (sidecar <bank>.json): the scene's geometry hash, and the pose of every
    // object at the bake (e.g. the Church's doors closed, frame 0 of its take). Equal poses are exact: the bank holds a
    // closed room only while nothing that bounds it has moved.
    [Serializable] public sealed class Binding
    {
        public string geometryHash;
        public Pose[] poses;
        [Serializable] public sealed class Pose { public int index; public float[] position,rotation; public bool active; }
    }

    // admittance[m][b]: each bank material's real normalised admittance at the eight band centres.
    public RoomModes(RoomModeBank bank,double[][] admittance,double speed,double reference,int outputRate,int kernelSamples,int radius,Binding binding=null)
    {
        if(admittance.Length!=bank.Materials.Length)throw new ArgumentException("One admittance curve per bank material.");
        Bank=bank;Speed=speed;Reference=reference;OutputRate=outputRate;KernelSamples=kernelSamples;Radius=Math.Max(1,radius);this.binding=binding;
        Delta=Damping(bank,admittance,speed);
        SchroederHz=SchroederFrequency(bank.K,Delta,speed);
        if(!(SchroederHz>0))throw new InvalidDataException("The room mode bank does not reach the Schroeder frequency; bake to a higher fmax.");
        TopHz=bank.TopHz(speed);
        if(TopHz<SchroederHz*Math.Sqrt(2)*(1-1e-3))throw new InvalidDataException($"The bank ends at {TopHz:F0} Hz, below the crossover's top {SchroederHz*Math.Sqrt(2):F0} Hz.");
        Share=PathShare(SchroederHz);
        // The smallest divisor of the output rate at least four times the top mode (the output rate itself if lower).
        Rate=outputRate;
        for(int d=1;d<=outputRate;d++)if(outputRate%d==0 && d>=4*TopHz){Rate=d;break;}
        Ratio=outputRate/Rate;
        Length=checked((kernelSamples+Ratio-1)/Ratio+Radius+1);
        FftSize=1;while(FftSize<2*Length)FftSize=checked(FftSize*2);
        Weights=new double[FftSize/2+1];
        for(int k=0;k<Weights.Length;k++)Weights[k]=Crossover((double)k*Rate/FftSize,SchroederHz);
        Interpolation=new double[Ratio*2*Radius];
        for(int phase=0;phase<Ratio;phase++)
        {
            double frac=(double)phase/Ratio,sum=0;
            for(int i=0;i<2*Radius;i++){double u=Radius-1-i+frac,w=Math.Abs(u)<Radius?Sinc(u)*Sinc(u/Radius):0;Interpolation[phase*2*Radius+i]=w;sum+=w;}
            for(int i=0;i<2*Radius;i++)Interpolation[phase*2*Radius+i]/=sum;
        }
        int N=bank.Modes;decay=new double[N];spread=new double[N];omega=new double[N];overdamped=new bool[N];
        // The static (constant) mode has w_n ~ 0 < delta: overdamped, with a slow pole s1 ~ w_n^2 / (2 delta) ~ 0, a
        // pressure step that never decays. A finite kernel cannot hold it; its slow pole is raised to the decay at which
        // the kernel's float32 values reach their precision at the kernel's end, ln(2^24) / T (about 0.02 dB at 20 Hz
        // for a 2 s kernel), so the step dies inside the kernel instead of ending in a click. Its response
        // (e^{-s1 t} - e^{-s2 t}) / (s2 - s1) = e^{-s t} (1 - e^{-d t}) / d, s the slower pole and d >= 0 the poles' difference,
        // is evaluated as e^{-s t} t phi(d t), phi(x) = (1 - e^{-x}) / x, which stays exact when the raised pole meets the other.
        double floor=24*Math.Log(2)/((double)kernelSamples/outputRate);
        for(int n=0;n<N;n++)
        {
            double w=bank.K[n]*speed,d=Delta[n];
            if(w>d){decay[n]=d;omega[n]=Math.Sqrt(w*w-d*d);}
            else{double r=Math.Sqrt(d*d-w*w),s1=Math.Max(d-r,floor),s2=d+r;overdamped[n]=true;decay[n]=Math.Min(s1,s2);spread[n]=Math.Abs(s2-s1);}
        }
        if(binding!=null && binding.poses==null)binding.poses=new Binding.Pose[0];
    }

    // The scene's bank (settings.roomModesBank), or null when off. A relative path is read from StreamingAssets (how a
    // build ships a bank: StreamingAssets/<file>.roommodes.bytes with its sidecar). Materials: the scene's, by name; a
    // bank material "opening" (faces onto unglazed openings, church_modes.py) loses as an open window, absorption 1.
    // An opening is not a locally reacting wall, so it enters the first-order damping through the admittance whose
    // diffuse-field loss is that absorption: averaged over the modes (c/2) beta S is Sabine's c alpha A / (8 V), so
    // beta = alpha / 8 = 1/8.
    public static RoomModes ForScene(AcousticTake.Scene scene,TakeRender.Settings settings,int rate,int kernelSamples)
    {
        if(string.IsNullOrEmpty(settings.roomModesBank))return null;
        if(scene.dimension!=AcousticTake.Dimension.Spatial3D)throw new InvalidOperationException("Room modes need a 3D scene.");
        string path=Path.IsPathRooted(settings.roomModesBank)?settings.roomModesBank:Path.Combine(Application.streamingAssetsPath,settings.roomModesBank);
        var bank=RoomModeBank.Load(path);
        Binding binding=null;string sidecar=path+".json";
        if(File.Exists(sidecar))binding=JsonUtility.FromJson<Binding>(File.ReadAllText(sidecar));
        var admittance=new double[bank.Materials.Length][];
        for(int m=0;m<admittance.Length;m++)
        {
            string name=bank.Materials[m];
            var definition=scene.materials==null?null:Array.Find(scene.materials,x=>x.name==name);
            if(definition!=null)admittance[m]=MaterialAdmittance(definition);
            else if(name=="opening"){admittance[m]=new double[8];Array.Fill(admittance[m],1.0/8);}
            else throw new InvalidDataException($"Room mode bank material '{name}' is not in the scene.");
        }
        var modes=new RoomModes(bank,admittance,settings.speedOfSound,settings.referenceDistance,rate,kernelSamples,settings.sincRadius,binding);
        if(binding!=null && !string.IsNullOrEmpty(binding.geometryHash) && !string.IsNullOrEmpty(scene.geometryHash) && binding.geometryHash!=scene.geometryHash)
        {
            modes.Disabled=$"room modes off: the bank was baked for geometry {binding.geometryHash}, the scene's is {scene.geometryHash}";
            Debug.LogWarning(modes.Disabled);
        }
        return modes;
    }

    // Whether the bank holds the scene's room now: its configuration's objects at their baked poses.
    public bool Applies(AcousticTake.State[] states)
    {
        if(Disabled!=null)return false;
        if(binding==null)return true;
        foreach(var p in binding.poses)
        {
            if(p.index<0 || p.index>=states.Length)return false;
            var s=states[p.index];bool active=(s.flags&AcousticTake.StateFlags.Active)!=0;
            if(active!=p.active)return false;
            if(s.position.x!=p.position[0] || s.position.y!=p.position[1] || s.position.z!=p.position[2])return false;
            if(s.rotation.x!=p.rotation[0] || s.rotation.y!=p.rotation[1] || s.rotation.z!=p.rotation[2] || s.rotation.w!=p.rotation[3])return false;
        }
        return true;
    }

    // One pair's modal amplitudes A_n = 4 pi c^2 psi_n(s) psi_n(r) (times the reference distance: free field
    // reference / r) and the direct sound the path method renders, gain reference / max(r, minimum) at r/c if the bank's
    // air sees it. False when either end is outside the bank's air.
    public bool Pair(Vector3 source,Vector3 listener,double minimumDistance,double[] amplitudes,out double direct,out double delay)
    {
        direct=delay=0;
        probeS??=new double[Bank.Modes];probeR??=new double[Bank.Modes];scratch??=new double[Bank.Modes];
        if(!Bank.Probe(source,probeS,scratch) || !Bank.Probe(listener,probeR,scratch))return false;
        double scale=4*Math.PI*Speed*Speed*Reference;
        for(int n=0;n<Bank.Modes;n++)amplitudes[n]=scale*probeS[n]*probeR[n];
        double r=Vector3.Distance(source,listener);delay=r/Speed;
        direct=Bank.LineOfSight(source,listener)?Reference/Math.Max(r,minimumDistance):0;
        return true;
    }

    // The pair's modal difference dh = W_m (h_modal - h_direct) at the output rate, as kernel samples (dh(t) / rate at
    // t = n / rate, n < output.Length <= KernelSamples): the CPU reference of GpuAudio.Modes.
    public void Synthesize(double[] amplitudes,double direct,double delay,double[] output)
    {
        int N=FftSize;
        re??=new double[N];im??=new double[N];
        ModalImpulse(amplitudes);
        Array.Clear(re,0,N);Array.Clear(im,0,N);Array.Copy(h,re,Length);
        Fft(re,im,false);
        double cycles=delay*Rate/N;   // direct pulse phase per bin, cycles
        for(int k=0;k<N;k++)
        {
            int kk=k<=N/2?k:k-N;double w=Weights[Math.Abs(kk)],phase=-2*Math.PI*Fraction(cycles*kk);
            double xr=re[k]-Rate*direct*Math.Cos(phase),xi=im[k]-Rate*direct*Math.Sin(phase);
            re[k]=w*xr;im[k]=w*xi;
        }
        Fft(re,im,true);
        for(int n=0;n<output.Length;n++)output[n]=Interpolate(re,n)/OutputRate;
    }
    // h_modal at the decimated samples j < Length (the synthesis stage).
    internal double[] ModalImpulse(double[] amplitudes)
    {
        h??=new double[Length];
        Array.Clear(h,0,h.Length);
        for(int n=0;n<Bank.Modes;n++)
        {
            double a=amplitudes[n];if(a==0)continue;
            if(overdamped[n])
            {
                for(int j=0;j<Length;j++){double t=(double)j/Rate;h[j]+=a*Math.Exp(-decay[n]*t)*t*Phi(spread[n]*t);}
            }
            else
            {
                // e^{-delta t} sin(w_d t) / w_d, by rotation of exp((-delta + i w_d) / rate)
                double r=Math.Exp(-decay[n]/Rate),zr=r*Math.Cos(omega[n]/Rate),zi=r*Math.Sin(omega[n]/Rate),vr=a/omega[n],vi=0;
                for(int j=0;j<Length;j++){h[j]+=vi;double t=vr*zr-vi*zi;vi=vr*zi+vi*zr;vr=t;}
            }
        }
        return h;
    }
    static double Fraction(double x)=>x-Math.Floor(x);
    // (1 - e^{-x}) / x for x >= 0, without cancellation (its series below 0.01: error under 1e-8 relative; the GPU's too).
    internal static double Phi(double x)=>x<.01?1-x/2+x*x/6:(1-Math.Exp(-x))/x;
    // dh(n / rate) from the decimated samples (periodic over the FFT), by the renderer's windowed sinc, normalised.
    double Interpolate(double[] x,int n)
    {
        int j0=n/Ratio,first=j0-Radius+1,N=FftSize,taps=2*Radius,o=(n%Ratio)*taps;double sum=0;
        for(int i=0;i<taps;i++)sum+=Interpolation[o+i]*x[((first+i)%N+N)%N];
        return sum;
    }
    static double Sinc(double x)=>Math.Abs(x)<1e-12?1:Math.Sin(Math.PI*x)/(Math.PI*x);

    // GPU tables (GpuAudio.Modes): per mode (per-sample decay (overdamped: of the slower pole), the poles' difference per
    // sample (overdamped) or 0, 1 / w_d (overdamped: 1 / rate, as t = j / rate), 1 if overdamped), and the oscillation's
    // phase step as 32-bit fixed-point cycles.
    public Vector4[] ModeTable()
    {
        var table=new Vector4[Bank.Modes];
        for(int n=0;n<table.Length;n++)table[n]=new Vector4((float)(decay[n]/Rate),overdamped[n]?(float)(spread[n]/Rate):0,(float)(overdamped[n]?1.0/Rate:1/omega[n]),overdamped[n]?1:0);
        return table;
    }
    public uint[] ModePhases()
    {
        var phases=new uint[Bank.Modes];
        for(int n=0;n<phases.Length;n++)phases[n]=overdamped[n]?0:Fixed(omega[n]/(2*Math.PI*Rate));
        return phases;
    }
    // Cycles as 32-bit fixed point (the fraction): products with a sample index wrap exactly modulo one cycle.
    public static uint Fixed(double cycles)=>unchecked((uint)(ulong)Math.Round(Fraction(cycles)*4294967296.0));
    public uint DirectPhase(double delay)=>Fixed(delay*Rate/FftSize);

    // ---------------------------------------------------------------- the method's pieces (room_modes.py)

    // Paris's random-incidence absorption of a locally reacting wall with real impedance zeta, inverted on the hard
    // branch (zeta >= 1.567, where it falls monotonically): the normalised admittance 1 / zeta. Absorption past the
    // branch's maximum (0.951) is held there.
    public static double AdmittanceFromAbsorption(double alpha)
    {
        if(!(alpha>0))return 0;
        alpha=Math.Min(alpha,StatisticalAbsorptionMax);
        double lo=Math.Log(ZetaAtMax),hi=Math.Log(1e9);
        for(int i=0;i<200;i++){double mid=.5*(lo+hi);if(StatisticalAbsorption(Math.Exp(mid))>alpha)lo=mid;else hi=mid;}
        return 1/Math.Exp(.5*(lo+hi));
    }
    static double StatisticalAbsorption(double z)=>8/(z*z)*(z+1-1/(1+z)-2*Math.Log(1+z));
    // A scene material's admittance at the band centres: its local admittance if it has one, else from the energy that
    // leaves through it, absorption + transmission (curves linear in Hz, clamped: linear-hz-clamped-v1).
    public static double[] MaterialAdmittance(AcousticTake.MaterialDefinition m)
    {
        var result=new double[8];
        for(int b=0;b<8;b++)
        {
            if(m.HasAdmittance){result[b]=Curve(m.frequencies,m.admittanceReal,Bands[b]);continue;}
            double loss=Curve(m.frequencies,m.absorption,Bands[b])+Curve(m.frequencies,m.transmission,Bands[b]);
            result[b]=AdmittanceFromAbsorption(loss);
        }
        return result;
    }
    static double Curve(float[] hz,float[] values,double f)
    {
        if(values==null || values.Length==0)return 0;
        if(hz==null || hz.Length!=values.Length || values.Length==1)return values[0];
        if(f<=hz[0])return values[0];
        for(int i=1;i<hz.Length;i++)if(f<=hz[i])return values[i-1]+(values[i]-values[i-1])*(f-hz[i-1])/(hz[i]-hz[i-1]);
        return values[values.Length-1];
    }
    // delta_n = (c / 2) sum_m beta_m(f_n) S_nm, beta linear in log2 f between the band centres, held outside.
    public static double[] Damping(RoomModeBank bank,double[][] admittance,double speed)
    {
        var delta=new double[bank.Modes];int M=bank.Materials.Length;
        for(int n=0;n<bank.Modes;n++)
        {
            double f=bank.K[n]*speed/(2*Math.PI),x=Math.Log(Math.Max(f,1e-3)/Bands[0],2),sum=0;
            int i=(int)Math.Floor(x);double t=x-i;
            for(int m=0;m<M;m++)
            {
                var a=admittance[m];double beta=x<=0?a[0]:x>=7?a[7]:a[i]+(a[i+1]-a[i])*t;
                sum+=beta*bank.Surface[n*M+m];
            }
            delta[n]=.5*speed*sum;
        }
        return delta;
    }
    // The lowest frequency at which the modal overlap reaches three: modes per Hz times their mean half-power bandwidth
    // delta / pi, both over f +- f (sqrt2 - 1/sqrt2) / 2 (room_modes.schroeder_frequency, its 400-point grid).
    public static double SchroederFrequency(double[] k,double[] delta,double speed)
    {
        var f=new double[k.Length];double top=0,first=double.MaxValue;
        for(int n=0;n<k.Length;n++){f[n]=k[n]*speed/(2*Math.PI);top=Math.Max(top,f[n]);if(k[n]*speed>delta[n])first=Math.Min(first,f[n]);}
        // (from the lowest oscillating mode up, as room_modes.schroeder_frequency: the constant mode, overdamped, has a
        // wavenumber of rounding size and would stretch the grid over decades)
        double half=.5*(Math.Sqrt(2)-1/Math.Sqrt(2)),lo=Math.Log(first),hi=Math.Log(top/(1+half));
        for(int g=0;g<400;g++)
        {
            double fc=Math.Exp(lo+(hi-lo)*g/399.0),sum=0;int count=0;
            for(int n=0;n<f.Length;n++)if(f[n]>=fc*(1-half) && f[n]<fc*(1+half)){count++;sum+=delta[n];}
            if(count<2)continue;
            if(count/(2*half*fc)*(sum/count/Math.PI)>=SchroederOverlap)return fc;
        }
        return double.NaN;
    }
    // Modal weight W_m (amplitude): 1 below fs / sqrt2, 0 above fs sqrt2, cos^2 (pi/2 x) in power between.
    public static double Crossover(double f,double fs)
    {
        double x=Math.Clamp(Math.Log(Math.Max(f,1e-30)/fs,2)+.5,0,1);
        return Math.Cos(.5*Math.PI*x);
    }
    // g_b: the share of band b's filter energy left to the path method's incoherent part, W_g^2 = 1 - W_m^2.
    public static double[] PathShare(double fs)
    {
        var g=new double[Bands.Length];const int points=20001;
        for(int b=0;b<g.Length;b++)
        {
            double num=0,den=0;
            for(int i=0;i<points;i++)
            {
                double x=-1+2.0*i/(points-1),f=Bands[b]*Math.Pow(2,x),c=Math.Cos(.5*Math.PI*x),w=c*c*f*(i==0||i==points-1?.5:1),m=Crossover(f,fs);
                num+=w*(1-m*m);den+=w;
            }
            g[b]=num/den;
        }
        return g;
    }

    // In-place radix-2 FFT (exp(-i...) forward; the inverse includes 1/N).
    public static void Fft(double[] re,double[] im,bool inverse)
    {
        int n=re.Length;
        for(int i=1,j=0;i<n;i++)
        {
            int bit=n>>1;for(;(j&bit)!=0;bit>>=1)j^=bit;j^=bit;
            if(i<j){(re[i],re[j])=(re[j],re[i]);(im[i],im[j])=(im[j],im[i]);}
        }
        for(int length=2;length<=n;length<<=1)
        {
            double angle=2*Math.PI/length*(inverse?1:-1);
            for(int k=0;k<length/2;k++)
            {
                double cr=Math.Cos(angle*k),ci=Math.Sin(angle*k);
                for(int i=0;i<n;i+=length)
                {
                    int a=i+k,b=a+length/2;
                    double tr=re[b]*cr-im[b]*ci,ti=re[b]*ci+im[b]*cr;
                    re[b]=re[a]-tr;im[b]=im[a]-ti;re[a]+=tr;im[a]+=ti;
                }
            }
        }
        if(inverse)for(int i=0;i<n;i++){re[i]/=n;im[i]/=n;}
    }
}
