using System;
using System.Collections.Generic;
using NUnit.Framework;
using UnityEngine;

// Analytic controls of the path method (the earlier pressure and energy methods' values are in
// docs/benchmarks/methods-2026-11-06), and the room fixtures the path checks share.
public partial class TakeChecks
{
    TakeRender.Settings ControlSettings(TakeRender.Method method, int rays, float horizon)
    {
        var settings = SpectralSettings();
        settings.method = method; settings.methodRays = rays; settings.speedOfSound = 343; settings.minimumDistance = .01f;
        settings.maximumDelaySeconds = horizon; settings.diffractionFactor = 0; settings.airAbsorption = false; settings.useBvh = true;
        settings.seed = 11;
        // Opt-in: the Path method's beams-only mode in every check (README part 20 runs them so).
        settings.pathBeamsOnly = Environment.GetEnvironmentVariable("ACOUSTIC_PATH_BEAMS_ONLY") == "1";
        return settings;
    }
    // Band energies per 1 ms bin of the latest solve, direction-summed.
    static double[] ControlEnergy(TransportSolver solver, int band)
    {
        var m = solver.Method; var energy = new double[m.Bins];
        for (int k = 0; k < m.Bins; k++) for (int d = 0; d < 6; d++) energy[k] += m.Energy(d, band, k);
        return energy;
    }
    static double Window(double[] values, int from, int to) { double s = 0; for (int k = Math.Max(0, from); k < Math.Min(values.Length, to); k++) s += values[k]; return s; }
    static AcousticTake.MaterialDefinition RigidControl(float absorption = 0, float scattering = 0) =>
        new AcousticTake.MaterialDefinition {name = "control", frequencies = new[] {0f}, absorption = new[] {absorption}, scattering = new[] {scattering}, transmission = new[] {0f}};

    // Free field: the direct sound arrives at r/c with energy 1/r^2 in every band.
    // Slow like every method check: compiling the method shader costs the shared
    // runner shader-compiler memory, which the quick tier cannot spare.
    [TestCase(false), TestCase(true), Category("GPU"), Category("Methods"), Category("Slow"), Timeout(1800000)]
    public void MethodsMatchTheFreeField(bool persistent)
    {
        var scene = Scene(); scene.materials = new[] {RigidControl()};
        var state = States(); state[0].position = Vector3.zero; state[1].position = new Vector3(3, .4f, -1.2f);
        var settings = ControlSettings(TakeRender.Method.Path, 256, .1f); settings.useBvh = false; PersistentControl(settings, persistent, 2);
        using var solver = new TransportSolver(scene, new AcousticTake.Primitive[0], settings, RequireGpu(), sampleRate: 8000);
        solver.Solve(state, new TakeRender.DirectPath[1]); AssertNoOverflow(solver);
        double r = state[1].position.magnitude; int bin = (int)Math.Floor(r / 343 / .001);
        for (int b = 0; b < 8; b++)
        {
            var e = ControlEnergy(solver, b); double total = Window(e, 0, e.Length);
            TestContext.WriteLine($"band {b}: total {total * r * r:F6} x 1/r^2, bin {bin} holds {e[bin] / total:F3}");
            Assert.That(total * r * r, Is.EqualTo(1).Within(1e-6));
            Assert.That(Window(e, bin - 4, bin + 5) / total, Is.GreaterThan(.999));
            Assert.That(e[bin] / total, Is.GreaterThan(.999));
        }
    }

    // One rigid plane: direct sound plus its image, nothing else (diffraction off).
    [TestCase(false), TestCase(true), Category("GPU"), Category("Methods"), Category("Slow"), Timeout(1800000)]
    public void MethodsReproduceARigidPlaneImage(bool persistent)
    {
        var scene = Scene(2); scene.listeners = new[] {1}; scene.materials = new[] {RigidControl()};
        var a = new Vector3(-100, 0, -100); var b = new Vector3(100, 0, -100); var c = new Vector3(100, 0, 100); var d = new Vector3(-100, 0, 100);
        var geometry = new[] {new AcousticTake.Primitive {objectIndex = 2, a = a, b = c, c = b}, new AcousticTake.Primitive {objectIndex = 2, a = a, b = d, c = c}};
        scene.objects[2].primitiveCount = geometry.Length;
        var state = States(2); state[0].position = new Vector3(0, 1.5f, 0); state[1].position = new Vector3(4, 1.2f, .5f);
        state[2].position = Vector3.zero; state[2].flags = AcousticTake.StateFlags.Active | AcousticTake.StateFlags.Geometry;
        var settings = ControlSettings(TakeRender.Method.Path, 1 << 18, .1f);
        settings.useBvh = false; // two triangles; spares the BVH kernel's compile
        PersistentControl(settings, persistent, 1);
        using var solver = new TransportSolver(scene, geometry, settings, RequireGpu(), sampleRate: 8000);
        solver.Solve(state, new TakeRender.DirectPath[1]); AssertNoOverflow(solver);
        var image = new Vector3(0, -1.5f, 0);
        var m = solver.Method; var counts = new System.Text.StringBuilder();
        for (int k = 0; k < m.TimeBins.Count; k++) if (m.Counts[k] > 0) counts.Append($"{k}:{m.Counts[k]} ");
        TestContext.WriteLine($"contributions per bin {counts}; rejected deposits {m.Counter(6)}");

        double r = (state[1].position - state[0].position).magnitude, ri = (state[1].position - image).magnitude;
        int bi = (int)Math.Floor(ri / 343 / .001);
        var e = ControlEnergy(solver, 2); double held = Window(e, bi, bi + 1), total = Window(e, 0, e.Length);
        TestContext.WriteLine($"image {held * ri * ri:F4} x 1/ri^2, total {total / (1 / (r * r) + 1 / (ri * ri)):F4} x expected, chains {solver.Method.Counter(8)}");
        Assert.That(held * ri * ri, Is.EqualTo(1).Within(.15));
        Assert.That(total / (1 / (r * r) + 1 / (ri * ri)), Is.EqualTo(1).Within(.08));

    }

    // Rigid-walled shoebox (alpha = 0.1, no scattering): the energy histogram is the
    // image lattice; the pressure trains, band-filtered, match the lattice's arrivals.
    [TestCase(false), TestCase(true), Category("GPU"), Category("Methods"), Category("Slow"), Timeout(1800000)]
    public void MethodsReproduceTheShoeboxImageLattice(bool persistent)
    {
        var size = new Vector3(5, 4, 3); var source = new Vector3(-1.1f, .4f, .3f); var listener = new Vector3(1.3f, -.6f, -.2f);
        LateRoom(AcousticTake.Dimension.Spatial3D, size, LateMaterial(.1f, 0), source, listener, out var scene, out var geometry, out var state);
        var settings = ControlSettings(TakeRender.Method.Path, 1 << 16, .2f);
        // Every path lives to the horizon here, so the 20 ms windows test the connections at full
        // sampling density (survival in time is checked by PathMethodLeaksThroughWalls...).
        settings.pathSurvivalSeconds = 0; PersistentControl(settings, persistent, 8);
        using var solver = new TransportSolver(scene, geometry, settings, RequireGpu(), sampleRate: 8000);
        solver.Solve(state, new TakeRender.DirectPath[1]); AssertNoOverflow(solver);
        var corner = size * .5f; int bins = solver.Method.Bins;
        var e = ControlEnergy(solver, 0); var want = ImageLattice(size, source + corner, listener + corner, .1, .001, bins, 0);
        double worst = 0;
        for (int w = 0; w + 20 <= bins; w += 20)
        {
            double ratio = Window(e, w, w + 20) / Window(want, w, w + 20);
            worst = Math.Max(worst, Math.Abs(ratio - 1));
            TestContext.WriteLine($"{w}-{w + 20} ms: {ratio:F4}");
        }
        double total = Window(e, 0, bins) / Window(want, 0, bins);
        TestContext.WriteLine($"total {total:F4}, worst 20 ms window {worst:F4}, chains {solver.Method.Counter(8)}");
        Assert.That(total, Is.EqualTo(1).Within(.05));
        Assert.That(worst, Is.LessThan(.15));

    }

    // Rigid half-plane, listener in the shadow: the path method's UTD band energy matches the
    // independent BTM band energy within 1.5 dB.
    [TestCase(false), TestCase(true), Category("GPU"), Category("Methods"), Category("Slow"), Timeout(1800000)]
    public void MethodsMatchTheHalfPlaneReference(bool persistent)
    {
        KnifeEdgeFixture(false, out var scene, out var geometry, out var state);
        var settings = ControlSettings(TakeRender.Method.Path, 1 << 16, .3f); settings.diffractionFactor = 1; PersistentControl(settings, persistent, 2);
        using var solver = new TransportSolver(scene, geometry, settings, RequireGpu(), sampleRate: 8000);
        solver.Solve(state, new TakeRender.DirectPath[1]); AssertNoOverflow(solver);
        // Independent BTM band energies of this fixture (docs/benchmarks/methods-2026-11-06/
        // btm_reference.py: float64 edge integral, the comparison's band filters).
        double[] btm = {0.009199, 0.006599, 0.004427, 0.002758, 0.001590};
        for (int band = 1; band <= 4; band++)
        {
            double have = Window(ControlEnergy(solver, band), 0, 300);
            TestContext.WriteLine($"band {MethodBands.Centers[band]} Hz: UTD energy {have:G5}, BTM {btm[band]:G5}, {10 * Math.Log10(have / btm[band]):F2} dB");
            Assert.That(Math.Abs(10 * Math.Log10(have / btm[band])), Is.LessThan(1.5));
        }
    }

    // Rigid cube (2 m), source and listener on opposite sides: only double diffraction
    // (DD over parallel edges) and longer chains reach the listener; the UTD energy (DD
    // connections) is checked on single walls against btm_reference.py.
    [TestCase(false), TestCase(true), Category("GPU"), Category("Methods"), Category("Slow"), Timeout(1800000)]
    public void MethodsReachTheFarSideOfASolid(bool persistent)
    {
        KnifeEdgeFixture(false, out var scene, out _, out var state);
        var geometry = TransportBox(scene.dimension, 2, 0, 1); scene.objects[2].primitiveCount = geometry.Length;
        state[0].position = Vector3.left * 3; state[1].position = Vector3.right * 3;
        var settings = ControlSettings(TakeRender.Method.Path, 1 << 17, .3f); settings.diffractionFactor = 1; PersistentControl(settings, persistent, 2);
        var solver = new TransportSolver(scene, geometry, settings, RequireGpu(), sampleRate: 8000);
        try
        {
            solver.Solve(state, new TakeRender.DirectPath[1]); AssertNoOverflow(solver);
            TestContext.WriteLine($"edges {solver.Method.EdgeCount}, edge connections {solver.Method.Counter(10)}, edge events {solver.Method.Counter(12)}");
            for (int depth = 1; depth < 6; depth++)
            {
                var d = solver.Method.Depth(0, depth);
                if (d.count > 0) TestContext.WriteLine($"  depth {depth}: {d.count} contributions, mean |c| {d.sum / d.count:G4}, max {d.max:G4}, sum|c|/rays {d.sum / settings.methodRays:G4}");
            }
            // The cube's four DD paths have equal delays: BTM adds them coherently (16x one
            // path's energy), any energy method incoherently (4x), 6 dB apart. The
            // control therefore uses walls with a single dominant DD path over the top,
            // against independent BTM band energies (btm_reference.py). Ray-based double
            // diffraction needs the edges to be far apart in wavelengths: asserted for a 2 m
            // wall (kd = 18-73 at 500-2000 Hz); a 0.25 m wall (the suite's walls) is measured.
            solver.Dispose();
            var cases = new[] {(width: 2f, btm: new[] {1.620450e-3, 7.792134e-4, 3.207570e-4, 1.120553e-4, 3.379192e-5}),
                               (width: .25f, btm: new[] {2.606818e-3, 1.510434e-3, 7.953154e-4, 3.818016e-4, 1.673416e-4})};
            foreach (var wallCase in cases)
            {
                var wall = TransportBox(scene.dimension, 2, 0, 1);
                for (int i = 0; i < wall.Length; i++)
                {
                    var p = wall[i]; Vector3 size = new Vector3(wallCase.width / 2, 5, 30), offset = new Vector3(0, -4, 0);
                    p.a = Vector3.Scale(p.a, size) + offset; p.b = Vector3.Scale(p.b, size) + offset; p.c = Vector3.Scale(p.c, size) + offset; wall[i] = p;
                }
                using var wallSolver = new TransportSolver(scene, wall, settings, RequireGpu(), sampleRate: 8000);
                wallSolver.Solve(state, new TakeRender.DirectPath[1]);
                for (int band = 0; band <= 4; band++)
                {
                    double have = Window(ControlEnergy(wallSolver, band), 0, 300), error = 10 * Math.Log10(have / wallCase.btm[band]);
                    TestContext.WriteLine($"{wallCase.width} m wall, band {MethodBands.Centers[band]} Hz: UTD energy {have:G5}, BTM {wallCase.btm[band]:G5}, {error:F2} dB");
                    if (wallCase.width > 1 && band >= 2) Assert.That(Math.Abs(error), Is.LessThan(2), "UTD double diffraction over a 2 m wall agrees with BTM.");
                }
            }
        }
        finally { solver.Dispose(); }
    }
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
    static double Sum(double[] values){double s=0;foreach(var v in values)s+=v;return s;}
}
