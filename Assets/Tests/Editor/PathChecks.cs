using System;
using System.Collections.Generic;
using NUnit.Framework;
using UnityEngine;

// Checks of the Path method (one energy path tracer, MethodPath.hlsl) and of its audio
// synthesis (PathReverb); docs/benchmarks/path-tracer-2026-11-12 reports the values. The analytic
// controls shared with the other methods (free field, rigid plane, image lattice, half-plane,
// far side of a solid) are MethodChecks cases with TakeRender.Method.Path.
public partial class TakeChecks
{
    // Fully diffusing shoebox: the path tracer decays at the Eyring rate.
    [Test, Category("GPU"), Category("Methods"), Category("Slow"), Timeout(1800000)]
    public void PathMethodDecaysAtTheEyringRate()
    {
        var size = new Vector3(6, 5, 4); const float absorption = .2f;
        LateRoom(AcousticTake.Dimension.Spatial3D, size, LateMaterial(absorption, 1), new Vector3(-1.5f, .5f, .4f), new Vector3(1.4f, -.7f, -.3f),
                 out var scene, out var geometry, out var state);
        var settings = ControlSettings(TakeRender.Method.Path, 8192, 1);
        using var solver = new TransportSolver(scene, geometry, settings, RequireGpu(), sampleRate: 8000);
        solver.Solve(state, new TakeRender.DirectPath[1]);
        var e = ControlEnergy(solver, 0);
        var curve = new double[e.Length]; double tail = 0;
        for (int k = e.Length - 1; k >= 0; k--) { tail += e[k]; curve[k] = tail; }
        double x = 0, y = 0, xx = 0, xy = 0; int n = 0;
        for (int k = 0; k < e.Length; k++)
        {
            double level = 10 * Math.Log10(curve[k] / curve[0]); if (level > -5 || level < -25) continue;
            double t = (k + .5) * .001; x += t; y += level; xx += t * t; xy += t * level; n++;
        }
        double t60 = -60 / ((n * xy - x * y) / (n * xx - x * x));
        double volume = size.x * size.y * size.z, surface = 2 * (size.x * size.y + size.y * size.z + size.z * size.x);
        double eyring = .161 * volume / (-surface * Math.Log(1 - absorption));
        TestContext.WriteLine($"T60 {t60:F3} s, Eyring {eyring:F3} s, counted contributions {Sum(solver.Method.Counts)}");
        Assert.That(t60 / eyring, Is.EqualTo(1).Within(.15));
    }
    // Sound leaking out of a scattering room through its walls (t = 1e-3), listener outside: the path
    // method's connections through walls (traced with probability = the first wall's transmission,
    // weighted back) give the energy method's leaked energy. Both are estimates; the tolerance covers
    // their seed noise.
    [Test, Category("GPU"), Category("Methods"), Category("Slow"), Timeout(1800000)]
    public void PathMethodLeaksThroughWallsLikeTheEnergyMethod()
    {
        var size = new Vector3(6, 5, 4);
        double Leak(TakeRender.Method method, int seed, float survival = -1, int flags = 0)
        {
            LateRoom(AcousticTake.Dimension.Spatial3D, size, LateMaterial(.1f, .5f, 1e-3f), new Vector3(-1.5f, .5f, .4f), new Vector3(3.8f, -.3f, .2f),
                     out var scene, out var geometry, out var state);
            var settings = ControlSettings(method, 1 << 16, .4f); settings.seed = seed; settings.specularDetectionRadius = .5f;
            if (survival >= 0) settings.pathSurvivalSeconds = survival;
            using var solver = new TransportSolver(scene, geometry, settings, RequireGpu(), sampleRate: 8000);
            if (method == TakeRender.Method.Path) solver.Method.ProfileFlags = flags;
            solver.Solve(state, new TakeRender.DirectPath[1]);
            double direct = (new Vector3(3.8f, -.3f, .2f) - new Vector3(-1.5f, .5f, .4f)).magnitude / 343;
            return Window(ControlEnergy(solver, 2), (int)Math.Ceiling(direct * 1000) + 5, int.MaxValue);
        }
        double path = 0, all = 0, lifelong = 0, energy = 0;
        for (int seed = 0; seed < 2; seed++)
        {
            path += Leak(TakeRender.Method.Path, seed) / 2; all += Leak(TakeRender.Method.Path, seed, -1, 16) / 2;
            lifelong += Leak(TakeRender.Method.Path, seed, 0, 16) / 2; energy += Leak(TakeRender.Method.Energy, seed) / 2;
        }
        TestContext.WriteLine($"leaked energy: path {path:G4}; every blocked connection traced {all:G4}; and no survival in time {lifelong:G4}; energy method {energy:G4}");
        Assert.That(path / all, Is.EqualTo(1).Within(.2), "Tracing blocked connections with probability t (weighted 1/t) keeps the mean.");
        Assert.That(all / lifelong, Is.EqualTo(1).Within(.2), "Survival in time keeps the mean.");
    }
    // Diffuse-field leakage, an analytic reference for sound through walls. In a closed room whose field
    // is diffuse (isotropic, uniform), the radiance L at a wall is E_in / (4 pi), E_in the reverberant
    // energy measured just inside it; a wall transmitting tau = (1 - a) t (undeviated) shows the
    // listener outside radiance tau L over the solid angle Omega it subtends, so the reverberant energy
    // outside is tau E_in Omega / (4 pi). The direct sound through the wall, (1 - a) t / r^2, is
    // subtracted from the outside total, and the direct sound from the inside one. s = 1 makes the
    // field diffuse; s = 0.5 leaves half of every reflection specular (the field is then only
    // approximately diffuse, and its specular part leaks through chains of reflections).
    // Opt-in (ACOUSTIC_BOUNDARY=1): a finite rigid plate (2 m x 2 m) under the source; listeners on a line
    // across the plate's reflection boundary (at x = 2 m), with and without UTD edge diffraction. README
    // part 8 compares these with the exact connection that the beams replaced.
    [Test, Category("GPU"), Category("Methods"), Category("Slow"), Timeout(1800000)]
    public void MeasureReflectionBoundary()
    {
        if (string.IsNullOrEmpty(Environment.GetEnvironmentVariable("ACOUSTIC_BOUNDARY"))) Assert.Ignore("Set ACOUSTIC_BOUNDARY.");
        var scene = Scene(2); scene.listeners = new[] {1}; scene.materials = new[] {RigidControl()};
        var a = new Vector3(-1, 0, -1); var b = new Vector3(1, 0, -1); var c = new Vector3(1, 0, 1); var d = new Vector3(-1, 0, 1);
        var geometry = new[] {new AcousticTake.Primitive {objectIndex = 2, a = a, b = c, c = b}, new AcousticTake.Primitive {objectIndex = 2, a = a, b = d, c = c}};
        scene.objects[2].primitiveCount = geometry.Length;
        foreach (float x in new[] {1f, 1.5f, 1.8f, 1.9f, 2f, 2.1f, 2.2f, 2.5f, 3f})
        {
            var line = new System.Text.StringBuilder(FormattableString.Invariant($"x {x:F1}:"));
                foreach (float diffraction in new[] {0f, 1f})
                {
                    var state = States(2); state[0].position = new Vector3(0, 1, 0); state[1].position = new Vector3(x, 1, 0);
                    state[2].position = Vector3.zero; state[2].flags = AcousticTake.StateFlags.Active | AcousticTake.StateFlags.Geometry;
                    var settings = ControlSettings(TakeRender.Method.Path, 1 << 18, .1f); settings.useBvh = false;
                    settings.diffractionFactor = diffraction;
                    using var solver = new TransportSolver(scene, geometry, settings, RequireGpu(), sampleRate: 8000);
                    solver.Solve(state, new TakeRender.DirectPath[1]);
                    line.Append(diffraction > 0 ? " beam+UTD" : " beam");
                    foreach (int band in new[] {1, 3, 5}) line.Append(FormattableString.Invariant($" {Window(ControlEnergy(solver, band), 0, int.MaxValue) - 1 / (x * x):G4}"));
                    line.Append(";");
                }
            TestContext.WriteLine(line.ToString()); Debug.Log("Boundary " + line);
        }
    }
    // Opt-in (ACOUSTIC_LEAK_SCAN=1): the leak fixture's outside energy (direct removed) for wall transmissions 1e-3, 1e-5, 0.
    [Test, Category("GPU"), Category("Methods"), Category("Slow"), Timeout(1800000)]
    public void MeasureLeakScan()
    {
        if (string.IsNullOrEmpty(Environment.GetEnvironmentVariable("ACOUSTIC_LEAK_SCAN"))) Assert.Ignore("Set ACOUSTIC_LEAK_SCAN.");
        var size = new Vector3(6, 5, 4); var source = new Vector3(-1.5f, .5f, .4f); var outside = new Vector3(3.8f, -.3f, .2f);
        foreach (float t in new[] {1e-3f, 1e-5f, 0f})
            foreach (float s in new[] {1f, .5f, 0f})
            {
                LateRoom(AcousticTake.Dimension.Spatial3D, size, LateMaterial(.1f, s, t), source, outside, out var scene, out var geometry, out var state);
                var settings = ControlSettings(TakeRender.Method.Path, 1 << 16, 1.5f); settings.specularDetectionRadius = .5f;
                using var solver = new TransportSolver(scene, geometry, settings, RequireGpu(), sampleRate: 8000);
                solver.Solve(state, new TakeRender.DirectPath[1]);
                double total = Window(ControlEnergy(solver, 2), 0, int.MaxValue) - (1 - .1) * t / (outside - source).sqrMagnitude;
                Debug.Log($"Leak scan t {t} s {s}: outside {total:G4}, escapes {solver.Method.Counter(2)}, events {solver.Method.Counter(1)}");
                TestContext.WriteLine($"t {t} s {s}: outside {total:G4}, escapes {solver.Method.Counter(2)}, events {solver.Method.Counter(1)}");
            }
    }
    [TestCase(TakeRender.Method.Path, 1f), TestCase(TakeRender.Method.Path, .5f), TestCase(TakeRender.Method.Energy, 1f), TestCase(TakeRender.Method.Energy, .5f),
     Category("GPU"), Category("Methods"), Category("Slow"), Timeout(1800000)]
    public void MethodsLeakAsTheDiffuseFieldPredicts(TakeRender.Method method, float scattering)
    {
        var size = new Vector3(6, 5, 4); const float a = .1f, t = 1e-3f;
        var source = new Vector3(-1.5f, .5f, .4f); var inside = new Vector3(2.2f, -.3f, .2f); var outside = new Vector3(3.8f, -.3f, .2f);
        double Total(Vector3 listener, int seed)
        {
            LateRoom(AcousticTake.Dimension.Spatial3D, size, LateMaterial(a, scattering, t), source, listener, out var scene, out var geometry, out var state);
            var settings = ControlSettings(method, 1 << 16, 1.5f); settings.seed = seed; settings.specularDetectionRadius = .5f;
            using var solver = new TransportSolver(scene, geometry, settings, RequireGpu(), sampleRate: 8000);
            solver.Solve(state, new TakeRender.DirectPath[1]);
            return Window(ControlEnergy(solver, 2), 0, int.MaxValue);
        }
        double tau = (1 - a) * t;
        double inRev = (Total(inside, 0) + Total(inside, 1)) / 2 - 1 / (inside - source).sqrMagnitude;
        double outRev = (Total(outside, 0) + Total(outside, 1)) / 2 - tau / (outside - source).sqrMagnitude;
        double h = outside.x - size.x / 2, omega = 0;
        foreach (double ay in new[] {size.y / 2 - outside.y, size.y / 2 + outside.y})
            foreach (double bz in new[] {size.z / 2 - outside.z, size.z / 2 + outside.z})
                omega += Math.Atan(ay * bz / (h * Math.Sqrt(h * h + ay * ay + bz * bz)));
        double predicted = tau * inRev * omega / (4 * Math.PI);
        TestContext.WriteLine($"{method}, s = {scattering}: reverberant energy inside {inRev:G4}, outside {outRev:G4}, diffuse-field prediction {predicted:G4} (wall solid angle {omega:F3} sr), ratio {outRev / predicted:F3}");
        Assert.That(outRev / predicted, Is.EqualTo(1).Within(scattering >= 1 ? .2 : .35));
    }
    static long Sum(uint[] values) { long s = 0; foreach (var v in values) s += v; return s; }

    // The tau = 1 limit of the leak check: an open hole. The room (6 x 5 x 4 m, open sheets, a = 0.1,
    // s = 1) has a 1.2 m x 1.0 m hole in its +x wall, centred at (y, z) = (-0.3, 0.2); outside, the
    // hole shows the listener the inside radiance E_in / (4 pi) over its solid angle, so the reverberant
    // energy outside is E_in Omega / (4 pi). Optional outside surfaces (ApertureCase): a ground at the
    // floor's height out to 40 m (a = 0.3, s = 1) and an absorbing screen at x = 3.4 that hides the hole
    // from the listener, so that only sound scattered by the ground reaches it.
    // Measured (12 November): both methods 1.15 times the prediction in view (the field near the hole
    // is not perfectly diffuse; the methods agree to 0.2%), and within 15% of each other behind the screen.
    static readonly Vector3 ApertureSize = new Vector3(6, 5, 4), ApertureSource = new Vector3(-1.5f, .5f, .4f);
    static readonly Vector3 ApertureInside = new Vector3(1.2f, -.3f, .2f), ApertureOutside = new Vector3(3.8f, -.3f, .2f);
    const float ApertureY = -.3f, ApertureZ = .2f, ApertureHalfY = .6f, ApertureHalfZ = .5f;
    static void ApertureRoom(bool ground, bool screen, Vector3 listener, out AcousticTake.Scene scene, out AcousticTake.Primitive[] geometry, out AcousticTake.State[] states, float transmission = 0, float groundScattering = 1, float screenGap = 0, bool porch = false, float roomScattering = 1)
    {
        LateRoom(AcousticTake.Dimension.Spatial3D, ApertureSize, LateMaterial(.1f, roomScattering, transmission), ApertureSource, listener, out scene, out var room, out states);
        var list = new List<AcousticTake.Primitive>();
        var objects = new List<AcousticTake.ObjectDefinition>(scene.objects);
        void Quad(Vector3 a, Vector3 b, Vector3 c, Vector3 d, int material)
        {
            list.Add(new AcousticTake.Primitive {materialIndex = material, a = a, b = b, c = c});
            list.Add(new AcousticTake.Primitive {materialIndex = material, a = a, b = c, c = d});
        }
        void Add(string id, int first)
        {
            for (int i = first; i < list.Count; i++) { var q = list[i]; q.objectIndex = objects.Count; list[i] = q; }
            objects.Add(new AcousticTake.ObjectDefinition {id = id, name = id, scale = Vector3.one, primitiveStart = first, primitiveCount = list.Count - first});
        }
        // Keep every face but +x (face 1, object 3), then the +x wall as a frame around the hole.
        var kept = new List<AcousticTake.ObjectDefinition>();
        foreach (var o in objects)
        {
            if (o.id == "face-1") continue;
            int first = o.primitiveStart; var copy = o;
            if (o.primitiveCount > 0)
            {
                copy.primitiveStart = list.Count;
                for (int i = 0; i < o.primitiveCount; i++) { var q = room[first + i]; q.objectIndex = kept.Count; list.Add(q); }
            }
            kept.Add(copy);
        }
        objects = kept;
        float x = ApertureSize.x / 2, y0 = -ApertureSize.y / 2, y1 = ApertureSize.y / 2, z0 = -ApertureSize.z / 2, z1 = ApertureSize.z / 2;
        float hy0 = ApertureY - ApertureHalfY, hy1 = ApertureY + ApertureHalfY, hz0 = ApertureZ - ApertureHalfZ, hz1 = ApertureZ + ApertureHalfZ;
        int start = list.Count;
        Quad(new Vector3(x, y0, z0), new Vector3(x, hy0, z0), new Vector3(x, hy0, z1), new Vector3(x, y0, z1), 0);   // below the hole
        Quad(new Vector3(x, hy1, z0), new Vector3(x, y1, z0), new Vector3(x, y1, z1), new Vector3(x, hy1, z1), 0);   // above
        Quad(new Vector3(x, hy0, z0), new Vector3(x, hy1, z0), new Vector3(x, hy1, hz0), new Vector3(x, hy0, hz0), 0); // beside, -z
        Quad(new Vector3(x, hy0, hz1), new Vector3(x, hy1, hz1), new Vector3(x, hy1, z1), new Vector3(x, hy0, z1), 0); // beside, +z
        Add("wall-with-hole", start);
        var materials = new List<AcousticTake.MaterialDefinition>(scene.materials);
        if (ground)
        {
            materials.Add(groundScattering >= 1 ? LateMaterial(.3f, 1) : LateMaterial(.01f, groundScattering)); start = list.Count;
            Quad(new Vector3(x, y0, -40), new Vector3(40, y0, -40), new Vector3(40, y0, 40), new Vector3(x, y0, 40), materials.Count - 1);
            Add("ground", start);
        }
        if (screen)
        {
            materials.Add(LateMaterial(1, 0)); start = list.Count;
            float bottom = y0 + screenGap;
            Quad(new Vector3(3.4f, bottom, -3), new Vector3(3.4f, 3.5f, -3), new Vector3(3.4f, 3.5f, 3), new Vector3(3.4f, bottom, 3), materials.Count - 1);
            Add("screen", start);
        }
        if (porch)
        {
            // A duct of the hole's cross-section, PorchLength deep: floor, roof and two side walls (open sheets, specular).
            materials.Add(LateMaterial(PorchAbsorption, 0)); start = list.Count; int m = materials.Count - 1; float x1 = x + PorchLength;
            Quad(new Vector3(x, hy0, hz0), new Vector3(x1, hy0, hz0), new Vector3(x1, hy0, hz1), new Vector3(x, hy0, hz1), m); // floor
            Quad(new Vector3(x, hy1, hz0), new Vector3(x1, hy1, hz0), new Vector3(x1, hy1, hz1), new Vector3(x, hy1, hz1), m); // roof
            Quad(new Vector3(x, hy0, hz0), new Vector3(x1, hy0, hz0), new Vector3(x1, hy1, hz0), new Vector3(x, hy1, hz0), m); // side, -z
            Quad(new Vector3(x, hy0, hz1), new Vector3(x1, hy0, hz1), new Vector3(x1, hy1, hz1), new Vector3(x, hy1, hz1), m); // side, +z
            Add("porch", start);
        }
        scene.objects = objects.ToArray(); scene.materials = materials.ToArray(); geometry = list.ToArray();
        var old = states; states = new AcousticTake.State[objects.Count];
        for (int i = 0; i < states.Length; i++)
            states[i] = i < 2 ? old[i] : new AcousticTake.State {rotation = Quaternion.identity, flags = AcousticTake.StateFlags.Active | AcousticTake.StateFlags.Geometry};
    }
    double ApertureTotal(TakeRender.Method method, bool ground, bool screen, Vector3 listener, int seed, int rays = 1 << 16, float transmission = 0,
                         float groundScattering = 1, float screenGap = 0, float radius = .5f, bool porch = false, float roomScattering = 1)
    {
        ApertureRoom(ground, screen, listener, out var scene, out var geometry, out var state, transmission, groundScattering, screenGap, porch, roomScattering);
        var settings = ControlSettings(method, rays, 1.5f); settings.seed = seed; settings.specularDetectionRadius = radius;
        using var solver = new TransportSolver(scene, geometry, settings, RequireGpu(), sampleRate: 8000);
        solver.Solve(state, new TakeRender.DirectPath[1]);
        return Window(ControlEnergy(solver, 2), 0, int.MaxValue);
    }
    // The porch: the hole opens into a duct of its own cross-section (1.2 m x 1.0 m), 3 m deep, whose
    // floor, roof and side walls are specular (a = 0.05, s = 0). The listener stands 4 m beyond the
    // mouth and off its axis (PorchListener), where every ray it sends back through the mouth meets
    // the duct walls at least twice before it reaches the hole. Reference (PorchReference): unfold the
    // duct (its walls form a rectangular lattice of images of the hole) and integrate over the mouth
    // the radiance through the hole times (1 - a)^k, k the reflections of the unfolded ray, plus the
    // source's own image paths through the hole. The radiance through the hole is kappa E_in / (4 pi),
    // E_in the reverberant energy inside, with kappa measured in view of the bare hole in the same
    // direction as the porch's rays (PorchCalibration; both methods agree there to 0.2%).
    const float PorchLength = 3, PorchAbsorption = .05f;
    static readonly Vector3 PorchListener = new Vector3(10, 2.1f, 2.2f), PorchCalibration = new Vector3(4.5f, .6f, .95f);
    // Integral over a rectangle (x = constant, y0..y1, z0..z1) of cos / r^2 seen from L, with each point's weight w(P).
    static double MouthIntegral(Vector3 L, double x, double y0, double y1, double z0, double z1, Func<double, double, double> weight, int n = 600)
    {
        double sum = 0, dy = (y1 - y0) / n, dz = (z1 - z0) / n;
        for (int i = 0; i < n; i++) for (int j = 0; j < n; j++)
        {
            double y = y0 + (i + .5) * dy, z = z0 + (j + .5) * dz, vx = x - L.x, vy = y - L.y, vz = z - L.z, r2 = vx * vx + vy * vy + vz * vz;
            sum += weight(y, z) * Math.Abs(vx) / (r2 * Math.Sqrt(r2)) * dy * dz;
        }
        return sum;
    }
    // The direct sound at an outside point if its line to the source passes through the hole, else 0.
    static double DirectThroughHole(Vector3 L)
    {
        var S = ApertureSource; double t = (ApertureSize.x / 2 - S.x) / (L.x - S.x), y = S.y + t * (L.y - S.y), z = S.z + t * (L.z - S.z);
        bool open = Math.Abs(y - ApertureY) <= ApertureHalfY && Math.Abs(z - ApertureZ) <= ApertureHalfZ;
        return open ? 1 / (L - S).sqrMagnitude : 0;
    }
    // Reflections of an unfolded coordinate in a lattice of walls at lo + i (hi - lo); the folded coordinate.
    static int Crossings(double u, double lo, double hi, out double folded)
    {
        double w = hi - lo, c = Math.Floor((u - lo) / w), f = u - lo - c * w;
        int i = (int)c; folded = lo + ((i & 1) == 0 ? f : w - f);
        return Math.Abs(i);
    }
    // The porch's exact energy at L: radiance x sum over the mouth of (1 - a)^k cos / r^2, and the source's image paths.
    static double PorchReference(Vector3 L, double radiance, out double reverberant, out double source, out double meanReflections)
    {
        double hy0 = ApertureY - ApertureHalfY, hy1 = ApertureY + ApertureHalfY, hz0 = ApertureZ - ApertureHalfZ, hz1 = ApertureZ + ApertureHalfZ;
        double x0 = ApertureSize.x / 2, x1 = x0 + PorchLength, rho = 1 - PorchAbsorption, kSum = 0, wSum = 0;
        reverberant = radiance * MouthIntegral(L, x1, hy0, hy1, hz0, hz1, (y, z) =>
        {
            double t = (x1 - x0) / (x1 - L.x); // continue the ray L -> mouth point on to x0 (unfolded)
            int k = Crossings(y + (y - L.y) * -t, hy0, hy1, out _) + Crossings(z + (z - L.z) * -t, hz0, hz1, out _);
            double w = Math.Pow(rho, k); kSum += k * w; wSum += w;
            return w;
        });
        meanReflections = kSum / wSum;
        source = 0;
        var S = ApertureSource;
        for (int iy = -12; iy <= 12; iy++) for (int iz = -12; iz <= 12; iz++)
        {
            // The image of S in cell (iy, iz) of the lattice (only the walls between x0 and x1 reflect; the room behind is diffuse).
            double Wy = hy1 - hy0, Wz = hz1 - hz0;
            double sy = hy0 + iy * Wy + ((iy & 1) == 0 ? S.y - hy0 : hy1 - S.y), sz = hz0 + iz * Wz + ((iz & 1) == 0 ? S.z - hz0 : hz1 - S.z);
            double dx = S.x - L.x, dy = sy - L.y, dz = sz - L.z;
            double at1 = (x1 - L.x) / dx, at0 = (x0 - L.x) / dx;
            double my = L.y + dy * at1, mz = L.z + dz * at1, hy = L.y + dy * at0, hz = L.z + dz * at0;
            if (my < hy0 || my > hy1 || mz < hz0 || mz > hz1) continue;                      // enters through the mouth
            if (hy < hy0 + iy * Wy || hy > hy0 + (iy + 1) * Wy || hz < hz0 + iz * Wz || hz > hz0 + (iz + 1) * Wz) continue; // leaves through the hole's image
            source += Math.Pow(rho, Math.Abs(iy) + Math.Abs(iz)) / (dx * dx + dy * dy + dz * dz);
        }
        return reverberant + source;
    }
    // Diffuse room (s = 1): the porch listener's energy against the image reference. Measured over 128
    // seeds (README part 7): energy method 1.050 +- 0.013, path method 1.089 +- 0.016 (exact connection;
    // 1.040 with the beams of part 8) times the
    // reference, whose radiance factor kappa is taken in one direction only (it is 1.15 on the hole's
    // axis and 1.04 at the porch's 38 degrees), so the reference itself is good to about 5%.
    [TestCase(TakeRender.Method.Path), TestCase(TakeRender.Method.Energy), Category("GPU"), Category("Methods"), Category("Slow"), Timeout(1800000)]
    public void MethodsSeeThroughAPorchAsTheImageReferencePredicts(TakeRender.Method method)
    {
        double hy0 = ApertureY - ApertureHalfY, hy1 = ApertureY + ApertureHalfY, hz0 = ApertureZ - ApertureHalfZ, hz1 = ApertureZ + ApertureHalfZ;
        double omega = MouthIntegral(PorchCalibration, ApertureSize.x / 2, hy0, hy1, hz0, hz1, (y, z) => 1);
        double inside = 0, outside = 0, calibration = 0; const int seeds = 8;
        for (int seed = 0; seed < seeds; seed++)
        {
            inside += (ApertureTotal(method, false, false, ApertureInside, seed, porch: true) - 1 / (ApertureInside - ApertureSource).sqrMagnitude) / seeds;
            outside += ApertureTotal(method, false, false, PorchListener, seed, porch: true) / seeds;
            calibration += (ApertureTotal(method, false, false, PorchCalibration, seed) - DirectThroughHole(PorchCalibration)) / seeds;
        }
        double kappa = calibration * 4 * Math.PI / (inside * omega);
        double reference = PorchReference(PorchListener, kappa * inside / (4 * Math.PI), out _, out double source, out double reflections);
        TestContext.WriteLine($"{method}: porch {outside:G4}, image reference {reference:G4} (kappa {kappa:F3}, source paths {source:G3}, {reflections:F2} reflections on average), ratio {outside / reference:F3}");
        Assert.That(outside / reference, Is.EqualTo(1).Within(.2));
    }
    // Opt-in (ACOUSTIC_PORCH=1): per method and seed, the inside reverberant energy and the porch listener's
    // total, and the calibration in view of the bare hole; ACOUSTIC_PORCH_SEEDS (default 16), _RAYS, _RADIUS.
    [Test, Category("GPU"), Category("Methods"), Category("Slow"), Timeout(7200000)]
    public void MeasurePorch()
    {
        if (string.IsNullOrEmpty(Environment.GetEnvironmentVariable("ACOUSTIC_PORCH"))) Assert.Ignore("Set ACOUSTIC_PORCH.");
        int seeds = int.Parse(Environment.GetEnvironmentVariable("ACOUSTIC_PORCH_SEEDS") ?? "16"), first = int.Parse(Environment.GetEnvironmentVariable("ACOUSTIC_PORCH_FIRST") ?? "0");
        int rays = int.Parse(Environment.GetEnvironmentVariable("ACOUSTIC_PORCH_RAYS") ?? "65536");
        float radius = float.Parse(Environment.GetEnvironmentVariable("ACOUSTIC_PORCH_RADIUS") ?? ".5", System.Globalization.CultureInfo.InvariantCulture);
        string methods = Environment.GetEnvironmentVariable("ACOUSTIC_PORCH_METHODS") ?? "Energy,Path";
        float roomScattering = float.Parse(Environment.GetEnvironmentVariable("ACOUSTIC_PORCH_ROOM_SCATTERING") ?? "1", System.Globalization.CultureInfo.InvariantCulture);
        double hy0 = ApertureY - ApertureHalfY, hy1 = ApertureY + ApertureHalfY, hz0 = ApertureZ - ApertureHalfZ, hz1 = ApertureZ + ApertureHalfZ;
        double omega = MouthIntegral(PorchCalibration, ApertureSize.x / 2, hy0, hy1, hz0, hz1, (y, z) => 1);
        PorchReference(PorchListener, 1, out double unit, out double source, out double meanK);
        Debug.Log($"Porch geometry: calibration solid angle {omega:G6} sr, porch unit-radiance integral {unit:G6}, source image paths {source:G6}, mean reflections {meanK:F2}");
        foreach (string name in methods.Split(','))
        {
            var method = (TakeRender.Method)Enum.Parse(typeof(TakeRender.Method), name);
            for (int seed = first; seed < first + seeds; seed++)
            {
                double inside = ApertureTotal(method, false, false, ApertureInside, seed, rays, 0, 1, 0, radius, true, roomScattering) - 1 / (ApertureInside - ApertureSource).sqrMagnitude;
                double outside = ApertureTotal(method, false, false, PorchListener, seed, rays, 0, 1, 0, radius, true, roomScattering);
                double calibration = ApertureTotal(method, false, false, PorchCalibration, seed, rays, 0, 1, 0, radius, false, roomScattering) - DirectThroughHole(PorchCalibration);
                double calibrationInside = inside; // the porch returns no sound into the room: its walls are parallel to the axis
                Debug.Log(FormattableString.Invariant($"Porch {name} seed {seed}: inside {inside:R} outside {outside:R} calibration inside {calibrationInside:R} calibration {calibration:R}"));
            }
        }
    }
    // Opt-in (ACOUSTIC_APERTURE=1): the hole's outside energy per method, in view without and with the
    // ground, and behind the screen. Options: ACOUSTIC_APERTURE_T (wall transmission), _S (ground
    // scattering; below 1 the ground is a = 0.01), _GAP (the screen's height above the ground),
    // _SEEDS (more than 4: the screen case only), _RADIUS (detection radius).
    [Test, Category("GPU"), Category("Methods"), Category("Slow"), Timeout(3600000)]
    public void MeasureApertureCases()
    {
        if (string.IsNullOrEmpty(Environment.GetEnvironmentVariable("ACOUSTIC_APERTURE"))) Assert.Ignore("Set ACOUSTIC_APERTURE.");
        float transmission = float.Parse(Environment.GetEnvironmentVariable("ACOUSTIC_APERTURE_T") ?? "0", System.Globalization.CultureInfo.InvariantCulture);
        float groundScattering = float.Parse(Environment.GetEnvironmentVariable("ACOUSTIC_APERTURE_S") ?? "1", System.Globalization.CultureInfo.InvariantCulture);
        float gap = float.Parse(Environment.GetEnvironmentVariable("ACOUSTIC_APERTURE_GAP") ?? "0", System.Globalization.CultureInfo.InvariantCulture);
        float radius = float.Parse(Environment.GetEnvironmentVariable("ACOUSTIC_APERTURE_RADIUS") ?? ".5", System.Globalization.CultureInfo.InvariantCulture);
        foreach (var (name, ground, screen) in new[] {("in view", false, false), ("in view, ground", true, false), ("behind the screen, ground", true, true)})
            foreach (var method in new[] {TakeRender.Method.Energy, TakeRender.Method.Path})
            {
                int seeds = int.Parse(Environment.GetEnvironmentVariable("ACOUSTIC_APERTURE_SEEDS") ?? "4");
                if (seeds > 4 && !screen) continue;
                var values = new double[seeds];
                for (int seed = 0; seed < seeds; seed++)
                    values[seed] = ApertureTotal(method, ground, screen, ApertureOutside, seed, 1 << 16, transmission, groundScattering, gap, radius) -
                                   (screen ? 0 : 1 / (ApertureOutside - ApertureSource).sqrMagnitude);
                TestContext.WriteLine($"{name}, {method}: reverberant outside {string.Join(", ", Array.ConvertAll(values, v => v.ToString("G4")))}");
                Debug.Log($"Aperture {name}, {method}: {string.Join(", ", Array.ConvertAll(values, v => v.ToString("G4")))}");
            }
    }
    [TestCase(TakeRender.Method.Path), TestCase(TakeRender.Method.Energy), Category("GPU"), Category("Methods"), Category("Slow"), Timeout(1800000)]
    public void MethodsRadiateThroughAHoleAsTheDiffuseFieldPredicts(TakeRender.Method method)
    {
        double inRev = 0, outRev = 0;
        for (int seed = 0; seed < 2; seed++)
        {
            inRev += (ApertureTotal(method, false, false, ApertureInside, seed) - 1 / (ApertureInside - ApertureSource).sqrMagnitude) / 2;
            outRev += (ApertureTotal(method, false, false, ApertureOutside, seed) - 1 / (ApertureOutside - ApertureSource).sqrMagnitude) / 2;
        }
        double h = ApertureOutside.x - ApertureSize.x / 2, omega = 4 * Math.Atan(ApertureHalfY * ApertureHalfZ / (h * Math.Sqrt(h * h + ApertureHalfY * ApertureHalfY + ApertureHalfZ * ApertureHalfZ)));
        double predicted = inRev * omega / (4 * Math.PI);
        TestContext.WriteLine($"{method}: reverberant energy inside {inRev:G4}, outside {outRev:G4}, diffuse-field prediction {predicted:G4} (hole solid angle {omega:F3} sr), ratio {outRev / predicted:F3}");
        Assert.That(outRev / predicted, Is.EqualTo(1).Within(.2));
    }
    // Behind the screen only sound scattered by the ground reaches the listener (no analytic value):
    // the two methods must agree. A specular ground seen beneath a raised screen is measured by
    // MeasureApertureCases (the Path method's estimate there is much noisier; README part 6).
    [Test, Category("GPU"), Category("Methods"), Category("Slow"), Timeout(1800000)]
    public void MethodsAgreeOnSoundScatteredOutsideAHole()
    {
        double energy = 0, path = 0;
        for (int seed = 0; seed < 4; seed++)
        {
            energy += ApertureTotal(TakeRender.Method.Energy, true, true, ApertureOutside, seed) / 4;
            path += ApertureTotal(TakeRender.Method.Path, true, true, ApertureOutside, seed) / 4;
        }
        TestContext.WriteLine($"behind the screen, scattering ground: energy method {energy:G4}, path method {path:G4}, ratio {path / energy:F3}");
        Assert.That(path / energy, Is.EqualTo(1).Within(.3));
    }

    // The audio synthesis reproduces the histogram: one fine-bin arrival becomes one band-shaped
    // click at its delay, and a late exponential decay keeps its band energies (measured through
    // the comparison's band filters). Each impulse's spectrum interpolates the band amplitudes
    // between band centres (amplitude-complementary weights), so a band measured through the
    // comparison's filter also sees its neighbours: the expectation is that interpolated
    // spectrum's band energy (equal to the histogram's for a flat spectrum, a little below the
    // band value where the spectrum is steep; both are printed).
    [Test, Category("Core")]
    public void PathReverbReproducesTheHistogram()
    {
        const int rate = 32000;
        var settings = SpectralSettings(); settings.method = TakeRender.Method.Path; settings.maximumDelaySeconds = 1.5f;
        settings.responseStepSamples = 1024; settings.fftSize = 1024; settings.listenerOutput = TakeRender.ListenerOutput.Mono;
        var bins = new PathBins(settings);
        var scene = Scene(); var state = States();
        double[] click = {1, .7, .5, .35, .25, .18, .12, .09};
        double Histogram(int pair, int d, int b, int k)
        {
            double t = bins.Start(k), e = 0;
            if (k == bins.Of(.01)) e += d == 0 ? click[b] : 0;                                   // one arrival at 10 ms, from +x
            if (t >= .05) e += d == 2 ? .002 * Math.Pow(.8, b) * bins.Width(k) / .001 : 0;  // stationary late field, -1 dB per octave
            return e;
        }
        var pcm = new float[1]; pcm[0] = 1;
        var signal = new TakeRender.Signal(pcm); signal.SetActive(0, true);
        signal.Append(new AcousticTake.VoiceEvent {action = AcousticTake.VoiceAction.Play, gain = 1});
        var reverb = new PathReverb(scene, rate, settings, new[] {signal});
        reverb.SetResponse(0, true, Histogram, state);
        int length = (int)(1.6 * rate), latency = settings.DspLatencySamples;
        var output = new float[length];
        var block = new float[1024];
        for (int at = 0; at < length; at += 1024)
        {
            int count = Math.Min(1024, length - at);
            Array.Clear(block, 0, block.Length); reverb.Render(at, count, block); Array.Copy(block, 0, output, at, count);
        }
        // Response (latency removed), in 1 ms band energies.
        var h = new double[length - latency];
        for (int n = 0; n < h.Length; n++) h[n] = output[n + latency];
        for (int b = 0; b < 6; b++)
        {
            var e = ImpulseBandEnergy(h, rate, b, 1600);
            double clickEnergy = Window(e, 5, 16), want = Interpolated(click, b);
            TestContext.WriteLine($"band {MethodBands.Centers[b]} Hz: click {clickEnergy:G4} (interpolated spectrum {want:G4}, histogram {click[b]:G4}), peak bin {Array.IndexOf(e, Max(e, 0, 30))} ms");
            Assert.That(clickEnergy / want, Is.EqualTo(1).Within(.15), "A single arrival keeps its band energy.");
            // Late field (random signs): band energy over 0.1-1.4 s; its relative spread is about one over
            // sqrt(bandwidth x duration), about 10% at 125 Hz.
            double late = Window(e, 100, 1400), wantLate = 0, bandLate = 0;
            var bandEnergies = new double[8];
            for (int k = 0; k < bins.Count; k++)
            {
                double s0 = bins.Start(k); if (s0 < .1 || s0 >= 1.4) continue;
                for (int q = 0; q < 8; q++) bandEnergies[q] = Histogram(0, 2, q, k);
                wantLate += Interpolated(bandEnergies, b); bandLate += bandEnergies[b];
            }
            TestContext.WriteLine($"   late 0.1-1.4 s {late:G4} (interpolated spectrum {wantLate:G4}, histogram {bandLate:G4})");
            Assert.That(late / wantLate, Is.EqualTo(1).Within(b == 0 ? .3 : .15), "The late field keeps its band energy.");
        }
    }
    // Updates every step between two histograms that split a late field over disjoint bins (the
    // extreme of two independent noisy estimates) must sound as loud as their mean, held fixed.
    [Test, Category("Core")]
    public void PathReverbCrossfadeKeepsThePower()
    {
        const int rate = 32000;
        var settings = SpectralSettings(); settings.method = TakeRender.Method.Path; settings.maximumDelaySeconds = .5f;
        settings.responseStepSamples = 512; settings.fftSize = 512; settings.listenerOutput = TakeRender.ListenerOutput.Mono;
        var bins = new PathBins(settings);
        var scene = Scene(); var state = States();
        double Late(int k, int b) => bins.Start(k) >= .02 && bins.Start(k) < .4 ? .002 * Math.Pow(.8, b) * bins.Width(k) / .001 : 0;
        Func<int, int, int, int, double> mean = (p, d, b, k) => d == 2 ? Late(k, b) : 0;
        Func<int, int, int, int, double> even = (p, d, b, k) => d == 2 && k % 2 == 0 ? 2 * Late(k, b) : 0;
        Func<int, int, int, int, double> odd = (p, d, b, k) => d == 2 && k % 2 == 1 ? 2 * Late(k, b) : 0;
        int length = 3 * rate, step = settings.responseStepSamples;
        var noise = new float[length]; var rng = new System.Random(1);
        for (int n = 0; n < length; n++) noise[n] = (float)(rng.NextDouble() * 2 - 1);
        double Energy(bool alternate)
        {
            var signal = new TakeRender.Signal(noise); signal.SetActive(0, true);
            signal.Append(new AcousticTake.VoiceEvent {action = AcousticTake.VoiceAction.Play, gain = 1});
            var reverb = new PathReverb(scene, rate, settings, new[] {signal});
            var block = new float[step]; double energy = 0;
            for (int at = 0, i = 0; at < length; at += step, i++)
            {
                reverb.SetResponse(at, false, alternate ? (i % 2 == 0 ? even : odd) : mean, state);
                Array.Clear(block, 0, step); reverb.Render(at, step, block);
                if (at >= rate / 2) foreach (var x in block) energy += (double)x * x;
            }
            return energy;
        }
        double fixedEnergy = Energy(false), alternating = Energy(true);
        TestContext.WriteLine($"alternating / fixed output energy {alternating / fixedEnergy:F3} ({10 * Math.Log10(alternating / fixedEnergy):+0.00;-0.00} dB; a plain linear crossfade gives 0.70, -1.55 dB)");
        Assert.That(alternating / fixedEnergy, Is.EqualTo(1).Within(.05), "Crossfading between updates keeps the power.");
    }
    // Band energies in 1 ms bins of a sampled impulse response (unit sample = 1 m free field), with
    // the comparison's zero-phase band filters (tools/method_compare.py energy_from_impulse).
    static double[] ImpulseBandEnergy(double[] h, int rate, int band, int bins)
    {
        int n = 1; while (n < 2 * h.Length) n <<= 1;
        var re = new double[n]; var im = new double[n];
        Array.Copy(h, re, h.Length);
        MethodBands.Fft(re, im, n, false);
        double fc = MethodBands.Centers[band];
        for (int k = 0; k < n; k++)
        {
            double f = (k <= n / 2 ? k : n - k) * (double)rate / n, g = Math.Sqrt(MethodBands.Power(f, fc));
            re[k] *= g; im[k] *= g;
        }
        MethodBands.Fft(re, im, n, true);
        // A unit sample has energy 1 / rate per sample squared; the filter's energy normalises the band.
        double norm = rate / MethodBands.FilterEnergy(fc);
        var e = new double[bins];
        for (int i = 0; i < n; i++)
        {
            int t = i < n / 2 ? i : i - n; int k = Math.Max(0, (int)Math.Floor(t * 1000.0 / rate));
            if (k < bins) e[k] += re[i] * re[i] * norm;
        }
        return e;
    }
    // Band energy, through the comparison's filter of `band`, of one impulse whose amplitude
    // spectrum interpolates sqrt(energies) with PathReverb's band weights.
    static double Interpolated(double[] energies, int band)
    {
        double fc = MethodBands.Centers[band], sum = 0, norm = 0;
        for (int i = 0; i < 4000; i++)
        {
            double f = fc / 2 * Math.Pow(4, (i + .5) / 4000), df = f * Math.Log(4) / 4000, a = 0;
            for (int q = 0; q < 8; q++) a += Math.Sqrt(energies[q]) * PathReverb.BandWeight(q, f);
            double g = MethodBands.Power(f, fc);
            sum += a * a * g * df; norm += g * df;
        }
        return sum / norm;
    }
    static double Max(double[] values, int from, int to) { double m = double.MinValue; for (int i = from; i < to; i++) m = Math.Max(m, values[i]); return m; }
}
