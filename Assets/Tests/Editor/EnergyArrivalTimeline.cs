using System;
using System.Globalization;
using System.IO;
using System.Text;
using NUnit.Framework;
using UnityEngine;

// Diagnostic export (opt-in): the energy method's discrete arrivals and diffuse
// histogram at every response of a take, solved exactly as TakeRender.RenderDirect
// solves them (same anchors, settings and audio-mode solver). Used to inspect a
// listening render (docs/benchmarks/energy-audio-2026-11-09/inspect.py).
//   ACOUSTIC_ENERGY_TIMELINE=/absolute/job.json   (TakeBatch job: take, output, settings)
// Writes <output>/solves.csv, arrivals.csv (every arrival of pair 0 with its rank in
// the deterministic audio order; rank < methodArrivalSlots plays as a path) and
// late.f32 [solve][band][late bin] (direction-summed diffuse energy).
// Optional ACOUSTIC_ENERGY_TIMELINE_SOLVES="first,count,stride" exports a subset of the responses.
public partial class TakeChecks
{
    [Test, Category("QualityMeasurement"), Timeout(int.MaxValue)]
    public void MeasureEnergyArrivalTimeline()
    {
        string file = Environment.GetEnvironmentVariable("ACOUSTIC_ENERGY_TIMELINE");
        if (string.IsNullOrEmpty(file)) Assert.Ignore("Set ACOUSTIC_ENERGY_TIMELINE to a TakeBatch job (several: separated by ';').");
        foreach (string one in file.Split(';')) ExportTimeline(one);
    }
    void ExportTimeline(string file)
    {
        var job = TakeBatch.ParseJob(File.ReadAllText(file));
        var settings = job.settings;
        Assert.That(settings.method, Is.EqualTo(TakeRender.Method.Energy));
        using var take = AcousticTake.Open(job.take);
        var scene = take.Metadata.scene; int rate = take.Metadata.sampleRate;
        Directory.CreateDirectory(job.output);
        var state = new AcousticTake.State[scene.objects.Length];
        long length = take.Metadata.durationSamples + settings.tailSamples;
        var invariant = CultureInfo.InvariantCulture;
        var solves = new StringBuilder("solve,sample,anchor,lx,ly,lz,sx,sy,sz,arrivals\n");
        var arrivalsText = new StringBuilder("solve,family,depth,identity,delay,length,dx,dy,dz,rate,e0,e1,e2,e3,e4,e5,e6,e7,rank\n");
        using var late = new BinaryWriter(File.Create(Path.Combine(job.output, "late.f32")));
        using var solver = new TransportSolver(scene, take.Geometry, settings, RequireGpu(), 1, rate, true, methodAudio: true);
        int index = 0, first = 0, count = int.MaxValue, stride = 1;
        string subset = Environment.GetEnvironmentVariable("ACOUSTIC_ENERGY_TIMELINE_SOLVES");
        if (!string.IsNullOrEmpty(subset)) { var v = subset.Split(','); first = int.Parse(v[0]); count = int.Parse(v[1]); stride = int.Parse(v[2]); }
        for (long response = first; response * settings.responseStepSamples < length && index < count; response += stride, index++)
        {
            long sample = response * settings.responseStepSamples;
            long anchor = settings.ResponseSample(sample);
            take.Evaluate(Math.Min(anchor, take.Metadata.durationSamples), state);
            solver.Solve(state, new TakeRender.DirectPath[scene.sources.Length * scene.listeners.Length]);
            var all = solver.Method.ReadArrivals();
            var keys = new double[all.Length]; var order = new int[all.Length];
            for (int a = 0; a < all.Length; a++) { order[a] = a; for (int b = 0; b < 8; b++) keys[a] += all[a].Energy(b); }
            Array.Sort(order, (x, y) => keys[x] != keys[y] ? keys[y].CompareTo(keys[x]) : all[x].identity != all[y].identity ? all[x].identity.CompareTo(all[y].identity) : x.CompareTo(y));
            var rank = new int[all.Length]; for (int r = 0; r < order.Length; r++) rank[order[r]] = r;
            Vector3 l = state[scene.listeners[0]].position, s = state[scene.sources[0].objectIndex].position;
            solves.AppendLine(string.Format(invariant, "{0},{1},{2},{3},{4},{5},{6},{7},{8},{9}", index, sample, anchor, l.x, l.y, l.z, s.x, s.y, s.z, solver.Method.ArrivalTotal()));
            for (int a = 0; a < all.Length; a++)
            {
                var x = all[a];
                arrivalsText.Append(string.Format(invariant, "{0},{1},{2},{3},{4:R},{5:R},{6},{7},{8},{9:R}", index, x.Family, x.Depth, x.identity, x.delay, x.length,
                                                  x.direction.x, x.direction.y, x.direction.z, x.direction.w));
                for (int b = 0; b < 8; b++) arrivalsText.Append(string.Format(invariant, ",{0:R}", x.Energy(b)));
                arrivalsText.Append(string.Format(invariant, ",{0}\n", rank[a]));
            }
            var frame = solver.Late(0);
            for (int b = 0; b < 8; b++)
                for (int k = 0; k < solver.Method.LateBins; k++)
                {
                    double e = 0;
                    for (int d = 0; d < 6; d++) e += frame.Energy(0, d, b, k);
                    late.Write((float)e);
                }
        }
        File.WriteAllText(Path.Combine(job.output, "solves.csv"), solves.ToString());
        File.WriteAllText(Path.Combine(job.output, "arrivals.csv"), arrivalsText.ToString());
        File.WriteAllText(Path.Combine(job.output, "meta.json"), JsonUtility.ToJson(new TimelineMeta {solves = index, lateBins = solver.Method.LateBins,
            lateBinSeconds = settings.LatePartition / (float)rate, sampleRate = rate, slots = settings.methodArrivalSlots, planes = solver.Method.PlaneCount,
            edges = solver.Method.EdgeCount, order = solver.Method.ArrivalOrder, families = solver.Method.ArrivalFamilies}, true));
        TestContext.WriteLine($"{index} solves written to {job.output}");
    }
    [Serializable] sealed class TimelineMeta { public int solves, lateBins, sampleRate, slots, planes, edges, order, families; public float lateBinSeconds; }
}
