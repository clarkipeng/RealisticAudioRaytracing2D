using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Runtime.InteropServices;
using NUnit.Framework;
using UnityEditor;
using UnityEngine;
using Stopwatch = System.Diagnostics.Stopwatch;

public partial class TakeChecks
{
    [StructLayout(LayoutKind.Sequential)]
    struct ExperimentEdge { public Vector4 a, b, face, normal; public uint owner, firstFace, secondFace, pad; }
    [StructLayout(LayoutKind.Sequential)]
    struct ExperimentNode { public Vector4 pointWeight; public uint edge, pad0, pad1, pad2; }
    [StructLayout(LayoutKind.Sequential)]
    struct ExperimentMaterial { public Vector4 surface, medium; }
    static object Field(object owner, string name) => owner.GetType().GetField(name,
        BindingFlags.NonPublic | BindingFlags.Instance).GetValue(owner);
    static string Csv(params object[] values) => string.Join(",", values.Select(x =>
        Convert.ToString(x, CultureInfo.InvariantCulture)));

    // Finite-edge quadrature defines this pilot's discrete target. It is not the
    // production Cauchy sampler or a substitute for a wave-equation reference.
    static ExperimentNode[] Nodes(ExperimentEdge[] edges, AcousticTake.State[] states, int order)
    {
        var nodes = new List<ExperimentNode>();
        for (int e = 0; e < edges.Length; e++)
        {
            var edge = edges[e]; var pose = states[edge.owner];
            Vector3 a = pose.position + pose.rotation * (Vector3)edge.a;
            Vector3 b = pose.position + pose.rotation * (Vector3)edge.b;
            for (int i = 0; i < order; i++)
            {
                double x = Math.Cos(Math.PI * (i + .75) / (order + .5)), derivative = 0;
                for (int iteration = 0; iteration < 20; iteration++)
                {
                    double previous = 1, current = x;
                    for (int k = 2; k <= order; k++)
                    { double next = ((2 * k - 1) * x * current - (k - 1) * previous) / k; previous = current; current = next; }
                    derivative = order * (x * current - previous) / (x * x - 1);
                    double change = current / derivative; x -= change;
                    if (Math.Abs(change) < 1e-14) break;
                }
                Vector4 point = Vector3.Lerp(a, b, (float)((x + 1) / 2));
                point.w = (float)(Vector3.Distance(a, b) / ((1 - x * x) * derivative * derivative));
                nodes.Add(new ExperimentNode {pointWeight = point, edge = (uint)e});
            }
        }
        return nodes.ToArray();
    }

    // Reverse Dijkstra on visible connections between actual edge quadrature
    // points. A graph route only ranks proposals; every sampled leg is ray-tested.
    static double[] DistanceField(float[] links, int count, int goal)
    {
        var cost = Enumerable.Repeat(double.PositiveInfinity, count).ToArray();
        var done = new bool[count]; cost[goal] = 0;
        for (int step = 0; step < count; step++)
        {
            int u = -1;
            for (int i = 0; i < count; i++) if (!done[i] && (u < 0 || cost[i] < cost[u])) u = i;
            if (u < 0 || double.IsInfinity(cost[u])) break;
            done[u] = true;
            for (int v = 0; v < count; v++)
                if (links[v * count + u] > 0)
                    cost[v] = Math.Min(cost[v], cost[u] + links[v * count + u]);
        }
        return cost;
    }

    static Vector2[] Proposal(ExperimentNode[] nodes, float[] links, Vector3 source, Vector3 listener,
        bool reverse, bool guided)
    {
        int n = nodes.Length, count = n + 2, origin = reverse ? n + 1 : n, goal = reverse ? n : n + 1;
        var costs = guided ? DistanceField(links, count, goal) : null;
        var table = new Vector2[(n + 1) * n]; var weight = new double[n];
        double measure = nodes.Sum(x => (double)x.pointWeight.w);
        double scale = Vector3.Distance(source, listener) * .25, floor = scale * .04;
        const double exploration = .15; // Experiment parameter, not a runtime default.
        for (int row = 0; row <= n; row++)
        {
            int u = row == 0 ? origin : row - 1;
            double total = 0;
            for (int v = 0; v < n; v++)
            {
                double d = links[u * count + v];
                weight[v] = d > 0 ? nodes[v].pointWeight.w / Math.Max(floor * floor, d * d) : 0;
                if (guided && !double.IsInfinity(costs[u]))
                    weight[v] *= Math.Exp(-Math.Max(0, d + costs[v] - costs[u]) / scale);
                total += weight[v];
            }
            double cdf = 0; float previous = 0;
            for (int v = 0; v < n; v++)
            {
                double uniform = nodes[v].pointWeight.w / measure;
                cdf += total > 0 ? (1 - exploration) * weight[v] / total + exploration * uniform : uniform;
                float next = v == n - 1 ? 1 : (float)cdf;
                // Stored PDF is exactly the interval used by the GPU CDF search.
                table[row * n + v] = new Vector2(next - previous, next); previous = next;
                if (table[row * n + v].x <= 0) throw new InvalidOperationException("Candidate lost support.");
            }
        }
        return table;
    }

    static double Probability(Vector2[] forward, Vector2[] reverse, int n, int a, int b, int method)
    {
        double f = (double)forward[a].x * forward[(a + 1) * n + b].x;
        double r = (double)reverse[b].x * reverse[(b + 1) * n + a].x;
        double join = (double)forward[a].x * reverse[b].x;
        return method == 0 ? f : method == 1 ? r : method == 2 ? join : (f + r + join) / 3;
    }

    [Test, Category("SamplingExperiment")]
    public void CompareEdgeSampling()
    {
        Assert.That(Application.isBatchMode, Is.True, "Run silently in the isolated batch project.");
        AudioListener.volume = 0;
        string output = Environment.GetEnvironmentVariable("ACOUSTIC_SAMPLING_OUTPUT");
        Assert.That(output, Is.Not.Null.And.Not.Empty); Directory.CreateDirectory(output);
        using var metrics = new StreamWriter(Path.Combine(output, "metrics.csv"));
        using var trials = new StreamWriter(Path.Combine(output, "trials.csv"));
        using var reference = new StreamWriter(Path.Combine(output, "reference.csv"));
        using var setup = new StreamWriter(Path.Combine(output, "setup.csv"));
        File.WriteAllText(Path.Combine(output, "device.txt"), SystemInfo.graphicsDeviceName + "\nUnity " + Application.unityVersion + "\n");
        setup.WriteLine("scene,nodes,connections_ms,baseline_tables_ms,guided_tables_ms,proposal_pair_bytes,first_connections_ms");
        metrics.WriteLine("scene,hz,nodes,guided,method,samples,target_relative_rmse,trials,repeat,build_ms,upload_ms,batch_ms,single_ms,readback_bytes,reference_real,reference_imag,expected_rmse,empirical_rmse,mean_real,mean_imag");
        trials.WriteLine("scene,hz,guided,method,samples,target_relative_rmse,repeat,trial,real,imag");
        reference.WriteLine("scene,hz,quadrature_order,nodes,real,imag,enumeration_ms,reciprocity_relative_l2");
        var shader = UnityEngine.Object.Instantiate(AssetDatabase.LoadAssetAtPath<ComputeShader>("Assets/Shaders/SamplingExperiment.compute"));
        Assert.That(shader, Is.Not.Null);
        try
        {
            int enumerate = shader.FindKernel("Enumerate"), connections = shader.FindKernel("Connections"), estimate = shader.FindKernel("Estimate");
            foreach (int fixture in new[] {0, 1})
            {
                string name = fixture == 0 ? "box" : "staggered";
                int boxes = fixture + 1;
                var scene = Scene(1 + boxes); scene.listeners = new[] {1};
                var state = States(1 + boxes);
                state[0].position = new Vector3(-3, -.4f, .2f);
                state[1].position = new Vector3(3, .7f, -.1f);
                scene.materials = new[] {new AcousticTake.MaterialDefinition {frequencies = new[] {0f},
                    absorption = new[] {0f}, scattering = new[] {0f}, transmission = new[] {0f}}};
                var geometry = new List<AcousticTake.Primitive>();
                for (int b = 0; b < boxes; b++)
                {
                    int owner = b + 2;
                    state[owner].position = fixture == 0 ? Vector3.zero : new Vector3(b == 0 ? -.8f : .8f, b == 0 ? -.9f : .9f, 0);
                    state[owner].flags = AcousticTake.StateFlags.Active | AcousticTake.StateFlags.Geometry;
                    Vector3 scale = fixture == 0 ? new Vector3(1, 1.6f, 1.2f) : new Vector3(.3f, 1.1f, b == 0 || Environment.GetEnvironmentVariable("ACOUSTIC_SAMPLING_COPLANAR") == "1" ? 1.4f : 1.7f);
                    var mesh = TransportBox(AcousticTake.Dimension.Spatial3D, owner, 0, 1);
                    scene.objects[owner].primitiveStart = geometry.Count; scene.objects[owner].primitiveCount = mesh.Length;
                    foreach (var triangle in mesh)
                    { var p = triangle; p.a = Vector3.Scale(p.a, scale); p.b = Vector3.Scale(p.b, scale); p.c = Vector3.Scale(p.c, scale); geometry.Add(p); }
                }
                var settings = EdgeSettings(); settings.useBvh = true;
                using var solver = new TransportSolver(scene, geometry.ToArray(), settings, RequireGpu(), sampleRate: 8000);
                solver.Solve(state, new TakeRender.DirectPath[1]);
                var edgeBuffer = (ComputeBuffer)Field(Field(solver, "transport"), "edgeBuffer");
                var edges = new ExperimentEdge[edgeBuffer.count]; edgeBuffer.GetData(edges);
                shader.SetInt("BvhNodeCount", ((DirectBvh)Field(solver, "bvh")).NodeCount);
                shader.SetInt("PrimitiveCount", geometry.Count);
                shader.SetInt("ObjectCount", state.Length);
                int continuous = shader.FindKernel("Continuous");
                shader.SetInt("EdgeCount", edges.Length);
                shader.SetBuffer(continuous, "EdgeRanges", (ComputeBuffer)Field(Field(solver, "transport"), "edgeRangeBuffer"));
                shader.SetBuffer(continuous, "EdgeIndices", (ComputeBuffer)Field(Field(solver, "transport"), "edgeIndexBuffer"));
                foreach (int kernel in new[] {enumerate, connections, estimate, continuous})
                {
                    shader.SetBuffer(kernel, "Edges", edgeBuffer);
                    foreach (string field in new[] {"primitives", "states", "bvhNodes", "bvhIndices"})
                        shader.SetBuffer(kernel, char.ToUpperInvariant(field[0]) + field.Substring(1), (ComputeBuffer)Field(solver, field));
                }
                foreach (int order in new[] {16, 32, 64})
                {
                    var nodes = Nodes(edges, state, order); int n = nodes.Length;
                    using var nodeBuffer = new ComputeBuffer(n, 32); nodeBuffer.SetData(nodes);
                    using var result = new ComputeBuffer(n * n, 8);
                    shader.SetInt("NodeCount", n);
                    shader.SetVector("ExperimentSource", state[0].position); shader.SetVector("ExperimentListener", state[1].position);
                    foreach (int kernel in new[] {enumerate, connections, estimate}) shader.SetBuffer(kernel, "SampleNodes", nodeBuffer);
                    shader.SetBuffer(enumerate, "ExperimentOutput", result);
                    // Only order 32 benchmarks; other orders expose target discretization error.
                    var watch = Stopwatch.StartNew();
                    Vector2[][] proposals = null; float[] links = null; double[] buildMs = new double[2];
                    if (order == 32)
                    {
                        using var connectionBuffer = new ComputeBuffer((n + 2) * (n + 2), 4);
                        shader.SetBuffer(connections, "ConnectionLengths", connectionBuffer);
                        shader.Dispatch(connections, (connectionBuffer.count + 63) / 64, 1, 1);
                        links = new float[connectionBuffer.count]; connectionBuffer.GetData(links);
                        double firstConnectionsMs = watch.Elapsed.TotalMilliseconds;
                        watch.Restart();
                        shader.Dispatch(connections, (connectionBuffer.count + 63) / 64, 1, 1); connectionBuffer.GetData(links);
                        double connectionsMs = watch.Elapsed.TotalMilliseconds;
                        Assert.That(links[n * (n + 2) + n + 1], Is.LessThan(0), "Fixture must block the direct segment.");
                        proposals = new Vector2[4][];
                        watch.Restart();
                        for (int i = 0; i < 2; i++) proposals[i] = Proposal(nodes, links, state[0].position, state[1].position, i % 2 == 1, false);
                        double baselineMs = watch.Elapsed.TotalMilliseconds;
                        watch.Restart();
                        for (int i = 2; i < 4; i++) proposals[i] = Proposal(nodes, links, state[0].position, state[1].position, i % 2 == 1, true);
                        double guidedMs = watch.Elapsed.TotalMilliseconds;
                        buildMs[0] = connectionsMs + baselineMs; buildMs[1] = connectionsMs + guidedMs;
                        setup.WriteLine(Csv(name, n, connectionsMs, baselineMs, guidedMs, (long)(n + 1) * n * 8 * 2, firstConnectionsMs)); setup.Flush();
                    }
                    foreach (float hz in new[] {62.5f, 250f})
                    {
                        shader.SetFloat("ExperimentFrequency", hz);
                        watch.Restart(); shader.Dispatch(enumerate, (n * n + 63) / 64, 1, 1);
                        var exact = new Vector2[n * n]; result.GetData(exact);
                        double enumerationMs = watch.Elapsed.TotalMilliseconds;
                        double real = exact.Sum(x => (double)x.x), imag = exact.Sum(x => (double)x.y);
                        Assert.That(double.IsNaN(real) || double.IsInfinity(real) || double.IsNaN(imag) || double.IsInfinity(imag), Is.False);
                        Assert.That(real * real + imag * imag, Is.GreaterThan(0));
                        shader.SetVector("ExperimentSource", state[1].position); shader.SetVector("ExperimentListener", state[0].position);
                        shader.Dispatch(enumerate, (n * n + 63) / 64, 1, 1);
                        var reversed = new Vector2[n * n]; result.GetData(reversed);
                        double delta = 0, norm = 0;
                        for (int a = 0; a < n; a++) for (int b = 0; b < n; b++)
                        { delta += (exact[a * n + b] - reversed[b * n + a]).sqrMagnitude; norm += exact[a * n + b].sqrMagnitude; }
                        double reciprocity = Math.Sqrt(delta / norm);
                        if (reciprocity >= .002 && links != null)
                        {
                            int count = n + 2;
                            var worst = Enumerable.Range(0, n * n).OrderByDescending(i =>
                                (exact[i] - reversed[(i % n) * n + i / n]).sqrMagnitude).Take(24);
                            using var audit = new StreamWriter(Path.Combine(output, "reciprocity-" + name + "-" + hz + ".csv"));
                            audit.WriteLine("a,b,edge_a,edge_b,ax,ay,az,bx,by,bz,real,imag,reverse_real,reverse_imag,S_a,a_b,b_L,L_b,b_a,a_S");
                            foreach (int i in worst)
                            {
                                int a = i / n, b = i % n; var x = nodes[a]; var y = nodes[b]; var rev = reversed[b * n + a];
                                audit.WriteLine(Csv(a,b,x.edge,y.edge,x.pointWeight.x,x.pointWeight.y,x.pointWeight.z,
                                    y.pointWeight.x,y.pointWeight.y,y.pointWeight.z,exact[i].x,exact[i].y,rev.x,rev.y,
                                    links[n*count+a],links[a*count+b],links[b*count+n+1],links[(n+1)*count+b],links[b*count+a],links[a*count+n]));
                            }
                        }
                        reference.WriteLine(Csv(name, hz, order, n, real, imag, enumerationMs, reciprocity)); reference.Flush();
                        Assert.That(reciprocity, Is.LessThan(.002), "Rigid two-edge target must be reciprocal.");
                        shader.SetVector("ExperimentSource", state[0].position); shader.SetVector("ExperimentListener", state[1].position);
                        if (order == 32) Benchmark(shader, estimate, nodes, proposals, exact, real, imag, name, hz, buildMs, metrics, trials);
                        if (order == 64) ProductionBenchmark(solver, shader, real, imag, name, hz, metrics, trials);
                    }
                }
            }
        }
        finally { UnityEngine.Object.DestroyImmediate(shader); }
    }

    static void Benchmark(ComputeShader shader, int kernel, ExperimentNode[] nodes, Vector2[][] proposals,
        Vector2[] exact, double real, double imag, string scene, float hz, double[] buildMs, StreamWriter metrics, StreamWriter trials)
    {
        int n = nodes.Length; const int trialCount = 32;
        using var forward = new ComputeBuffer((n + 1) * n, 8);
        using var reverse = new ComputeBuffer((n + 1) * n, 8);
        var variance = new double[8]; var budgets = new (int samples, double target)[8][];
        for (int variant = 0; variant < 8; variant++)
        {
            int guided = variant / 4, method = variant % 4;
            var f = proposals[guided * 2]; var r = proposals[guided * 2 + 1];
            double secondMoment = 0;
            for (int a = 0; a < n; a++) for (int b = 0; b < n; b++)
            { var v = exact[a * n + b]; secondMoment += ((double)v.x * v.x + (double)v.y * v.y) / Probability(f, r, n, a, b, method); }
            variance[variant] = Math.Max(0, secondMoment - real * real - imag * imag);
            int Samples(double error) => checked(Math.Max(64, (int)Math.Ceiling(variance[variant] / (error * error * (real * real + imag * imag)) / 64) * 64));
            budgets[variant] = new[] {(64, 0d), (512, 0d), (4096, 0d), (Samples(.2), .2), (Samples(.1), .1)};
        }
        using var output = new ComputeBuffer(budgets.SelectMany(x => x).Max(x => x.samples) * trialCount / 64, 8);
        shader.SetBuffer(kernel, "ForwardProposal", forward); shader.SetBuffer(kernel, "ReverseProposal", reverse);
        shader.SetBuffer(kernel, "ExperimentOutput", output); shader.SetInt("TrialCount", trialCount);
        string[] methods = {"source", "listener", "join", "bidirectional_mis"};
        // Alternate order each repeat. Timings include dispatch + blocking readback;
        // graph build and table upload are separately reported, never hidden.
        for (int repeat = -1; repeat < 5; repeat++) for (int choice = 0; choice < 8; choice++)
        {
            int variant = repeat % 2 == 0 ? 7 - choice : choice, guided = variant / 4, method = variant % 4;
            var f = proposals[guided * 2]; var r = proposals[guided * 2 + 1];
            var watch = Stopwatch.StartNew(); forward.SetData(f); reverse.SetData(r);
            double uploadMs = watch.Elapsed.TotalMilliseconds;
            shader.SetInt("Strategy", method); shader.SetInt("SeedOffset", 1000 + (repeat + 1) * trialCount);
            foreach (var budget in budgets[variant])
            {
                int samples = budget.samples;
                shader.SetInt("SampleCount", samples); int groupsPerTrial = samples / 64;
                var sums = new Vector2[trialCount * groupsPerTrial];
                shader.SetInt("TrialCount", trialCount);
                watch.Restart(); shader.Dispatch(kernel, sums.Length, 1, 1); output.GetData(sums, 0, 0, sums.Length);
                double elapsed = watch.Elapsed.TotalMilliseconds;
                var single = new Vector2[groupsPerTrial]; shader.SetInt("TrialCount", 1);
                watch.Restart(); shader.Dispatch(kernel, single.Length, 1, 1); output.GetData(single, 0, 0, single.Length);
                double singleMs = watch.Elapsed.TotalMilliseconds;
                Assert.That(single.SequenceEqual(sums.Take(single.Length)), Is.True, "Single estimate must equal first batched estimate.");
                if (repeat < 0) continue;
                var stats = EstimateStatistics(sums, trialCount, real, imag, trials, scene, hz, guided, methods[method], samples, budget.target, repeat);
                double meanReal = stats.real, meanImag = stats.imag;
                double bias = Math.Sqrt((meanReal - real) * (meanReal - real) + (meanImag - imag) * (meanImag - imag));
                Assert.That(bias, Is.LessThan(6 * Math.Sqrt(variance[variant] / (samples * trialCount)) + 1e-7), "Estimator mean differs from enumerated target.");
                metrics.WriteLine(Csv(scene, hz, n, guided, methods[method], samples, budget.target, trialCount, repeat,
                    buildMs[guided], uploadMs, elapsed, singleMs, sums.Length * 8, real, imag, Math.Sqrt(variance[variant] / samples),
                    stats.rmse, meanReal, meanImag)); metrics.Flush();
            }
        }
        trials.Flush();
    }

    static (double real, double imag, double rmse) EstimateStatistics(Vector2[] sums, int trialCount,
        double real, double imag, StreamWriter trials, string scene, float hz, int guided, string method, int samples, double target, int repeat)
    {
        double error = 0, meanReal = 0, meanImag = 0; int groups = sums.Length / trialCount;
        for (int trial = 0; trial < trialCount; trial++)
        {
            double x = 0, y = 0;
            for (int i = 0; i < groups; i++) { var v = sums[trial * groups + i]; x += v.x; y += v.y; }
            Assert.That(double.IsNaN(x) || double.IsInfinity(x) || double.IsNaN(y) || double.IsInfinity(y), Is.False);
            error += (x - real) * (x - real) + (y - imag) * (y - imag); meanReal += x; meanImag += y;
            trials.WriteLine(Csv(scene, hz, guided, method, samples, target, repeat, trial, x, y));
        }
        return (meanReal / trialCount, meanImag / trialCount, Math.Sqrt(error / trialCount));
    }

    static void ProductionBenchmark(TransportSolver solver, ComputeShader reducer, double real, double imag,
        string scene, float hz, StreamWriter metrics, StreamWriter trials)
    {
        const int trialCount = 32;
        var budgets = new List<(int samples, double target)> {(64, 0), (512, 0), (4096, 0), (16384, 0)};
        string calibration = Path.Combine(Environment.GetEnvironmentVariable("ACOUSTIC_SAMPLING_OUTPUT"), "calibration.csv");
        int seedBase = File.Exists(calibration) ? 7000001 : 0;
        if (File.Exists(calibration)) foreach (string line in File.ReadAllLines(calibration).Skip(1))
        {
            var columns = line.Split(',');
            if (columns[0] == scene && float.Parse(columns[1], CultureInfo.InvariantCulture) == hz)
                budgets.Add((int.Parse(columns[2], CultureInfo.InvariantCulture), double.Parse(columns[3], CultureInfo.InvariantCulture)));
        }
        int maximumSamples = budgets.Max(x => x.samples);
        var watch = Stopwatch.StartNew();
        var transport = Field(solver, "transport"); var shader = (ComputeShader)Field(solver, "shader");
        int trace = shader.FindKernel("Trace3DBvh"), reduce = reducer.FindKernel("ReduceProduction"), continuous = reducer.FindKernel("Continuous");
        using var paths = new ComputeBuffer(trialCount * (1 + 3 * maximumSamples), 32);
        using var output = new ComputeBuffer(trialCount * maximumSamples / 64, 8);
        using var sources = new ComputeBuffer(trialCount, 4); sources.SetData(new uint[trialCount]);
        using var enabled = new ComputeBuffer(trialCount, 4); enabled.SetData(Enumerable.Repeat(1u, trialCount).ToArray());
        var materials = (ComputeBuffer)Field(transport, "materialBuffer");
        var original = new ExperimentMaterial[materials.count]; materials.GetData(original);
        var absorbing = (ExperimentMaterial[])original.Clone();
        for (int i = 0; i < absorbing.Length; i++) absorbing[i].surface = new Vector4(1, 0, 0, 1);
        materials.SetData(absorbing);
        shader.SetBuffer(trace, "RayPaths", paths); shader.SetBuffer(trace, "Sources", sources); shader.SetBuffer(trace, "RayPairEnabled", enabled);
        shader.SetInt("FrequencyBins", 1); shader.SetInt("MaxBounces", 2); shader.SetInt("FrameCount", 1);
        reducer.SetBuffer(reduce, "ProductionPaths", paths); reducer.SetBuffer(reduce, "ExperimentOutput", output);
        reducer.SetBuffer(continuous, "ExperimentOutput", output);
        double buildMs = watch.Elapsed.TotalMilliseconds;
        try
        {
            for (int repeat = -1; repeat < 5; repeat++) foreach (string method in repeat % 2 == 0 ? new[] {"cauchy", "production"} : new[] {"production", "cauchy"}) foreach (var budget in budgets)
            {
                int samples = budget.samples;
                shader.SetInt("RaysPerBin", samples); shader.SetInt("RaysPerPair", samples); shader.SetInt("SlotsPerPair", 1 + 3 * samples);
                shader.SetInt("RaySeed", seedBase + (repeat + 1) * 104729); reducer.SetInt("SampleCount", samples);
                reducer.SetInt("SeedOffset", seedBase + (repeat + 1) * trialCount);
                var sums = new Vector2[trialCount * samples / 64]; var single = new Vector2[samples / 64];
                void Dispatch(int count)
                {
                    if (method == "cauchy") reducer.Dispatch(continuous, count * samples / 64, 1, 1);
                    else
                    {
                        shader.SetInt("PairCount", count); shader.Dispatch(trace, count * samples / 64, 1, 1);
                        reducer.Dispatch(reduce, count * samples / 64, 1, 1);
                    }
                }
                if (repeat == -1 && samples == 64 && method == "production")
                {
                    shader.SetFloat("DiffractionFactor", 0); Dispatch(1); output.GetData(single, 0, 0, single.Length);
                    Assert.That(single.All(x => x == Vector2.zero), Is.True, "Production control must contain no surface contribution.");
                    shader.SetFloat("DiffractionFactor", 1);
                }
                watch.Restart(); Dispatch(trialCount); output.GetData(sums, 0, 0, sums.Length);
                double batchMs = watch.Elapsed.TotalMilliseconds;
                watch.Restart(); Dispatch(1); output.GetData(single, 0, 0, single.Length);
                double singleMs = watch.Elapsed.TotalMilliseconds;
                Assert.That(single.SequenceEqual(sums.Take(single.Length)), Is.True, "Production control single/batch packing differs.");
                if (repeat < 0) continue;
                var stats = EstimateStatistics(sums, trialCount, real, imag, trials, scene, hz, 0, method, samples, budget.target, repeat);
                metrics.WriteLine(Csv(scene, hz, 0, 0, method, samples, budget.target, trialCount, repeat, buildMs, 0,
                    batchMs, singleMs, sums.Length * 8, real, imag, double.NaN, stats.rmse, stats.real, stats.imag)); metrics.Flush();
            }
        }
        finally
        {
            materials.SetData(original);
            shader.SetBuffer(trace, "RayPaths", (ComputeBuffer)Field(transport, "pathBuffer"));
            shader.SetBuffer(trace, "Sources", (ComputeBuffer)Field(solver, "sources"));
            shader.SetBuffer(trace, "RayPairEnabled", (ComputeBuffer)Field(transport, "enabledBuffer"));
        }
        trials.Flush();
    }
}
