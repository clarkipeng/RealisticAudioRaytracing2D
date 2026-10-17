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
        setup.WriteLine("scene,nodes,connections_ms,baseline_tables_ms,guided_tables_ms,proposal_pair_bytes");
        metrics.WriteLine("scene,hz,nodes,guided,method,samples,trials,repeat,build_ms,upload_ms,batch_ms,readback_bytes,reference_real,reference_imag,expected_rmse,empirical_rmse,mean_real,mean_imag");
        trials.WriteLine("scene,hz,guided,method,samples,repeat,trial,real,imag");
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
                foreach (int kernel in new[] {enumerate, connections, estimate})
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
                    Vector2[][] proposals = null; float[] links = null; double buildMs = 0;
                    if (order == 32)
                    {
                        using var connectionBuffer = new ComputeBuffer((n + 2) * (n + 2), 4);
                        shader.SetBuffer(connections, "ConnectionLengths", connectionBuffer);
                        shader.Dispatch(connections, (connectionBuffer.count + 63) / 64, 1, 1);
                        links = new float[connectionBuffer.count]; connectionBuffer.GetData(links);
                        double connectionsMs = watch.Elapsed.TotalMilliseconds;
                        Assert.That(links[n * (n + 2) + n + 1], Is.LessThan(0), "Fixture must block the direct segment.");
                        proposals = new Vector2[4][];
                        watch.Restart();
                        for (int i = 0; i < 2; i++) proposals[i] = Proposal(nodes, links, state[0].position, state[1].position, i % 2 == 1, false);
                        double baselineMs = watch.Elapsed.TotalMilliseconds;
                        watch.Restart();
                        for (int i = 2; i < 4; i++) proposals[i] = Proposal(nodes, links, state[0].position, state[1].position, i % 2 == 1, true);
                        double guidedMs = watch.Elapsed.TotalMilliseconds;
                        buildMs = connectionsMs + baselineMs + guidedMs;
                        setup.WriteLine(Csv(name, n, connectionsMs, baselineMs, guidedMs, (long)(n + 1) * n * 8 * 2)); setup.Flush();
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
                        if (order != 32) continue;
                        Benchmark(shader, estimate, nodes, proposals, exact, real, imag, name, hz, buildMs, metrics, trials);
                    }
                }
            }
        }
        finally { UnityEngine.Object.DestroyImmediate(shader); }
    }

    static void Benchmark(ComputeShader shader, int kernel, ExperimentNode[] nodes, Vector2[][] proposals,
        Vector2[] exact, double real, double imag, string scene, float hz, double buildMs, StreamWriter metrics, StreamWriter trials)
    {
        int n = nodes.Length; const int trialCount = 32;
        using var forward = new ComputeBuffer((n + 1) * n, 8);
        using var reverse = new ComputeBuffer((n + 1) * n, 8);
        using var output = new ComputeBuffer(4096 * trialCount / 64, 8);
        shader.SetBuffer(kernel, "ForwardProposal", forward); shader.SetBuffer(kernel, "ReverseProposal", reverse);
        shader.SetBuffer(kernel, "ExperimentOutput", output); shader.SetInt("TrialCount", trialCount);
        string[] methods = {"source", "listener", "join", "bidirectional_mis"};
        // Alternate order each repeat. Timings include dispatch + blocking readback;
        // graph build and table upload are separately reported, never hidden.
        for (int repeat = -1; repeat < 3; repeat++) for (int choice = 0; choice < 8; choice++)
        {
            int variant = repeat % 2 == 0 ? 7 - choice : choice, guided = variant / 4, method = variant % 4;
            var f = proposals[guided * 2]; var r = proposals[guided * 2 + 1];
            double secondMoment = 0;
            for (int a = 0; a < n; a++) for (int b = 0; b < n; b++)
            {
                Vector2 value = exact[a * n + b];
                secondMoment += ((double)value.x * value.x + (double)value.y * value.y) / Probability(f, r, n, a, b, method);
            }
            double variance = Math.Max(0, secondMoment - real * real - imag * imag);
            var watch = Stopwatch.StartNew(); forward.SetData(f); reverse.SetData(r);
            double uploadMs = watch.Elapsed.TotalMilliseconds;
            shader.SetInt("Strategy", method); shader.SetInt("SeedOffset", (repeat + 1) * trialCount);
            foreach (int samples in new[] {64, 512, 4096})
            {
                shader.SetInt("SampleCount", samples); int groupsPerTrial = samples / 64;
                var sums = new Vector2[trialCount * groupsPerTrial];
                watch.Restart(); shader.Dispatch(kernel, sums.Length, 1, 1); output.GetData(sums, 0, 0, sums.Length);
                double elapsed = watch.Elapsed.TotalMilliseconds;
                if (repeat < 0) continue;
                double error = 0, meanReal = 0, meanImag = 0;
                for (int trial = 0; trial < trialCount; trial++)
                {
                    double x = 0, y = 0;
                    for (int i = 0; i < groupsPerTrial; i++) { var v = sums[trial * groupsPerTrial + i]; x += v.x; y += v.y; }
                    Assert.That(double.IsNaN(x) || double.IsInfinity(x) || double.IsNaN(y) || double.IsInfinity(y), Is.False);
                    error += (x - real) * (x - real) + (y - imag) * (y - imag); meanReal += x; meanImag += y;
                    trials.WriteLine(Csv(scene, hz, guided, methods[method], samples, repeat, trial, x, y));
                }
                meanReal /= trialCount; meanImag /= trialCount;
                double bias = Math.Sqrt((meanReal - real) * (meanReal - real) + (meanImag - imag) * (meanImag - imag));
                Assert.That(bias, Is.LessThan(6 * Math.Sqrt(variance / (samples * trialCount)) + 1e-7), "Estimator mean differs from enumerated target.");
                metrics.WriteLine(Csv(scene, hz, n, guided, methods[method], samples, trialCount, repeat,
                    buildMs, uploadMs, elapsed, sums.Length * 8, real, imag, Math.Sqrt(variance / samples),
                    Math.Sqrt(error / trialCount), meanReal, meanImag)); metrics.Flush();
            }
        }
        trials.Flush();
    }
}
