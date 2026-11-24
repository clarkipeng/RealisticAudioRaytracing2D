using System;
using System.Collections.Generic;
using NUnit.Framework;
using UnityEngine;

// The merged walk (README part 56): every pair of a frame walked in one dispatch must give the outputs of walking the pairs
// one by one, bit for bit (the histograms are integer sums and every ray carries its own random sequence).
public partial class TakeChecks
{
    [Test, Category("GPU"), Category("Methods"), Timeout(900000)]
    public void PathMergedWalkIsExact()
    {
        ApertureRoom(true, false, ApertureOutside, out var scene, out var geometry, out var state, transmission: .05f, roomScattering: .5f);
        // Two sources and two listeners: four pairs, each with its own ends, edge distribution, pools and ring.
        int source2 = scene.objects.Length, listener2 = source2 + 1;
        var objects = new List<AcousticTake.ObjectDefinition>(scene.objects)
        {
            new AcousticTake.ObjectDefinition {id = "source-2", name = "Source 2", scale = Vector3.one},
            new AcousticTake.ObjectDefinition {id = "listener-2", name = "Listener 2", scale = Vector3.one}
        };
        scene.objects = objects.ToArray();
        scene.sources = new[] {scene.sources[0], new AcousticTake.SourceDefinition {objectIndex = source2, clipIndex = 0}};
        scene.listeners = new[] {1, listener2};
        var states = new List<AcousticTake.State>(state)
        {
            new AcousticTake.State {position = new Vector3(1.5f, 1, -1), rotation = Quaternion.identity, flags = AcousticTake.StateFlags.Active | AcousticTake.StateFlags.Source},
            new AcousticTake.State {position = new Vector3(-.5f, -1, 1), rotation = Quaternion.identity, flags = AcousticTake.StateFlags.Active | AcousticTake.StateFlags.Listener}
        };
        state = states.ToArray();
        foreach (bool moving in new[] {false, true})
        {
            string Hash(bool merged)
            {
                var settings = ControlSettings(TakeRender.Method.Path, 4096, .3f); settings.diffractionFactor = 1;
                settings.transport = TakeRender.Transport.Raytraced; settings.synthesis = TakeRender.Synthesis.Spectral;
                settings.pathPersistent = true; settings.responseStepSamples = 256; settings.pathPopulation = 8000;
                using var solver = new TransportSolver(scene, geometry, settings, RequireGpu(), sampleRate: 8000, methodAudio: true);
                var m = solver.Method; m.MergedWalk = merged;
                Assert.That(m.Pairs, Is.EqualTo(4));
                using var sha = System.Security.Cryptography.SHA256.Create();
                var posed = (AcousticTake.State[])state.Clone();
                for (int n = 0; n < 12; n++)
                {
                    if (moving)
                    {
                        posed[0].velocity = new Vector3(20, 0, 5); posed[0].flags |= AcousticTake.StateFlags.VelocityValid;
                        posed[0].position = state[0].position + posed[0].velocity * (n * .032f);
                    }
                    solver.Solve(posed, new TakeRender.DirectPath[m.Pairs]);
                    var words = new List<uint>(m.Late); words.AddRange(m.LateRates); words.AddRange(m.Counts);
                    var bytes = new byte[words.Count * 4 + solver.RayPaths.Length * RayTransport.PathStride];
                    Buffer.BlockCopy(words.ToArray(), 0, bytes, 0, words.Count * 4);
                    var paths = System.Runtime.InteropServices.MemoryMarshal.AsBytes(new ReadOnlySpan<RayTransport.Path>(solver.RayPaths));
                    paths.CopyTo(new Span<byte>(bytes, words.Count * 4, paths.Length));
                    sha.TransformBlock(bytes, 0, bytes.Length, null, 0);
                }
                if (moving) Assert.That(m.Doppler, "the moving source runs the Doppler walk");
                Assert.That(m.MergedFrames, merged ? Is.EqualTo(12) : Is.EqualTo(0));
                for (int pair = 0; pair < m.Pairs; pair++)
                {
                    long heard = 0; for (int k = 0; k < m.TimeBins.Count; k++) heard += m.HistogramCount(0, pair, k);
                    Assert.That(heard, Is.GreaterThan(0), $"pair {pair} hears the room");
                }
                AssertNoOverflow(solver);
                sha.TransformFinalBlock(Array.Empty<byte>(), 0, 0);
                return BitConverter.ToString(sha.Hash).Replace("-", "");
            }
            string apart = Hash(false), merged = Hash(true);
            TestContext.WriteLine($"{(moving ? "moving" : "static")}: pair by pair {apart}, merged {merged}");
            Assert.That(merged, Is.EqualTo(apart));
        }
    }
}
