using System;
using System.Collections.Generic;
using System.IO;
using NUnit.Framework;
using UnityEngine;

public partial class TakeChecks
{
    [Serializable] sealed class WaveWall { public float[] center, half; }
    [Serializable] sealed class WaveScene
    {
        public string physicalModel;
        public float[] source, points, frequencies;
        public WaveWall[] walls;
        public int[] rays, seeds;
        public int sampleRate, fftSize, maxBounces;
        public float speedOfSound, maximumDelaySeconds;
    }
    static Vector3 WaveVector(float[] values, int offset=0) =>
        new Vector3(values[offset], values[offset+1], values[offset+2]);

    // Opt-in measurement, not a test that declares the current renderer physically correct.
    [Test, Category("QualityMeasurement")]
    public void MeasureWaveField()
    {
        string output=Environment.GetEnvironmentVariable("ACOUSTIC_WAVE_OUTPUT");
        if(string.IsNullOrEmpty(output))Assert.Ignore("Set ACOUSTIC_WAVE_OUTPUT to a prepared pressure-map directory.");
        Assert.That(Application.isBatchMode,Is.True,"Use an isolated batch project.");
        AudioListener.volume=0;
        var config=JsonUtility.FromJson<WaveScene>(File.ReadAllText(Path.Combine(output,"scene.json")));
        Assert.That(config.physicalModel,Is.Null.Or.Empty.Or.EqualTo("point-extruded"),"Unsupported source/geometry model.");
        var scene=Scene(1+config.walls.Length);scene.listeners=new[]{1};
        if(config.physicalModel=="point-extruded")
        {
            scene.dimension=AcousticTake.Dimension.Planar2D;
            Assert.That(config.source[2],Is.Zero);
            for(int p=2;p<config.points.Length;p+=3)Assert.That(config.points[p],Is.Zero);
        }
        var state=States(1+config.walls.Length);state[0].position=WaveVector(config.source);
        scene.materials=new[]{new AcousticTake.MaterialDefinition {name="rigid",frequencies=new[]{0f},
            absorption=new[]{0f},scattering=new[]{0f},transmission=new[]{0f}}};
        var geometry=new List<AcousticTake.Primitive>();
        for(int w=0;w<config.walls.Length;w++)
        {
            int owner=w+2;var wall=config.walls[w];
            state[owner].position=WaveVector(wall.center);
            state[owner].flags=AcousticTake.StateFlags.Active|AcousticTake.StateFlags.Geometry;
            var mesh=TransportBox(scene.dimension,owner,0,1);
            scene.objects[owner].primitiveStart=geometry.Count;scene.objects[owner].primitiveCount=mesh.Length;
            foreach(var triangle in mesh)
            {
                var p=triangle;var half=WaveVector(wall.half);
                p.a=Vector3.Scale(p.a,half);p.b=Vector3.Scale(p.b,half);p.c=Vector3.Scale(p.c,half);
                geometry.Add(p);
            }
        }
        using var csv=new StreamWriter(Path.Combine(output,"renderer.csv"));
        csv.WriteLine("diffraction,rays,seed,point,hz,real,imag,solve_seconds,direct_visible");
        var shader=RequireGpu();var direct=new TakeRender.DirectPath[1];
        foreach(int rays in config.rays)foreach(int seed in config.seeds)foreach(int diffraction in new[]{0,1})
        {
            var settings=EdgeSettings();settings.useBvh=true;settings.airAbsorption=false;
            settings.fftSize=config.fftSize;settings.spectralHop=config.fftSize/2;
            settings.raysPerBin=rays;settings.seed=seed;settings.maxBounces=config.maxBounces;
            settings.diffractionFactor=diffraction;settings.speedOfSound=config.speedOfSound;
            settings.maximumDelaySeconds=config.maximumDelaySeconds;settings.minimumDistance=.01f;
            using var solver=new TransportSolver(scene,geometry.ToArray(),settings,shader,sampleRate:config.sampleRate);
            state[1].position=WaveVector(config.points);solver.Solve(state,direct); // Exclude cold dispatch.
            for(int point=0;point<config.points.Length/3;point++)
            {
                state[1].position=WaveVector(config.points,point*3);
                long started=System.Diagnostics.Stopwatch.GetTimestamp();solver.Solve(state,direct);
                var paths=solver.RayPaths;double seconds=TakeRender.Elapsed(started);
                foreach(float hz in config.frequencies)
                {
                    float bin=hz*config.fftSize/config.sampleRate;
                    Assert.That(bin,Is.EqualTo(Mathf.Round(bin)),"Reference frequencies must be exact configured FFT bins.");
                    var p=EdgeResponse(paths,(int)bin,hz);
                    Assert.That(double.IsFinite(p.Real)&&double.IsFinite(p.Imaginary),Is.True);
                    csv.WriteLine(FormattableString.Invariant($"{diffraction},{rays},{seed},{point},{hz:R},{p.Real:R},{p.Imaginary:R},{seconds:R},{(direct[0].pressureGain!=0?1:0)}"));
                }
            }
            csv.Flush();Debug.Log($"Wave field: diffraction={diffraction}, rays={rays}, seed={seed}");
        }
    }
}
