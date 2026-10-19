using System;
using System.Collections.Generic;
using System.IO;
using NUnit.Framework;
using UnityEngine;

public partial class TakeChecks
{
    [Serializable] sealed class WaveWall { public float[] center, half; }
    [Serializable] sealed class WaveMotion { public float[] velocity; }
    [Serializable] sealed class WaveScene
    {
        public string physicalModel;
        public float[] source, points, frequencies;
        public WaveWall[] walls;
        public int[] rays, seeds;
        public int sampleRate, fftSize, maxBounces;
        public float speedOfSound, maximumDelaySeconds;
        public bool decompose;
        public float[] triangles;
        public int[] triangleMaterials;
        public AcousticTake.MaterialDefinition[] materials;
        public WaveMotion receiverMotion;
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
        string compiled=Path.Combine(output,"renderer-scene.json");
        var config=JsonUtility.FromJson<WaveScene>(File.ReadAllText(File.Exists(compiled)?compiled:Path.Combine(output,"scene.json")));
        Assert.That(config.physicalModel,Is.Null.Or.Empty.Or.EqualTo("point-extruded").Or.EqualTo("point-room-3d"),"Unsupported source/geometry model.");
        int owners=config.triangles==null?config.walls.Length:1;
        var scene=Scene(1+owners);scene.listeners=new[]{1};
        if(config.physicalModel=="point-extruded")
        {
            scene.dimension=AcousticTake.Dimension.Planar2D;
            Assert.That(config.source[2],Is.Zero);
            for(int p=2;p<config.points.Length;p+=3)Assert.That(config.points[p],Is.Zero);
        }
        var state=States(1+owners);state[0].position=WaveVector(config.source);
        if(config.receiverMotion?.velocity!=null)
        {state[1].velocity=WaveVector(config.receiverMotion.velocity);state[1].flags|=AcousticTake.StateFlags.VelocityValid;}
        scene.materials=config.materials??new[]{new AcousticTake.MaterialDefinition {name="rigid",frequencies=new[]{0f},
            absorption=new[]{0f},scattering=new[]{0f},transmission=new[]{0f}}};
        var geometry=new List<AcousticTake.Primitive>();
        if(config.triangles!=null)
        {
            Assert.That(config.triangles.Length,Is.EqualTo(config.triangleMaterials.Length*9));
            state[2].position=Vector3.zero;state[2].flags=AcousticTake.StateFlags.Active|AcousticTake.StateFlags.Geometry;
            var bounds=new Bounds(WaveVector(config.triangles),Vector3.zero);
            for(int triangle=0;triangle<config.triangleMaterials.Length;triangle++)
            {
                // FEM normals leave the air. The tracer needs normals leaving the solid.
                geometry.Add(new AcousticTake.Primitive {objectIndex=2,materialIndex=config.triangleMaterials[triangle],
                    a=WaveVector(config.triangles,9*triangle),b=WaveVector(config.triangles,9*triangle+6),c=WaveVector(config.triangles,9*triangle+3)});
                var p=geometry[triangle];bounds.Encapsulate(p.a);bounds.Encapsulate(p.b);bounds.Encapsulate(p.c);
            }
            // Close the opaque solid outside the air cavity. These outer faces are
            // inaccessible from inside; they preserve solid winding and wedge angles.
            foreach(var face in TransportBox(scene.dimension,2,0,1))
            {
                var p=face;var half=bounds.extents+Vector3.one;
                p.a=bounds.center+Vector3.Scale(p.a,half);p.b=bounds.center+Vector3.Scale(p.b,half);p.c=bounds.center+Vector3.Scale(p.c,half);
                geometry.Add(p);
            }
            scene.objects[2].primitiveCount=geometry.Count;
        }
        for(int w=0;config.triangles==null && w<config.walls.Length;w++)
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
        using var terms=config.decompose?new StreamWriter(Path.Combine(output,"path-orders.csv")):null;
        terms?.WriteLine("diffraction,rays,seed,point,hz,slot,real,imag,arrivals,maximum_contribution");
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
                    if(terms!=null)for(int slot=0;slot<settings.maxBounces+2;slot++)
                    {
                        var value=System.Numerics.Complex.Zero;int count=0;double largest=0;
                        for(int i=0;i<paths.Length;i++)
                        {
                            int kind=i<settings.FrequencyBins?0:1+(i-settings.FrequencyBins)%(settings.maxBounces+1);
                            var path=paths[i];if(kind!=slot||path.frequencyBin!=(int)bin)continue;
                            var gain=new System.Numerics.Complex(path.pressureGain,path.pressureImag);
                            if(gain==System.Numerics.Complex.Zero)continue;
                            value+=gain*System.Numerics.Complex.Exp(new System.Numerics.Complex(0,-2*Math.PI*hz*path.delaySeconds));
                            count++;largest=Math.Max(largest,gain.Magnitude);
                        }
                        terms.WriteLine(FormattableString.Invariant($"{diffraction},{rays},{seed},{point},{hz:R},{slot},{value.Real:R},{value.Imaginary:R},{count},{largest:R}"));
                    }
                }
            }
            csv.Flush();Debug.Log($"Wave field: diffraction={diffraction}, rays={rays}, seed={seed}");
        }
    }

    // Analytic direct + first-image control, not a full reverberation/wave reference.
    [Test, Category("QualityMeasurement")]
    public void MeasureRoomMaterialMotion()
    {
        string output=Environment.GetEnvironmentVariable("ACOUSTIC_WAVE_OUTPUT");
        if(string.IsNullOrEmpty(output))Assert.Ignore("Set ACOUSTIC_WAVE_OUTPUT for the silent room control.");
        Assert.That(Application.isBatchMode,Is.True);AudioListener.volume=0;
        float[] frequencies={125,250,500};
        float[][] absorption={new[]{0f,0f,0f},new[]{.2f,.2f,.2f},new[]{.6f,.6f,.6f},new[]{.9f,.9f,.9f},new[]{.1f,.4f,.8f}};
        var half=new Vector3(3,2,1.5f);var velocity=new Vector3(1.25f,0,0);var shader=RequireGpu();
        using var csv=new StreamWriter(Path.Combine(output,"room-material-motion.csv"));
        csv.WriteLine("dimension,material,frame,time,hz,absorption,real,imag,reference_real,reference_imag,solve_seconds");
        foreach(var dimension in new[]{AcousticTake.Dimension.Planar2D,AcousticTake.Dimension.Spatial3D})
        for(int material=0;material<absorption.Length;material++)
        {
            var scene=Scene(2);scene.dimension=dimension;scene.listeners=new[]{1};
            scene.materials=new[]{new AcousticTake.MaterialDefinition {name="synthetic-"+material,
                frequencies=frequencies,absorption=absorption[material],scattering=new float[3],transmission=new float[3],
                ior=new[]{1f,1f,1f},damping=new float[3]}};
            var geometry=TransportBox(dimension,2,0,1);scene.objects[2].primitiveCount=geometry.Length;
            for(int i=0;i<geometry.Length;i++)
            {geometry[i].a=Vector3.Scale(geometry[i].a,half);geometry[i].b=Vector3.Scale(geometry[i].b,half);geometry[i].c=Vector3.Scale(geometry[i].c,half);}
            var state=States(2);state[0].position=new Vector3(-1.5f,-.5f,0);state[2].position=Vector3.zero;
            state[2].flags=AcousticTake.StateFlags.Active|AcousticTake.StateFlags.Geometry;
            state[1].velocity=velocity;state[1].flags|=AcousticTake.StateFlags.VelocityValid;
            var images=new List<Vector3>{state[0].position};
            for(int axis=0;axis<(dimension==AcousticTake.Dimension.Planar2D?2:3);axis++)
            foreach(int side in new[]{-1,1})
            {var image=state[0].position;image[axis]=2*side*half[axis]-image[axis];images.Add(image);}
            var settings=EdgeSettings();settings.useBvh=true;settings.airAbsorption=false;settings.diffractionFactor=0;
            settings.fftSize=8;settings.spectralHop=4;settings.maxBounces=1;settings.raysPerBin=geometry.Length;
            settings.maximumDelaySeconds=.1f;settings.speedOfSound=343;settings.minimumDistance=.01f;
            using var solver=new TransportSolver(scene,geometry,settings,shader,sampleRate:1000);
            var direct=new TakeRender.DirectPath[1];state[1].position=new Vector3(-1,.4f,0);solver.Solve(state,direct);
            double error=0,energy=0;
            for(int frame=0;frame<=32;frame++)
            {
                float time=frame/16f;state[1].position=new Vector3(-1,.4f,0)+velocity*time;
                long start=System.Diagnostics.Stopwatch.GetTimestamp();solver.Solve(state,direct);
                var paths=solver.RayPaths;double seconds=TakeRender.Elapsed(start);
                for(int f=0;f<frequencies.Length;f++)
                {
                    float hz=frequencies[f];var expected=System.Numerics.Complex.Zero;
                    for(int image=0;image<images.Count;image++)
                    {
                        double distance=Vector3.Distance(images[image],state[1].position);
                        double gain=(image==0?1:Math.Sqrt(1-absorption[material][f]))/distance;
                        expected+=System.Numerics.Complex.FromPolarCoordinates(gain,-2*Math.PI*hz*distance/343);
                    }
                    var actual=EdgeResponse(paths,(int)(hz*8/1000),hz);
                    error+=(actual-expected).Magnitude*(actual-expected).Magnitude;energy+=expected.Magnitude*expected.Magnitude;
                    csv.WriteLine(FormattableString.Invariant($"{dimension},{material},{frame},{time:R},{hz:R},{absorption[material][f]:R},{actual.Real:R},{actual.Imaginary:R},{expected.Real:R},{expected.Imaginary:R},{seconds:R}"));
                }
            }
            csv.Flush();Assert.That(Math.Sqrt(error/energy),Is.LessThan(.0001),"Early room images and synthetic material pressure must match within 0.01%.");
        }
    }
}
