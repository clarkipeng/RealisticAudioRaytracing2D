using NUnit.Framework;
using System;
using UnityEngine;

public partial class TakeChecks
{
    [TestCase("{\"transport\":1}",TakeRender.Synthesis.Spectral)]
    [TestCase("{\"synthesis\":0}",TakeRender.Synthesis.ImpulseResponse)]
    [TestCase("{\"synthesis\":1}",TakeRender.Synthesis.Spectral), Category("Core")]
    public void SynthesisSettingsUseDefaultOrExplicitSelection(string json,TakeRender.Synthesis expected)
    {
        Assert.That(JsonUtility.FromJson<TakeRender.Settings>(json).synthesis,Is.EqualTo(expected));
    }

    [Test, Category("GPU")]
    public void SourceAndListenerDoppler()
    {
        var shader=RequireGpu();
        const int rate=8192;const float speed=343;
        var motions=new[]{new Vector2(34.3f,0),new Vector2(-34.3f,0),new Vector2(0,34.3f),new Vector2(0,-34.3f),new Vector2(34.3f,34.3f)};
        foreach(var dimension in new[]{AcousticTake.Dimension.Planar2D,AcousticTake.Dimension.Spatial3D})foreach(var motion in motions)
        {
            var scene=Scene();scene.dimension=dimension;var state=States();state[0].position=Vector3.zero;state[1].position=new Vector3(100,0,0);
            state[0].velocity=new Vector3(motion.x,0,0);state[1].velocity=new Vector3(motion.y,0,0);
            state[0].flags|=AcousticTake.StateFlags.VelocityValid;state[1].flags|=AcousticTake.StateFlags.VelocityValid;
            var settings=SpectralSettings();settings.speedOfSound=speed;settings.maximumDelaySeconds=2;settings.raysPerBin=1;
            var cpu=new TakeRender.DirectPath[1];var gpu=new TakeRender.DirectPath[1];
            TakeRender.SolveDirect(scene,Array.Empty<AcousticTake.Primitive>(),state,settings,cpu);
            using var solver=new TransportSolver(scene,Array.Empty<AcousticTake.Primitive>(),settings,shader,sampleRate:rate);
            solver.Solve(state,gpu);double ratio=(speed-motion.y)/(speed-motion.x),delay=100/(speed-motion.x);
            Near(cpu[0].delaySeconds,(float)delay,"Retarded source travel time CPU",1e-6f);
            Near(gpu[0].delaySeconds,(float)delay,"Retarded source travel time GPU",1e-6f);
            Near(1-cpu[0].direction.w,(float)ratio,"Source and listener radial factors CPU",1e-6f);
            Near(1-gpu[0].direction.w,(float)ratio,"Source and listener radial factors GPU",1e-6f);
            var arrivals=solver.RayPaths;
            Near(arrivals[1].delaySeconds,(float)delay,"Spectral retarded travel time",1e-6f);
            Near(1-arrivals[1].direction.w,(float)ratio,"Spectral motion factor",1e-6f);
            if(dimension==AcousticTake.Dimension.Spatial3D)
            {
                foreach(bool direct in new[]{false,true})
                {
                    var audio=SpectralSettings();audio.speedOfSound=speed;audio.maximumDelaySeconds=2;audio.raysPerBin=1;
                    audio.transport=direct?TakeRender.Transport.Direct:TakeRender.Transport.Raytraced;audio.transitionSamples=0;
                    CheckMotionTone(scene,state,Array.Empty<AcousticTake.Primitive>(),audio,shader,motion,ratio,false);
                    if(motion.x==34.3f && motion.y==0)
                    {
                        audio.delayFilter=TakeRender.DelayFilter.WindowedSinc;
                        CheckMotionTone(scene,state,Array.Empty<AcousticTake.Primitive>(),audio,shader,motion,ratio,false);
                    }
                }
            }
        }
    }

    [TestCase(false, Category="GPU")]
    [TestCase(true, Category="GPU")]
    public void MovingMirrorDoppler(bool bvh)
    {
        var shader=RequireGpu();
        const int rate=8192;const float speed=343;
        foreach(float wallSpeed in new[]{-34.3f,34.3f})
        {
            var scene=Scene(2);scene.listeners=new[]{1};scene.objects[2].primitiveCount=1;
            scene.materials=new[]{new AcousticTake.MaterialDefinition{frequencies=new[]{0f},absorption=new[]{0f},scattering=new[]{0f},transmission=new[]{0f},ior=new[]{1f},damping=new[]{0f}}};
            var state=States(2);state[0].position=state[1].position=new Vector3(-100,0,0);state[2].position=Vector3.zero;
            state[2].velocity=new Vector3(wallSpeed,0,0);state[2].flags=AcousticTake.StateFlags.Active|AcousticTake.StateFlags.Geometry|AcousticTake.StateFlags.VelocityValid;
            var geometry=new[]{new AcousticTake.Primitive{objectIndex=2,a=new Vector3(0,-100,-100),b=new Vector3(0,100,-100),c=new Vector3(0,0,100)}};
            var settings=SpectralSettings();settings.useBvh=bvh;settings.speedOfSound=speed;settings.maximumDelaySeconds=2;settings.raysPerBin=1;
            using var solver=new TransportSolver(scene,geometry,settings,shader,sampleRate:rate);solver.Solve(state,new TakeRender.DirectPath[1]);var rays=solver.RayPaths;
            double ratio=(speed-wallSpeed)/(speed+wallSpeed),delay=200/(speed+wallSpeed);int mirror=settings.FrequencyBins;
            Near(rays[mirror].delaySeconds,(float)delay,"Moving mirror retarded round trip",1e-6f);
            Near(1-rays[mirror].direction.w,(float)ratio,"Moving mirror double Doppler",1e-6f);
            CheckMotionTone(scene,state,geometry,settings,shader,new Vector2(wallSpeed,0),ratio,true);
        }
    }

    [Test, Category("GPU")]
    public void MaterialSpeedDoppler()
    {
        var shader=RequireGpu();
        const int rate=8192;const float speed=343;
        // A source wholly inside a material uses that material's speed in both
        // the retarded delay and Doppler denominator, not the configured air speed.
        foreach(var dimension in new[]{AcousticTake.Dimension.Planar2D,AcousticTake.Dimension.Spatial3D})
        {
            var scene=Scene(2);scene.dimension=dimension;scene.listeners=new[]{1};
            var geometry=TransportBox(dimension,2,0,1000);scene.objects[2].primitiveCount=geometry.Length;
            scene.materials=new[]{new AcousticTake.MaterialDefinition{frequencies=new[]{0f},absorption=new[]{0f},scattering=new[]{0f},transmission=new[]{1f},ior=new[]{2f},damping=new[]{0f}}};
            var state=States(2);state[0].position=state[2].position=Vector3.zero;state[1].position=Vector3.right*100;
            state[0].velocity=Vector3.right*17.15f;state[0].flags|=AcousticTake.StateFlags.VelocityValid;
            state[2].flags=AcousticTake.StateFlags.Active|AcousticTake.StateFlags.Geometry;
            var settings=SpectralSettings();settings.speedOfSound=speed;settings.maximumDelaySeconds=2;settings.raysPerBin=1;
            using var solver=new TransportSolver(scene,geometry,settings,shader,sampleRate:rate);solver.Solve(state,new TakeRender.DirectPath[1]);
            var paths=solver.RayPaths;
            Near(paths[1].delaySeconds,100/(171.5f-17.15f),"Material-speed retarded source",1e-6f);
            Near(1-paths[1].direction.w,171.5f/(171.5f-17.15f),"Material-speed Doppler",1e-6f);
        }
    }

    [Test, Category("GPU")]
    public void TwoMediumRetardedRoot()
    {
        var shader=RequireGpu();
        const int rate=8192;const float speed=343;
        // This two-medium root has fixed-point slope -0.9, so 32 iterations
        // cannot resolve it. The analytic retarded source remains inside the solid.
        {
            var scene=Scene(2);scene.listeners=new[]{1};var geometry=TransportBox(AcousticTake.Dimension.Spatial3D,2,0,1000);
            scene.objects[2].primitiveCount=geometry.Length;
            scene.materials=new[]{new AcousticTake.MaterialDefinition{frequencies=new[]{0f},absorption=new[]{0f},scattering=new[]{0f},transmission=new[]{1f},ior=new[]{2f},damping=new[]{0f}}};
            var state=States(2);state[0].position=state[2].position=Vector3.zero;state[1].position=Vector3.right*1500;
            state[0].velocity=Vector3.left*450;state[0].flags|=AcousticTake.StateFlags.VelocityValid;
            state[2].flags=AcousticTake.StateFlags.Active|AcousticTake.StateFlags.Geometry;
            var settings=SpectralSettings();settings.speedOfSound=1000;settings.maximumDelaySeconds=4;settings.raysPerBin=1;
            using var solver=new TransportSolver(scene,geometry,settings,shader,sampleRate:rate);solver.Solve(state,new TakeRender.DirectPath[1]);
            var paths=solver.RayPaths;
            Near(paths[1].delaySeconds,2.5f/1.9f,"Bracketed retarded root through two media",2e-6f);
            Near(1-paths[1].direction.w,1/1.9f,"Two-medium emitted-frequency factor",1e-6f);
        }
    }

    [Test, Category("Core")]
    public void DopplerRejectsAliasing()
    {
        // Pitch-up must attenuate content that would fold above output Nyquist in sinc mode.
        var tone=new float[128];for(int n=0;n<tone.Length;n++)tone[n]=Mathf.Sin(2*Mathf.PI*.375f*n);
        var signal=new TakeRender.Signal(tone);signal.SetActive(0,true);signal.Append(new AcousticTake.VoiceEvent{action=AcousticTake.VoiceAction.Play,gain=1,loop=true});
        double energy=0;for(int n=0;n<256;n++){float v=Doppler.Read(signal,500.25+1.5*n,1.5,TakeRender.DelayFilter.WindowedSinc,32);energy+=v*v;}
        Check(Math.Sqrt(energy/256)<.01,"Doppler sinc low-pass rejects a tone shifted above Nyquist.");
    }
    void CheckMotionTone(AcousticTake.Scene scene,AcousticTake.State[] initial,AcousticTake.Primitive[] geometry,
        TakeRender.Settings settings,ComputeShader shader,Vector2 motion,double ratio,bool mirrorOnly)
    {
        const int rate=8192,total=16384;const double frequency=512;
        settings.responseStepSamples=128;settings.transitionSamples=0;
        var pcm=new float[128];for(int n=0;n<pcm.Length;n++)pcm[n]=Mathf.Sin((float)(2*Math.PI*frequency*n/rate));
        var signal=new TakeRender.Signal(pcm);signal.SetActive(0,true);signal.Append(new AcousticTake.VoiceEvent{action=AcousticTake.VoiceAction.Play,gain=1,loop=true});
        var mixer=new TakeRender.Mixer(scene,rate,settings,new[]{signal});var state=(AcousticTake.State[])initial.Clone();
        using var solver=new TransportSolver(scene,geometry,settings,shader,sampleRate:rate);

        var direct=new TakeRender.DirectPath[1];var block=new float[128];var output=new float[total];
        for(int sample=0;sample<total;sample+=128)
        {
            double seconds=(double)sample/rate;
            for(int i=0;i<state.Length;i++)state[i].position=initial[i].position+initial[i].velocity*(float)seconds;
            solver.Solve(state,direct);var arrivals=solver.RayPaths;
            if(mirrorOnly)
                for(int slot=0;slot<arrivals.Length;slot++)if(slot<settings.FrequencyBins || (slot-settings.FrequencyBins)%(settings.maxBounces+1)!=0)
                    arrivals[slot]=default;
            if(settings.transport==TakeRender.Transport.Direct)mixer.SetResponse(sample,direct,state);
            else mixer.SetResponse((long)sample,direct,state,false,arrivals,0);
            mixer.Render(sample,128,block);Array.Copy(block,0,output,sample,128);
        }
        int crossings=0;double re=0,im=0,energy=0;
        for(int n=8193;n<total;n++)
        {
            if(output[n-1]<0 && output[n]>=0)crossings++;
            double angle=2*Math.PI*frequency*ratio*n/rate;re+=output[n]*Math.Cos(angle);im+=output[n]*Math.Sin(angle);energy+=output[n]*output[n];
        }
        Check(Math.Abs(crossings-frequency*ratio)<=2,"Analytic Doppler tone: "+settings.transport+" mirror="+mirrorOnly+" velocities="+motion+" actual="+crossings+" expected="+frequency*ratio);
        Check((re*re+im*im)/(8191*energy)>.40,"Motion tone stays phase coherent without response fades.");
    }
}
