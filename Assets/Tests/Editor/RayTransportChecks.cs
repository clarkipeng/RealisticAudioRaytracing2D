using System;
using System.Collections.Generic;
using System.IO;
using System.Reflection;
using UnityEditor;
using UnityEngine;

public static partial class TakeChecks
{
    // Read the private GPU response contract without widening the production API for tests.
    static Array Arrivals(DirectSolver solver) => (Array)typeof(DirectSolver).GetProperty("RayPaths",BindingFlags.Instance|BindingFlags.NonPublic).GetValue(solver);
    static T ArrivalField<T>(Array paths,int index,string field) => (T)paths.GetValue(index).GetType().GetField(field).GetValue(paths.GetValue(index));
    static TakeRender.Settings SpectralSettings() => new TakeRender.Settings {transport=TakeRender.Transport.Raytraced,
        responseStepSamples=128,fftSize=128,spectralHop=32,raysPerPair=260,maxBounces=3,maximumDelaySeconds=.1f,
        maximumResponsesPerBatch=4,speedOfSound=8000,minimumDistance=1,airAbsorption=false,diffractionFactor=0,tailSamples=256};

    [MenuItem("Tools/Acoustics/Check directional spectral transport")]
    public static void RunTransport()
    {
        checks=0;directory=Path.Combine(Path.GetTempPath(),"acoustic-transport-"+Guid.NewGuid().ToString("N"));Directory.CreateDirectory(directory);
        try
        {
            var shader=AssetDatabase.LoadAssetAtPath<ComputeShader>("Assets/Shaders/Direct.compute");
            TransportMaterials();TransportGeometry(shader);TransportMedia(shader);TransportNativeVolumes(shader);SpectralIdentity(shader);SpectralDoppler(shader);SpectralTransitions(shader);SpectralHeadTurns(shader);
            SpectralStreaming(shader,AcousticTake.Dimension.Spatial3D);SpectralStreaming(shader,AcousticTake.Dimension.Planar2D);
            Debug.Log("ACOUSTIC_TRANSPORT_PASS: "+checks+" checks on "+SystemInfo.graphicsDeviceType);
        }
        finally {Directory.Delete(directory,true);}
    }
    static void TransportMaterials()
    {
        var scene=Scene();scene.materials=new[]{new AcousticTake.MaterialDefinition {name="glass",frequencies=new[]{100f,5000f},
            absorption=new[]{.1f,.2f},scattering=new[]{.3f,.4f},transmission=new[]{.8f,.6f},ior=new[]{.1f,.2f},damping=new[]{.01f,.02f}}};
        foreach(int version in new[]{1,2})
        {
            string path=Path.Combine(directory,"version-"+version);
            using(var writer=new AcousticTake.Writer(path,8000,scene,Array.Empty<AcousticTake.Primitive>(),new[]{new[]{1f}},version:version))
            {writer.Append(0,States());writer.Complete();}
            using var take=AcousticTake.Open(path);
            Check(take.Metadata.version==version,"Versioned material read.");
            if(version==1)Check(take.Metadata.scene.materials[0].transmission==null,"V1 does not authenticate new coefficients.");
            else Near(take.Metadata.scene.materials[0].transmission[1],.6f,"V2 transmission round trip");
        }
    }
    static void TransportGeometry(ComputeShader shader)
    {
        var scene=Scene(2);scene.listeners=new[]{1};scene.objects[2].primitiveCount=1;
        scene.materials=new[]{new AcousticTake.MaterialDefinition {name="wall",frequencies=new[]{0f},absorption=new[]{0f},scattering=new[]{.5f},transmission=new[]{0f},ior=new[]{1f},damping=new[]{0f}}};
        var state=States(2);state[0].position=new Vector3(-1,0,0);state[1].position=new Vector3(1,0,0);state[2].position=Vector3.zero;
        state[2].flags=AcousticTake.StateFlags.Active|AcousticTake.StateFlags.Geometry;
        var geometry=new[]{new AcousticTake.Primitive {objectIndex=2,a=new Vector3(0,-2,-2),b=new Vector3(0,2,-2),c=new Vector3(0,0,2)}};
        var settings=SpectralSettings();settings.speedOfSound=343;
        var direct=new TakeRender.DirectPath[1];
        foreach(var dimension in new[]{AcousticTake.Dimension.Spatial3D,AcousticTake.Dimension.Planar2D})
        {
            scene.dimension=dimension;
            using(var solver=new DirectSolver(scene,geometry,settings,shader,sampleRate:8000))
            {
                solver.Solve(state,direct);Near(direct[0].pressureGain,0,"Opaque diagnostic is blocked");
                var rays=Arrivals(solver);Near(ArrivalField<float>(rays,10,"pressureGain"),0,"Opaque spectral direct is blocked");
            }
            scene.materials[0].transmission[0]=1;
            using(var solver=new DirectSolver(scene,geometry,settings,shader,sampleRate:8000))
            {
                solver.Solve(state,direct);var rays=Arrivals(solver);
                Near(ArrivalField<float>(rays,10,"pressureGain"),.5f,"Transparent wall passes pressure");
            }
            scene.materials[0].transmission[0]=0;
            {
                settings.diffractionFactor=.5f;
                using var edgeSolver=new DirectSolver(scene,geometry,settings,shader,sampleRate:8000);
                edgeSolver.Solve(state,direct);var edgeRays=Arrivals(edgeSolver);double edgeEnergy=0;
                for(int i=65;i<edgeRays.Length;i++)edgeEnergy+=Math.Pow(ArrivalField<float>(edgeRays,i,"pressureGain"),2);
                Check(edgeEnergy>0,"Wavelength edge paths carry sound around an opaque geometry");settings.diffractionFactor=0;
            }
            state[1].position=new Vector3(-1,1,0);
            Array baseline=null;
            foreach(bool bvh in new[]{false,true})
            {
                settings.useBvh=bvh;
                using var solver=new DirectSolver(scene,geometry,settings,shader,2,8000);
                solver.Solve(state,direct);var rays=Arrivals(solver);double energy=0;
                for(int i=65;i<65+settings.raysPerPair*settings.maxBounces;i++)
                {
                    float value=ArrivalField<float>(rays,i,"pressureGain");Check(float.IsFinite(value),"Finite indirect pressure");energy+=value*value;
                    if(baseline!=null)Near(value,ArrivalField<float>(baseline,i,"pressureGain"),"BVH indirect agreement",1e-5f);
                }
                Check(energy>0,"Reflective wall produces indirect arrivals in "+dimension);
                if(!bvh)baseline=(Array)rays.Clone();
                // A moving wall must leave the static hierarchy and remain acoustically visible.
                state[2].position=Vector3.right;solver.Solve(state,direct);state[2].position=Vector3.zero;
                if(bvh)Check(solver.StaticPrimitiveCount==0,"Moving reflector leaves static tree");
            }
            state[1].position=Vector3.right;settings.useBvh=false;
            scene.materials[0].absorption[0]=1;
            using(var solver=new DirectSolver(scene,geometry,settings,shader,sampleRate:8000))
            {
                solver.Solve(state,direct);var rays=Arrivals(solver);
                for(int i=0;i<rays.Length;i++)Near(ArrivalField<float>(rays,i,"pressureGain"),0,"Absorber cannot create reflected pressure");
            }
            scene.materials[0].absorption[0]=0;
        }
        // World-space arrival direction survives transport; ear interpolation uses
        // each listener's orientation in the shared DSP, rather than GPU snapping.
        scene.dimension=AcousticTake.Dimension.Spatial3D;scene.objects[2].primitiveCount=0;
        foreach(float turn in new[]{0f,90f,180f,270f})
        {
            state[1].rotation=Quaternion.Euler(0,turn,0);
            using var solver=new DirectSolver(scene,Array.Empty<AcousticTake.Primitive>(),settings,shader,sampleRate:8000);
            solver.Solve(state,direct);var rays=Arrivals(solver);
            Near(Vector3.Distance((Vector3)ArrivalField<Vector4>(rays,10,"direction"),(Vector3)direct[0].direction),0,"Transport retains world arrival direction");
        }
    }
    static AcousticTake.Primitive[] TransportBox(AcousticTake.Dimension dimension,int owner,int material,float half)
    {
        var geometry=new List<AcousticTake.Primitive>();
        if(dimension==AcousticTake.Dimension.Planar2D)
        {
            var v=new[]{new Vector3(-half,-half,0),new Vector3(half,-half,0),new Vector3(half,half,0),new Vector3(-half,half,0)};
            for(int i=0;i<4;i++)geometry.Add(new AcousticTake.Primitive {objectIndex=owner,materialIndex=material,a=v[i],b=v[(i+1)%4]});
        }
        else
        {
            var v=new Vector3[8];for(int i=0;i<8;i++)v[i]=new Vector3((i&1)==0?-half:half,(i&2)==0?-half:half,(i&4)==0?-half:half);
            int[] indices={0,2,3,0,3,1,4,5,7,4,7,6,0,1,5,0,5,4,2,6,7,2,7,3,0,4,6,0,6,2,1,3,7,1,7,5};
            for(int i=0;i<indices.Length;i+=3)geometry.Add(new AcousticTake.Primitive {objectIndex=owner,materialIndex=material,a=v[indices[i]],b=v[indices[i+1]],c=v[indices[i+2]]});
        }
        return geometry.ToArray();
    }
    static void TransportMedia(ComputeShader shader)
    {
        foreach(var dimension in new[]{AcousticTake.Dimension.Planar2D,AcousticTake.Dimension.Spatial3D})foreach(bool bvh in new[]{false,true})
        {
            var scene=Scene(3);scene.dimension=dimension;scene.listeners=new[]{1};
            scene.materials=new[]{
                new AcousticTake.MaterialDefinition {frequencies=new[]{0f},absorption=new[]{0f},scattering=new[]{0f},transmission=new[]{1f},ior=new[]{.5f},damping=new[]{.1f}},
                new AcousticTake.MaterialDefinition {frequencies=new[]{0f},absorption=new[]{0f},scattering=new[]{0f},transmission=new[]{1f},ior=new[]{2f},damping=new[]{.3f}}};
            var outer=TransportBox(dimension,2,0,2);var inner=TransportBox(dimension,3,1,1);
            var geometry=new List<AcousticTake.Primitive>(outer);geometry.AddRange(inner);
            scene.objects[2].primitiveCount=outer.Length;scene.objects[3].primitiveStart=outer.Length;scene.objects[3].primitiveCount=inner.Length;
            var state=States(3);state[2].position=state[3].position=Vector3.zero;
            state[2].flags=state[3].flags=AcousticTake.StateFlags.Active|AcousticTake.StateFlags.Geometry;
            var settings=SpectralSettings();settings.speedOfSound=1000;settings.useBvh=bvh;settings.maxBounces=1;
            void Direct(float source,float listener,float time,float attenuation,string name)
            {
                state[0].position=Vector3.right*source;state[1].position=Vector3.right*listener;
                using var solver=new DirectSolver(scene,geometry.ToArray(),settings,shader,sampleRate:8000);
                solver.Solve(state,new TakeRender.DirectPath[1]);var rays=Arrivals(solver);
                Near(ArrivalField<float>(rays,10,"delaySeconds"),time,name+" travel time "+dimension,1e-7f);
                Near(ArrivalField<float>(rays,10,"pressureGain"),Mathf.Exp(-attenuation)/Mathf.Max(1,Mathf.Abs(listener-source)),name+" pressure loss",1e-5f);
            }
            Direct(-3,3,.007f,.8f,"Nested slabs restore outer medium");
            Direct(0,3,.0035f,.4f,"Source starts inside inner solid");
            Direct(-3,0,.0035f,.4f,"Listener ends inside inner solid");
            Direct(-.5f,.5f,.002f,.3f,"Both endpoints inside solid");
            state[3].flags=0;Direct(-3,3,.004f,.4f,"Inactive inner medium is absent");
            state[2].position=Vector3.right*5;Direct(-3,3,.006f,0,"Moved volume no longer intersects path");
            state[2].position=Vector3.zero;
            // Reverse all winding: geometry still bounds the same physical medium.
            for(int i=0;i<geometry.Count;i++){var p=geometry[i];(p.a,p.b)=(p.b,p.a);geometry[i]=p;}
            Direct(-3,3,.004f,.4f,"Reversed closed winding");
            // A sheet has zero thickness and must not turn the rest of space into its material.
            geometry.RemoveRange(1,geometry.Count-1);scene.objects[2].primitiveCount=1;scene.objects[3].primitiveCount=0;
            geometry[0]=new AcousticTake.Primitive {objectIndex=2,a=new Vector3(0,-4,-4),b=new Vector3(0,4,-4),c=new Vector3(0,0,4)};
            Direct(-3,3,.006f,0,"Open sheet leaves air speed unchanged");
            // A reflected path entirely inside one solid uses its speed and damping
            // for both legs, including the final connection to the listener.
            geometry=new List<AcousticTake.Primitive>(TransportBox(dimension,2,0,1));scene.objects[2].primitiveCount=geometry.Count;
            scene.materials[0].transmission[0]=0;state[0].position=state[1].position=Vector3.zero;
            using var reflected=new DirectSolver(scene,geometry.ToArray(),settings,shader,sampleRate:8000);
            reflected.Solve(state,new TakeRender.DirectPath[1]);var paths=Arrivals(reflected);int audible=0;
            for(int i=65;i<paths.Length;i++)
            {
                float gain=ArrivalField<float>(paths,i,"pressureGain");if(gain==0)continue;audible++;
                var direction=(Vector3)ArrivalField<Vector4>(paths,i,"direction");float cosine=Mathf.Max(Mathf.Abs(direction.x),Mathf.Abs(direction.y),Mathf.Abs(direction.z)),distance=1/cosine;
                Near(ArrivalField<float>(paths,i,"delaySeconds"),distance/1000,"Reflected interior travel time",1e-7f);
                Near(gain,.5f*cosine*(65f/settings.raysPerPair)*Mathf.Exp(-.1f*2*distance)/(2*distance),"Reflected interior damping",1e-5f);
            }
            Check(audible>0,"Interior reflection fixture exercised actual rays");
        }
    }
    static void TransportNativeVolumes(ComputeShader shader)
    {
        var material=ScriptableObject.CreateInstance<AudioMaterial>();
        try
        {
            foreach(var type in new[]{typeof(BoxCollider),typeof(SphereCollider),typeof(CapsuleCollider),typeof(BoxCollider2D),typeof(CircleCollider2D),typeof(CapsuleCollider2D)})
            {
                var obj=new GameObject("Volume fixture");
                try
                {
                    obj.AddComponent(type);var dimension=typeof(Collider2D).IsAssignableFrom(type)?AcousticTake.Dimension.Planar2D:AcousticTake.Dimension.Spatial3D;
                    var bindings=new List<RayTraceManager.TakeObjectBinding> {new RayTraceManager.TakeObjectBinding {id="solid",transform=obj.transform,geometry=true}};
                    foreach(float scale in new[]{1f,-1f})
                    {
                        obj.transform.localScale=new Vector3(scale,1,1);
                        // Curved 3D collider export requires uniform absolute scale.
                        var exported=SceneToTake.Compile(bindings,dimension,.01f,material,Array.Empty<Transform>(),Array.Empty<AudioClip>(),Array.Empty<Transform>(),out var geometry);
                        var scene=Scene(2);scene.dimension=dimension;scene.listeners=new[]{1};scene.objects[2].primitiveCount=geometry.Length;
                        for(int i=0;i<geometry.Length;i++)geometry[i].objectIndex=2;
                        scene.materials=new[]{new AcousticTake.MaterialDefinition {frequencies=new[]{0f,4000f},absorption=new[]{0f,0f},scattering=new[]{0f,0f},
                            transmission=new[]{1f,1f},ior=new[]{.5f,1.5f},damping=new[]{.1f,.1f}}};
                        var state=States(2);state[0].position=Vector3.down*.1f;state[1].position=Vector3.up*.1f;state[2].position=Vector3.zero;
                        state[2].flags=AcousticTake.StateFlags.Active|AcousticTake.StateFlags.Geometry;
                        var settings=SpectralSettings();settings.speedOfSound=1000;
                        using var solver=new DirectSolver(scene,geometry,settings,shader,sampleRate:8000);
                        solver.Solve(state,new TakeRender.DirectPath[1]);var paths=Arrivals(solver);
                        foreach(int bin in new[]{0,16,32,64})
                        {
                            Near(ArrivalField<float>(paths,bin,"delaySeconds"),.2f*(.5f+bin/64f)/1000,"Native "+type.Name+" volume, mirrored scale and frequency-dependent speed",1e-7f);
                            Near(ArrivalField<float>(paths,bin,"pressureGain"),Mathf.Exp(-.02f),"Native volume damping",1e-5f);
                        }
                    }
                }
                finally {UnityEngine.Object.DestroyImmediate(obj);}
            }
        }
        finally {UnityEngine.Object.DestroyImmediate(material);}
    }
    static void SpectralTransitions(ComputeShader shader)
    {
        var scene=Scene();var state=States();state[1].position=Vector3.zero;
        foreach(bool cut in new[]{false,true})
        {
            var settings=SpectralSettings();settings.responseStepSamples=512;settings.transitionSamples=64;
            var signal=new TakeRender.Signal(new[]{1f});signal.SetActive(0,true);signal.Append(new AcousticTake.VoiceEvent {action=AcousticTake.VoiceAction.Play,gain=1,loop=true});
            var mixer=new TakeRender.Mixer(scene,8000,settings,new[]{signal});
            using var solver=new DirectSolver(scene,Array.Empty<AcousticTake.Primitive>(),settings,shader,sampleRate:8000);
            var direct=new TakeRender.DirectPath[1];solver.Solve(state,direct);var rays=Arrivals(solver);
            var set=typeof(TakeRender.Mixer).GetMethod("SetResponse",BindingFlags.Instance|BindingFlags.NonPublic);
            set.Invoke(mixer,new object[]{0L,direct,state,false,rays,0});var output=new float[1024];mixer.Render(0,512,output);
            Array.Clear(rays,0,rays.Length);set.Invoke(mixer,new object[]{512L,direct,state,cut,rays,0});mixer.Render(512,512,output);
            for(int n=0;n<512;n++)
            {
                int sample=512+n;float expected=0;
                for(int frame=sample/settings.spectralHop*settings.spectralHop;frame>sample-settings.fftSize;frame-=settings.spectralHop)
                {
                    float gain=frame<512?1:cut?0:Mathf.Clamp01(1-(frame-512)/64f);
                    expected+=(1-Mathf.Cos(2*Mathf.PI*(sample-frame)/settings.fftSize))*settings.spectralHop/settings.fftSize*gain;
                }
                Near(output[n],expected,"Spectral response fade follows normalized overlapping windows; cut resets fade",2e-5f);
            }
        }
    }
    static void SpectralHeadTurns(ComputeShader shader)
    {
        var scene=Scene();var state=States();state[0].position=Vector3.forward;state[1].position=Vector3.zero;
        var settings=SpectralSettings();settings.listenerOutput=TakeRender.ListenerOutput.Binaural;settings.responseStepSamples=512;
        float[] Render(float turn)
        {
            state[1].rotation=Quaternion.Euler(0,turn,0);
            var signal=new TakeRender.Signal(new[]{1f});signal.SetActive(0,true);signal.Append(new AcousticTake.VoiceEvent {action=AcousticTake.VoiceAction.Play,gain=1});
            var mixer=new TakeRender.Mixer(scene,8000,settings,new[]{signal});
            using var solver=new DirectSolver(scene,Array.Empty<AcousticTake.Primitive>(),settings,shader,sampleRate:8000);
            var direct=new TakeRender.DirectPath[1];solver.Solve(state,direct);
            var set=typeof(TakeRender.Mixer).GetMethod("SetResponse",BindingFlags.Instance|BindingFlags.NonPublic);
            set.Invoke(mixer,new object[]{0L,direct,state,false,Arrivals(solver),0});var output=new float[1024];mixer.Render(0,512,output);return output;
        }
        var a=Render(2.499f);var b=Render(2.501f);float jump=0,signalPeak=0;
        for(int i=0;i<a.Length;i++){jump=Mathf.Max(jump,Mathf.Abs(a[i]-b[i]));signalPeak=Mathf.Max(signalPeak,Mathf.Abs(a[i]));}
        Check(signalPeak>.001f && jump<signalPeak*.001f,"Spectral ear output is continuous across the old nearest-direction boundary");
        a=Render(90);b=Render(-90);float symmetry=0;
        for(int i=0;i<a.Length;i+=2){symmetry=Mathf.Max(symmetry,Mathf.Abs(a[i]-b[i+1]));symmetry=Mathf.Max(symmetry,Mathf.Abs(a[i+1]-b[i]));}
        Near(symmetry,0,"Spectral head rotation preserves left/right convention",1e-6f);
    }
    static void SpectralIdentity(ComputeShader shader)
    {
        foreach(int fft in new[]{128,256})foreach(int divisor in new[]{2,4})
        {
            var settings=SpectralSettings();settings.fftSize=fft;settings.spectralHop=fft/divisor;settings.responseStepSamples=256;settings.tailSamples=512;
            var scene=Scene();var state=States();state[1].position=Vector3.zero;
            var pcm=new float[2048];for(int i=0;i<pcm.Length;i++)pcm[i]=.2f*Mathf.Sin(2*Mathf.PI*503*i/8000)+.1f*Mathf.Cos(2*Mathf.PI*1107*i/8000);
            string path=Path.Combine(directory,"identity-"+fft+"-"+divisor);
            using(var writer=new AcousticTake.Writer(path,8000,scene,Array.Empty<AcousticTake.Primitive>(),new[]{pcm}))
            {writer.Append(0,state);writer.Append(2047,state);writer.Append(new AcousticTake.VoiceEvent {sample=0,gain=1,action=AcousticTake.VoiceAction.Play});writer.Complete();}
            using var take=AcousticTake.Open(path);var report=TakeRender.RenderDirect(take,settings,path+"-render",shader:shader);
            var output=ReadWave(Path.Combine(path+"-render","listener-0.wav"));float error=0;
            for(int i=0;i<output.Length;i++){float expected=i>=fft && i<fft+pcm.Length?pcm[i-fft]:0;error=Mathf.Max(error,Mathf.Abs(output[i]-expected));}
            Near(error,0,"WOLA identity gain, latency and complete tail",2e-5f);Check(report.dspLatencySamples==fft,"Declared FFT latency");
        }
    }
    static void SpectralDoppler(ComputeShader shader)
    {
        foreach(int motion in new[]{-1,1})
        {
            var scene=Scene();var state=States();state[1].position=new Vector3(motion>0?10:80,0,0);
            float initial=state[1].position.x;var pcm=new float[256];
            for(int i=0;i<pcm.Length;i++)pcm[i]=Mathf.Sin(2*Mathf.PI*32*i/256);
            string path=Path.Combine(directory,"doppler-"+motion);
            using(var writer=new AcousticTake.Writer(path,8000,scene,Array.Empty<AcousticTake.Primitive>(),new[]{pcm}))
            {
                writer.Append(0,state);state[1].position.x=initial+motion*68.6f;writer.Append(16000,state);
                writer.Append(new AcousticTake.VoiceEvent {sample=0,gain=1,action=AcousticTake.VoiceAction.Play,loop=true});writer.Complete();
            }
            using var take=AcousticTake.Open(path);var settings=SpectralSettings();
            settings.fftSize=256;settings.responseStepSamples=32;settings.speedOfSound=343;settings.maximumDelaySeconds=1;
            TakeRender.RenderDirect(take,settings,path+"-render",shader:shader);
            var output=ReadWave(Path.Combine(path+"-render","listener-0.wav"));int crossings=0,expected=1000-motion*100;
            double wantedReal=0,wantedImag=0,originalReal=0,originalImag=0;
            for(int n=4001;n<12000;n++)
            {
                if(output[n-1]<0 && output[n]>=0)crossings++;
                double window=.5-.5*Math.Cos(2*Math.PI*(n-4000)/7999),value=output[n]*window;
                wantedReal+=value*Math.Cos(2*Math.PI*expected*n/8000);wantedImag+=value*Math.Sin(2*Math.PI*expected*n/8000);
                originalReal+=value*Math.Cos(2*Math.PI*1000*n/8000);originalImag+=value*Math.Sin(2*Math.PI*1000*n/8000);
            }
            Check(Math.Abs(crossings-expected)<=2,"Approaching/receding listener Doppler frequency");
            Check(wantedReal*wantedReal+wantedImag*wantedImag>100*(originalReal*originalReal+originalImag*originalImag),"Doppler tone dominates its unshifted bin");
        }
    }
    static void SpectralStreaming(ComputeShader shader,AcousticTake.Dimension dimension)
    {
        var scene=Scene(3);scene.dimension=dimension;
        scene.sources=new[]{new AcousticTake.SourceDefinition {objectIndex=0,clipIndex=0},new AcousticTake.SourceDefinition {objectIndex=1,clipIndex=0}};scene.listeners=new[]{2,3};
        var state=States(3);state[1].flags=AcousticTake.StateFlags.Active|AcousticTake.StateFlags.Source;
        var pcm=new float[256];for(int i=0;i<pcm.Length;i++)pcm[i]=.2f*Mathf.Sin(2*Mathf.PI*5*i/256);
        var events=new[]{new AcousticTake.VoiceEvent {sample=0,gain=.5f,action=AcousticTake.VoiceAction.Play,loop=true},
            new AcousticTake.VoiceEvent {sample=97,sourceIndex=1,gain=.3f,action=AcousticTake.VoiceAction.Play,loop=true},
            new AcousticTake.VoiceEvent {sample=611,sourceIndex=0,action=AcousticTake.VoiceAction.Stop}};
        long[] observations={0,61,137,321,577,699,811,1024};string path=Path.Combine(directory,"live-"+dimension);
        using(var writer=new AcousticTake.Writer(path,8000,scene,Array.Empty<AcousticTake.Primitive>(),new[]{pcm}))
        {
            foreach(long t in observations)
            {
                state[2].position=new Vector3(2+t*.001f+(t>=577?3:0),0,0);state[2].rotation=Quaternion.Euler(0,t*.05f,0);
                state[2].flags&=~AcousticTake.StateFlags.Cut;if(t==577)state[2].flags|=AcousticTake.StateFlags.Cut;writer.Append(t,state);
            }
            foreach(var e in events)writer.Append(e);writer.Complete();
        }
        using var take=AcousticTake.Open(path);var settings=SpectralSettings();settings.listenerOutput=TakeRender.ListenerOutput.Binaural;settings.transitionSamples=64;
        var report=TakeRender.RenderDirect(take,settings,path+"-render",shader:shader);
        var samples=new[]{new List<float>(),new List<float>()};Action<long,float[],int> collect=(at,block,count)=>{for(int n=0;n<count;n++)for(int l=0;l<2;l++)for(int ear=0;ear<2;ear++)samples[l].Add(block[n*4+l*2+ear]);};
        var left=new AcousticTake.State[4];var right=new AcousticTake.State[4];take.ReadStoredFrame(0,left);
        using(var live=new TakeRender.Live(scene,take.Geometry,new[]{pcm},8000,settings,null,0,0,shader))
        {
            live.Observe(0,0,left,left,collect);int eventIndex=0;
            for(int f=1;f<observations.Length;f++)
            {while(eventIndex<events.Length && events[eventIndex].sample<=observations[f])live.Append(events[eventIndex++]);take.ReadStoredFrame(f,right);live.Observe(observations[f-1],observations[f],left,right,collect);Array.Copy(right,left,4);}
            live.Finish(1024,left,collect);
        }
        for(int l=0;l<2;l++)
        {
            var expected=ReadWave(Path.Combine(path+"-render","listener-"+l+".wav"));Check(expected.Length==samples[l].Count,"Spectral timelines match");float error=0;
            for(int i=0;i<expected.Length;i++)error=Mathf.Max(error,Mathf.Abs(expected[i]-samples[l][i]));Near(error,0,"Spectral live/offline agreement",1e-5f);
        }
        settings.maximumResponsesPerBatch=1;TakeRender.RenderDirect(take,settings,path+"-single",shader:shader);
        for(int l=0;l<2;l++)Check(AcousticTake.HashFile(Path.Combine(path+"-single","listener-"+l+".wav"))==report.audioHashes[l],"GPU batch size preserves spectral PCM");
    }
}
