using System;
using System.IO;
using System.Reflection;
using UnityEditor;
using UnityEngine;

public static partial class TakeChecks
{
    sealed class GpuDsp : IDisposable
    {
        readonly Type type=typeof(TakeRender).Assembly.GetType("GpuAudio");
        readonly object instance;
        public GpuDsp(AcousticTake.Scene scene,int rate,TakeRender.Settings settings,TakeRender.Signal[] signals)
        {instance=Activator.CreateInstance(type,scene,rate,settings,signals,1);}
        public void Response(long sample,bool cut,ComputeBuffer buffer,AcousticTake.Scene scene,AcousticTake.State[] state)
        {type.GetMethod("SetResponse").Invoke(instance,new object[]{sample,cut,buffer,0,scene,state});}
        public void Render(long sample,int count,float[] pcm) {type.GetMethod("Render").Invoke(instance,new object[]{sample,count,pcm});}
        public void Dispose()=>((IDisposable)instance).Dispose();
    }
    [MenuItem("Tools/Acoustics/Check GPU audio against CPU reference")]
    public static void RunGpuAudio()
    {
        checks=0;var shader=AssetDatabase.LoadAssetAtPath<ComputeShader>("Assets/Shaders/Direct.compute");
        foreach(var method in new[]{TakeRender.Synthesis.ImpulseResponse,TakeRender.Synthesis.Spectral})
        foreach(var ears in new[]{TakeRender.ListenerOutput.Mono,TakeRender.ListenerOutput.Binaural})
        foreach(var delay in new[]{TakeRender.DelayFilter.Linear,TakeRender.DelayFilter.WindowedSinc})
            GpuAudioParity(shader,method,ears,delay,128);
        // Exercise the global-memory FFT fallback, rather than imposing an FFT-size cap.
        GpuAudioParity(shader,TakeRender.Synthesis.Spectral,TakeRender.ListenerOutput.Mono,TakeRender.DelayFilter.Linear,2048);
        // More path slots exercise the global grouping table at a small FFT.
        GpuAudioParity(shader,TakeRender.Synthesis.Spectral,TakeRender.ListenerOutput.Mono,TakeRender.DelayFilter.Linear,128,32);
        directory=Path.Combine(Path.GetTempPath(),"acoustic-gpu-audio-"+Guid.NewGuid().ToString("N"));Directory.CreateDirectory(directory);
        try
        {
            foreach(var dimension in new[]{AcousticTake.Dimension.Planar2D,AcousticTake.Dimension.Spatial3D})
            foreach(var method in new[]{TakeRender.Synthesis.ImpulseResponse,TakeRender.Synthesis.Spectral})
                SpectralStreaming(shader,dimension,TakeRender.AudioBackend.Gpu,method);
        }
        finally {Directory.Delete(directory,true);}
        Debug.Log("GPU audio reference checks passed: "+checks);
    }
    static void GpuAudioParity(ComputeShader shader,TakeRender.Synthesis method,TakeRender.ListenerOutput ears,TakeRender.DelayFilter delay,int size,int rays=1)
    {
        const int rate=8192;var scene=Scene(2);var state=States(2);
        scene.sources=new[]{scene.sources[0],new AcousticTake.SourceDefinition{objectIndex=2,clipIndex=0}};
        state[2].flags|=AcousticTake.StateFlags.Source;state[0].position=new Vector3(-2,0,0);
        var settings=new TakeRender.Settings{transport=TakeRender.Transport.Raytraced,synthesis=method,listenerOutput=ears,delayFilter=delay,
            fftSize=size,spectralHop=size/4,responseStepSamples=size,transitionSamples=size/4,maximumDelaySeconds=.2f,airAbsorption=true,
            raysPerBin=rays,maxBounces=1,speedOfSound=343,minimumDistance=1};
        var clip=new float[257];for(int n=0;n<clip.Length;n++)clip[n]=.2f*Mathf.Sin(n*.37f)+.1f*Mathf.Cos(n*.93f);
        var signals=new[]{new TakeRender.Signal(clip),new TakeRender.Signal(clip)};
        for(int s=0;s<2;s++){signals[s].SetActive(0,true);signals[s].Append(new AcousticTake.VoiceEvent{action=AcousticTake.VoiceAction.Play,gain=1-s*.3f,clipOffset=s*31,loop=true});}
        using var gpu=new GpuDsp(scene,rate,settings,signals);var cpu=new TakeRender.Mixer(scene,rate,settings,signals);
        using var solver=new DirectSolver(scene,Array.Empty<AcousticTake.Primitive>(),settings,shader,sampleRate:rate);
        var set=typeof(TakeRender.Mixer).GetMethod("SetResponse",BindingFlags.Instance|BindingFlags.NonPublic);
        var buffer=typeof(DirectSolver).GetProperty("RayBuffer",BindingFlags.Instance|BindingFlags.NonPublic);
        var direct=new TakeRender.DirectPath[4];var a=new float[size*2*settings.ChannelsPerListener];var b=new float[a.Length];
        double peak=0,error=0,energy=0;int length=0;
        for(int frame=0;frame<12;frame++)
        {
            long sample=frame*(long)size;bool cut=frame==7;
            if(frame<5 || frame>=7){state[1].position.x=1+frame*.07f;state[1].rotation=Quaternion.Euler(0,frame*19,0);}
            state[0].velocity=Vector3.right*12;state[0].flags|=AcousticTake.StateFlags.VelocityValid;
            if(frame==8)state[1].flags&=~AcousticTake.StateFlags.Active;
            if(frame==9)state[1].flags|=AcousticTake.StateFlags.Active;
            if(frame==9)signals[0].Append(new AcousticTake.VoiceEvent{sample=sample+17,action=AcousticTake.VoiceAction.Stop});
            solver.Solve(state,direct);var paths=Arrivals(solver);
            set.Invoke(cpu,new object[]{sample,direct,state,cut,paths,0});gpu.Response(sample,cut,(ComputeBuffer)buffer.GetValue(solver),scene,state);
            // Non-hop-aligned delivery exercises the saved output block and source ring.
            int at=0;foreach(int count in new[]{13,size-13})
            {
                cpu.Render(sample+at,count,a);gpu.Render(sample+at,count,b);
                for(int n=0;n<count*2*settings.ChannelsPerListener;n++)
                {double d=a[n]-b[n];peak=Math.Max(peak,Math.Abs(d));error+=d*d;energy+=a[n]*a[n];length++;}at+=count;
            }
        }
        Check(energy>1e-6,"GPU parity fixture contains audible output.");
        double relative=Math.Sqrt(error/energy);
        Check(peak<2e-4 && relative<5e-4,"GPU/CPU "+method+" "+ears+" "+delay+" FFT "+size+" peak="+peak+" relative="+relative);
    }
}
