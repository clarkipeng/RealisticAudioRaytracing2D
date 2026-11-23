using NUnit.Framework;
using System;
using System.Collections.Generic;
using System.IO;
using UnityEngine;

// The CPU spectral synthesis of exact arrivals (SpectralAudio, the path method's direct sound) against independent
// time-domain oracles, and the settings and fixtures the path checks share.
public partial class TakeChecks
{
    TakeRender.Settings SpectralSettings() => new TakeRender.Settings {transport=TakeRender.Transport.Raytraced,synthesis=TakeRender.Synthesis.Spectral,audioBackend=TakeRender.AudioBackend.CpuReference,
        responseStepSamples=128,fftSize=128,spectralHop=32,delayFilter=TakeRender.DelayFilter.Linear,maxBounces=3,maximumDelaySeconds=.1f,
        maximumResponsesPerBatch=4,speedOfSound=8000,minimumDistance=1,airAbsorption=false,diffractionFactor=0,tailSamples=256};

    AcousticTake.Primitive[] TransportBox(AcousticTake.Dimension dimension,int owner,int material,float half)
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
    // Independent time-domain oracle: no FFT/window formula in the expected output.
    [Test, Pairwise, Category("Core")]
    public void SpectralConvolutionChecks(
        [Values(64,128)] int fft,[Values(2,4)] int divisor,[Values(false,true)] bool ears,
        [Values(0f,1f,31f,32f,63f,129f,31.25f)] float delay,
        [Values(TakeRender.DelayFilter.Linear,TakeRender.DelayFilter.WindowedSinc)] TakeRender.DelayFilter filter)
    {
        var scene=Scene();var state=States();
        int hop=fft/divisor;
        var settings=SpectralSettings();settings.fftSize=fft;settings.spectralHop=hop;settings.responseStepSamples=128;
        settings.delayFilter=filter;settings.maxBounces=0;settings.listenerOutput=ears?TakeRender.ListenerOutput.Binaural:TakeRender.ListenerOutput.Mono;
        var pcm=new float[256];pcm[31]=1;pcm[127]=-.3f;
        var signal=new TakeRender.Signal(pcm);signal.SetActive(0,true);signal.Append(new AcousticTake.VoiceEvent{action=AcousticTake.VoiceAction.Play,gain=1});
        var mixer=new TakeRender.Mixer(scene,8192,settings,new[]{signal});
        var arrivals=new RayTransport.Path[settings.FrequencyBins];
        for(int b=0;b<settings.FrequencyBins;b++)
            arrivals[b]=new RayTransport.Path {frequencyBin=(uint)b,pressureGain=1,delaySeconds=delay/8192,direction=Vector3.forward};
        var output=new float[768*settings.ChannelsPerListener];var block=new float[128*settings.ChannelsPerListener];
        for(int at=0;at<768;at+=128)
        {
            mixer.SetResponse((long)at,new TakeRender.DirectPath[1],state,false,arrivals,0);
            // Changing callback chunk boundaries must not change convolution.
            for(int n=0;n<128;n+=16){mixer.Render(at+n,16,block);Array.Copy(block,0,output,(at+n)*settings.ChannelsPerListener,16*settings.ChannelsPerListener);}
        }
        var expected=new float[output.Length];var hrtf=ears?Hrtf.Load(8192):null;var blend=ears?hrtf.Interpolate(Vector3.forward):default;
        int whole=(int)delay;double fraction=delay-whole;
        int begin=filter==TakeRender.DelayFilter.Linear?0:1-settings.sincRadius;
        int end=filter==TakeRender.DelayFilter.Linear?1:settings.sincRadius;
        var taps=new double[end-begin+1];double normalization=0;
        for(int k=begin;k<=end;k++)
        {
            double x=k-fraction;
            double value=filter==TakeRender.DelayFilter.Linear?(k==0?1-fraction:fraction):
                Math.Abs(x)<1e-12?1:Math.Sin(Math.PI*x)/(Math.PI*x)*Math.Sin(Math.PI*x/settings.sincRadius)/(Math.PI*x/settings.sincRadius);
            taps[k-begin]=value;normalization+=value;
        }
        for(int n=0;n<pcm.Length;n++)for(int ear=0;ear<settings.ChannelsPerListener;ear++)for(int t=0;t<(hrtf?.Taps??1);t++)
        for(int k=begin;k<=end;k++)
        {
            int index=n+settings.DspLatencySamples+whole+t+k;double value=pcm[n]*(ears?hrtf.Coefficient(blend,ear,t):1);
            if(index>=0 && index<768)expected[index*settings.ChannelsPerListener+ear]+=(float)(value*taps[k-begin]/normalization);
        }
        float error=0;for(int i=0;i<output.Length;i++)error=Mathf.Max(error,Mathf.Abs(output[i]-expected[i]));
        Near(error,0,"Independent delayed FIR oracle: FFT "+fft+" hop "+hop+" ears "+ears+" delay "+delay+" filter "+filter,2e-5f);
    }

    [Test, Category("Core")]
    public void LongHrirConvolvesFullResponse()
    {
        var scene=Scene();var state=States();
        // A single-bin finite response with HRIR longer than the acoustic grid exercises the ungrouped path.
        var shortGrid=SpectralSettings();shortGrid.fftSize=64;shortGrid.spectralHop=16;shortGrid.maxBounces=0;
        shortGrid.listenerOutput=TakeRender.ListenerOutput.Binaural;
        var earSignal=new TakeRender.Signal(new[]{1f});earSignal.SetActive(0,true);
        earSignal.Append(new AcousticTake.VoiceEvent{action=AcousticTake.VoiceAction.Play,gain=1});
        var earMixer=new TakeRender.Mixer(scene,44100,shortGrid,new[]{earSignal});
        var earPaths=new RayTransport.Path[shortGrid.FrequencyBins];
        earPaths[7]=new RayTransport.Path {frequencyBin=7,pressureGain=.75f,delaySeconds=31.25f/44100,direction=Vector3.right};
        earMixer.SetResponse(0L,new TakeRender.DirectPath[1],state,false,earPaths,0);
        var earOutput=new float[1024];var earBlock=new float[256];
        for(int n=0;n<512;n+=128){earMixer.Render(n,128,earBlock);Array.Copy(earBlock,0,earOutput,n*2,256);}
        var earExpected=new double[1024];var longHrtf=Hrtf.Load(44100);var right=longHrtf.Interpolate(Vector3.right);
        double exactDelay=(double)(31.25f/44100)*44100;int baseDelay=(int)Math.Floor(exactDelay);double sub=exactDelay-baseDelay;
        for(int n=0;n<64;n++)for(int t=0;t<longHrtf.Taps;t++)for(int ear=0;ear<2;ear++)
        {
            double value=.75*2/64*Math.Cos(2*Math.PI*7*(n-32)/64)*longHrtf.Coefficient(right,ear,t);
            int at=baseDelay+32+n+t;earExpected[at*2+ear]+=value*(1-sub);earExpected[(at+1)*2+ear]+=value*sub;
        }
        double earError=0;for(int n=0;n<earOutput.Length;n++)earError=Math.Max(earError,Math.Abs(earOutput[n]-earExpected[n]));
        Near((float)earError,0,"Full long HRIR times finite single-bin response",2e-5f);
    }

    void KnifeEdgeFixture(bool planar,out AcousticTake.Scene scene,out AcousticTake.Primitive[] geometry,out AcousticTake.State[] state)
    {
        scene=Scene(2);scene.listeners=new[]{1};scene.dimension=planar?AcousticTake.Dimension.Planar2D:AcousticTake.Dimension.Spatial3D;
        scene.materials=new[]{new AcousticTake.MaterialDefinition {frequencies=new[]{0f},absorption=new[]{0f},transmission=new[]{0f},scattering=new[]{0f}}};
        var a=new Vector3(0,-100,planar?0:-100);var b=new Vector3(0,0,planar?0:-100);var c=new Vector3(0,0,100);var d=new Vector3(0,-100,100);
        geometry=planar?new[]{new AcousticTake.Primitive {objectIndex=2,a=a,b=b}}:
            new[]{new AcousticTake.Primitive {objectIndex=2,a=a,b=b,c=c},new AcousticTake.Primitive {objectIndex=2,a=a,b=c,c=d}};
        scene.objects[2].primitiveCount=geometry.Length;
        state=States(2);state[0].position=new Vector3(-2,1,0);state[1].position=new Vector3(2,-2,0);
        state[2].position=Vector3.zero;state[2].flags=AcousticTake.StateFlags.Active|AcousticTake.StateFlags.Geometry;
    }

}
