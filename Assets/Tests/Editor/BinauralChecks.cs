using NUnit.Framework;
using System;
using System.Collections.Generic;
using System.IO;
using UnityEngine;

public partial class TakeChecks
{
    float[] EarImpulse(Vector3 arrival,Quaternion rotation,int rate,TakeRender.DelayFilter filter=TakeRender.DelayFilter.Linear)
    {
        var state=States(); state[1].rotation=rotation;
        var signal=new TakeRender.Signal(new[]{1f}); signal.SetActive(0,true);
        signal.Append(new AcousticTake.VoiceEvent {action=AcousticTake.VoiceAction.Play,gain=1});
        var settings=new TakeRender.Settings {listenerOutput=TakeRender.ListenerOutput.Binaural,delayFilter=filter};
        var mixer=new TakeRender.Mixer(Scene(),rate,settings,new[]{signal});
        mixer.SetResponse(0,new[]{new TakeRender.DirectPath {direction=arrival,pressureGain=1}},state);
        var output=new float[1024]; mixer.Render(0,512,output); return output;
    }
    [Test, Category("Core")]
    public void BinauralChecks()
    {
        var hrtf=Hrtf.Load(44100); var right=EarImpulse(Vector3.right,Quaternion.identity,44100);
        var left=EarImpulse(Vector3.left,Quaternion.identity,44100);
        var turned=EarImpulse(Vector3.right,Quaternion.Euler(0,180,0),44100);
        var front=EarImpulse(Vector3.forward,Quaternion.identity,44100);
        int direction=hrtf.Nearest(Vector3.right),leftPeak=0,rightPeak=0;
        double leftEnergy=0,rightEnergy=0,mirrorError=0,turnError=0,filterError=0,frontError=0;
        for(int n=0;n<512;n++)
        {
            filterError=Math.Max(filterError,Math.Abs(right[n*2]-(n<hrtf.Taps ? hrtf.Coefficient(direction,0,n):0)));
            filterError=Math.Max(filterError,Math.Abs(right[n*2+1]-(n<hrtf.Taps ? hrtf.Coefficient(direction,1,n):0)));
            mirrorError=Math.Max(mirrorError,Math.Abs(right[n*2]-left[n*2+1]));
            turnError=Math.Max(turnError,Math.Abs(right[n*2]-turned[n*2+1]));
            frontError=Math.Max(frontError,Math.Abs(front[n*2]-front[n*2+1]));
            if(Math.Abs(right[n*2])>Math.Abs(right[leftPeak*2])) leftPeak=n;
            if(Math.Abs(right[n*2+1])>Math.Abs(right[rightPeak*2+1])) rightPeak=n;
            leftEnergy+=right[n*2]*right[n*2]; rightEnergy+=right[n*2+1]*right[n*2+1];
        }
        Check(filterError<1e-7,"Binaural impulse output matches the measured HRIR, including its delay.");
        Check(frontError<1e-7 && mirrorError<1e-7,"Front symmetry and left/right ear convention.");
        Check(turnError<1e-7,"Rotating the head 180 degrees exchanges ears for a fixed lateral source.");
        Check(rightPeak<leftPeak && rightEnergy>leftEnergy,"Right-side source reaches the right ear earlier and with greater energy.");
        var resampled=EarImpulse(Vector3.right,Quaternion.identity,48000);
        int peak48=0; for(int n=0;n<512;n++) if(Math.Abs(resampled[n*2+1])>Math.Abs(resampled[peak48*2+1])) peak48=n;
        Check(Math.Abs((double)peak48/48000-(double)rightPeak/44100)<2.0/44100,"Resampling retains physical ear timing.");
        var sinc=EarImpulse(Vector3.right,Quaternion.identity,44100,TakeRender.DelayFilter.WindowedSinc);
        float error=0; for(int n=0;n<hrtf.Taps;n++) error=Mathf.Max(error,Mathf.Abs(sinc[(n+16)*2]-right[n*2]));
        Near(error,0,"HRIR composes with declared fractional-delay FIR latency",1e-6f);
    }

    // Diffuse-field equalised HRTF: the solid-angle-weighted mean power over all directions
    // and both ears is flat (third octaves 300 Hz - 12 kHz within 2 dB of the 200-400 Hz
    // level), so the ears do not colour a diffuse field; interaural differences are kept.
    [Test]
    public void HrtfIsDiffuseFieldEqualised()
    {
        var hrtf=Hrtf.Load(44100);int n=1024,bins=n/2+1;var spectra=Hrtf.Spectra(hrtf,n);
        var rings=new Dictionary<int,int>();
        for(int i=0;i<hrtf.DirectionCount;i++){int key=Mathf.RoundToInt(Mathf.Asin(Mathf.Clamp(hrtf.Direction(i).y,-1,1))*Mathf.Rad2Deg);rings[key]=rings.TryGetValue(key,out int c)?c+1:1;}
        var power=new double[bins];double total=0;
        for(int i=0;i<hrtf.DirectionCount;i++)
        {
            float elevation=Mathf.Asin(Mathf.Clamp(hrtf.Direction(i).y,-1,1));double w=Math.Max(.05,Math.Cos(elevation))/rings[Mathf.RoundToInt(elevation*Mathf.Rad2Deg)];
            for(int ear=0;ear<2;ear++)for(int k=0;k<bins;k++){var h=spectra[(i*2+ear)*bins+k];power[k]+=w*(h.x*h.x+h.y*h.y);}
            total+=2*w;
        }
        double Band(double f){double s=0;int m=0;for(int k=1;k<bins;k++){double hz=k*44100.0/n;if(hz>=f/Math.Pow(2,1/6.0)&&hz<f*Math.Pow(2,1/6.0)){s+=power[k]/total;m++;}}return s/m;}
        double reference=0;int r=0;for(int k=1;k<bins;k++){double hz=k*44100.0/n;if(hz>=200&&hz<=400){reference+=power[k]/total;r++;}}reference/=r;
        double worst=0;
        foreach(double f in new[]{315.0,500,800,1250,2000,2500,3150,4000,5000,6300,8000,10000,12500})worst=Math.Max(worst,Math.Abs(10*Math.Log10(Band(f)/reference)));
        TestContext.WriteLine($"largest third-octave deviation of the diffuse-field average: {worst:F2} dB");
        Assert.That(worst,Is.LessThan(2),"The diffuse-field average is flat after equalisation.");
        // Interaural level difference for a source on the right stays large (localisation kept).
        var right=hrtf.Nearest(Vector3.right);double left=0,rightEar=0;
        for(int k=1;k<bins;k++){double hz=k*44100.0/n;if(hz<2000||hz>6000)continue;var l=spectra[(right*2)*bins+k];var q=spectra[(right*2+1)*bins+k];left+=l.x*l.x+l.y*l.y;rightEar+=q.x*q.x+q.y*q.y;}
        double ild=10*Math.Log10(rightEar/left);TestContext.WriteLine($"interaural level difference at 90 degrees, 2-6 kHz: {ild:F1} dB");
        Assert.That(ild,Is.GreaterThan(6),"A source on the right is much louder in the right ear at 2-6 kHz.");
    }

    [Test, Category("Core")]
    public void StereoQueuePreservesListenerChannels()
    {
        var queue=new AudioManager.PcmQueue(8,44100,null,2); var block=new float[8];
        queue.Publish(0,new[]{99f,98f,1f,2f,99f,98f,3f,4f,99f,98f,5f,6f,99f,98f,7f,8f},4,4,2); queue.Consume(block,2);
        Check(block[0]==1 && block[1]==2 && block[7]==8,"Monitoring selects both channels of the requested listener.");
        queue.Publish(4,new[]{9f,10f,11f,12f,13f,14f,15f,16f},4,2); queue.Consume(block,2);
        Check(block[0]==9 && block[7]==16,"Stereo queue wraps in sample frames.");
        queue.Complete(); long before=GC.GetAllocatedBytesForCurrentThread();
        for(int i=0;i<100;i++) queue.Consume(block,2);
        Check(GC.GetAllocatedBytesForCurrentThread()==before,"Stereo callback remains allocation-free.");
    }

    [Test, Category("Core")]
    public void StereoQueueRejectsMonoDevice()
    {
        var monoDevice=new AudioManager.PcmQueue(8,44100,null,2); monoDevice.Publish(0,new float[8],4,2); monoDevice.Consume(new float[4],1);
        Check(monoDevice.ChannelMismatchSamples==4,"Unsupported device channel layout is counted.");
    }

    [Test, Category("Core")]
    public void HrtfInterpolation()
    {
        var hrtf=Hrtf.Load(44100);float measuredError=0;
        for(int i=0;i<hrtf.DirectionCount;i++)
        {
            var blend=hrtf.Interpolate(hrtf.Direction(i));
            Near(blend.weights.x+blend.weights.y+blend.weights.z+blend.weights.w,1,"HRTF weights preserve gain",1e-6f);
            for(int ear=0;ear<2;ear++)for(int k=0;k<hrtf.Taps;k++)
                measuredError=Mathf.Max(measuredError,Mathf.Abs(hrtf.Coefficient(blend,ear,k)-hrtf.Coefficient(i,ear,k)));
        }
        Near(measuredError,0,"Interpolation reproduces every measured HRIR",2e-5f);
        // Halfway between front and five degrees is the half-weighted measured pair.
        var halfway=EarImpulse(Quaternion.Euler(0,2.5f,0)*Vector3.forward,Quaternion.identity,44100);
        int front=hrtf.Nearest(Vector3.forward),next=hrtf.Nearest(Quaternion.Euler(0,5,0)*Vector3.forward);float error=0;
        for(int ear=0;ear<2;ear++)for(int k=0;k<hrtf.Taps;k++)
            error=Mathf.Max(error,Mathf.Abs(halfway[k*2+ear]-.5f*(hrtf.Coefficient(front,ear,k)+hrtf.Coefficient(next,ear,k))));
        Near(error,0,"Direct binaural output interpolates between measured directions",1e-6f);
        foreach(float yaw in new[]{2.5f,5f,177.5f,180f,357.5f})foreach(float pitch in new[]{0f,-10f,-40f,-80f,-90f,40f})
        {
            var a=hrtf.Interpolate(Quaternion.Euler(pitch-.0001f,yaw-.0001f,0)*Vector3.forward);
            var b=hrtf.Interpolate(Quaternion.Euler(pitch+.0001f,yaw+.0001f,0)*Vector3.forward);float jump=0;
            for(int ear=0;ear<2;ear++)for(int k=0;k<hrtf.Taps;k++)jump=Mathf.Max(jump,Mathf.Abs(hrtf.Coefficient(a,ear,k)-hrtf.Coefficient(b,ear,k)));
            Check(jump<.0002f,"No filter jump across azimuth/elevation cell edges or rear seam: yaw="+yaw+", pitch="+pitch+", jump="+jump);
        }
    }

    [TestCase(false, Category="Core")]
    [TestCase(true, Category="GPU,Slow")]
    public void BinauralStreaming(bool gpu)
    {
        var shader=gpu?RequireGpu():null;
        const int rate=48000,duration=4800;
        var scene=Scene(3); scene.sources=new[]{new AcousticTake.SourceDefinition {objectIndex=0,clipIndex=0},new AcousticTake.SourceDefinition {objectIndex=1,clipIndex=0}}; scene.listeners=new[]{2,3};
        var states=States(3); states[1].flags=AcousticTake.StateFlags.Active|AcousticTake.StateFlags.Source;
        states[0].position=new Vector3(3,1,4); states[1].position=new Vector3(-2,1,3);
        states[2].position=Vector3.zero; states[3].position=new Vector3(-1,0,0);
        var pcm=new float[256]; for(int i=0;i<pcm.Length;i++) pcm[i]=Mathf.Sin(i*.2f)*.2f;
        string name=shader==null ? "binaural-cpu":"binaural-gpu",path=Path.Combine(directory,name);
        var play=new AcousticTake.VoiceEvent {action=AcousticTake.VoiceAction.Play,gain=1,loop=true};
        using(var writer=new AcousticTake.Writer(path,rate,scene,new AcousticTake.Primitive[0],new[]{pcm}))
        {
            for(int n=0;n<=duration;n+=1200) {states[2].rotation=Quaternion.Euler(0,n*.017f,0); writer.Append(n,states);}
            writer.Append(play); play.sourceIndex=1; writer.Append(play); writer.Complete();
        }
        using(var take=AcousticTake.Open(path))
        {
            var settings=new TakeRender.Settings {listenerOutput=TakeRender.ListenerOutput.Binaural,responseStepSamples=128,transitionSamples=32,tailSamples=512,maximumDelaySeconds=.1f};
            var report=TakeRender.RenderDirect(take,settings,Path.Combine(path,"offline"),null,shader);
            var rendered=new[]{new List<float>(),new List<float>()};
            Action<long,float[],int> collect=(sample,block,count)=> {for(int n=0;n<count;n++) for(int l=0;l<2;l++) for(int ear=0;ear<2;ear++) rendered[l].Add(block[(n*2+l)*2+ear]);};
            var a=new AcousticTake.State[4]; var b=new AcousticTake.State[4]; take.ReadStoredFrame(0,a);
            using(var live=new TakeRender.Live(scene,take.Geometry,new[]{pcm},rate,settings,null,0,0,shader))
            {
                play.sourceIndex=0; live.Append(play); play.sourceIndex=1; live.Append(play); live.Observe(0,0,a,a,collect);
                for(int f=1;f<take.Metadata.frameCount;f++)
                {take.ReadStoredFrame(f,b); live.Observe(take.FrameSample(f-1),take.FrameSample(f),a,b,collect); Array.Copy(b,a,4);}
                live.Finish(duration,a,collect);
            }
            for(int l=0;l<2;l++)
            {
                var expected=ReadWave(Path.Combine(path,"offline","listener-"+l+".wav")); Check(expected.Length==rendered[l].Count,"Binaural live/offline frame counts align.");
                float error=0,difference=0;
                for(int i=0;i<expected.Length;i++) error=Mathf.Max(error,Mathf.Abs(expected[i]-rendered[l][i]));
                for(int i=0;i<expected.Length;i+=2) difference=Mathf.Max(difference,Mathf.Abs(expected[i]-expected[i+1]));
                Near(error,0,"Binaural live/offline samples agree",1e-6f); Check(difference>.001f,"Export contains distinct left/right signals.");
            }
            Check(report.hrtfHash==Hrtf.Load(rate).Hash && report.channelsPerListener==2 && report.formatVersion==3,"Export identifies its channel/direction format and HRTF.");
            Check(new FileInfo(Path.Combine(path,"offline","responses.bin")).Length==(long)report.responseCount*(8+4*24),"Direction-bearing response rows have the declared layout.");
            using(var reader=new BinaryReader(File.OpenRead(Path.Combine(path,"offline","responses.bin"))))
            {reader.ReadInt64(); reader.ReadSingle(); reader.ReadSingle(); Vector3 direction=new Vector3(reader.ReadSingle(),reader.ReadSingle(),reader.ReadSingle()); Near(Vector3.Distance(direction,(states[0].position-states[2].position).normalized),0,"Exported direction points toward the source",1e-6f);}
        }
    }
}
