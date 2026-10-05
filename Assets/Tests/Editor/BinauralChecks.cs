using System;
using System.Collections.Generic;
using System.IO;
using UnityEngine;

public static partial class TakeChecks
{
    static float[] EarImpulse(Vector3 arrival,Quaternion rotation,int rate,TakeRender.DelayFilter filter=TakeRender.DelayFilter.Linear)
    {
        var state=States(); state[1].rotation=rotation;
        var signal=new TakeRender.Signal(new[]{1f}); signal.SetActive(0,true);
        signal.Append(new AcousticTake.VoiceEvent {action=AcousticTake.VoiceAction.Play,gain=1});
        var settings=new TakeRender.Settings {listenerOutput=TakeRender.ListenerOutput.Binaural,delayFilter=filter};
        var mixer=new TakeRender.Mixer(Scene(),rate,settings,new[]{signal});
        mixer.SetResponse(0,new[]{new TakeRender.DirectPath {direction=arrival,pressureGain=1}},state);
        var output=new float[1024]; mixer.Render(0,512,output); return output;
    }
    static void BinauralChecks()
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
        var queue=new AudioManager.PcmQueue(8,44100,null,2); var block=new float[8];
        queue.Publish(0,new[]{99f,98f,1f,2f,99f,98f,3f,4f,99f,98f,5f,6f,99f,98f,7f,8f},4,4,2); queue.Consume(block,2);
        Check(block[0]==1 && block[1]==2 && block[7]==8,"Monitoring selects both channels of the requested listener.");
        queue.Publish(4,new[]{9f,10f,11f,12f,13f,14f,15f,16f},4,2); queue.Consume(block,2);
        Check(block[0]==9 && block[7]==16,"Stereo queue wraps in sample frames.");
        queue.Complete(); long before=GC.GetAllocatedBytesForCurrentThread();
        for(int i=0;i<100;i++) queue.Consume(block,2);
        Check(GC.GetAllocatedBytesForCurrentThread()==before,"Stereo callback remains allocation-free.");
        var settings=new TakeRender.Settings {listenerOutput=TakeRender.ListenerOutput.Binaural};
        string output=Path.Combine(directory,"stereo-monitor");
        using(var capture=new AudioManager.PlaybackCapture(output,44100,8,4,0,0,settings))
        {
            var stream=new AudioManager.PcmQueue(8,44100,capture,2);
            stream.Publish(0,new[]{1f,2f,3f,4f,5f,6f,7f,8f},4,2); stream.Complete(); stream.Consume(block,2); capture.Drain();
            var pcm=ReadWave(Path.Combine(output,"output.wav"));
            using(var reader=new BinaryReader(File.OpenRead(Path.Combine(output,"output.wav"))))
            {reader.BaseStream.Position=22; Check(reader.ReadUInt16()==2,"Monitor WAV declares stereo.");}
            Check(capture.Metadata.complete && capture.Metadata.channelsPerListener==2 && pcm[1]==2 && pcm[7]==8,"Recorded output preserves distinct ear samples.");
            var metrics=TakeComparison.CompareWave(Path.Combine(output,"output.wav"),Path.Combine(output,"output.wav"));
            Check(metrics.channels==2 && metrics.samples==4 && metrics.exactMatch,"Stereo comparison counts sample frames, not interleaved values.");
        }
        var monoDevice=new AudioManager.PcmQueue(8,44100,null,2); monoDevice.Publish(0,new float[8],4,2); monoDevice.Consume(new float[4],1);
        Check(monoDevice.ChannelMismatchSamples==4,"Unsupported device channel layout is counted.");
    }

    static void BinauralStreaming(ComputeShader shader=null)
    {
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
            Check(report.hrtfHash==Hrtf.Load(rate).Hash && report.channelsPerListener==2 && report.formatVersion==2,"Export identifies its channel/direction format and HRTF.");
            Check(new FileInfo(Path.Combine(path,"offline","responses.bin")).Length==(long)report.responseCount*(8+4*20),"Direction-bearing response rows have the declared layout.");
            using(var reader=new BinaryReader(File.OpenRead(Path.Combine(path,"offline","responses.bin"))))
            {reader.ReadInt64(); reader.ReadSingle(); reader.ReadSingle(); Vector3 direction=new Vector3(reader.ReadSingle(),reader.ReadSingle(),reader.ReadSingle()); Near(Vector3.Distance(direction,(states[0].position-states[2].position).normalized),0,"Exported direction points toward the source",1e-6f);}
        }
    }
}
