using System;
using System.Collections;
using Concentus.Common;
using UnityEngine;

// Device capture is opt-in. Replay never opens the device; it reads saved canonical PCM.
[DefaultExecutionOrder(-40)]
public sealed class MicrophoneInput : MonoBehaviour
{
    public RaytracedAudioSource source;
    public string device;
    public bool monitorSelf;
    public bool Active => clip!=null;
    public string Status {get;private set;}="Microphone off";
    public event Action<float[],int> VoiceSamples;
    public event Action<bool,double> VoiceState;
    public Func<double> Clock;
    public int CaptureRate {get;private set;}
    public double EpochTime {get;private set;}
    AudioClip clip;
    string openedDevice;
    SourceStream bound;
    PcmRateConverter deviceRate,outputRate;
    float[] raw,mono;
    int read,cursor;
    long available;
    double lastPoll;
    bool requesting;
    public void Toggle() {if(Active)StopCapture();else if(!requesting)StartCoroutine(StartCapture());}
    IEnumerator StartCapture()
    {
        if(source==null || !source.streamed || source.rayTraceManager.IsSourceReplay)yield break;
        requesting=true;Status="Waiting for microphone permission";
        yield return Application.RequestUserAuthorization(UserAuthorization.Microphone);
        requesting=false;
        if(!isActiveAndEnabled || source.rayTraceManager.IsSourceReplay)yield break;
        if(!Application.HasUserAuthorization(UserAuthorization.Microphone)) {Status="Microphone permission denied";yield break;}
        if(!string.IsNullOrEmpty(device) && Array.IndexOf(Microphone.devices,device)<0) {Status="Microphone device is unavailable";yield break;}
        if(Microphone.devices.Length==0) {Status="No microphone found";yield break;}
        try
        {
            bound=source.Stream;
            if(bound==null || bound.Finished)throw new InvalidOperationException("Start an acoustic session before enabling the microphone.");
            var manager=source.rayTraceManager;long now=manager.LiveSample;
            bound.SilenceUntil(now,now);
            EpochTime=Clock?.Invoke()??AudioSettings.dspTime;
            openedDevice=string.IsNullOrEmpty(device)?null:device;
            clip=Microphone.Start(openedDevice,true,2,48000);
            if(clip==null)throw new InvalidOperationException("Microphone did not start.");
            CaptureRate=clip.frequency;read=cursor=0;available=0;lastPoll=Time.realtimeSinceStartupAsDouble;
            raw=new float[Math.Max(1,CaptureRate/50)*clip.channels];mono=new float[raw.Length/clip.channels];
            deviceRate=new PcmRateConverter(CaptureRate,48000,SendVoice);
            outputRate=new PcmRateConverter(48000,manager.sampleRate,WriteSource);
            manager.RecordStreamEpoch(source,new AcousticTake.StreamEpoch {sample=bound.End,rate=CaptureRate,channels=clip.channels,
                participant=manager.SessionInfo?.localSlot.ToString()??"local",epoch=Guid.NewGuid().ToString("N"),remoteTime=EpochTime,uncertaintySeconds=-1,
                processing="Unity Microphone; device/OS processing unknown; Speex quality 8; canonical sample clock, ADC timing unmeasured"});
            VoiceState?.Invoke(true,EpochTime);Status="Microphone on";
        }
        catch(Exception e) {Status=e.Message;StopDevice();Debug.LogWarning(Status,this);}
    }
    void SendVoice(float[] data,int count) {VoiceSamples?.Invoke(data,count);outputRate.Process(data,count);}
    void WriteSource(float[] data,int count)
    {
        if(bound==null || bound.Finished)return;
        bound.Append(data,0,count,source.rayTraceManager.LiveSample);
    }
    void Update()
    {
        if(source==null || source.rayTraceManager==null)return;
        var manager=source.rayTraceManager;
        if(!manager.IsLiveSimulation || manager.IsSourceReplay)
        {if(Active)StopDevice();return;}
        if(bound!=source.Stream) {StopDevice();bound=source.Stream;Status="Microphone off";}
        if(bound==null || bound.Finished)return;
        try
        {
            long now=manager.LiveSample;
            if(!Active) {bound.SilenceUntil(now,now);return;}
            double wall=Time.realtimeSinceStartupAsDouble;
            if(wall-lastPoll>=clip.length || !Microphone.IsRecording(openedDevice))
            {
                StopDevice();bound.SilenceUntil(now,now,SourceStream.Status.Missing|SourceStream.Status.Discontinuity);
                Status="Microphone interrupted; enable it to start a new capture epoch";return;
            }
            lastPoll=wall;Drain(false);
        }
        catch(Exception e) {StopDevice();Status=e.Message;Debug.LogWarning(Status,this);}
    }
    void Drain(bool final)
    {
        int next=Microphone.GetPosition(openedDevice);
        if(next<0)throw new InvalidOperationException("Microphone cursor unavailable.");
        available+=(next-cursor+clip.samples)%clip.samples;cursor=next;
        while(available>=mono.Length || (final && available>0))
        {
            int frames=(int)Math.Min(available,mono.Length);
            var data=frames==mono.Length?raw:new float[frames*clip.channels];
            if(!clip.GetData(data,read))throw new InvalidOperationException("Cannot read microphone samples.");
            for(int n=0;n<frames;n++) {float sum=0;for(int c=0;c<clip.channels;c++)sum+=data[n*clip.channels+c];mono[n]=sum/clip.channels;}
            deviceRate.Process(mono,frames);read=(read+frames)%clip.samples;available-=frames;
        }
    }
    public void StopCapture()
    {
        try {if(Active && bound!=null && !bound.Finished) {Drain(true);deviceRate.Finish();outputRate.Finish();}}
        finally {StopDevice();Status="Microphone off";}
    }
    void StopDevice()
    {
        var captured=clip;clip=null;
        try {if(captured!=null)VoiceState?.Invoke(false,Clock?.Invoke()??AudioSettings.dspTime);}
        finally
        {
            if(captured!=null) {Microphone.End(openedDevice);Destroy(captured);}
            deviceRate?.Dispose();outputRate?.Dispose();deviceRate=outputRate=null;
        }
    }
    void OnDisable()=>StopDevice();
}

// Stateful anti-aliased conversion shared by devices and received voice.
internal sealed class PcmRateConverter : IDisposable
{
    readonly SpeexResampler converter;
    readonly float[] output=new float[4096];
    readonly Action<float[],int> consume;
    readonly int from,to;
    long inputSamples,outputSamples;
    bool finished;
    public PcmRateConverter(int from,int to,Action<float[],int> consume)
    {
        if(from<=0 || to<=0)throw new ArgumentOutOfRangeException();
        this.consume=consume;this.from=from;this.to=to;
        if(from!=to) {converter=new SpeexResampler(1,from,to,8);converter.SkipZeroes();}
    }
    public void Process(float[] input,int count)
    {
        if(finished || count<0 || count>input.Length)throw new ArgumentException("Invalid resampler input.");
        inputSamples+=count;
        if(converter==null) {consume(input,count);outputSamples+=count;return;}
        Convert(input,count,long.MaxValue);
    }
    void Convert(float[] input,int count,long limit)
    {
        int offset=0;
        while(offset<count && outputSamples<limit)
        {
            int read=count-offset,written=output.Length;
            converter.Process(0,input.AsSpan(offset,read),ref read,output.AsSpan(),ref written);
            if(read==0 && written==0)throw new InvalidOperationException("Resampler made no progress.");
            offset+=read;written=(int)Math.Min(written,limit-outputSamples);
            if(written>0) {consume(output,written);outputSamples+=written;}
        }
    }
    public void Finish()
    {
        if(finished)return;finished=true;
        if(converter==null)return;
        long end=checked((inputSamples*to+from-1)/from);
        var silence=new float[Math.Max(1,converter.InputLatency+1)];
        while(outputSamples<end)Convert(silence,silence.Length,end);
    }
    public void Dispose()=>converter?.Dispose();
}
