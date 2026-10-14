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
    public double CaptureTime(long sample48)=>captureClock?.At((double)sample48*CaptureRate/48000)??EpochTime+(double)sample48/48000;
    AudioClip clip;
    string openedDevice;
    SourceStream bound;
    PcmRateConverter deviceRate,outputRate;
    float[] raw,mono;
    int read,cursor;
    long available,capturedSamples;
    double lastPoll;
    bool requesting,started;
    SampleClock captureClock;
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
            openedDevice=string.IsNullOrEmpty(device)?null:device;
            clip=Microphone.Start(openedDevice,true,2,48000);
            if(clip==null)throw new InvalidOperationException("Microphone did not start.");
            CaptureRate=clip.frequency;read=cursor=0;available=capturedSamples=0;started=false;captureClock=new SampleClock(CaptureRate);lastPoll=Time.realtimeSinceStartupAsDouble;
            raw=new float[Math.Max(1,CaptureRate/50)*clip.channels];mono=new float[raw.Length/clip.channels];
            deviceRate=new PcmRateConverter(CaptureRate,48000,SendVoice);
            outputRate=new PcmRateConverter(48000,manager.sampleRate,WriteSource);
            Status="Microphone starting";
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
        int added=(next-cursor+clip.samples)%clip.samples;available+=added;capturedSamples+=added;cursor=next;
        if(capturedSamples==0)return;
        captureClock.Observe(capturedSamples,Clock?.Invoke()??AudioSettings.dspTime);
        if(!started)
        {
            started=true;EpochTime=captureClock.At(0);
            var manager=source.rayTraceManager;long now=manager.LiveSample;
            bound.SilenceUntil(Math.Max(bound.End,now-(long)Math.Round((double)capturedSamples*manager.sampleRate/CaptureRate)),now);
            manager.RecordStreamEpoch(source,new AcousticTake.StreamEpoch {sample=bound.End,rate=CaptureRate,channels=clip.channels,
                participant=manager.SessionInfo?.localSlot.ToString()??"local",epoch=Guid.NewGuid().ToString("N"),remoteTime=EpochTime,uncertaintySeconds=-1,
                processing="Unity cursor/sample clock; ADC latency unknown; Speex quality 8"});
            VoiceState?.Invoke(true,EpochTime);Status="Microphone on";
        }
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
        try {if(started)VoiceState?.Invoke(false,Clock?.Invoke()??AudioSettings.dspTime);}
        finally
        {
            started=false;
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
    readonly bool variableRate;
    double inputRate,targetOutput;
    bool finished;
    public double PendingSamples=>targetOutput-outputSamples;
    public PcmRateConverter(int from,int to,Action<float[],int> consume,bool variableRate=false)
    {
        if(from<=0 || to<=0)throw new ArgumentOutOfRangeException();
        this.consume=consume;this.from=from;this.to=to;this.variableRate=variableRate;inputRate=from;
        if(from!=to || variableRate)
        {
            var ratio=Ratio(from,to);
            converter=variableRate?new SpeexResampler(1,ratio.n,ratio.d,from,to,8):new SpeexResampler(1,from,to,8);
            if(variableRate)inputRate=(double)ratio.n*to/ratio.d;
            converter.SkipZeroes();
        }
    }
    public void SetInputRate(double rate)
    {
        if(converter==null || !double.IsFinite(rate) || rate<=0)throw new ArgumentException("Invalid variable resampler rate.");
        var ratio=Ratio(rate,to);double actual=(double)ratio.n*to/ratio.d;
        if(actual==inputRate)return;inputRate=actual;
        converter.SetRateFraction(ratio.n,ratio.d,from,to);
    }
    static (int n,int d) Ratio(double rate,int outputRate)
    {
        int n=checked((int)Math.Round(rate*1000)),d=checked(outputRate*1000),a=n,b=d;
        while(b!=0) {int remainder=a%b;a=b;b=remainder;}n/=a;d/=a;
        // Concentus multiplies old and new phase denominators as signed int.
        // Keep their product in range; common exact ratios remain exact.
        if(d>32768) {d=32768;n=checked((int)Math.Round(rate/outputRate*d));}
        return (n,d);
    }
    public void Process(float[] input,int count)
    {
        if(finished || count<0 || count>input.Length)throw new ArgumentException("Invalid resampler input.");
        inputSamples+=count;
        targetOutput=variableRate?targetOutput+(double)count*to/inputRate:(double)inputSamples*to/from;
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
        long end=checked((long)Math.Ceiling(targetOutput-1e-7));
        var silence=new float[Math.Max(1,converter.InputLatency+1)];
        while(outputSamples<end)Convert(silence,silence.Length,end);
    }
    public void Dispose()=>converter?.Dispose();
}

// Paired cursor/time observations separate device startup and drift from packet/update timing.
internal sealed class SampleClock
{
    readonly double nominal;
    double origin,rate,lastSample,lastTime;
    bool initialized;
    public SampleClock(double rate) {if(!double.IsFinite(rate) || rate<=0)throw new ArgumentOutOfRangeException(nameof(rate));nominal=this.rate=rate;}
    public double At(double sample)=>origin+sample/rate;
    public void Observe(double sample,double time)
    {
        if(!double.IsFinite(sample) || !double.IsFinite(time) || sample<0)throw new ArgumentException("Invalid capture clock.");
        if(!initialized) {initialized=true;origin=time-sample/rate;lastSample=sample;lastTime=time;return;}
        if(sample<lastSample)throw new InvalidOperationException("Capture clock reset; start a new epoch.");
        if(sample==lastSample)return;
        if(time<lastTime)throw new InvalidOperationException("Capture clock reset; start a new epoch.");
        double elapsed=time-lastTime;if(elapsed<1 || sample==lastSample)return;
        double predicted=At(sample),observed=(sample-lastSample)/elapsed;
        rate+=.1*(Math.Clamp(observed,nominal*.995,nominal*1.005)-rate);
        origin=predicted-sample/rate+Math.Clamp((time-predicted)*.1,-.001*elapsed,.001*elapsed);
        lastSample=sample;lastTime=time;
    }
}
