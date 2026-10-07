using System;
using System.Collections.Generic;
using System.IO;
using NUnit.Framework;
using UnityEngine;

public partial class TakeChecks
{
    [TestCase(false),TestCase(true),Category("Voice")]
    public void SelfMutedPendingMicrophoneDoesNotBlockOtherAudio(bool gpu)
    {
        var scene=Scene();var states=States();states[1].flags|=AcousticTake.StateFlags.Source;
        scene.sources=new[]{scene.sources[0],new AcousticTake.SourceDefinition {objectIndex=1,clipIndex=-1,streamed=true,mutedListenerPlusOne=1}};
        var settings=gpu?SpectralSettings():new TakeRender.Settings {responseStepSamples=128};settings.raysPerBin=1;settings.maxBounces=1;
        using var stream=new SourceStream(128);
        using var live=new TakeRender.Live(scene,Array.Empty<AcousticTake.Primitive>(),new[]{new[]{.1f}},8192,settings,null,0,0,gpu?RequireGpu():null,new[]{null,stream});
        live.Append(new AcousticTake.VoiceEvent {gain=1,loop=true});live.Observe(0,0,states,states);
        double energy=0;live.Observe(0,1024,states,states,(sample,pcm,count)=> {for(int n=0;n<count;n++)energy+=pcm[n]*pcm[n];});live.Pump(true);
        Assert.That(live.Cursor,Is.EqualTo(1024));Assert.That(energy,Is.GreaterThan(.001));Assert.That(stream.End,Is.Zero);
    }
    [Test,Category("Voice"),Category("GPU")]
    public void UnusedSlotsDoNotTraceButDeactivationRetainsArrivalPaths()
    {
        var scene=Scene();var states=States();var settings=SpectralSettings();settings.raysPerBin=1;settings.maxBounces=1;
        using var solver=new TransportSolver(scene,Array.Empty<AcousticTake.Primitive>(),settings,RequireGpu(),sampleRate:8192);
        var direct=new TakeRender.DirectPath[1];states[0].flags&=~AcousticTake.StateFlags.Active;
        solver.Solve(states,direct);foreach(var path in solver.RayPaths)Assert.That(path.pressureGain,Is.Zero);
        states[0].flags|=AcousticTake.StateFlags.Active;solver.Solve(states,direct);
        Assert.That(solver.RayPaths[0].pressureGain,Is.GreaterThan(0));
        states[0].flags&=~AcousticTake.StateFlags.Active;solver.Solve(states,direct);
        Assert.That(solver.RayPaths[0].pressureGain,Is.GreaterThan(0),"Past emission must retain its geometric path after deactivation.");
    }
    [TestCase(false,TakeRender.Synthesis.ImpulseResponse),TestCase(true,TakeRender.Synthesis.ImpulseResponse)]
    [TestCase(false,TakeRender.Synthesis.Spectral),TestCase(true,TakeRender.Synthesis.Spectral),Category("Voice")]
    public void SelfMonitorMaskPreservesTheOtherListenersVoice(bool gpu,TakeRender.Synthesis synthesis)
    {
        if(gpu)RequireGpu();
        var scene=Scene(2);scene.sources[0].mutedListenerPlusOne=1;var states=States(2);
        var settings=SpectralSettings();settings.synthesis=synthesis;settings.raysPerBin=1;settings.maxBounces=1;
        var signal=new TakeRender.Signal(new[]{.1f});signal.SetActive(0,true);signal.Append(new AcousticTake.VoiceEvent {gain=1,loop=true});
        var arrivals=new RayTransport.Path[settings.SpectralSlotsPerPair*2];
        for(int pair=0;pair<2;pair++)for(int bin=0;bin<settings.FrequencyBins;bin++)
            arrivals[pair*settings.SpectralSlotsPerPair+bin]=new RayTransport.Path {frequencyBin=(uint)bin,pressureGain=1,delaySeconds=.01f};
        using var buffer=gpu?new ComputeBuffer(arrivals.Length,32):null;buffer?.SetData(arrivals);
        using var accelerated=gpu?new GpuAudio(scene,8192,settings,new[]{signal}):null;
        var reference=gpu?null:new TakeRender.Mixer(scene,8192,settings,new[]{signal});
        var block=new float[settings.responseStepSamples*2];double energy=0;
        for(int frame=0;frame<8;frame++)
        {
            long sample=frame*(long)settings.responseStepSamples;
            if(gpu) {accelerated.SetResponse(sample,false,buffer,0,scene,states);accelerated.Render(sample,settings.responseStepSamples,block);}
            else {reference.SetResponse(sample,new TakeRender.DirectPath[2],states,false,arrivals,0);reference.Render(sample,settings.responseStepSamples,block);}
            for(int i=0;i<block.Length;i+=2) {Assert.That(block[i],Is.Zero);energy+=block[i+1]*block[i+1];}
        }
        Assert.That(energy,Is.GreaterThan(.01));
    }
    [Test,Category("Voice")]
    public void FinishingWaitsForOutputSpaceWithoutDroppingTheTail()
    {
        var scene=Scene();var states=States();var settings=new TakeRender.Settings {tailSamples=64,responseStepSamples=32};
        var queue=new AudioManager.PcmQueue(32,8000);
        using var live=new TakeRender.Live(scene,Array.Empty<AcousticTake.Primitive>(),new[]{new[]{.1f}},8000,settings,queue,0,0);
        live.Append(new AcousticTake.VoiceEvent {gain=1,loop=true});live.Observe(0,0,states,states);live.Observe(0,64,states,states);
        live.Finish(64,states,wait:false);
        Assert.That(live.Completed,Is.False,"A full device queue must defer the remaining tail.");
        for(int step=0;!live.Completed && step<32;step++) {queue.Consume(new float[16],1);live.Pump();}
        Assert.That(live.Completed,Is.True);Assert.That(queue.WrittenSample,Is.EqualTo(129));
        Assert.That(queue.OverflowSamples,Is.Zero);Assert.That(queue.UnderrunSamples,Is.Zero);
    }
    [Test,Category("Voice")]
    public void SourceHistoryDistinguishesPendingSilenceAndExpiredSamples()
    {
        using var stream=new SourceStream(4);
        Assert.Throws<InvalidOperationException>(()=>stream.Read(0));
        stream.Append(new[]{.1f,.2f,.3f},0,3,3);
        stream.SilenceUntil(4,4);
        Assert.That(stream.Read(3),Is.Zero);
        Assert.Throws<InvalidOperationException>(()=>stream.Append(new[]{1f},0,1,5));
        stream.Prune(2);stream.Append(new[]{.4f,.5f},0,2,6);
        Assert.That(stream.Read(2),Is.EqualTo(.3f));Assert.That(stream.Read(5),Is.EqualTo(.5f));
        Assert.Throws<InvalidOperationException>(()=>stream.Read(1));
        stream.Finish();Assert.That(stream.Read(6),Is.Zero);
        Assert.Throws<ArgumentException>(()=>stream.Append(new[]{1f},0,1,7));
    }
    [Test,Category("Voice")]
    public void SavedSourcePayloadAndGapLogRoundTripAndRejectTampering()
    {
        string path=Path.Combine(directory,"voice");var scene=Scene();scene.clips=Array.Empty<AcousticTake.ClipDefinition>();scene.sources[0].streamed=true;scene.sources[0].clipIndex=-1;
        using(var source=new SourceStream(128))
        using(var writer=new AcousticTake.Writer(path,48000,scene,Array.Empty<AcousticTake.Primitive>(),Array.Empty<float[]>()))
        {
            using var capture=new SourceStream.Capture(path,new[]{source},writer.Metadata);
            source.SilenceUntil(8,8);source.Append(new[]{.1f,-.2f,.3f},0,3,14);source.SilenceUntil(32,32,SourceStream.Status.Missing);
            source.Finish();capture.Complete(writer.Metadata);
            writer.Append(0,States());writer.Append(31,States());writer.Append(new AcousticTake.VoiceEvent {gain=1});writer.Complete();
        }
        using(var take=AcousticTake.Open(path))
        {
            Assert.That(take.ReadStream(0).Read(8),Is.EqualTo(.1f));Assert.That(take.ReadStream(0).Read(11),Is.Zero);
            Assert.That(take.Metadata.streams[0].samples,Is.EqualTo(32));Assert.That(AcousticTake.SourceHash(take.Metadata),Is.Not.Null);
        }
        using(var file=File.OpenWrite(Path.Combine(path,"source-0.f32")))file.WriteByte(1);
        Assert.Throws<InvalidDataException>(()=>AcousticTake.Open(path));
    }
    [TestCase(TakeRender.Synthesis.ImpulseResponse,false),TestCase(TakeRender.Synthesis.Spectral,false)]
    [TestCase(TakeRender.Synthesis.ImpulseResponse,true),TestCase(TakeRender.Synthesis.Spectral,true),Category("Voice")]
    public void StreamedAndClipSourcesProduceTheSameAudio(TakeRender.Synthesis synthesis,bool gpu)
    {
        var shader=RequireGpu();
        const int rate=8192,size=128,length=2048;
        var scene=Scene();var state=States();var settings=SpectralSettings();
        settings.synthesis=synthesis;settings.fftSize=size;settings.spectralHop=32;settings.responseStepSamples=size;
        settings.raysPerBin=1;settings.maxBounces=1;
        var pcm=new float[length];for(int i=0;i<pcm.Length;i++)pcm[i]=.2f*Mathf.Sin(i*.17f);
        using var stream=new SourceStream(length);stream.Append(pcm,0,pcm.Length,length);stream.Finish();
        var signals=new[]{new TakeRender.Signal(pcm),new TakeRender.Signal(stream,32)};
        foreach(var signal in signals) {signal.SetActive(0,true);signal.Append(new AcousticTake.VoiceEvent {gain=1});}
        using var solver=new TransportSolver(scene,Array.Empty<AcousticTake.Primitive>(),settings,shader,sampleRate:rate);
        var direct=new TakeRender.DirectPath[1];solver.Solve(state,direct);
        using var a=gpu?new GpuAudio(scene,rate,settings,new[]{signals[0]}):null;
        using var b=gpu?new GpuAudio(scene,rate,settings,new[]{signals[1]}):null;
        var cpuA=gpu?null:new TakeRender.Mixer(scene,rate,settings,new[]{signals[0]});
        var cpuB=gpu?null:new TakeRender.Mixer(scene,rate,settings,new[]{signals[1]});
        var left=new float[size];var right=new float[size];double energy=0;
        for(int offset=0;offset<length;offset+=size)
        {
            if(gpu)
            {
                a.SetResponse(offset,false,solver.RayBuffer,0,scene,state);b.SetResponse(offset,false,solver.RayBuffer,0,scene,state);
                a.Render(offset,size,left);b.Render(offset,size,right);
            }
            else
            {
                cpuA.SetResponse(offset,direct,state,false,solver.RayPaths,0);cpuB.SetResponse(offset,direct,state,false,solver.RayPaths,0);
                cpuA.Render(offset,size,left);cpuB.Render(offset,size,right);
            }
            Assert.That(right,Is.EqualTo(left));foreach(float value in left)energy+=value*value;
        }
        Assert.That(energy,Is.GreaterThan(.001));
    }
    [Test,Category("Voice")]
    public void VoicePacketReorderingIsExactAndLateLossIsFinal()
    {
        var packets=new List<byte[]>();var encoder=new VoiceCodec.Encoder((sequence,data,count)=> {var p=new byte[count];Array.Copy(data,p,count);packets.Add(p);});
        var pcm=new float[VoiceCodec.Frame*4];for(int i=0;i<pcm.Length;i++)pcm[i]=.3f*Mathf.Sin(2*Mathf.PI*440*i/48000);
        encoder.Append(pcm,pcm.Length);encoder.Finish();Assert.That(packets.Count,Is.EqualTo(4));
        var ordered=new List<float>();var shuffled=new List<float>();
        using(var a=new VoiceCodec.Receiver(48000,(data,count,status)=> {for(int i=0;i<count;i++)ordered.Add(data[i]);}))
        using(var b=new VoiceCodec.Receiver(48000,(data,count,status)=> {for(int i=0;i<count;i++)shuffled.Add(data[i]);}))
        {
            for(int i=0;i<4;i++)Assert.That(a.Add(i,packets[i]),Is.True);
            foreach(int i in new[]{2,0,3,1})Assert.That(b.Add(i,packets[i]),Is.True);
            Assert.That(b.Add(0,packets[0]),Is.False);a.Stop(4);b.Stop(4);
            while(a.DecodeNext())Assert.That(b.DecodeNext(),Is.True);
            Assert.That(b.DecodeNext(),Is.False);Assert.That(shuffled,Is.EqualTo(ordered));
        }
        using(var loss=new VoiceCodec.Receiver(48000,(data,count,status)=> {}))
        {
            loss.Add(0,packets[0]);loss.Add(2,packets[2]);loss.Stop(3);
            loss.DecodeNext();loss.DecodeNext();Assert.That(loss.Add(1,packets[1]),Is.False);loss.DecodeNext();
            Assert.That(loss.Concealed,Is.EqualTo(1));Assert.That(loss.Late,Is.EqualTo(1));Assert.That(loss.DecodeNext(),Is.False);
        }
    }
    [Test,Category("Voice")]
    public void RateConversionDoesNotDependOnCaptureBlockBoundaries()
    {
        var pcm=new float[4800];for(int i=0;i<pcm.Length;i++)pcm[i]=Mathf.Sin(i*.2f);
        var a=new List<float>();var b=new List<float>();
        using(var whole=new PcmRateConverter(48000,44100,(data,count)=> {for(int i=0;i<count;i++)a.Add(data[i]);})) {whole.Process(pcm,pcm.Length);whole.Finish();}
        using(var split=new PcmRateConverter(48000,44100,(data,count)=> {for(int i=0;i<count;i++)b.Add(data[i]);}))
        {
            for(int at=0;at<pcm.Length;) {int count=Math.Min(173,pcm.Length-at);var block=new float[count];Array.Copy(pcm,at,block,0,count);split.Process(block,count);at+=count;}
            split.Finish();
        }
        Assert.That(a.Count,Is.EqualTo(4410));Assert.That(b,Is.EqualTo(a));
    }
}

// Explicit: opens the local device briefly; saves/transmits no microphone samples.
public sealed class MicrophoneDeviceTests : SceneEndToEndTests
{
    protected override string ScenePath=>"Assets/Scenes/Church.unity";
    [UnityEngine.TestTools.UnityTest,Explicit("Uses the permitted default microphone locally."),Category("Voice"),Category("Device")]
    public System.Collections.IEnumerator MicrophoneCapturesAndClosesItsOpenedDevice()
    {
        if(Microphone.devices.Length==0 || !Application.HasUserAuthorization(UserAuthorization.Microphone))
            Assert.Ignore("Grant microphone permission manually before selecting the device test.");
        yield return StartSession();session.SaveRecording();
        yield return Until(()=>!session.Busy,"saving before device capture");
        Assert.That(session.manager.IsRecordingTake,Is.False);Assert.That(session.multiplayer.Connected,Is.False);
        var microphone=session.multiplayer.microphone;var stream=microphone.source.Stream;
        Assert.That(stream.Recorded,Is.Null,"No source recording may be attached during this device test.");
        string selected=microphone.device,opened=string.IsNullOrEmpty(selected)?null:selected;long samples=0;
        try
        {
            stream.Recorded=(first,data,offset,count,available,status)=> {if(status==SourceStream.Status.Data)samples+=count;};
            microphone.Toggle();yield return Until(()=>microphone.Active,"microphone startup",5);
            yield return Until(()=>samples>=session.manager.sampleRate,"one second of canonical device samples",5);
            int rate=microphone.CaptureRate;
            microphone.device="Unopened device (test)";microphone.StopCapture();
            Assert.That(microphone.Active,Is.False);Assert.That(Microphone.IsRecording(opened),Is.False,"Stop must close the opened device even after the selection changes.");
            Assert.That(session.manager.IsRecordingTake,Is.False);Assert.That(session.multiplayer.Connected,Is.False);
            TestContext.Progress.WriteLine($"Device rate {rate}; canonical rate {session.manager.sampleRate}; received {samples} samples; no audio saved or transmitted.");
        }
        finally {microphone.device=selected;microphone.StopCapture();stream.Recorded=null;}
    }
}
