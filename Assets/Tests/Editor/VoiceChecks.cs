using System;
using System.Collections.Generic;
using System.IO;
using NUnit.Framework;
using UnityEngine;

public partial class TakeChecks
{
    sealed class GatedRecordingStream : MemoryStream
    {
        public readonly System.Threading.ManualResetEventSlim entered=new System.Threading.ManualResetEventSlim(),release=new System.Threading.ManualResetEventSlim();
        public override void Write(byte[] buffer,int offset,int count) {entered.Set();release.Wait();base.Write(buffer,offset,count);}
    }
    [Test,Category("Voice")]
    public void FailedRecordingDetachesWithoutWaitingForDisk()
    {
        string path=Path.Combine(directory,"stalled-writer");Directory.CreateDirectory(path);
        using var source=new SourceStream(1);using var capture=new SourceStream.Capture(path,new[]{source},new AcousticTake.Manifest());
        using var gate=new GatedRecordingStream();
        var files=(BinaryWriter[])typeof(SourceStream.Capture).GetField("pcm",System.Reflection.BindingFlags.Instance|System.Reflection.BindingFlags.NonPublic).GetValue(capture);
        files[0].Dispose();files[0]=new BinaryWriter(gate);
        try
        {
            source.Append(new[]{.25f},0,1,1);source.Prune(1);
            Assert.That(gate.entered.Wait(5000),Is.True,"Writer must be blocked inside disk IO.");
            for(int i=0;i<129;i++){source.Append(new[]{.25f},0,1,i+2);source.Prune(source.End);}
            Assert.That(capture.Failure,Is.TypeOf<IOException>());
            var pending=capture.Abort();Assert.That(source.Recorded,Is.Null);Assert.That(pending.IsCompleted,Is.False);
            gate.release.Set();Assert.That(pending.Wait(5000),Is.True);Assert.That(capture.Abort(),Is.SameAs(pending));
            Assert.Throws<IOException>(()=>capture.Complete(new AcousticTake.Manifest()));
        }
        finally {gate.release.Set();}
    }

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
    [TestCase(true,TakeRender.Synthesis.Spectral,2048)]
    public void SelfMonitorMaskPreservesTheOtherListenersVoice(bool gpu,TakeRender.Synthesis synthesis,int size=128)
    {
        if(gpu)RequireGpu();
        var scene=Scene(2);scene.sources[0].mutedListenerPlusOne=1;var states=States(2);
        var settings=SpectralSettings();settings.synthesis=synthesis;settings.raysPerBin=1;settings.maxBounces=1;
        settings.fftSize=size;settings.spectralHop=size/4;settings.responseStepSamples=size;
        var signal=new TakeRender.Signal(new[]{.1f});signal.SetActive(0,true);signal.Append(new AcousticTake.VoiceEvent {gain=1,loop=true});
        var arrivals=new RayTransport.Path[settings.SpectralSlotsPerPair*2];
        for(int pair=0;pair<2;pair++)for(int bin=0;bin<settings.FrequencyBins;bin++)
            arrivals[pair*settings.SpectralSlotsPerPair+bin]=new RayTransport.Path {frequencyBin=(uint)bin,pressureGain=1,delaySeconds=.01f};
        using var buffer=gpu?new ComputeBuffer(arrivals.Length,RayTransport.PathStride):null;buffer?.SetData(arrivals);
        using var accelerated=gpu?new GpuAudio(scene,8192,settings,new[]{signal}):null;
        if(gpu)Assert.That(accelerated.Diagnostics.filterBytes,Is.EqualTo(2L*settings.SpectralSlotsPerPair*(size+1)*8),"Only the audible pair owns both filter banks.");
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
        encoder.Append(pcm,pcm.Length);encoder.Finish();Assert.That(packets.Count,Is.EqualTo(5));
        var ordered=new List<float>();var shuffled=new List<float>();
        using(var a=new VoiceCodec.Receiver(48000,(data,count,status)=> {for(int i=0;i<count;i++)ordered.Add(data[i]);},encoder.DelaySamples))
        using(var b=new VoiceCodec.Receiver(48000,(data,count,status)=> {for(int i=0;i<count;i++)shuffled.Add(data[i]);},encoder.DelaySamples))
        {
            for(int i=0;i<packets.Count;i++)Assert.That(a.Add(i*VoiceCodec.Frame,packets[i]),Is.True);
            foreach(int i in new[]{2,0,4,3,1})Assert.That(b.Add(i*VoiceCodec.Frame,packets[i]),Is.True);
            Assert.That(b.Add(0,packets[0]),Is.False);a.Stop(pcm.Length);b.Stop(pcm.Length);
            while(a.DecodeNext())Assert.That(b.DecodeNext(),Is.True);
            Assert.That(b.DecodeNext(),Is.False);Assert.That(shuffled,Is.EqualTo(ordered));Assert.That(ordered.Count,Is.EqualTo(pcm.Length));
        }
        using(var loss=new VoiceCodec.Receiver(48000,(data,count,status)=> {}))
        {
            loss.Add(0,packets[0]);loss.Add(2*VoiceCodec.Frame,packets[2]);loss.Stop(3*VoiceCodec.Frame);
            loss.DecodeNext();loss.DecodeNext();Assert.That(loss.Add(VoiceCodec.Frame,packets[1]),Is.False);loss.DecodeNext();
            Assert.That(loss.Concealed,Is.EqualTo(1));Assert.That(loss.Late,Is.EqualTo(1));Assert.That(loss.DecodeNext(),Is.False);
        }
    }
    [Test,Category("Voice")]
    public void EpochMetadataAcceptsEarlierPayloadAndRestartRetainsPendingSpeech()
    {
        var root=new GameObject("Voice epoch fixture");
        const System.Reflection.BindingFlags members=System.Reflection.BindingFlags.Instance|System.Reflection.BindingFlags.Public|System.Reflection.BindingFlags.NonPublic;
        using var stream=new SourceStream(8192);
        try
        {
            var network=root.AddComponent<MultiplayerSession>();network.session=root.AddComponent<SceneSession>();network.session.manager=root.AddComponent<RayTraceManager>();
            network.session.manager.sampleRate=48000;
            var source=root.AddComponent<RaytracedAudioSource>();source.Stream=stream;network.remoteVoices=new[]{source};
            var peerType=typeof(MultiplayerSession).GetNestedType("VoicePeer",System.Reflection.BindingFlags.NonPublic);
            var peers=Array.CreateInstance(peerType,2);for(int i=0;i<2;i++)peers.SetValue(Activator.CreateInstance(peerType,true),i);
            typeof(MultiplayerSession).GetField("voices",members).SetValue(network,peers);
            var get=typeof(MultiplayerSession).GetMethod("GetVoice",members);
            var first=get.Invoke(network,new object[]{1,1});
            var pending=(List<(long sample,double time,byte[] data)>)peerType.GetField("pending",members).GetValue(first);
            var encoder=new VoiceCodec.Encoder((sample,data,count)=> {var copy=new byte[count];Array.Copy(data,copy,count);pending.Insert(0,(sample,10+(double)sample/48000,copy));});
            var input=new float[2500];for(int i=0;i<input.Length;i++)input[i]=.2f*Mathf.Sin(i*.1f);
            encoder.Append(input,input.Length);encoder.Finish();peerType.GetField("final",members).SetValue(first,(long)input.Length);
            typeof(MultiplayerSession).GetMethod("StartVoice",members).Invoke(network,new object[]{1,1,10.0,0L,encoder.DelaySamples});
            var decoder=(VoiceCodec.Receiver)peerType.GetField("decoder",members).GetValue(first);
            Assert.That(pending.Count,Is.Zero);Assert.That(decoder.Final,Is.EqualTo(input.Length));
            var next=get.Invoke(network,new object[]{1,2});
            Assert.That(next,Is.Not.SameAs(first));Assert.That(get.Invoke(network,new object[]{1,1}),Is.SameAs(first));
            while(decoder.DecodeNext()) {}
            Assert.That(stream.End,Is.EqualTo(input.Length));Assert.That(decoder.Concealed,Is.Zero);
            Assert.That(network.ReceivedPackets,Is.EqualTo(3));
        }
        finally {UnityEngine.Object.DestroyImmediate(root);}
    }
    [TestCase(0,48000),TestCase(1,48000),TestCase(959,48000),TestCase(960,48000),TestCase(961,44100),Category("Voice")]
    public void CodecStopPreservesExactInputLengthIncludingTheDelayedTail(int length,int outputRate)
    {
        var packets=new List<(long sample,byte[] data)>();
        var encoder=new VoiceCodec.Encoder((sample,data,count)=> {var copy=new byte[count];Array.Copy(data,copy,count);packets.Add((sample,copy));});
        var input=new float[length];for(int i=0;i<length;i++)input[i]=.2f*Mathf.Sin(i*.1f);
        encoder.Append(input,length);encoder.Finish();encoder.Finish();
        int total=0;double tailEnergy=0;
        using var receiver=new VoiceCodec.Receiver(outputRate,(data,count,status)=>
        {total+=count;for(int i=0;i<count;i++) {Assert.That(float.IsFinite(data[i]));tailEnergy+=data[i]*data[i];}},encoder.DelaySamples);
        receiver.Stop(length); // Reliable stop can arrive before the final unreliable payload.
        foreach(var packet in packets)Assert.That(receiver.Add(packet.sample,packet.data));
        while(receiver.DecodeNext()) {}
        Assert.That(total,Is.EqualTo((long)Math.Ceiling((double)length*outputRate/48000)));
        if(length>100)Assert.That(tailEnergy,Is.GreaterThan(.001));
    }
    [Test,Category("Voice")]
    public void CodecImpulseUsesCaptureSampleTimeAfterDelayCompensation()
    {
        var packets=new List<(long sample,byte[] data)>();var output=new List<float>();
        var encoder=new VoiceCodec.Encoder((sample,data,count)=> {var copy=new byte[count];Array.Copy(data,copy,count);packets.Add((sample,copy));});
        var input=new float[4096];input[512]=.8f;input[4000]=.8f;encoder.Append(input,input.Length);encoder.Finish();
        using var receiver=new VoiceCodec.Receiver(48000,(data,count,status)=> {for(int i=0;i<count;i++)output.Add(data[i]);},encoder.DelaySamples);
        foreach(var packet in packets)receiver.Add(packet.sample,packet.data);receiver.Stop(input.Length);while(receiver.DecodeNext()) {}
        foreach(int impulse in new[]{512,4000})
        {
            int peak=impulse-32;for(int i=peak+1;i<=impulse+32;i++)if(Math.Abs(output[i])>Math.Abs(output[peak]))peak=i;
            Assert.That(peak,Is.EqualTo(impulse).Within(2));Assert.That(Math.Abs(output[peak]),Is.GreaterThan(.05));
        }
    }
    [Test,Category("Voice")]
    public void CaptureClockAccountsForStartupAndTracksDeviceDrift()
    {
        const double actualRate=48009.6,origin=17.3;var clock=new SampleClock(48000);
        clock.Observe(960,origin+960/actualRate);
        Assert.That(clock.At(0),Is.EqualTo(origin).Within(.00001));
        for(int second=1;second<=120;second++)clock.Observe(second*actualRate,origin+second);
        Assert.That(clock.At(120*actualRate),Is.EqualTo(origin+120).Within(.002));
        Assert.Throws<InvalidOperationException>(()=>clock.Observe(1,origin+121));
    }
    [Test,Category("Voice")]
    public void VariableRateConversionCorrectsDriftWithoutLosingTheFinalSamples()
    {
        long count=0;var input=new float[48048];for(int i=0;i<input.Length;i++)input[i]=.1f*Mathf.Sin(i*.1f);
        using var converter=new PcmRateConverter(48000,48000,(data,n)=> {count+=n;for(int i=0;i<n;i++)Assert.That(float.IsFinite(data[i]));},variableRate:true);
        converter.SetInputRate(48048);
        for(int second=0;second<10;second++)converter.Process(input,input.Length);
        converter.Finish();Assert.That(count,Is.EqualTo(480000));
    }
    [TestCase(44100),TestCase(48000),Category("Voice")]
    public void ChangingFractionalRatesPreservesResamplerPhaseWithoutOverflow(int rate)
    {
        long count=0;double expected=0;var input=new float[960];
        using var converter=new PcmRateConverter(48000,rate,(data,n)=> {count+=n;for(int i=0;i<n;i++)Assert.That(float.IsFinite(data[i]));},variableRate:true);
        for(int frame=0;frame<300;frame++)
        {
            double frequency=48000+180*Math.Sin(frame*.19);
            converter.SetInputRate(frequency);expected+=960*rate/frequency;
            for(int i=0;i<input.Length;i++)input[i]=.1f*Mathf.Sin((frame*960+i)*.1f);
            converter.Process(input,input.Length);
        }
        converter.Finish();Assert.That(count,Is.EqualTo(expected).Within(8));
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
