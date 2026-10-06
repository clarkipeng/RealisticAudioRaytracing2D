using UnityEngine;
using System.IO;
using System;
using System.Threading;
using System.Diagnostics;
using Debug = UnityEngine.Debug;

public partial class AudioManager : MonoBehaviour
{
    // One main-thread producer, one audio-thread consumer. Publish after writing the entire block.
    public sealed class PcmQueue
    {
        readonly float[] samples;
        internal readonly int rate;
        internal readonly PlaybackCapture capture;
        long read,write,finalSample=long.MaxValue;
        public long UnderrunSamples,OverflowSamples,LateSamples,CallbackCount,MaximumCallbackTicks,OverBudgetCallbacks,ChannelMismatchSamples;
        public int Channels {get;private set;}
        public long ReadSample => Volatile.Read(ref read);
        public long WrittenSample => Volatile.Read(ref write);
        public int Capacity => samples.Length/Channels;
        public bool IsDrained => ReadSample>=Volatile.Read(ref finalSample);
        public PcmQueue(int capacity,int rate=0,PlaybackCapture capture=null,int outputChannels=1)
        {
            if(capacity<=0 || rate<0 || (outputChannels!=1 && outputChannels!=2)) throw new ArgumentOutOfRangeException(nameof(capacity));
            if(capture!=null && capture.Metadata.channelsPerListener!=outputChannels) throw new ArgumentException("Capture/queue channel mismatch.");
            this.capture=capture; Channels=outputChannels; samples=new float[checked(capacity*Channels)]; this.rate=rate;
        }
        public bool Publish(long first,float[] interleaved,int frames,int channels=1,int channel=0)
        {
            if (first<0 || frames<0 || channels<1 || channel<0 || channel>=channels || (interleaved!=null && (channel+Channels>channels || interleaved.Length<(long)frames*channels)))
                throw new ArgumentException("Invalid PCM publication.");
            long consumed=Volatile.Read(ref read),end=checked(first+frames);
            if (end-consumed>Capacity) { OverflowSamples+=frames; return false; }
            long begin=Math.Max(first,consumed);
            LateSamples+=Math.Max(0,Math.Min(end,consumed)-first);
            if (end<=begin) return false;
            long committed=Volatile.Read(ref write);
            if (first<committed) throw new InvalidOperationException("PCM publications must not overlap.");
            for (long i=Math.Max(committed,consumed); i<begin; i++) for(int c=0;c<Channels;c++) samples[(i%Capacity)*Channels+c]=0;
            for (long i=begin; i<end; i++) for(int c=0;c<Channels;c++) samples[(i%Capacity)*Channels+c]=interleaved==null ? 0:interleaved[(i-first)*channels+channel+c];
            Volatile.Write(ref write,end); return true;
        }
        public void Complete() { Volatile.Write(ref finalSample,WrittenSample); capture?.CompleteProducer(WrittenSample); }
        public void Consume(float[] data,int channels,bool silence=false,double dspTime=double.NaN)
        {
            long start=Stopwatch.GetTimestamp(),cursor=read,available=Volatile.Read(ref write),missed=0;
            bool mismatch=channels<Channels;
            if(mismatch) Interlocked.Add(ref ChannelMismatchSamples,data.Length/channels);
            for (int n=0; n<data.Length/channels; n++,cursor++)
            {
                if (cursor>=available && cursor<Volatile.Read(ref finalSample)) missed++;
                for(int c=0;c<channels;c++)
                    data[n*channels+c]=!silence && !mismatch && cursor<available && (Channels==1 || c<Channels) ? samples[(cursor%Capacity)*Channels+(Channels==1 ? 0:c)]:0;
            }
            capture?.Publish(read,data,channels,start,dspTime);
            Volatile.Write(ref read,cursor);
            Interlocked.Add(ref UnderrunSamples,missed); Interlocked.Increment(ref CallbackCount);
            long elapsed=Stopwatch.GetTimestamp()-start;
            if (rate>0 && elapsed>(double)(data.Length/channels)*Stopwatch.Frequency/rate) Interlocked.Increment(ref OverBudgetCallbacks);
            if (elapsed>MaximumCallbackTicks) Interlocked.Exchange(ref MaximumCallbackTicks,elapsed);
        }
    }
    // Audio thread writes bounded PCM/metadata; the main thread owns every file operation.
    public sealed class PlaybackCapture : IDisposable
    {
        [Serializable] public sealed class Report
        {
            public int formatVersion=3,channelsPerListener;
            public string scope="selected listener PCM; mono duplicates to device channels, binaural maps left/right; before Unity mixer/device processing";
            public string hrtfDataset,hrtfHash;
            public int hrtfTaps;
            public string callbackLayout="int64 firstSample, int32 frames, int32 channels, int64 startTicks, int64 durationTicks, float64 dspTimeSeconds";
            public string responseLayout="int64 scheduledSample, int64 consumedSampleAtApplication, int64 applicationTicks, int32 pairs, float32(delaySeconds,gain,directionX,directionY,directionZ,delayRate) per pair; world listener-to-arrival-origin";
            public int sampleRate,leadSamples,listener,capacity;
            public double inputDspOriginSeconds;
            public long samples,callbacks,overflowSamples,channelMismatchSamples,stopwatchFrequency=Stopwatch.Frequency,requestedEndSample;
            public bool complete,aborted;
            public string audioHash,callbackHash,responseHash;
            public AcousticTake.Provenance provenance;
            public TakeRender.Settings settings;
        }
        struct Block { public long first,startTicks,durationTicks; public int frames,channels; public double dspTime; }
        readonly float[] samples;
        readonly Block[] blocks;
        readonly string directory;
        readonly BinaryWriter audio,callbacks,responses;
        long writtenFrames,readFrames,writtenBlocks,readBlocks,finishAfter=long.MaxValue,overflow,channelMismatch;
        readonly int outputChannels,capacityFrames;
        volatile bool done,closed;
        bool producerComplete;
        public bool Closed => closed;
        public Report Metadata {get;private set;}
        public PlaybackCapture(string directory,int rate,int capacity,int callbackFrames,int lead,int listener,TakeRender.Settings settings,ComputeShader shader=null)
        {
            if (rate<=0 || capacity<1 || callbackFrames<1) throw new ArgumentOutOfRangeException(nameof(capacity));
            if (Directory.Exists(directory)) throw new IOException("Output capture directory already exists.");
            this.directory=directory; Directory.CreateDirectory(directory);
            outputChannels=settings.ChannelsPerListener; capacityFrames=capacity;
            samples=new float[checked(capacity*outputChannels)]; blocks=new Block[Math.Max(2,capacity/callbackFrames+2)];
            Metadata=new Report {sampleRate=rate,capacity=capacity,channelsPerListener=outputChannels,leadSamples=lead,listener=listener,settings=settings,provenance=AcousticTake.Provenance.Capture(shader,settings)};
            if(settings.transport==TakeRender.Transport.Raytraced)Metadata.responseLayout="Direct opaque diagnostic probes only; PCM includes indirect/transmitted transport. "+Metadata.responseLayout;
            if(outputChannels==2) {var hrtf=Hrtf.Load(rate); Metadata.hrtfDataset=Hrtf.Dataset; Metadata.hrtfHash=hrtf.Hash; Metadata.hrtfTaps=hrtf.Taps;}
            File.WriteAllText(Path.Combine(directory,"output.json"),JsonUtility.ToJson(Metadata,true));
            try
            {
                audio=new BinaryWriter(File.Create(Path.Combine(directory,"output.wav"))); TakeRender.WriteWaveHeader(audio,rate,0,outputChannels);
                callbacks=new BinaryWriter(File.Create(Path.Combine(directory,"callbacks.bin")));
                responses=new BinaryWriter(File.Create(Path.Combine(directory,"applications.bin")));
            }
            catch { audio?.Dispose(); callbacks?.Dispose(); responses?.Dispose(); throw; }
        }
        public void RecordResponse(long scheduled,long consumed,TakeRender.DirectPath[] paths)
        {
            if (producerComplete || closed) return;
            responses.Write(scheduled); responses.Write(consumed); responses.Write(Stopwatch.GetTimestamp()); responses.Write(paths.Length);
            foreach (var path in paths) TakeRender.WritePath(responses,path);
        }
        public void CompleteProducer(long sample)
        {
            if (producerComplete || closed) return;
            responses.Flush(); producerComplete=true; Metadata.requestedEndSample=sample;
            Volatile.Write(ref finishAfter,sample);
        }
        public void Publish(long first,float[] data,int channels,long startTicks,double dspTime=double.NaN)
        {
            if (done || closed) return;
            int frames=data.Length/channels; long frame=writtenFrames,block=writtenBlocks;
            if (frame+frames-Volatile.Read(ref readFrames)>capacityFrames || block-Volatile.Read(ref readBlocks)>=blocks.Length)
            { Interlocked.Add(ref overflow,frames); done=true; return; }
            if(channels<outputChannels) Interlocked.Add(ref channelMismatch,frames);
            for (int i=0;i<frames;i++) for(int c=0;c<outputChannels;c++) samples[((frame+i)%capacityFrames)*outputChannels+c]=c<channels ? data[i*channels+c]:0;
            blocks[block%blocks.Length]=new Block {first=first,frames=frames,channels=channels,startTicks=startTicks,durationTicks=Stopwatch.GetTimestamp()-startTicks,dspTime=dspTime};
            writtenFrames=frame+frames; Volatile.Write(ref writtenBlocks,block+1);
            if (first+frames>=Volatile.Read(ref finishAfter)) done=true;
        }
        public void Drain()
        {
            if (closed) return;
            bool completionObserved=done;
            long end=Volatile.Read(ref writtenBlocks);
            while (readBlocks<end)
            {
                var block=blocks[readBlocks%blocks.Length];
                if ((readFrames+block.frames)>(uint.MaxValue-36)/(4L*outputChannels)) { Finish(true); return; }
                callbacks.Write(block.first); callbacks.Write(block.frames); callbacks.Write(block.channels); callbacks.Write(block.startTicks); callbacks.Write(block.durationTicks); callbacks.Write(block.dspTime);
                for (int i=0;i<block.frames;i++) for(int c=0;c<outputChannels;c++) audio.Write(samples[((readFrames+i)%capacityFrames)*outputChannels+c]);
                Volatile.Write(ref readFrames,readFrames+block.frames); Volatile.Write(ref readBlocks,readBlocks+1);
            }
            if (completionObserved && (producerComplete || Interlocked.Read(ref overflow)>0)) Finish(false);
        }
        void Finish(bool aborted)
        {
            if (closed) return;
            closed=true; Metadata.samples=readFrames; Metadata.callbacks=readBlocks; Metadata.overflowSamples=Interlocked.Read(ref overflow); Metadata.channelMismatchSamples=Interlocked.Read(ref channelMismatch);
            Metadata.aborted=aborted; Metadata.complete=!aborted && producerComplete && Metadata.overflowSamples==0 && Metadata.channelMismatchSamples==0;
            try { audio.BaseStream.Position=0; TakeRender.WriteWaveHeader(audio,Metadata.sampleRate,readFrames,outputChannels); }
            finally { CloseFiles(); }
            Metadata.audioHash=AcousticTake.HashFile(Path.Combine(directory,"output.wav"));
            Metadata.callbackHash=AcousticTake.HashFile(Path.Combine(directory,"callbacks.bin"));
            Metadata.responseHash=AcousticTake.HashFile(Path.Combine(directory,"applications.bin"));
            File.WriteAllText(Path.Combine(directory,"output.json"),JsonUtility.ToJson(Metadata,true));
        }
        void CloseFiles()
        {
            try { audio.Dispose(); } finally { try { callbacks.Dispose(); } finally { responses.Dispose(); } }
        }
        public void Abort() { if (!closed) { closed=true; CloseFiles(); } }
        public void Dispose() { if (!closed) { try { Drain(); Finish(true); } finally { Abort(); } } }
    }

    PcmQueue pcmQueue;
    public PlaybackCapture OutputCapture {get;private set;}
    public bool IsCapturingOutput => OutputCapture!=null;
    public PlaybackCapture.Report LastOutputCapture {get;private set;}
    AudioClip driverClip;
    AudioSource driverSource;
    public long LegacyBufferBytes => ringBuffer==null ? 0:(long)ringBuffer.Length*4;
    volatile bool pcmMode;
    public PcmQueue OutputQueue => Volatile.Read(ref pcmQueue);
    public PcmQueue StartPcmStream(int rate,int capacity,int leadSamples,PlaybackCapture capture=null)
    {
        if (OutputCapture!=null) throw new InvalidOperationException("The previous output capture is still draining.");
        if (rate!=AudioSettings.outputSampleRate) throw new InvalidOperationException("Live monitoring requires the audio device sample rate; offline takes can use other rates.");
        if (leadSamples<0 || leadSamples>capacity) throw new ArgumentOutOfRangeException(nameof(leadSamples));
        var queue=new PcmQueue(capacity,rate,capture,capture?.Metadata.channelsPerListener ?? 1); queue.Publish(0,null,leadSamples);
        StartPcmStream(queue); return queue;
    }
    // Activate only after solver and recorder preparation; callbacks must not consume lead time during setup.
    public void StartPcmStream(PcmQueue queue)
    {
        if (queue==null || queue.rate!=AudioSettings.outputSampleRate) throw new ArgumentException("Prepared queue must use the output device sample rate.");
        if (OutputCapture!=null || OutputQueue!=null) throw new InvalidOperationException("Finish the previous output stream before activating another.");
        sampleRate=queue.rate; OutputCapture=queue.capture; pcmMode=true; Volatile.Write(ref pcmQueue,queue);
    }
    public void StopPcmStream()
    {
        Volatile.Write(ref pcmQueue,null);
        OutputCapture?.Dispose(); OutputCapture=null;
    }

    [Header("Debug")]
    public int writeHead;
    public int readHead_Debug;
    public int bufferCapacity;
    public long writeSampleCursor_Debug;
    public long readSampleCursor_Debug;
    public int writeLoopCount_Debug;
    public int readLoopCount_Debug;
    public int underrunCount_Debug;
    
    public RayTraceManager rayTraceManager;
    public bool paused;
    int sampleRate;

    void Awake()
    {
        pcmMode=rayTraceManager!=null && rayTraceManager.useTakeRenderer;
        sampleRate = AudioSettings.outputSampleRate;
        
        var src = driverSource = gameObject.AddComponent<AudioSource>();
        src.playOnAwake = true;
        src.loop = true;
        AudioSettings.GetDSPBufferSize(out int frames,out var buffers);
        var clip = driverClip = AudioClip.Create("Acoustic output driver", frames, 2, sampleRate, false);
        src.clip = clip;
        src.Play();
    }

    void OnAudioFilterRead(float[] data, int channels)
    {
        if (pcmMode)
        {
            var queue=Volatile.Read(ref pcmQueue);
            if (queue==null) Array.Clear(data,0,data.Length);
            else queue.Consume(data,channels,paused,AudioSettings.dspTime);
            return;
        }
        ConsumeLegacy(data,channels);
    }

    void Update()
    {
        if (OutputCapture!=null)
        {
            try { OutputCapture.Drain(); }
            catch { OutputCapture.Abort(); OutputCapture=null; throw; }
            if (OutputCapture.Closed)
            {
                LastOutputCapture=OutputCapture.Metadata;
                if (!LastOutputCapture.complete) Debug.LogError("Acoustic output capture is incomplete; inspect monitor/output.json.",this);
                OutputCapture=null;
            }
        }
        if (pcmMode)
        {
            var queue=OutputQueue;
            if (queue!=null)
            {
                if (queue.IsDrained && OutputCapture==null) Volatile.Write(ref pcmQueue,null);
                bufferCapacity=queue.Capacity; readSampleCursor_Debug=queue.ReadSample; writeSampleCursor_Debug=queue.WrittenSample;
                underrunCount_Debug=(int)Math.Min(int.MaxValue,Interlocked.Read(ref queue.UnderrunSamples));
            }
            return;
        }
        UpdateLegacy();
    }

    void OnDestroy()
    {
        StopPcmStream();
        if (driverSource!=null) driverSource.Stop();
        if (driverClip!=null) { if (Application.isPlaying) Destroy(driverClip); else DestroyImmediate(driverClip); }
        DisposeLegacy();
    }
}
