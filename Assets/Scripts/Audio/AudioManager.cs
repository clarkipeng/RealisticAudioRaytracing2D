using UnityEngine;
using System.Collections.Generic;
using System.IO;
using Helpers;
using System;
using System.Threading;
using System.Diagnostics;
using Debug = UnityEngine.Debug;

public class AudioManager : MonoBehaviour
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
            public int formatVersion=2,channelsPerListener;
            public string scope="selected listener PCM; mono duplicates to device channels, binaural maps left/right; before Unity mixer/device processing";
            public string hrtfDataset,hrtfHash;
            public int hrtfTaps;
            public string callbackLayout="int64 firstSample, int32 frames, int32 channels, int64 startTicks, int64 durationTicks, float64 dspTimeSeconds";
            public string responseLayout="int64 scheduledSample, int64 consumedSampleAtApplication, int64 applicationTicks, int32 pairs, float32(delaySeconds,gain,directionX,directionY,directionZ) per pair; world listener-to-arrival-origin";
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
            Metadata=new Report {sampleRate=rate,capacity=capacity,channelsPerListener=outputChannels,leadSamples=lead,listener=listener,settings=settings,provenance=AcousticTake.Provenance.Capture(shader)};
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
    
    [Header("Waveform Visualization")]
    public ComputeShader shader;
    public RayTraceManager rayTraceManager;
    public bool showWaveform = true;
    public Vector2 waveformTextureSize = new Vector2(1024, 256);
    [Range(0.1f, 100f)] public float waveformGain = 10.0f;
    public RenderTexture waveformTexture;

    int safetyBuffer = 4096; // Minimum samples between read and write heads to avoid underrun

    float[] ringBuffer;
    int readHead, sampleRate, bufferSize;
    long readSampleCursor;
    long writeSampleCursor;
    long nextUnderrunLogSample;
    readonly object bufferLock = new object();
    ComputeBuffer ringBufferGPU;

    const string WaveformKernelName = "DrawRingBufferWaveform";

    public bool paused = false;

    // Recording
    bool isRecording;
    List<float> recordingBuffer;

    public int SampleRate => sampleRate;
    public int WriteHead => writeHead;
    public int ReadHead { get { lock (bufferLock) return readHead; } }
    public bool IsRecording => isRecording;

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

    void Start()
    {
        if (shader == null && rayTraceManager != null)
            shader = rayTraceManager.raytraceShader;

    }

    void EnsureLegacyBuffer()
    {
        int size=checked(sampleRate*16);
        if (ringBuffer==null || ringBuffer.Length!=size) ringBuffer=new float[size];
        bufferSize=size;
    }

    public void StartStreaming(int audioSampleRate)
    {
        Debug.Log($"AudioManager starting streaming with sample rate {audioSampleRate}.");
        if (OutputCapture!=null || OutputQueue!=null) throw new InvalidOperationException("Stop the direct output session before starting legacy playback.");
        sampleRate = audioSampleRate;
        EnsureLegacyBuffer(); pcmMode=false;
        lock (bufferLock) 
        { 
            readHead = 0;
            writeHead = 0;
            readSampleCursor = 0;
            writeSampleCursor = 0;
            underrunCount_Debug = 0;
            System.Array.Clear(ringBuffer, 0, ringBuffer.Length); 
        }
    }

    /// <summary>
    /// Add audio chunk to ring buffer just ahead of read position for immediate playback
    /// </summary>
    public (long, int) QueueAudioChunk(float[] audio, int audioStartPos=0, long? forcedWritePos = null)
    {
        if (audio == null || audio.Length == 0)
            return (writeSampleCursor, audioStartPos);

        EnsureLegacyBuffer();
        lock (bufferLock)
        {
            long safeStart = readSampleCursor + safetyBuffer;
            long writePos = forcedWritePos ?? safeStart;

            if (writePos < safeStart)
            {
                long skippedSamples = safeStart - writePos;
                audioStartPos += skippedSamples > int.MaxValue ? int.MaxValue : (int)skippedSamples;
                writePos = safeStart;
                LogUnderrun($"Audio computation is behind playback by {skippedSamples} samples; skipping ahead to avoid writing in the past.");
            }

            audioStartPos = Mathf.Clamp(audioStartPos, 0, audio.Length);
            int samplesToWrite = audio.Length - audioStartPos;
            if (samplesToWrite <= 0)
                return (writePos, audioStartPos);

            long maxWritableEnd = readSampleCursor + bufferSize;
            if (writePos + samplesToWrite > maxWritableEnd)
            {
                samplesToWrite = Mathf.Max(0, (int)(maxWritableEnd - writePos));
                Debug.LogWarning($"Audio queue is more than one ring buffer ahead; truncating chunk to {samplesToWrite} samples.");
            }

            for (int i = 0; i < samplesToWrite; i++)
            {
                int idx = Mod(writePos + i, bufferSize);
                ringBuffer[idx] += audio[i + audioStartPos];
            }

            writeSampleCursor = System.Math.Max(writeSampleCursor, writePos + samplesToWrite);
            writeHead = Mod(writeSampleCursor, bufferSize);
            return (writePos, audioStartPos);
        }
    }

    static int Mod(long value, int modulus) => (int)((value % modulus + modulus) % modulus);

    void OnAudioFilterRead(float[] data, int channels)
    {
        if (pcmMode)
        {
            var queue=Volatile.Read(ref pcmQueue);
            if (queue==null) Array.Clear(data,0,data.Length);
            else queue.Consume(data,channels,paused,AudioSettings.dspTime);
            return;
        }
        if (paused || ringBuffer==null) { Array.Clear(data,0,data.Length); return; }
        lock (bufferLock)
        {
            int samplesPerChannel = data.Length / channels;
            for (int i = 0; i < samplesPerChannel; i++)
            {
                if (readSampleCursor >= writeSampleCursor)
                    LogUnderrun($"Audio read cursor passed write cursor at sample {readSampleCursor}. Computation is too slow or audio has not been queued yet.");

                float s = ringBuffer[readHead];
                ringBuffer[readHead] = 0; // Clear after reading
                readHead++;
                if (readHead >= bufferSize) readHead -= bufferSize;
                readSampleCursor++;
                for (int c = 0; c < channels; c++)
                    data[i * channels + c] = s;

                if (isRecording)
                    recordingBuffer.Add(s);
            }
        }
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
        bufferCapacity = bufferSize;
        readHead_Debug = readHead;
        readSampleCursor_Debug = readSampleCursor;
        writeSampleCursor_Debug = writeSampleCursor;
        readLoopCount_Debug = bufferSize > 0 ? (int)(readSampleCursor / bufferSize) : 0;
        writeLoopCount_Debug = bufferSize > 0 ? (int)(writeSampleCursor / bufferSize) : 0;

        if (Input.GetKeyDown(KeyCode.R))
            ToggleRecording();

        if (Input.GetKeyDown(KeyCode.P))
            TogglePause();

        if (showWaveform && shader != null)
            DrawCyclingWaveform();
    }

    public void TogglePause()
    {
        paused = !paused;
        Debug.Log(paused ? "Audio paused." : "Audio resumed.");
    }

    void LogUnderrun(string message)
    {
        underrunCount_Debug++;
        if (readSampleCursor < nextUnderrunLogSample)
            return;

        nextUnderrunLogSample = readSampleCursor + sampleRate / 2;
        Debug.LogWarning(message);
    }

    public void ToggleRecording()
    {
        if (isRecording)
        {
            StopRecordingAndSave();
            return;
        }

        recordingBuffer = new List<float>();
        isRecording = true;
        Debug.Log("Recording started.");
    }

    public void StopRecordingAndSave()
    {
        if (!isRecording) return;

        isRecording = false;
        Debug.Log($"Recording stopped. {recordingBuffer.Count} samples captured.");
        SaveRecordingToWav();
    }

    void SaveRecordingToWav()
    {
        if (recordingBuffer == null || recordingBuffer.Count == 0)
        {
            Debug.LogWarning("No recording data to save.");
            return;
        }

        float[] samples = recordingBuffer.ToArray();
        string projectRoot = Directory.GetParent(Application.dataPath).FullName;
        string recordingDirectory = Path.Combine(projectRoot, "Recordings");
        Directory.CreateDirectory(recordingDirectory);
        string path = Path.Combine(recordingDirectory, $"recording_{System.DateTime.Now:yyyyMMdd_HHmmss}.wav");

        using (var fs = new FileStream(path, FileMode.Create))
        using (var writer = new BinaryWriter(fs))
        {
            int channels = 1;
            int bitsPerSample = 16;
            int byteRate = sampleRate * channels * bitsPerSample / 8;
            int blockAlign = channels * bitsPerSample / 8;
            int dataSize = samples.Length * blockAlign;

            // RIFF header
            writer.Write(System.Text.Encoding.ASCII.GetBytes("RIFF"));
            writer.Write(36 + dataSize);
            writer.Write(System.Text.Encoding.ASCII.GetBytes("WAVE"));

            // fmt chunk
            writer.Write(System.Text.Encoding.ASCII.GetBytes("fmt "));
            writer.Write(16);
            writer.Write((short)1); // PCM
            writer.Write((short)channels);
            writer.Write(sampleRate);
            writer.Write(byteRate);
            writer.Write((short)blockAlign);
            writer.Write((short)bitsPerSample);

            // data chunk
            writer.Write(System.Text.Encoding.ASCII.GetBytes("data"));
            writer.Write(dataSize);

            for (int i = 0; i < samples.Length; i++)
            {
                float clamped = Mathf.Clamp(samples[i], -1f, 1f);
                writer.Write((short)(clamped * 32767f));
            }
        }

        Debug.Log($"Recording saved to {path}");
    }
    
    void DrawCyclingWaveform()
    {
        if (ringBuffer==null) return;
        if (shader == null)
            return;
        if (!shader.HasKernel(WaveformKernelName))
        {
            Debug.LogWarning($"Waveform kernel '{WaveformKernelName}' not found on shader '{shader.name}'.");
            return;
        }

        // Lazy initialize waveform texture
        if (waveformTexture == null || waveformTexture.width != (int)waveformTextureSize.x || waveformTexture.height != (int)waveformTextureSize.y)
        {
            if (waveformTexture != null)
                waveformTexture.Release();
            
            waveformTexture = new RenderTexture((int)waveformTextureSize.x, (int)waveformTextureSize.y, 0);
            waveformTexture.enableRandomWrite = true;
            waveformTexture.filterMode = FilterMode.Point;
            waveformTexture.Create();
            
            // Clear to visible color
            RenderTexture.active = waveformTexture;
            UnityEngine.GL.Clear(true, true, new Color(0.05f, 0.05f, 0.1f, 1));
            RenderTexture.active = null;
        }
        
        // Create or update GPU buffer
        if (ringBufferGPU == null || ringBufferGPU.count != bufferSize)
        {
            ringBufferGPU?.Release();
            ringBufferGPU = new ComputeBuffer(bufferSize, sizeof(float));
        }
        
        // Copy ring buffer to GPU
        lock (bufferLock)
        {
            ringBufferGPU.SetData(ringBuffer);
        }
        
        // Dispatch compute shader
        int kernel = shader.FindKernel(WaveformKernelName);
        shader.SetBuffer(kernel, "RingBufferData", ringBufferGPU);
        shader.SetInt("RingBufferSize", bufferSize);
        shader.SetInt("RingBufferReadHead", readHead);
        shader.SetInt("RingBufferWriteHead", writeHead);
        shader.SetTexture(kernel, "DebugTexture", waveformTexture);
        shader.SetInt("TexWidth", waveformTexture.width);
        shader.SetInt("TexHeight", waveformTexture.height);
        shader.SetFloat("WaveformGain", waveformGain);
        
        ComputeHelper.Dispatch(shader, waveformTexture.width, waveformTexture.height, 1, kernel);
    }
    
    void OnGUI()
    {
        if (showWaveform && waveformTexture != null)
        {
            float w = Screen.width * 0.4f;
            float h = Screen.height * 0.15f;

            // Position below RayTraceManager's two textures using the same on-screen size
            float yOffset = 10;
            if (rayTraceManager != null && rayTraceManager.showDebugTexture)
            {
                yOffset = 30 + h * 2;
            }
            GUI.DrawTexture(new Rect(10, yOffset, w, h), waveformTexture);

        }
    }
    
    void OnDestroy()
    {
        StopPcmStream();
        if (driverSource!=null) driverSource.Stop();
        if (driverClip!=null) { if (Application.isPlaying) Destroy(driverClip); else DestroyImmediate(driverClip); }
        ringBufferGPU?.Release();
        if (waveformTexture != null)
        {
            waveformTexture.Release();
            waveformTexture = null;
        }
    }
}
