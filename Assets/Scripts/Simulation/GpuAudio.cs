using System;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Rendering;

// Owns GPU DSP and its asynchronous PCM readback. Scene/timeline contracts are
// shared with the CPU reference, but no path or intermediate spectrum is read back.
internal sealed class GpuAudio : IDisposable
{
    readonly ComputeShader shader;
    readonly CommandBuffer commands;
    readonly TakeRender.Settings settings;
    readonly TakeRender.Signal[] signals;
    readonly int pairs,listeners,channels,capacity,fft,bins,history,maximumBlock;
    readonly int reset,compare,commit,groups,filter,input,inverse,write,materialGlobal,padGlobal,filterGlobal,inputGlobal,sumGlobal,saveGlobal,reverse,butterfly;
    readonly int[] kernels,transformKernels;
    readonly List<ComputeBuffer> owned=new List<ComputeBuffer>();
    readonly ComputeBuffer args,headsPose,source,work,transform,output;
    readonly float[] download;
    readonly Vector4[] poses;
    readonly Quaternion[] lastHeads;
    readonly bool[] lastEnabled,audibleSources;
    readonly Action<AsyncGPUReadbackRequest> completion;
    AsyncGPUReadbackRequest request;
    bool initialized,pending,completed,failed,disposed;
    long cursor,uploaded,responseSample,readbackStarted,readbackCompleted;
    int requested,queued,poseCursor;
    uint previousResponses,previousReused;
    const int Header=8;
    public TakeRender.AudioDiagnostics Diagnostics {get;}=new TakeRender.AudioDiagnostics();
    public long NumericBytes {get;private set;}
    public int ResponseSupportSamples {get;private set;}
    public bool IsPending=>pending;
    public long UploadedBytes {get;private set;}
    public long ReadbackBytes {get;private set;}
    public static bool Selected(TakeRender.Settings settings,ComputeShader transport) =>
        transport!=null && settings.transport==TakeRender.Transport.Raytraced && settings.audioBackend==TakeRender.AudioBackend.Gpu;
    public GpuAudio(AcousticTake.Scene scene,int rate,TakeRender.Settings settings,TakeRender.Signal[] signals,int maximumResponses=1)
    {
        audibleSources=new bool[signals.Length];
        for(int s=0;s<signals.Length;s++)audibleSources[s]=scene.listeners.Length>1 || scene.sources[s].mutedListenerPlusOne!=1;
        this.settings=settings;this.signals=signals;listeners=scene.listeners.Length;channels=settings.ChannelsPerListener;pairs=signals.Length*listeners;
        capacity=checked(pairs*settings.SpectralSlotsPerPair);maximumBlock=checked(settings.responseStepSamples*maximumResponses);
        var hrtf=channels==2?Hrtf.Load(rate):null;int taps=hrtf?.Taps??1;
        fft=1;while(fft<settings.fftSize+taps+settings.spectralHop-2)fft=checked(fft*2);bins=fft/2+1;
        history=checked((int)settings.HistorySamples(rate)+fft+maximumBlock);
        download=new float[checked(maximumBlock*listeners*channels+Header)];
        poses=new Vector4[listeners*2];lastHeads=new Quaternion[listeners];lastEnabled=new bool[listeners];completion=Complete;
        var asset=Resources.Load<ComputeShader>("Acoustics/GpuAudio");
        if(asset==null)throw new InvalidOperationException("Missing GPU audio compute shader.");
        try
        {
            commands=new CommandBuffer {name="Acoustics GPU audio"};
            shader=UnityEngine.Object.Instantiate(asset);
            reset=Kernel("AudioReset");compare=Kernel("AudioCompare");commit=Kernel("AudioCommit");groups=Kernel("AudioGroups");filter=Kernel("AudioFilter");input=Kernel("AudioInput");inverse=Kernel("AudioInverse");write=Kernel("AudioWrite");
            materialGlobal=Kernel("AudioMaterialGlobal");padGlobal=Kernel("AudioPadGlobal");filterGlobal=Kernel("AudioFilterGlobal");inputGlobal=Kernel("AudioInputGlobal");sumGlobal=Kernel("AudioSumGlobal");saveGlobal=Kernel("AudioSaveGlobal");reverse=Kernel("AudioBitReverse");butterfly=Kernel("AudioButterfly");
            kernels=new[]{reset,compare,commit,groups,filter,input,inverse,write,materialGlobal,padGlobal,filterGlobal,inputGlobal,sumGlobal,saveGlobal,reverse,butterfly};
            transformKernels=new[]{materialGlobal,padGlobal,sumGlobal,saveGlobal,reverse,butterfly};
            int hash=checked(settings.SpectralSlotsPerPair*2+1);
            Buffer("AudioMeta",16,4).SetData(new uint[16]);
            Buffer("AudioLast",capacity,32).SetData(new RayTransport.Path[capacity]);
            Buffer("AudioPaths",checked(capacity*2),32);Buffer("AudioCounts",pairs*2,4).SetData(new uint[pairs*2]);
            args=Buffer("AudioArgs",pairs*6,4,ComputeBufferType.IndirectArguments|ComputeBufferType.Structured);args.SetData(new uint[pairs*6]);
            Buffer("AudioTable",checked(pairs*hash),8);Buffer("AudioHeads",capacity,4);Buffer("AudioNext",capacity,4);Buffer("AudioBlends",checked(2*capacity),32);
            headsPose=Buffer("AudioHeadsPose",checked(poses.Length*maximumResponses),16);
            source=Buffer("AudioSource",checked(signals.Length*history),4,mode:ComputeBufferMode.SubUpdates);
            // Material spectra are shared by both ears; retain each bank's head blend for crossfades.
            Buffer("AudioFilters",checked(2*capacity*bins),8);
            Diagnostics.pathCapacity=capacity;Diagnostics.filterBytes=2L*capacity*bins*8;
            // Shared FFTs only write their nonredundant bins. Global filter preparation
            // runs pair by pair, so its separate transform scratch can reuse the same rows.
            int workStride=fft<=2048?bins:fft;
            work=Buffer("AudioWork",checked(capacity*workStride),8);
            transform=fft>2048?Buffer("AudioTransform",checked(Math.Max(settings.SpectralSlotsPerPair,listeners*channels)*fft),8):work;
            Buffer("AudioBlocks",checked(2*listeners*channels*settings.spectralHop),4).SetData(new float[2*listeners*channels*settings.spectralHop]);
            output=Buffer("AudioOutput",download.Length,4);
            var pairMask=new float[pairs];for(int p=0;p<pairs;p++)pairMask[p]=scene.sources[p/listeners].mutedListenerPlusOne==p%listeners+1?0:1;
            Buffer("AudioPairEnabled",pairs,4).SetData(pairMask);
            PrepareEars(hrtf);
            Set("AudioPairs",pairs);Set("AudioListeners",listeners);Set("AudioChannels",channels);Set("AudioSlots",settings.SpectralSlotsPerPair);Set("AudioCapacity",capacity);Set("AudioHashSize",hash);
            Set("AudioSize",settings.fftSize);Set("AudioFft",fft);Set("AudioBins",bins);Set("AudioHop",settings.spectralHop);Set("AudioRate",rate);Set("AudioHrir",taps);Set("AudioHistory",history);
            Set("AudioWorkStride",workStride);
            Set("AudioTransition",settings.transitionSamples);Set("AudioSpectral",settings.synthesis==TakeRender.Synthesis.Spectral?1:0);
            Set("AudioRadius",settings.sincRadius);Set("AudioSinc",settings.delayFilter==TakeRender.DelayFilter.WindowedSinc?1:0);
            Set("AudioLatency",settings.fftSize/2+(settings.delayFilter==TakeRender.DelayFilter.WindowedSinc?settings.sincRadius:0));shader.SetFloat("AudioMaximumDelay",settings.maximumDelaySeconds);
            NumericBytes+=(long)download.Length*4+poses.Length*16L+lastHeads.Length*16L+lastEnabled.Length+audibleSources.Length;
            Submit();
        }
        catch {Dispose();throw;}
    }
    int Kernel(string name) {int k=shader.FindKernel(name);if(!shader.IsSupported(k))throw new NotSupportedException("GPU audio kernel unavailable: "+name);return k;}
    ComputeBuffer Buffer(string name,int count,int stride,ComputeBufferType type=ComputeBufferType.Default,ComputeBufferMode mode=ComputeBufferMode.Immutable)
    {
        var buffer=new ComputeBuffer(count,stride,type,mode);owned.Add(buffer);NumericBytes+=(long)count*stride;
        foreach(int k in kernels)shader.SetBuffer(k,name,buffer);return buffer;
    }
    void Set(string name,int value)=>commands.SetComputeIntParam(shader,name,value);
    void Clock(string name,long sample)=>commands.SetComputeIntParams(shader,name,unchecked((int)sample),unchecked((int)(sample>>32)));
    void Submit(){Graphics.ExecuteCommandBuffer(commands);commands.Clear();}
    void PrepareEars(Hrtf hrtf)
    {
        int directions=hrtf?.DirectionCount??1;var spectra=new Vector2[checked(directions*channels*bins)];var data=new Vector2[fft];
        for(int direction=0;direction<directions;direction++)for(int ear=0;ear<channels;ear++)
        {
            Array.Clear(data,0,fft);for(int t=0;t<(hrtf?.Taps??1);t++)data[t].x=hrtf==null?1:hrtf.Coefficient(direction,ear,t);
            ImpulseResponseAudio.Fft(data,false);Array.Copy(data,0,spectra,(direction*channels+ear)*bins,bins);
        }
        Buffer("AudioEars",spectra.Length,8).SetData(spectra);
        var rings=new SortedDictionary<float,List<Vector2>>();
        for(int i=0;i<directions;i++)
        {
            Vector3 d=hrtf==null?Vector3.forward:hrtf.Direction(i);float elevation=Mathf.Asin(Mathf.Clamp(d.y,-1,1));
            if(!rings.TryGetValue(elevation,out var ring)){ring=new List<Vector2>();rings.Add(elevation,ring);}ring.Add(new Vector2(Mathf.Atan2(d.x,d.z),i));
        }
        var rows=new Vector4[rings.Count];var angles=new Vector2[directions];int row=0,offset=0;
        foreach(var item in rings){item.Value.Sort((a,b)=>a.x.CompareTo(b.x));rows[row++]=new Vector4(item.Key,offset,item.Value.Count,0);foreach(var angle in item.Value)angles[offset++]=angle;}
        Buffer("AudioRings",rows.Length,16).SetData(rows);Buffer("AudioAngles",angles.Length,8).SetData(angles);Set("AudioRingCount",rows.Length);
    }
    public void SetResponse(long sample,bool cut,ComputeBuffer arrivals,int offset,AcousticTake.Scene scene,AcousticTake.State[] states)
    {
        if(disposed || pending || sample!=cursor)throw new InvalidOperationException("GPU response must match the owned audio cursor.");
        bool changed=false;
        foreach(var state in states)cut|=(state.flags&AcousticTake.StateFlags.Cut)!=0;
        for(int l=0;l<listeners;l++)
        {
            var state=states[scene.listeners[l]];bool enabled=TakeRender.Enabled(state,AcousticTake.StateFlags.Listener);
            changed|=!lastHeads[l].Equals(state.rotation) || lastEnabled[l]!=enabled;lastHeads[l]=state.rotation;lastEnabled[l]=enabled;
            poses[l]=new Vector4(state.rotation.x,state.rotation.y,state.rotation.z,state.rotation.w);poses[listeners+l]=new Vector4(enabled?1:0,0,0,0);
        }
        headsPose.SetData(poses,0,poseCursor*poses.Length,poses.Length);UploadedBytes+=poses.Length*16;
        Set("AudioPoseOffset",poseCursor*poses.Length);poseCursor++;
        foreach(int k in kernels)commands.SetComputeBufferParam(shader,k,"AudioArrivals",arrivals);
        Set("AudioOffset",offset);Set("AudioChanged",changed?1:0);Set("AudioInitialized",initialized?1:0);Set("AudioCut",cut?1:0);Clock("AudioResponseSample",sample);
        commands.BeginSample("Acoustics.GPU.Compare");
        commands.DispatchCompute(shader,reset,1,1,1);commands.DispatchCompute(shader,compare,(capacity+63)/64,1,1);commands.DispatchCompute(shader,commit,1,1,1);
        commands.EndSample("Acoustics.GPU.Compare");
        commands.BeginSample("Acoustics.GPU.Group");commands.DispatchCompute(shader,groups,pairs,1,1);commands.EndSample("Acoustics.GPU.Group");
        commands.BeginSample("Acoustics.GPU.Filters");
        Set("AudioWhich",0);Set("AudioBuildingFilter",1);
        for(int pair=0;pair<pairs;pair++)
        {
            Set("AudioPair",pair);
            if(fft<=2048)Dispatch(filter,pair,0);
            else
            {
                BindTransform(transform);Set("AudioTransformSize",settings.fftSize);Dispatch(materialGlobal,pair,0);
                FftGlobal(transform,settings.fftSize,true,pair,0);BindTransform(transform);Dispatch(padGlobal,pair,0);
                FftGlobal(work,fft,false,pair,0);Dispatch(filterGlobal,pair,0);
            }
        }
        commands.EndSample("Acoustics.GPU.Filters");Submit();initialized=true;responseSample=sample;
    }
    void Dispatch(int kernel,int pair,int which)=>commands.DispatchCompute(shader,kernel,args,(uint)((which*pairs+pair)*3*4));
    void BindTransform(ComputeBuffer buffer)
    {
        foreach(int k in transformKernels)commands.SetComputeBufferParam(shader,k,"AudioTransform",buffer);
    }
    void FftGlobal(ComputeBuffer buffer,int size,bool inverse,int pair,int which,int count=0)
    {
        BindTransform(buffer);Set("AudioTransformSize",size);Set("AudioTransformInverse",inverse?1:0);
        Set("AudioTransformBase",buffer==work && count==0?pair*settings.SpectralSlotsPerPair:0);
        if(count==0)Dispatch(reverse,pair,which);else commands.DispatchCompute(shader,reverse,count,1,1);
        for(int stage=2;;stage=checked(stage*2))
        {Set("AudioStage",stage);if(count==0)Dispatch(butterfly,pair,which);else commands.DispatchCompute(shader,butterfly,count,1,1);if(stage==size)break;}
    }
    public void PrepareInput(long end)
    {
        if(disposed || pending || queued!=0 || end<cursor || end-cursor>maximumBlock)throw new InvalidOperationException("Source uploads require a completed previous batch and a bounded input range.");
        for(int s=0;s<signals.Length;s++)if(audibleSources[s] && signals[s].ReadyEnd<end)throw new InvalidOperationException("GPU source input is not finalized.");
        while(uploaded<end)
        {
            int ring=(int)(uploaded%history),count=(int)Math.Min(Math.Min(end-uploaded,history-ring),maximumBlock);
            for(int s=0;s<signals.Length;s++)
            {
                // The previous PCM readback fences all source readers. Write this batch
                // before dispatch, retaining history + the entire pending batch in the ring.
                var mapped=source.BeginWrite<float>(s*history+ring,count);
                try {for(int n=0;n<count;n++)mapped[n]=audibleSources[s]?signals[s].Read(uploaded+n):0;}
                finally {source.EndWrite<float>(count);}
                UploadedBytes+=count*4L;
            }
            uploaded+=count;
        }
    }
    public void QueueRender(long sample,int count)
    {
        if(disposed || pending || sample!=cursor || count<1 || count>maximumBlock-queued)throw new InvalidOperationException("Invalid GPU audio submission.");
        if(!initialized)throw new InvalidOperationException("GPU audio needs a response before rendering.");
        // Uploaded samples were finalized by PrepareInput and remain owned for this batch.
        if(uploaded<checked(sample+count))PrepareInput(sample+count);
        Set("AudioBuildingFilter",0);
        for(int at=0;at<count;)
        {
            long now=sample+at,block=now-now%settings.spectralHop;Clock("AudioBlockSample",block);
            if(now%settings.spectralHop==0)
            {
                Set("AudioRingAt",(int)(block%history));Set("AudioBlockNegative",block<=int.MaxValue?1:0);
                Set("AudioAvailableStart",checked((int)(Math.Max(0,uploaded-history)-block)));Set("AudioAvailableEnd",checked((int)(sample+count-block)));
                for(int which=0;which<2;which++)
                {
                    // No sample in this hop can use the previous response after its fade ends.
                    if(which!=0 && block-responseSample>=settings.transitionSamples)continue;
                    Set("AudioWhich",which);
                    commands.BeginSample("Acoustics.GPU.InputFFT");
                    for(int pair=0;pair<pairs;pair++)
                    {
                        Set("AudioPair",pair);Dispatch(fft<=2048?input:inputGlobal,pair,which);
                        if(fft>2048)FftGlobal(work,fft,false,pair,which);
                    }
                    commands.EndSample("Acoustics.GPU.InputFFT");commands.BeginSample("Acoustics.GPU.Mix");
                    if(fft<=2048)commands.DispatchCompute(shader,inverse,listeners*channels,1,1);
                    else {BindTransform(transform);commands.DispatchCompute(shader,sumGlobal,listeners*channels,1,1);FftGlobal(transform,fft,true,0,which,listeners*channels);commands.DispatchCompute(shader,saveGlobal,listeners*channels,1,1);}
                    commands.EndSample("Acoustics.GPU.Mix");
                }
            }
            int first=(int)(now%settings.spectralHop),length=Math.Min(count-at,settings.spectralHop-first);
            Set("AudioWriteOffset",queued+at);Set("AudioWriteAt",first);Set("AudioWriteCount",length);
            commands.BeginSample("Acoustics.GPU.PCM");commands.DispatchCompute(shader,write,(length*listeners*channels+63)/64,1,1);commands.EndSample("Acoustics.GPU.PCM");at+=length;
        }
        Submit();cursor+=count;queued+=count;
    }
    public void BeginReadback()
    {
        if(disposed || pending || queued==0)throw new InvalidOperationException("No queued GPU audio to read back.");
        requested=queued;queued=0;poseCursor=0;
        completed=false;failed=false;int bytes=checked((requested*listeners*channels+Header)*4);ReadbackBytes+=bytes;
        readbackStarted=System.Diagnostics.Stopwatch.GetTimestamp();
        if(SystemInfo.supportsAsyncGPUReadback)request=AsyncGPUReadback.Request(output,bytes,0,completion);
        else {output.GetData(download,0,0,bytes/4);readbackCompleted=System.Diagnostics.Stopwatch.GetTimestamp();completed=true;}
        pending=true;
    }
    void Complete(AsyncGPUReadbackRequest value)
    {
        failed=value.hasError;if(!failed){var data=value.GetData<float>();for(int i=0;i<data.Length;i++)download[i]=data[i];}
        readbackCompleted=System.Diagnostics.Stopwatch.GetTimestamp();completed=true;
    }
    public bool TryComplete(float[] pcm,bool wait=false)
    {
        if(!pending)throw new InvalidOperationException("No GPU audio submission is owned.");
        if(!completed){if(wait)request.WaitForCompletion();else request.Update();}if(!completed)return false;
        pending=false;if(failed)throw new InvalidOperationException("GPU audio readback failed.");
        double latency=(double)(readbackCompleted-readbackStarted)/System.Diagnostics.Stopwatch.Frequency;
        Diagnostics.readbacks++;Diagnostics.readbackSeconds+=latency;Diagnostics.maximumReadbackSeconds=Math.Max(Diagnostics.maximumReadbackSeconds,latency);
        Diagnostics.maximumConsumeDelaySeconds=Math.Max(Diagnostics.maximumConsumeDelaySeconds,TakeRender.Elapsed(readbackCompleted));
        if(download[0]!=0)throw new InvalidOperationException("GPU audio validation failed ("+download[0]+"): 1 invalid path, 2 missing source history, 3 non-finite audio, 4 invalid path grouping.");
        ResponseSupportSamples=(int)download[1];int length=requested*listeners*channels;
        uint responses=unchecked((uint)BitConverter.SingleToInt32Bits(download[2])),reused=unchecked((uint)BitConverter.SingleToInt32Bits(download[3]));
        Diagnostics.responseUpdates+=unchecked(responses-previousResponses);Diagnostics.reusedResponses+=unchecked(reused-previousReused);
        previousResponses=responses;previousReused=reused;
        Diagnostics.activeGroups=BitConverter.SingleToInt32Bits(download[4]);Diagnostics.peakGroups=BitConverter.SingleToInt32Bits(download[5]);
        for(int i=0;i<length;i++){float value=download[Header+i];if(!float.IsFinite(value))throw new InvalidOperationException("Non-finite GPU audio.");pcm[i]=value;}
        return true;
    }
    public void Render(long sample,int count,float[] pcm) {QueueRender(sample,count);BeginReadback();TryComplete(pcm,true);}
    public void Dispose()
    {
        if(disposed)return;if(pending && !completed)request.WaitForCompletion();pending=false;disposed=true;
        foreach(var buffer in owned)buffer.Dispose();owned.Clear();
        commands?.Dispose();
        if(shader!=null){if(Application.isPlaying)UnityEngine.Object.Destroy(shader);else UnityEngine.Object.DestroyImmediate(shader);}
    }
}
