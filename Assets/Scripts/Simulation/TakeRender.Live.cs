using System;
using System.Diagnostics;
using System.Collections.Generic;
using UnityEngine;

public static partial class TakeRender
{
    // Control-thread producer. Every queued interval owns its poses until its PCM is rendered.
    public sealed class Live : IDisposable
    {
        sealed class Observation
        {
            public long from,to,end;
            public AcousticTake.State[] left,right;
            public Action<long,float[],int> output;
        }
        readonly AcousticTake.Scene scene;
        readonly Settings settings;
        readonly int sampleRate,listener,lead,lookahead;
        readonly bool bursts;
        readonly long historySamples;
        readonly TransportSolver solver;
        readonly Mixer mixer;
        readonly GpuAudio gpuAudio;
        readonly Signal[] signals;
        readonly AcousticTake.State[] evaluated,batchStates,headStates;
        readonly DirectPath[] paths,batchPaths;
        readonly float[] block,gpuBlock;
        readonly AudioManager.PcmQueue queue;
        Observation[] observations;
        int head,pending,retained,batchCount;
        long batchFirst,batchEnd,previousResponse=-1;
        bool batchActive,batchReady,audioQueued,audioReady,batchDiagnostics,prepared;
        readonly List<long> cuts=new List<long>();
        public long Cursor {get;private set;}
        public double SolveSeconds,AudioSeconds,ResponsePreparationSeconds,MaximumSolveSeconds;
        public int ResponseCount,MaximumPendingObservations;
        public int PendingObservations => pending;
        public bool CanObserve => pending<settings.maximumPendingObservations;
        public int RetainedEvents { get { int count=cuts.Count; foreach (var signal in signals) count+=signal.RetainedEvents; return count; } }
        public double CompletionSeconds => solver.CompletionSeconds;
        public double MaximumCompletionSeconds => solver.MaximumCompletionSeconds;
        public long PairPrimitiveUpperBound => solver.PairPrimitiveUpperBound;
        public long ResponseSupportSamples => gpuAudio?.ResponseSupportSamples??mixer.ResponseSupportSamples;
        public bool TailCoversResponseSupport => settings.tailSamples>=ResponseSupportSamples;
        public Action<long,long,DirectPath[]> ResponseApplied;
        public string Backend => solver.Backend;
        public AudioDiagnostics GpuDiagnostics => gpuAudio?.Diagnostics;
        // Live path method: the persistent tracer (for diagnostics) and the denoiser switch (GPU audio).
        internal MethodTransport Method => solver.Method;
        public bool HasPathDenoiser => gpuAudio?.HasPathDenoiser??false;
        public bool PathDenoising {get=>gpuAudio!=null && gpuAudio.PathDenoising;set{if(gpuAudio!=null)gpuAudio.PathDenoising=value;}}
        // Pipeline trace (diagnostics): with -liveTrace FILE on the command line, one record per pump and per batch
        // event (dispatch with its CPU time, GPU audio ready with its latency, the pump's CPU time), written as CSV when the renderer is disposed.
        readonly LiveTrace trace=LiveTrace.FromCommandLine();
        public Live(AcousticTake.Scene scene,AcousticTake.Primitive[] geometry,float[][] pcm,int rate,Settings settings,
            AudioManager.PcmQueue queue,int lead,int listener,ComputeShader shader=null,SourceStream[] streams=null)
        {
            Validate(settings);bursts=Emission.HasBursts(scene);historySamples=settings.HistorySamples(rate);
            if(bursts)foreach(var p in scene.planes)historySamples=Math.Max(historySamples,settings.HistorySamples(rate)+p.burstDurationSamples+2L*settings.responseStepSamples);
            lookahead=(int)Math.Max(settings.ResponseSample(0),bursts?settings.responseStepSamples:0);
            if (listener<0 || listener>=scene.listeners.Length) throw new ArgumentOutOfRangeException(nameof(listener));
            this.scene=scene; sampleRate=rate; this.settings=settings; this.queue=queue; this.lead=lead; this.listener=listener;
            if(queue!=null && lead<lookahead)throw new ArgumentException("Playback lead must cover acoustic response lookahead.");
            if(queue!=null && queue.Channels!=settings.ChannelsPerListener) throw new ArgumentException("Queue/output channel mismatch.");
            signals=new Signal[scene.sources.Length];
            for (int i=0;i<signals.Length;i++) signals[i]=scene.sources[i].streamed?new Signal(streams[i],settings.maximumSignalEvents):new Signal(pcm[scene.sources[i].clipIndex],settings.maximumSignalEvents);
            evaluated=new AcousticTake.State[scene.objects.Length];headStates=settings.ChannelsPerListener==2?new AcousticTake.State[evaluated.Length]:null; paths=new DirectPath[scene.sources.Length*scene.listeners.Length];
            batchStates=new AcousticTake.State[checked(evaluated.Length*settings.maximumResponsesPerBatch)];
            batchPaths=new DirectPath[checked(paths.Length*settings.maximumResponsesPerBatch)];
            block=new float[checked(settings.responseStepSamples*scene.listeners.Length*settings.ChannelsPerListener)];
            observations=new Observation[settings.maximumPendingObservations];
            for (int i=0;i<observations.Length;i++) observations[i]=new Observation {left=new AcousticTake.State[evaluated.Length],right=new AcousticTake.State[evaluated.Length]};
            try
            {
                if(GpuAudio.Selected(settings,shader))
                {
                    gpuBlock=new float[checked(block.Length*settings.maximumResponsesPerBatch)];
                    gpuAudio=new GpuAudio(scene,rate,settings,signals,settings.maximumResponsesPerBatch,transported:true);
                }
                else mixer=new Mixer(scene,rate,settings,signals);
                solver=new TransportSolver(scene,geometry,settings,shader,settings.maximumResponsesPerBatch,rate,gpuAudio==null,methodAudio:true);
            }
            catch {gpuAudio?.Dispose();throw;}
        }
        public void Prepare(AcousticTake.State[] state)
        {
            if(gpuAudio==null)return;
            Array.Copy(state,batchStates,state.Length);solver.BeginSolve(batchStates,1,false);
            gpuAudio.SetResponse(0,true,solver.RayBuffer,0,scene,state,late:solver.Late(0));
            gpuAudio.CompletePreparation();solver.CompleteAfterGpuAudio();prepared=!bursts;
        }
        // D3D12 builds a kernel's pipeline on its first dispatch, which took 15.6 s for the path kernels on a cold
        // driver cache, while the output clock ran on and the take stopped. A throwaway renderer of the same scene and
        // settings runs every live kernel over `updates` updates of the first pose, before the clock starts; it leaves
        // the live renderer's state untouched. `diagnostics` selects the readback path the live renderer will use.
        public static void WarmUp(AcousticTake.Scene scene,AcousticTake.Primitive[] geometry,float[][] pcm,int rate,Settings settings,
            int listener,ComputeShader shader,AcousticTake.State[] state,bool diagnostics,int updates=16)
        {
            var streams=new SourceStream[scene.sources.Length];   // streamed sources read silence
            for(int i=0;i<streams.Length;i++)if(scene.sources[i].streamed){streams[i]=new SourceStream(1);streams[i].Finish();}
            using var warm=new Live(scene,geometry,pcm,rate,settings,null,0,listener,shader,streams);
            if(diagnostics)warm.ResponseApplied=(_,_,_)=>{};
            warm.Prepare(state);warm.Observe(0,(long)updates*settings.responseStepSamples,state,state);warm.Pump(true);
            if(warm.Cursor<(long)updates*settings.responseStepSamples)throw new InvalidOperationException("Kernel warm-up did not render.");
        }
        public void Append(AcousticTake.VoiceEvent e) { AudioSourcePlane.ValidateEvent(scene,e); signals[e.sourceIndex].Append(e); }
        public void Observe(long from,long to,AcousticTake.State[] left,AcousticTake.State[] right,Action<long,float[],int> output=null)
        {
            // Include the newest interval before dispatching the next GPU batch.
            // Otherwise every completed batch can launch with a frame of stale input.
            if (!CanObserve) Pump();
            foreach (var state in right) if ((state.flags&AcousticTake.StateFlags.Cut)!=0)
            { if (cuts.Count==0 || cuts[cuts.Count-1]!=to) cuts.Add(to); break; }
            for (int i=0;i<signals.Length;i++) signals[i].SetActive(to,Enabled(right[scene.sources[i].objectIndex],AcousticTake.StateFlags.Source));
            Enqueue(from,to,left,right,to,output); Pump();
        }
        void Enqueue(long from,long to,AcousticTake.State[] left,AcousticTake.State[] right,long end,Action<long,float[],int> output)
        {
            if (!CanObserve) throw new InvalidOperationException($"Live observation queue is full: rendered={Cursor}, observed={to}, sourceReady={ReadyEnd}, outputRead={queue?.ReadSample}, outputWritten={queue?.WrittenSample}, batchEnd={batchEnd}, GPU pending={audioQueued && !audioReady}. Reduce live compute cost.");
            if (from>to || to>end || left.Length!=evaluated.Length || right.Length!=evaluated.Length) throw new ArgumentException("Invalid observation.");
            if(pending+retained==observations.Length)
            {
                var expanded=new Observation[checked(observations.Length*2)];
                for(int i=0;i<retained+pending;i++)expanded[i]=observations[(head-retained+i+observations.Length)%observations.Length];
                for(int i=retained+pending;i<expanded.Length;i++)expanded[i]=new Observation {left=new AcousticTake.State[evaluated.Length],right=new AcousticTake.State[evaluated.Length]};
                observations=expanded;head=retained;
            }
            var frame=observations[(head+pending)%observations.Length];
            frame.from=from; frame.to=to; frame.end=end; frame.output=output;
            Array.Copy(left,frame.left,left.Length); Array.Copy(right,frame.right,right.Length);
            pending++; MaximumPendingObservations=Math.Max(MaximumPendingObservations,pending);
        }
        AcousticTake.State[] finalState;
        long finalSample;
        Action<long,float[],int> finalOutput;
        public bool Finishing {get;private set;}
        public bool Completed=>Finishing && finalState==null && pending==0;
        public void Finish(long sample,AcousticTake.State[] final,Action<long,float[],int> output=null,bool wait=true)
        {
            if(Finishing)throw new InvalidOperationException("The renderer is already finishing.");
            foreach(var signal in signals)signal.EndSample=sample;
            finalSample=sample;finalState=(AcousticTake.State[])final.Clone();finalOutput=output;Finishing=true;
            Pump(wait);
            if(wait && queue==null && !Completed)throw new InvalidOperationException("Renderer did not complete during shutdown.");
        }
        public void Pump(bool wait=false)
        {
            if(trace==null){PumpCore(wait);return;}
            long start=Stopwatch.GetTimestamp();
            trace.Add(LiveTrace.Kind.Pump,Cursor,queue,pending,batchFirst,batchEnd,batchCount,0);
            PumpCore(wait);
            trace.Add(LiveTrace.Kind.PumpEnd,Cursor,queue,pending,batchFirst,batchEnd,batchCount,Elapsed(start)*1000);
        }
        void PumpCore(bool wait)
        {
            while(pending>0 || finalState!=null)
            {
                if(pending==0)
                {
                    Enqueue(finalSample,finalSample,finalState,finalState,checked(finalSample+1+settings.tailSamples),finalOutput);
                    finalState=null;finalOutput=null;
                }
                var frame=observations[head];
                if (!RenderTo(frame,wait)) return;
                frame.output=null; head=(head+1)%observations.Length; pending--;retained++;
                long next=Cursor+(settings.responseStepSamples-Cursor%settings.responseStepSamples)%settings.responseStepSamples;
                long before=bursts?Cursor-historySamples:Math.Min(Cursor,settings.ResponseSample(next));
                while(retained>0 && observations[(head-retained+observations.Length)%observations.Length].end<=before)retained--;
                foreach (var signal in signals) signal.Prune(Cursor-historySamples);
                int last=cuts.BinarySearch(previousResponse); if (last<0) last=~last-1;
                if (last>0) cuts.RemoveRange(0,last);
            }
        }
        long ReadyEnd {get {long end=long.MaxValue;for(int s=0;s<signals.Length;s++)if(scene.listeners.Length>1 || scene.sources[s].mutedListenerPlusOne!=1)end=Math.Min(end,signals[s].ReadyEnd);return end;}}
        long WritableEnd=>queue==null?ReadyEnd:Math.Min(ReadyEnd,queue.ReadSample+queue.Capacity-lead);
        void EvaluateObserved(long sample,AcousticTake.State[] state)
        {
            int interval=-retained;
            while(interval+1<pending && sample>=observations[(head+interval+observations.Length)%observations.Length].end)interval++;
            var observed=observations[(head+interval+observations.Length)%observations.Length];
            if(sample>observed.end && finalState!=null){Array.Copy(finalState,state,state.Length);FreezeMotion(state);return;}
            AcousticTake.Interpolate(Math.Min(sample,observed.to),observed.from,observed.to,sampleRate,observed.left,observed.right,state);
            if(sample>observed.to)FreezeMotion(state);
        }
        void PrepareBatch()
        {
            var newest=observations[(head+pending-1)%observations.Length];
            long knownEnd=Math.Min(WritableEnd,newest.end);
            if(newest.end==newest.to && finalState==null)knownEnd=Math.Min(knownEnd,newest.end-lookahead);
            // A procedural response includes births through the following interval.
            // Wait for complete input intervals so future birth poses are measured.
            if(bursts && !Finishing)knownEnd=knownEnd/settings.responseStepSamples*settings.responseStepSamples;
            batchFirst=Cursor%settings.responseStepSamples==0?Cursor:Cursor+settings.responseStepSamples-Cursor%settings.responseStepSamples;
            batchEnd=Math.Min(knownEnd,checked((Cursor/settings.responseStepSamples+settings.maximumResponsesPerBatch)*settings.responseStepSamples));
            // Bound live submissions by playback lead, retaining batching when catching up.
            if(queue!=null)batchEnd=Math.Min(batchEnd,Cursor+Math.Max(settings.responseStepSamples,lead));
            // Path GPU audio reads half a partition ahead within a block (GpuAudio.InputEnd): stop at a partition
            // boundary until the source has the samples.
            if(gpuAudio!=null && gpuAudio.InputEnd(batchEnd)>ReadyEnd)batchEnd=Math.Max(Cursor,batchEnd/settings.LatePartition*settings.LatePartition);
            batchCount=batchEnd<=batchFirst?0:(int)(1+(batchEnd-1-batchFirst)/settings.responseStepSamples);
            if(batchEnd<=Cursor)return;
            if(!prepared && Cursor==0 && settings.ResponseSample(0)>0)
            {
                EvaluateObserved(0,evaluated);solver.Solve(evaluated,paths);
                if(gpuAudio!=null){gpuAudio.SetResponse(0,true,solver.RayBuffer,0,scene,evaluated,0,late:solver.Late(0));gpuAudio.CompletePreparation();}
                else mixer.SetResponse(0,paths,evaluated,true,solver.RayPaths,0,0,late:solver.Late(0));
                prepared=true;
            }
            for(int f=0;f<batchCount;f++)
            {
                long sample=settings.ResponseSample(batchFirst+(long)f*settings.responseStepSamples);
                EvaluateObserved(sample,evaluated);
                Array.Copy(evaluated,0,batchStates,f*evaluated.Length,evaluated.Length);
                int cut=cuts.BinarySearch(sample);if(cut<0)cut=~cut-1;
                if(cut>=0 && cuts[cut]>settings.ResponseSample(batchFirst+(long)(f-1)*settings.responseStepSamples))solver.FlushRays(f); // persistent rays: a cut empties them
            }
            if(batchCount>0)
            {
                batchDiagnostics=gpuAudio==null || ResponseApplied!=null;
                long start=Stopwatch.GetTimestamp();solver.BeginSolve(batchStates,batchCount,batchDiagnostics);
                double elapsed=Elapsed(start);SolveSeconds+=elapsed;MaximumSolveSeconds=Math.Max(MaximumSolveSeconds,elapsed);
            }
            batchActive=true;batchReady=batchCount==0;audioQueued=audioReady=false;
        }
        void ApplyResponse(long sample)
        {
            int index=(int)((sample-batchFirst)/settings.responseStepSamples);
            if(gpuAudio==null)Array.Copy(batchPaths,index*paths.Length,paths,0,paths.Length);
            Array.Copy(batchStates,index*evaluated.Length,evaluated,0,evaluated.Length);
            long anchor=settings.ResponseSample(sample);if(headStates!=null)EvaluateObserved(sample,headStates);
            int cut=cuts.BinarySearch(anchor);if(cut<0)cut=~cut-1;
            long start=Stopwatch.GetTimestamp();
            if(gpuAudio!=null)gpuAudio.SetResponse(sample,cut>=0 && cuts[cut]>previousResponse,solver.RayBuffer,index*solver.RayPathsPerFrame,scene,evaluated,anchor,headStates,solver.Late(index));
            else mixer.SetResponse(sample,paths,evaluated,cut>=0 && cuts[cut]>previousResponse,solver.RayPaths,index*solver.RayPathsPerFrame,anchor,headStates,solver.Late(index));
            previousResponse=anchor;
            double elapsed=Elapsed(start);AudioSeconds+=elapsed;ResponsePreparationSeconds+=elapsed;
            ResponseCount++;
            if(gpuAudio==null)ResponseApplied?.Invoke(sample+lead,queue==null?-1:queue.ReadSample,paths);
        }
        bool CompleteBatch(bool wait)
        {
            if(batchReady)return true;
            if (!batchDiagnostics)
            {
                if (!audioReady) return false;
                solver.CompleteAfterGpuAudio();
                batchReady = true;
                return true;
            }
            if(!solver.TryComplete(batchPaths,wait))return false;
            batchReady=true;
            if(gpuAudio!=null && ResponseApplied!=null)
                for(int f=0;f<batchCount;f++)
                {
                    Array.Copy(batchPaths,f*paths.Length,paths,0,paths.Length);
                    ResponseApplied(batchFirst+(long)f*settings.responseStepSamples+lead,queue==null?-1:queue.ReadSample,paths);
                }
            return true;
        }
        long audioFirst,dispatchedAt,preparedAt;
        bool RenderTo(Observation frame,bool wait)
        {
            while(Cursor<frame.end)
            {
                if(!batchActive) {if(WritableEnd<=Cursor)return false;preparedAt=Stopwatch.GetTimestamp();PrepareBatch();if(!batchActive)return false;}
                if(gpuAudio==null && !CompleteBatch(wait))return false;
                if(gpuAudio!=null && !audioReady)
                {
                    if(!audioQueued)
                    {
                        audioFirst=Cursor;gpuAudio.PrepareInput(gpuAudio.InputEnd(batchEnd));
                        // Transport and DSP stay ordered on the GPU; diagnostic
                        // readback is not a prerequisite for submitting audio.
                        // Submit every known interval together. A readback per observation
                        // would limit throughput to the GPU completion cadence.
                        for(long sample=Cursor;sample<batchEnd;)
                        {
                            if(sample%settings.responseStepSamples==0)ApplyResponse(sample);
                            int length=(int)Math.Min(batchEnd-sample,settings.responseStepSamples-sample%settings.responseStepSamples);
                            long start=Stopwatch.GetTimestamp();gpuAudio.QueueRender(sample,length);AudioSeconds+=Elapsed(start);sample+=length;
                        }
                        gpuAudio.BeginReadback();audioQueued=true;dispatchedAt=Stopwatch.GetTimestamp();
                        trace?.Add(LiveTrace.Kind.Dispatch,Cursor,queue,pending,batchFirst,batchEnd,batchCount,Elapsed(preparedAt)*1000);
                    }
                    long completion=Stopwatch.GetTimestamp();audioReady=gpuAudio.TryComplete(gpuBlock,wait);AudioSeconds+=Elapsed(completion);
                    if(!audioReady)return false;
                    trace?.Add(LiveTrace.Kind.Ready,Cursor,queue,pending,batchFirst,batchEnd,batchCount,Elapsed(dispatchedAt)*1000);
                }
                if(gpuAudio!=null && !CompleteBatch(wait))return false;
                if(gpuAudio==null && Cursor%settings.responseStepSamples==0)ApplyResponse(Cursor);
                int count=(int)Math.Min(Math.Min(frame.end,batchEnd)-Cursor,settings.responseStepSamples-Cursor%settings.responseStepSamples);
                long audioStart=Stopwatch.GetTimestamp();
                if(gpuAudio!=null)Array.Copy(gpuBlock,checked((int)(Cursor-audioFirst)*scene.listeners.Length*settings.ChannelsPerListener),block,0,count*scene.listeners.Length*settings.ChannelsPerListener);
                else mixer.Render(Cursor,count,block);
                AudioSeconds+=Elapsed(audioStart);
                queue?.Publish(Cursor+lead,block,count,scene.listeners.Length*settings.ChannelsPerListener,listener*settings.ChannelsPerListener);
                frame.output?.Invoke(Cursor,block,count);Cursor+=count;
                if(Cursor==batchEnd)batchActive=false;
            }
            return true;
        }
        public void Dispose() { gpuAudio?.Dispose();solver.Dispose();trace?.Write(); }
    }

    // The live renderer's pipeline over time (diagnostics; TakeRender.Live, -liveTrace FILE): records in a preallocated
    // array, so tracing allocates nothing while it runs; written as CSV at the end.
    internal sealed class LiveTrace
    {
        public enum Kind : byte { Pump, PumpEnd, Dispatch, Ready }
        struct Record { public long ticks,cursor,read,written,first,end; public int frame,pending,count,gc0,gc2; public float ms,main,present,render,gpu; public Kind kind; }
        readonly Record[] records=new Record[1<<17];
        readonly FrameTiming[] timing=new FrameTiming[1];
        readonly string file;
        readonly long origin=Stopwatch.GetTimestamp();
        int used;
        LiveTrace(string file){this.file=file;InstallLoopMarkers();}
        // The player loop's own time: a marker between every pair of its systems records a timestamp; a system that
        // takes longer than 40 ms in a frame is written to FILE.stalls (what the main thread did during a stall), with
        // what the main thread and the process did meanwhile: the main thread's CPU time (near the wall time: it was
        // computing; near zero: it was blocked or preempted), the process's page faults that needed I/O (paging), its
        // voluntary context switches (waits) and involuntary ones (preemption by other work), and its garbage collections.
        static readonly List<(string name,long ticks)> marks=new List<(string,long)>(512);
        struct Usage { public double cpu; public long major,voluntary,involuntary; public int gc; }
        static readonly List<(long ticks,int frame,string name,double ms,Usage used)> stalls=new List<(long,int,string,double,Usage)>();
        static Usage lastUsage;
        static bool installed,usageAvailable=true;
        [System.Runtime.InteropServices.StructLayout(System.Runtime.InteropServices.LayoutKind.Sequential)]
        struct TimeSpec { public long seconds,nanoseconds; }
        [System.Runtime.InteropServices.StructLayout(System.Runtime.InteropServices.LayoutKind.Sequential)]
        struct ResourceUsage { public long userSeconds,userMicroseconds,systemSeconds,systemMicroseconds,maxrss,ixrss,idrss,isrss,minflt,majflt,nswap,inblock,oublock,msgsnd,msgrcv,nsignals,nvcsw,nivcsw; }
        [System.Runtime.InteropServices.DllImport("/usr/lib/libSystem.B.dylib")] static extern int clock_gettime(int clock,out TimeSpec value);
        [System.Runtime.InteropServices.DllImport("/usr/lib/libSystem.B.dylib")] static extern int getrusage(int who,out ResourceUsage value);
        const int ThreadCpuClock=16,Self=0;   // macOS: CLOCK_THREAD_CPUTIME_ID, RUSAGE_SELF
        static Usage Now()
        {
            var u=new Usage{gc=GC.CollectionCount(0),cpu=double.NaN};
            if(!usageAvailable)return u;
            try
            {
                if(clock_gettime(ThreadCpuClock,out var t)==0)u.cpu=t.seconds*1e3+t.nanoseconds*1e-6;
                if(getrusage(Self,out var r)==0){u.major=r.majflt;u.voluntary=r.nvcsw;u.involuntary=r.nivcsw;}
            }
            catch(Exception){usageAvailable=false;}   // (not macOS: wall time only)
            return u;
        }
        static void InstallLoopMarkers()
        {
            if(installed)return;installed=true;
            var loop=UnityEngine.LowLevel.PlayerLoop.GetCurrentPlayerLoop();
            var phases=loop.subSystemList;
            for(int p=0;p<phases.Length;p++)
            {
                var systems=phases[p].subSystemList;if(systems==null)continue;
                var marked=new List<UnityEngine.LowLevel.PlayerLoopSystem>();
                for(int i=0;i<systems.Length;i++)
                {
                    string name=phases[p].type.Name+"."+systems[i].type?.Name;
                    marked.Add(systems[i]);
                    bool last=p==phases.Length-1 && i==systems.Length-1;
                    marked.Add(new UnityEngine.LowLevel.PlayerLoopSystem{type=typeof(LiveTrace),updateDelegate=()=>Mark(name,last)});
                }
                phases[p].subSystemList=marked.ToArray();
            }
            loop.subSystemList=phases;UnityEngine.LowLevel.PlayerLoop.SetPlayerLoop(loop);
        }
        static long frameStart;
        static void Mark(string name,bool last)
        {
            long now=Stopwatch.GetTimestamp();var usage=Now();
            long previous=marks.Count>0?marks[marks.Count-1].ticks:frameStart;
            if(previous>0 && (now-previous)*1000.0/Stopwatch.Frequency>40)
                stalls.Add((now,Time.frameCount,name,(now-previous)*1000.0/Stopwatch.Frequency,new Usage{cpu=usage.cpu-lastUsage.cpu,major=usage.major-lastUsage.major,
                    voluntary=usage.voluntary-lastUsage.voluntary,involuntary=usage.involuntary-lastUsage.involuntary,gc=usage.gc-lastUsage.gc}));
            marks.Add((name,now));lastUsage=usage;
            if(last){frameStart=now;marks.Clear();}
        }
        public static LiveTrace FromCommandLine()
        {
            var args=Environment.GetCommandLineArgs();int at=Array.IndexOf(args,"-liveTrace");
            return at>=0 && at+1<args.Length?new LiveTrace(args[at+1]):null;
        }
        public void Add(Kind kind,long cursor,AudioManager.PcmQueue queue,int pending,long first,long end,int count,double ms)
        {
            if(used==records.Length)return;
            // The latest frame's timings (frame timing stats: main thread, its wait for presentation, render thread, GPU).
            bool timed=kind==Kind.Pump && FrameTimingManager.GetLatestTimings(1,timing)>0;
            var t=timing[0];
            records[used++]=new Record{main=timed?(float)t.cpuMainThreadFrameTime:float.NaN,present=timed?(float)t.cpuMainThreadPresentWaitTime:float.NaN,
                render=timed?(float)t.cpuRenderThreadFrameTime:float.NaN,gpu=timed?(float)t.gpuFrameTime:float.NaN,ticks=Stopwatch.GetTimestamp(),frame=Time.frameCount,kind=kind,cursor=cursor,read=queue?.ReadSample??-1,written=queue?.WrittenSample??-1,
                pending=pending,first=first,end=end,count=count,ms=(float)ms,gc0=GC.CollectionCount(0),gc2=GC.CollectionCount(2)};
        }
        public void Write()
        {
            var text=new System.Text.StringBuilder("seconds,frame,kind,cursor,read,written,pending,batchFirst,batchEnd,updates,ms,gc0,gc2,mainMs,presentWaitMs,renderThreadMs,gpuMs\n");
            for(int i=0;i<used;i++)
            {
                var r=records[i];
                text.Append(FormattableString.Invariant($"{(r.ticks-origin)/(double)Stopwatch.Frequency:F4},{r.frame},{r.kind},{r.cursor},{r.read},{r.written},{r.pending},{r.first},{r.end},{r.count},{r.ms:F2},{r.gc0},{r.gc2},{r.main:F2},{r.present:F2},{r.render:F2},{r.gpu:F2}\n"));
            }
            System.IO.File.WriteAllText(file,text.ToString());
            var lines=new System.Text.StringBuilder("seconds,frame,system,ms,mainThreadCpuMs,majorFaults,voluntarySwitches,involuntarySwitches,collections\n");
            foreach(var (ticks,frame,name,ms,u) in stalls)lines.Append(FormattableString.Invariant($"{(ticks-origin)/(double)Stopwatch.Frequency:F4},{frame},{name},{ms:F1},{u.cpu:F1},{u.major},{u.voluntary},{u.involuntary},{u.gc}\n"));
            System.IO.File.WriteAllText(file+".stalls",lines.ToString());
        }
    }
}
