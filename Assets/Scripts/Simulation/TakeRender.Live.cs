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
        readonly int sampleRate,listener,lead;
        readonly long historySamples;
        readonly DirectSolver solver;
        readonly Mixer mixer;
        readonly GpuAudio gpuAudio;
        readonly Signal[] signals;
        readonly AcousticTake.State[] evaluated,batchStates;
        readonly DirectPath[] paths,batchPaths;
        readonly float[] block,gpuBlock;
        readonly AudioManager.PcmQueue queue;
        readonly Observation[] observations;
        int head,pending,batchCount;
        long batchFirst,batchEnd,previousResponse=-1;
        bool batchActive,batchReady,audioQueued,audioReady;
        readonly List<long> cuts=new List<long>();
        public long Cursor {get;private set;}
        public double SolveSeconds,AudioSeconds,ResponsePreparationSeconds,MaximumSolveSeconds;
        public int ResponseCount,MaximumPendingObservations;
        public int PendingObservations => pending;
        public bool CanObserve => pending<observations.Length;
        public int RetainedEvents { get { int count=cuts.Count; foreach (var signal in signals) count+=signal.RetainedEvents; return count; } }
        public double CompletionSeconds => solver.CompletionSeconds;
        public double MaximumCompletionSeconds => solver.MaximumCompletionSeconds;
        public long PairPrimitiveUpperBound => solver.PairPrimitiveUpperBound;
        public long ResponseSupportSamples => gpuAudio?.ResponseSupportSamples??mixer.ResponseSupportSamples;
        public bool TailCoversResponseSupport => settings.tailSamples>=ResponseSupportSamples;
        public Action<long,long,DirectPath[]> ResponseApplied;
        public string Backend => solver.Backend;
        public Live(AcousticTake.Scene scene,AcousticTake.Primitive[] geometry,float[][] pcm,int rate,Settings settings,
            AudioManager.PcmQueue queue,int lead,int listener,ComputeShader shader=null,SourceStream[] streams=null)
        {
            Validate(settings); historySamples=settings.HistorySamples(rate);
            if (listener<0 || listener>=scene.listeners.Length) throw new ArgumentOutOfRangeException(nameof(listener));
            this.scene=scene; sampleRate=rate; this.settings=settings; this.queue=queue; this.lead=lead; this.listener=listener;
            if(queue!=null && queue.Channels!=settings.ChannelsPerListener) throw new ArgumentException("Queue/output channel mismatch.");
            signals=new Signal[scene.sources.Length];
            for (int i=0;i<signals.Length;i++) signals[i]=scene.sources[i].streamed?new Signal(streams[i],settings.maximumSignalEvents):new Signal(pcm[scene.sources[i].clipIndex],settings.maximumSignalEvents);
            evaluated=new AcousticTake.State[scene.objects.Length]; paths=new DirectPath[scene.sources.Length*scene.listeners.Length];
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
                    gpuAudio=new GpuAudio(scene,rate,settings,signals,settings.maximumResponsesPerBatch);
                }
                else mixer=new Mixer(scene,rate,settings,signals);
                solver=new DirectSolver(scene,geometry,settings,shader,settings.maximumResponsesPerBatch,rate,gpuAudio==null);
            }
            catch {gpuAudio?.Dispose();throw;}
        }
        public void Append(AcousticTake.VoiceEvent e) { signals[e.sourceIndex].Append(e); }
        public void Observe(long from,long to,AcousticTake.State[] left,AcousticTake.State[] right,Action<long,float[],int> output=null)
        {
            // Include the newest interval before dispatching the next GPU batch.
            // Otherwise every completed batch can launch with a frame of stale input.
            if (pending==observations.Length) Pump();
            foreach (var state in right) if ((state.flags&AcousticTake.StateFlags.Cut)!=0)
            { if (cuts.Count==0 || cuts[cuts.Count-1]!=to) cuts.Add(to); break; }
            for (int i=0;i<signals.Length;i++) signals[i].SetActive(to,Enabled(right[scene.sources[i].objectIndex],AcousticTake.StateFlags.Source));
            Enqueue(from,to,left,right,to,output); Pump();
        }
        void Enqueue(long from,long to,AcousticTake.State[] left,AcousticTake.State[] right,long end,Action<long,float[],int> output)
        {
            if (pending==observations.Length) throw new InvalidOperationException($"Live observation queue is full: rendered={Cursor}, observed={to}, sourceReady={ReadyEnd}, outputRead={queue?.ReadSample}, outputWritten={queue?.WrittenSample}, batchEnd={batchEnd}, GPU pending={audioQueued && !audioReady}. Reduce live compute cost.");
            if (from>to || to>end || left.Length!=evaluated.Length || right.Length!=evaluated.Length) throw new ArgumentException("Invalid observation.");
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
            while(pending>0 || finalState!=null)
            {
                if(pending==0)
                {
                    Enqueue(finalSample,finalSample,finalState,finalState,checked(finalSample+1+settings.tailSamples),finalOutput);
                    finalState=null;finalOutput=null;
                }
                var frame=observations[head];
                if (!RenderTo(frame,wait)) return;
                frame.output=null; head=(head+1)%observations.Length; pending--;
                foreach (var signal in signals) signal.Prune(Cursor-historySamples);
                int last=cuts.BinarySearch(previousResponse); if (last<0) last=~last-1;
                if (last>0) cuts.RemoveRange(0,last);
            }
        }
        long ReadyEnd {get {long end=long.MaxValue;for(int s=0;s<signals.Length;s++)if(scene.listeners.Length>1 || scene.sources[s].mutedListenerPlusOne!=1)end=Math.Min(end,signals[s].ReadyEnd);return end;}}
        long WritableEnd=>queue==null?ReadyEnd:Math.Min(ReadyEnd,queue.ReadSample+queue.Capacity-lead);
        void PrepareBatch()
        {
            long knownEnd=Math.Min(WritableEnd,observations[(head+pending-1)%observations.Length].end);
            batchFirst=Cursor%settings.responseStepSamples==0?Cursor:Cursor+settings.responseStepSamples-Cursor%settings.responseStepSamples;
            batchEnd=Math.Min(knownEnd,checked((Cursor/settings.responseStepSamples+settings.maximumResponsesPerBatch)*settings.responseStepSamples));
            batchCount=batchEnd<=batchFirst?0:(int)(1+(batchEnd-1-batchFirst)/settings.responseStepSamples);
            int interval=0;
            for(int f=0;f<batchCount;f++)
            {
                long sample=batchFirst+(long)f*settings.responseStepSamples;
                while(interval+1<pending && sample>=observations[(head+interval)%observations.Length].end)interval++;
                var observed=observations[(head+interval)%observations.Length];
                AcousticTake.Interpolate(Math.Min(sample,observed.to),observed.from,observed.to,sampleRate,observed.left,observed.right,evaluated);
                if(sample>observed.to)FreezeMotion(evaluated);
                Array.Copy(evaluated,0,batchStates,f*evaluated.Length,evaluated.Length);
            }
            if(batchCount>0)
            {
                long start=Stopwatch.GetTimestamp();solver.BeginSolve(batchStates,batchCount);
                double elapsed=Elapsed(start);SolveSeconds+=elapsed;MaximumSolveSeconds=Math.Max(MaximumSolveSeconds,elapsed);
            }
            batchActive=true;batchReady=batchCount==0;audioQueued=audioReady=false;
        }
        void ApplyResponse(long sample)
        {
            int index=(int)((sample-batchFirst)/settings.responseStepSamples);
            if(gpuAudio==null)Array.Copy(batchPaths,index*paths.Length,paths,0,paths.Length);
            Array.Copy(batchStates,index*evaluated.Length,evaluated,0,evaluated.Length);
            int cut=cuts.BinarySearch(sample);if(cut<0)cut=~cut-1;
            long start=Stopwatch.GetTimestamp();
            if(gpuAudio!=null)gpuAudio.SetResponse(sample,cut>=0 && cuts[cut]>previousResponse,solver.RayBuffer,index*solver.RayPathsPerFrame,scene,evaluated);
            else mixer.SetResponse(sample,paths,evaluated,cut>=0 && cuts[cut]>previousResponse,solver.RayPaths,index*solver.RayPathsPerFrame);
            previousResponse=sample;
            double elapsed=Elapsed(start);AudioSeconds+=elapsed;ResponsePreparationSeconds+=elapsed;
            ResponseCount++;
            if(gpuAudio==null)ResponseApplied?.Invoke(sample+lead,queue==null?-1:queue.ReadSample,paths);
        }
        bool CompleteBatch(bool wait)
        {
            if(batchReady)return true;
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
        long audioFirst;
        bool RenderTo(Observation frame,bool wait)
        {
            while(Cursor<frame.end)
            {
                if(!batchActive) {if(WritableEnd<=Cursor)return false;PrepareBatch();}
                if(gpuAudio==null && !CompleteBatch(wait))return false;
                if(gpuAudio!=null && !audioReady)
                {
                    if(!audioQueued)
                    {
                        audioFirst=Cursor;gpuAudio.PrepareInput(batchEnd);
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
                        gpuAudio.BeginReadback();audioQueued=true;
                    }
                    long completion=Stopwatch.GetTimestamp();audioReady=gpuAudio.TryComplete(gpuBlock,wait);AudioSeconds+=Elapsed(completion);
                    if(!audioReady)return false;
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
        public void Dispose() { gpuAudio?.Dispose();solver.Dispose(); }
    }
}
