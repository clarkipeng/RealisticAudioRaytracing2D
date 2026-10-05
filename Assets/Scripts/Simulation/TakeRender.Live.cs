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
        readonly Signal[] signals;
        readonly AcousticTake.State[] evaluated,batchStates;
        readonly DirectPath[] paths,batchPaths;
        readonly float[] block;
        readonly AudioManager.PcmQueue queue;
        readonly Observation[] observations;
        int head,pending,batchCount;
        long batchFirst,batchEnd,previousResponse=-1;
        bool batchReady;
        readonly List<long> cuts=new List<long>();
        public long Cursor {get;private set;}
        public double SolveSeconds,AudioSeconds,MaximumSolveSeconds;
        public int ResponseCount,MaximumPendingObservations;
        public int PendingObservations => pending;
        public int RetainedEvents { get { int count=cuts.Count; foreach (var signal in signals) count+=signal.RetainedEvents; return count; } }
        public double CompletionSeconds => solver.CompletionSeconds;
        public double MaximumCompletionSeconds => solver.MaximumCompletionSeconds;
        public long PairPrimitiveUpperBound => solver.PairPrimitiveUpperBound;
        public Action<long,long,DirectPath[]> ResponseApplied;
        public string Backend => solver.Backend;
        public Live(AcousticTake.Scene scene,AcousticTake.Primitive[] geometry,float[][] pcm,int rate,Settings settings,
            AudioManager.PcmQueue queue,int lead,int listener,ComputeShader shader=null)
        {
            Validate(settings); historySamples=settings.HistorySamples(rate);
            if (listener<0 || listener>=scene.listeners.Length) throw new ArgumentOutOfRangeException(nameof(listener));
            this.scene=scene; sampleRate=rate; this.settings=settings; this.queue=queue; this.lead=lead; this.listener=listener;
            if(queue!=null && queue.Channels!=settings.ChannelsPerListener) throw new ArgumentException("Queue/output channel mismatch.");
            signals=new Signal[scene.sources.Length];
            for (int i=0;i<signals.Length;i++) signals[i]=new Signal(pcm[scene.sources[i].clipIndex],settings.maximumSignalEvents);
            mixer=new Mixer(scene,rate,settings,signals);
            evaluated=new AcousticTake.State[scene.objects.Length]; paths=new DirectPath[scene.sources.Length*scene.listeners.Length];
            batchStates=new AcousticTake.State[checked(evaluated.Length*settings.maximumResponsesPerBatch)];
            batchPaths=new DirectPath[checked(paths.Length*settings.maximumResponsesPerBatch)];
            block=new float[checked(settings.responseStepSamples*scene.listeners.Length*settings.ChannelsPerListener)];
            observations=new Observation[settings.maximumPendingObservations];
            for (int i=0;i<observations.Length;i++) observations[i]=new Observation {left=new AcousticTake.State[evaluated.Length],right=new AcousticTake.State[evaluated.Length]};
            solver=new DirectSolver(scene,geometry,settings,shader,settings.maximumResponsesPerBatch,rate);
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
            if (pending==observations.Length) throw new InvalidOperationException("Live observation queue is full; reduce compute cost or increase Maximum Pending Observations.");
            if (from>to || to>end || left.Length!=evaluated.Length || right.Length!=evaluated.Length) throw new ArgumentException("Invalid observation.");
            var frame=observations[(head+pending)%observations.Length];
            frame.from=from; frame.to=to; frame.end=end; frame.output=output;
            Array.Copy(left,frame.left,left.Length); Array.Copy(right,frame.right,right.Length);
            pending++; MaximumPendingObservations=Math.Max(MaximumPendingObservations,pending);
        }
        public void Finish(long sample,AcousticTake.State[] final,Action<long,float[],int> output=null)
        {
            foreach (var signal in signals) signal.EndSample=sample;
            Pump(true);
            Enqueue(sample,sample,final,final,checked(sample+1+settings.tailSamples),output); Pump(true);
            if (pending!=0) throw new InvalidOperationException("GPU response did not complete during shutdown.");
        }
        public void Pump(bool wait=false)
        {
            while (pending>0)
            {
                var frame=observations[head];
                if (!RenderTo(frame,wait)) return;
                frame.output=null; head=(head+1)%observations.Length; pending--;
                foreach (var signal in signals) signal.Prune(Cursor-historySamples);
                int last=cuts.BinarySearch(previousResponse); if (last<0) last=~last-1;
                if (last>0) cuts.RemoveRange(0,last);
            }
        }
        bool RenderTo(Observation frame,bool wait)
        {
            while (Cursor<frame.end)
            {
                if (Cursor%settings.responseStepSamples==0)
                {
                    if (batchCount==0)
                    {
                        batchFirst=Cursor;
                        // Batch all known queued intervals so readback latency cannot limit
                        // throughput to one observation per GPU completion.
                        long knownEnd=observations[(head+pending-1)%observations.Length].end;
                        batchCount=(int)Math.Min(settings.maximumResponsesPerBatch,1+(knownEnd-1-Cursor)/settings.responseStepSamples);
                        batchEnd=Math.Min(knownEnd,checked(Cursor+(long)batchCount*settings.responseStepSamples));
                        int interval=0;
                        for (int f=0;f<batchCount;f++)
                        {
                            long sample=Cursor+(long)f*settings.responseStepSamples;
                            while (interval+1<pending && sample>=observations[(head+interval)%observations.Length].end) interval++;
                            var observed=observations[(head+interval)%observations.Length];
                            AcousticTake.Interpolate(Math.Min(sample,observed.to),observed.from,observed.to,sampleRate,observed.left,observed.right,evaluated);
                            Array.Copy(evaluated,0,batchStates,f*evaluated.Length,evaluated.Length);
                        }
                        long start=Stopwatch.GetTimestamp(); solver.BeginSolve(batchStates,batchCount);
                        double elapsed=Elapsed(start); SolveSeconds+=elapsed; MaximumSolveSeconds=Math.Max(MaximumSolveSeconds,elapsed);
                    }
                    if (!batchReady) { if (!solver.TryComplete(batchPaths,wait)) return false; batchReady=true; }
                    int index=(int)((Cursor-batchFirst)/settings.responseStepSamples);
                    Array.Copy(batchPaths,index*paths.Length,paths,0,paths.Length);
                    Array.Copy(batchStates,index*evaluated.Length,evaluated,0,evaluated.Length);
                    int cut=cuts.BinarySearch(Cursor); if (cut<0) cut=~cut-1;
                    long responseStart=Stopwatch.GetTimestamp();
                    mixer.SetResponse(Cursor,paths,evaluated,cut>=0 && cuts[cut]>previousResponse,solver.RayPaths,index*solver.RayPathsPerFrame); previousResponse=Cursor;AudioSeconds+=Elapsed(responseStart);
                    ResponseCount++; ResponseApplied?.Invoke(Cursor+lead,queue==null ? -1:queue.ReadSample,paths);
                }
                int count=(int)Math.Min(frame.end-Cursor,settings.responseStepSamples-Cursor%settings.responseStepSamples);
                long audioStart=Stopwatch.GetTimestamp(); mixer.Render(Cursor,count,block); AudioSeconds+=Elapsed(audioStart);
                queue?.Publish(Cursor+lead,block,count,scene.listeners.Length*settings.ChannelsPerListener,listener*settings.ChannelsPerListener);
                frame.output?.Invoke(Cursor,block,count);
                Cursor+=count;
                if (batchCount>0 && Cursor>=batchEnd) { batchCount=0; batchReady=false; }
            }
            return true;
        }
        public void Dispose() { solver.Dispose(); }
    }
}
