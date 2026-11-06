using System;
using System.Collections.Generic;
using System.IO;
using UnityEngine;

// Offline measurement entry points for the late-energy benchmark
// (docs/benchmarks/late-energy-2026-11-04). Batch mode only; never starts audio.
public static partial class LateEnergyMeasurement
{
    static string Arg(string name)
    {
        var args=Environment.GetCommandLineArgs();int i=Array.IndexOf(args,name);
        return i<0 || i+1>=args.Length?null:args[i+1];
    }

    // Unity -batchmode -executeMethod LateEnergyMeasurement.DeriveSegment
    //   -segmentTake <take> -segmentOutput <new take> -segmentStart <s> -segmentEnd <s>
    // Copies poses and source events of [start, end) into a new take starting at
    // sample 0. Sources that never emit in the window (silent streams, idle clips)
    // are dropped; playing voices restart at sample 0 with the clip position they
    // had at the window start. Seeded bursts derive their schedule from the voice
    // epoch, so their realization differs from the original take, identically for
    // every renderer that reads the derived take.
    public static void DeriveSegment()
    {
        if(!Application.isBatchMode)throw new InvalidOperationException("Offline only.");
        string input=Arg("-segmentTake"),output=Arg("-segmentOutput");
        double start=double.Parse(Arg("-segmentStart"),System.Globalization.CultureInfo.InvariantCulture);
        double end=double.Parse(Arg("-segmentEnd"),System.Globalization.CultureInfo.InvariantCulture);
        using var take=AcousticTake.Open(input);var meta=take.Metadata;int rate=meta.sampleRate;
        long first=(long)Math.Round(start*rate),last=(long)Math.Round(end*rate);
        if(first<0 || last<=first || last>meta.durationSamples)throw new ArgumentException("Segment outside the take.");
        var scene=JsonUtility.FromJson<AcousticTake.Scene>(JsonUtility.ToJson(meta.scene));
        // Voice state at the window start, and whether each source emits inside it.
        var playing=new AcousticTake.VoiceEvent?[scene.sources.Length];var emits=new bool[scene.sources.Length];
        foreach(var e in take.Events)
        {
            if(e.sample<=first)
            {
                if(e.action==AcousticTake.VoiceAction.Play)playing[e.sourceIndex]=e;
                else if(e.action==AcousticTake.VoiceAction.Stop)playing[e.sourceIndex]=null;
                else if(playing[e.sourceIndex] is AcousticTake.VoiceEvent p){p.gain=e.gain;playing[e.sourceIndex]=p;}
            }
            else if(e.sample<last && e.action==AcousticTake.VoiceAction.Play)emits[e.sourceIndex]=true;
        }
        var keep=new List<int>();
        for(int s=0;s<scene.sources.Length;s++)if(!scene.sources[s].streamed && (emits[s] || playing[s]!=null))keep.Add(s);
        var remap=new int[scene.sources.Length];Array.Fill(remap,-1);for(int i=0;i<keep.Count;i++)remap[keep[i]]=i;
        var sources=new AcousticTake.SourceDefinition[keep.Count];for(int i=0;i<keep.Count;i++)sources[i]=scene.sources[keep[i]];
        scene.sources=sources;
        if(scene.planes!=null)
        {
            var planes=new List<AcousticTake.EmissionPlane>();
            foreach(var plane in scene.planes)if(remap[plane.source]>=0){plane.source=remap[plane.source];planes.Add(plane);}
            scene.planes=planes.ToArray();
        }
        var pcm=new float[scene.clips.Length][];for(int c=0;c<pcm.Length;c++)pcm[c]=take.ReadClip(c);
        var state=new AcousticTake.State[scene.objects.Length];
        using(var writer=new AcousticTake.Writer(output,rate,scene,take.Geometry,pcm))
        {
            take.Evaluate(first,state);writer.Append(0,state);
            for(int f=0;f<meta.frameCount;f++)
            {
                long sample=take.FrameSample(f);if(sample<=first || sample>=last)continue;
                take.ReadStoredFrame(f,state);writer.Append(sample-first,state);
            }
            foreach(int s in keep)if(playing[s] is AcousticTake.VoiceEvent p)
            {
                // Seeded bursts read clips by burst age, so their voice offset is not a clock.
                bool bursts=scene.planes!=null && Array.Exists(scene.planes,x=>x.source==remap[s] && x.generatorVersion!=0);
                long offset=bursts?p.clipOffset:p.clipOffset+(first-p.sample);int length=pcm[scene.sources[remap[s]].clipIndex].Length;
                if(p.loop)offset%=length;else if(offset>length)continue;
                writer.Append(new AcousticTake.VoiceEvent{sample=0,sourceIndex=remap[s],action=AcousticTake.VoiceAction.Play,clipOffset=offset,loop=p.loop,gain=p.gain});
            }
            foreach(var e in take.Events)
            {
                if(e.sample<=first || e.sample>=last || remap[e.sourceIndex]<0)continue;
                var copy=e;copy.sample-=first;copy.sourceIndex=remap[e.sourceIndex];writer.Append(copy);
            }
            take.Evaluate(last,state);writer.Append(last-first,state);writer.Complete();
        }
        Debug.Log("LATE_ENERGY_SEGMENT "+output+" sources="+string.Join(",",keep));
    }
}
