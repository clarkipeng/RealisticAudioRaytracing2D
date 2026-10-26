using System;
using System.IO;
using System.Linq;
using UnityEngine;

public static partial class ChurchPresentationCapture
{
    // Numeric diagnostic derivatives: same geometry, stored poses, event times
    // and source identities. Their own manifests authenticate the selected mix.
    public static void PrepareWeatherProbes()
    {
        if(!Application.isBatchMode||Application.isPlaying)throw new InvalidOperationException("Offline probes only.");
        string directory=Argument("-sequenceOutput");
        using var take=AcousticTake.Open(Path.Combine(directory,"take"));
        var meta=take.Metadata;var pcm=Enumerable.Range(0,meta.scene.clips.Length).Select(take.ReadClip).ToArray();
        foreach(string name in new[]{"rain","wind","short"})
        {
            var scene=JsonUtility.FromJson<AcousticTake.Scene>(JsonUtility.ToJson(meta.scene));
            int[] indices=Enumerable.Range(0,scene.sources.Length).Where(i=>name=="short"?scene.planes.Any(p=>p.source==i):
                scene.objects[scene.sources[i].objectIndex].name.IndexOf(name,StringComparison.OrdinalIgnoreCase)>=0).ToArray();
            if(indices.Length==0)throw new InvalidDataException("Missing weather probe source: "+name);
            scene.sources=indices.Select(i=>scene.sources[i]).ToArray();
            scene.planes=scene.planes.Where(p=>Array.IndexOf(indices,p.source)>=0).ToArray();
            foreach(var plane in scene.planes)plane.source=Array.IndexOf(indices,plane.source);
            long end=name=="short"?4L*meta.sampleRate:meta.durationSamples;
            string output=Path.Combine(directory,name+"-take");
            using(var writer=new AcousticTake.Writer(output,meta.sampleRate,scene,take.Geometry,pcm))
            {
                var state=new AcousticTake.State[scene.objects.Length];
                for(int f=0;f<meta.frameCount&&take.FrameSample(f)<end;f++)
                {take.ReadStoredFrame(f,state);writer.Append(take.FrameSample(f),state);}
                take.Evaluate(end,state);writer.Append(end,state);
                foreach(var original in take.Events)
                {
                    var e=original;int source=Array.IndexOf(indices,e.sourceIndex);
                    if(source<0||e.sample>end)continue;e.sourceIndex=source;writer.Append(e);
                }
                if(name=="short")for(int i=0;i<scene.sources.Length;i++)
                    writer.Append(new AcousticTake.VoiceEvent{sample=end,sourceIndex=i,action=AcousticTake.VoiceAction.Stop});
                writer.Complete();
            }
            var job=TakeBatch.ParseJob(File.ReadAllText(Path.Combine(directory,"render-job.json")));
            job.take=name+"-take";job.output=name+"-audio";
            File.WriteAllText(Path.Combine(directory,name+"-job.json"),Newtonsoft.Json.JsonConvert.SerializeObject(job,Newtonsoft.Json.Formatting.Indented));
        }
        Debug.Log("CHURCH_WEATHER_PROBES prepared");
    }
}
