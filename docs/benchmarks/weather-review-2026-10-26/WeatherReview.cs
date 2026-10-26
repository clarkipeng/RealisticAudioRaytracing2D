using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using UnityEngine;

// Diagnostic derivatives share excitation, geometry and listener poses.
// Copy to an isolated runner's Assets/Tests/Editor; never starts live audio.
public static class WeatherReview
{
    [Serializable] sealed class Candidate { public string name; public int period,duration,capacity,sites,visibleOutside,visibleInside; public double minOutsidePower,maxOutsidePower,meanOutsidePower; }
    [Serializable] sealed class Coverage { public Candidate[] candidates; public string interpretation="Expected direct power envelope from visibility, Hann weights and inverse distance; not rendered sound or transmission."; }
    static string Arg(string n) {var a=Environment.GetCommandLineArgs();int i=Array.IndexOf(a,n);return i<0?null:a[i+1];}
    public static void Prepare()
    {
        if(!Application.isBatchMode||Application.isPlaying)throw new InvalidOperationException("Offline only.");
        string root=Arg("-reviewOutput");Directory.CreateDirectory(root);
        using var take=AcousticTake.Open(Arg("-reviewTake"));var meta=take.Metadata;int rate=meta.sampleRate;
        var atOutside=new AcousticTake.State[meta.scene.objects.Length];take.Evaluate(rate,atOutside);
        var atInside=new AcousticTake.State[meta.scene.objects.Length];take.Evaluate(18L*rate,atInside);
        int head=meta.scene.listeners[0];var pcm=Enumerable.Range(0,meta.scene.clips.Length).Select(take.ReadClip).ToArray();
        var candidates=new[]{new Candidate{name="wind-baseline",period=4096,duration=22050,capacity=24},new Candidate{name="wind-long",period=4096,duration=88200,capacity=40},new Candidate{name="wind-dense",period=2048,duration=88200,capacity=80},new Candidate{name="wind-slow",period=4096,duration=176400,capacity=64}};
        foreach(string label in new[]{"rain-outside","rain-inside","wind-baseline","wind-long","wind-dense","wind-slow"})
        {
            bool rain=label.StartsWith("rain");var scene=JsonUtility.FromJson<AcousticTake.Scene>(JsonUtility.ToJson(meta.scene));
            int index=Array.FindIndex(scene.sources,s=>scene.objects[s.objectIndex].name.IndexOf(rain?"rain":"wind",StringComparison.OrdinalIgnoreCase)>=0);
            scene.sources=new[]{scene.sources[index]};var plane=scene.planes.Single(p=>p.source==index);plane.source=0;scene.planes=new[]{plane};
            var poses=(AcousticTake.State[])atInside.Clone();
            if(rain)poses[head]=label.EndsWith("outside")?atOutside[head]:atInside[head];
            else
            {
                var c=candidates.Single(c=>c.name==label);plane.burstPeriodSamples=c.period;plane.burstDurationSamples=c.duration;plane.samples=c.capacity;
                scene.objects=scene.objects.Concat(new[]{new AcousticTake.ObjectDefinition{id="weather-audit-interior",name="Interior comparison listener",scale=Vector3.one}}).ToArray();
                poses=poses.Concat(new[]{atInside[head]}).ToArray();poses[head]=atOutside[head];scene.listeners=new[]{head,scene.objects.Length-1};
            }
            foreach(int listener in scene.listeners){poses[listener].velocity=poses[listener].angularVelocity=poses[listener].acceleration=poses[listener].angularAcceleration=Vector3.zero;}
            long end=6L*rate;string path=Path.Combine(root,label+"-take");
            using(var writer=new AcousticTake.Writer(path,rate,scene,take.Geometry,pcm))
            {
                writer.Append(0,poses);writer.Append(end,poses);
                foreach(var original in take.Events)
                {if(original.sourceIndex!=index||original.sample>=end)continue;var e=original;e.sourceIndex=0;writer.Append(e);}
                writer.Append(new AcousticTake.VoiceEvent{sample=end,sourceIndex=0,action=AcousticTake.VoiceAction.Stop});writer.Complete();
            }
            var settings=new TakeRender.Settings{transport=TakeRender.Transport.Raytraced,synthesis=TakeRender.Synthesis.Spectral,audioBackend=TakeRender.AudioBackend.Gpu,fftSize=128,spectralHop=32,responseStepSamples=2048,maximumDelaySeconds=1.25f,maximumResponsesPerBatch=1,useBvh=true,maxBounces=32,raysPerBin=2,diffractionFactor=0,tailSamples=(int)(1.5*rate)};
            File.WriteAllText(Path.Combine(root,label+"-job.json"),"{\"take\":\""+label+"-take\",\"output\":\""+label+"-audio\",\"shader\":\"Assets/Shaders/Transport.compute\",\"settings\":"+JsonUtility.ToJson(settings)+"}");
            if(!rain)
            {
                var c=candidates.Single(c=>c.name==label);using var probe=AcousticTake.Open(path);var signals=TakeRender.Signal.FromTake(probe,pcm);
                var batch=new Emission.Batch(scene,settings,rate,1,take.Geometry);var points=new SortedDictionary<long,Vector4>();
                for(long t=0;t<end;t+=settings.responseStepSamples)
                {batch.Prepare(signals,t,1,(_,output)=>Array.Copy(poses,output,poses.Length));var range=batch.Ranges[0];for(int n=0;n<range.count;n++){int at=range.first+n;var clock=batch.Clocks[at];points[(long)(((ulong)clock.high<<32)|clock.low)]=batch.Points[at];}}
                var bvh=new DirectBvh(scene,take.Geometry);bvh.Update(poses);var sites=new List<(long birth,double factor)>();c.sites=points.Count;
                foreach(var pair in points)
                {
                    Vector3 point=pair.Value;bool outside=!bvh.Blocked(point,poses[head].position,poses),inside=!bvh.Blocked(point,poses[scene.listeners[1]].position,poses);
                    if(outside){c.visibleOutside++;sites.Add((pair.Key,pair.Value.w*pair.Value.w/(point-poses[head].position).sqrMagnitude));}if(inside)c.visibleInside++;
                }
                var powers=new List<double>();for(long t=Math.Max(2L*rate,c.duration);t<end;t+=rate/50)
                {double power=0;foreach(var site in sites){double w=Emission.Window(t-site.birth,c.duration);power+=site.factor*w*w;}powers.Add(power);}
                c.minOutsidePower=powers.Min();c.maxOutsidePower=powers.Max();c.meanOutsidePower=powers.Average();
            }
        }
        File.WriteAllText(Path.Combine(root,"coverage.json"),JsonUtility.ToJson(new Coverage{candidates=candidates},true));
        Debug.Log("WEATHER_REVIEW_PROBES prepared");
    }
}
