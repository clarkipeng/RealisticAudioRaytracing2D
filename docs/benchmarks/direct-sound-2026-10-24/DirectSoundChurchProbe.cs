using System;
using System.IO;
using System.Linq;
using NUnit.Framework;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;

// Copy into the isolated runner's Assets/Tests/Editor; never enter Play mode.
public sealed class DirectSoundChurchProbe
{
    [Serializable] sealed class Evidence
    {
        public int registeredSources,autostartSources,triangles,colliders,planeSites,emittingSites;
        public int directSlots,previous32BounceSlots;
        public bool settingsPreserved;
        public string[] sourceNames;
        public float[] clipExcerptRms;
        public Vector3[] listenerPositions;
        public int clipOffsetSamples;
        public TakeRender.Report[] renders;
    }
    [Test] public void SavedChurchProfileRendersDirectSound()
    {
        Assert.That(Application.isBatchMode && !Application.isPlaying);
        string root=Environment.GetEnvironmentVariable("ACOUSTIC_DIRECT_PROBE");
        Assert.That(root,Is.Not.Null.And.Not.Empty);
        EditorSceneManager.OpenScene("Assets/Scenes/Church.unity");
        var session=UnityEngine.Object.FindAnyObjectByType<SceneSession>();var manager=session.manager;
        string originalSettings=JsonUtility.ToJson(manager.directSettings);
        Assert.That(manager.directSettings.maxBounces,Is.Zero);
        var sources=manager.takeSources.Select(s=>s.transform).ToArray();
        var clips=manager.takeSources.Select(s=>s.streamed?null:s.clip).ToArray();
        var scene=SceneToTake.Compile(manager.takeObjects,manager.takeDimension,manager.takeCurveTolerance,
            manager.takeDefaultMaterial,sources,clips,manager.takeListeners.ToArray(),out var geometry);
        int rate=manager.sampleRate,duration=rate*2;
        var pcm=new float[scene.clips.Length][];
        for(int i=0;i<clips.Length;i++)if(clips[i]!=null)pcm[scene.sources[i].clipIndex]=RayTraceManager.LoadSample(clips[i],rate);
        var result=new Evidence {registeredSources=sources.Length,autostartSources=session.startSounds.Length,
            triangles=geometry.Length,colliders=UnityEngine.Object.FindObjectsByType<Collider>(FindObjectsInactive.Include,FindObjectsSortMode.None).Length,
            planeSites=scene.planes.Sum(p=>p.samples),sourceNames=manager.takeSources.Select(s=>s.name).ToArray(),renders=new TakeRender.Report[2],
            clipOffsetSamples=7*rate,clipExcerptRms=new float[pcm.Length],listenerPositions=new Vector3[2]};
        for(int c=0;c<pcm.Length;c++)
        {
            double energy=0;for(int n=0;n<duration;n++){float value=pcm[c][(result.clipOffsetSamples+n)%pcm[c].Length];energy+=value*value;}
            result.clipExcerptRms[c]=(float)Math.Sqrt(energy/duration);
            Assert.That(result.clipExcerptRms[c],Is.GreaterThan(.001f),"Every clip must exercise audible input in this excerpt.");
        }
        foreach(var plane in scene.planes)for(int i=0;i<plane.samples;i++)if(AudioSourcePlane.Sample(plane,i).w>0)result.emittingSites++;
        Assert.That(result.registeredSources,Is.EqualTo(8));Assert.That(result.autostartSources,Is.EqualTo(4));
        Assert.That(result.triangles,Is.EqualTo(17321));Assert.That(result.emittingSites,Is.EqualTo(5));
        result.directSlots=manager.directSettings.FrequencyBins*(sources.Length-scene.planes.Length+result.planeSites)*scene.listeners.Length;
        result.previous32BounceSlots=result.directSlots+manager.directSettings.FrequencyBins*manager.directSettings.RaysPerBin*33*sources.Length*scene.listeners.Length;
        for(int scenario=0;scenario<2;scenario++)
        {
            string name=scenario==0?"outdoor":"indoor",path=Path.Combine(root,name+"-take");
            var states=manager.takeObjects.Select((b,i)=>SceneToTake.Observe(scene,i,b.transform)).ToArray();
            // An outdoor position and the saved player spawn inside the church.
            if(scenario==0)states[scene.listeners[0]].position=new Vector3(-20,2.7f,-25);
            result.listenerPositions[scenario]=states[scene.listeners[0]].position;
            var streams=new SourceStream[sources.Length];
            try
            {
                for(int i=0;i<streams.Length;i++)if(scene.sources[i].streamed)streams[i]=new SourceStream(duration+1);
                using var writer=new AcousticTake.Writer(path,rate,scene,geometry,pcm);
                using(var capture=new SourceStream.Capture(path,streams,writer.Metadata))
                {
                    foreach(var stream in streams)if(stream!=null){stream.SilenceUntil(duration+1,duration+1);stream.Finish();}
                    capture.Complete(writer.Metadata);
                }
                writer.Append(0,states);writer.Append(duration,states);
                foreach(var source in session.startSounds)
                    writer.Append(new AcousticTake.VoiceEvent {sourceIndex=manager.takeSources.IndexOf(source),gain=source.gain,loop=source.loop,
                        clipOffset=result.clipOffsetSamples});
                foreach(var source in session.startSounds)
                    writer.Append(new AcousticTake.VoiceEvent {sample=duration,sourceIndex=manager.takeSources.IndexOf(source),action=AcousticTake.VoiceAction.Stop});
                writer.Complete();
            }
            finally {foreach(var stream in streams)stream?.Dispose();}
            using var take=AcousticTake.Open(path);
            var settings=JsonUtility.FromJson<TakeRender.Settings>(originalSettings);
            result.renders[scenario]=TakeRender.RenderDirect(take,settings,Path.Combine(root,name+"-render"),shader:manager.directShader);
            Assert.That(result.renders[scenario].complete);
            Assert.That(result.renders[scenario].gpuAudio.pathCapacity,Is.EqualTo(result.directSlots));
            Debug.Log("DIRECT_CHURCH_RENDERED "+name);
        }
        result.settingsPreserved=JsonUtility.ToJson(manager.directSettings)==originalSettings;
        Assert.That(result.settingsPreserved);
        File.WriteAllText(Path.Combine(root,"church-probe.json"),JsonUtility.ToJson(result,true));
    }
}
