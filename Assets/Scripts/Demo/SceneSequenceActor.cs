using System;
using System.Collections.Generic;
using UnityEngine;

// The session drives poses. Its capture adapter separately drains acoustic commands.
public sealed class SceneSequenceActor : SceneActor
{
    [Serializable] public struct Binding {public string id;public Transform target;}
    public SceneSequence sequence;
    public Binding[] bindings=Array.Empty<Binding>();
    [NonSerialized] public int sampleRate=44100;
    public double Seconds {get;private set;}
    public SceneSequence Definition => snapshot!=null?snapshot:sequence;
    SceneSequence snapshot;
    SceneSequence.Run run;
    string initial;
    Binding[] initialBindings;
    long ticks,audioOrigin;float step;
    public override string SettingsJson=>JsonUtility.ToJson(sequence);
    public override bool SettingsUnchanged
    {
        get
        {
            if(initial!=SettingsJson||bindings.Length!=initialBindings.Length)return false;
            for(int i=0;i<bindings.Length;i++)if(bindings[i].id!=initialBindings[i].id||bindings[i].target!=initialBindings[i].target)return false;
            return true;
        }
    }
    public Transform Target(string id)
    {
        foreach(var b in bindings)if(b.id==id)return b.target;
        throw new InvalidOperationException("Missing sequence binding: "+id);
    }
    public bool Owns(Transform source)
    {
        foreach(var c in Definition.cues)if(Target(c.target)==source)return true;
        foreach(var l in Definition.levels)if(Target(l.target)==source)return true;
        return false;
    }
    public override void Initialize()
    {
        if(sequence==null)throw new InvalidOperationException("Assign a scene sequence.");
        sequence.Validate();initial=SettingsJson;initialBindings=(Binding[])bindings.Clone();
        for(int i=0;i<bindings.Length;i++)
        {
            if(string.IsNullOrWhiteSpace(bindings[i].id)||bindings[i].target==null)throw new InvalidOperationException("Invalid sequence binding.");
            for(int j=0;j<i;j++)if(bindings[i].id==bindings[j].id||bindings[i].target==bindings[j].target)throw new InvalidOperationException("Duplicate sequence binding.");
        }
        Release();snapshot=ScriptableObject.CreateInstance<SceneSequence>();JsonUtility.FromJsonOverwrite(JsonUtility.ToJson(sequence),snapshot);
        snapshot.hideFlags=HideFlags.HideAndDontSave;run=new SceneSequence.Run(snapshot,sampleRate);
        foreach(var c in run.Commands)Target(c.target);
        foreach(var p in snapshot.poses)Target(p.target);
        ticks=audioOrigin=0;step=0;Apply(0);
    }
    public void Apply(double seconds)
    {
        Seconds=seconds;
        for(int i=0;i<Definition.poses.Length;i++)
        {Definition.Pose(i,seconds,out var p,out var q);Target(Definition.poses[i].target).SetPositionAndRotation(p,q);}
    }
    public void Drain(Action<Transform,AcousticTake.VoiceEvent> accept)
    {
        run.Advance(SceneSequence.Sample(Seconds,sampleRate),c=>{var e=c.audio;e.sample-=audioOrigin;accept(Target(c.target),e);});
    }
    // A new capture can start mid-sequence (e.g. after Save). Restore active
    // voices at their current phase and rebase future commands to its clock.
    public void BeginAudio(int rate,Action<Transform,AcousticTake.VoiceEvent> accept)
    {
        sampleRate=rate;run=new SceneSequence.Run(Definition,rate);audioOrigin=SceneSequence.Sample(Seconds,rate);
        var active=new Dictionary<string,AcousticTake.VoiceEvent>();
        run.Advance(audioOrigin,c=> {
            var e=c.audio;
            if(e.action==AcousticTake.VoiceAction.Stop)active.Remove(c.target);
            else if(e.action==AcousticTake.VoiceAction.Play)active[c.target]=e;
            else if(active.TryGetValue(c.target,out var voice)){voice.gain=e.gain;active[c.target]=voice;}
        });
        foreach(var binding in bindings)if(active.TryGetValue(binding.id,out var e))
        {
            var source=binding.target.GetComponent<RaytracedAudioSource>();
            if(source==null||source.streamed||source.clip==null)throw new InvalidOperationException("Sequence cues require fixed clip sources.");
            long length=Mathf.RoundToInt(source.clip.samples/((float)source.clip.frequency/rate));
            e.clipOffset+=audioOrigin-e.sample;e.sample=0;
            if(e.loop)e.clipOffset%=length;
            else if(e.clipOffset>=length)continue;
            accept(binding.target,e);
        }
    }
    public override bool Step(SceneRecording.Input input,Transform player,float seconds)
    {
        if(!(seconds>0)|| (step!=0&&step!=seconds))throw new InvalidOperationException("Sequence simulation step changed.");
        step=seconds;Apply(++ticks*(double)step);return Definition.poses.Length>0;
    }
    void Release()
    {
        if(snapshot!=null){if(Application.isPlaying)Destroy(snapshot);else DestroyImmediate(snapshot);snapshot=null;}
    }
    void OnDestroy()=>Release();
}
