using System;
using System.Collections.Generic;
using UnityEngine;

// Authored inputs are immutable during a run. Each session owns a compiled cursor.
[CreateAssetMenu(menuName="Acoustics/Scene sequence")]
public sealed class SceneSequence : ScriptableObject
{
    public enum CueKind { Sound, Bell, Bird, Lightning }
    public enum LevelKind { Rain, Wind }
    [Serializable] public struct Cue
    {
        public string id,target;
        public CueKind kind;
        public float time,duration,gain,clipOffset;
        public bool loop;
    }
    [Serializable] public struct WeatherKey
    {
        public float time,cloud,rain;
        public Vector3 wind;
    }
    public struct Weather {public float cloud,rain,wetness;public Vector3 wind;}
    [Serializable] public struct LevelTrack {public string id,target;public LevelKind kind;public float start,end,gain;}
    [Serializable] public struct PositionKey {public float time;public Vector3 position;public float yaw;}
    [Serializable] public struct PoseTrack {public string target;public PositionKey[] keys;}
    public int version=1,seed=1;
    public float duration=60;
    public Cue[] cues=Array.Empty<Cue>();
    public WeatherKey[] weather=Array.Empty<WeatherKey>();
    public LevelTrack[] levels=Array.Empty<LevelTrack>();
    public PoseTrack[] poses=Array.Empty<PoseTrack>();

    public static long Sample(double seconds,int rate)=>checked((long)Math.Round(seconds*rate,MidpointRounding.AwayFromZero));
    public Weather At(double seconds)
    {
        if(weather.Length==0)return default;
        float t=(float)Math.Max(0,seconds);var a=weather[0];var b=a;
        double water=0;
        for(int i=1;i<weather.Length;i++)
        {
            var left=weather[i-1];var right=weather[i];float end=Mathf.Min(t,right.time);
            if(end>left.time)
            {
                float q=(end-left.time)/(right.time-left.time);
                water+=(end-left.time)*(left.rain+Mathf.Lerp(left.rain,right.rain,q))*.5;
            }
            if(t>=right.time)a=b=right;
            else if(t>=left.time){a=left;b=right;}
        }
        var last=weather[weather.Length-1];if(t>last.time)water+=(t-last.time)*last.rain;
        float blend=b.time>a.time?Mathf.Clamp01((t-a.time)/(b.time-a.time)):0;
        var wind=Vector3.Lerp(a.wind,b.wind,blend);
        // Stable bounded gust, shared by presentation and the rustle envelope.
        float gust=1+.16f*Mathf.Sin(t*1.7f+seed*.73f)+.08f*Mathf.Sin(t*4.1f+seed*1.13f);
        return new Weather{cloud=Mathf.Lerp(a.cloud,b.cloud,blend),rain=Mathf.Lerp(a.rain,b.rain,blend),
            wind=wind*gust,wetness=Mathf.Clamp01((float)water*.065f)};
    }
    public void Pose(int track,double seconds,out Vector3 position,out Quaternion rotation)
    {
        var keys=poses[track].keys;var a=keys[0];var b=a;
        for(int i=1;i<keys.Length;i++)
        {
            if(seconds>=keys[i].time)a=b=keys[i];
            else {b=keys[i];break;}
        }
        float t=b.time>a.time?Mathf.Clamp01((float)(seconds-a.time)/(b.time-a.time)):0;
        t=t*t*(3-2*t);
        position=Vector3.Lerp(a.position,b.position,t);rotation=Quaternion.Euler(0,Mathf.LerpAngle(a.yaw,b.yaw,t),0);
    }
    public sealed class Run
    {
        public struct Command
        {
            public string occurrence,target;
            public AcousticTake.VoiceEvent audio;
            internal int order;
        }
        public readonly Command[] Commands;
        public readonly int Rate;
        int cursor;long through=-1;
        public Run(SceneSequence sequence,int rate)
        {
            if(rate<1)throw new ArgumentOutOfRangeException(nameof(rate));
            sequence.Validate();Rate=rate;var result=new List<Command>();int ordinal=0;
            void Add(string id,string target,double seconds,AcousticTake.VoiceAction action,float gain=0,bool loop=false,double offset=0)
            {
                result.Add(new Command{occurrence=id,target=target,order=ordinal++,audio=new AcousticTake.VoiceEvent {
                    sample=Sample(seconds,rate),action=action,gain=gain,loop=loop,clipOffset=Sample(offset,rate)}});
            }
            foreach(var cue in sequence.cues)
            {
                Add(cue.id,cue.target,cue.time,AcousticTake.VoiceAction.Play,cue.gain,cue.loop,cue.clipOffset);
                Add(cue.id,cue.target,cue.time+cue.duration,AcousticTake.VoiceAction.Stop);
            }
            foreach(var track in sequence.levels)
            {
                float Level(double t){var w=sequence.At(t);return track.gain*(track.kind==LevelKind.Rain?w.rain:w.wind.magnitude/8f);}
                Add(track.id,track.target,track.start,AcousticTake.VoiceAction.Play,Level(track.start),true);
                // Fixed 50 Hz automation is independent of display/DSP batch rates.
                for(int step=1;track.start+step/50.0<track.end;step++)
                {double t=track.start+step/50.0;Add(track.id,track.target,t,AcousticTake.VoiceAction.Gain,Level(t));}
                Add(track.id,track.target,track.end,AcousticTake.VoiceAction.Stop);
            }
            result.Sort((a,b)=>{int c=a.audio.sample.CompareTo(b.audio.sample);return c!=0?c:a.order.CompareTo(b.order);});
            Commands=result.ToArray();
        }
        public void Advance(long sample,Action<Command> accept)
        {
            if(sample<through)throw new InvalidOperationException("Reset the sequence before seeking backwards.");
            while(cursor<Commands.Length && Commands[cursor].audio.sample<=sample){accept(Commands[cursor]);cursor++;}
            through=sample;
        }
        public void Reset(){cursor=0;through=-1;}
    }
    public void Validate()
    {
        bool Finite(float f)=>float.IsFinite(f);
        bool Vector(Vector3 v)=>Finite(v.x)&&Finite(v.y)&&Finite(v.z);
        if(version!=1 || !Finite(duration) || duration<=0 || cues==null || levels==null || weather==null || poses==null)
            throw new InvalidOperationException("Invalid scene sequence definition.");
        var ids=new HashSet<string>(StringComparer.Ordinal);
        var intervals=new Dictionary<string,List<(float start,float end)>>();
        void Activity(string id,string target,float start,float length,float gain)
        {
            if(string.IsNullOrWhiteSpace(id)||!ids.Add(id)||string.IsNullOrWhiteSpace(target)||!Finite(start)||start<0||
                !Finite(length)||length<=0||start+length>duration+.0001f||!Finite(gain)||gain<0)
                throw new InvalidOperationException("Sequence activities need unique IDs, valid targets, times and gains.");
            if(!intervals.TryGetValue(target,out var spans))intervals[target]=spans=new List<(float,float)>();
            foreach(var span in spans)if(start<=span.end && start+length>=span.start)
                throw new InvalidOperationException("Overlapping or touching activities restart one source: "+target);
            spans.Add((start,start+length));
        }
        foreach(var cue in cues)
        {
            Activity(cue.id,cue.target,cue.time,cue.duration,cue.gain);
            if(!Finite(cue.clipOffset)||cue.clipOffset<0||!Enum.IsDefined(typeof(CueKind),cue.kind))throw new InvalidOperationException("Invalid cue.");
        }
        foreach(var level in levels)
        {
            Activity(level.id,level.target,level.start,level.end-level.start,level.gain);
            if(!Enum.IsDefined(typeof(LevelKind),level.kind))throw new InvalidOperationException("Invalid activity level.");
        }
        float previous=-1;
        foreach(var key in weather)
        {
            if(!Finite(key.time)||key.time<0||key.time<=previous||!Finite(key.cloud)||key.cloud<0||key.cloud>1||
                !Finite(key.rain)||key.rain<0||key.rain>1||!Vector(key.wind))throw new InvalidOperationException("Invalid weather keys.");
            previous=key.time;
        }
        if(weather.Length>0&&weather[0].time!=0)throw new InvalidOperationException("Weather starts at time zero.");
        ids.Clear();
        foreach(var track in poses)
        {
            if(string.IsNullOrWhiteSpace(track.target)||!ids.Add(track.target)||track.keys==null||track.keys.Length==0||track.keys[0].time!=0)
                throw new InvalidOperationException("Pose tracks require unique bindings and an initial pose.");
            previous=-1;
            foreach(var key in track.keys)
            {if(!Finite(key.time)||key.time<=previous||!Vector(key.position)||!Finite(key.yaw))throw new InvalidOperationException("Invalid pose keys.");previous=key.time;}
        }
    }
}
