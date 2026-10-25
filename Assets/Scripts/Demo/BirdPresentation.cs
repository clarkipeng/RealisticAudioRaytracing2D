using UnityEngine;

public sealed class BirdPresentation : MonoBehaviour
{
    public SceneSequenceActor sequence;
    public string target="bird";
    public Transform leftWing,rightWing,beak;
    public float[] callEnvelope;
    public float flightStart=21,flightEnd=25;
    void LateUpdate()=>Evaluate(sequence.Seconds);
    public void Evaluate(double seconds)
    {
        float t=(float)seconds,call=0;
        foreach(var cue in sequence.Definition.cues)
            if(cue.kind==SceneSequence.CueKind.Bird&&cue.target==target&&t>=cue.time&&t<cue.time+cue.duration)
            {
                int frame=Mathf.FloorToInt((t-cue.time+cue.clipOffset)*100);
                if(callEnvelope!=null&&frame>=0&&frame<callEnvelope.Length)call=Mathf.Max(call,callEnvelope[frame]);
            }
        bool flying=t>flightStart&&t<flightEnd;
        float flap=flying?Mathf.Sin(t*38)*48:8;
        leftWing.localRotation=Quaternion.Euler(0,flying?0:-75,-flap);
        rightWing.localRotation=Quaternion.Euler(0,flying?0:75,flap);
        beak.localRotation=Quaternion.Euler(call*16,0,0);
    }
}
