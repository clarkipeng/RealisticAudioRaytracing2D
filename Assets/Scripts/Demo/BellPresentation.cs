using UnityEngine;

// The phrase's measured PCM onsets drive contact; this is animation, not vibration synthesis.
public sealed class BellPresentation : MonoBehaviour
{
    public SceneSequenceActor sequence;
    public Transform body,clapper;
    public string target="bell";
    public float[] strikes;
    void LateUpdate()=>Evaluate(sequence.Seconds);
    public void Evaluate(double seconds)
    {
        float angle=0,contact=0;
        foreach(var cue in sequence.Definition.cues)
        {
            if(cue.kind!=SceneSequence.CueKind.Bell||cue.target!=target||strikes==null||strikes.Length<2)continue;
            float t=(float)(seconds-cue.time);if(t<-.9f||t>cue.duration)continue;
            int i=0;while(i<strikes.Length-2&&t>strikes[i+1])i++;
            float phase=(t-strikes[i])/(strikes[i+1]-strikes[i]);
            float envelope=Mathf.SmoothStep(0,1,(t+.9f)/.6f)*Mathf.Exp(-Mathf.Max(0,t-strikes[strikes.Length-1])/2);
            angle=28*Mathf.Cos((i+phase)*Mathf.PI)*envelope;
            contact=-29*Mathf.Cos((i+phase)*Mathf.PI)*envelope;
        }
        body.localRotation=Quaternion.Euler(0,0,angle);
        clapper.localRotation=Quaternion.Euler(0,0,contact);
    }
}
