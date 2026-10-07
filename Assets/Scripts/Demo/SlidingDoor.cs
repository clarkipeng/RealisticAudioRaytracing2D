using System;
using UnityEngine;

public sealed class SlidingDoor : SceneActor
{
    public Vector3 openOffset=new Vector3(2.2f,0,0);
    [Min(.1f)] public float speed=1.8f,interactionDistance=2.2f;
    public bool initiallyOpen;
    public bool IsOpen {get;private set;}
    Vector3 closedPosition;
    (Vector3,float,float,bool) initial;
    public override string SettingsJson => JsonUtility.ToJson(new Settings {offset=openOffset,speed=speed,distance=interactionDistance,open=initiallyOpen});
    [Serializable] struct Settings { public Vector3 offset; public float speed,distance; public bool open; }
    public override bool SettingsUnchanged => initial.Equals((openOffset,speed,interactionDistance,initiallyOpen));
    public override void Initialize()
    {
        closedPosition=transform.localPosition; IsOpen=initiallyOpen;
        initial=(openOffset,speed,interactionDistance,initiallyOpen);
    }
    public void Interact(Vector3 playerPosition)
    {
        var anchor=transform.parent!=null?transform.parent.TransformPoint(closedPosition):closedPosition;
        if(Vector3.Distance(playerPosition+Vector3.up,anchor)<=interactionDistance)IsOpen=!IsOpen;
    }
    public override bool Step(SceneRecording.Input input,Transform player,float seconds)
    {
        if(input.interact)Interact(player.position);
        var previous=transform.localPosition;
        transform.localPosition=Vector3.MoveTowards(previous,closedPosition+(IsOpen?openOffset:Vector3.zero),speed*seconds);
        return transform.localPosition!=previous;
    }
}
