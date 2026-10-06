using System;
using UnityEngine;

// Automatic station service; SceneSession owns its clock, capture and player.
public sealed class StationDemo : SceneActor
{
    public Transform train,carriage,boardingPlatform;
    public StationLoop route;
    [Min(0)] public float trainSpeed=12,startDistance=22;
    [Min(.1f)] public float trainAcceleration=3,carriageSpacing=7.2f,stationDwell=10;
    public float CurrentTrainSpeed {get;private set;}
    public double TrainDistance => trainDistance%route.Length;
    public bool AtStation {get;private set;}=true;
    public float DwellRemaining {get;private set;}
    double trainDistance,nextStopDistance;
    [Serializable] struct Settings
    {
        public float speed,start,acceleration,spacing,dwell,straightLength,radius;
    }
    Settings Configuration => new Settings {speed=trainSpeed,start=startDistance,acceleration=trainAcceleration,
        spacing=carriageSpacing,dwell=stationDwell,straightLength=route.straightLength,radius=route.radius};
    Settings initial;
    public override string SettingsJson => JsonUtility.ToJson(Configuration);
    public override bool SettingsUnchanged => initial.Equals(Configuration);
    public override void Initialize()
    {
        if(train==null || carriage==null || route==null) throw new InvalidOperationException("Assign the train, carriage and route.");
        initial=Configuration;
        trainDistance=nextStopDistance=startDistance; AtStation=true;
        DwellRemaining=stationDwell; CurrentTrainSpeed=0; PlaceTrain();
    }
    public override bool Step(SceneRecording.Input input,Transform player,float seconds)
    {
        double previous=trainDistance; AdvanceTrain(seconds); return trainDistance!=previous;
    }
    void AdvanceTrain(float seconds)
    {
        if(AtStation)
        {
            float wait=Mathf.Min(seconds,DwellRemaining); DwellRemaining-=wait; seconds-=wait;
            if(DwellRemaining>0 || seconds<=0) return;
            AtStation=false; nextStopDistance=trainDistance+route.Length;
        }
        float speed=CurrentTrainSpeed;
        double remaining=nextStopDistance-trainDistance;
        double travel=StationLoop.AdvanceToStop(ref speed,remaining,trainSpeed,trainAcceleration,seconds);
        trainDistance+=travel; CurrentTrainSpeed=speed;
        if(remaining-travel<=1e-6) {trainDistance=nextStopDistance; AtStation=true; DwellRemaining=stationDwell;}
        if(travel>0) PlaceTrain();
    }

    public void PlaceTrain()
    {
        if(route==null) return;
        route.Place(train,trainDistance); if(carriage!=null) route.Place(carriage,trainDistance-carriageSpacing);
    }
}
