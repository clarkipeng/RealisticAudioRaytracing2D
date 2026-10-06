using System.Linq;
using UnityEditor;
using UnityEngine;

public static partial class TakeDemo
{
    static void ConfigureStationBoarding(StationDemo demo,SceneSession session)
    {
        if(demo.boardingPlatform!=null) return;
        if(PrefabUtility.IsPartOfPrefabInstance(demo.train)) PrefabUtility.UnpackPrefabInstance(demo.train.gameObject,PrefabUnpackMode.OutermostRoot,InteractionMode.AutomatedAction);
        var wood=AssetDatabase.LoadAssetAtPath<Material>(StationAssets+"/Timber.mat");
        var steel=AssetDatabase.LoadAssetAtPath<Material>(StationAssets+"/Steel.mat");
        var teal=AssetDatabase.LoadAssetAtPath<Material>(StationAssets+"/Wayfinding.mat");
        var yellow=AssetDatabase.LoadAssetAtPath<Material>(StationAssets+"/Safety yellow.mat");
        var pale=AssetDatabase.LoadAssetAtPath<Material>(StationAssets+"/Masonry.mat");
        var concrete=AssetDatabase.LoadAssetAtPath<Material>(StationAssets+"/Concrete.mat");
        foreach(string name in new[]{"train-carriage-box","Carriage acoustic proxy"})
        {
            var old=demo.carriage.Find(name); if(old!=null) Object.DestroyImmediate(old.gameObject);
        }
        // Reuse the imported wheel assemblies. The passenger body is ordinary editable geometry.
        StationModel(demo.carriage,"train-carriage-flatbed",Vector3.zero,2.6f);
        var baseModel=demo.carriage.Find("train-carriage-flatbed");
        baseModel.GetComponentsInChildren<MeshRenderer>().First(r=>r.name=="train-carriage-flatbed").enabled=false;
        StationBox(demo.carriage,"Passenger floor",new Vector3(0,.96f,0),new Vector3(2.9f,.48f,6.6f),wood);
        StationBox(demo.carriage,"Open carriage roof",new Vector3(0,3.7f,0),new Vector3(3.12f,.16f,6.9f),pale);
        StationBox(demo.carriage,"Outer side",new Vector3(1.4f,1.7f,0),new Vector3(.12f,1,6.4f),teal);
        foreach(float z in new[]{-2f,2f}) StationBox(demo.carriage,"Boarding side",new Vector3(-1.4f,1.7f,z),new Vector3(.12f,1,2.4f),teal);
        foreach(float z in new[]{-3.2f,3.2f})
        {
            StationBox(demo.carriage,"End panel",new Vector3(0,1.7f,z),new Vector3(2.9f,1,.12f),teal);
            StationBox(demo.carriage,"Coupling",new Vector3(0,.6f,z*1.09f),new Vector3(.2f,.2f,.7f),steel,false);
            foreach(float x in new[]{-1.4f,1.4f}) StationBox(demo.carriage,"Roof post",new Vector3(x,2.4f,z),new Vector3(.12f,2.4f,.12f),steel,false);
        }
        foreach(float z in new[]{-1.6f,1.6f})
        {
            StationBox(demo.carriage,"Passenger bench",new Vector3(.94f,1.6f,z),new Vector3(.65f,.12f,2.25f),wood);
            StationBox(demo.carriage,"Bench back",new Vector3(1.24f,1.96f,z),new Vector3(.1f,.72f,2.25f),wood);
            foreach(float dz in new[]{-.9f,.9f}) StationBox(demo.carriage,"Bench leg",new Vector3(.94f,1.37f,z+dz),new Vector3(.5f,.34f,.12f),steel,false);
        }

        var platform=new GameObject("Boarding platform").transform; platform.SetParent(session.manager.takeRoot,false); demo.boardingPlatform=platform;
        // Author the stop beside the starting carriage pose, including a level crossing from the platform into the carriage.
        var stop=demo.route.PoseAt(demo.startDistance-demo.carriageSpacing);
        platform.SetPositionAndRotation(stop.position,stop.rotation);
        StationBox(platform,"Boarding deck",new Vector3(-2.725f,.6f,0),new Vector3(2.35f,1.2f,8),concrete);
        StationBox(platform,"Boarding edge",new Vector3(-1.67f,1.215f,0),new Vector3(.16f,.03f,8),yellow,false);
        for(int i=0;i<6;i++)
        {
            float height=(i+1)*.2f;
            StationBox(platform,"Step "+(i+1),new Vector3(-6.65f+i*.5f,height/2-.02f,0),new Vector3(.5f,height,2.4f),concrete);
            StationBox(platform,"Step nosing",new Vector3(-6.86f+i*.5f,height-.012f,0),new Vector3(.08f,.016f,2.4f),yellow,false);
        }
        foreach(float z in new[]{-3.8f,3.8f}) StationBox(platform,"Platform end rail",new Vector3(-2.725f,1.7f,z),new Vector3(2.35f,1.04f,.12f),steel);
        demo.trainSpeed=12; demo.trainAcceleration=3; demo.stationDwell=10;
        session.player.GetComponentInChildren<MouseLook>().sensitivity=180;
        session.manager.takePath=""; // The freight-car roster belongs to its old take; record this assembly afresh.
        EditorUtility.SetDirty(session.player.GetComponentInChildren<MouseLook>());
    }
}
