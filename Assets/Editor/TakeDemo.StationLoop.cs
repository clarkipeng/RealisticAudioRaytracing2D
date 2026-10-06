using System;
using System.Collections.Generic;
using System.Linq;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;

public static partial class TakeDemo
{
    [MenuItem("Tools/Acoustics/Upgrade station loop")]
    public static void UpgradeStationLoop()
    {
        if(EditorApplication.isPlaying) throw new InvalidOperationException("Stop Play mode before editing the station.");
        var demo=UnityEngine.Object.FindAnyObjectByType<StationDemo>();
        if(demo==null) throw new InvalidOperationException("Open TrainStation first.");
        var session=demo.GetComponent<SceneSession>();
        ConfigureStationLoop(demo,session); session.manager.RegisterTakeObjects();
        EditorSceneManager.MarkSceneDirty(demo.gameObject.scene); EditorSceneManager.SaveScene(demo.gameObject.scene); AssetDatabase.SaveAssets();
    }
    static void ConfigureStationLoop(StationDemo demo,SceneSession session)
    {
        var world=session.manager.takeRoot;
        if(demo.route==null)
        {
            var go=new GameObject("Train loop"); go.transform.SetParent(world,false); go.transform.position=new Vector3(-.5f,.02f,0);
            demo.route=go.AddComponent<StationLoop>();
        }
        var architecture=world.Find("Station architecture");
        if(architecture!=null)
        {
            var old=Enumerable.Range(0,architecture.childCount).Select(i=>architecture.GetChild(i)).Where(t=>t.name=="Track bed" || t.name=="railroad-straight").ToArray();
            if(old.Length>0)
            {
                if(PrefabUtility.IsPartOfPrefabInstance(architecture)) PrefabUtility.UnpackPrefabInstance(architecture.gameObject,PrefabUnpackMode.OutermostRoot,InteractionMode.AutomatedAction);
                foreach(var t in old) UnityEngine.Object.DestroyImmediate(t.gameObject);
                PrefabUtility.SaveAsPrefabAssetAndConnect(architecture.gameObject,"Assets/Prefabs/Station/Platform.prefab",InteractionMode.AutomatedAction);
            }
        }
        if(demo.carriage==null)
        {
            if(PrefabUtility.IsPartOfPrefabInstance(demo.train)) PrefabUtility.UnpackPrefabInstance(demo.train.gameObject,PrefabUnpackMode.OutermostRoot,InteractionMode.AutomatedAction);
            var carriage=new GameObject("Following carriage").transform; carriage.SetParent(demo.train,false); carriage.localPosition=Vector3.back*demo.carriageSpacing;
            foreach(string name in new[]{"train-carriage-box","Carriage acoustic proxy"})
            {
                var child=demo.train.Find(name); if(child==null) throw new InvalidOperationException("Missing train part: "+name);
                child.SetParent(carriage,true);
            }
            demo.carriage=carriage;
            session.manager.takeObjects.Add(new RayTraceManager.TakeObjectBinding {transform=carriage});
        }
        ConfigureStationBoarding(demo,session);
        PrefabUtility.SaveAsPrefabAssetAndConnect(demo.train.gameObject,"Assets/Prefabs/Station/Train.prefab",InteractionMode.AutomatedAction);
        foreach(var source in demo.train.GetComponentsInChildren<RaytracedAudioSource>()) {source.rayTraceManager=session.manager; PrefabUtility.RecordPrefabInstancePropertyModifications(source);}
        RebuildStationTrack(demo.route);
        var panel=demo.GetComponent<RaytraceControlPanel>(); if(panel!=null) {panel.buttonSize=new Vector2(260,30); panel.spacing=3;}
        EditorUtility.SetDirty(demo); EditorUtility.SetDirty(session.manager);
    }

    public static void RebuildStationTrack(StationLoop route)
    {
        if(EditorApplication.isPlaying) throw new InvalidOperationException("Stop Play mode before rebuilding track.");
        // Warp the existing Kenney rail/tie mesh onto exactly the curve used by the vehicles.
        var prefab=AssetDatabase.LoadAssetAtPath<GameObject>(Models+"railroad-straight.fbx");
        var filter=prefab.GetComponentInChildren<MeshFilter>(); var input=filter.sharedMesh;
        const float scale=2.6f;
        var vertices=new List<Vector3>(); var uv=new List<Vector2>(); var rails=new List<int>(); var bed=new List<int>();
        var inputVertices=input.vertices; var inputUv=input.uv; var triangles=input.triangles;
        int segments=Mathf.CeilToInt((float)route.Length/(input.bounds.size.z*scale)); double segmentLength=route.Length/segments;
        for(int s=0;s<segments;s++)
        {
            int offset=vertices.Count;
            for(int i=0;i<inputVertices.Length;i++)
            {
                var v=inputVertices[i]; var pose=route.LocalPose(segmentLength*(s+(v.z-input.bounds.min.z)/input.bounds.size.z));
                vertices.Add(pose.position+pose.rotation*new Vector3(v.x*scale,(v.y-input.bounds.max.y)*scale,0)); uv.Add(inputUv[i]);
            }
            foreach(int index in triangles) rails.Add(offset+index);
        }
        int bedStart=vertices.Count, steps=Mathf.CeilToInt((float)route.Length);
        for(int i=0;i<=steps;i++)
        {
            var pose=route.LocalPose(route.Length*i/steps);
            foreach(var v in new[]{new Vector3(-2.1f,-.25f,0),new Vector3(2.1f,-.25f,0),new Vector3(-2.1f,-.34f,0),new Vector3(2.1f,-.34f,0)})
            {vertices.Add(pose.position+pose.rotation*v); uv.Add(new Vector2(v.x,i));}
            if(i==steps) continue;
            int a=bedStart+i*4,b=a+4;
            bed.AddRange(new[]{a,b,a+1,a+1,b,b+1,a+2,b+2,a,a,b+2,b,a+1,b+1,a+3,a+3,b+1,b+3});
        }
        const string path="Assets/Prefabs/Station/LoopTrack.asset";
        var saved=AssetDatabase.LoadAssetAtPath<Mesh>(path);
        if(saved==null) {saved=new Mesh {name="Station loop track"}; AssetDatabase.CreateAsset(saved,path);}
        // Update the existing mesh through its API so render buffers and saved geometry change together.
        saved.Clear(); saved.indexFormat=UnityEngine.Rendering.IndexFormat.UInt32;
        saved.SetVertices(vertices); saved.SetUVs(0,uv); saved.subMeshCount=2; saved.SetTriangles(rails,0); saved.SetTriangles(bed,1);
        saved.RecalculateNormals(); saved.RecalculateBounds(); EditorUtility.SetDirty(saved);
        var mf=route.GetComponent<MeshFilter>(); if(mf==null) mf=route.gameObject.AddComponent<MeshFilter>(); mf.sharedMesh=saved;
        var mr=route.GetComponent<MeshRenderer>(); if(mr==null) mr=route.gameObject.AddComponent<MeshRenderer>();
        mr.sharedMaterials=new[]{filter.GetComponent<MeshRenderer>().sharedMaterial,AssetDatabase.LoadAssetAtPath<Material>(StationAssets+"/Ballast.mat")};
        foreach(var demo in UnityEngine.Object.FindObjectsByType<StationDemo>()) if(demo.route==route)
        {
            Undo.RecordObjects(new UnityEngine.Object[]{demo.train,demo.carriage},"Align train to route");
            route.Place(demo.train,demo.startDistance); route.Place(demo.carriage,demo.startDistance-demo.carriageSpacing);
            if(demo.boardingPlatform!=null)
            {
                Undo.RecordObject(demo.boardingPlatform,"Align boarding platform to stop");
                route.Place(demo.boardingPlatform,demo.startDistance-demo.carriageSpacing);
            }
            PrefabUtility.RecordPrefabInstancePropertyModifications(demo.train); PrefabUtility.RecordPrefabInstancePropertyModifications(demo.carriage);
        }
        EditorSceneManager.MarkSceneDirty(route.gameObject.scene); AssetDatabase.SaveAssets();
    }
}

[CustomEditor(typeof(StationLoop))]
public sealed class StationLoopEditor : Editor
{
    public override void OnInspectorGUI()
    {
        using(new EditorGUI.DisabledScope(EditorApplication.isPlaying))
        {
            DrawDefaultInspector();
            if(GUILayout.Button("Rebuild rails / align train")) TakeDemo.RebuildStationTrack((StationLoop)target);
        }
        EditorGUILayout.HelpBox("Length/radius are metres at unit scale. Rebuild after editing the route or stop distance; this also aligns the boarding platform. Record a new take after scene edits.",MessageType.Info);
    }
}
