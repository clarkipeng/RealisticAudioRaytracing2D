using System;
using System.IO;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;
using UnityEngine.Rendering;

public static partial class TakeDemo
{
    const string StationAssets="Assets/Materials/Station";
    const string Models="Assets/External/KenneyTrainKit/Models/FBX format/";

    // Rebuild only on request. The resulting scene consists of ordinary editable Unity objects.
    [MenuItem("Tools/Acoustics/Build train station")]
    public static void BuildStation()
    {
        if(Application.isPlaying) throw new InvalidOperationException("Stop Play mode before building a scene.");
        if(!Application.isBatchMode && !EditorSceneManager.SaveCurrentModifiedScenesIfUserWantsTo()) return;
        Directory.CreateDirectory(StationAssets);
        Directory.CreateDirectory("Assets/Prefabs/Station");
        AssetDatabase.Refresh();
        var concrete=StationMaterial("Concrete",new Color(.59f,.63f,.64f));
        var pale=StationMaterial("Masonry",new Color(.83f,.82f,.73f));
        var steel=StationMaterial("Steel",new Color(.055f,.12f,.17f),.2f);
        StationMaterial("Safety yellow",new Color(.96f,.66f,.12f));
        var teal=StationMaterial("Wayfinding",new Color(.04f,.4f,.43f));
        var timber=StationMaterial("Timber",new Color(.54f,.3f,.14f));
        StationMaterial("Ballast",new Color(.27f,.3f,.29f));
        var landscape=StationMaterial("Ground",new Color(.32f,.4f,.32f));
        var lightMat=StationMaterial("Diffuser",new Color(.93f,.95f,.88f));
        lightMat.EnableKeyword("_EMISSION"); lightMat.SetColor("_EmissionColor",new Color(.5f,.55f,.5f));
        EditorUtility.SetDirty(lightMat);
        var scene=EditorSceneManager.NewScene(NewSceneSetup.EmptyScene,NewSceneMode.Single);
        var world=new GameObject("Station world").transform;
        var architecture=new GameObject("Station architecture").transform; architecture.SetParent(world);
        StationBox(architecture,"Terrain",new Vector3(0,-.55f,0),new Vector3(70,.5f,130),landscape);
        StationBox(architecture,"Platform",new Vector3(3.8f,-.3f,0),new Vector3(12,.6f,44),concrete);
        var shelter=new GameObject("Covered platform").transform; shelter.SetParent(architecture);
        foreach(float x in new[]{.2f,7.4f}) foreach(float z in new[]{-11f,-2f})
        {
            StationBox(shelter,"Column",new Vector3(x,1.8f,z),new Vector3(.22f,3.6f,.22f),steel);
            StationBox(shelter,"Column base",new Vector3(x,.12f,z),new Vector3(.48f,.24f,.48f),concrete,false);
        }
        StationBox(shelter,"Canopy",new Vector3(3.8f,3.7f,-6.5f),new Vector3(9,.24f,11),pale);
        foreach(float x in new[]{-.55f,8.15f}) StationBox(shelter,"Fascia",new Vector3(x,3.75f,-6.5f),new Vector3(.22f,.5f,11),steel,false);
        foreach(float z in new[]{-11f,-2f}) StationBox(shelter,"Cross beam",new Vector3(3.8f,3.46f,z),new Vector3(8,.2f,.2f),steel,false);
        foreach(float z in new[]{-9f,-4f})
        {
            StationBox(shelter,"Ceiling strip",new Vector3(3.8f,3.56f,z),new Vector3(2.6f,.055f,.12f),lightMat,false);
            StationBench(shelter,new Vector3(6.8f,0,z),timber,steel);
        }
        StationSign(shelter,"NORTHLINE",new Vector3(3.8f,2.95f,-11.7f),new Vector2(5,.65f),steel,Color.white);

        var room=new GameObject("Waiting room").transform; room.SetParent(architecture);
        StationBox(room,"West wall",new Vector3(1.8f,1.7f,10),new Vector3(.24f,3.4f,9),pale);
        StationBox(room,"East wall",new Vector3(9.8f,1.7f,10),new Vector3(.24f,3.4f,9),pale);
        StationBox(room,"Rear wall",new Vector3(5.8f,1.7f,14.5f),new Vector3(8,3.4f,.24f),pale);
        StationBox(room,"Front left",new Vector3(3.2f,1.7f,5.5f),new Vector3(2.8f,3.4f,.24f),pale);
        StationBox(room,"Front right",new Vector3(8.2f,1.7f,5.5f),new Vector3(3.2f,3.4f,.24f),pale);
        StationBox(room,"Door header",new Vector3(5.6f,3,5.5f),new Vector3(2,.8f,.24f),pale);
        StationBox(room,"Roof",new Vector3(5.8f,3.6f,10),new Vector3(8.8f,.4f,9.8f),steel);
        StationBox(room,"Front lower trim",new Vector3(3.2f,.3f,5.35f),new Vector3(2.8f,.6f,.05f),steel,false);
        StationBox(room,"Front lower trim",new Vector3(8.2f,.3f,5.35f),new Vector3(3.2f,.6f,.05f),steel,false);
        StationBox(room,"Door track",new Vector3(6.7f,2.72f,5.23f),new Vector3(4.5f,.13f,.16f),steel,false);
        StationBox(room,"Room flooring",new Vector3(5.8f,.015f,10),new Vector3(7.76f,.03f,8.76f),timber,false);
        StationSign(room,"WAITING ROOM",new Vector3(3.25f,2.35f,5.345f),new Vector2(2.3f,.52f),teal,Color.white);
        StationBench(room,new Vector3(8.8f,.03f,11.5f),timber,steel);
        foreach(float z in new[]{8f,12f})
        {
            StationBox(room,"Ceiling strip",new Vector3(5.8f,3.38f,z),new Vector3(3,.06f,.16f),lightMat,false);
            var lamp=new GameObject("Room light").AddComponent<Light>(); lamp.transform.SetParent(room);
            lamp.transform.position=new Vector3(5.8f,2.9f,z); lamp.type=LightType.Point; lamp.range=6; lamp.intensity=1.6f; lamp.color=new Color(1,.91f,.76f);
        }
        StationBox(room,"Speaker housing",new Vector3(5.6f,2.1f,14.17f),new Vector3(.55f,.75f,.3f),steel);
        foreach(float z in new[]{-16f,1f,18f})
        {
            StationBox(architecture,"Platform lamp post",new Vector3(9.3f,2.6f,z),new Vector3(.16f,5.2f,.16f),steel);
            StationBox(architecture,"Lamp head",new Vector3(8.95f,5.15f,z),new Vector3(.9f,.12f,.3f),lightMat,false);
        }
        PrefabUtility.SaveAsPrefabAssetAndConnect(architecture.gameObject,"Assets/Prefabs/Station/Platform.prefab",InteractionMode.AutomatedAction);

        var door=StationBox(world,"Sliding door",new Vector3(5.6f,1.3f,5.4f),new Vector3(2,2.6f,.16f),teal);
        StationBox(door.transform,"Handle",new Vector3(-.35f,-.06f,-.75f),new Vector3(.025f,.19f,.5f),pale,false);
        // Keep a unit root so children/proxy dimensions remain easy to edit.
        var train=new GameObject("Moving train").transform; train.SetParent(world); train.position=new Vector3(-5,.02f,-14); // Wheel bottoms meet the rail heads at y=.02.
        StationModel(train,"train-diesel-c",Vector3.zero,2.6f);
        StationProxy(train,"Body acoustic proxy",new Vector3(0,1.25f,0),new Vector3(2.55f,1.8f,6.1f));
        StationProxy(train,"Cab acoustic proxy",new Vector3(0,2.95f,-1.7f),new Vector3(2.55f,1.1f,1.9f));
        StationModel(train,"train-carriage-box",new Vector3(0,0,-7.2f),2.6f);
        StationProxy(train,"Carriage acoustic proxy",new Vector3(0,1.92f,-7.2f),new Vector3(3.05f,2.6f,6.9f));
        PrefabUtility.SaveAsPrefabAssetAndConnect(train.gameObject,"Assets/Prefabs/Station/Train.prefab",InteractionMode.AutomatedAction);

        var host=new GameObject("Acoustic simulation");
        var manager=host.AddComponent<RayTraceManager>();
        manager.monitorLive=true; manager.takeRoot=world; manager.takeDimension=AcousticTake.Dimension.Spatial3D;
        manager.sampleRate=AudioSettings.outputSampleRate;
        manager.takeDefaultMaterial=AssetDatabase.LoadAssetAtPath<AudioMaterial>("Assets/Materials/Material.asset");
        manager.directShader=AssetDatabase.LoadAssetAtPath<ComputeShader>("Assets/Shaders/Transport.compute");

        manager.directSettings.listenerOutput=TakeRender.ListenerOutput.Mono;
        manager.directSettings.transport=TakeRender.Transport.Raytraced; manager.directSettings.useBvh=true;
        manager.directSettings.responseStepSamples=1024; manager.directSettings.transitionSamples=512;
        manager.directSettings.tailSamples=manager.sampleRate;
        manager.monitorLeadSamples=16384;
        manager.audioManager=host.AddComponent<AudioManager>();
        var horn=StationSource(train,"Train horn",new Vector3(0,4,1.8f),AssetDatabase.LoadAssetAtPath<AudioClip>("Assets/Audio/train_horn.mp3"),manager,true);
        var signal=StationSource(world,"Room signal",new Vector3(5.6f,2.1f,13.98f),StationSignal(),manager,true);
        var player=SharedPlayer(world,new Vector3(4,0,-20));
        var camera=player.GetComponentInChildren<Camera>();

        manager.takeListeners.Add(camera.transform);
        manager.takeObjects.Add(new RayTraceManager.TakeObjectBinding{transform=player});
        manager.takeObjects.Add(new RayTraceManager.TakeObjectBinding{transform=train});
        var session=host.AddComponent<SceneSession>(); session.manager=manager; session.player=player; session.startSounds=new[]{horn,signal};
        var demo=host.AddComponent<StationDemo>(); demo.id="train"; demo.train=train;
        var sliding=door.AddComponent<SlidingDoor>(); sliding.id="waiting-room-door";
        session.actors=new SceneActor[]{demo,sliding};
        ConfigureStationLoop(demo,session); manager.RegisterTakeObjects();
        // SceneSession checks replay mode before writing motion and keeps UI/escape available.
        var panel=host.AddComponent<RaytraceControlPanel>(); panel.rayTraceManager=manager; panel.session=session; panel.buttonSize=new Vector2(280,32); panel.spacing=6; panel.anchoredPosition=new Vector2(24,-24);
        var sun=new GameObject("Sun").AddComponent<Light>(); sun.type=LightType.Directional; sun.transform.rotation=Quaternion.Euler(42,-32,0); sun.intensity=1.2f; sun.color=new Color(1,.94f,.83f); sun.shadows=LightShadows.Soft;
        var sky=AssetDatabase.LoadAssetAtPath<Material>(StationAssets+"/Sky.mat");
        if(sky==null) {sky=new Material(Shader.Find("Skybox/Procedural")); AssetDatabase.CreateAsset(sky,StationAssets+"/Sky.mat");}
        sky.SetFloat("_AtmosphereThickness",.85f); sky.SetFloat("_Exposure",1.15f); EditorUtility.SetDirty(sky);
        RenderSettings.skybox=sky; RenderSettings.sun=sun; RenderSettings.ambientMode=AmbientMode.Trilight;
        RenderSettings.ambientSkyColor=new Color(.62f,.72f,.8f); RenderSettings.ambientEquatorColor=new Color(.44f,.46f,.44f); RenderSettings.ambientGroundColor=new Color(.27f,.28f,.25f);
        RenderSettings.fog=true; RenderSettings.fogColor=new Color(.68f,.75f,.78f); RenderSettings.fogMode=FogMode.Linear; RenderSettings.fogStartDistance=65; RenderSettings.fogEndDistance=170;
        AssetDatabase.SaveAssets();
        ConfigureMultiplayer(session);
        FinishStation(session,demo);
        AssetDatabase.SaveAssets();
        EditorSceneManager.SaveScene(scene,"Assets/Scenes/TrainStation.unity");
        if(SceneView.lastActiveSceneView!=null) SceneView.lastActiveSceneView.LookAt(new Vector3(0,1,0),Quaternion.Euler(35,35,0),38);
        Selection.activeGameObject=host;
        Debug.Log("TrainStation ready: two sources, one listener, editable acoustic boxes, sliding door and looping train. Play, then right click to walk; Esc returns to the controls.");
    }

    [MenuItem("Tools/Acoustics/Polish train station")]
    public static void PolishStation()
    {
        if(EditorApplication.isPlaying) throw new InvalidOperationException("Stop Play mode before editing the station.");
        var demo=UnityEngine.Object.FindAnyObjectByType<StationDemo>();
        if(demo==null) throw new InvalidOperationException("Open TrainStation first.");
        var session=demo.GetComponent<SceneSession>();
        Undo.RegisterFullObjectHierarchyUndo(session.manager.takeRoot.gameObject,"Polish station");
        FinishStation(session,demo);
        EditorSceneManager.MarkSceneDirty(demo.gameObject.scene);
        // Leave saving to the editor, as with other scene edits.
        AssetDatabase.SaveAssets();
    }
    static void FinishStation(SceneSession session,StationDemo demo)
    {
        var concrete=StationMaterial("Concrete",new Color(.59f,.62f,.60f));
        var pale=StationMaterial("Masonry",new Color(.82f,.8f,.7f));
        var steel=StationMaterial("Steel",new Color(.085f,.15f,.18f),.35f);
        var teal=StationMaterial("Wayfinding",new Color(.08f,.37f,.37f),.1f);
        var wood=StationMaterial("Timber",new Color(.46f,.27f,.15f));
        var yellow=StationMaterial("Safety yellow",new Color(.95f,.72f,.24f));
        StationMaterial("Ground",new Color(.31f,.38f,.29f));
        StationMaterial("Ballast",new Color(.26f,.28f,.27f));
        var sky=AssetDatabase.LoadAssetAtPath<Material>(StationAssets+"/Sky.mat");
        Undo.RecordObject(sky,"Edit station sky");sky.SetColor("_GroundColor",new Color(.68f,.75f,.78f));EditorUtility.SetDirty(sky);
        var architecture=session.manager.takeRoot.Find("Station architecture");
        architecture.Find("Terrain").localScale=new Vector3(400,.5f,400); // Ground reaches past the fog horizon, with matching collision/acoustics.
        var details=FinishRoot(architecture);
        // Surface trim is visual only: it does not add simulation geometry or colliders.
        StationBox(details,"Platform edge trim",new Vector3(-2.215f,-.13f,0),new Vector3(.05f,.2f,44),steel,false);
        StationBox(details,"Platform edge trim",new Vector3(9.815f,-.13f,0),new Vector3(.05f,.2f,44),steel,false);
        StationBox(details,"Platform coping",new Vector3(3.8f,.008f,-21.6f),new Vector3(11.8f,.016f,.3f),pale,false);
        StationBox(details,"Canopy lower trim",new Vector3(3.8f,3.64f,-12.02f),new Vector3(8.7f,.12f,.08f),teal,false);
        StationBox(details,"Canopy lower trim",new Vector3(3.8f,3.64f,-.98f),new Vector3(8.7f,.12f,.08f),teal,false);
        foreach(float x in new[]{1.75f,9.85f})
        {
            StationBox(details,"Waiting room plinth",new Vector3(x,.16f,10),new Vector3(.34f,.32f,9.2f),steel,false);
            StationBox(details,"Waiting room cornice",new Vector3(x,3.35f,10),new Vector3(.34f,.12f,9.2f),teal,false);
        }
        foreach(float x in new[]{4.54f,6.66f}) StationBox(details,"Door jamb",new Vector3(x,1.3f,5.17f),new Vector3(.08f,2.6f,.1f),steel,false);
        StationBox(details,"Door threshold",new Vector3(5.6f,.016f,5.3f),new Vector3(2.1f,.032f,.65f),steel,false);
        for(int i=0;i<5;i++) StationBox(details,"Speaker grille",new Vector3(5.6f,1.86f+i*.12f,14.012f),new Vector3(.42f,.026f,.025f),pale,false);
        var door=architecture.parent.Find("Sliding door");
        var doorDetails=FinishRoot(door);
        StationBox(doorDetails,"Door inset",new Vector3(0,.12f,-.515f),new Vector3(.84f,.55f,.03f),steel,false);
        StationSign(details,"TRAIN  >",new Vector3(7.4f,2.65f,-11.85f),new Vector2(1.8f,.45f),teal,Color.white);

        var boarding=FinishRoot(demo.boardingPlatform);
        StationBox(boarding,"Approach paving",new Vector3(-7.9f,-.01f,0),new Vector3(2,.02f,2.4f),concrete,false);
        StationBox(boarding,"Landing trim",new Vector3(-3.89f,.6f,-2.6f),new Vector3(.04f,1.2f,2.8f),steel,false);
        StationBox(boarding,"Landing trim",new Vector3(-3.89f,.6f,2.6f),new Vector3(.04f,1.2f,2.8f),steel,false);
        StationSign(boarding,"NORTHLINE",new Vector3(-2.72f,1.8f,-3.87f),new Vector2(1.95f,.4f),teal,Color.white);
        var carriage=FinishRoot(demo.carriage);
        foreach(float x in new[]{-1.5f,1.5f})
            StationBox(carriage,"Roof fascia",new Vector3(x,3.65f,0),new Vector3(.1f,.25f,6.94f),steel,false);
        foreach(float z in new[]{-3.42f,3.42f})
            StationBox(carriage,"Roof fascia",new Vector3(0,3.65f,z),new Vector3(3.1f,.25f,.1f),steel,false);
        StationBox(carriage,"Outer waist trim",new Vector3(1.468f,2.15f,0),new Vector3(.022f,.08f,6.4f),pale,false);
        foreach(float z in new[]{-2f,2f})
            StationBox(carriage,"Boarding waist trim",new Vector3(-1.468f,2.15f,z),new Vector3(.022f,.08f,2.4f),pale,false);
        StationBox(carriage,"Entry threshold",new Vector3(-1.32f,1.215f,0),new Vector3(.2f,.026f,1.52f),yellow,false);
        // Reuse the existing avatar collider; the body and face make heading legible.
        FinishPlayers(session);
        var panel=session.GetComponent<RaytraceControlPanel>(); panel.buttonSize=new Vector2(280,32);panel.spacing=6;
        EditorUtility.SetDirty(panel);
    }
    static Transform FinishRoot(Transform parent)
    {
        var existing=parent.Find("Visual details");
        if(existing!=null) UnityEngine.Object.DestroyImmediate(existing.gameObject);
        var root=new GameObject("Visual details").transform; root.SetParent(parent,false); return root;
    }
    static void FinishPlayers(SceneSession session)
    {
        if(session.multiplayer==null)return;
        var cloth=StationMaterial("Player jacket",new Color(.1f,.43f,.46f));
        var face=StationMaterial("Player face",new Color(.85f,.72f,.55f));
        var dark=AssetDatabase.LoadAssetAtPath<Material>(StationAssets+"/Steel.mat");
        foreach(var voice in session.multiplayer.remoteVoices)
        {
            var avatar=voice.transform.parent;
            var body=avatar.Find("Body");body.GetComponent<Renderer>().enabled=false;
            var collider=body.GetComponent<CapsuleCollider>();collider.radius=.3f;collider.height=1.8f;collider.center=Vector3.down*.1f;
            var details=FinishRoot(avatar);
            StationBox(details,"Jacket",new Vector3(0,1.12f,0),new Vector3(.48f,.76f,.3f),cloth,false);
            foreach(float x in new[]{-.15f,.15f})
                StationBox(details,"Leg",new Vector3(x,.37f,0),new Vector3(.21f,.74f,.26f),dark,false);
            var head=FinishRoot(voice.transform); // Voice already follows the received/replayed head orientation.
            StationBox(head,"Head",Vector3.zero,new Vector3(.32f,.38f,.32f),face,false);
            StationBox(head,"Face",new Vector3(0,.02f,.167f),new Vector3(.23f,.065f,.025f),dark,false);
            foreach(float x in new[]{-.32f,.32f})
                StationBox(details,"Sleeve",new Vector3(x,1.13f,0),new Vector3(.14f,.52f,.27f),cloth,false);
        }
    }

    static Material StationMaterial(string name,Color color,float metal=0)
    {
        string path=StationAssets+"/"+name+".mat"; var material=AssetDatabase.LoadAssetAtPath<Material>(path);
        if(material==null) {material=new Material(Shader.Find("Standard")); AssetDatabase.CreateAsset(material,path);}
        Undo.RecordObject(material,"Edit station material");
        material.color=color; material.SetFloat("_Metallic",metal); material.SetFloat("_Glossiness",.25f); EditorUtility.SetDirty(material); return material;
    }
    static GameObject StationBox(Transform parent,string name,Vector3 position,Vector3 size,Material material,bool acoustic=true)
    {
        var obj=GameObject.CreatePrimitive(PrimitiveType.Cube); obj.name=name; obj.transform.SetParent(parent,false); obj.transform.localPosition=position; obj.transform.localScale=size;
        obj.GetComponent<Renderer>().sharedMaterial=material;
        if(acoustic) obj.AddComponent<AcousticSurface>().geometrySource=AcousticSurface.GeometrySource.Collider;
        else UnityEngine.Object.DestroyImmediate(obj.GetComponent<Collider>());
        return obj;
    }
    static void StationProxy(Transform parent,string name,Vector3 center,Vector3 size)
    {
        var obj=new GameObject(name); obj.transform.SetParent(parent,false); obj.transform.localPosition=center; obj.AddComponent<BoxCollider>().size=size;
        obj.AddComponent<AcousticSurface>().geometrySource=AcousticSurface.GeometrySource.Collider;
    }
    static void StationModel(Transform parent,string model,Vector3 position,float scale)
    {
        var prefab=AssetDatabase.LoadAssetAtPath<GameObject>(Models+model+".fbx"); if(prefab==null) throw new FileNotFoundException(model);
        var obj=(GameObject)PrefabUtility.InstantiatePrefab(prefab,parent); obj.transform.localPosition=position; obj.transform.localRotation=Quaternion.identity; obj.transform.localScale=Vector3.one*scale;
    }
    static void StationBench(Transform parent,Vector3 position,Material wood,Material steel)
    {
        var root=new GameObject("Bench").transform; root.SetParent(parent,false); root.localPosition=position;
        StationBox(root,"Seat",new Vector3(0,.47f,0),new Vector3(.62f,.12f,2),wood);
        StationBox(root,"Back",new Vector3(.25f,.82f,0),new Vector3(.1f,.65f,2),wood);
        foreach(float z in new[]{-.7f,.7f}) StationBox(root,"Leg",new Vector3(0,.22f,z),new Vector3(.45f,.44f,.08f),steel,false);
    }
    static void StationSign(Transform parent,string text,Vector3 position,Vector2 size,Material background,Color ink,float yaw=0)
    {
        var root=new GameObject(text).transform; root.SetParent(parent,false); root.localPosition=position; root.localRotation=Quaternion.Euler(0,yaw,0);
        StationBox(root,"Sign panel",Vector3.zero,new Vector3(size.x,size.y,.045f),background,false);
        var label=new GameObject("Lettering").AddComponent<TextMesh>(); label.transform.SetParent(root,false); label.transform.localPosition=new Vector3(0,0,-.03f);
        label.text=text; label.fontSize=72; label.characterSize=Mathf.Min(size.y*.5f,size.x/(text.Length*.52f))/7.2f; label.anchor=TextAnchor.MiddleCenter; label.alignment=TextAlignment.Center; label.color=ink;
        var renderer=label.GetComponent<MeshRenderer>();
        string path=StationAssets+"/Lettering.mat";
        var material=AssetDatabase.LoadAssetAtPath<Material>(path);
        if(material==null) {material=new Material(Shader.Find("Acoustics/World text")); AssetDatabase.CreateAsset(material,path);}
        material.mainTexture=renderer.sharedMaterial.mainTexture; renderer.sharedMaterial=material; EditorUtility.SetDirty(material);
    }
    static RaytracedAudioSource StationSource(Transform parent,string name,Vector3 position,AudioClip clip,RayTraceManager manager,bool loop)
    {
        var prefab=AssetDatabase.LoadAssetAtPath<GameObject>("Assets/Prefabs/TestSpeaker.prefab");
        var obj=(GameObject)PrefabUtility.InstantiatePrefab(prefab,parent); obj.name=name; obj.transform.localPosition=position;
        var source=obj.GetComponent<RaytracedAudioSource>(); source.rayTraceManager=manager; source.clip=clip; source.loop=loop; return source;
    }
    static AudioClip StationSignal()
    {
        const string path="Assets/Audio/station_signal.wav";
        // Original deterministic four-second diagnostic chime, not third-party speech/music.
        const int rate=48000,count=4*rate;
        double[] tones={440,554.365,659.255,880};
        using(var writer=new BinaryWriter(File.Create(path)))
        {
            writer.Write(System.Text.Encoding.ASCII.GetBytes("RIFF")); writer.Write(36+count*2); writer.Write(System.Text.Encoding.ASCII.GetBytes("WAVEfmt "));
            writer.Write(16); writer.Write((short)1); writer.Write((short)1); writer.Write(rate); writer.Write(rate*2); writer.Write((short)2); writer.Write((short)16);
            writer.Write(System.Text.Encoding.ASCII.GetBytes("data")); writer.Write(count*2);
            for(int i=0;i<count;i++)
            {
                double t=(double)i/rate,beat=t%1; double hz=tones[(int)t];
                double envelope=Math.Min(1,beat/.015)*Math.Exp(-beat*4)*Math.Min(1,(1-beat)/.03);
                double sample=.25*envelope*(Math.Sin(2*Math.PI*hz*t)+.25*Math.Sin(2*Math.PI*hz*2*t)); writer.Write((short)(sample*32767));
            }
        }
        AssetDatabase.ImportAsset(path,ImportAssetOptions.ForceSynchronousImport);
        var importer=(AudioImporter)AssetImporter.GetAtPath(path); var settings=importer.defaultSampleSettings; settings.loadType=AudioClipLoadType.DecompressOnLoad; settings.compressionFormat=AudioCompressionFormat.PCM; settings.preloadAudioData=true; importer.defaultSampleSettings=settings; importer.SaveAndReimport();
        return AssetDatabase.LoadAssetAtPath<AudioClip>(path);
    }
}
