using System;
using System.IO;
using System.Collections.Generic;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;
using UnityEngine.Rendering;

// Ordinary editable geometry; the runtime uses the same session as TrainStation.
public static partial class TakeDemo
{
    [MenuItem("Tools/Acoustics/Build church")]
    public static void BuildChurch()
    {
        if(Application.isPlaying) throw new InvalidOperationException("Stop Play mode before building a scene.");
        if(!Application.isBatchMode && !EditorSceneManager.SaveCurrentModifiedScenesIfUserWantsTo()) return;
        var scene=EditorSceneManager.NewScene(NewSceneSetup.EmptyScene,NewSceneMode.Single);
        const string assets="Assets/External/Church/";
        var world=new GameObject("Church world").transform;
        var model=(GameObject)PrefabUtility.InstantiatePrefab(AssetDatabase.LoadAssetAtPath<GameObject>(assets+"Church.fbx"),world);
        var proxy=(GameObject)PrefabUtility.InstantiatePrefab(AssetDatabase.LoadAssetAtPath<GameObject>(assets+"ChurchCollision.fbx"));
        var masonry=ChurchAcoustics("Church stone",2400,.9f,.025f,.055f,.12f);
        var timber=ChurchAcoustics("Church wood",700,.6f,.12f,.3f,.25f);
        var doors=new List<SceneActor>();
        foreach(var filter in proxy.GetComponentsInChildren<MeshFilter>())
        {
            var target=model.transform.Find(filter.name.Replace("_Acoustic",""));
            if(target==null) throw new InvalidOperationException("Missing church mesh: "+filter.name);
            target.gameObject.AddComponent<MeshCollider>().sharedMesh=filter.sharedMesh;
            var surface=target.gameObject.AddComponent<AcousticSurface>(); surface.geometrySource=AcousticSurface.GeometrySource.Collider;
            surface.material=target.name.Contains("Door") && !target.name.Contains("Frame") ? timber:masonry;
            if(target.name.StartsWith("FrontDoor") && !target.name.EndsWith("Frame"))
            {
                var door=target.gameObject.AddComponent<SlidingDoor>(); door.id=target.name;
                door.interactionDistance=3.5f; door.openOffset=new Vector3(target.name.EndsWith("Left")?-2:2,0,0); doors.Add(door);
            }
        }
        UnityEngine.Object.DestroyImmediate(proxy);
        var wood=Array.Find(model.transform.Find("Bell").GetComponent<Renderer>().sharedMaterials,m=>m.name=="Wood");
        var stone=AssetDatabase.LoadAssetAtPath<Material>(StationAssets+"/Masonry.mat");
        var metal=AssetDatabase.LoadAssetAtPath<Material>(StationAssets+"/Steel.mat");
        string floorPath=assets+"Floor.mat";
        var floor=AssetDatabase.LoadAssetAtPath<Material>(floorPath);
        if(floor==null)
        {
            floor=new Material(Shader.Find("Standard")); floor.mainTexture=AssetDatabase.LoadAssetAtPath<Texture2D>(assets+"Textures/marble.jpg");
            floor.mainTextureScale=new Vector2(24,36); floor.SetFloat("_Glossiness",.3f); AssetDatabase.CreateAsset(floor,floorPath);
        }
        StationBox(world,"Stone floor and courtyard",new Vector3(0,.75f,-7),new Vector3(58,.5f,86),floor).GetComponent<AcousticSurface>().material=masonry;
        var furniture=new GameObject("Pews and sanctuary").transform; furniture.SetParent(world,false);
        foreach(float z in new[]{-26f,-22f,-18f,-14f,-10f,-6f}) foreach(float x in new[]{-4f,4f})
        {
            var pew=new GameObject("Pew").transform; pew.SetParent(furniture,false); pew.localPosition=new Vector3(x,1,z);
            StationBox(pew,"Seat",new Vector3(0,.48f,0),new Vector3(4,.16f,.65f),wood);
            StationBox(pew,"Back",new Vector3(0,.95f,-.3f),new Vector3(4,.8f,.14f),wood);
            foreach(float side in new[]{-1.7f,1.7f}) StationBox(pew,"Support",new Vector3(side,.22f,0),new Vector3(.15f,.44f,.55f),wood,false);
        }
        StationBox(furniture,"Altar",new Vector3(0,1.6f,12),new Vector3(4,1.2f,1.4f),stone).GetComponent<AcousticSurface>().material=masonry;
        StationBox(furniture,"Organ case",new Vector3(0,2,19),new Vector3(6,2,1.5f),wood);
        for(int i=-5;i<=5;i++)
        {
            float height=4-Mathf.Abs(i)*.35f;
            var pipe=GameObject.CreatePrimitive(PrimitiveType.Cylinder); pipe.name="Organ pipe"; pipe.transform.SetParent(furniture,false);
            pipe.transform.localPosition=new Vector3(i*.48f,3+height*.5f,19); pipe.transform.localScale=new Vector3(.3f,height*.5f,.3f);
            pipe.GetComponent<Renderer>().sharedMaterial=metal; UnityEngine.Object.DestroyImmediate(pipe.GetComponent<Collider>());
        }
        foreach(var surface in furniture.GetComponentsInChildren<AcousticSurface>()) if(surface.material==null) surface.material=timber;
        var host=new GameObject("Acoustic simulation"); var manager=host.AddComponent<RayTraceManager>();
        manager.monitorLive=true; manager.takeRoot=world; manager.takeDimension=AcousticTake.Dimension.Spatial3D;
        manager.sampleRate=AudioSettings.outputSampleRate; manager.takeDefaultMaterial=masonry;
        manager.directShader=AssetDatabase.LoadAssetAtPath<ComputeShader>("Assets/Shaders/Transport.compute");
        manager.directSettings.listenerOutput=TakeRender.ListenerOutput.Mono;
        manager.directSettings.transport=TakeRender.Transport.Raytraced; manager.directSettings.audioBackend=TakeRender.AudioBackend.Gpu;
        manager.directSettings.useBvh=true; manager.directSettings.raysPerBin=2; manager.directSettings.maxBounces=32;
        manager.directSettings.diffractionFactor=0; // Experimental edge estimates remain excluded from this room preset.
        manager.directSettings.responseStepSamples=1024; manager.directSettings.transitionSamples=512;
        manager.directSettings.tailSamples=3*manager.sampleRate; manager.monitorLeadSamples=16384;
        manager.audioManager=host.AddComponent<AudioManager>();
        var player=SharedPlayer(world,new Vector3(0,1,-30)); manager.takeListeners.Add(player.GetComponentInChildren<Camera>().transform);
        manager.takeObjects.Add(new RayTraceManager.TakeObjectBinding {transform=player});
        var bell=StationSource(world,"Tower bells",new Vector3(0,52,-4),AssetDatabase.LoadAssetAtPath<AudioClip>(assets+"Audio/Bell.wav"),manager,true);
        var organ=StationSource(world,"Pipe organ",new Vector3(0,5,18),AssetDatabase.LoadAssetAtPath<AudioClip>(assets+"Audio/Organ.wav"),manager,true);
        // Represent tower radiation at its open belfry; the interior bell is screened by masonry.
        bell.gain=6; organ.gain=2;
        foreach(var source in new[]{bell,organ})
        {
            if(source.clip==null) throw new FileNotFoundException("Missing church audio: "+source.name);
            PrefabUtility.RecordPrefabInstancePropertyModifications(source);
        }
        var session=host.AddComponent<SceneSession>(); session.manager=manager; session.player=player; session.actors=doors.ToArray(); session.startSounds=new[]{bell,organ};
        var panel=host.AddComponent<RaytraceControlPanel>(); panel.rayTraceManager=manager; panel.session=session;
        panel.buttonSize=new Vector2(280,32); panel.spacing=6; panel.anchoredPosition=new Vector2(24,-24);
        var sun=new GameObject("Late afternoon sun").AddComponent<Light>(); sun.type=LightType.Directional;
        sun.transform.rotation=Quaternion.Euler(35,-55,0); sun.intensity=1; sun.color=new Color(1,.91f,.76f); sun.shadows=LightShadows.Soft;
        var glow=AssetDatabase.LoadAssetAtPath<Material>(StationAssets+"/Diffuser.mat");
        foreach(float z in new[]{-24f,-12f,0f,12f})
        {
            StationBox(furniture,"Suspension",new Vector3(0,18,z),new Vector3(.035f,22,.035f),metal,false);
            StationBox(furniture,"Pendant",new Vector3(0,7,z),new Vector3(.5f,.1f,.5f),metal,false);
            StationBox(furniture,"Lantern",new Vector3(0,7.3f,z),new Vector3(.25f,.5f,.25f),glow,false);
            var light=new GameObject("Warm nave light").AddComponent<Light>(); light.transform.SetParent(world,false); light.transform.localPosition=new Vector3(0,6.9f,z);
            light.type=LightType.Point; light.range=19; light.intensity=2; light.color=new Color(1,.75f,.43f);
        }
        RenderSettings.ambientMode=AmbientMode.Trilight; RenderSettings.ambientSkyColor=new Color(.35f,.4f,.5f);
        RenderSettings.ambientEquatorColor=new Color(.22f,.24f,.29f); RenderSettings.ambientGroundColor=new Color(.12f,.1f,.08f);
        RenderSettings.fog=false; RenderSettings.skybox=AssetDatabase.LoadAssetAtPath<Material>(StationAssets+"/Sky.mat");
        manager.RegisterTakeObjects(); ConfigureMultiplayer(session);
        EditorSceneManager.SaveScene(scene,"Assets/Scenes/Church.unity"); AssetDatabase.SaveAssets();
        Debug.Log("Church saved. WASD to walk, E for entrance doors, Esc for microphone/multiplayer, R to save.");
    }

    static AudioMaterial ChurchAcoustics(string name,float density,float rigidity,float low,float high,float scatter)
    {
        string path="Assets/Materials/"+name+".asset";
        var material=AssetDatabase.LoadAssetAtPath<AudioMaterial>(path);
        if(material!=null) return material;
        material=ScriptableObject.CreateInstance<AudioMaterial>(); material.density=density; material.rigidity=rigidity;
        material.porosity=0; material.roughness=scatter;
        // Illustrative energy coefficients; no measured absorption/RT60 claim.
        material.energyCurve=new[]{new AudioMaterial.EnergyKnot {frequencyHz=125,absorption=low,scattering=scatter},
            new AudioMaterial.EnergyKnot {frequencyHz=8000,absorption=high,scattering=scatter}};
        AssetDatabase.CreateAsset(material,path); return material;
    }

    static Transform SharedPlayer(Transform parent,Vector3 position)
    {
        var prefab=AssetDatabase.LoadAssetAtPath<GameObject>("Assets/Prefabs/Player.prefab");
        if(prefab==null) throw new FileNotFoundException("Missing shared Player prefab.");
        var player=((GameObject)PrefabUtility.InstantiatePrefab(prefab,parent)).transform;
        player.localPosition=position; return player;
    }
    public static void ConfigureMultiplayer(SceneSession session,int players=4)
    {
        var manager=session.manager;
        var network=session.GetComponent<MultiplayerSession>()??session.gameObject.AddComponent<MultiplayerSession>();
        session.multiplayer=network;network.session=session;
        var mic=session.GetComponent<MicrophoneInput>()??session.gameObject.AddComponent<MicrophoneInput>();network.microphone=mic;
        if(mic.source==null)
        {
            var mouth=new GameObject("Microphone source");mouth.transform.SetParent(manager.takeListeners[0],false);
            mic.source=mouth.AddComponent<RaytracedAudioSource>();mic.source.rayTraceManager=manager;
            mic.source.streamed=true;mic.source.mutedListenerPlusOne=1;
            manager.takeSources.Add(mic.source);
            manager.takeObjects.Add(new RayTraceManager.TakeObjectBinding {id="voice-local",transform=mouth.transform});
        }
        if(network.remoteVoices.Length==0)
        {
            network.remoteVoices=new RaytracedAudioSource[players-1];
            for(int i=0;i<network.remoteVoices.Length;i++)
            {
                var avatar=new GameObject("Remote player "+(i+1));avatar.transform.SetParent(manager.takeRoot,false);
                var shape=GameObject.CreatePrimitive(PrimitiveType.Capsule);shape.name="Body";shape.transform.SetParent(avatar.transform,false);shape.transform.localPosition=Vector3.up;
                var mouth=new GameObject("Voice");mouth.transform.SetParent(avatar.transform,false);mouth.transform.localPosition=Vector3.up*1.65f;
                var voice=mouth.AddComponent<RaytracedAudioSource>();voice.rayTraceManager=manager;voice.streamed=true;
                network.remoteVoices[i]=voice;manager.takeSources.Add(voice);
                manager.takeObjects.Add(new RayTraceManager.TakeObjectBinding {id="voice-remote-"+i,transform=mouth.transform});
                manager.takeObjects.Add(new RayTraceManager.TakeObjectBinding {id="player-remote-"+i,transform=avatar.transform});
                avatar.SetActive(false);
            }
        }
        var shared=new List<Transform>();
        foreach(var actor in session.actors)
        {
            if(actor is StationDemo train) {shared.Add(train.train);shared.Add(train.carriage);}
            else if(actor is SlidingDoor door)shared.Add(door.transform);
        }
        network.sharedObjects=shared.ToArray();
        manager.takeObjects.Sort((a,b)=>StringComparer.Ordinal.Compare(a.id,b.id));
        if(network.spawnPoints.Length==0)
        {
            network.spawnPoints=new Transform[network.Capacity];
            for(int i=0;i<network.Capacity;i++)
            {
                var point=new GameObject("Player spawn "+i).transform;point.SetParent(manager.takeRoot,false);
                point.SetPositionAndRotation(session.player.position+session.player.right*(i*1.2f),session.player.rotation);
                network.spawnPoints[i]=point;
            }
        }
        FinishPlayers(session);
        PlayerSettings.macOS.microphoneUsageDescription="Use your microphone as an acoustic source and save it with recorded runs.";
    }
}
