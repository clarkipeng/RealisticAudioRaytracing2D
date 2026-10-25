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
        var details=AssetDatabase.LoadAssetAtPath<GameObject>(assets+"Details.fbx");
        GameObject Detail(string name,Transform parent,Vector3 position)
        {
            var obj=UnityEngine.Object.Instantiate(details.transform.Find(name).gameObject,parent);
            obj.name=name; obj.transform.localPosition=position; return obj;
        }
        var floor=AssetDatabase.LoadAssetAtPath<Material>(assets+"Floor.mat");
        StationBox(world,"Stone floor and courtyard",new Vector3(0,.75f,-7),new Vector3(58,.5f,86),floor).GetComponent<AcousticSurface>().material=masonry;
        foreach(string name in new[]{"Windows","Meadow"})
        {
            var obj=Detail(name,world,Vector3.zero); obj.AddComponent<MeshCollider>().sharedMesh=obj.GetComponent<MeshFilter>().sharedMesh;
            var surface=obj.AddComponent<AcousticSurface>(); surface.geometrySource=AcousticSurface.GeometrySource.Collider;
            surface.material=name=="Windows" ? ChurchAcoustics("Church glass",2500,.95f,.04f,.025f,.025f):ChurchAcoustics("Church ground",1400,.15f,.35f,.8f,.55f);
        }
        var furniture=new GameObject("Pews and sanctuary").transform; furniture.SetParent(world,false);
        foreach(float z in new[]{-26f,-22f,-18f,-14f,-10f,-6f}) foreach(float x in new[]{-4f,4f})
        {
            // Keep metre-sized collision shapes outside the imported mesh's axis/unit transform.
            var pew=new GameObject("Pew").transform; pew.SetParent(furniture,false); pew.localPosition=new Vector3(x,1,z);
            Detail("Pew",pew,Vector3.zero);
            StationProxy(pew,"Seat",new Vector3(0,.48f,0),new Vector3(4,.16f,.65f));
            StationProxy(pew,"Back",new Vector3(0,.95f,-.3f),new Vector3(4,.8f,.14f));
        }
        Detail("Altar",furniture,new Vector3(0,1,12));
        StationProxy(furniture,"Altar body",new Vector3(0,1.7f,12),new Vector3(4.5f,1.4f,1.65f));
        furniture.Find("Altar body").GetComponent<AcousticSurface>().material=masonry;
        Detail("Organ",furniture,new Vector3(0,1,19));
        StationProxy(furniture,"Organ case",new Vector3(0,2,19),new Vector3(6,2,1.5f));
        foreach(float x in new[]{-2.1f,0,2.1f})
            StationProxy(furniture,"Organ tower",new Vector3(x,x==0?5.85f:5.3f,19.42f),new Vector3(1.7f,x==0?5.3f:4.2f,.5f));
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
        var sun=new GameObject("Moonlight").AddComponent<Light>(); sun.type=LightType.Directional;
        sun.transform.rotation=Quaternion.Euler(25,-70,0); sun.intensity=.18f; sun.color=new Color(.5f,.65f,1); sun.shadows=LightShadows.Soft;
        foreach(float z in new[]{-24f,-12f,0f,12f})
        {
            Detail("Chandelier",furniture,new Vector3(0,6.5f,z));
            var light=new GameObject("Candlelight").AddComponent<Light>(); light.transform.SetParent(world,false); light.transform.localPosition=new Vector3(0,6.2f,z);
            light.type=LightType.Point; light.range=12; light.intensity=z==12?2.2f:1.5f; light.color=new Color(1,.72f,.42f);
        }
        foreach(float z in new[]{-20.32f,-6.04f})
        {
            var light=new GameObject("Window light").AddComponent<Light>(); light.transform.SetParent(world,false); light.transform.localPosition=new Vector3(9,9,z);
            light.transform.LookAt(new Vector3(-1,1,z+4)); light.type=LightType.Spot; light.spotAngle=38; light.range=25; light.intensity=1.4f;
            light.color=new Color(.36f,.55f,1); light.cookie=AssetDatabase.LoadAssetAtPath<Texture2D>(assets+"Textures/WindowLight.png");
            light.shadows=LightShadows.Soft; light.shadowResolution=LightShadowResolution.Medium; light.renderMode=LightRenderMode.ForcePixel;
        }
        RenderSettings.ambientMode=AmbientMode.Trilight; RenderSettings.ambientSkyColor=new Color(.14f,.17f,.23f);
        RenderSettings.ambientEquatorColor=new Color(.06f,.08f,.12f); RenderSettings.ambientGroundColor=new Color(.025f,.03f,.04f);
        RenderSettings.fog=true; RenderSettings.fogMode=FogMode.Linear; RenderSettings.fogStartDistance=90; RenderSettings.fogEndDistance=270; RenderSettings.fogColor=new Color(.055f,.075f,.11f);
        RenderSettings.skybox=AssetDatabase.LoadAssetAtPath<Material>(assets+"Dusk.mat");
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
