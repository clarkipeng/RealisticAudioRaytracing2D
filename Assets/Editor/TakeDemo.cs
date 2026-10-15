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
        var world=new GameObject("Church world").transform;
        var stone=AssetDatabase.LoadAssetAtPath<Material>(StationAssets+"/Masonry.mat");
        var floor=AssetDatabase.LoadAssetAtPath<Material>(StationAssets+"/Concrete.mat");
        var wood=AssetDatabase.LoadAssetAtPath<Material>(StationAssets+"/Timber.mat");
        var steel=AssetDatabase.LoadAssetAtPath<Material>(StationAssets+"/Steel.mat");
        var hall=new GameObject("Nave and aisles").transform; hall.SetParent(world,false);
        StationBox(hall,"Floor",new Vector3(3,-.25f,0),new Vector3(20,.5f,32),floor);
        StationBox(hall,"West wall",new Vector3(-6.25f,7.5f,0),new Vector3(.5f,15,30),stone);
        StationBox(hall,"Front wall",new Vector3(0,7.5f,15.25f),new Vector3(13,15,.5f),stone);
        StationBox(hall,"Rear wall",new Vector3(0,7.5f,-15.25f),new Vector3(13,15,.5f),stone);
        // The east opening leads to an adjoining room. No coincident acoustic wall behind the door.
        StationBox(hall,"East wall rear",new Vector3(6.25f,7.5f,-4.1f),new Vector3(.5f,15,21.8f),stone);
        StationBox(hall,"East wall front",new Vector3(6.25f,7.5f,12.1f),new Vector3(.5f,15,5.8f),stone);
        StationBox(hall,"Door lintel",new Vector3(6.25f,8.9f,8),new Vector3(.5f,12.2f,2.4f),stone);
        foreach(float x in new[]{-3.8f,3.8f}) foreach(float z in new[]{-9f,-3f,3f,9f})
        {
            StationBox(hall,"Column",new Vector3(x,5,z),new Vector3(.65f,10,.65f),stone);
            StationBox(hall,"Column capital",new Vector3(x,10.15f,z),new Vector3(1,.3f,1),stone);
        }
        foreach(float x in new[]{-3.8f,3.8f})
            StationBox(hall,"Arcade beam",new Vector3(x,10.6f,0),new Vector3(.7f,.6f,30),stone);
        foreach(float z in new[]{-8f,-5f,-2f,1f,4f}) foreach(float x in new[]{-2f,2f})
        {
            var pew=new GameObject("Pew").transform; pew.SetParent(hall,false); pew.localPosition=new Vector3(x,0,z);
            StationBox(pew,"Seat",new Vector3(0,.48f,0),new Vector3(2.5f,.16f,.6f),wood);
            StationBox(pew,"Back",new Vector3(0,.95f,-.28f),new Vector3(2.5f,.8f,.12f),wood);
            foreach(float side in new[]{-1f,1f}) StationBox(pew,"Support",new Vector3(side,.22f,0),new Vector3(.12f,.44f,.5f),wood,false);
        }
        var room=new GameObject("Side room").transform; room.SetParent(world,false);
        StationBox(room,"East wall",new Vector3(12.25f,2,8),new Vector3(.5f,4,6.5f),stone);
        foreach(float z in new[]{4.75f,11.25f}) StationBox(room,"End wall",new Vector3(9.25f,2,z),new Vector3(6.5f,4,.5f),stone);
        StationBox(room,"Roof",new Vector3(9.25f,4.25f,8),new Vector3(6.5f,.5f,7),stone);
        var door=StationBox(world,"Side room door",new Vector3(6.25f,1.4f,8),new Vector3(.2f,2.8f,2.4f),wood).AddComponent<SlidingDoor>();
        door.id="side-room-door"; door.openOffset=new Vector3(0,0,-2.6f);
        var host=new GameObject("Acoustic simulation");
        var manager=host.AddComponent<RayTraceManager>();
        manager.monitorLive=true; manager.takeRoot=world; manager.takeDimension=AcousticTake.Dimension.Spatial3D;
        manager.sampleRate=AudioSettings.outputSampleRate;
        manager.takeDefaultMaterial=AssetDatabase.LoadAssetAtPath<AudioMaterial>("Assets/Materials/Border.asset");
        manager.directShader=AssetDatabase.LoadAssetAtPath<ComputeShader>("Assets/Shaders/Transport.compute");

        manager.directSettings.listenerOutput=TakeRender.ListenerOutput.Mono;
        manager.directSettings.transport=TakeRender.Transport.Raytraced;
        manager.directSettings.audioBackend=TakeRender.AudioBackend.Gpu; manager.directSettings.useBvh=true;
        manager.directSettings.responseStepSamples=1024; manager.directSettings.transitionSamples=512;
        manager.directSettings.tailSamples=3*manager.sampleRate; manager.monitorLeadSamples=16384;
        manager.audioManager=host.AddComponent<AudioManager>();
        var player=SharedPlayer(world,new Vector3(0,0,-12.5f));
        var head=player.GetComponentInChildren<Camera>().transform;
        var speaker=StationSource(world,"Test speaker",new Vector3(0,1.6f,12),AssetDatabase.LoadAssetAtPath<AudioClip>("Assets/Audio/station_signal.wav"),manager,true);
        StationBox(speaker.transform,"Housing",new Vector3(0,0,.25f),new Vector3(.55f,.8f,.4f),steel,false);
        manager.takeListeners.Add(head);
        manager.takeObjects.Add(new RayTraceManager.TakeObjectBinding {transform=player});
        var session=host.AddComponent<SceneSession>(); session.manager=manager; session.player=player;
        session.actors=new SceneActor[]{door}; session.startSounds=new[]{speaker};
        var panel=host.AddComponent<RaytraceControlPanel>(); panel.rayTraceManager=manager;
        panel.session=session; panel.buttonSize=new Vector2(280,32); panel.spacing=6; panel.anchoredPosition=new Vector2(24,-24);
        var sun=new GameObject("Daylight").AddComponent<Light>(); sun.type=LightType.Directional;
        sun.transform.rotation=Quaternion.Euler(55,-25,0); sun.intensity=.35f; sun.shadows=LightShadows.Soft;
        foreach(float z in new[]{-9f,1f,11f})
        {
            var lamp=new GameObject("Nave light").AddComponent<Light>(); lamp.transform.position=new Vector3(0,7,z);
            lamp.type=LightType.Point; lamp.range=15; lamp.intensity=1.1f; lamp.color=new Color(1,.9f,.75f);
        }
        var roomLight=new GameObject("Side room light").AddComponent<Light>(); roomLight.transform.position=new Vector3(9,3,8);
        roomLight.range=7; roomLight.intensity=1.1f; roomLight.color=new Color(1,.92f,.8f);
        RenderSettings.ambientMode=AmbientMode.Flat; RenderSettings.ambientLight=new Color(.22f,.24f,.28f); RenderSettings.fog=false;
        FinishChurch(session);
        manager.RegisterTakeObjects();
        ConfigureMultiplayer(session);
        EditorSceneManager.SaveScene(scene,"Assets/Scenes/Church.unity"); AssetDatabase.SaveAssets();
        if(SceneView.lastActiveSceneView!=null) SceneView.lastActiveSceneView.LookAt(new Vector3(0,3,0),Quaternion.Euler(20,35,0),24);
        Selection.activeGameObject=host;
        Debug.Log("Church saved. Play to walk; Esc for microphone and multiplayer; R saves. Acoustic materials are illustrative, not measured church data.");
    }

    // Shared by the builder and scene authoring; all geometry remains ordinary editable objects.
    public static void FinishChurch(SceneSession session)
    {
        var manager=session.manager; var world=manager.takeRoot; var hall=world.Find("Nave and aisles");
        foreach(string name in new[]{"Ceiling","Vault and sanctuary"})
        {
            var previous=hall.Find(name); if(previous!=null) UnityEngine.Object.DestroyImmediate(previous.gameObject);
        }
        var stone=AssetDatabase.LoadAssetAtPath<Material>(StationAssets+"/Masonry.mat");
        var wood=AssetDatabase.LoadAssetAtPath<Material>(StationAssets+"/Timber.mat");
        var glass=AssetDatabase.LoadAssetAtPath<Material>(StationAssets+"/Diffuser.mat");
        var details=new GameObject("Vault and sanctuary").transform; details.SetParent(hall,false);
        foreach(float side in new[]{-1f,1f})
        {
            var roof=StationBox(details,"Vault",new Vector3(side*3,13.2f,0),new Vector3(6.96f,.4f,31),stone);
            roof.transform.localRotation=Quaternion.Euler(0,0,-side*26.565f);
            foreach(float z in new[]{-9f,-3f,3f,9f})
            {
                var rib=StationBox(details,"Vault rib",new Vector3(side*3,12.91f,z),new Vector3(6.96f,.2f,.28f),stone,false);
                rib.transform.localRotation=roof.transform.localRotation;
                // Inset luminous panels sit on the sealed walls; they are visual details, not holes.
                StationBox(details,"Clerestory window",new Vector3(side*5.985f,8,z),new Vector3(.025f,4,1.5f),glass,false);
                StationBox(details,"Window mullion",new Vector3(side*5.95f,8,z),new Vector3(.07f,4,.09f),wood,false);
                StationBox(details,"Window transom",new Vector3(side*5.95f,7,z),new Vector3(.07f,.09f,1.5f),wood,false);
            }
        }
        StationBox(details,"Altar",new Vector3(0,.6f,12.8f),new Vector3(3,1.2f,1),stone);
        StationBox(details,"Altar top",new Vector3(0,1.27f,12.8f),new Vector3(3.3f,.14f,1.3f),stone,false);
        StationBox(details,"Cross upright",new Vector3(0,6,14.95f),new Vector3(.22f,3.8f,.12f),wood,false);
        StationBox(details,"Cross arm",new Vector3(0,6.8f,14.95f),new Vector3(2,.22f,.12f),wood,false);
        // Thin inlays have no colliders or acoustic surfaces: the existing floor owns both.
        foreach(float x in new[]{-.65f,.65f})
            StationBox(details,"Aisle border",new Vector3(x,.006f,0),new Vector3(.04f,.012f,29),stone,false);
        for(int z=-14;z<15;z+=2)
            StationBox(details,"Floor joint",new Vector3(0,.003f,z),new Vector3(12,.006f,.025f),wood,false);
        var masonry=ChurchAcoustics("Church stone",2400,.9f,.025f,.055f,.12f);
        var timber=ChurchAcoustics("Church wood",700,.6f,.12f,.3f,.25f);
        manager.takeDefaultMaterial=masonry;
        foreach(var surface in world.GetComponentsInChildren<AcousticSurface>(true))
            surface.material=surface.GetComponent<Renderer>()?.sharedMaterial==wood ? timber:masonry;
        manager.directSettings.raysPerBin=2; manager.directSettings.maxBounces=32;
        manager.directSettings.diffractionFactor=0; // Keep the experimental edge estimator out of this reflective-room preset.
        manager.directSettings.listenerOutput=TakeRender.ListenerOutput.Mono;
        var speaker=session.startSounds[0]; speaker.name="Clap source";
        speaker.transform.localPosition=new Vector3(0,1.8f,12.8f);
        speaker.clip=AssetDatabase.LoadAssetAtPath<AudioClip>("Assets/Audio/church_clap.wav"); speaker.gain=2;
        if(speaker.clip==null) throw new FileNotFoundException("Missing dry church clap.");
        speaker.rayTraceManager=manager; speaker.loop=true;
        PrefabUtility.RecordPrefabInstancePropertyModifications(speaker);
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
