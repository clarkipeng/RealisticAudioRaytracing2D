using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;

// Builds an ordinary editable Unity scene using the project's existing components/assets.
public static partial class TakeDemo
{
    [MenuItem("Tools/Acoustics/Create 3D walkthrough")]
    public static void Create()
    {
        if (!Application.isBatchMode && !EditorSceneManager.SaveCurrentModifiedScenesIfUserWantsTo()) return;
        EditorSceneManager.NewScene(NewSceneSetup.EmptyScene,NewSceneMode.Single);
        var root=new GameObject("Acoustic scene");
        var host=new GameObject("Acoustic simulation");
        var manager=host.AddComponent<RayTraceManager>();
        manager.useTakeRenderer=true; manager.takeRoot=root.transform;
        manager.sampleRate=AudioSettings.outputSampleRate;
        manager.takeDefaultMaterial=AssetDatabase.LoadAssetAtPath<AudioMaterial>("Assets/Materials/Material.asset");
        manager.showDebugTexture=false; manager.enableKeyboardShortcuts=false;
        manager.directSettings.responseStepSamples=256; manager.directSettings.transitionSamples=64;
        manager.audioManager=host.AddComponent<AudioManager>(); manager.audioManager.showWaveform=false;
        manager.audioManager.rayTraceManager=manager;
        Box(root.transform,"Floor",new Vector3(0,-.25f,0),new Vector3(12,.5f,16));
        Box(root.transform,"Left wall",new Vector3(-6,1.5f,0),new Vector3(.25f,3,16));
        Box(root.transform,"Right wall",new Vector3(6,1.5f,0),new Vector3(.25f,3,16));
        Box(root.transform,"Back wall",new Vector3(0,1.5f,8),new Vector3(12,3,.25f));
        var shutter=Box(root.transform,"Moving occluder",new Vector3(-3,1.5f,1),new Vector3(2,3,.25f));
        var motion=shutter.AddComponent<Oscillator>(); motion.moving=true; motion.speed=.7f; motion.distance=6; motion.startAudioOnMove=false;
        var emitter=new GameObject("Emitter"); emitter.transform.SetParent(root.transform); emitter.transform.position=new Vector3(0,1.5f,5);
        var source=emitter.AddComponent<RaytracedAudioSource>(); source.rayTraceManager=manager;
        source.clip=AssetDatabase.LoadAssetAtPath<AudioClip>("Assets/Audio/sine_440Hz_48k.wav"); source.prepareOnStart=false;
        manager.source=emitter.transform; manager.inputClip=source.clip;
        var player=new GameObject("Walking listener"); player.transform.SetParent(root.transform); player.transform.position=new Vector3(0,1,-5);
        player.AddComponent<CharacterController>(); player.AddComponent<PlayerController>();
        var camera=new GameObject("Listener camera"); camera.transform.SetParent(player.transform,false); camera.transform.localPosition=new Vector3(0,.6f,0);
        camera.AddComponent<Camera>(); camera.AddComponent<AudioListener>(); camera.tag="MainCamera";
        camera.AddComponent<MouseLook>().playerBody=player.transform; manager.listener=camera.transform;
        var light=new GameObject("Light").AddComponent<Light>(); light.type=LightType.Directional; light.transform.rotation=Quaternion.Euler(45,-30,0);
        manager.RegisterTakeObjects(); Selection.activeObject=host; EditorSceneManager.MarkSceneDirty(host.scene);
        Debug.Log("Walkthrough ready. Save the scene, enter Play, Record take, then use the emitter's Play Raytraced Audio context menu. WASD/mouse move the listener. Stop/save and compare two response budgets on the manager.");
    }
    static GameObject Box(Transform root,string name,Vector3 position,Vector3 scale)
    {
        var obj=GameObject.CreatePrimitive(PrimitiveType.Cube); obj.name=name; obj.transform.SetParent(root);
        obj.transform.position=position; obj.transform.localScale=scale; obj.AddComponent<AcousticSurface>(); return obj;
    }
}
