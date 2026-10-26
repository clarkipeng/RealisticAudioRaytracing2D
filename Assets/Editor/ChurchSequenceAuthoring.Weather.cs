using System;
using System.Linq;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;
using Object=UnityEngine.Object;

public static partial class ChurchSequenceAuthoring
{
    static void ConfigureSurfaceRain(RaytracedAudioSource rain)
    {
        rain.name="Rain surfaces";
        rain.transform.position=new Vector3(0,100,-7);
        var drops=rain.GetComponent<AudioSourcePlane>();
        drops.oneShotImpacts=true;drops.projectImpacts=true;drops.mask=null;
        // This mask now describes incoming rainfall. Geometry supplies shelter.
        drops.burstPeriodSamples=344;drops.samples=192;
    }
    public static void UpdateRainSurfaces()
    {
        if(!Application.isBatchMode||Application.isPlaying)throw new InvalidOperationException("Author offscreen in batch mode.");
        foreach(string path in new[]{"Assets/Scenes/Church.unity","Assets/Scenes/ChurchWeatherComparison.unity"})
        {
            EditorSceneManager.OpenScene(path);
            var actor=Object.FindAnyObjectByType<SceneSequenceActor>();
            ConfigureSurfaceRain(actor.Target("rain").GetComponent<RaytracedAudioSource>());
            EditorSceneManager.MarkSceneDirty(actor.gameObject.scene);EditorSceneManager.SaveScene(actor.gameObject.scene);
        }
        Debug.Log("CHURCH_SURFACE_RAIN_AUTHORED first acoustic surface from above");
    }
    static void ConfigureWindCoverage(AudioSourcePlane canopy)
    {
        // About 43 overlapping grains at 44.1 kHz. Capacity includes the
        // propagation tail and both response banks, not just audible grains.
        canopy.samples=80;canopy.burstPeriodSamples=2048;canopy.burstDurationSamples=88200;
    }
    public static void UpdateWindCoverage()
    {
        if(!Application.isBatchMode||Application.isPlaying)throw new InvalidOperationException("Author offscreen in batch mode.");
        foreach(string path in new[]{"Assets/Scenes/Church.unity","Assets/Scenes/ChurchWeatherComparison.unity"})
        {
            EditorSceneManager.OpenScene(path);
            var actor=Object.FindAnyObjectByType<SceneSequenceActor>();
            ConfigureWindCoverage(actor.Target("wind").GetComponent<AudioSourcePlane>());
            EditorSceneManager.MarkSceneDirty(actor.gameObject.scene);EditorSceneManager.SaveScene(actor.gameObject.scene);
        }
        Debug.Log("CHURCH_WIND_COVERAGE_AUTHORED overlapping canopy grains");
    }
    // Run after tools/rain_impact.py. Saves both the normal sequence and a
    // repeatable constant-weather route; no foreground editor or Play mode.
    public static void UpdateWeather()
    {
        if(!Application.isBatchMode||Application.isPlaying)throw new InvalidOperationException("Author offscreen in batch mode.");
        AssetDatabase.Refresh();
        string path="Assets/External/Church/Audio/GroundDrop.wav";
        var importer=(AudioImporter)AssetImporter.GetAtPath(path);
        var import=importer.defaultSampleSettings;import.loadType=AudioClipLoadType.DecompressOnLoad;
        import.compressionFormat=AudioCompressionFormat.PCM;import.sampleRateSetting=AudioSampleRateSetting.PreserveSampleRate;
        import.preloadAudioData=true;importer.defaultSampleSettings=import;importer.SaveAndReimport();
        EditorSceneManager.OpenScene("Assets/Scenes/Church.unity");
        var actor=Object.FindAnyObjectByType<SceneSequenceActor>();var session=Object.FindAnyObjectByType<SceneSession>();
        var rain=actor.Target("rain").GetComponent<RaytracedAudioSource>();
        rain.clip=AssetDatabase.LoadAssetAtPath<AudioClip>(path);rain.loop=true;
        ConfigureSurfaceRain(rain);
        var wind=actor.Target("wind").GetComponent<RaytracedAudioSource>();wind.loop=true;
        var canopy=wind.GetComponent<AudioSourcePlane>();if(canopy==null)canopy=wind.gameObject.AddComponent<AudioSourcePlane>();
        wind.transform.position=new Vector3(0,6,-15);canopy.size=new Vector2(80,70);
        canopy.seed=317;ConfigureWindCoverage(canopy);canopy.oneShotImpacts=false;
        var leaves=Object.FindAnyObjectByType<EnvironmentPresentation>().foliage;
        var mask=new Texture2D(160,140,TextureFormat.RFloat,false,true){name="Canopy rustle density",wrapMode=TextureWrapMode.Clamp};
        for(int y=0;y<mask.height;y++)for(int x=0;x<mask.width;x++)
        {
            var point=wind.transform.position+new Vector3(((x+.5f)/mask.width-.5f)*canopy.size.x,0,((y+.5f)/mask.height-.5f)*canopy.size.y);
            float density=0;
            foreach(var leaf in leaves)
            {
                var b=leaf.bounds;float dx=(point.x-b.center.x)/b.extents.x,dz=(point.z-b.center.z)/b.extents.z;
                density=Mathf.Max(density,Mathf.Clamp01(1-dx*dx-dz*dz));
            }
            mask.SetPixel(x,y,new Color(density,0,0,1));
        }
        mask.Apply(false,false);canopy.mask=Save(mask,"Canopy rustle density");
        for(int i=0;i<actor.sequence.levels.Length;i++)
        {
            var level=actor.sequence.levels[i];level.gain=level.kind==SceneSequence.LevelKind.Rain?.006f:.06f;actor.sequence.levels[i]=level;
        }
        EditorUtility.SetDirty(actor.sequence);
        // Explicit interactive weather budget. Reflections retain 32 bounces;
        // temporal/frequency resolution and the 1.25 s path horizon are a preview.
        var settings=session.manager.directSettings;settings.fftSize=128;settings.spectralHop=32;
        settings.responseStepSamples=2048;settings.maximumDelaySeconds=1.25f;settings.maximumResponsesPerBatch=1;
        settings.raysPerBin=2;settings.raysPerPair=0;settings.maxBounces=32;settings.tailSamples=2*session.manager.sampleRate;
        EditorSceneManager.MarkSceneDirty(session.gameObject.scene);EditorSceneManager.SaveScene(session.gameObject.scene);
        var comparison=ScriptableObject.CreateInstance<SceneSequence>();comparison.name="Weather shelter comparison";
        comparison.seed=actor.sequence.seed;comparison.duration=36;
        comparison.weather=new[]{new SceneSequence.WeatherKey{time=0,cloud=1,rain=1,wind=new Vector3(6,0,2.1f)},
            new SceneSequence.WeatherKey{time=35.8f,cloud=1,rain=1,wind=new Vector3(6,0,2.1f)},
            new SceneSequence.WeatherKey{time=36,cloud=1,rain=0,wind=Vector3.zero}};
        comparison.levels=actor.sequence.levels.Select(l=>{l.start=0;l.end=36;return l;}).ToArray();
        actor.sequence=Save(comparison,"Weather shelter comparison");
        session.player.SetPositionAndRotation(new Vector3(2,1.05f,-40),Quaternion.identity);
        EditorSceneManager.MarkSceneDirty(session.gameObject.scene);
        EditorSceneManager.SaveScene(session.gameObject.scene,"Assets/Scenes/ChurchWeatherComparison.unity");
        AssetDatabase.SaveAssets();
        Debug.Log("CHURCH_WEATHER_AUTHORED first-surface impacts + canopy grains + steady shelter comparison");
    }
}
