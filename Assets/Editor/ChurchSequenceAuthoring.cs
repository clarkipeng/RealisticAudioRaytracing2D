using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;
using Object=UnityEngine.Object;

// Original reusable props are saved as mesh assets. No geometry generation in gameplay.
public static partial class ChurchSequenceAuthoring
{
    const string Folder="Assets/External/Church/Sequence";
    static T Save<T>(T asset,string name) where T:Object
    {
        string path=Folder+"/"+typeof(T).Name+"-"+name+".asset";var existing=AssetDatabase.LoadAssetAtPath<T>(path);
        if(existing!=null){EditorUtility.CopySerialized(asset,existing);Object.DestroyImmediate(asset);return existing;}
        AssetDatabase.CreateAsset(asset,path);return asset;
    }
    static Material Material(string name,Color color,string shader="Standard",float metal=0)
    {
        var m=new Material(Shader.Find(shader)){name=name,color=color};
        if(m.HasProperty("_Metallic"))m.SetFloat("_Metallic",metal);
        if(m.HasProperty("_Glossiness"))m.SetFloat("_Glossiness",.35f);
        return Save(m,name);
    }
    static Transform Node(string name,Transform parent,Vector3 position)
    {var t=new GameObject(name).transform;t.SetParent(parent,false);t.localPosition=position;return t;}
    static Transform Part(string name,Transform parent,Mesh mesh,Material material,Vector3 position=default)
    {
        var t=Node(name,parent,position);t.gameObject.AddComponent<MeshFilter>().sharedMesh=mesh;
        t.gameObject.AddComponent<MeshRenderer>().sharedMaterial=material;return t;
    }
    public static void Build()
    {
        if(!Application.isBatchMode||Application.isPlaying)throw new InvalidOperationException("Author offscreen in batch mode.");
        Directory.CreateDirectory(Folder);AssetDatabase.Refresh();
        foreach(string name in new[]{"RobinCall","TreeWind","ThunderCue"})
        {
            var importer=(AudioImporter)AssetImporter.GetAtPath("Assets/External/Church/Audio/"+name+".wav");
            var settings=importer.defaultSampleSettings;settings.loadType=AudioClipLoadType.DecompressOnLoad;
            settings.compressionFormat=AudioCompressionFormat.PCM;settings.sampleRateSetting=AudioSampleRateSetting.PreserveSampleRate;
            settings.preloadAudioData=true;importer.defaultSampleSettings=settings;importer.loadInBackground=false;importer.SaveAndReimport();
        }
        EditorSceneManager.OpenScene("Assets/Scenes/Church.unity");
        var session=Object.FindAnyObjectByType<SceneSession>();var manager=session.manager;
        if(Object.FindAnyObjectByType<SceneSequenceActor>()!=null)throw new InvalidOperationException("Sequence already authored; edit its saved asset/bindings.");
        var root=Node("Church sequence",manager.takeRoot,Vector3.zero);
        var actor=root.gameObject.AddComponent<SceneSequenceActor>();actor.id="church-sequence";
        var environment=root.gameObject.AddComponent<EnvironmentPresentation>();environment.sequence=actor;
        var bindings=new List<SceneSequenceActor.Binding>();
        RaytracedAudioSource Bind(string id,RaytracedAudioSource source)
        {bindings.Add(new SceneSequenceActor.Binding{id=id,target=source.transform});return source;}
        RaytracedAudioSource Source(string id,string clip,Vector3 position)
        {
            var source=Node(id,root,position).gameObject.AddComponent<RaytracedAudioSource>();source.rayTraceManager=manager;
            source.clip=AssetDatabase.LoadAssetAtPath<AudioClip>("Assets/External/Church/Audio/"+clip+".wav");
            manager.takeObjects.Add(new RayTraceManager.TakeObjectBinding{id="sequence-"+id,transform=source.transform,geometry=false});return Bind(id,source);
        }
        var bell=Bind("bell",manager.takeSources.Single(s=>s.clip!=null&&s.clip.name=="Bell"));
        Bind("rain",manager.takeSources.Single(s=>s.GetComponent<AudioSourcePlane>()!=null));
        var thunder=Bind("thunder",manager.takeSources.Single(s=>s.clip!=null&&s.clip.name=="Thunder"));
        thunder.clip=AssetDatabase.LoadAssetAtPath<AudioClip>("Assets/External/Church/Audio/ThunderCue.wav");thunder.transform.position=new Vector3(45,48,-58);
        var bird=Source("bird","RobinCall",new Vector3(26,5.6f,-35.8f));
        var wind=Source("wind","TreeWind",new Vector3(25,5,-28));
        var sequence=ScriptableObject.CreateInstance<SceneSequence>();sequence.name="Church walkthrough";sequence.seed=173;sequence.duration=60;
        sequence.cues=new[]{
            new SceneSequence.Cue{id="tower-phrase",target="bell",kind=SceneSequence.CueKind.Bell,time=4,duration=17,gain=26},
            new SceneSequence.Cue{id="robin-first",target="bird",kind=SceneSequence.CueKind.Bird,time=15,duration=3.12f,gain=1.4f},
            new SceneSequence.Cue{id="robin-answer",target="bird",kind=SceneSequence.CueKind.Bird,time=19,duration=3.12f,gain=1.4f},
            new SceneSequence.Cue{id="storm-strike",target="thunder",kind=SceneSequence.CueKind.Lightning,time=45,duration=11.3f,gain=40}};
        sequence.levels=new[]{new SceneSequence.LevelTrack{id="grove-gust",target="wind",kind=SceneSequence.LevelKind.Wind,start=23,end=60,gain=6},
            new SceneSequence.LevelTrack{id="courtyard-rain",target="rain",kind=SceneSequence.LevelKind.Rain,start=33,end=60,gain=.1f}};
        SceneSequence.WeatherKey Weather(float t,float cloud,float rain,float windSpeed)=>new SceneSequence.WeatherKey {time=t,cloud=cloud,rain=rain,wind=new Vector3(windSpeed,0,windSpeed*.35f)};
        sequence.weather=new[]{Weather(0,.08f,0,.5f),Weather(23,.15f,0,.7f),Weather(32,.85f,0,5),Weather(35,1,.15f,7),
            Weather(41,1,.9f,8),Weather(52,1,1,6),Weather(58,1,.9f,5),Weather(60,1,0,0)};
        SceneSequence.PositionKey Key(float t,Vector3 p,float yaw)=>new SceneSequence.PositionKey{time=t,position=p,yaw=yaw};
        sequence.poses=new[]{new SceneSequence.PoseTrack{target="bird",keys=new[]{Key(0,bird.transform.position,220),Key(22.2f,bird.transform.position,220),
            Key(25,new Vector3(22,9,-23),330),Key(28,new Vector3(16,11,-12),330)}}};
        actor.sequence=Save(sequence,"Church walkthrough");actor.bindings=bindings.ToArray();
        session.actors=session.actors.Concat(new SceneActor[]{actor}).ToArray();
        session.startSounds=Array.Empty<RaytracedAudioSource>();
        session.player.position=new Vector3(26,1.05f,-43);session.player.rotation=Quaternion.identity;
        session.GetComponent<ChurchVisualPreset>().Apply(ChurchVisualPreset.Preset.PolishedLowPoly);
        BuildBell(actor,bell.transform,root);
        BuildBird(actor,bird.transform);
        BuildTrees(root,environment);
        environment.sun=Object.FindObjectsByType<Light>(FindObjectsSortMode.None).First(l=>l.type==LightType.Directional);
        environment.sun.transform.rotation=Quaternion.Euler(42,-28,0);
        environment.sky=Save(new Material(Shader.Find("Church/Sequence Sky")),"Weather sky");
        environment.rainMaterial=Save(new Material(Shader.Find("Church/Sequence Rain")),"Rain streaks");
        environment.flash=Node("Lightning illumination",root,thunder.transform.position).gameObject.AddComponent<Light>();
        environment.flash.type=LightType.Point;environment.flash.range=150;environment.flash.color=new Color(.72f,.83f,1);environment.flash.intensity=0;
        environment.lightning=Node("Lightning channel",root,Vector3.zero).gameObject.AddComponent<LineRenderer>();
        environment.lightning.sharedMaterial=Material("Lightning",new Color(.8f,.9f,1),"Unlit/Color");
        environment.lightning.positionCount=19;environment.lightning.startWidth=.5f;environment.lightning.endWidth=.13f;environment.lightning.enabled=false;
        environment.wetSurfaces=Object.FindObjectsByType<MeshRenderer>(FindObjectsSortMode.None).Where(r=>r.name=="Stone floor and courtyard").Cast<Renderer>().ToArray();
        Physics.SyncTransforms();
        var heights=new Texture2D(128,192,TextureFormat.RFloat,false,true){name="Static shelter height",wrapMode=TextureWrapMode.Clamp};
        for(int y=0;y<heights.height;y++)for(int x=0;x<heights.width;x++)
        {
            var origin=new Vector3(environment.rainMin.x+(x+.5f)/heights.width*environment.rainSize.x,100,
                environment.rainMin.y+(y+.5f)/heights.height*environment.rainSize.y);
            float height=1;foreach(var hit in Physics.RaycastAll(origin,Vector3.down,105))if(!(hit.collider is CharacterController))height=Mathf.Max(height,hit.point.y);
            heights.SetPixel(x,y,new Color(height/100,0,0,1));
        }
        heights.Apply(false,false);environment.shelterHeights=Save(heights,"Static shelter height");
        manager.RegisterTakeObjects();actor.Initialize();
        if(manager.directSettings.maxBounces!=32)throw new InvalidOperationException("Preserve Church's acoustic profile.");
        EditorSceneManager.SaveScene(session.gameObject.scene);AssetDatabase.SaveAssets();
        var report=new List<string>{"sources="+manager.takeSources.Count,"bounces="+manager.directSettings.maxBounces};
        foreach(var door in session.actors.OfType<SlidingDoor>())report.Add("door "+door.name+" "+door.transform.position+" range="+door.interactionDistance);
        foreach(var r in Object.FindObjectsByType<MeshRenderer>(FindObjectsSortMode.None).Where(r=>r.name.Contains("Bell")))report.Add("bell "+r.name+" bounds="+r.bounds);
        Directory.CreateDirectory(".context/church-sequence");File.WriteAllLines(".context/church-sequence/authoring.txt",report);
        Debug.Log(string.Join("\n",report));
    }
    static void BuildBell(SceneSequenceActor actor,Transform source,Transform parent)
    {
        foreach(var r in Object.FindObjectsByType<MeshRenderer>(FindObjectsSortMode.None).Where(r=>r.name=="Bell"))r.enabled=false;
        var bronze=Material("Bell bronze",new Color(.52f,.32f,.12f),metal:.72f);var iron=Material("Bell iron",new Color(.10f,.12f,.13f),metal:.65f);
        var oak=Material("Bell oak",new Color(.22f,.10f,.045f));
        var rig=Node("Animated belfry bell",parent,new Vector3(0,54,3.92f));
        var stand=new Draft();stand.Beam(new Vector3(-3,-3,0),new Vector3(-3,1,0),.3f);stand.Beam(new Vector3(3,-3,0),new Vector3(3,1,0),.3f);
        stand.Beam(new Vector3(-3,1,0),new Vector3(3,1,0),.4f);Part("Fixed bell frame",rig,Save(stand.Mesh(),"Bell frame"),oak);
        var body=Node("Bell swing",rig,Vector3.zero);
        var shape=new Draft();shape.Lathe(new[]{new Vector2(0,0),new Vector2(.72f,-.15f),new Vector2(.86f,-.5f),new Vector2(.95f,-1.5f),
            new Vector2(1.2f,-2.1f),new Vector2(1.75f,-2.65f),new Vector2(1.8f,-2.83f),new Vector2(1.62f,-2.83f),new Vector2(1.1f,-2.1f),new Vector2(.74f,-.6f),new Vector2(0,-.4f)},32);
        Part("Bronze bell",body,Save(shape.Mesh(),"Bell body"),bronze);
        var rod=new Draft();rod.Beam(Vector3.zero,new Vector3(0,-2.2f,0),.12f);rod.Ellipsoid(new Vector3(0,-2.35f,0),Vector3.one*.23f);
        var clapper=Part("Clapper",body,Save(rod.Mesh(),"Bell clapper"),iron,new Vector3(0,-.2f,0));
        var presentation=rig.gameObject.AddComponent<BellPresentation>();presentation.sequence=actor;presentation.body=body;presentation.clapper=clapper;
        presentation.strikes=new[]{.08f,2.03f,4.02f,6.02f,8.02f,10.01f};
    }
    static void BuildBird(SceneSequenceActor actor,Transform root)
    {
        var brown=Material("Robin feather",new Color(.29f,.22f,.14f));var orange=Material("Robin breast",new Color(.81f,.28f,.06f));
        var black=Material("Robin eyes",new Color(.015f,.02f,.02f));var pale=Material("Robin belly",new Color(.63f,.60f,.46f));
        var d=new Draft();d.Ellipsoid(Vector3.zero,new Vector3(.21f,.24f,.33f));d.Ellipsoid(new Vector3(0,.23f,.24f),Vector3.one*.17f);
        d.Beam(new Vector3(0,0,-.25f),new Vector3(0,-.05f,-.55f),.11f);Part("Robin body",root,Save(d.Mesh(),"Robin body"),brown);
        d=new Draft();d.Ellipsoid(new Vector3(0,.13f,.30f),new Vector3(.145f,.16f,.09f));Part("Orange breast",root,Save(d.Mesh(),"Robin breast"),orange);
        d=new Draft();d.Ellipsoid(new Vector3(0,-.09f,.15f),new Vector3(.17f,.14f,.21f));Part("Pale belly",root,Save(d.Mesh(),"Robin belly"),pale);
        d=new Draft();d.Ellipsoid(new Vector3(-.14f,.27f,.31f),Vector3.one*.025f);d.Ellipsoid(new Vector3(.14f,.27f,.31f),Vector3.one*.025f);
        Part("Eyes",root,Save(d.Mesh(),"Robin eyes"),black);
        var beakDraft=new Draft();beakDraft.Beam(Vector3.zero,new Vector3(0,0,.15f),.038f);
        var beak=Part("Beak",root,Save(beakDraft.Mesh(),"Robin beak"),black,new Vector3(0,.23f,.38f));
        d=new Draft();d.Ellipsoid(new Vector3(-.25f,0,0),new Vector3(.3f,.035f,.16f));var left=Part("Left wing",root,Save(d.Mesh(),"Robin left wing"),brown,new Vector3(-.1f,.05f,0));
        d=new Draft();d.Ellipsoid(new Vector3(.25f,0,0),new Vector3(.3f,.035f,.16f));var right=Part("Right wing",root,Save(d.Mesh(),"Robin right wing"),brown,new Vector3(.1f,.05f,0));
        var p=root.gameObject.AddComponent<BirdPresentation>();p.sequence=actor;p.leftWing=left;p.rightWing=right;p.beak=beak;p.flightStart=22.2f;p.flightEnd=28;
        var samples=RayTraceManager.LoadSample(root.GetComponent<RaytracedAudioSource>().clip,44100);
        p.callEnvelope=new float[(samples.Length+440)/441];float peak=0;
        for(int i=0;i<p.callEnvelope.Length;i++)
        {double power=0;for(int j=i*441;j<Math.Min(samples.Length,(i+1)*441);j++)power+=samples[j]*samples[j];p.callEnvelope[i]=(float)Math.Sqrt(power/441);peak=Mathf.Max(peak,p.callEnvelope[i]);}
        for(int i=0;i<p.callEnvelope.Length;i++)p.callEnvelope[i]=Mathf.Clamp01((p.callEnvelope[i]/peak-.08f)/.6f);
    }
    static void BuildTrees(Transform root,EnvironmentPresentation environment)
    {
        var bark=Material("Tree bark",new Color(.23f,.14f,.07f));
        var green=Material("Tree leaves",new Color(.20f,.36f,.10f),"Church/Sequence Foliage");
        var light=Material("Tree leaves sun",new Color(.33f,.46f,.13f),"Church/Sequence Foliage");
        var trunk=new Draft();trunk.Beam(Vector3.zero,new Vector3(.2f,5.4f,0),.45f);
        var leaves=new[]{new Draft(),new Draft()};
        for(int i=0;i<17;i++)
        {
            float a=i*2.4f,h=3+i*.18f;var end=new Vector3(Mathf.Cos(a)*2.1f,h+.6f,Mathf.Sin(a)*2.1f);
            trunk.Beam(new Vector3(.1f,h-1,0),end,.14f);leaves[i%2].Ellipsoid(end,new Vector3(1.5f,.9f,1.4f));
        }
        trunk.Beam(new Vector3(0,3.7f,0),new Vector3(1,4.6f,-3.8f),.12f);
        var trunkMesh=Save(trunk.Mesh(),"Tree branches");var leafMeshes=leaves.Select((d,i)=>Save(d.Mesh(),"Tree foliage "+i)).ToArray();
        var foliage=new List<Renderer>();var positions=new[]{new Vector3(25,1,-32),new Vector3(27,1,-21),new Vector3(25,1,-10),
            new Vector3(-25,1,-39),new Vector3(-27,1,-22),new Vector3(35,1,6),new Vector3(-32,1,8)};
        for(int i=0;i<positions.Length;i++)
        {
            var tree=Node("Courtyard tree "+(i+1),root,positions[i]);Part("Branches",tree,trunkMesh,bark);
            for(int j=0;j<2;j++)foliage.Add(Part("Leaves",tree,leafMeshes[j],j==0?green:light).GetComponent<Renderer>());
        }
        environment.foliage=foliage.ToArray();
    }
    sealed class Draft
    {
        readonly List<Vector3> vertices=new List<Vector3>();readonly List<int> indices=new List<int>();
        void Triangle(Vector3 a,Vector3 b,Vector3 c){int i=vertices.Count;vertices.Add(a);vertices.Add(b);vertices.Add(c);indices.Add(i);indices.Add(i+1);indices.Add(i+2);}
        void Quad(Vector3 a,Vector3 b,Vector3 c,Vector3 d){Triangle(a,b,c);Triangle(a,c,d);}
        public void Beam(Vector3 a,Vector3 b,float width)
        {
            var q=Quaternion.FromToRotation(Vector3.up,b-a);var u=q*Vector3.right*width*.5f;var v=q*Vector3.forward*width*.5f;
            void Q(Vector3 w,Vector3 x,Vector3 y,Vector3 z)=>Quad(z,y,x,w);
            Q(a-u-v,a+u-v,b+u-v,b-u-v);Q(a+u-v,a+u+v,b+u+v,b+u-v);
            Q(a+u+v,a-u+v,b-u+v,b+u+v);Q(a-u+v,a-u-v,b-u-v,b-u+v);
            Q(a-u-v,a-u+v,a+u+v,a+u-v);Q(b-u-v,b+u-v,b+u+v,b-u+v);
        }
        public void Ellipsoid(Vector3 center,Vector3 radius)
        {
            Vector3 P(int x,int y){float a=x*Mathf.PI*2/12,b=y*Mathf.PI/6;return center+Vector3.Scale(radius,new Vector3(Mathf.Sin(b)*Mathf.Cos(a),Mathf.Cos(b),Mathf.Sin(b)*Mathf.Sin(a)));}
            for(int y=0;y<6;y++)for(int x=0;x<12;x++)Quad(P(x,y),P(x+1,y),P(x+1,y+1),P(x,y+1));
        }
        public void Lathe(Vector2[] rings,int sides)
        {
            Vector3 P(Vector2 r,int x){float a=x*Mathf.PI*2/sides;return new Vector3(r.x*Mathf.Cos(a),r.y,r.x*Mathf.Sin(a));}
            for(int j=0;j<rings.Length-1;j++)for(int i=0;i<sides;i++)Quad(P(rings[j],i),P(rings[j],i+1),P(rings[j+1],i+1),P(rings[j+1],i));
        }
        public Mesh Mesh(){var m=new Mesh{indexFormat=UnityEngine.Rendering.IndexFormat.UInt32};m.SetVertices(vertices);m.SetTriangles(indices,0);m.RecalculateNormals();m.RecalculateBounds();return m;}
    }
}
