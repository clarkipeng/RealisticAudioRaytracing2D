using System;
using UnityEngine;
using UnityEngine.Rendering;

// Absolute-time visual evaluation. No random global state and no acoustic commands.
public sealed class EnvironmentPresentation : MonoBehaviour
{
    public SceneSequenceActor sequence;
    public Light sun,flash;
    public Material sky,rainMaterial;
    public Renderer[] foliage=Array.Empty<Renderer>(),wetSurfaces=Array.Empty<Renderer>();
    public Texture2D shelterHeights;
    public Vector2 rainMin=new Vector2(-29,-50),rainSize=new Vector2(58,86);
    public LineRenderer lightning;
    [Range(0,12000)] public int maximumDrops=7000;
    Mesh rainMesh;
    Material skyInstance;
    GameObject rainObject;
    Vector3[] vertices;
    Color[] colors;
    MaterialPropertyBlock block;
    Color[] dryColors;
    void LateUpdate()=>Evaluate(sequence.Seconds,Camera.main);
    public static float Hash(uint value)
    {unchecked{value^=value>>16;value*=0x7feb352d;value^=value>>15;value*=0x846ca68b;value^=value>>16;}return (value&0xffffff)/16777216f;}
    void Initialize()
    {
        if(rainMesh!=null)return;
        block=new MaterialPropertyBlock();skyInstance=new Material(sky){hideFlags=HideFlags.HideAndDontSave};
        dryColors=new Color[wetSurfaces.Length];for(int i=0;i<dryColors.Length;i++)dryColors[i]=wetSurfaces[i].sharedMaterial.color;
        rainObject=new GameObject("Weather rain presentation"){hideFlags=HideFlags.DontSave};
        rainObject.transform.SetParent(transform,false);
        rainMesh=new Mesh{name="Bounded visual rain",indexFormat=IndexFormat.UInt32,hideFlags=HideFlags.HideAndDontSave};rainMesh.MarkDynamic();
        vertices=new Vector3[maximumDrops*4];colors=new Color[vertices.Length];var indices=new int[maximumDrops*6];
        for(int i=0;i<maximumDrops;i++){int v=i*4,j=i*6;indices[j]=v;indices[j+1]=v+1;indices[j+2]=v+2;indices[j+3]=v;indices[j+4]=v+2;indices[j+5]=v+3;}
        rainMesh.vertices=vertices;rainMesh.triangles=indices;
        rainObject.AddComponent<MeshFilter>().sharedMesh=rainMesh;
        var renderer=rainObject.AddComponent<MeshRenderer>();renderer.sharedMaterial=rainMaterial;
        renderer.shadowCastingMode=ShadowCastingMode.Off;renderer.receiveShadows=false;
    }
    public void Evaluate(double seconds,Camera view)
    {
        Initialize();float t=(float)seconds;var weather=sequence.Definition.At(seconds);
        float illumination=0;
        foreach(var cue in sequence.Definition.cues)
        {
            if(cue.kind!=SceneSequence.CueKind.Lightning)continue;
            float age=t-cue.time;
            if(age>=0&&age<.5f)
            {
                illumination=Mathf.Max(illumination,Mathf.Exp(-age*23)+.65f*Mathf.Exp(-Mathf.Abs(age-.16f)*70));
                var center=sequence.Target(cue.target).position;
                for(int i=0;i<lightning.positionCount;i++)
                {
                    float q=i/(float)(lightning.positionCount-1);
                    var p=center+new Vector3((Hash((uint)(sequence.Definition.seed+i*17))-.5f)*7,38-65*q,(Hash((uint)(i*97+93))-.5f)*4);
                    lightning.SetPosition(i,p);
                }
                flash.transform.position=center;
            }
        }
        lightning.enabled=illumination>.13f;flash.intensity=illumination*8;
        sun.intensity=Mathf.Lerp(1.25f,.24f,weather.cloud)+illumination*2;
        sun.color=Color.Lerp(new Color(1,.87f,.66f),new Color(.6f,.72f,.84f),weather.cloud);
        RenderSettings.ambientMode=AmbientMode.Trilight;
        RenderSettings.ambientSkyColor=Color.Lerp(new Color(.47f,.62f,.75f),new Color(.22f,.29f,.37f),weather.cloud)+Color.white*illumination*.5f;
        RenderSettings.ambientEquatorColor=RenderSettings.ambientSkyColor*.65f;
        RenderSettings.ambientGroundColor=new Color(.10f,.12f,.13f);
        var ambient=new SphericalHarmonicsL2();ambient.AddAmbientLight(RenderSettings.ambientSkyColor*.6f+RenderSettings.ambientEquatorColor*.4f);RenderSettings.ambientProbe=ambient;
        RenderSettings.fog=true;RenderSettings.fogMode=FogMode.Linear;
        RenderSettings.fogStartDistance=Mathf.Lerp(80,35,weather.rain);RenderSettings.fogEndDistance=Mathf.Lerp(240,145,weather.rain);
        RenderSettings.fogColor=Color.Lerp(new Color(.57f,.7f,.76f),new Color(.24f,.30f,.36f),weather.cloud);
        skyInstance.SetFloat("_Cloud",weather.cloud);skyInstance.SetFloat("_SequenceTime",t);
        skyInstance.SetFloat("_Flash",illumination);skyInstance.SetVector("_Wind",weather.wind);RenderSettings.skybox=skyInstance;
        block.Clear();block.SetVector("_Wind",weather.wind);block.SetFloat("_SequenceTime",t);
        foreach(var renderer in foliage)renderer.SetPropertyBlock(block);
        block.Clear();block.SetFloat("_Glossiness",Mathf.Lerp(.15f,.8f,weather.wetness));
        for(int i=0;i<wetSurfaces.Length;i++)
        {block.SetColor("_Color",dryColors[i]*Mathf.Lerp(1,.72f,weather.wetness));wetSurfaces[i].SetPropertyBlock(block);}
        int count=Mathf.RoundToInt(maximumDrops*weather.rain);
        var side=(view!=null?view.transform.right:Vector3.right)*.012f;
        var fall=new Vector3(weather.wind.x*.045f,-1.1f,weather.wind.z*.045f);
        for(int i=0;i<maximumDrops;i++)
        {
            uint id=unchecked((uint)(i+sequence.Definition.seed*129));
            float x=Mathf.Repeat(Hash(id*3)*rainSize.x+t*weather.wind.x*.3f,rainSize.x);
            float z=Mathf.Repeat(Hash(id*3+1)*rainSize.y+t*weather.wind.z*.3f,rainSize.y);
            float y=1+Mathf.Repeat(Hash(id*3+2)*74-t*23,74);
            float roof=shelterHeights.GetPixelBilinear(x/rainSize.x,z/rainSize.y).r*100;
            var p=new Vector3(rainMin.x+x,y,rainMin.y+z);
            float alpha=i<count&&y+fall.y>roof+.04f?.24f:0;
            int v=i*4;vertices[v]=p-side;vertices[v+1]=p+side;vertices[v+2]=p+fall+side;vertices[v+3]=p+fall-side;
            for(int k=0;k<4;k++)colors[v+k]=new Color(.7f,.85f,1,alpha);
        }
        rainMesh.vertices=vertices;rainMesh.colors=colors;rainMesh.bounds=new Bounds(new Vector3(0,40,-7),new Vector3(65,85,95));
    }
    void OnDestroy()
    {
        if(RenderSettings.skybox==skyInstance)RenderSettings.skybox=null;
        if(Application.isPlaying){Destroy(rainObject);Destroy(rainMesh);Destroy(skyInstance);}
        else {DestroyImmediate(rainObject);DestroyImmediate(rainMesh);DestroyImmediate(skyInstance);}
    }
}
