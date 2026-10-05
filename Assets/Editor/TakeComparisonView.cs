using System;
using System.Collections.Generic;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;
using UnityEngine.SceneManagement;

// Isolated visual/collider clone: gameplay, audio and rigidbody simulation never run.
internal sealed class TakeComparisonView : IDisposable
{
    readonly AcousticTake take;
    readonly Scene scene;
    readonly Dictionary<Transform,Transform> copies=new Dictionary<Transform,Transform>();
    readonly Transform[] objects;
    readonly AcousticTake.State[] state;
    readonly int[] order;
    readonly Camera listenerCamera,topCamera;
    readonly CharacterController cameraController;
    readonly HashSet<KeyCode> keys=new HashSet<KeyCode>();
    readonly float[] listenerFov;
    RenderTexture listenerTexture,topTexture;
    bool dragging,disposed;
    float yaw,pitch;
    double nextRender,lastTick;
    long lastSample=-1;
    int lastListener=-1;
    bool dirty=true;
    public bool FreeCamera {get;private set;}
    public bool NeedsRender => dirty;
    public long Sample {get;private set;}
    public Camera ListenerCamera => listenerCamera;
    public Camera TopCamera => topCamera;
    public RenderTexture ListenerTexture => listenerTexture;
    public RenderTexture TopTexture => topTexture;
    public int ObjectCount => objects.Length;
    public Vector3 ObjectPosition(int i) => objects[i].position;
    public AcousticTake.Scene Description => take.Metadata.scene;

    public TakeComparisonView(string path,TakeComparison.Report report)
    {
        if(EditorApplication.isPlaying) throw new InvalidOperationException("Stop Play mode before loading recorded views.");
        take=AcousticTake.Open(path);
        try
        {
            var meta=take.Metadata;
            if(meta.sceneHash!=report.sceneHash || meta.stateHash!=report.stateHash || meta.eventHash!=report.eventHash) throw new InvalidOperationException("The selected take does not match this comparison.");
            RayTraceManager manager=null;
            foreach(var candidate in UnityEngine.Object.FindObjectsByType<RayTraceManager>())
            {
                if(candidate.takeObjects.Count!=meta.scene.objects.Length) continue;
                bool match=true;
                for(int i=0;i<candidate.takeObjects.Count;i++) match &= candidate.takeObjects[i].transform!=null && candidate.takeObjects[i].id==meta.scene.objects[i].id;
                if(match) {manager=candidate; break;}
            }
            if(manager==null) throw new InvalidOperationException("Open the original scene with matching registered object IDs to view this take. Audio comparison remains available.");
            scene=EditorSceneManager.NewPreviewScene();
            foreach(var root in manager.gameObject.scene.GetRootGameObjects()) Copy(root.transform,null);
            objects=new Transform[meta.scene.objects.Length]; state=new AcousticTake.State[objects.Length]; order=new int[objects.Length];
            for(int i=0;i<objects.Length;i++) {objects[i]=copies[manager.takeObjects[i].transform]; order[i]=i;}
            Array.Sort(order,(a,b)=>Depth(objects[a]).CompareTo(Depth(objects[b])));
            listenerFov=new float[meta.scene.listeners.Length];
            for(int l=0;l<listenerFov.Length;l++) {var camera=manager.takeObjects[meta.scene.listeners[l]].transform.GetComponent<Camera>(); listenerFov[l]=camera==null ? 70:camera.fieldOfView;}
            listenerCamera=Camera("Recorded listener"); topCamera=Camera("Top down");
            listenerCamera.backgroundColor=new Color(.53f,.66f,.76f); listenerCamera.nearClipPlane=.05f;
            cameraController=listenerCamera.gameObject.AddComponent<CharacterController>();
            cameraController.radius=.15f; cameraController.height=.3f; cameraController.stepOffset=0;
            cameraController.skinWidth=.01f; cameraController.minMoveDistance=0; cameraController.enabled=false;
            topCamera.orthographic=true; topCamera.backgroundColor=new Color(.12f,.15f,.17f);
            if(meta.scene.dimension==AcousticTake.Dimension.Planar2D) topCamera.transform.rotation=Quaternion.identity;
            else topCamera.transform.rotation=Quaternion.LookRotation(Vector3.down,Vector3.forward);
            // The preview has its own light objects; ambient fill is explicit rather than changing global RenderSettings.
            var fill=NewObject("Preview fill").AddComponent<Light>(); fill.type=LightType.Directional; fill.intensity=.45f;
            fill.transform.rotation=Quaternion.Euler(55,145,0); fill.shadows=LightShadows.None;
            SetSample(0,0); FrameTop();
        }
        catch {Dispose(); throw;}
    }
    GameObject NewObject(string name)
    {
        var go=EditorUtility.CreateGameObjectWithHideFlags(name,HideFlags.HideAndDontSave);
        SceneManager.MoveGameObjectToScene(go,scene); return go;
    }
    void Copy(Transform source,Transform parent)
    {
        var go=NewObject(source.name); var target=go.transform; target.SetParent(parent,false);
        target.localPosition=source.localPosition; target.localRotation=source.localRotation; target.localScale=source.localScale; copies.Add(source,target);
        var text=source.GetComponent<TextMesh>(); if(text!=null) EditorUtility.CopySerialized(text,go.AddComponent<TextMesh>());
        var mesh=source.GetComponent<MeshFilter>(); if(mesh!=null) go.AddComponent<MeshFilter>().sharedMesh=mesh.sharedMesh;
        var renderer=source.GetComponent<MeshRenderer>();
        if(renderer!=null) {var clone=go.GetComponent<MeshRenderer>(); if(clone==null) clone=go.AddComponent<MeshRenderer>(); EditorUtility.CopySerialized(renderer,clone); clone.lightmapIndex=-1;}
        // Use the authored solid colliders, including invisible acoustic proxies.
        foreach(var collider in source.GetComponents<Collider>())
            if(collider.enabled && !collider.isTrigger && !(collider is CharacterController))
                EditorUtility.CopySerialized(collider,go.AddComponent(collider.GetType()));
        var light=source.GetComponent<Light>(); if(light!=null) EditorUtility.CopySerialized(light,go.AddComponent<Light>());
        for(int i=0;i<source.childCount;i++) Copy(source.GetChild(i),target);
        go.SetActive(source.gameObject.activeSelf);
    }
    Camera Camera(string name)
    {
        var camera=NewObject(name).AddComponent<Camera>(); camera.enabled=false; camera.scene=scene;
        camera.clearFlags=CameraClearFlags.SolidColor; camera.nearClipPlane=.1f; camera.farClipPlane=2000;
        camera.allowHDR=false; camera.allowMSAA=false; camera.cameraType=CameraType.Preview;
        return camera;
    }
    static int Depth(Transform transform) {int n=0; while(transform!=null) {n++; transform=transform.parent;} return n;}
    public void SetSample(long sample,int listener)
    {
        sample=Math.Clamp(sample,0,take.Metadata.durationSamples);
        if(sample==lastSample && listener==lastListener && !dirty) return;
        if(listener<0 || listener>=take.Metadata.scene.listeners.Length) throw new ArgumentOutOfRangeException(nameof(listener));
        take.Evaluate(sample,state);
        foreach(int i in order)
        {
            objects[i].SetPositionAndRotation(state[i].position,state[i].rotation);
            objects[i].gameObject.SetActive((state[i].flags&AcousticTake.StateFlags.Active)!=0);
        }
        if(!FreeCamera)
        {
            int i=take.Metadata.scene.listeners[listener]; listenerCamera.transform.SetPositionAndRotation(state[i].position,state[i].rotation);
            listenerCamera.fieldOfView=listenerFov[listener];
        }
        Sample=lastSample=sample; lastListener=listener; dirty=true;
    }
    public void FollowListener()
    {
        cameraController.enabled=false; FreeCamera=false; EndDrag(); dirty=true; SetSample(Math.Max(0,lastSample),Math.Max(0,lastListener));
    }
    public void FrameTop()
    {
        var bounds=new Bounds(objects[0].position,Vector3.zero);
        foreach(var t in objects) bounds.Encapsulate(t.position);
        // Frame registered object origins; scroll/pan can reveal the wider visual environment.
        if(take.Metadata.scene.dimension==AcousticTake.Dimension.Planar2D)
        {topCamera.transform.position=new Vector3(bounds.center.x,bounds.center.y,bounds.min.z-50); topCamera.orthographicSize=Mathf.Max(2,bounds.extents.y,bounds.extents.x/1.6f)*1.1f;}
        else
        {topCamera.transform.position=new Vector3(bounds.center.x,bounds.max.y+50,bounds.center.z); topCamera.orthographicSize=Mathf.Max(2,bounds.extents.z,bounds.extents.x/1.6f)*1.1f;}
        dirty=true;
    }
    public void Render(int width,int height)
    {
        width=Mathf.Clamp(width,160,1280); height=Mathf.Clamp(height,90,720);
        if(listenerTexture==null || listenerTexture.width!=width || listenerTexture.height!=height)
        {
            ReleaseTextures(); listenerTexture=Texture(width,height); topTexture=Texture(width,height); dirty=true;
        }
        if(!dirty || EditorApplication.timeSinceStartup<nextRender) return;
        nextRender=EditorApplication.timeSinceStartup+1.0/30;
        listenerCamera.targetTexture=listenerTexture; topCamera.targetTexture=topTexture;
        listenerCamera.aspect=topCamera.aspect=(float)width/height;
        listenerCamera.Render(); topCamera.Render(); dirty=false;
    }
    static RenderTexture Texture(int width,int height)
    {
        var value=new RenderTexture(width,height,24,RenderTextureFormat.ARGB32) {hideFlags=HideFlags.HideAndDontSave}; value.Create(); return value;
    }
    public void Tick()
    {
        double now=EditorApplication.timeSinceStartup; float dt=(float)Math.Min(.1,now-lastTick); lastTick=now;
        if(!dragging) return;
        Vector3 movement=new Vector3((keys.Contains(KeyCode.D)?1:0)-(keys.Contains(KeyCode.A)?1:0),(keys.Contains(KeyCode.E)?1:0)-(keys.Contains(KeyCode.Q)?1:0),(keys.Contains(KeyCode.W)?1:0)-(keys.Contains(KeyCode.S)?1:0));
        if(movement!=Vector3.zero) MoveCamera(listenerCamera.transform.TransformDirection(movement.normalized)*dt*(keys.Contains(KeyCode.LeftShift)?15:5));
    }
    public void MoveCamera(Vector3 displacement)
    {
        if(!FreeCamera) return;
        Physics.SyncTransforms(); // Recorded objects may have moved since the previous camera sweep.
        cameraController.Move(displacement); dirty=true;
    }
    public void Input(Event e,Rect listenerRect,Rect topRect)
    {
        if(e.type==EventType.MouseDown && e.button==1 && listenerRect.Contains(e.mousePosition))
        {
            FreeCamera=true; cameraController.enabled=true; dragging=true; keys.Clear(); Vector3 angles=listenerCamera.transform.eulerAngles; yaw=angles.y; pitch=Mathf.DeltaAngle(0,angles.x);
            EditorGUIUtility.SetWantsMouseJumping(1); e.Use();
        }
        if(dragging)
        {
            if(e.type==EventType.MouseDrag && e.button==1) {yaw+=e.delta.x*.25f; pitch=Mathf.Clamp(pitch-e.delta.y*.25f,-89,89); listenerCamera.transform.rotation=Quaternion.Euler(pitch,yaw,0); dirty=true; e.Use();}
            if(e.type==EventType.KeyDown) {if(e.keyCode==KeyCode.Escape) EndDrag(); else keys.Add(e.keyCode); e.Use();}
            if(e.type==EventType.KeyUp) {keys.Remove(e.keyCode); e.Use();}
            if(e.type==EventType.MouseUp && e.button==1) {EndDrag(); e.Use();}
        }
        if(topRect.Contains(e.mousePosition))
        {
            if(e.type==EventType.ScrollWheel) {topCamera.orthographicSize=Mathf.Clamp(topCamera.orthographicSize*Mathf.Pow(1.1f,e.delta.y),.5f,1000); dirty=true; e.Use();}
            if(e.type==EventType.MouseDrag && e.button==2)
            {topCamera.transform.position+=(-topCamera.transform.right*e.delta.x+topCamera.transform.up*e.delta.y)*(topCamera.orthographicSize*2/topRect.height); dirty=true; e.Use();}
        }
    }
    public void EndDrag() {if(dragging) EditorGUIUtility.SetWantsMouseJumping(0); dragging=false; keys.Clear();}
    void ReleaseTextures()
    {
        if(listenerCamera!=null) listenerCamera.targetTexture=null; if(topCamera!=null) topCamera.targetTexture=null;
        foreach(var texture in new[]{listenerTexture,topTexture}) if(texture!=null) {texture.Release(); UnityEngine.Object.DestroyImmediate(texture);}
        listenerTexture=topTexture=null;
    }
    public void Dispose()
    {
        if(disposed) return; disposed=true; EndDrag(); ReleaseTextures();
        if(scene.IsValid()) EditorSceneManager.ClosePreviewScene(scene); take?.Dispose();
    }
}
