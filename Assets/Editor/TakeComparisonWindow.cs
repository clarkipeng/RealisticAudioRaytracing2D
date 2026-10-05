using System;
using System.IO;
using UnityEditor;
using UnityEngine;

public sealed class TakeComparisonWindow : EditorWindow
{
    [SerializeField] string takePath="",comparisonPath="";
    [SerializeField] TakeRender.Settings settings=new TakeRender.Settings();
    [SerializeField] ComputeShader shader;
    [SerializeField] int highStep=128,listener,fft=2048,hop=512,mode;
    [SerializeField] double startSeconds,lengthSeconds=10;
    [SerializeField] float gain=1,floorDb=-90;
    [SerializeField] bool renderFoldout=true,details,checkReference,showViews=true;
    [SerializeField] TakeComparison.ReferenceCheck reference=new TakeComparison.ReferenceCheck {responseStepSamples=64};
    [NonSerialized] TakeComparison.Report report;
    [NonSerialized] TakeComparisonView view;
    string viewMessage="";
    double nextRepaint;
    [NonSerialized] RayTraceManager.LiveDiagnostics live;
    [NonSerialized] AudioManager.PlaybackCapture.Report capture;
    [NonSerialized] TakeComparisonAudio.Wave lowWave,highWave;
    [NonSerialized] TakeComparisonAudio.Preview preview;
    [NonSerialized] float[] low,high;
    [NonSerialized] Vector2[][] envelopes;
    [NonSerialized] Texture2D[] spectra;
    [NonSerialized] TakeComparisonAudio.Spectrum[] spectrumData;
    string error="",liveNotice="";
    long firstFrame;
    int cursor,frames;
    float peak;
    Vector2 scroll;
    static readonly string[] modes={"Low","High","Difference"};

    [MenuItem("Tools/Acoustics/Compare audio")]
    public static void ShowWindow()
    {
        var window=GetWindow<TakeComparisonWindow>("Acoustic comparison"); window.minSize=new Vector2(880,600);
        if(string.IsNullOrEmpty(window.takePath)) window.UseManager();
    }
    public static void OpenComparison(string path)
    {
        ShowWindow(); var window=GetWindow<TakeComparisonWindow>(); window.Try(()=>window.LoadComparison(path));
    }
    void OnEnable()
    {
        titleContent=new GUIContent("Acoustic comparison");
        preview=new TakeComparisonAudio.Preview(); EditorApplication.update+=Tick; EditorApplication.playModeStateChanged+=PlayModeChanged;
        if(!string.IsNullOrEmpty(comparisonPath) && File.Exists(comparisonPath)) Try(()=>LoadComparison(comparisonPath));
    }
    void OnDisable()
    {
        EditorApplication.update-=Tick; EditorApplication.playModeStateChanged-=PlayModeChanged;
        Release(); preview?.Dispose();
    }
    void PlayModeChanged(PlayModeStateChange state)
    {
        Pause(); view?.Dispose(); view=null;
        if(state==PlayModeStateChange.EnteredEditMode && report!=null && showViews) LoadViews();
    }
    void OnLostFocus() {view?.EndDrag();}
    void Tick()
    {
        bool playing=preview!=null && preview.Active;
        if(playing) Try(()=>cursor=Math.Min(frames,preview.Poll()));
        view?.Tick();
        if((playing || (view!=null && view.NeedsRender)) && EditorApplication.timeSinceStartup>=nextRepaint)
        {nextRepaint=EditorApplication.timeSinceStartup+1.0/30; Repaint();}
    }
    void Try(Action action)
    {
        try {action(); error="";}
        catch(Exception e) {error=e.GetBaseException().Message; preview?.Dispose(); Repaint();}
    }
    void UseManager()
    {
        var manager=Selection.activeGameObject==null ? null:Selection.activeGameObject.GetComponent<RayTraceManager>();
        if(manager==null) manager=UnityEngine.Object.FindAnyObjectByType<RayTraceManager>();
        if(manager==null) return;
        takePath=manager.takePath; settings=JsonUtility.FromJson<TakeRender.Settings>(JsonUtility.ToJson(manager.directSettings)); shader=manager.directShader;
    }
    void Release()
    {
        view?.Dispose(); view=null; preview?.Dispose(); lowWave?.Dispose(); highWave?.Dispose(); lowWave=highWave=null;
        DestroyTextures(); low=high=null; envelopes=null; spectrumData=null; report=null; live=null; capture=null;
    }
    void DestroyTextures() {if(spectra!=null) foreach(var texture in spectra) if(texture!=null) DestroyImmediate(texture); spectra=null;}

    public void LoadComparison(string path)
    {
        Release(); comparisonPath=Path.GetFullPath(path);
        try
        {
            report=TakeComparisonAudio.OpenReport(comparisonPath); listener=Mathf.Clamp(listener,0,report.low.listenerCount-1);
            LoadLiveDiagnostics(); LoadListener(); renderFoldout=false; if(showViews) LoadViews();
        }
        catch {Release(); throw;}
        Repaint();
    }
    void LoadLiveDiagnostics()
    {
        liveNotice="No matching recorded live diagnostics. Offline renders do not have playback deadlines.";
        string relative=report.takeDirectory;
        string root=string.IsNullOrEmpty(relative) ? Path.GetDirectoryName(Path.GetDirectoryName(comparisonPath)):Path.GetFullPath(Path.Combine(Path.GetDirectoryName(comparisonPath),relative));
        if(!File.Exists(Path.Combine(root,"take.json"))) return;
        var take=JsonUtility.FromJson<AcousticTake.Manifest>(File.ReadAllText(Path.Combine(root,"take.json")));
        if(take==null || !take.complete || take.sceneHash!=report.sceneHash || take.stateHash!=report.stateHash || take.eventHash!=report.eventHash) return;
        takePath=root;
        string livePath=Path.Combine(root,"live.json"),capturePath=Path.Combine(root,"monitor","output.json");
        if(File.Exists(livePath)) live=JsonUtility.FromJson<RayTraceManager.LiveDiagnostics>(File.ReadAllText(livePath));
        if(File.Exists(capturePath)) capture=JsonUtility.FromJson<AudioManager.PlaybackCapture.Report>(File.ReadAllText(capturePath));
    }
    void LoadListener()
    {
        preview.Dispose(); lowWave?.Dispose(); highWave?.Dispose(); lowWave=highWave=null;
        low=high=null; DestroyTextures(); envelopes=null; spectrumData=null;
        string root=Path.GetDirectoryName(comparisonPath),file="listener-"+listener+".wav";
        lowWave=new TakeComparisonAudio.Wave(Path.Combine(root,"low",file),report.low.audioHashes[listener],report.low);
        try {highWave=new TakeComparisonAudio.Wave(Path.Combine(root,"high",file),report.high.audioHashes[listener],report.high); LoadRange();}
        catch {lowWave.Dispose(); highWave?.Dispose(); lowWave=highWave=null; throw;}
    }
    public void LoadRange()
    {
        preview.Dispose();
        if(lowWave==null) return;
        if(!double.IsFinite(startSeconds) || !double.IsFinite(lengthSeconds) || lengthSeconds<=0 || lengthSeconds>30) throw new ArgumentException("Preview duration must be greater than zero and at most 30 seconds.");
        long first=(long)Math.Round(Math.Clamp(startSeconds,0,(double)(lowWave.Frames-1)/lowWave.Rate)*lowWave.Rate);
        int count=(int)Math.Min(Math.Min(lowWave.Frames-first,Math.Max(1,Math.Round(lengthSeconds*lowWave.Rate))),TakeComparisonAudio.MaximumPreviewFrames);
        var a=lowWave.Read(first,count); var b=highWave.Read(first,count); float maximum=0;
        foreach(var pcm in new[]{a,b}) foreach(float value in pcm) maximum=Mathf.Max(maximum,Mathf.Abs(value));
        var env=new Vector2[lowWave.Channels*2][]; var analysis=new TakeComparisonAudio.Spectrum[env.Length];
        var textures=new Texture2D[env.Length];
        try
        {
            for(int side=0;side<2;side++) for(int ear=0;ear<lowWave.Channels;ear++)
            {
                int i=side*lowWave.Channels+ear; var pcm=side==0 ? a:b;
                env[i]=TakeComparisonAudio.Envelope(pcm,lowWave.Channels,ear,768);
                analysis[i]=TakeComparisonAudio.Spectrogram(pcm,lowWave.Channels,ear,fft,hop);
                textures[i]=TakeComparisonAudio.Texture(analysis[i],floorDb);
            }
        }
        catch {foreach(var texture in textures) if(texture!=null) DestroyImmediate(texture); throw;}
        DestroyTextures(); firstFrame=first; frames=count; low=a; high=b; cursor=0; peak=maximum==0 ? 1:maximum;
        envelopes=env; spectrumData=analysis; spectra=textures;
        startSeconds=(double)firstFrame/lowWave.Rate; Repaint();
    }
    void RenderComparison()
    {
        if(EditorApplication.isPlaying) throw new InvalidOperationException("Stop Play mode before rendering comparisons.");
        var lowSettings=JsonUtility.FromJson<TakeRender.Settings>(JsonUtility.ToJson(settings));
        var highSettings=JsonUtility.FromJson<TakeRender.Settings>(JsonUtility.ToJson(settings)); highSettings.responseStepSamples=highStep;
        string output=Path.Combine(takePath,"comparison-"+DateTime.UtcNow.ToString("yyyyMMdd-HHmmss-fff"));
        preview.Dispose();
        try
        {
            using(var take=AcousticTake.Open(takePath)) TakeComparison.Run(take,lowSettings,highSettings,output,shader,p=>
            {if(EditorUtility.DisplayCancelableProgressBar("Rendering acoustic comparison",output,p)) throw new OperationCanceledException("Comparison cancelled; incomplete output retained.");},checkReference ? reference:null);
            LoadComparison(Path.Combine(output,"comparison.json"));
        }
        finally {EditorUtility.ClearProgressBar();}
    }
    void Pause() {if(preview?.Active==true) {cursor=preview.Poll(); preview.Pause();}}
    void Play()
    {
        if(EditorApplication.isPlaying) return;
        if(cursor>=frames) cursor=0;
        preview.Play(low,high,lowWave.Channels,cursor,mode,gain,lowWave.Rate);
    }
    void Seek(int frame)
    {
        bool resume=preview.Active; preview.Pause(); cursor=Mathf.Clamp(frame,0,frames-1); if(resume) Play();
    }

    void OnGUI()
    {
        using(new EditorGUILayout.HorizontalScope(EditorStyles.toolbar))
        {
            if(GUILayout.Button("Open comparison…",EditorStyles.toolbarButton,GUILayout.Width(135)))
            {string path=EditorUtility.OpenFilePanel("Open comparison.json",string.IsNullOrEmpty(comparisonPath)?"Recordings":Path.GetDirectoryName(comparisonPath),"json"); if(path!="") Try(()=>LoadComparison(path));}
            if(GUILayout.Button("Use scene manager",EditorStyles.toolbarButton,GUILayout.Width(135))) Try(UseManager);
            GUILayout.FlexibleSpace();
            if(report!=null && GUILayout.Button("Show files",EditorStyles.toolbarButton,GUILayout.Width(80))) EditorUtility.RevealInFinder(comparisonPath);
        }
        scroll=EditorGUILayout.BeginScrollView(scroll);
        if(!string.IsNullOrEmpty(error)) EditorGUILayout.HelpBox(error,MessageType.Error);
        renderFoldout=EditorGUILayout.Foldout(renderFoldout,"Render a saved take at two response budgets",true);
        if(renderFoldout) DrawRenderControls();
        if(report==null) EditorGUILayout.HelpBox("Open a completed comparison.json, or render a recorded take above. The scene does not need to be running.",MessageType.Info);
        else DrawComparison();
        EditorGUILayout.EndScrollView();
    }
    void DrawRenderControls()
    {
        using(new EditorGUI.DisabledScope(EditorApplication.isPlaying))
        {
            using(new EditorGUILayout.HorizontalScope())
            {
                takePath=EditorGUILayout.TextField("Recorded take",takePath);
                if(GUILayout.Button("Choose…",GUILayout.Width(80))) {string path=EditorUtility.OpenFolderPanel("Recorded take",takePath,""); if(path!="") takePath=path;}
            }
            shader=(ComputeShader)EditorGUILayout.ObjectField("Shader (empty = CPU)",shader,typeof(ComputeShader),false);
            using(new EditorGUILayout.HorizontalScope())
            {settings.responseStepSamples=EditorGUILayout.IntField("Low step (samples)",settings.responseStepSamples); highStep=EditorGUILayout.IntField("High step (samples)",highStep);}
            settings.listenerOutput=(TakeRender.ListenerOutput)EditorGUILayout.EnumPopup("Listener output",settings.listenerOutput);
            settings.tailSamples=EditorGUILayout.IntField("Tail (samples)",settings.tailSamples);
            settings.transitionSamples=EditorGUILayout.IntField("Transition (samples)",settings.transitionSamples);
            settings.delayFilter=(TakeRender.DelayFilter)EditorGUILayout.EnumPopup("Delay filter",settings.delayFilter);
            if(settings.delayFilter==TakeRender.DelayFilter.WindowedSinc) settings.sincRadius=EditorGUILayout.IntField("Sinc radius",settings.sincRadius);
            checkReference=EditorGUILayout.Toggle("Check finer reference",checkReference);
            if(checkReference)
            {
                reference.responseStepSamples=EditorGUILayout.IntField("Reference step",reference.responseStepSamples);
                reference.maximumRmsError=EditorGUILayout.DoubleField("Maximum RMS error",reference.maximumRmsError);
                reference.maximumAbsoluteError=EditorGUILayout.DoubleField("Maximum absolute error",reference.maximumAbsoluteError);
            }
            EditorGUILayout.LabelField("Both runs share all other settings. Current model: direct sound with opaque occlusion.",EditorStyles.wordWrappedMiniLabel);
            if(GUILayout.Button("Render low + high and open comparison")) Try(RenderComparison);
        }
    }
    void DrawComparison()
    {
        EditorGUILayout.LabelField(Path.GetFileName(Path.GetDirectoryName(comparisonPath)),EditorStyles.boldLabel);
        var names=new string[report.low.listenerCount];
        for(int l=0;l<names.Length;l++) names[l]=report.listenerNames!=null && l<report.listenerNames.Length ? l+": "+report.listenerNames[l]:"Listener "+l;
        int selected=EditorGUILayout.Popup("Listener",listener,names);
        if(selected!=listener) {listener=selected; Try(LoadListener);}
        var metric=report.listeners[listener];
        EditorGUILayout.LabelField($"Whole take  •  RMS error {metric.rmsError:G4}  •  Peak error {metric.maximumAbsoluteError:G4}  •  Reference RMS {metric.referenceRms:G4}  •  "+(metric.exactMatch?"Exact match":metric.snrDefined?$"SNR {metric.snrDb:F2} dB":"SNR undefined (silent reference)"),EditorStyles.wordWrappedLabel);
        EditorGUILayout.LabelField($"Low {report.low.wallSeconds:F3}s / High {report.high.wallSeconds:F3}s wall time  •  High/low cost {report.measuredSpeedup:F2}×  •  Steps {report.low.settings.responseStepSamples} / {report.high.settings.responseStepSamples} samples",EditorStyles.wordWrappedMiniLabel);
        if(live!=null) EditorGUILayout.LabelField($"Recorded live run  •  Lead {1000.0*live.leadSamples/live.sampleRate:F1} ms  •  Underrun {live.underrunSamples} / late {live.lateSamples} / overflow {live.overflowSamples} frames  •  Callback overruns {live.overBudgetCallbacks}",EditorStyles.wordWrappedMiniLabel);
        using(new EditorGUILayout.HorizontalScope())
        {
            startSeconds=EditorGUILayout.DoubleField("Start (s)",startSeconds); lengthSeconds=EditorGUILayout.DoubleField("Duration (s), ≤30",lengthSeconds);
            using(new EditorGUI.DisabledScope(lowWave==null)) if(GUILayout.Button("Load range",GUILayout.Width(90))) Try(LoadRange);
        }
        if(low==null || spectra==null) return;
        using(new EditorGUILayout.HorizontalScope())
        {
            using(new EditorGUI.DisabledScope(!preview.Supported || EditorApplication.isPlaying))
            {
                if(GUILayout.Button(preview.Active?"Pause":"Play",GUILayout.Width(65))) Try(()=> {if(preview.Active) Pause(); else Play();});
                if(GUILayout.Button("Stop",GUILayout.Width(55))) Try(()=> {Pause(); cursor=0;});
                int next=GUILayout.Toolbar(mode,modes,GUILayout.Width(260));
                if(next!=mode) Try(()=> {bool resume=preview.Active; Pause(); mode=next; if(resume) Play();});
                float nextGain=EditorGUILayout.Slider("Shared gain",gain,0,2);
                if(nextGain!=gain) Try(()=> {bool resume=preview.Active; Pause(); gain=nextGain; if(resume) Play();});
            }
        }
        if(!preview.Supported) EditorGUILayout.HelpBox("Unity preview API unavailable; plots and exported WAVs remain usable.",MessageType.Warning);
        if(EditorApplication.isPlaying) EditorGUILayout.HelpBox("Stop Play mode to audition without the scene's live audio.",MessageType.Info);
        EditorGUI.BeginChangeCheck(); float at=EditorGUILayout.Slider("Shared cursor (s)",(float)((firstFrame+cursor)/(double)lowWave.Rate),(float)((double)firstFrame/lowWave.Rate),(float)((double)(firstFrame+frames-1)/lowWave.Rate));
        if(EditorGUI.EndChangeCheck()) Try(()=>Seek((int)Math.Round(at*lowWave.Rate-firstFrame)));
        EditorGUILayout.LabelField($"Loaded {firstFrame/(double)lowWave.Rate:F3}–{(firstFrame+frames)/(double)lowWave.Rate:F3}s  •  {lowWave.Rate} Hz / {lowWave.Channels} channel(s)  •  shared waveform scale ±{peak:G3}  •  audition fades 5 ms, no independent normalization",EditorStyles.wordWrappedMiniLabel);
        DrawViews();
        using(new EditorGUILayout.HorizontalScope())
        {
            fft=EditorGUILayout.IntField("Display FFT",fft); hop=EditorGUILayout.IntField("Requested hop",hop);
            float db=EditorGUILayout.FloatField("dBFS floor",floorDb);
            if(db!=floorDb && float.IsFinite(db)) {floorDb=Mathf.Clamp(db,-160,-10); Try(()=> {DestroyTextures(); spectra=new Texture2D[spectrumData.Length]; for(int i=0;i<spectra.Length;i++) spectra[i]=TakeComparisonAudio.Texture(spectrumData[i],floorDb);});}
            if(GUILayout.Button("Analyze",GUILayout.Width(70))) Try(LoadRange);
        }
        EditorGUILayout.LabelField($"Spectrogram: Hann / FFT {spectrumData[0].Fft} / displayed hop {spectrumData[0].Hop} samples; {floorDb:F0} to 0 dBFS, linear 0–{lowWave.Rate/2000.0:F2} kHz. Display columns may be decimated; final windows zero-pad. Analysis only, not training tensors.",EditorStyles.wordWrappedMiniLabel);
        for(int ear=0;ear<lowWave.Channels;ear++)
        {
            using(new EditorGUILayout.HorizontalScope())
                for(int side=0;side<2;side++)
                    using(new EditorGUILayout.VerticalScope())
                    {
                        int index=side*lowWave.Channels+ear;
                        EditorGUILayout.LabelField((side==0?"LOW":"HIGH")+" · "+(lowWave.Channels==1?"MONO":ear==0?"LEFT":"RIGHT"),EditorStyles.boldLabel);
                        DrawWave(GUILayoutUtility.GetRect(100,58,GUILayout.ExpandWidth(true)),envelopes[index],side==0?new Color(.25f,.75f,.95f):new Color(1,.65f,.3f));
                        Rect rect=GUILayoutUtility.GetRect(100,105,GUILayout.ExpandWidth(true));
                        GUI.DrawTexture(rect,spectra[index],ScaleMode.StretchToFill); DrawCursor(rect); ClickSeek(rect);
                        using(new EditorGUILayout.HorizontalScope())
                        {GUILayout.Label($"{firstFrame/(double)lowWave.Rate:F2}s",EditorStyles.miniLabel); GUILayout.FlexibleSpace(); GUILayout.Label($"{(firstFrame+frames)/(double)lowWave.Rate:F2}s",EditorStyles.miniLabel);}
                    }
        }
        details=EditorGUILayout.Foldout(details,"Timings, recorded deadlines and provenance",true); if(details) DrawDetails();
    }
    void LoadViews()
    {
        view?.Dispose(); view=null;
        try {view=new TakeComparisonView(takePath,report); viewMessage="";}
        catch(Exception e) {viewMessage=e.GetBaseException().Message;}
    }
    void DrawViews()
    {
        using(new EditorGUILayout.HorizontalScope())
        {
            bool show=EditorGUILayout.ToggleLeft("Recorded scene views",showViews,GUILayout.Width(165));
            if(show!=showViews) {showViews=show; if(show) LoadViews(); else {view?.Dispose(); view=null;}}
            if(!showViews) return;
            if(GUILayout.Button("Reload visuals",GUILayout.Width(110))) LoadViews();
            if(view!=null)
            {
                if(GUILayout.Button("Follow listener",GUILayout.Width(110))) view.FollowListener();
                if(GUILayout.Button("Frame top view",GUILayout.Width(110))) view.FrameTop();
            }
        }
        if(view==null) {EditorGUILayout.HelpBox(viewMessage,MessageType.Info); return;}
        Rect area=GUILayoutUtility.GetRect(100,210,GUILayout.ExpandWidth(true));
        float half=(area.width-8)/2;
        Rect left=new Rect(area.x,area.y+19,half,area.height-19),right=new Rect(area.x+half+8,area.y+19,half,area.height-19);
        GUI.Label(new Rect(area.x,area.y,half,18),view.FreeCamera?"FREE CAMERA · audio follows recorded listener":"LISTENER PERSPECTIVE",EditorStyles.boldLabel);
        GUI.Label(new Rect(right.x,area.y,half,18),"TOP DOWN · sources + listeners",EditorStyles.boldLabel);
        view.Input(Event.current,left,right);
        if(Event.current.type==EventType.Repaint)
        {
            try {view.SetSample(firstFrame+cursor,listener); view.Render((int)half,(int)left.height);}
            catch(Exception e) {viewMessage=e.Message; view.Dispose(); view=null; return;}
        }
        if(view.ListenerTexture!=null) GUI.DrawTexture(left,view.ListenerTexture,ScaleMode.StretchToFill);
        if(view.TopTexture!=null) GUI.DrawTexture(right,view.TopTexture,ScaleMode.StretchToFill);
        DrawMarkers(right);
        EditorGUILayout.LabelField("Listener view: hold right mouse + WASD, Q/E up/down, Shift faster. Top view: scroll to zoom, middle-drag to pan. Follow listener restores the recorded camera. Visuals use the open scene's current static meshes/materials and recorded poses; audio keeps its recorded receiver.",EditorStyles.wordWrappedMiniLabel);
    }
    void DrawMarkers(Rect rect)
    {
        var scene=view.Description;
        for(int i=0;i<scene.sources.Length;i++) Marker(rect,view.ObjectPosition(scene.sources[i].objectIndex),"S"+i,new Color(1,.7f,.15f));
        for(int i=0;i<scene.listeners.Length;i++) Marker(rect,view.ObjectPosition(scene.listeners[i]),"L"+i,i==listener?Color.cyan:Color.white);
    }
    void Marker(Rect rect,Vector3 position,string label,Color color)
    {
        var p=view.TopCamera.WorldToViewportPoint(position);
        if(p.z<0 || p.x<0 || p.x>1 || p.y<0 || p.y>1) return;
        Vector2 at=new Vector2(rect.x+p.x*rect.width,rect.y+(1-p.y)*rect.height);
        EditorGUI.DrawRect(new Rect(at.x-3,at.y-3,6,6),color);
        var style=new GUIStyle(EditorStyles.miniBoldLabel); style.normal.textColor=color;
        GUI.Label(new Rect(at.x+5,at.y-8,35,18),label,style);
    }
    void DrawWave(Rect rect,Vector2[] envelope,Color color)
    {
        EditorGUI.DrawRect(rect,new Color(.07f,.075f,.085f));
        if(Event.current.type==EventType.Repaint)
        {
            Handles.BeginGUI(); Handles.color=new Color(.35f,.35f,.35f); Handles.DrawLine(new Vector3(rect.x,rect.center.y),new Vector3(rect.xMax,rect.center.y)); Handles.color=color;
            for(int x=0;x<envelope.Length;x++) {float px=rect.x+(x+.5f)*rect.width/envelope.Length; Handles.DrawLine(new Vector3(px,rect.center.y-envelope[x].x/peak*rect.height*.47f),new Vector3(px,rect.center.y-envelope[x].y/peak*rect.height*.47f));}
            Handles.EndGUI();
        }
        DrawCursor(rect); ClickSeek(rect);
    }
    void DrawCursor(Rect rect) {EditorGUI.DrawRect(new Rect(rect.x+Mathf.Clamp01((float)cursor/frames)*rect.width,rect.y,1,rect.height),Color.white);}
    void ClickSeek(Rect rect)
    {
        if(Event.current.type==EventType.MouseDown && Event.current.button==0 && rect.Contains(Event.current.mousePosition))
        {int frame=(int)((Event.current.mousePosition.x-rect.x)/rect.width*frames); Try(()=>Seek(frame)); Event.current.Use();}
    }
    void DrawDetails()
    {
        Row("Stage (seconds)","Low","High");
        Row("Preparation",report.low.preparationSeconds,report.high.preparationSeconds); Row("Evaluate poses",report.low.evaluationSeconds,report.high.evaluationSeconds);
        Row("Solve (incl. sync readback)",report.low.solveSeconds,report.high.solveSeconds); Row("DSP",report.low.audioSeconds,report.high.audioSeconds); Row("File writing",report.low.writeSeconds,report.high.writeSeconds);
        Row("Maximum solve",report.low.maximumSolveSeconds,report.high.maximumSolveSeconds);
        Row("Wall / audio duration",report.low.wallSeconds/(report.low.outputSamples/(double)report.low.sampleRate),report.high.wallSeconds/(report.high.outputSamples/(double)report.high.sampleRate));
        Row("Numeric working MiB (not process)",report.low.workingNumericBytes/1048576.0,report.high.workingNumericBytes/1048576.0);
        Row("Backend",report.low.backend,report.high.backend);
        EditorGUILayout.LabelField(report.referenceStatus,EditorStyles.wordWrappedLabel);
        EditorGUILayout.LabelField("Temporal training gate: "+(report.eligibleForTemporalTraining?"passed mixed-audio reference check; per-pair validation still required":"not accepted")+". Direct model only.",EditorStyles.wordWrappedMiniLabel);
        if(live==null) EditorGUILayout.LabelField(liveNotice,EditorStyles.wordWrappedMiniLabel);
        else
        {
            EditorGUILayout.LabelField($"Recorded live settings: {live.backend}, {live.sampleRate} Hz, buffer {live.deviceBufferSamples}, queue {live.queueCapacity}, {live.responseCount} responses / {live.observedSamples/(double)live.sampleRate:F2}s. These counters describe capture, not either offline budget.",EditorStyles.wordWrappedLabel);
            EditorGUILayout.LabelField($"Validation {live.validationSeconds:F4}s; solve submission {live.solveSeconds:F4}s; DSP {live.audioSeconds:F4}s. Max observation gap {1000.0*live.maximumObservationGapSamples/live.sampleRate:F2} ms; GPU completion {live.maximumCompletionSeconds*1000:F2} ms; callback {live.maximumCallbackSeconds*1000:F3} ms; channel mismatches {live.channelMismatchSamples}.",EditorStyles.wordWrappedMiniLabel);
            if(capture!=null) EditorGUILayout.LabelField($"Component output capture: {(capture.complete?"complete":"incomplete")}, {capture.samples} frames; capture overflow {capture.overflowSamples}, channel mismatch {capture.channelMismatchSamples}.",EditorStyles.wordWrappedMiniLabel);
        }
        EditorGUILayout.LabelField("Timing is a single run, not a percentile benchmark. GPU completion includes scheduling/readback. Stages can overlap and are not summed as latency.",EditorStyles.wordWrappedMiniLabel);
        if(report.listenerIds!=null && listener<report.listenerIds.Length) EditorGUILayout.LabelField("Listener ID",report.listenerIds[listener]);
        EditorGUILayout.SelectableLabel("Scene "+report.sceneHash+"\nStates "+report.stateHash+"\nEvents "+report.eventHash+"\nLow settings "+report.low.settingsHash+"\nHigh settings "+report.high.settingsHash+"\nHRTF "+report.high.hrtfHash,EditorStyles.miniLabel,GUILayout.Height(85));
    }
    static void Row(string label,double low,double high) => Row(label,low.ToString("F4"),high.ToString("F4"));
    static void Row(string label,string low,string high)
    {
        using(new EditorGUILayout.HorizontalScope()) {EditorGUILayout.LabelField(label,GUILayout.Width(230)); EditorGUILayout.LabelField(low); EditorGUILayout.LabelField(high);}
    }
}
