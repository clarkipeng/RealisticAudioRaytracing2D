using System;
using System.IO;
using UnityEditor;
using UnityEngine;

[assembly: System.Runtime.CompilerServices.InternalsVisibleTo("Acoustics.Tests")]

[CustomEditor(typeof(RayTraceManager))]
public sealed class TakeEditor : Editor
{
    TakeRender.Settings renderSettings = new TakeRender.Settings();
    int comparisonStep=128;
    ComputeShader renderShader;
    bool checkReference;
    TakeComparison.ReferenceCheck referenceCheck=new TakeComparison.ReferenceCheck();

    void OnEnable()
    {
        var manager=(RayTraceManager)target;
        renderSettings=JsonUtility.FromJson<TakeRender.Settings>(JsonUtility.ToJson(manager.directSettings));
        renderShader=manager.directShader;
        comparisonStep=Math.Max(renderSettings.spectralHop,renderSettings.responseStepSamples/2);
    }

    public override void OnInspectorGUI()
    {
        var manager=(RayTraceManager)target;
        var station=manager.GetComponent<SceneSession>();
        using (new EditorGUI.DisabledScope(manager.Mode!=RayTraceManager.SessionMode.Idle && !manager.IsReplayingTake)) DrawConfiguration(manager);
        EditorGUILayout.Space();
        EditorGUILayout.LabelField("Recorded takes",EditorStyles.boldLabel);
        EditorGUILayout.HelpBox(manager.TakeStatus+" ("+manager.Mode+")",MessageType.Info);
        if (manager.Mode==RayTraceManager.SessionMode.Draining) EditorGUILayout.HelpBox("Finishing output recording. Keep Play mode running until this completes.",MessageType.Info);
        using (new EditorGUI.DisabledScope(Application.isPlaying))
            if (GUILayout.Button("Register scene objects / persist IDs")) Run(() =>
            {
                Undo.RecordObject(manager,"Register acoustic take objects");
                manager.RegisterTakeObjects(); EditorUtility.SetDirty(manager);
                PrefabUtility.RecordPrefabInstancePropertyModifications(manager);
            });
        using (new EditorGUI.DisabledScope(!Application.isPlaying))
        {
            if (station==null && manager.Mode==RayTraceManager.SessionMode.Idle && GUILayout.Button("Record take")) Run(() =>
            {
                string parent=EditorUtility.OpenFolderPanel("Save a new take under","Recordings","");
                if (string.IsNullOrEmpty(parent)) return;
                manager.BeginTakeRecording(Path.Combine(parent,"take-"+DateTime.UtcNow.ToString("yyyyMMdd-HHmmss-fff")));
            });
            if (manager.IsRecordingTake && GUILayout.Button("Save run")) Run(()=> {if(station!=null)station.SaveRecording();else manager.EndTakeRecording();});
            if(station!=null && !manager.IsRecordingTake && !station.Busy && GUILayout.Button("Replay saved run from beginning")) Run(()=>station.ReplayInputs(manager.takePath));
            if (manager.monitorLive && !manager.IsRecordingTake && !manager.IsReplayingTake)
            {
                if (manager.Mode==RayTraceManager.SessionMode.Idle && GUILayout.Button("Start live preview")) Run(manager.BeginLiveSimulation);
                if (manager.IsLiveSimulation && GUILayout.Button("Stop live preview")) Run(manager.StopLiveSimulation);
            }
            if (station==null && manager.Mode==RayTraceManager.SessionMode.Idle && GUILayout.Button("Replay recorded poses")) Run(() => manager.BeginTakeReplay(manager.takePath));
            if (manager.IsReplayingTake)
            {
                if (GUILayout.Button(manager.IsTakeReplayPlaying ? "Pause replay / enable scrubbing" : "Resume replay"))
                    Run(() => manager.SetTakeReplayPlaying(!manager.IsTakeReplayPlaying));
                EditorGUILayout.HelpBox("While paused, edit Replay Seconds above to scrub recorded poses.",MessageType.None);
                if (GUILayout.Button("Stop replay / restore scene")) Run(manager.StopTakeReplay);
            }
        }
        if (GUILayout.Button("Choose saved take"))
        {
            string path=EditorUtility.OpenFolderPanel("Choose a take directory",manager.takePath,"");
            if (!string.IsNullOrEmpty(path)) { Undo.RecordObject(manager,"Choose take"); manager.takePath=path; EditorUtility.SetDirty(manager); }
        }
        if(station!=null && !manager.IsRecordingTake && GUILayout.Button("Export input-only run")) Run(()=>
        {
            string parent=EditorUtility.OpenFolderPanel("Export inputs under","Recordings","");
            if(string.IsNullOrEmpty(parent))return;
            string output=Path.Combine(parent,"inputs-"+DateTime.UtcNow.ToString("yyyyMMdd-HHmmss-fff"));
            SceneRecording.ExportInputs(manager.takePath,output);Debug.Log("Input-only run: "+output,manager);
        });
        if (GUILayout.Button("Open audio comparison window")) TakeComparisonWindow.ShowWindow();
        if (manager.takeDimension == AcousticTake.Dimension.Planar2D && GUILayout.Button("Open 2D ray debugger")) RayDebugWindow.ShowWindow();
        EditorGUILayout.LabelField("Offline acoustic render",EditorStyles.boldLabel);
        DrawTransportSettings(renderSettings);
        EditorGUILayout.HelpBox("Renders saved inputs through the selected shared transport. Compact recordings first reconstruct their scene motion.",MessageType.None);
        renderShader=(ComputeShader)EditorGUILayout.ObjectField("Shader (direct CPU if empty)",renderShader,typeof(ComputeShader),false);
        renderSettings.listenerOutput=(TakeRender.ListenerOutput)EditorGUILayout.EnumPopup("Listener output",renderSettings.listenerOutput);
        renderSettings.useBvh=EditorGUILayout.Toggle("Static geometry BVH",renderSettings.useBvh);
        if(renderSettings.transport==TakeRender.Transport.Direct)
        {
        renderSettings.transitionSamples=EditorGUILayout.IntField("Transition (samples)",renderSettings.transitionSamples);
        renderSettings.delayFilter=(TakeRender.DelayFilter)EditorGUILayout.EnumPopup("Delay filter",renderSettings.delayFilter);
        if (renderSettings.delayFilter==TakeRender.DelayFilter.WindowedSinc) renderSettings.sincRadius=EditorGUILayout.IntField("Sinc radius / DSP latency",renderSettings.sincRadius);
        }
        renderSettings.maximumDelaySeconds=EditorGUILayout.FloatField("Maximum delay (seconds)",renderSettings.maximumDelaySeconds);
        renderSettings.responseStepSamples=EditorGUILayout.IntField("Response step (samples)",renderSettings.responseStepSamples);
        renderSettings.tailSamples=EditorGUILayout.IntField("Tail (samples)",renderSettings.tailSamples);
        renderSettings.speedOfSound=EditorGUILayout.FloatField("Sound speed (m/s)",renderSettings.speedOfSound);
        renderSettings.referenceDistance=EditorGUILayout.FloatField("Reference distance (m)",renderSettings.referenceDistance);
        renderSettings.minimumDistance=EditorGUILayout.FloatField("Minimum distance (m)",renderSettings.minimumDistance);
        using (new EditorGUI.DisabledScope(manager.Mode!=RayTraceManager.SessionMode.Idle))
            if (GUILayout.Button("Render saved take to responses + WAV")) Run(() =>
            {
                string output=Path.Combine(manager.takePath,"render-"+DateTime.UtcNow.ToString("yyyyMMdd-HHmmss-fff"));
                try
                {
                    using (var take=SceneReplay.Open(manager.takePath,output))
                        TakeRender.RenderDirect(take,renderSettings,output,p =>
                        {
                            if (EditorUtility.DisplayCancelableProgressBar("Rendering recorded take",output,p)) throw new OperationCanceledException();
                        },renderShader);
                    Debug.Log("Acoustic render saved to "+output,manager);
                }
                finally { EditorUtility.ClearProgressBar(); }
            });
        comparisonStep=EditorGUILayout.IntField("Comparison finer step",comparisonStep);
        checkReference=EditorGUILayout.Toggle("Check finer reference",checkReference);
        if (checkReference)
        {
            referenceCheck.responseStepSamples=EditorGUILayout.IntField("Reference response step",referenceCheck.responseStepSamples);
            referenceCheck.maximumRmsError=EditorGUILayout.DoubleField("Maximum reference RMS error",referenceCheck.maximumRmsError);
            referenceCheck.maximumAbsoluteError=EditorGUILayout.DoubleField("Maximum reference peak error",referenceCheck.maximumAbsoluteError);
        }
        using (new EditorGUI.DisabledScope(manager.Mode!=RayTraceManager.SessionMode.Idle))
            if (GUILayout.Button("Compare two response budgets")) Run(() =>
            {
                var high=JsonUtility.FromJson<TakeRender.Settings>(JsonUtility.ToJson(renderSettings)); high.responseStepSamples=comparisonStep;
                string output=Path.Combine(manager.takePath,"comparison-"+DateTime.UtcNow.ToString("yyyyMMdd-HHmmss-fff"));
                try
                {
                    using (var take=SceneReplay.Open(manager.takePath,output))
                        TakeComparison.Run(take,renderSettings,high,output,renderShader,p=>
                        { if (EditorUtility.DisplayCancelableProgressBar("Comparing temporal budgets",output,p)) throw new OperationCanceledException(); },checkReference ? referenceCheck:null);
                    Debug.Log("Comparison saved to "+output,manager);
                    TakeComparisonWindow.OpenComparison(Path.Combine(output,"comparison.json"));
                }
                finally { EditorUtility.ClearProgressBar(); }
            });

    }

    internal static void DrawTransportSettings(TakeRender.Settings settings)
    {
        settings.transport=(TakeRender.Transport)EditorGUILayout.EnumPopup("Transport",settings.transport);
        if(settings.transport!=TakeRender.Transport.Raytraced)return;
        settings.synthesis=(TakeRender.Synthesis)EditorGUILayout.EnumPopup("Synthesis",settings.synthesis);
        settings.audioBackend=(TakeRender.AudioBackend)EditorGUILayout.EnumPopup("Audio backend",settings.audioBackend);
        settings.raysPerBin=EditorGUILayout.IntField("Rays per frequency bin",settings.RaysPerBin);settings.raysPerPair=0;
        settings.seed=EditorGUILayout.IntField("Sampling seed",settings.seed);
        settings.maxBounces=EditorGUILayout.IntField("Maximum bounces (0 = direct only)",settings.maxBounces);
        settings.fftSize=EditorGUILayout.IntField("Acoustic frequency grid (FFT size)",settings.fftSize);
        settings.spectralHop=EditorGUILayout.IntField("Convolution block (samples)",settings.spectralHop);
        settings.diffractionFactor=EditorGUILayout.FloatField("Diffraction factor",settings.diffractionFactor);
        settings.airAbsorption=EditorGUILayout.Toggle("Air absorption",settings.airAbsorption);
    }
    void DrawConfiguration(RayTraceManager manager)
    {
        serializedObject.Update();
        foreach(string name in new[]{"monitorLive","audioManager","directShader","directSettings","monitorListener",
            "monitorLeadSamples","monitorCapacitySamples","takeDimension","takeRoot","takeDefaultMaterial","takeCurveTolerance",
            "takeObjects","takeSources","takeListeners","replayMotionWriters","takePath","replaySeconds"})
            EditorGUILayout.PropertyField(serializedObject.FindProperty(name),true);
        serializedObject.ApplyModifiedProperties();
        using(new EditorGUI.DisabledScope(true))EditorGUILayout.PropertyField(serializedObject.FindProperty("liveDiagnostics"),true);
        int rate=Application.isPlaying?AudioSettings.outputSampleRate:manager.sampleRate;
        EditorGUILayout.LabelField("Monitoring lead",(1000.0*manager.monitorLeadSamples/rate).ToString("F1")+" ms + device latency");
    }

    static void Run(Action action)
    {
        try { action(); }
        catch (Exception e) { Debug.LogException(e); }
    }
}

public static class TakeBatch
{
    [Serializable] public sealed class Job
    {
        public string take, output, shader;
        public TakeRender.Settings comparison;
        public TakeComparison.ReferenceCheck referenceCheck;
        public TakeRender.Settings settings = new TakeRender.Settings();
        public double workFraction=.2;
    }

    public static Job ParseJob(string json)
    {
        // Unity's inline serializer instantiates omitted optional objects, including reference jobs.
        var job=Newtonsoft.Json.JsonConvert.DeserializeObject<Job>(json);
        if(job==null || string.IsNullOrEmpty(job.take) || string.IsNullOrEmpty(job.output) || job.settings==null)
            throw new ArgumentException("Job requires take/output paths and render settings.");
        if(job.referenceCheck!=null && job.comparison==null)throw new ArgumentException("A reference check requires a comparison.");
        if(!double.IsFinite(job.workFraction)||job.workFraction<=0||job.workFraction>1)throw new ArgumentException("Work fraction must be in (0,1].");
        return job;
    }

    // Unity -batchmode -projectPath ... -executeMethod TakeBatch.Render -takeJob /absolute/job.json
    // The first baseline is CPU-only and can also run with -nographics.
    public static void Render()
    {
        try
        {
            var args=Environment.GetCommandLineArgs();
            int index=Array.IndexOf(args,"-takeJob");
            if (index < 0 || index+1 >= args.Length) throw new ArgumentException("Pass -takeJob /absolute/job.json");
            string path=Path.GetFullPath(args[index+1]);
            var job=ParseJob(File.ReadAllText(path));
            string root=Path.GetDirectoryName(path);
            string output=Path.GetFullPath(Path.Combine(root,job.output));
            using (var take=SceneReplay.Open(Path.GetFullPath(Path.Combine(root,job.take)),output))
            {
                ComputeShader shader=null;
                if (!string.IsNullOrEmpty(job.shader))
                {
                    shader=AssetDatabase.LoadAssetAtPath<ComputeShader>(job.shader);
                    if (shader==null) throw new ArgumentException("Cannot load job shader asset: "+job.shader);
                }
                var clock=System.Diagnostics.Stopwatch.StartNew();
                Action<float> pace=_=>
                {
                    // Progress runs after the batch's GPU readback completes.
                    int idle=(int)Math.Min(int.MaxValue,Math.Ceiling(clock.Elapsed.TotalMilliseconds*(1/job.workFraction-1)));
                    System.Threading.Thread.Sleep(idle);clock.Restart();
                };
                object report=job.comparison==null ? (object)TakeRender.RenderDirect(take,job.settings,output,pace,shader):TakeComparison.Run(take,job.settings,job.comparison,output,shader,pace,job.referenceCheck);
                Debug.Log(JsonUtility.ToJson(report,true));
            }
            if (Application.isBatchMode) EditorApplication.Exit(0);
        }
        catch (Exception e)
        {
            Debug.LogException(e);
            if (Application.isBatchMode) EditorApplication.Exit(1);
            else throw;
        }
    }
}

[CustomEditor(typeof(AudioManager))]
public sealed class AudioOutputEditor : Editor
{
    public override void OnInspectorGUI()
    {
        var audio=(AudioManager)target;
        serializedObject.Update();
        EditorGUILayout.PropertyField(serializedObject.FindProperty("paused"));
        serializedObject.ApplyModifiedProperties();
        var queue=audio.OutputQueue;
        if(queue!=null)EditorGUILayout.HelpBox("Buffered: "+Math.Max(0,queue.WrittenSample-queue.ReadSample)+
            " samples. Underrun / late / overflow: "+queue.UnderrunSamples+" / "+queue.LateSamples+" / "+queue.OverflowSamples,MessageType.None);
    }
}
