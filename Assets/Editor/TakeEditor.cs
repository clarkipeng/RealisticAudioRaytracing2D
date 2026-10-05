using System;
using System.IO;
using UnityEditor;
using UnityEngine;

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
    }

    public override void OnInspectorGUI()
    {
        var manager=(RayTraceManager)target;
        var station=manager.GetComponent<StationDemo>();
        using (new EditorGUI.DisabledScope(manager.Mode!=RayTraceManager.SessionMode.Idle && !manager.IsReplayingTake)) DrawDefaultInspector();
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
            if(station!=null && !manager.IsRecordingTake && !station.Busy && GUILayout.Button("Replay saved inputs from beginning")) Run(()=>station.ReplayInputs(manager.takePath));
            if (manager.useTakeRenderer && !manager.IsRecordingTake && !manager.IsReplayingTake)
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
            StationRecording.ExportInputs(manager.takePath,output);Debug.Log("Input-only run: "+output,manager);
        });
        if (GUILayout.Button("Open audio comparison window")) TakeComparisonWindow.ShowWindow();
        EditorGUILayout.LabelField("Offline direct-path baseline",EditorStyles.boldLabel);
        EditorGUILayout.HelpBox("Renders recorded 2D/3D inputs without rerunning gameplay. Direct sound and opaque occlusion only; this is not the legacy reflection model.",MessageType.None);
        renderShader=(ComputeShader)EditorGUILayout.ObjectField("Direct shader (empty = CPU)",renderShader,typeof(ComputeShader),false);
        renderSettings.listenerOutput=(TakeRender.ListenerOutput)EditorGUILayout.EnumPopup("Listener output",renderSettings.listenerOutput);
        renderSettings.transitionSamples=EditorGUILayout.IntField("Transition (samples)",renderSettings.transitionSamples);
        renderSettings.delayFilter=(TakeRender.DelayFilter)EditorGUILayout.EnumPopup("Delay filter",renderSettings.delayFilter);
        if (renderSettings.delayFilter==TakeRender.DelayFilter.WindowedSinc) renderSettings.sincRadius=EditorGUILayout.IntField("Sinc radius / DSP latency",renderSettings.sincRadius);
        renderSettings.maximumDelaySeconds=EditorGUILayout.FloatField("Maximum delay (seconds)",renderSettings.maximumDelaySeconds);
        renderSettings.responseStepSamples=EditorGUILayout.IntField("Response step (samples)",renderSettings.responseStepSamples);
        renderSettings.tailSamples=EditorGUILayout.IntField("Tail (samples)",renderSettings.tailSamples);
        renderSettings.speedOfSound=EditorGUILayout.FloatField("Sound speed (m/s)",renderSettings.speedOfSound);
        renderSettings.referenceDistance=EditorGUILayout.FloatField("Reference distance (m)",renderSettings.referenceDistance);
        renderSettings.minimumDistance=EditorGUILayout.FloatField("Minimum distance (m)",renderSettings.minimumDistance);
        using (new EditorGUI.DisabledScope(manager.Mode!=RayTraceManager.SessionMode.Idle))
            if (GUILayout.Button("Render saved take to responses + WAV")) Run(() =>
            {
                string output=Path.Combine(manager.takePath,"direct-"+DateTime.UtcNow.ToString("yyyyMMdd-HHmmss-fff"));
                try
                {
                    using (var take=AcousticTake.Open(manager.takePath))
                        TakeRender.RenderDirect(take,renderSettings,output,p =>
                        {
                            if (EditorUtility.DisplayCancelableProgressBar("Rendering recorded take",output,p)) throw new OperationCanceledException();
                        },renderShader);
                    Debug.Log("Direct baseline saved to "+output,manager);
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
                    using (var take=AcousticTake.Open(manager.takePath))
                        TakeComparison.Run(take,renderSettings,high,output,renderShader,p=>
                        { if (EditorUtility.DisplayCancelableProgressBar("Comparing temporal budgets",output,p)) throw new OperationCanceledException(); },checkReference ? referenceCheck:null);
                    Debug.Log("Comparison saved to "+output,manager);
                    TakeComparisonWindow.OpenComparison(Path.Combine(output,"comparison.json"));
                }
                finally { EditorUtility.ClearProgressBar(); }
            });

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
            var job=JsonUtility.FromJson<Job>(File.ReadAllText(path));
            if (job == null || string.IsNullOrEmpty(job.take) || string.IsNullOrEmpty(job.output)) throw new ArgumentException("Job requires take and output paths.");
            string root=Path.GetDirectoryName(path);
            using (var take=AcousticTake.Open(Path.GetFullPath(Path.Combine(root,job.take))))
            {
                ComputeShader shader=null;
                if (!string.IsNullOrEmpty(job.shader))
                {
                    shader=AssetDatabase.LoadAssetAtPath<ComputeShader>(job.shader);
                    if (shader==null) throw new ArgumentException("Cannot load job shader asset: "+job.shader);
                }
                string output=Path.GetFullPath(Path.Combine(root,job.output));
                object report=job.comparison==null ? (object)TakeRender.RenderDirect(take,job.settings,output,null,shader):TakeComparison.Run(take,job.settings,job.comparison,output,shader,null,job.referenceCheck);
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
