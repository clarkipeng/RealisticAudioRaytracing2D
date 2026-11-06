using System;
using System.Collections.Generic;
using UnityEditor;
using UnityEngine;
using Kind = TransportSolver.RayEvent;

// A frozen probe of the actual transport. No audio playback or second ray solver.
public sealed class RayDebugWindow : EditorWindow
{
    RayTraceManager manager;
    int source, listener, bin = 1, rays = 64, selected = -1, pathFilter = -1;
    bool playing, showRejected;
    float time, duration, horizon, zoom = 1;
    Vector2 center;
    double previousUpdate;
    string status = "Open a 2D room, choose its manager, then Capture.";
    string error;
    AcousticTake.Scene scene;
    AcousticTake.Primitive[] geometry;
    AcousticTake.State[] states;
    TransportSolver.RaySegment[] segments;
    Vector3 emitter, receiver;
    string[] paths;
    int[] pathIds;
    static readonly Color[] colors = { Color.clear,
                                       new Color(.25f, .8f, 1),
                                       new Color(1, .75f, .3f),
                                       new Color(.35f, .9f, .55f),
                                       new Color(.85f, .5f, 1),
                                       Color.white,
                                       new Color(.4f, .45f, .5f),
                                       new Color(1, .35f, .35f) };

    [MenuItem("Acoustics/2D Ray Debugger")]
    public static void ShowWindow() => GetWindow<RayDebugWindow>("2D Ray Debugger");

    void OnEnable()
    {
        minSize = new Vector2(620, 400);
        manager = FindAnyObjectByType<RayTraceManager>();
        previousUpdate = EditorApplication.timeSinceStartup;
        EditorApplication.update += Tick;
    }
    void OnDisable() => EditorApplication.update -= Tick;
    void Tick()
    {
        double now = EditorApplication.timeSinceStartup;
        if (playing)
        {
            // One sweep per four wall-clock seconds, independent of acoustic delay range.
            time = Mathf.Min(duration, time + (float)(now - previousUpdate) * duration / 4);
            if (time >= duration)
                playing = false;
            Repaint();
        }
        previousUpdate = now;
    }

    public void Capture(RayTraceManager target, int sourceIndex, int listenerIndex, int frequencyBin,
                        int rayCount)
    {
        if (target == null || target.takeDimension != AcousticTake.Dimension.Planar2D)
            throw new InvalidOperationException(
                "Choose a manager configured for Planar2D (for example SmollRoom).");
        if (target.directShader == null || rayCount < 1 || sourceIndex < 0 ||
            sourceIndex >= target.takeSources.Count || listenerIndex < 0 ||
            listenerIndex >= target.takeListeners.Count)
            throw new InvalidOperationException(
                "Assign a transport shader, registered source/listener and positive ray count.");
        var settings = JsonUtility.FromJson<TakeRender.Settings>(JsonUtility.ToJson(target.directSettings));
        settings.transport = TakeRender.Transport.Raytraced;
        settings.raysPerPair = 0;
        settings.raysPerBin = rayCount;
        var sources = target.takeSources.ConvertAll(s => s.transform).ToArray();
        var clips = target.takeSources.ConvertAll(s => s.streamed ? null : s.clip).ToArray();
        var snapshot = SceneToTake.Compile(target.takeObjects, target.takeDimension,
                                           target.takeCurveTolerance, target.takeDefaultMaterial, sources,
                                           clips, target.takeListeners.ToArray(), out var primitives);
        var poses = new AcousticTake.State[snapshot.objects.Length];
        for (int i = 0; i < poses.Length; i++)
            poses[i] = SceneToTake.Observe(snapshot, i, target.takeObjects[i].transform);
        int rate =
            Application.isPlaying && target.monitorLive ? AudioSettings.outputSampleRate : target.sampleRate;
        using var solver = new TransportSolver(snapshot, primitives, settings, target.directShader,
                                               sampleRate: rate, readbackPaths: false);
        var captured =
            solver.CaptureRays(poses, sourceIndex * snapshot.listeners.Length + listenerIndex, frequencyBin);
        // Commit the view only after successful extraction and GPU readback.
        manager = target;
        source = sourceIndex;
        listener = listenerIndex;
        bin = frequencyBin;
        rays = rayCount;
        scene = snapshot;
        geometry = primitives;
        states = poses;
        segments = captured;
        emitter = poses[snapshot.sources[source].objectIndex].position;
        receiver = poses[snapshot.listeners[listener]].position;
        horizon = settings.maximumDelaySeconds;
        duration = 0;
        int count = 0, arrivals = 0;
        var ids = new SortedSet<int>();
        foreach (var segment in segments)
        {
            if (segment.kind == Kind.None)
                continue;
            count++;
            ids.Add((int)segment.path);
            if (segment.kind == Kind.Listener)
                arrivals++;
            if (segment.kind != Kind.Rejected && segment.kind != Kind.Escape)
                duration = Mathf.Max(duration, segment.end.w);
        }
        if (duration <= 0)
            duration = horizon;
        duration = Mathf.Min(duration, horizon);
        time = duration;
        playing = false;
        selected = -1;
        pathFilter = -1;
        var names = new List<string> { "All paths" };
        var values = new List<int> { -1 };
        foreach (int id in ids)
        {
            names.Add(PathName((uint)id));
            values.Add(id);
        }
        paths = names.ToArray();
        pathIds = values.ToArray();
        status =
            $"Frozen XY probe · {frequencyBin * (float)rate / settings.fftSize:0.#} Hz · {count} segments · {arrivals} listener arrivals";
        error = null;
        Fit();
        Repaint();
    }

    // Direct, first-image mirrors (1 + 2 * candidate) and complete routes (identity hashes).
    // The late energy field is a histogram, not captured rays.
    static string PathName(uint id) => id == 0 ? "Direct"
                                               : id % 2 == 1 && id < 0x10000 ? $"Mirror {(id - 1) / 2}" : $"Route {id:x8}";
    Vector3 World(AcousticTake.Primitive primitive, Vector3 point) => states[primitive.objectIndex].position
                                                                      + states[primitive.objectIndex].rotation
                                                                            * point;
    bool Active(AcousticTake.Primitive primitive) => (states[primitive.objectIndex].flags &
                                                      (AcousticTake.StateFlags.Active |
                                                       AcousticTake.StateFlags.Geometry)) ==
                                                     (AcousticTake.StateFlags.Active |
                                                      AcousticTake.StateFlags.Geometry);
    void Fit()
    {
        Vector2 min = Vector2.Min(emitter, receiver), max = Vector2.Max(emitter, receiver);
        foreach (var primitive in geometry)
        {
            if (!Active(primitive))
                continue;
            Vector2 a = World(primitive, primitive.a), b = World(primitive, primitive.b);
            min = Vector2.Min(min, Vector2.Min(a, b));
            max = Vector2.Max(max, Vector2.Max(a, b));
        }
        center = (min + max) * .5f;
        zoom = Mathf.Min(Mathf.Max(1, position.width - 100) / Mathf.Max(1, max.x - min.x),
                         Mathf.Max(1, position.height - 260) / Mathf.Max(1, max.y - min.y));
    }

    void OnGUI()
    {
        manager = (RayTraceManager)EditorGUILayout.ObjectField("Scene manager", manager,
                                                               typeof(RayTraceManager), true);
        if (manager != null)
        {
            using (new EditorGUILayout.HorizontalScope())
            {
                source = EditorGUILayout.Popup(
                    "Source", source,
                    manager.takeSources.ConvertAll(s => s == null ? "Missing" : s.name).ToArray());
                listener = EditorGUILayout.Popup(
                    "Listener", listener,
                    manager.takeListeners.ConvertAll(t => t == null ? "Missing" : t.name).ToArray());
            }
            using (new EditorGUILayout.HorizontalScope())
            {
                bin = EditorGUILayout.IntField("Frequency bin", bin);
                rays = EditorGUILayout.IntField("Probe rays / bin", rays);
                if (GUILayout.Button("Capture", GUILayout.Width(80)))
                    try
                    {
                        Capture(manager, source, listener, bin, rays);
                    }
                    catch (Exception e)
                    {
                        error = e.Message;
                        playing = false;
                    }
            }
        }
        if (error != null)
            EditorGUILayout.HelpBox(error, MessageType.Error);
        EditorGUILayout.LabelField(status, EditorStyles.miniLabel);
        if (segments == null)
            return;
        using (new EditorGUILayout.HorizontalScope())
        {
            if (GUILayout.Button(playing ? "Pause" : "Play", GUILayout.Width(55)))
            {
                if (time >= duration)
                    time = 0;
                playing = !playing;
            }
            EditorGUI.BeginChangeCheck();
            time = EditorGUILayout.Slider("Travel ms", time * 1000, 0, duration * 1000) / 1000;
            if (EditorGUI.EndChangeCheck())
                playing = false;
            GUILayout.Label(
                new GUIContent("Range ms", "Visible travel-time range; maximum is the solver delay horizon."),
                GUILayout.Width(60));
            duration = Mathf.Clamp(EditorGUILayout.FloatField(duration * 1000, GUILayout.Width(65)) / 1000,
                                   .000001f, horizon);
            time = Mathf.Min(time, duration);
        }
        using (new EditorGUILayout.HorizontalScope())
        {
            pathFilter = EditorGUILayout.IntPopup(pathFilter, paths, pathIds);
            showRejected = GUILayout.Toggle(showRejected, "Rejected candidates", GUILayout.Width(150));
            if (GUILayout.Button("Fit", GUILayout.Width(45)))
                Fit();
        }
        using (new EditorGUILayout.HorizontalScope()) for (int i = 1; i < colors.Length; i++)
        {
            var prior = GUI.contentColor;
            GUI.contentColor = colors[i];
            GUILayout.Label(((Kind)i).ToString(), EditorStyles.miniLabel);
            GUI.contentColor = prior;
        }
        var area = GUILayoutUtility.GetRect(1, 1, GUILayout.ExpandWidth(true), GUILayout.ExpandHeight(true));
        DrawCanvas(area);
        string detail = "Click a segment to inspect · wheel to zoom · drag to pan. Frozen geometry; " +
                        "motion/Doppler paused.";
        if (selected >= 0)
        {
            var s = segments[selected];
            string material = s.material >= 0 ? scene.materials[s.material].name : "—";
            detail =
                $"{PathName(s.path)} · {s.kind} · material {material}\n" +
                $"{s.start.w * 1000:0.###} → {s.end.w * 1000:0.###} ms · signed weight {s.pressure:G5} · ΔZ {s.end.z - s.start.z:0.###} m";
        }
        GUILayout.Label(detail, EditorStyles.miniLabel, GUILayout.Height(32));
        GUILayout.Label("Sampled paths, not a wave field. XY projection; extruded-edge samples keep their " +
                            "full 3D travel time.",
                        EditorStyles.miniLabel);
    }

    void DrawCanvas(Rect area)
    {
        EditorGUI.DrawRect(area, new Color(.065f, .075f, .09f));
        var e = Event.current;
        Vector2 Point(Vector3 p) => area.size * .5f + new Vector2(p.x - center.x, center.y - p.y) * zoom;
        if (area.Contains(e.mousePosition))
        {
            if (e.type == EventType.ScrollWheel)
            {
                var at = e.mousePosition - area.center;
                var before = new Vector2(at.x, -at.y) / zoom;
                zoom *= Mathf.Exp(-e.delta.y * .05f);
                center += before - new Vector2(at.x, -at.y) / zoom;
                e.Use();
                Repaint();
            }
            if (e.type == EventType.MouseDrag)
            {
                center += new Vector2(-e.delta.x, e.delta.y) / zoom;
                e.Use();
                Repaint();
            }
        }
        bool pick = area.Contains(e.mousePosition) && e.type == EventType.MouseDown && e.button == 0;
        float closest = 8;
        GUI.BeginClip(area);
        Handles.BeginGUI();
        Handles.color = new Color(.6f, .65f, .7f);
        foreach (var p in geometry)
            if (Active(p))
                Handles.DrawAAPolyLine(3, Point(World(p, p.a)), Point(World(p, p.b)));
        for (int i = 0; i < segments.Length; i++)
        {
            var s = segments[i];
            if (s.kind == Kind.None || s.start.w > time || (!showRejected && s.kind == Kind.Rejected) ||
                (pathFilter >= 0 && s.path != pathFilter))
                continue;
            float fraction =
                s.end.w > s.start.w ? Mathf.Clamp01((time - s.start.w) / (s.end.w - s.start.w)) : 1;
            Vector2 a = Point(s.start), b = Point(Vector3.Lerp(s.start, s.end, fraction));
            Handles.color = colors[(int)s.kind];
            if (s.kind == Kind.Rejected)
                Handles.DrawDottedLine(a, b, 5);
            else
                Handles.DrawAAPolyLine(selected == i ? 4 : 1.5f, a, b);
            if (s.kind != Kind.Escape && s.kind != Kind.Rejected)
                Handles.DrawSolidDisc(b, Vector3.forward, selected == i ? 4 : 2);
            if (pick)
            {
                float d = HandleUtility.DistancePointLine(e.mousePosition, a, b);
                if (d < closest)
                {
                    closest = d;
                    selected = i;
                }
            }
        }
        Handles.color = new Color(.3f, 1, .6f);
        Handles.DrawSolidDisc(Point(emitter), Vector3.forward, 5);
        Handles.color = Color.white;
        Handles.DrawWireDisc(Point(receiver), Vector3.forward, 6);
        GUI.Label(new Rect(Point(emitter) + Vector2.one * 8, new Vector2(100, 20)), "Source");
        GUI.Label(new Rect(Point(receiver) + Vector2.one * 8, new Vector2(100, 20)), "Listener");
        Handles.EndGUI();
        GUI.EndClip();
        if (pick)
        {
            if (closest == 8)
                selected = -1;
            e.Use();
            Repaint();
        }
    }
}
