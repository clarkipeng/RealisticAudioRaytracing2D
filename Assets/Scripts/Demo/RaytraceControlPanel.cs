using System;
using System.IO;
using System.Globalization;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.EventSystems;
using UnityEngine.UI;

// The demo UI, the same in every scene: a HUD while walking (H hides it), and a menu while the cursor is free (Escape).
// In the live showcase (PathLive) the menu is the scene picker; in authored scenes it holds the session's runs, voice and
// multiplayer; in the 2D rooms (no SceneSession) it starts and stops the audio.
public class RaytraceControlPanel : MonoBehaviour
{
    public RayTraceManager rayTraceManager;
    public SceneSession session;
    public bool buildOnStart = true;
    [Header("Layout")]
    public Vector2 anchoredPosition = new Vector2(24, -24);
    public Vector2 buttonSize = new Vector2(280, 32);
    public int spacing = 6;

    static readonly Color Ink = new Color(.9f, .94f, .94f);
    static readonly Color Muted = new Color(.58f, .68f, .7f);
    static readonly Color Surface = new Color(.13f, .2f, .23f);
    static readonly Color Accent = new Color(.25f, .75f, .67f);
    static readonly Color Backdrop = new Color(.035f, .065f, .08f, .97f);
    Font font;
    GameObject controls, connectionSetup, hudBox;
    Text status, hudTitle, hudStatus, hudKeys, microphoneStatus, networkStatus, error;
    PathLive live;
    bool hudVisible = true;
    Button save, restart, replay, microphone, host, join, copyCode;
    Dropdown takes, mode;
    InputField address;
    readonly List<string> takePaths = new List<string>();
    float nextUpdate;
    bool wasRecording;

    void Start()
    {
        if (rayTraceManager == null) rayTraceManager = FindAnyObjectByType<RayTraceManager>();
        if (buildOnStart) Build();
    }

    public void Build()
    {
        if (controls != null) return;
        if (FindAnyObjectByType<EventSystem>() == null)
            new GameObject("EventSystem", typeof(EventSystem), typeof(StandaloneInputModule));
        font = Resources.GetBuiltinResource<Font>("LegacyRuntime.ttf");
        var canvas = new GameObject("Raytrace Controls", typeof(Canvas), typeof(CanvasScaler), typeof(GraphicRaycaster));
        canvas.transform.SetParent(transform, false);
        canvas.GetComponent<Canvas>().renderMode = RenderMode.ScreenSpaceOverlay;
        canvas.GetComponent<Canvas>().sortingOrder = 100;
        var scaler = canvas.GetComponent<CanvasScaler>();
        scaler.uiScaleMode = CanvasScaler.ScaleMode.ScaleWithScreenSize;
        scaler.referenceResolution = new Vector2(1280, 720);
        scaler.screenMatchMode = CanvasScaler.ScreenMatchMode.Expand;
        var panel = Stack(canvas.transform, "Session controls");
        controls = panel.gameObject;
        panel.anchorMin = panel.anchorMax = panel.pivot = new Vector2(0, 1);
        panel.anchoredPosition = anchoredPosition;
        panel.sizeDelta = new Vector2(buttonSize.x + 32, 0);
        panel.GetComponent<VerticalLayoutGroup>().padding = new RectOffset(16, 16, 14, 14);
        panel.gameObject.AddComponent<Image>().color = Backdrop;
        if (session == null)
        {
            Label(panel, "2D ROOM", 12, 22).color = Accent;
            Label(panel, "A planar scene for the 2D transport and the 2D Ray Debugger.", 13, 0).color = Muted;
            if (rayTraceManager != null)
            {
                var actions2D = Row(panel);
                AddButton(actions2D, "Play", () => { foreach (var source in rayTraceManager.takeSources) source.Play(); });
                AddButton(actions2D, "Stop", rayTraceManager.StopLiveSimulation);
            }
            if (FindAnyObjectByType<Oscillator>() != null) Label(panel, "M moves the source", 13, 22).color = Muted;
            return;
        }

        live = session.GetComponent<PathLive>();
        if (live != null) BuildScenePicker(panel);
        else BuildSessionMenu(panel);
        BuildSettings(panel);
        BuildHud(canvas.transform);
        wasRecording = rayTraceManager.IsRecordingTake;
        RefreshState();
    }

    // The live showcase's menu: its scenes, the current one marked.
    void BuildScenePicker(Transform panel)
    {
        Label(panel, "LIVE ACOUSTIC RAY TRACING", 12, 22).color = Accent;
        Label(panel, "Path method: persistent rays, a learned denoiser and a GPU reverb, live.", 13, 0).color = Muted;
        Label(panel, "SCENES", 12, 26).color = Muted;
        var buttons = new List<Button>();
        foreach (var (title, path) in PathLive.Scenes)
        {
            bool here = path == session.gameObject.scene.path;
            Button button = null;
            button = AddButton(panel, here ? title + "   ·   you are here" : title, () => {
                if (session.Busy) return;
                foreach (var b in buttons) b.interactable = false;
                Caption(button).text = "Loading " + title + "…";
                session.LoadScene(path);
            });
            buttons.Add(button);
            button.interactable = !here;
            if (here)
            {
                var colors = button.colors; colors.disabledColor = Color.white; button.colors = colors;
                button.GetComponent<Image>().color = new Color(.12f, .39f, .36f);
            }
        }
    }

    // The authored scenes' menu: the session's runs, then voice and multiplayer.
    void BuildSessionMenu(Transform panel)
    {
        Label(panel, "SESSION", 12, 22).color = Accent;
        status = Label(panel, "Recording from scene start", 15, 22);
        var actions = Row(panel);
        save = AddButton(actions, "Save run  [R]", session.SaveRecording);
        save.GetComponent<Image>().color = new Color(.12f, .39f, .36f);
        restart = AddButton(actions, "New run", () => {
            if (session.multiplayer != null) session.multiplayer.Restart(null);
            else session.RestartScene();
        });
        Label(panel, "REPLAY", 12, 22).color = Muted;
        takes = AddDropdown(panel, "Saved runs");
        RefreshTakes();
        replay = AddButton(panel, "Replay selected run", () => {
            try { session.ReplayInputs(takePaths[takes.value]); }
            catch (Exception e) { error.text = e.Message; error.gameObject.SetActive(true); }
        });
        error = Label(panel, "", 13, 0);
        error.color = new Color(1, .7f, .5f);
        error.gameObject.SetActive(false);
        if (session.multiplayer != null)
        {
            var network = session.multiplayer;
            Label(panel, "VOICE & MULTIPLAYER", 12, 22).color = Muted;
            microphone = AddButton(panel, "Enable microphone", network.microphone.Toggle);
            microphoneStatus = Label(panel, "", 13, 0);
            networkStatus = Label(panel, "Solo", 13, 0);
            var setup = Stack(panel, "Connection setup"); connectionSetup = setup.gameObject;
            mode = AddDropdown(setup, "Connection type", "Online / join code", "LAN / IP address");
            mode.SetValueWithoutNotify(network.UsesRelay ? 0 : 1);
            address = Mount(DefaultControls.CreateInputField(new DefaultControls.Resources()), setup, "Join code or address").GetComponent<InputField>();
            address.characterLimit = 255;
            address.placeholder.color = Muted; ((Text)address.placeholder).fontStyle = FontStyle.Normal;
            address.onValueChanged.AddListener(_ => UpdateJoin());
            mode.onValueChanged.AddListener(_ => SetAddress());
            SetAddress();
            var connections = Row(setup);
            host = AddButton(connections, "Host", () => network.Restart(mode.value == 0 ? "relay-host" : "host"));
            join = AddButton(connections, "Join", () => {
                network.address = address.text.Trim();
                network.Restart(mode.value == 0 ? "relay-client" : "client");
            });
            copyCode = AddButton(panel, "Copy join code", () => GUIUtility.systemCopyBuffer = network.JoinCode);
            copyCode.gameObject.SetActive(false);
        }
    }

    void BuildSettings(Transform panel)
    {
        Label(panel, "MOUSE", 12, 26).color = Muted;
        var look = session.player.GetComponentInChildren<MouseLook>();
        var sensitivityLabel = Label(panel, "", 13, 24); sensitivityLabel.color = Muted;
        var sensitivity = Mount(DefaultControls.CreateSlider(new DefaultControls.Resources()), panel, "Mouse sensitivity", 18).GetComponent<Slider>();
        sensitivity.minValue = 30; sensitivity.maxValue = 360; sensitivity.wholeNumbers = true;
        sensitivity.onValueChanged.AddListener(value => {
            look.sensitivity = value; sensitivityLabel.text = "Mouse sensitivity   " + value.ToString("F0");
        });
        sensitivity.value = look.sensitivity;
        sensitivityLabel.text = "Mouse sensitivity   " + look.sensitivity.ToString("F0");
        sensitivity.fillRect.GetComponent<Image>().color = Accent;
        sensitivity.handleRect.GetComponent<Image>().color = Ink;
        Label(panel, "Click the scene to walk   ·   Esc for this menu", 13, 26).color = Muted;
    }

    // The HUD while walking: the scene, its live figures or the session's state, and the keys.
    void BuildHud(Transform canvas)
    {
        var box = Stack(canvas, "HUD"); hudBox = box.gameObject;
        box.anchorMin = box.anchorMax = box.pivot = new Vector2(0, 1);
        box.anchoredPosition = anchoredPosition;
        var layout = box.GetComponent<VerticalLayoutGroup>(); layout.padding = new RectOffset(14, 14, 10, 10); layout.spacing = 3;
        box.GetComponent<ContentSizeFitter>().horizontalFit = ContentSizeFitter.FitMode.PreferredSize;
        box.gameObject.AddComponent<Image>().color = new Color(Backdrop.r, Backdrop.g, Backdrop.b, .8f);
        hudTitle = Label(box, PathLive.Title(session.gameObject.scene) + (live != null ? "   ·   path method, live" : ""), 18, 0);
        hudTitle.fontStyle = FontStyle.Bold;
        hudStatus = Label(box, "", 15, 0);
        hudKeys = Label(box, live != null ? "WASD move   ·   Mouse look   ·   E doors   ·   N denoiser   ·   H hide HUD   ·   Esc menu"
                                          : "WASD move   ·   Mouse look   ·   E doors   ·   R save run   ·   H hide HUD   ·   Esc menu", 14, 0);
        hudKeys.color = Muted;
        overlay = FindAnyObjectByType<AudioSourceOverlay>();
    }
    AudioSourceOverlay overlay;

    bool Walking => live != null && live.IsSmokeRun ? !live.SmokeShowsMenu : Cursor.lockState == CursorLockMode.Locked;

    void Update()
    {
        if (session == null || hudBox == null) return;
        // H toggles the HUD and the source markers together (while walking, so typing in the menu's fields is safe).
        if (Walking && Input.GetKeyDown(KeyCode.H))
        {
            hudVisible = !hudVisible;
            if (overlay != null) overlay.visible = hudVisible;
            nextUpdate = 0;
        }
        if (Time.unscaledTime < nextUpdate) return;
        nextUpdate = Time.unscaledTime + .1f;
        RefreshState();
    }

    void RefreshState()
    {
        bool walking = Walking;
        controls.SetActive(!walking); hudBox.SetActive(walking && hudVisible);
        if (live != null)
        {
            hudStatus.text = string.IsNullOrEmpty(live.StageTimes) ? live.Stats : live.Stats + "\n" + live.StageTimes;
            return;
        }
        bool waiting = session.Busy || rayTraceManager.Mode == RayTraceManager.SessionMode.Draining;
        bool recording = rayTraceManager.IsRecordingTake, playing = session.IsInputReplay;
        status.text = waiting ? "Finishing audio…" : playing ? (session.InputReplayFinished ? "Replay finished" : "Replaying saved run")
            : recording ? "Recording from scene start" : rayTraceManager.RecordingFailed ? "Recording failed — see diagnostics" : rayTraceManager.IsLiveSimulation ? "Run saved · live audio continues" : "Session stopped";
        save.interactable = !waiting && recording && !playing;
        restart.interactable = !waiting;
        Caption(restart).text = playing ? "Return to live" : session.multiplayer != null && session.multiplayer.Connected ? "Leave session" : "New run";
        replay.interactable = !waiting && !recording && takePaths.Count > 0;
        takes.interactable = !waiting && takePaths.Count > 0;
        Caption(replay).text = recording ? "Save current run to replay" : "Replay selected run";
        if (wasRecording && !recording) RefreshTakes();
        wasRecording = recording;
        hudStatus.text = waiting ? "Finishing audio…" : recording ? "Recording this run   ·   R saves it" : status.text;
        if (microphone == null) return;
        var network = session.multiplayer; var mic = network.microphone;
        microphone.interactable = !playing && !waiting && rayTraceManager.IsLiveSimulation;
        Caption(microphone).text = mic.Active ? "Disable microphone" : "Enable microphone";
        microphone.GetComponent<Image>().color = mic.Active ? new Color(.12f, .39f, .36f) : Surface;
        microphoneStatus.text = playing ? "Recorded voice · microphone disabled" : mic.Status;
        microphoneStatus.gameObject.SetActive(playing || (mic.Status != "Microphone off" && !mic.Active));
        if (mic.Active) hudStatus.text += "   ·   microphone on";
        networkStatus.text = network.Status + (network.Connected ? "\n" + network.ConnectedPlayers + " / " + network.Capacity + " players" : "");
        connectionSetup.SetActive(!network.Connected && !playing);
        host.interactable = !waiting && !network.Connecting;
        UpdateJoin();
        copyCode.gameObject.SetActive(network.Connected && network.JoinCode != null);
    }

    void SetAddress()
    {
        address.SetTextWithoutNotify(mode.value == 0 ? "" : session.multiplayer.UsesRelay ? "127.0.0.1" : session.multiplayer.address);
        ((Text)address.placeholder).text = mode.value == 0 ? "Enter host’s join code" : "Host IP address";
        UpdateJoin();
    }
    void UpdateJoin()
    {
        if (join != null) join.interactable = !session.Busy && !session.multiplayer.Connecting && !string.IsNullOrWhiteSpace(address.text);
    }
    void RefreshTakes()
    {
        takePaths.Clear(); var names = new List<string>();
        if (Directory.Exists(SceneSession.RecordingsDirectory))
        {
            var paths = Directory.GetDirectories(SceneSession.RecordingsDirectory);
            Array.Sort(paths, StringComparer.Ordinal); Array.Reverse(paths);
            foreach (string path in paths)
            {
                try
                {
                    var meta = AcousticTake.ReadManifest(path); if (!meta.complete) continue;
                    string scene = meta.session?.scene;
                    if (scene == null && File.Exists(Path.Combine(path, "input.json")))
                        scene = JsonUtility.FromJson<SceneRecording.Manifest>(File.ReadAllText(Path.Combine(path, "input.json"))).scene;
                    if (scene != session.gameObject.scene.path) continue;
                    takePaths.Add(path);
                    string name = Path.GetFileName(path);
                    names.Add(name.Length >= 19 && DateTime.TryParseExact(name.Substring(name.Length - 19),
                        "yyyyMMdd-HHmmss-fff", CultureInfo.InvariantCulture, DateTimeStyles.AssumeUniversal, out var time)
                        ? time.ToLocalTime().ToString("MMM d · HH:mm:ss.fff") : name);
                }
                catch (Exception e) when (e is IOException || e is InvalidDataException || e is ArgumentException) { }
            }
        }
        takes.ClearOptions(); takes.AddOptions(names.Count > 0 ? names : new List<string> { "No saved runs in this scene" });
    }

    RectTransform Stack(Transform parent, string name)
    {
        var obj = new GameObject(name, typeof(RectTransform), typeof(VerticalLayoutGroup), typeof(ContentSizeFitter));
        obj.transform.SetParent(parent, false);
        var rect = obj.GetComponent<RectTransform>(); rect.sizeDelta = new Vector2(buttonSize.x, 0);
        var layout = obj.GetComponent<VerticalLayoutGroup>(); layout.spacing = spacing;
        layout.childControlWidth = true; layout.childControlHeight = true;
        layout.childForceExpandWidth = true; layout.childForceExpandHeight = false;
        obj.GetComponent<ContentSizeFitter>().verticalFit = ContentSizeFitter.FitMode.PreferredSize;
        return rect;
    }
    Transform Row(Transform parent)
    {
        var obj = new GameObject("Actions", typeof(RectTransform), typeof(HorizontalLayoutGroup), typeof(LayoutElement));
        obj.transform.SetParent(parent, false); obj.GetComponent<LayoutElement>().preferredHeight = buttonSize.y;
        var layout = obj.GetComponent<HorizontalLayoutGroup>(); layout.spacing = spacing;
        layout.childControlWidth = true; layout.childControlHeight = true; layout.childForceExpandWidth = true;
        return obj.transform;
    }
    GameObject Mount(GameObject obj, Transform parent, string name, float height = -1)
    {
        obj.name = name; obj.transform.SetParent(parent, false);
        obj.AddComponent<LayoutElement>().preferredHeight = height < 0 ? buttonSize.y : height;
        foreach (var image in obj.GetComponentsInChildren<Image>(true)) image.color = Surface;
        foreach (var text in obj.GetComponentsInChildren<Text>(true))
        { text.font = font; text.fontSize = 14; text.color = Ink; text.raycastTarget = false; }
        var selectable = obj.GetComponent<Selectable>();
        if (selectable != null)
        {
            var colors = selectable.colors;
            colors.highlightedColor = new Color(1.3f, 1.4f, 1.4f);
            colors.selectedColor = colors.highlightedColor; colors.pressedColor = new Color(.7f, .85f, .85f);
            colors.disabledColor = new Color(.55f, .55f, .55f, .7f); selectable.colors = colors;
            selectable.navigation = new Navigation { mode = Navigation.Mode.Automatic };
        }
        return obj;
    }
    Dropdown AddDropdown(Transform parent, string name, params string[] options)
    {
        var dropdown = Mount(DefaultControls.CreateDropdown(new DefaultControls.Resources()), parent, name).GetComponent<Dropdown>();
        dropdown.ClearOptions(); dropdown.AddOptions(new List<string>(options));
        dropdown.template.sizeDelta = new Vector2(0, 150);
        dropdown.captionText.rectTransform.offsetMin = new Vector2(10, 0);
        var arrow = dropdown.transform.Find("Arrow"); arrow.GetComponent<Image>().enabled = false;
        var mark = Label(arrow, "▾", 14, 0); mark.color = Muted; mark.alignment = TextAnchor.MiddleCenter;
        mark.rectTransform.anchorMin = Vector2.zero; mark.rectTransform.anchorMax = Vector2.one;
        mark.rectTransform.offsetMin = mark.rectTransform.offsetMax = Vector2.zero;
        dropdown.template.GetComponentInChildren<Toggle>(true).graphic.color = Accent;
        return dropdown;
    }
    Text Label(Transform parent, string value, int size, float height)
    {
        var obj = new GameObject("Label", typeof(RectTransform), typeof(Text), typeof(LayoutElement)); obj.transform.SetParent(parent, false);
        var text = obj.GetComponent<Text>(); text.font = font; text.fontSize = size; text.text = value;
        text.color = Ink; text.raycastTarget = false; text.alignment = TextAnchor.MiddleLeft;
        if (height > 0) obj.GetComponent<LayoutElement>().preferredHeight = height;
        return text;
    }
    Button AddButton(Transform parent, string text, UnityEngine.Events.UnityAction action)
    {
        var button = Mount(DefaultControls.CreateButton(new DefaultControls.Resources()), parent, text).GetComponent<Button>();
        Caption(button).text = text; button.onClick.AddListener(action); return button;
    }
    static Text Caption(Button button) => button.GetComponentInChildren<Text>();
}
