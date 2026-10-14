using System;
using System.IO;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.EventSystems;
using UnityEngine.UI;

public class RaytraceControlPanel : MonoBehaviour
{
    public RayTraceManager rayTraceManager;
    public SceneSession session;
    public bool buildOnStart = true;

    [Header("Layout")]
    public Vector2 anchoredPosition = new Vector2(16, -16);
    public Vector2 buttonSize = new Vector2(180, 32);
    public int spacing = 8;

    Font buttonFont;
    Text recordLabel;
    GameObject sensitivityControls;
    float nextUiUpdate;
    Text microphoneLabel,networkLabel;
    Dropdown takes;
    Button replayButton;
    Button copyJoinCode;
    readonly List<string> takePaths=new List<string>();
    bool wasRecording;
    void RefreshTakes()
    {
        if(takes==null)return;
        takePaths.Clear();var names=new List<string>();
        if(Directory.Exists(SceneSession.RecordingsDirectory))
        {
            var paths=Directory.GetDirectories(SceneSession.RecordingsDirectory);Array.Sort(paths,StringComparer.Ordinal);Array.Reverse(paths);
            foreach(string path in paths)
            {
                try
                {
                    var meta=AcousticTake.ReadManifest(path);if(!meta.complete)continue;
                    string scene=meta.session?.scene;
                    if(scene==null && File.Exists(Path.Combine(path,"input.json")))scene=JsonUtility.FromJson<SceneRecording.Manifest>(File.ReadAllText(Path.Combine(path,"input.json"))).scene;
                    if(scene!=session.gameObject.scene.path)continue;
                    takePaths.Add(path);names.Add(Path.GetFileName(path));
                }
                catch(Exception e) when(e is IOException || e is InvalidDataException || e is ArgumentException) {} // Invalid/unfinished runs are not replay choices.
            }
        }
        takes.ClearOptions();takes.AddOptions(names);
    }

    void Start()
    {
        if (rayTraceManager == null) rayTraceManager = FindObjectOfType<RayTraceManager>();
        if (buildOnStart) Build();
    }

    public void Build()
    {
        EnsureEventSystem();
        buttonFont = Resources.GetBuiltinResource<Font>("LegacyRuntime.ttf");

        Transform panel = CreatePanel();
        if(session!=null)
        {
            recordLabel=AddButton(panel,"Save run  [R]",session.SaveRecording);
            sensitivityControls=new GameObject("Look settings",typeof(RectTransform),typeof(VerticalLayoutGroup));
            sensitivityControls.transform.SetParent(panel,false);
            sensitivityControls.GetComponent<RectTransform>().sizeDelta=new Vector2(buttonSize.x,40);
            var layout=sensitivityControls.GetComponent<VerticalLayoutGroup>(); layout.childControlHeight=false; layout.childForceExpandHeight=false;
            var look=session.player.GetComponentInChildren<MouseLook>();
            var sensitivityLabel=AddLabel(sensitivityControls.transform,"Mouse sensitivity: "+look.sensitivity.ToString("F0"),13,18);
            var sensitivityObject=DefaultControls.CreateSlider(new DefaultControls.Resources()); sensitivityObject.name="Mouse sensitivity"; sensitivityObject.transform.SetParent(sensitivityControls.transform,false);
            sensitivityObject.GetComponent<RectTransform>().sizeDelta=new Vector2(buttonSize.x,18);
            var sensitivity=sensitivityObject.GetComponent<Slider>(); sensitivity.minValue=30; sensitivity.maxValue=360; sensitivity.value=look.sensitivity;
            sensitivity.onValueChanged.AddListener(value=> {look.sensitivity=value; sensitivityLabel.text="Mouse sensitivity: "+value.ToString("F0");});
            if(session.multiplayer!=null)
            {
                var network=session.multiplayer;
                microphoneLabel=AddButton(sensitivityControls.transform,"Microphone off",()=> {if(!session.IsInputReplay)network.microphone.Toggle();});
                networkLabel=AddLabel(sensitivityControls.transform,"Solo",12,32);
                var addressObject=DefaultControls.CreateInputField(new DefaultControls.Resources());addressObject.transform.SetParent(sensitivityControls.transform,false);
                addressObject.GetComponent<RectTransform>().sizeDelta=buttonSize;
                var address=addressObject.GetComponent<InputField>();address.text="";address.placeholder.GetComponent<Text>().text="Join code";
                var modeObject=DefaultControls.CreateDropdown(new DefaultControls.Resources());modeObject.transform.SetParent(sensitivityControls.transform,false);
                modeObject.GetComponent<RectTransform>().sizeDelta=buttonSize;
                var mode=modeObject.GetComponent<Dropdown>();mode.ClearOptions();mode.AddOptions(new System.Collections.Generic.List<string>{"Online / join code","LAN / address"});
                mode.onValueChanged.AddListener(value=> {address.text=value==0?network.JoinCode??"":network.address;address.placeholder.GetComponent<Text>().text=value==0?"Join code":"IP address";});
                AddButton(sensitivityControls.transform,"Host",()=>network.Restart(mode.value==0?"relay-host":"host"));
                AddButton(sensitivityControls.transform,"Join",()=> {network.address=address.text.Trim();network.Restart(mode.value==0?"relay-client":"client");});
                copyJoinCode=AddButton(sensitivityControls.transform,"Copy join code",()=> {GUIUtility.systemCopyBuffer=network.JoinCode;address.text=network.JoinCode;}).GetComponentInParent<Button>();
                copyJoinCode.gameObject.SetActive(false);
                AddButton(sensitivityControls.transform,"Solo / restart",()=>network.Restart(null));
            }
            var chooser=DefaultControls.CreateDropdown(new DefaultControls.Resources());chooser.transform.SetParent(sensitivityControls.transform,false);
            chooser.GetComponent<RectTransform>().sizeDelta=buttonSize;takes=chooser.GetComponent<Dropdown>();RefreshTakes();
            replayButton=AddButton(sensitivityControls.transform,"Replay saved run",()=> {
                if(takePaths.Count==0 || session.Busy)return;
                session.ReplayInputs(takePaths[takes.value]);
            }).GetComponentInParent<Button>(true);
            wasRecording=rayTraceManager.IsRecordingTake;
            sensitivityControls.AddComponent<ContentSizeFitter>().verticalFit=ContentSizeFitter.FitMode.PreferredSize;
            return;
        }
        if (rayTraceManager != null)
        {
            AddButton(panel, "Play", () => {
                foreach (var source in rayTraceManager.takeSources) source.Play();
            });
            AddButton(panel, "Stop", rayTraceManager.StopLiveSimulation);
        }
    }

    void Update()
    {
        if(recordLabel==null || Time.unscaledTime<nextUiUpdate) return;
        nextUiUpdate=Time.unscaledTime+.2f;
        bool waiting=session.Busy || rayTraceManager.Mode==RayTraceManager.SessionMode.Draining;
        bool recording=rayTraceManager.IsRecordingTake;
        recordLabel.text=waiting?"…":session.IsInputReplay?(session.InputReplayFinished?"Replay finished":"Replaying"):recording?"Save run  [R]":rayTraceManager.RecordingFailed?"Save failed":"Saved";
        recordLabel.GetComponentInParent<Button>(true).interactable=!waiting && recording && !rayTraceManager.IsReplayingTake && !session.IsInputReplay;
        sensitivityControls.SetActive(Cursor.lockState!=CursorLockMode.Locked);
        if(replayButton!=null)replayButton.interactable=!waiting && !recording && takePaths.Count>0;
        if(wasRecording && !recording)RefreshTakes();wasRecording=recording;
        if(microphoneLabel!=null)
        {
            copyJoinCode.gameObject.SetActive(session.multiplayer.Connected && session.multiplayer.JoinCode!=null);
            microphoneLabel.text=session.multiplayer.microphone.Status;
            microphoneLabel.GetComponentInParent<Button>(true).interactable=!session.IsInputReplay && !waiting;
            networkLabel.text=session.multiplayer.Status+(session.multiplayer.Connected?" ("+session.multiplayer.ConnectedPlayers+"/"+session.multiplayer.Capacity+")":"");
        }
    }

    Text AddLabel(Transform parent,string value,int size,float height)
    {
        var obj=new GameObject("Label",typeof(RectTransform),typeof(Text)); obj.transform.SetParent(parent,false);
        obj.GetComponent<RectTransform>().sizeDelta=new Vector2(buttonSize.x,height);
        var label=obj.GetComponent<Text>(); label.font=buttonFont; label.fontSize=size;
        label.text=value; label.color=new Color(.92f,.96f,1); label.raycastTarget=false;
        label.alignment=TextAnchor.MiddleLeft; return label;
    }

    Transform CreatePanel()
    {
        var canvasObject = new GameObject("Raytrace Controls", typeof(Canvas), typeof(CanvasScaler), typeof(GraphicRaycaster));
        canvasObject.transform.SetParent(transform, false);

        Canvas canvas = canvasObject.GetComponent<Canvas>();
        canvas.renderMode = RenderMode.ScreenSpaceOverlay;
        canvas.sortingOrder = 100;

        CanvasScaler scaler = canvasObject.GetComponent<CanvasScaler>();
        scaler.uiScaleMode = CanvasScaler.ScaleMode.ScaleWithScreenSize;
        scaler.referenceResolution = session!=null ? new Vector2(1280,720):new Vector2(1920,1080);

        var panelObject = new GameObject("Panel", typeof(RectTransform), typeof(VerticalLayoutGroup));
        panelObject.transform.SetParent(canvasObject.transform, false);

        RectTransform rect = panelObject.GetComponent<RectTransform>();
        rect.anchorMin = new Vector2(0, 1);
        rect.anchorMax = new Vector2(0, 1);
        rect.pivot = new Vector2(0, 1);
        rect.anchoredPosition = anchoredPosition;

        VerticalLayoutGroup layout = panelObject.GetComponent<VerticalLayoutGroup>();
        layout.spacing = spacing;
        if(session!=null)
        {
            rect.sizeDelta=new Vector2(buttonSize.x+24,0);
            panelObject.AddComponent<ContentSizeFitter>().verticalFit=ContentSizeFitter.FitMode.PreferredSize;
            layout.padding=new RectOffset(12,12,12,12);
            panelObject.AddComponent<Image>().color=new Color(.025f,.055f,.075f,.92f);
        }
        layout.childControlWidth = false;
        layout.childControlHeight = false;
        layout.childForceExpandWidth = false;
        layout.childForceExpandHeight = false;

        return panelObject.transform;
    }

    Text AddButton(Transform parent, string label, UnityEngine.Events.UnityAction action)
    {
        var buttonObject = new GameObject(label, typeof(RectTransform), typeof(Image), typeof(Button));
        buttonObject.transform.SetParent(parent, false);

        RectTransform rect = buttonObject.GetComponent<RectTransform>();
        rect.sizeDelta = buttonSize;

        Image image = buttonObject.GetComponent<Image>();
        image.color = new Color(0.12f, 0.14f, 0.18f, 0.92f);

        Button button = buttonObject.GetComponent<Button>();
        button.onClick.AddListener(action);

        var textObject = new GameObject("Text", typeof(RectTransform), typeof(Text));
        textObject.transform.SetParent(buttonObject.transform, false);

        RectTransform textRect = textObject.GetComponent<RectTransform>();
        textRect.anchorMin = Vector2.zero;
        textRect.anchorMax = Vector2.one;
        textRect.offsetMin = Vector2.zero;
        textRect.offsetMax = Vector2.zero;

        Text text = textObject.GetComponent<Text>();
        text.text = label;
        text.font = buttonFont;
        text.fontSize = 16;
        text.color = Color.white;
        text.alignment = TextAnchor.MiddleCenter;
        return text;
    }

    static void EnsureEventSystem()
    {
        if (FindObjectOfType<EventSystem>() != null) return;
        new GameObject("EventSystem", typeof(EventSystem), typeof(StandaloneInputModule));
    }
}
