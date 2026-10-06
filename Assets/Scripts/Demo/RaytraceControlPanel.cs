using UnityEngine;
using UnityEngine.EventSystems;
using UnityEngine.UI;

public class RaytraceControlPanel : MonoBehaviour
{
    public RayTraceManager rayTraceManager;
    public AudioManager audioManager;
    public Oscillator trainMotion;
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

    void Start()
    {
        if (rayTraceManager == null) rayTraceManager = FindObjectOfType<RayTraceManager>();
        if (audioManager == null) audioManager = FindObjectOfType<AudioManager>();
        if (trainMotion == null && rayTraceManager != null && rayTraceManager.source != null)
            trainMotion = rayTraceManager.source.GetComponent<Oscillator>();
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
            return;
        }
        if(rayTraceManager!=null && rayTraceManager.useTakeRenderer)
        {
            AddButton(panel,"Start preview",rayTraceManager.BeginLiveSimulation);
            AddButton(panel,"Stop preview",rayTraceManager.StopLiveSimulation);
            return;
        }
        AddButton(panel, "Stream", () => rayTraceManager?.StartStreaming());
        AddButton(panel, "Stop", () => rayTraceManager?.StopStreaming());
        AddButton(panel, "Precompute", () => rayTraceManager?.StartPrecomputedPlayback());
        AddButton(panel, "Play Precomputed", () => rayTraceManager?.PlayPrecomputedClip());
        AddButton(panel, "Move Train", ToggleTrainMotion);
        AddButton(panel, "Reset", () => rayTraceManager?.ResetSimulation());
        AddButton(panel, "Test Sine", () => rayTraceManager?.QueueTestSineWave());
        AddButton(panel, "Pause Audio", () => audioManager?.TogglePause());
        AddButton(panel, "Record", () => audioManager?.ToggleRecording());
    }

    void Update()
    {
        if(recordLabel==null || Time.unscaledTime<nextUiUpdate) return;
        nextUiUpdate=Time.unscaledTime+.2f;
        bool waiting=session.Busy || rayTraceManager.Mode==RayTraceManager.SessionMode.Draining;
        bool recording=rayTraceManager.IsRecordingTake;
        recordLabel.text=waiting?"…":session.IsInputReplay?(session.InputReplayFinished?"Replay finished":"Replaying"):recording?"Save run  [R]":"Saved";
        recordLabel.GetComponentInParent<Button>().interactable=!waiting && recording && !rayTraceManager.IsReplayingTake && !session.IsInputReplay;
        sensitivityControls.SetActive(Cursor.lockState!=CursorLockMode.Locked);
    }

    Text AddLabel(Transform parent,string value,int size,float height)
    {
        var obj=new GameObject("Label",typeof(RectTransform),typeof(Text)); obj.transform.SetParent(parent,false);
        obj.GetComponent<RectTransform>().sizeDelta=new Vector2(buttonSize.x,height);
        var label=obj.GetComponent<Text>(); label.font=buttonFont; label.fontSize=size;
        label.text=value; label.color=new Color(.92f,.96f,1); label.raycastTarget=false;
        label.alignment=TextAnchor.MiddleLeft; return label;
    }

    void ToggleTrainMotion()
    {
        if (trainMotion == null)
        {
            Debug.LogWarning("No train Oscillator assigned to RaytraceControlPanel.");
            return;
        }

        trainMotion.ToggleMoving();
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
