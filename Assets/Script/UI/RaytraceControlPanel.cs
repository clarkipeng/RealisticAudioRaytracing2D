using UnityEngine;
using UnityEngine.EventSystems;
using UnityEngine.UI;

public class RaytraceControlPanel : MonoBehaviour
{
    public RayTraceManager rayTraceManager;
    public AudioManager audioManager;
    public Oscillator trainMotion;
    public bool buildOnStart = true;

    [Header("Layout")]
    public Vector2 anchoredPosition = new Vector2(16, -16);
    public Vector2 buttonSize = new Vector2(180, 32);
    public int spacing = 8;

    Font buttonFont;

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
        buttonFont = Resources.GetBuiltinResource<Font>("Arial.ttf");

        Transform panel = CreatePanel();
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
        scaler.referenceResolution = new Vector2(1920, 1080);

        var panelObject = new GameObject("Panel", typeof(RectTransform), typeof(VerticalLayoutGroup));
        panelObject.transform.SetParent(canvasObject.transform, false);

        RectTransform rect = panelObject.GetComponent<RectTransform>();
        rect.anchorMin = new Vector2(0, 1);
        rect.anchorMax = new Vector2(0, 1);
        rect.pivot = new Vector2(0, 1);
        rect.anchoredPosition = anchoredPosition;

        VerticalLayoutGroup layout = panelObject.GetComponent<VerticalLayoutGroup>();
        layout.spacing = spacing;
        layout.childControlWidth = false;
        layout.childControlHeight = false;
        layout.childForceExpandWidth = false;
        layout.childForceExpandHeight = false;

        return panelObject.transform;
    }

    void AddButton(Transform parent, string label, UnityEngine.Events.UnityAction action)
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
    }

    static void EnsureEventSystem()
    {
        if (FindObjectOfType<EventSystem>() != null) return;
        new GameObject("EventSystem", typeof(EventSystem), typeof(StandaloneInputModule));
    }
}
