using UnityEngine;
using UnityEngine.UI;

// Position annotations only: no ray queries, audio readback or playback-state inference.
[RequireComponent(typeof(Camera)), DisallowMultipleComponent]
public sealed class AudioSourceOverlay : MonoBehaviour
{
    public RayTraceManager manager;
    [Tooltip("F6 toggles emitter positions. Markers do not indicate loudness or playback state.")]
    public bool visible = true;
    Canvas canvas;
    Camera view;
    Text[] labels;

    void Update() { if (Input.GetKeyDown(KeyCode.F6)) visible = !visible; }
    void LateUpdate() => Refresh();

    public void Refresh()
    {
        if (manager == null) return;
        if (canvas != null && labels.Length != manager.takeSources.Count) Clear();
        if (canvas == null) Build();
        canvas.enabled = visible;
        if (!visible) return;
        var size = ((RectTransform)canvas.transform).rect.size;
        var limit = new Vector2(Mathf.Max(.01f, .5f - 125 / size.x), Mathf.Max(.01f, .5f - 34 / size.y));
        int receiver = manager.takeListeners.IndexOf(transform) + 1;
        for (int i = 0; i < labels.Length; i++)
        {
            var source = manager.takeSources[i];
            bool show = source != null && source.isActiveAndEnabled &&
                (receiver == 0 || source.mutedListenerPlusOne != receiver);
            labels[i].transform.parent.gameObject.SetActive(show);
            if (!show) continue;
            var p = view.WorldToViewportPoint(source.transform.position);
            var direction = new Vector2(p.x - .5f, p.y - .5f) * (p.z < 0 ? -1 : 1);
            bool edge = p.z <= 0 || Mathf.Abs(direction.x) > limit.x || Mathf.Abs(direction.y) > limit.y;
            string mark = "●";
            if (edge)
            {
                if (direction.sqrMagnitude < .000001f) direction = Vector2.down;
                direction /= Mathf.Max(Mathf.Abs(direction.x) / limit.x, Mathf.Abs(direction.y) / limit.y);
                mark = Mathf.Abs(direction.x) > Mathf.Abs(direction.y)
                    ? (direction.x > 0 ? "→" : "←") : (direction.y > 0 ? "↑" : "↓");
            }
            var rect = (RectTransform)labels[i].transform.parent;
            Vector2 anchor = Vector2.one * .5f + direction;
            bool crowded = false;
            for (int attempt = 0; attempt <= 2 * i; attempt++)
            {
                crowded = false;
                for (int j = 0; j < i; j++)
                {
                    var other = (RectTransform)labels[j].transform.parent;
                    Vector2 delta = anchor - other.anchorMin;
                    crowded |= other.gameObject.activeSelf && Mathf.Abs(delta.x) < 240 / size.x && Mathf.Abs(delta.y) < 54 / size.y;
                }
                if (!crowded) break;
                anchor.y -= 56 / size.y;
                if (anchor.y < .5f - limit.y) anchor.y = .5f + limit.y;
            }
            rect.gameObject.SetActive(!crowded);
            rect.anchorMin = rect.anchorMax = anchor;
            rect.anchoredPosition = Vector2.zero;
            int distance = Mathf.RoundToInt(Vector3.Distance(transform.position, source.transform.position));
            labels[i].text = mark + "  " + source.name + "\n" + distance + " m · " +
                (source.streamed ? "voice emitter" : "clip emitter") + (p.z < 0 ? " · behind" : "");
        }
    }

    void Build()
    {
        view = GetComponent<Camera>();
        var root = new GameObject("Audio source positions", typeof(Canvas), typeof(CanvasScaler));
        root.transform.SetParent(transform, false);
        canvas = root.GetComponent<Canvas>();
        canvas.renderMode = RenderMode.ScreenSpaceCamera;
        canvas.worldCamera = view;
        canvas.planeDistance = view.nearClipPlane + .01f;
        canvas.sortingOrder = 20;
        var scaler = root.GetComponent<CanvasScaler>();
        scaler.uiScaleMode = CanvasScaler.ScaleMode.ScaleWithScreenSize;
        scaler.referenceResolution = new Vector2(1280, 720);
        scaler.matchWidthOrHeight = .5f;
        var font = Resources.GetBuiltinResource<Font>("LegacyRuntime.ttf");
        labels = new Text[manager.takeSources.Count];
        for (int i = 0; i < labels.Length; i++)
        {
            var tag = new GameObject("Source position", typeof(RectTransform), typeof(Image));
            tag.transform.SetParent(root.transform, false);
            var rect = (RectTransform)tag.transform;
            rect.sizeDelta = new Vector2(230, 48);
            var background = tag.GetComponent<Image>();
            background.color = new Color(.025f, .045f, .06f, .88f);
            background.raycastTarget = false;
            var text = new GameObject("Label", typeof(RectTransform), typeof(Text)).GetComponent<Text>();
            text.transform.SetParent(tag.transform, false);
            text.rectTransform.anchorMin = Vector2.zero; text.rectTransform.anchorMax = Vector2.one;
            text.rectTransform.offsetMin = new Vector2(10, 4); text.rectTransform.offsetMax = new Vector2(-8, -4);
            text.font = font; text.fontSize = 15; text.alignment = TextAnchor.MiddleLeft;
            text.raycastTarget = false;
            text.color = manager.takeSources[i] != null && manager.takeSources[i].streamed
                ? new Color(.4f, .9f, 1) : new Color(1, .8f, .4f);
            labels[i] = text;
        }
    }

    void OnDisable() { if (canvas != null) canvas.enabled = false; }
    void OnDestroy() => Clear();
    void Clear()
    {
        if (canvas == null) return;
        canvas.enabled = false;
        if (Application.isPlaying) Destroy(canvas.gameObject);
        else DestroyImmediate(canvas.gameObject);
        canvas = null;
    }
}
