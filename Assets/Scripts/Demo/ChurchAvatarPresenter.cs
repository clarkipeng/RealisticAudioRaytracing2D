using UnityEngine;
using UnityEngine.Rendering;

// Presentation follows the existing controller; animation never moves the player.
[DisallowMultipleComponent]
public sealed class ChurchAvatarPresenter : MonoBehaviour
{
    public Animator animator;
    public Renderer[] renderers;
    Vector3 previousPosition;
    static readonly int Speed = Animator.StringToHash("Speed");

    void Awake() => Initialize();
    public void Initialize()
    {
        previousPosition = transform.position;
        SetVisible(false);
    }
    void LateUpdate() => ManualUpdate(Time.deltaTime);
    public void ManualUpdate(float seconds)
    {
        if (seconds <= 0 || animator == null) return;
        var displacement = transform.position - previousPosition;
        previousPosition = transform.position;
        animator.SetFloat(Speed, new Vector2(displacement.x, displacement.z).magnitude / seconds);
        if (!Application.isPlaying) animator.Update(seconds);
    }
    public void SetVisible(bool visible)
    {
        foreach (var renderer in renderers)
            renderer.shadowCastingMode = visible ? ShadowCastingMode.On : ShadowCastingMode.ShadowsOnly;
    }
}
